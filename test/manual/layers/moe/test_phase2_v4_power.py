"""Pure-CPU regression tests for the Phase 2 v4 evaluator/power contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


QUALITY_DIR = Path(__file__).with_name("quality")
sys.path.insert(0, str(QUALITY_DIR))

from phase2_v4_power import (  # noqa: E402
    BLOCKING_GATE_COUNT,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CAPS,
    CI_GATE_COUNT,
    EFFECT_GATE_NAMES,
    EFFECT_SCENARIO_COUNT,
    EXPECTED_POWER_GATE_NAMES,
    EXPECTED_PROMPTS,
    EXPECTED_QUESTIONS,
    EXPECTED_TARGET_TOKENS,
    MONTE_CARLO_REPLICATES,
    POWER_INPUT_SCHEMA,
    PROVENANCE_KEYS,
    RAW_DIRECTORY_NAME,
    REPEATS_PER_MODE,
    REQUIRED_SOURCE_SHA_KEYS,
    SPAN_GATE_COUNT,
    WINDOW_VIEWS,
    QualityContractError,
    PowerBackendError,
    ScoreWindow,
    SimulationConfig,
    assess_power_input,
    assess_upper_noninferiority,
    batch_upper_decisions,
    bootstrap_count_matrix,
    classify_failure,
    compute_power_assessment,
    clustered_bootstrap_mean,
    main as power_main,
    validate_power_input,
    validate_repeat_tags,
    validate_window_coverage,
    validate_window_views,
    verify_scalar_batch_equivalence,
    _require_numpy,
    _precompute_gate_vectors,
    load_raw_control_assets,
    exact_margin_injection,
    iter_bootstrap_count_rows,
    upper_margin_predicate,
)
from quality_gate_eval import (  # noqa: E402
    ArtifactError,
    GateRecorder,
    RECEIPT_CONTRACT,
    TARGETS_FILE,
    _check_exact_receipt_set,
    _cluster_reduce,
    _is_finite_logprob,
    _is_nonnegative_token_id,
    _pair_metrics,
    _validate_ci_gate_coverage,
)


TEST_REPLICATES = 400


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def _make_raw_power_fixture(
    tmp_path: Path, backend: tuple[str, str, str] | None = None
) -> Path:
    raw = tmp_path / RAW_DIRECTORY_NAME
    raw.mkdir()
    generator = tmp_path / "logprob_client.py"
    generator.write_text("# raw-power fixture generator\n", encoding="utf-8")
    generator_sha = _digest(generator)
    target_batching = {
        "schema": "phase2-v4-request-batching-v1",
        "tokenize_endpoint": "/v1/tokenize",
        "tokenize_batch_size": 128,
        "tokenize_response_identity": "server-request-order-contract",
        "generate_endpoint": "/generate",
        "generate_batch_size": 16,
        "generate_response_identity": "request-order-and-rid-echo",
    }
    bundle_batching = {
        "schema": "phase2-v4-request-batching-v1",
        "generate_endpoint": "/generate",
        "generate_batch_size": 16,
        "response_identity": "request-order-and-rid-echo",
    }
    target_rows = []
    for index in range(EXPECTED_PROMPTS):
        prompt_ids = [index + 1] * 256
        target_ids = [1000 + index] * EXPECTED_TARGET_TOKENS
        target_rows.append({
            "prompt_id": index,
            "text_sha": hashlib.sha256(f"text-{index}".encode()).hexdigest(),
            "prompt_ids": prompt_ids,
            "prompt_len": len(prompt_ids),
            "prompt_sha256": hashlib.sha256(json.dumps(prompt_ids, separators=(",", ":")).encode()).hexdigest(),
            "target_ids": target_ids,
            "target_sha256": hashlib.sha256(json.dumps(target_ids, separators=(",", ":")).encode()).hexdigest(),
        })
    targets = {
        "protocol": "phase2-v4",
        "seed": 196_944_571,
        "dataset_sha256": "d" * 64,
        "tokenizer_sha256": "e" * 64,
        "generator_sha256": generator_sha,
        "request_batching": target_batching,
        "rows": target_rows,
    }
    targets_path = raw / "targets-freeze.json"
    _write_json(targets_path, targets)
    targets_sha = _digest(targets_path)

    modes = {"t-s": ("split", "target-build")}
    modes.update({f"d{i}": ("deepep", "bundle") for i in range(1, 4)})
    modes.update({f"s{i}": ("split", "bundle") for i in range(1, 4)})
    modes.update({f"p{i}": ("split", "bundle") for i in range(1, 4)})
    for offset, (tag, (mode, stage)) in enumerate(modes.items()):
        config = {
            "schema": "phase2-v4-path-config-v1", "mode": mode, "tag": tag,
            "stage": stage, "port": 30100 + offset, "free_run": 0,
            "startup_ok": 1, "prefill_backend": "disabled", "cuda_graph": False,
            "radix_cache": False, "overlap_schedule": False,
            "flashinfer_autotune": False, "attention_backend": "dsv4",
        }
        config_path = raw / f"path-config-receipt-{tag}-{stage}.json"
        _write_json(config_path, config)
        config_sha = _digest(config_path)
        outputs = {"targets_freeze": targets_sha, "gsm8k_csv": None, "gsm8k_json": None, "teacher512": None, "free_run_info": None}
        client_rc = {"target": 0, "gsm8k": None, "teacher512": None, "free_run_info": None}
        if stage == "bundle":
            teacher = {
                "protocol": "phase2-v4", "mode": mode, "tag": tag,
                "targets_sha256": targets_sha,
                "path_config_receipt_sha256": config_sha,
                "request_batching": bundle_batching,
                "rows": [{
                    "prompt_id": row["prompt_id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "target_sha256": row["target_sha256"],
                    "token_ids": row["target_ids"],
                    "logprobs": [-0.25] * EXPECTED_TARGET_TOKENS,
                    "top1_ids": row["target_ids"],
                } for row in target_rows],
            }
            teacher_path = raw / f"teacher512-{tag}.json"
            _write_json(teacher_path, teacher)
            csv_path = raw / f"gsm8k-{tag}.csv"
            csv_path.write_text(
                "idx,pred,gold,correct,text_sha256\n" + "".join(
                    f"{index},1,1,1,{hashlib.sha256(f'{tag}-{index}'.encode()).hexdigest()}\n"
                    for index in range(EXPECTED_QUESTIONS)
                ), encoding="utf-8",
            )
            summary_path = raw / f"gsm8k-{tag}.json"
            _write_json(summary_path, {
                "schema": "phase2-v4-gsm8k-v1", "tag": tag,
                "n": EXPECTED_QUESTIONS, "correct": EXPECTED_QUESTIONS,
                "accuracy": 1.0, "dataset_sha256": "f" * 64,
                "csv_sha256": _digest(csv_path), "request_batching": bundle_batching,
            })
            outputs.update({
                "gsm8k_csv": _digest(csv_path), "gsm8k_json": _digest(summary_path),
                "teacher512": _digest(teacher_path),
            })
            client_rc = {"target": None, "gsm8k": 0, "teacher512": 0, "free_run_info": None}
        receipt = {
            "schema": "phase2-v4-session-receipt-v1", "mode": mode, "tag": tag,
            "stage": stage, "rc": 0, "path_ok": 1, "free_run": 0,
            "path_config_sha256": config_sha, "client_rc": client_rc,
            "outputs": outputs,
        }
        _write_json(raw / f"path-receipt-{tag}-{stage}.json", receipt)

    artifacts = {path.name: _digest(path) for path in raw.iterdir()}
    assert set(artifacts) == REQUIRED_SOURCE_SHA_KEYS
    manifest = valid_power_input()
    manifest["artifacts"] = artifacts
    manifest["provenance"].update({
        "dataset_sha256": "d" * 64, "gsm8k_sha256": "f" * 64,
        "tokenizer_sha256": "e" * 64, "generator_sha256": generator_sha,
        "evaluator_sha256": _digest(QUALITY_DIR / "phase2_v4_power.py"),
    })
    if backend is not None:
        numpy_version, openblas_version, host_class = backend
        manifest["provenance"].update({
            "numpy_version": numpy_version,
            "openblas_version": openblas_version,
            "host_class": host_class,
        })
    manifest_path = tmp_path / "phase2-v4-power-input.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def valid_power_input() -> dict:
    return {
        "schema": POWER_INPUT_SCHEMA,
        "candidate_blind": True,
        "control_only": True,
        "source_root": RAW_DIRECTORY_NAME,
        "artifacts": {key: "a" * 64 for key in REQUIRED_SOURCE_SHA_KEYS},
        "provenance": {
            key: (
                "b" * 40 if key.endswith("_head")
                else "test-host" if key == "host_class"
                else "1.0" if key in {"numpy_version", "openblas_version"}
                else "c" * 64
            )
            for key in PROVENANCE_KEYS
        },
    }


def test_five_windows_exact_and_off_by_one_rejected():
    assert [(window.start, window.stop) for window in WINDOW_VIEWS] == [
        (0, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (0, 512),
    ]
    assert [window.width for window in WINDOW_VIEWS] == [64, 64, 128, 256, 512]
    bad = list(WINDOW_VIEWS)
    bad[1] = ScoreWindow("t064_127", 65, 128)
    with pytest.raises(QualityContractError, match="frozen score windows changed"):
        validate_window_views(tuple(bad))


def test_missing_or_duplicate_window_fails_closed():
    with pytest.raises(QualityContractError, match="all five score windows"):
        validate_window_coverage([window.name for window in WINDOW_VIEWS[:-1]], "x")
    duplicate = [window.name for window in WINDOW_VIEWS] + [WINDOW_VIEWS[0].name]
    with pytest.raises(QualityContractError, match="exactly once"):
        validate_window_coverage(duplicate, "x")


def test_fixed_mean_p95_flip_caps_are_exact():
    assert (
        CAPS.control_mean_abs_logprob,
        CAPS.control_p95_abs_logprob,
        CAPS.control_top1_flip,
    ) == (0.02, 0.10, 0.02)
    assert (
        CAPS.fused_self_mean_abs_logprob,
        CAPS.fused_self_p95_abs_logprob,
        CAPS.fused_self_top1_flip,
    ) == (0.02, 0.10, 0.02)
    assert (
        CAPS.fused_split_mean_abs_logprob,
        CAPS.fused_split_p95_abs_logprob,
        CAPS.fused_split_top1_flip,
    ) == (0.02, 0.10, 0.02)
    assert (
        CAPS.deepep_prefix_mean_abs_logprob,
        CAPS.deepep_prefix_p95_abs_logprob,
        CAPS.deepep_prefix_top1_flip,
    ) == (0.08, 0.30, 0.02)


def test_one_sided_u95_boundary_and_q975_is_report_only():
    at_cap = assess_upper_noninferiority(
        [0.02] * EXPECTED_PROMPTS,
        margin=0.02,
        cluster_kind="prompt",
        expected_clusters=EXPECTED_PROMPTS,
        replicates=TEST_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    assert at_cap.passed and at_cap.bootstrap.gate_upper95 == 0.02
    over = assess_upper_noninferiority(
        [0.020001] * EXPECTED_PROMPTS,
        margin=0.02,
        cluster_kind="prompt",
        expected_clusters=EXPECTED_PROMPTS,
        replicates=TEST_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    assert not over.passed

    asymmetric = clustered_bootstrap_mean(
        [0.0] * 120 + [1.0] * 8,
        cluster_kind="prompt",
        expected_clusters=EXPECTED_PROMPTS,
        replicates=2000,
        seed=BOOTSTRAP_SEED,
    )
    assert asymmetric.gate_upper95 < asymmetric.report_upper975


def test_p95_is_per_prompt_nearest_rank_then_pair_median():
    window = WINDOW_VIEWS[0]
    left = [
        {"logprobs": [0.0] * 512, "top1_ids": [0] * 512}
        for _ in range(EXPECTED_PROMPTS)
    ]
    right_logprobs = [0.0] * 60 + [0.2] * 4 + [0.0] * (512 - 64)
    right = [
        {"logprobs": right_logprobs, "top1_ids": [0] * 512}
        for _ in range(EXPECTED_PROMPTS)
    ]
    metrics = _pair_metrics(left, right, window)
    assert metrics["p95"] == [0.2] * EXPECTED_PROMPTS
    assert metrics["mean"] == [0.0125] * EXPECTED_PROMPTS
    pair_medians = _cluster_reduce(
        [[0.01] * 128, [0.03] * 128, [0.02] * 128],
        __import__("statistics").median,
    )
    assert pair_medians == [0.02] * 128


def test_old_triangle_passes_but_directional_quality_fails():
    band = 0.10
    assert 0.10 <= band
    assert 0.60 - 0.50 <= band
    new_gate = assess_upper_noninferiority(
        [0.10] * EXPECTED_PROMPTS,
        margin=CAPS.directional_delta_target_error,
        cluster_kind="prompt",
        expected_clusters=EXPECTED_PROMPTS,
        replicates=TEST_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    assert not new_gate.passed


def test_three_repeats_and_f2_f3_missing_fail_closed():
    assert validate_repeat_tags(("f3", "f1", "f2"), "f") == ("f1", "f2", "f3")
    with pytest.raises(QualityContractError, match="f2.*f3"):
        validate_repeat_tags(("f1",), "f")


def test_token_level_bootstrap_is_rejected():
    with pytest.raises(QualityContractError, match="token-level resampling"):
        clustered_bootstrap_mean(
            [0.0] * (EXPECTED_PROMPTS * 512),
            cluster_kind="token",
            expected_clusters=EXPECTED_PROMPTS * 512,
            replicates=TEST_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )


def test_candidate_only_failure_classification():
    assert classify_failure(scope="candidate", target_path_hit=True, controls_stable=True) == "FAIL"
    assert classify_failure(scope="candidate", target_path_hit=False, controls_stable=True) == "INVALID"
    assert classify_failure(scope="candidate", target_path_hit=True, controls_stable=False) == "INVALID"
    assert classify_failure(scope="control", target_path_hit=True, controls_stable=True) == "INVALID"


def test_raw_manifest_strict_schema_and_no_self_reported_rates():
    payload = valid_power_input()
    assert validate_power_input(payload) == payload
    assessment = assess_power_input(payload)
    assert assessment["overall_pass"] is False
    assert "compute_power_assessment" in assessment["error"]

    payload["candidate_blind"] = False
    assert assess_power_input(payload)["overall_pass"] is False
    with pytest.raises(QualityContractError, match="candidate_blind"):
        validate_power_input(payload)


def test_reported_rate_injection_and_unknown_artifact_fail_closed():
    reported = valid_power_input()
    reported["families"] = {"full73": {"pass_rate": 1.0}}
    with pytest.raises(QualityContractError, match="key set mismatch"):
        validate_power_input(reported)

    unknown = valid_power_input()
    unknown["artifacts"]["teacher512-f1.json"] = "d" * 64
    with pytest.raises(QualityContractError, match="artifact set"):
        validate_power_input(unknown)


def test_frozen_power_gate_cardinality_and_independent_pseudo_f_contract():
    assert CI_GATE_COUNT == len(EXPECTED_POWER_GATE_NAMES) == 70
    assert SPAN_GATE_COUNT == 3
    assert BLOCKING_GATE_COUNT == 73
    assert EFFECT_SCENARIO_COUNT == len(EFFECT_GATE_NAMES) == 22
    artifacts = REQUIRED_SOURCE_SHA_KEYS
    assert all(f"teacher512-p{i}.json" in artifacts for i in range(1, 4))
    assert all(f"teacher512-f{i}.json" not in artifacts for i in range(1, 4))


def test_raw_dsp_manifest_recomputes_exact_gate_vectors_and_rejects_unknown(tmp_path):
    manifest_path = _make_raw_power_fixture(tmp_path)
    raw = load_raw_control_assets(manifest_path)
    assert set(raw.teachers) == {f"{family}{index}" for family in "dsp" for index in range(1, 4)}
    gates, spans, margins = _precompute_gate_vectors(raw)
    assert len(gates) == len(margins) == CI_GATE_COUNT
    assert set(gates) == EXPECTED_POWER_GATE_NAMES
    assert set(spans) == {
        "5-gsm8k-deepep-accuracy-span",
        "5-gsm8k-split-accuracy-span",
        "5-gsm8k-fused-accuracy-span",
    }

    extra = tmp_path / RAW_DIRECTORY_NAME / "teacher512-f1.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(QualityContractError, match="file set mismatch"):
        load_raw_control_assets(manifest_path)


def test_missing_power_asset_cli_is_explicit_no_go(tmp_path, capsys):
    assert power_main([str(tmp_path / "missing.json")]) == 2
    assert "overall_pass=0" in capsys.readouterr().out


def test_exact_10_receipt_names_and_f2_f3_missing(tmp_path):
    expected = {
        "path-receipt-t-s-target-build.json",
        *(f"path-receipt-{family}{index}-bundle.json" for family in "dsf" for index in range(1, 4)),
    }
    assert len(expected) == 10
    assert {
        f"path-receipt-{tag}-{stage}.json"
        for tag, (_, stage) in RECEIPT_CONTRACT.items()
    } == expected
    for name in expected:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    _check_exact_receipt_set(str(tmp_path))
    (tmp_path / "path-receipt-f2-bundle.json").unlink()
    with pytest.raises(ArtifactError, match="exactly 10"):
        _check_exact_receipt_set(str(tmp_path))


def test_ci_gate_matrix_is_exact_and_missing_window_gate_fails():
    assert len(EXPECTED_POWER_GATE_NAMES) == 70
    recorder = GateRecorder()
    recorder.gates = [
        {"gate": name, "detail": {"bootstrap": {}}, "pass": True, "class": "x"}
        for name in EXPECTED_POWER_GATE_NAMES
    ]
    _validate_ci_gate_coverage(recorder)
    recorder.gates.pop()
    with pytest.raises(QualityContractError, match="CI gate matrix incomplete"):
        _validate_ci_gate_coverage(recorder)


def test_evaluator_has_only_v4_artifact_names():
    source = (QUALITY_DIR / "quality_gate_eval.py").read_text(encoding="utf-8")
    assert TARGETS_FILE == "targets-freeze.json"
    for forbidden in ("logprob-targets.json", "logprob-score-", "longgen-"):
        assert forbidden not in source
    assert "teacher512-{tag}.json" in source
    assert "free-run-info-{tag}.json" in source


def test_bool_logprob_and_negative_top1_are_rejected():
    assert not _is_finite_logprob(True)
    assert not _is_finite_logprob(float("inf"))
    assert _is_finite_logprob(-1.25)
    assert not _is_nonnegative_token_id(True)
    assert not _is_nonnegative_token_id(-1)
    assert _is_nonnegative_token_id(0)


def test_bootstrap_fixed_seed_is_reproducible():
    kwargs = dict(
        cluster_kind="prompt",
        expected_clusters=EXPECTED_PROMPTS,
        replicates=TEST_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    assert clustered_bootstrap_mean(range(EXPECTED_PROMPTS), **kwargs) == clustered_bootstrap_mean(range(EXPECTED_PROMPTS), **kwargs)


def test_missing_numpy_is_explicit_no_go(monkeypatch):
    import builtins

    original = builtins.__import__

    def reject_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("injected missing numpy")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_numpy)
    with pytest.raises(PowerBackendError, match="no scalar fallback"):
        _require_numpy()


def test_python_random_count_stream_and_margin_predicate_are_frozen():
    assert list(iter_bootstrap_count_rows(5, 3, 123)) == [
        [2, 0, 2, 1, 0],
        [2, 0, 0, 1, 2],
        [1, 2, 2, 0, 0],
    ]
    assert upper_margin_predicate(0.02 - 1e-9, 0.02)
    assert upper_margin_predicate(0.02, 0.02)
    assert not upper_margin_predicate(0.02 + 1e-9, 0.02)


def test_degenerate_effect_source_uses_explicit_margin_plus_25pct_policy():
    shift, target, degenerate = exact_margin_injection([0.0] * 128, 0.02)
    assert degenerate is True
    assert target == pytest.approx(0.025)
    assert shift == pytest.approx(0.025)
    shift, target, degenerate = exact_margin_injection([0.0, 0.01] * 64, 0.02)
    assert degenerate is False
    assert target == 0.02
    assert shift == pytest.approx(0.015)


def test_optional_numpy_batch_decision_matches_scalar(monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    loaded, _configuration, openblas_version = _require_numpy()
    assert loaded is np
    assert openblas_version
    assert len(verify_scalar_batch_equivalence(np)) == 64
    counts = bootstrap_count_matrix(8, 200, BOOTSTRAP_SEED, np)
    vectors = np.asarray([[0.02] * 8, [0.03] * 8])
    passed, means = batch_upper_decisions(vectors, np.asarray([0.02, 0.02]), counts, np)
    assert passed.tolist() == [True, False]
    assert means.shape == (2, 200)


def test_optional_numpy_small_outer_raw_recomputation(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("PHASE2_POWER_HOST_CLASS", "pytest-host")
    loaded, _configuration, openblas_version = _require_numpy()
    assert loaded is np
    manifest = _make_raw_power_fixture(
        tmp_path, (np.__version__, openblas_version, "pytest-host")
    )
    result = compute_power_assessment(
        manifest,
        simulation=SimulationConfig(
            outer_trials=12, bootstrap_replicates=100, outer_chunk=4
        ),
    )
    assert result["contract"]["ci_gates"] == 70
    assert result["contract"]["blocking_gates"] == 73
    assert result["contract"]["effect_scenarios"] == 22
    assert result["pass_vector"]["bits"] == 12 * 120
    assert result["overall_pass"] is True
    assert all(
        record["injection_policy"] == "margin-plus-25pct"
        for record in result["exact_margin_effects"].values()
    )
