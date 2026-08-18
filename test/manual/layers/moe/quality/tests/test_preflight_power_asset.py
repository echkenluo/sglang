import hashlib
import json
from pathlib import Path
import sys


QUALITY_DIR = Path(__file__).resolve().parents[1]
if str(QUALITY_DIR) not in sys.path:
    sys.path.insert(0, str(QUALITY_DIR))

import preflight_manifest as preflight  # noqa: E402


EVALUATOR_SOURCE = r'''
import hashlib, json
from pathlib import Path

EXPECTED_POWER_GATE_NAMES = frozenset(f"gate-{i:02d}" for i in range(70))
EFFECT_GATE_NAMES = tuple(f"effect-{i:02d}" for i in range(22))

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_power_input(payload):
    if payload.get("schema") != "phase2-v4-raw-power-input-v2":
        raise ValueError("raw schema rejected")
    if set(payload) != {"schema", "candidate_blind", "control_only", "source_root", "artifacts", "provenance"}:
        raise ValueError("self-reported or unknown fields rejected")
    return payload

def compute_power_assessment(input_path):
    payload = json.loads(Path(input_path).read_text())
    qsrc = Path(input_path).parent
    fail = payload.get("provenance", {}).get("dataset_sha256") == "0" * 64
    gate_fail = payload.get("provenance", {}).get("gsm8k_sha256") == "0" * 64
    families = {name: {"pass": not fail} for name in ("teacher68", "gsm5", "full73")}
    gates = {name: {"pass": not gate_fail} for name in EXPECTED_POWER_GATE_NAMES}
    effects = {name: {"pass": not fail} for name in EFFECT_GATE_NAMES}
    provenance = dict(payload["provenance"])
    provenance["generator_sha256"] = _sha(qsrc / "logprob_client.py")
    provenance["evaluator_sha256"] = _sha(__file__)
    return {
        "schema": "phase2-v4-raw-power-assessment-v2",
        "candidate_blind": True,
        "control_only": True,
        "raw_manifest_sha256": _sha(input_path),
        "raw_source_digest": "b" * 64,
        "source_shas": payload["artifacts"],
        "provenance": provenance,
        "contract": {"ci_gates": 70, "span_gates": 3, "blocking_gates": 73,
                     "effect_scenarios": 22, "outer_trials": 10000,
                     "bootstrap_replicates": 10000, "outer_chunk": 32,
                     "formal_outer_chunk": 32},
        "zero_degradation_families": families,
        "single_ci_gates": gates,
        "exact_margin_effects": effects,
        "pass_vector": {
            "sha256": "c" * 64, "bits": 1200000, "ones": 900000,
            "popcounts": {"ci_pass": 600000, "span_pass": 20000,
                          "family_pass": 20000, "effect_gate_rejection": 130000,
                          "effect_family_failure": 130000},
        },
        "overall_pass": not fail and not gate_fail,
        "error": None,
    }
'''


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def power_input(*, family_fail=False, gate_fail=False):
    return {
        "schema": preflight.POWER_INPUT_SCHEMA,
        "candidate_blind": True,
        "control_only": True,
        "source_root": "phase2-v4-power-raw",
        "artifacts": {"raw-placeholder": "a" * 64},
        "provenance": {
            "sglang_head": "1" * 40,
            "mok_head": "2" * 40,
            "dataset_sha256": "0" * 64 if family_fail else "3" * 64,
            "gsm8k_sha256": "0" * 64 if gate_fail else "4" * 64,
            "tokenizer_sha256": "5" * 64,
            "generator_sha256": "6" * 64,
            "evaluator_sha256": "7" * 64,
            "numpy_version": "2.1.0",
            "openblas_version": "0.3.27",
            "host_class": "gpu9-control",
        },
    }


def write_fixture(qsrc, payload):
    evaluator = qsrc / preflight.POWER_EVALUATOR_NAME
    evaluator.write_text(EVALUATOR_SOURCE, encoding="utf-8")
    (qsrc / "logprob_client.py").write_text("# fixture\n", encoding="utf-8")
    power_path = qsrc / preflight.POWER_INPUT_NAME
    power_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return power_path


def run_verify(expected, qsrc):
    preflight.FAILS.clear()
    result = preflight.verify_power_asset(expected, str(qsrc))
    return result, list(preflight.FAILS)


def test_power_asset_is_recomputed_and_recorded(tmp_path):
    power_path = write_fixture(tmp_path, power_input())
    result, failures = run_verify(
        {"phase2_v4_power_input": {"path": power_path.name, "sha256": sha256(power_path)}},
        tmp_path,
    )
    assert failures == []
    assert result["verified"] is True
    assert result["assessment"]["contract"]["blocking_gates"] == 73
    assert result["assessment"]["raw_manifest_sha256"] == sha256(power_path)
    assert result["assessment"]["pass_vector"]["sha256"] == "c" * 64


def test_absolute_manifest_path_is_rejected(tmp_path):
    power_path = write_fixture(tmp_path, power_input())
    result, failures = run_verify(
        {"phase2_v4_power_input": {"path": str(power_path), "sha256": sha256(power_path)}},
        tmp_path,
    )
    assert result["verified"] is False
    assert failures == ["power_input_config"]


def test_manifest_symlink_is_rejected(tmp_path):
    original = write_fixture(tmp_path, power_input())
    link = tmp_path / "linked-power.json"
    link.symlink_to(original.name)
    result, failures = run_verify(
        {"phase2_v4_power_input": {"path": link.name, "sha256": sha256(original)}},
        tmp_path,
    )
    assert result["verified"] is False
    assert failures == ["power_input_symlink"]


def test_tamper_fails_before_evaluator(tmp_path):
    power_path = write_fixture(tmp_path, power_input())
    frozen = sha256(power_path)
    power_path.write_text(power_path.read_text() + "\n", encoding="utf-8")
    result, failures = run_verify(
        {"phase2_v4_power_input": {"path": power_path.name, "sha256": frozen}}, tmp_path
    )
    assert result["verified"] is False
    assert failures == ["power_input_sha256"]
    assert result["assessment"] is None


def test_computed_family_or_single_gate_no_go_is_fail_closed(tmp_path):
    for payload, expected_kind in (
        (power_input(family_fail=True), "power_family_gate"),
        (power_input(gate_fail=True), "power_single_gate"),
    ):
        case = tmp_path / expected_kind
        case.mkdir()
        power_path = write_fixture(case, payload)
        result, failures = run_verify(
            {"phase2_v4_power_input": {"path": power_path.name, "sha256": sha256(power_path)}}, case
        )
        assert result["verified"] is False
        assert expected_kind in failures
        assert "power_assessment_no_go" in failures


def test_pending_expected_sha_makes_formal_preflight_fail(tmp_path):
    result, failures = run_verify(
        {"phase2_v4_power_input": {"path": preflight.POWER_INPUT_NAME, "sha256": "PENDING_POWER"}}, tmp_path
    )
    assert result["verified"] is False
    assert failures == ["power_input_expected_pending"]
