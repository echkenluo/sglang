"""CPU-only synthetic end-to-end tests for all Phase 2 v4 evaluator entries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


QUALITY_DIR = Path(__file__).with_name("quality")
sys.path.insert(0, str(QUALITY_DIR))

import quality_gate_eval as evaluator  # noqa: E402
from longgen_client import compare_runs  # noqa: E402
from phase2_v4_power import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CAPS,
    EXPECTED_POWER_GATE_NAMES,
    MONTE_CARLO_REPLICATES,
    POWER_ASSESSMENT_SCHEMA,
    POWER_INPUT_SCHEMA,
    PROVENANCE_KEYS,
    RAW_DIRECTORY_NAME,
    REPEATS_PER_MODE,
    REQUIRED_SOURCE_SHA_KEYS,
    WINDOW_VIEWS,
)


HEAD = "1" * 40
GSM_SHA = "2" * 64
DATASET_SHA = "3" * 64
TOKENIZER_SHA = "4" * 64


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def power_payload() -> dict:
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


def power_assessment(manifest_sha: str) -> dict:
    return {
        "schema": POWER_ASSESSMENT_SCHEMA,
        "candidate_blind": True,
        "control_only": True,
        "raw_manifest_sha256": manifest_sha,
        "overall_pass": True,
        "error": None,
    }


def path_config(mode: str, tag: str, stage: str, free_run: int) -> dict:
    return {
        "schema": "phase2-v4-path-config-v1",
        "mode": mode,
        "tag": tag,
        "stage": stage,
        "port": 30061,
        "free_run": free_run,
        "startup_ok": 1,
        "prefill_backend": "disabled",
        "cuda_graph": False,
        "radix_cache": False,
        "overlap_schedule": False,
        "flashinfer_autotune": False,
        "attention_backend": "dsv4",
    }


def make_free_run(targets: dict, mode: str, tag: str, targets_sha: str, config_sha: str, divergent: bool) -> dict:
    refs = [
        {
            "prompt_id": row["prompt_id"],
            "prompt_sha256": row["prompt_sha256"],
            "prompt_ids": row["prompt_ids"],
        }
        for row in targets["rows"][:16]
    ]
    serial = [[7] * 512 for _ in range(16)]
    waves = [[list(tokens) for tokens in serial] for _ in range(3)]
    if divergent:
        waves[0][0][0] = 8
    runs = {"serial": serial, "wave1": waves[0], "wave2": waves[1], "wave3": waves[2]}
    pairs = {
        "serial_vs_wave1": ("serial", "wave1"),
        "serial_vs_wave2": ("serial", "wave2"),
        "serial_vs_wave3": ("serial", "wave3"),
        "wave1_vs_wave2": ("wave1", "wave2"),
        "wave1_vs_wave3": ("wave1", "wave3"),
        "wave2_vs_wave3": ("wave2", "wave3"),
    }
    info = {name: compare_runs(runs[left], runs[right], refs) for name, (left, right) in pairs.items()}
    return {
        "protocol": "phase2-v4",
        "kind": "free-run-info",
        "mode": mode,
        "tag": tag,
        "targets_sha256": targets_sha,
        "path_config_receipt_sha256": config_sha,
        "prompts": refs,
        "serial": serial,
        "waves": waves,
        "first_divergence": {name: value["first_divergence"] for name, value in info.items()},
        "info": info,
    }


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    qdir = tmp_path / "run"
    qsrc = tmp_path / "quality"
    future_f = tmp_path / "future-f"
    qdir.mkdir()
    qsrc.mkdir()
    future_f.mkdir()
    monkeypatch.setattr(evaluator, "QUALITY_SOURCE", qsrc)
    monkeypatch.setattr(evaluator, "BOOTSTRAP_REPLICATES", 100)

    generator = qsrc / "logprob_client.py"
    generator.write_text("# frozen synthetic generator\n", encoding="utf-8")
    power_path = qsrc / "phase2-v4-power-input.json"
    write_json(power_path, power_payload())
    expected_assets = {
        "sharegpt_sha256": DATASET_SHA,
        "gsm8k_sha256": GSM_SHA,
        "tokenizer_files": {"tokenizer.json": TOKENIZER_SHA},
        "phase2_v4_power_input": {
            "path": power_path.name,
            "sha256": digest(power_path),
        },
    }
    write_json(qsrc / "expected_assets.json", expected_assets)

    rows = []
    for index in range(128):
        prompt_ids = [index + 1] * 256
        target_ids = [1000 + index] * 512
        rows.append(
            {
                "prompt_id": index,
                "text_sha": hashlib.sha256(f"text-{index}".encode()).hexdigest(),
                "prompt_ids": prompt_ids,
                "prompt_len": 256,
                "prompt_sha256": evaluator.ids_sha256(prompt_ids),
                "target_ids": target_ids,
                "target_sha256": evaluator.ids_sha256(target_ids),
            }
        )
    targets = {
        "protocol": "phase2-v4",
        "seed": 196_944_571,
        "dataset_sha256": DATASET_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "generator_sha256": digest(generator),
        "request_batching": {
            "schema": "phase2-v4-request-batching-v1",
            "tokenize_endpoint": "/v1/tokenize",
            "tokenize_batch_size": 128,
            "tokenize_response_identity": "server-request-order-contract",
            "generate_endpoint": "/generate",
            "generate_batch_size": 16,
            "generate_response_identity": "request-order-and-rid-echo",
        },
        "rows": rows,
    }
    targets_path = qdir / "targets-freeze.json"
    write_json(targets_path, targets)
    targets_sha = digest(targets_path)

    preflight = {
        "verified": True,
        "sglang_head": HEAD,
        "power_analysis": {
            "verified": True,
            "actual_sha256": digest(power_path),
            "expected_sha256": digest(power_path),
            "assessment": power_assessment(digest(power_path)),
        },
        "numeric_audit": {
            "verified": True,
            "min_exact": 1.0,
            "max_relative_l2": 0.0,
            "live_records": 344,
            "live_layers": 43,
        },
    }
    write_json(qdir / "preflight-manifest.json", preflight)

    modes = {"t-s": "split", **{f"d{i}": "deepep" for i in range(1, 4)}, **{f"s{i}": "split" for i in range(1, 4)}, **{f"f{i}": "fused" for i in range(1, 4)}}
    for tag, mode in modes.items():
        artifact_dir = future_f if tag in {"f1", "f2", "f3"} else qdir
        stage = "target-build" if tag == "t-s" else "bundle"
        free_run = int(tag in {"d1", "s1", "f1"})
        config_path = artifact_dir / f"path-config-receipt-{tag}-{stage}.json"
        write_json(config_path, path_config(mode, tag, stage, free_run))
        config_sha = digest(config_path)

        outputs = {
            "targets_freeze": targets_sha,
            "gsm8k_csv": None,
            "gsm8k_json": None,
            "teacher512": None,
            "free_run_info": None,
        }
        client_rc = {"target": 0, "gsm8k": None, "teacher512": None, "free_run_info": None}
        if stage == "bundle":
            teacher = {
                "protocol": "phase2-v4",
                "mode": mode,
                "tag": tag,
                "targets_sha256": targets_sha,
                "path_config_receipt_sha256": config_sha,
                "request_batching": evaluator.TEACHER_BATCHING_CONTRACT,
                "rows": [
                    {
                        "prompt_id": row["prompt_id"],
                        "prompt_sha256": row["prompt_sha256"],
                        "target_sha256": row["target_sha256"],
                        "token_ids": row["target_ids"],
                        "logprobs": [-0.25] * 512,
                        "top1_ids": row["target_ids"],
                    }
                    for row in rows
                ],
            }
            teacher_path = artifact_dir / f"teacher512-{tag}.json"
            write_json(teacher_path, teacher)
            outputs["teacher512"] = digest(teacher_path)

            csv_path = artifact_dir / f"gsm8k-{tag}.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as stream:
                stream.write("idx,pred,gold,correct,text_sha256\n")
                for index in range(1314):
                    text_sha = hashlib.sha256(f"answer-{tag}-{index}".encode()).hexdigest()
                    stream.write(f"{index},1,1,1,{text_sha}\n")
            summary = {
                "schema": "phase2-v4-gsm8k-v1",
                "tag": tag,
                "n": 1314,
                "correct": 1314,
                "accuracy": 1.0,
                "dataset_sha256": GSM_SHA,
                "csv_sha256": digest(csv_path),
                "request_batching": evaluator.GSM_BATCHING_CONTRACT,
            }
            summary_path = artifact_dir / f"gsm8k-{tag}.json"
            write_json(summary_path, summary)
            outputs["gsm8k_csv"] = digest(csv_path)
            outputs["gsm8k_json"] = digest(summary_path)
            client_rc = {"target": None, "gsm8k": 0, "teacher512": 0, "free_run_info": 0 if free_run else None}

            if free_run:
                free_path = artifact_dir / f"free-run-info-{tag}.json"
                write_json(
                    free_path,
                    make_free_run(
                        targets,
                        mode,
                        tag,
                        targets_sha,
                        config_sha,
                        divergent=(tag == "f1"),
                    ),
                )
                outputs["free_run_info"] = digest(free_path)

        receipt = {
            "schema": "phase2-v4-session-receipt-v1",
            "mode": mode,
            "tag": tag,
            "stage": stage,
            "rc": 0,
            "path_ok": 1,
            "free_run": free_run,
            "path_config_sha256": config_sha,
            "client_rc": client_rc,
            "outputs": outputs,
        }
        write_json(artifact_dir / f"path-receipt-{tag}-{stage}.json", receipt)
    return qdir


def run_control_checkpoints(qdir: Path, monkeypatch) -> None:
    assert evaluator.main([str(qdir), HEAD, "deepep-env"]) == 0
    assert json.loads((qdir / "deepep-env-checkpoint.json").read_text())["status"] == "PASS"
    assert evaluator.main([str(qdir), HEAD, "split-aa-freeze"]) == 0
    split_path = qdir / "split-aa-freeze.json"
    assert json.loads(split_path.read_text())["status"] == "PASS"
    monkeypatch.setenv("SPLIT_AA_FREEZE_SHA", digest(split_path))
    future_f = qdir.parent / "future-f"
    for path in future_f.iterdir():
        path.rename(qdir / path.name)


def test_all_three_entries_pass_and_free_run_divergence_is_information_only(synthetic_run, monkeypatch):
    run_control_checkpoints(synthetic_run, monkeypatch)
    assert evaluator.main([str(synthetic_run), HEAD]) == 0
    verdict = json.loads((synthetic_run / "quality-gate-verdict.json").read_text())
    assert verdict["status"] == "PASS"
    assert verdict["free_run_information_only"]["blocking"] is False
    assert verdict["free_run_information_only"]["tags"]["f1"]["comparisons"]["serial_vs_wave1"]["token_divergence_count"] == 1


@pytest.mark.parametrize("tag,checkpoint,output", [("d2", "deepep-env", "deepep-env-checkpoint.json"), ("s2", "split-aa-freeze", "split-aa-freeze.json")])
def test_missing_control_artifact_is_invalid(synthetic_run, tag, checkpoint, output):
    if checkpoint == "split-aa-freeze":
        assert evaluator.main([str(synthetic_run), HEAD, "deepep-env"]) == 0
    (synthetic_run / f"teacher512-{tag}.json").unlink()
    assert evaluator.main([str(synthetic_run), HEAD, checkpoint]) == 2
    assert json.loads((synthetic_run / output).read_text())["status"] == "INVALID"


def test_candidate_artifact_before_split_freeze_is_invalid(synthetic_run):
    assert evaluator.main([str(synthetic_run), HEAD, "deepep-env"]) == 0
    future = synthetic_run.parent / "future-f" / "teacher512-f1.json"
    future.rename(synthetic_run / future.name)
    assert evaluator.main([str(synthetic_run), HEAD, "split-aa-freeze"]) == 2
    assert json.loads((synthetic_run / "split-aa-freeze.json").read_text())["status"] == "INVALID"


@pytest.mark.parametrize("tag", ["f2", "f3"])
def test_missing_candidate_repeat_after_f1_is_fail(synthetic_run, monkeypatch, tag):
    run_control_checkpoints(synthetic_run, monkeypatch)
    (synthetic_run / f"teacher512-{tag}.json").unlink()
    assert evaluator.main([str(synthetic_run), HEAD]) == 1
    assert json.loads((synthetic_run / "quality-gate-verdict.json").read_text())["status"] == "FAIL"


def test_malformed_candidate_free_run_is_fail_after_path_hit(synthetic_run, monkeypatch):
    run_control_checkpoints(synthetic_run, monkeypatch)
    write_json(synthetic_run / "free-run-info-f1.json", {"malformed": True})
    assert evaluator.main([str(synthetic_run), HEAD]) == 1
    verdict = json.loads((synthetic_run / "quality-gate-verdict.json").read_text())
    assert verdict["status"] == "FAIL"


@pytest.mark.parametrize("artifact", ["path-receipt-d1-bundle.json", "path-config-receipt-d1-bundle.json"])
def test_control_receipt_or_path_config_tamper_is_invalid(synthetic_run, artifact):
    path = synthetic_run / artifact
    payload = json.loads(path.read_text())
    payload["path_ok" if artifact.startswith("path-receipt") else "cuda_graph"] = 0 if artifact.startswith("path-receipt") else True
    write_json(path, payload)
    assert evaluator.main([str(synthetic_run), HEAD, "deepep-env"]) == 2
    assert json.loads((synthetic_run / "deepep-env-checkpoint.json").read_text())["status"] == "INVALID"


@pytest.mark.parametrize(
    "artifact,output_key",
    [("teacher512-d1.json", "teacher512"), ("gsm8k-d1.json", "gsm8k_json")],
)
def test_request_batching_contract_tamper_is_invalid(
    synthetic_run, artifact, output_key
):
    path = synthetic_run / artifact
    payload = json.loads(path.read_text())
    payload["request_batching"]["generate_batch_size"] = 1
    write_json(path, payload)
    receipt_path = synthetic_run / "path-receipt-d1-bundle.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["outputs"][output_key] = digest(path)
    write_json(receipt_path, receipt)
    assert evaluator.main([str(synthetic_run), HEAD, "deepep-env"]) == 2
    verdict = json.loads(
        (synthetic_run / "deepep-env-checkpoint.json").read_text()
    )
    assert verdict["status"] == "INVALID"
    assert "batching contract mismatch" in verdict["failure"]
