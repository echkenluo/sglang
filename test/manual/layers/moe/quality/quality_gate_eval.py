"""Phase 2 v4 quality evaluator and control checkpoints.

The blocking teacher metrics use prompt-clustered one-sided U95.  D/S/F each
have three independent server repeats, teacher outputs are exactly 128x512,
and free-running generation is completeness-checked but information-only.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Callable, Iterable, Sequence

from longgen_client import validate_free_run

from phase2_v4_power import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CAPS,
    EXPECTED_POWER_GATE_NAMES,
    EXPECTED_PROMPTS,
    EXPECTED_QUESTIONS,
    EXPECTED_TARGET_TOKENS,
    WINDOW_VIEWS,
    QualityContractError,
    ScoreWindow,
    assess_upper_noninferiority,
    classify_failure,
    nearest_rank,
    validate_repeat_tags,
    validate_window_coverage,
)


DEFAULT_QDIR = "/mok/claude-mok/quality-v4"
QUALITY_SOURCE = Path(__file__).resolve().parent
TARGETS_FILE = "targets-freeze.json"
PROTOCOL = "phase2-v4"
TARGET_SEED = 196_944_571
TARGET_BATCHING_CONTRACT = {
    "schema": "phase2-v4-request-batching-v1",
    "tokenize_endpoint": "/v1/tokenize",
    "tokenize_batch_size": 128,
    "tokenize_response_identity": "server-request-order-contract",
    "generate_endpoint": "/generate",
    "generate_batch_size": 16,
    "generate_response_identity": "request-order-and-rid-echo",
}
TEACHER_BATCHING_CONTRACT = {
    "schema": "phase2-v4-request-batching-v1",
    "generate_endpoint": "/generate",
    "generate_batch_size": 16,
    "response_identity": "request-order-and-rid-echo",
}
GSM_BATCHING_CONTRACT = dict(TEACHER_BATCHING_CONTRACT)

DEEP_TAGS = validate_repeat_tags(("d1", "d2", "d3"), "d")
SPLIT_TAGS = validate_repeat_tags(("s1", "s2", "s3"), "s")
FUSED_TAGS = validate_repeat_tags(("f1", "f2", "f3"), "f")
ALL_TAGS = DEEP_TAGS + SPLIT_TAGS + FUSED_TAGS

RECEIPT_CONTRACT = {
    "t-s": ("split", "target-build"),
    **{tag: ("deepep", "bundle") for tag in DEEP_TAGS},
    **{tag: ("split", "bundle") for tag in SPLIT_TAGS},
    **{tag: ("fused", "bundle") for tag in FUSED_TAGS},
}


class ArtifactError(RuntimeError):
    def __init__(self, scope: str, detail: str):
        super().__init__(detail)
        self.scope = scope
        self.detail = detail


class GateRecorder:
    def __init__(self) -> None:
        self.gates: list[dict[str, object]] = []

    def add(self, name: str, passed: bool, detail: object, *, gate_class: str) -> None:
        item = {
            "gate": name,
            "pass": bool(passed),
            "class": gate_class,
            "detail": detail,
        }
        self.gates.append(item)
        compact = json.dumps(detail, sort_keys=True, separators=(",", ":"))
        print(
            f"GATE|{name}|{'PASS' if passed else 'FAIL'}|class={gate_class}|"
            f"{compact[:1200]}",
            flush=True,
        )

    def all_pass(self, gate_class: str | None = None) -> bool:
        selected = (
            self.gates
            if gate_class is None
            else [gate for gate in self.gates if gate["class"] == gate_class]
        )
        return bool(selected) and all(bool(gate["pass"]) for gate in selected)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids: Sequence[int]) -> str:
    raw = json.dumps(ids, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_nonnegative_token_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite_logprob(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _scope_for_tag(tag: str) -> str:
    return "candidate" if tag in FUSED_TAGS else "control"


def _require_file(path: str | Path, scope: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise ArtifactError(scope, f"missing required artifact: {result}")
    return result


def _load_json(path: str | Path, scope: str) -> object:
    _require_file(path, scope)
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(scope, f"malformed JSON {path}: {error}") from error


def load_targets(qdir: str) -> dict:
    path = Path(qdir) / TARGETS_FILE
    document = _load_json(path, "control")
    try:
        expected_assets = _load_json(
            QUALITY_SOURCE / "expected_assets.json", "infrastructure"
        )
        if not isinstance(document, dict) or document.get("protocol") != PROTOCOL:
            raise ValueError("protocol mismatch")
        if document.get("seed") != TARGET_SEED:
            raise ValueError("target seed mismatch")
        if document.get("request_batching") != TARGET_BATCHING_CONTRACT:
            raise ValueError("target request batching contract mismatch")
        for field in ("dataset_sha256", "tokenizer_sha256", "generator_sha256"):
            if not _is_sha256(document.get(field)):
                raise ValueError(f"invalid {field}")
        expected_bindings = {
            "dataset_sha256": expected_assets["sharegpt_sha256"],
            "tokenizer_sha256": expected_assets["tokenizer_files"]["tokenizer.json"],
            "generator_sha256": sha256(QUALITY_SOURCE / "logprob_client.py"),
        }
        for field, expected in expected_bindings.items():
            if document.get(field) != expected:
                raise ValueError(f"{field} does not match frozen asset")
        rows = document.get("rows")
        if not isinstance(rows, list) or len(rows) != EXPECTED_PROMPTS:
            raise ValueError(f"expected {EXPECTED_PROMPTS} target rows")
        prompt_ids_seen: set[int] = set()
        prompt_shas_seen: set[str] = set()
        text_shas_seen: set[str] = set()
        for index, row in enumerate(rows):
            prompt_id = row["prompt_id"]
            prompt_len = row["prompt_len"]
            prompt_ids = row["prompt_ids"]
            target_ids = row["target_ids"]
            if (
                not isinstance(prompt_id, int)
                or isinstance(prompt_id, bool)
                or prompt_id < 0
                or prompt_id in prompt_ids_seen
            ):
                raise ValueError(f"invalid/duplicate prompt_id at row {index}")
            if not isinstance(prompt_len, int) or not 256 <= prompt_len <= 2048:
                raise ValueError(f"invalid prompt_len at row {index}")
            if len(prompt_ids) != prompt_len or len(target_ids) != EXPECTED_TARGET_TOKENS:
                raise ValueError(f"token length mismatch at row {index}")
            if not all(_is_nonnegative_token_id(token) for token in prompt_ids + target_ids):
                raise ValueError(f"invalid token ID at row {index}")
            if prompt_len + EXPECTED_TARGET_TOKENS > 32768:
                raise ValueError(f"context overflow at row {index}")
            prompt_sha = row["prompt_sha256"]
            target_sha = row["target_sha256"]
            text_sha = row["text_sha"]
            if ids_sha256(prompt_ids) != prompt_sha or ids_sha256(target_ids) != target_sha:
                raise ValueError(f"ID SHA mismatch at row {index}")
            if not _is_sha256(text_sha) or text_sha in text_shas_seen:
                raise ValueError(f"invalid/duplicate text_sha at row {index}")
            if prompt_sha in prompt_shas_seen:
                raise ValueError(f"duplicate prompt SHA at row {index}")
            prompt_ids_seen.add(prompt_id)
            prompt_shas_seen.add(prompt_sha)
            text_shas_seen.add(text_sha)
        return document
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError("control", f"malformed {TARGETS_FILE}: {error}") from error


def load_score(tag: str, qdir: str, targets: dict, receipt: dict) -> list[dict]:
    scope = _scope_for_tag(tag)
    path = Path(qdir) / f"teacher512-{tag}.json"
    document = _load_json(path, scope)
    target_rows = targets["rows"]
    expected_mode = RECEIPT_CONTRACT[tag][0]
    try:
        if document.get("protocol") != PROTOCOL:
            raise ValueError("protocol mismatch")
        if document.get("mode") != expected_mode or document.get("tag") != tag:
            raise ValueError("mode/tag mismatch")
        if document.get("targets_sha256") != sha256(Path(qdir) / TARGETS_FILE):
            raise ValueError("targets SHA mismatch")
        if document.get("path_config_receipt_sha256") != receipt["path_config_sha256"]:
            raise ValueError("path config receipt SHA mismatch")
        if document.get("request_batching") != TEACHER_BATCHING_CONTRACT:
            raise ValueError("teacher request batching contract mismatch")
        rows = document.get("rows")
        if not isinstance(rows, list) or len(rows) != EXPECTED_PROMPTS:
            raise ValueError(f"expected {EXPECTED_PROMPTS} score rows")
        for index, (row, target) in enumerate(zip(rows, target_rows)):
            if row.get("prompt_id") != target["prompt_id"]:
                raise ValueError(f"prompt_id mismatch at row {index}")
            if row.get("prompt_sha256") != target["prompt_sha256"]:
                raise ValueError(f"prompt SHA mismatch at row {index}")
            if row.get("target_sha256") != target["target_sha256"]:
                raise ValueError(f"target SHA mismatch at row {index}")
            for field in ("token_ids", "logprobs", "top1_ids"):
                if not isinstance(row.get(field), list) or len(row[field]) != EXPECTED_TARGET_TOKENS:
                    raise ValueError(f"{field} length mismatch at row {index}")
            if row["token_ids"] != target["target_ids"]:
                raise ValueError(f"target token mismatch at row {index}")
            if not all(_is_finite_logprob(value) for value in row["logprobs"]):
                raise ValueError(f"non-finite logprob at row {index}")
            if not all(_is_nonnegative_token_id(token) for token in row["top1_ids"]):
                raise ValueError(f"invalid top1 token at row {index}")
        return rows
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ArtifactError(scope, f"malformed teacher512-{tag}: {error}") from error


def load_gsm8k(tag: str, qdir: str) -> dict[int, tuple[str, int]]:
    scope = _scope_for_tag(tag)
    csv_path = _require_file(Path(qdir) / f"gsm8k-{tag}.csv", scope)
    json_path = Path(qdir) / f"gsm8k-{tag}.json"
    summary = _load_json(json_path, scope)
    try:
        expected_assets = _load_json(
            QUALITY_SOURCE / "expected_assets.json", "infrastructure"
        )
        expected_dataset_sha = expected_assets["gsm8k_sha256"]
        if not _is_sha256(expected_dataset_sha):
            raise ValueError("expected GSM8K SHA is not frozen")
        rows: dict[int, tuple[str, int]] = {}
        with open(csv_path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                index = int(row["idx"])
                correct = int(row["correct"])
                if index in rows or correct not in (0, 1):
                    raise ValueError(f"invalid/duplicate GSM8K index {index}")
                if not _is_sha256(row["text_sha256"]):
                    raise ValueError(f"invalid full text SHA at question {index}")
                rows[index] = (row["pred"], correct)
        if set(rows) != set(range(EXPECTED_QUESTIONS)):
            raise ValueError("GSM8K must contain exactly 1314 unique questions")
        accuracy = sum(value[1] for value in rows.values()) / EXPECTED_QUESTIONS
        if (
            summary.get("schema") != "phase2-v4-gsm8k-v1"
            or summary.get("tag") != tag
            or summary.get("n") != EXPECTED_QUESTIONS
        ):
            raise ValueError("GSM8K summary tag/count mismatch")
        if summary.get("dataset_sha256") != expected_dataset_sha:
            raise ValueError("GSM8K full dataset SHA mismatch")
        if summary.get("request_batching") != GSM_BATCHING_CONTRACT:
            raise ValueError("GSM8K request batching contract mismatch")
        if summary.get("correct") != sum(value[1] for value in rows.values()):
            raise ValueError("GSM8K summary correct count mismatch")
        summary_accuracy = summary.get("accuracy")
        if (
            not isinstance(summary_accuracy, (int, float))
            or isinstance(summary_accuracy, bool)
            or not math.isfinite(float(summary_accuracy))
            or not math.isclose(float(summary_accuracy), accuracy, abs_tol=1e-15)
        ):
            raise ValueError("GSM8K summary accuracy mismatch")
        if summary.get("csv_sha256") != sha256(csv_path):
            raise ValueError("GSM8K CSV SHA mismatch")
        return rows
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ArtifactError(scope, f"malformed GSM8K {tag}: {error}") from error


def _per_prompt_nll(rows: Sequence[dict], window: ScoreWindow) -> list[float]:
    return [
        -statistics.fmean(float(value) for value in row["logprobs"][window.start:window.stop])
        for row in rows
    ]


def _per_prompt_target_error(rows: Sequence[dict], window: ScoreWindow) -> list[float]:
    return [
        sum(
            top1 != target
            for top1, target in zip(
                row["top1_ids"][window.start:window.stop],
                row["token_ids"][window.start:window.stop],
            )
        ) / window.width
        for row in rows
    ]


def _pair_metrics(left: Sequence[dict], right: Sequence[dict], window: ScoreWindow) -> dict[str, list[float]]:
    mean_values, p95_values, flip_values = [], [], []
    for row_left, row_right in zip(left, right):
        deltas = [
            abs(float(a) - float(b))
            for a, b in zip(
                row_left["logprobs"][window.start:window.stop],
                row_right["logprobs"][window.start:window.stop],
            )
        ]
        mean_values.append(statistics.fmean(deltas))
        p95_values.append(nearest_rank(deltas, 0.95))
        flip_values.append(
            sum(
                a != b
                for a, b in zip(
                    row_left["top1_ids"][window.start:window.stop],
                    row_right["top1_ids"][window.start:window.stop],
                )
            ) / window.width
        )
    return {"mean": mean_values, "p95": p95_values, "flip": flip_values}


def _cluster_reduce(samples: Sequence[Sequence[float]], reducer: Callable[[Iterable[float]], float]) -> list[float]:
    if not samples or any(len(sample) != EXPECTED_PROMPTS for sample in samples):
        raise QualityContractError("prompt-cluster samples have inconsistent shapes")
    return [float(reducer(sample[index] for sample in samples)) for index in range(EXPECTED_PROMPTS)]


def _record_upper_gate(
    recorder: GateRecorder,
    name: str,
    values: Sequence[float],
    *,
    margin: float,
    cluster_kind: str,
    gate_class: str,
) -> dict[str, object]:
    expected_clusters = EXPECTED_PROMPTS if cluster_kind == "prompt" else EXPECTED_QUESTIONS
    result = assess_upper_noninferiority(
        values,
        margin=margin,
        cluster_kind=cluster_kind,
        expected_clusters=expected_clusters,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    detail = result.to_dict()
    recorder.add(name, result.passed, detail, gate_class=gate_class)
    return detail


def _control_admission(recorder: GateRecorder, scores: dict[str, list[dict]], tags: Sequence[str], label: str) -> dict[str, object]:
    pairs = tuple(itertools.combinations(tags, 2))
    bands: dict[str, object] = {}
    for window in WINDOW_VIEWS:
        pair_metrics = [_pair_metrics(scores[left], scores[right], window) for left, right in pairs]
        strict_caps = {
            "mean-abs-logprob": (
                _cluster_reduce([item["mean"] for item in pair_metrics], statistics.median),
                CAPS.control_mean_abs_logprob,
            ),
            "p95-abs-logprob": (
                _cluster_reduce([item["p95"] for item in pair_metrics], statistics.median),
                CAPS.control_p95_abs_logprob,
            ),
            "top1-flip": (
                _cluster_reduce([item["flip"] for item in pair_metrics], statistics.median),
                CAPS.control_top1_flip,
            ),
        }
        deep_prefix_caps = {
            "mean-abs-logprob": CAPS.deepep_prefix_mean_abs_logprob,
            "p95-abs-logprob": CAPS.deepep_prefix_p95_abs_logprob,
            "top1-flip": CAPS.deepep_prefix_top1_flip,
        }
        bands[window.name] = {}
        for metric_name, (values, split_margin) in strict_caps.items():
            is_gate = label == "split" or window == WINDOW_VIEWS[0]
            margin = (
                deep_prefix_caps[metric_name] if label == "deepep" else split_margin
            )
            if not is_gate:
                bands[window.name][metric_name] = {
                    "blocking": False,
                    "estimate": statistics.fmean(values),
                }
                continue
            detail = _record_upper_gate(
                recorder,
                f"1-{label}-{window.name}-{metric_name}-admission",
                values,
                margin=margin,
                cluster_kind="prompt",
                gate_class="environment",
            )
            bands[window.name][metric_name] = {
                "blocking": True,
                "margin": margin,
                "gate_upper95": detail["bootstrap"]["gate_upper95"],
            }
    validate_window_coverage(bands, f"{label} control admission")
    return bands


def _candidate_score_gates(recorder: GateRecorder, scores: dict[str, list[dict]]) -> None:
    fused_pairs = tuple(itertools.combinations(FUSED_TAGS, 2))
    fused_split_pairs = tuple(itertools.product(FUSED_TAGS, SPLIT_TAGS))
    covered = []
    for window in WINDOW_VIEWS:
        covered.append(window.name)
        for prefix, pairs, reducer, caps in (
            (
                "2-fused-self",
                fused_pairs,
                statistics.median,
                (CAPS.fused_self_mean_abs_logprob, CAPS.fused_self_p95_abs_logprob, CAPS.fused_self_top1_flip),
            ),
            (
                "3-fused-split",
                fused_split_pairs,
                statistics.median,
                (CAPS.fused_split_mean_abs_logprob, CAPS.fused_split_p95_abs_logprob, CAPS.fused_split_top1_flip),
            ),
        ):
            pair_metrics = [_pair_metrics(scores[left], scores[right], window) for left, right in pairs]
            for metric_name, key, margin in zip(
                ("mean-abs-logprob", "p95-abs-logprob", "top1-flip"),
                ("mean", "p95", "flip"),
                caps,
            ):
                _record_upper_gate(
                    recorder,
                    f"{prefix}-{window.name}-{metric_name}",
                    _cluster_reduce([item[key] for item in pair_metrics], reducer),
                    margin=margin,
                    cluster_kind="prompt",
                    gate_class="candidate",
                )

        family_metrics = {}
        for family, tags in (("fused", FUSED_TAGS), ("split", SPLIT_TAGS), ("deepep", DEEP_TAGS)):
            family_metrics[family] = {
                "nll": _cluster_reduce([_per_prompt_nll(scores[tag], window) for tag in tags], statistics.fmean),
                "target_error": _cluster_reduce([_per_prompt_target_error(scores[tag], window) for tag in tags], statistics.fmean),
            }
        for reference in ("split", "deepep"):
            for metric, margin in (
                ("nll", CAPS.directional_delta_nll),
                ("target_error", CAPS.directional_delta_target_error),
            ):
                values = [
                    candidate - baseline
                    for candidate, baseline in zip(family_metrics["fused"][metric], family_metrics[reference][metric])
                ]
                _record_upper_gate(
                    recorder,
                    f"4-fused-{reference}-{window.name}-delta-{metric.replace('_', '-')}",
                    values,
                    margin=margin,
                    cluster_kind="prompt",
                    gate_class="candidate",
                )
    validate_window_coverage(covered, "candidate score gates")


def _gsm_accuracy(gsm: dict[str, dict[int, tuple[str, int]]], tag: str) -> float:
    return statistics.fmean(gsm[tag][index][1] for index in range(EXPECTED_QUESTIONS))


def _record_gsm_span(recorder: GateRecorder, gsm: dict[str, dict[int, tuple[str, int]]], tags: Sequence[str], label: str, gate_class: str) -> None:
    accuracies = {tag: _gsm_accuracy(gsm, tag) for tag in tags}
    span = max(accuracies.values()) - min(accuracies.values())
    recorder.add(
        f"5-gsm8k-{label}-accuracy-span",
        span <= CAPS.gsm8k_accuracy_span or math.isclose(span, CAPS.gsm8k_accuracy_span, abs_tol=1e-12),
        {"accuracies": accuracies, "span": span, "margin": CAPS.gsm8k_accuracy_span},
        gate_class=gate_class,
    )


def _gsm8k_candidate_gates(recorder: GateRecorder, gsm: dict[str, dict[int, tuple[str, int]]]) -> None:
    family_error = {
        family: [statistics.fmean(1 - gsm[tag][index][1] for tag in tags) for index in range(EXPECTED_QUESTIONS)]
        for family, tags in (("fused", FUSED_TAGS), ("split", SPLIT_TAGS), ("deepep", DEEP_TAGS))
    }
    for reference in ("split", "deepep"):
        delta = [candidate - baseline for candidate, baseline in zip(family_error["fused"], family_error[reference])]
        _record_upper_gate(
            recorder,
            f"5-gsm8k-fused-{reference}-1pp",
            delta,
            margin=CAPS.gsm8k_delta_error,
            cluster_kind="question",
            gate_class="candidate",
        )
    _record_gsm_span(recorder, gsm, FUSED_TAGS, "fused", "candidate")


def _receipt_path(qdir: str, tag: str) -> Path:
    _, stage = RECEIPT_CONTRACT[tag]
    return Path(qdir) / f"path-receipt-{tag}-{stage}.json"


def _check_receipt_header(qdir: str, tag: str) -> tuple[dict, int]:
    scope = _scope_for_tag(tag) if tag != "t-s" else "control"
    expected_mode, expected_stage = RECEIPT_CONTRACT[tag]
    receipt_path = _receipt_path(qdir, tag)
    receipt = _load_json(receipt_path, scope)
    try:
        if (
            receipt.get("schema") != "phase2-v4-session-receipt-v1"
            or receipt.get("mode") != expected_mode
            or receipt.get("tag") != tag
            or receipt.get("stage") != expected_stage
            or receipt.get("rc") != 0
            or receipt.get("path_ok") != 1
        ):
            raise ValueError("schema/mode/tag/stage/rc/path mismatch")
        path_config = Path(qdir) / f"path-config-receipt-{tag}-{expected_stage}.json"
        if sha256(_require_file(path_config, scope)) != receipt.get("path_config_sha256"):
            raise ValueError("path-config SHA mismatch")
        path_payload = _load_json(path_config, scope)
        expected_free_run = int(tag in {"d1", "s1", "f1"})
        if tag == "t-s":
            expected_free_run = 0
        if (
            path_payload.get("schema") != "phase2-v4-path-config-v1"
            or path_payload.get("mode") != expected_mode
            or path_payload.get("tag") != tag
            or path_payload.get("stage") != expected_stage
            or path_payload.get("port") != 30061
            or path_payload.get("free_run") != expected_free_run
            or path_payload.get("startup_ok") != 1
            or path_payload.get("prefill_backend") != "disabled"
            or path_payload.get("cuda_graph") is not False
            or path_payload.get("radix_cache") is not False
            or path_payload.get("overlap_schedule") is not False
            or path_payload.get("flashinfer_autotune") is not False
            or path_payload.get("attention_backend") != "dsv4"
        ):
            raise ValueError("path-config content contract mismatch")
        if receipt.get("free_run") != expected_free_run:
            raise ValueError("receipt free_run mismatch")
        return receipt, expected_free_run
    except (AttributeError, TypeError, ValueError) as error:
        raise ArtifactError(scope, f"malformed receipt {tag}: {error}") from error


def _check_receipt(qdir: str, tag: str) -> dict:
    scope = _scope_for_tag(tag) if tag != "t-s" else "control"
    expected_mode, expected_stage = RECEIPT_CONTRACT[tag]
    receipt, expected_free_run = _check_receipt_header(qdir, tag)
    try:
        client_rc = receipt.get("client_rc")
        if not isinstance(client_rc, dict) or set(client_rc) != {
            "target", "gsm8k", "teacher512", "free_run_info"
        }:
            raise ValueError("client_rc key set mismatch")
        expected_rc = (
            {"target": 0, "gsm8k": None, "teacher512": None, "free_run_info": None}
            if expected_stage == "target-build"
            else {
                "target": None,
                "gsm8k": 0,
                "teacher512": 0,
                "free_run_info": 0 if expected_free_run else None,
            }
        )
        if client_rc != expected_rc:
            raise ValueError("client_rc value contract mismatch")
        outputs = receipt.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != {
            "targets_freeze", "gsm8k_csv", "gsm8k_json", "teacher512", "free_run_info"
        }:
            raise ValueError("receipt output key set mismatch")
        required = {"targets_freeze": TARGETS_FILE}
        if expected_stage == "bundle":
            required.update(
                {
                    "gsm8k_csv": f"gsm8k-{tag}.csv",
                    "gsm8k_json": f"gsm8k-{tag}.json",
                    "teacher512": f"teacher512-{tag}.json",
                }
            )
            if tag in {"d1", "s1", "f1"}:
                required["free_run_info"] = f"free-run-info-{tag}.json"
        for key, filename in required.items():
            artifact = _require_file(Path(qdir) / filename, scope)
            if outputs.get(key) != sha256(artifact):
                raise ValueError(f"output SHA mismatch for {key}")
        for key in set(outputs) - set(required):
            if outputs[key] is not None:
                raise ValueError(f"unexpected output SHA for {key}")
        return receipt
    except (AttributeError, TypeError, ValueError) as error:
        raise ArtifactError(scope, f"malformed receipt {tag}: {error}") from error


def _check_exact_receipt_set(qdir: str) -> None:
    expected = {_receipt_path(qdir, tag).name for tag in RECEIPT_CONTRACT}
    observed = {
        path.name for path in Path(qdir).glob("path-receipt-*.json")
    }
    if observed != expected or len(observed) != 10:
        raise ArtifactError(
            "infrastructure",
            f"receipt set must be exactly 10; missing={sorted(expected-observed)} extra={sorted(observed-expected)}",
        )


def _check_preflight(qdir: str, expected_head: str) -> dict:
    payload = _load_json(Path(qdir) / "preflight-manifest.json", "infrastructure")
    try:
        numeric = payload.get("numeric_audit", {})
        if not (
            payload.get("verified") is True
            and len(expected_head) == 40
            and payload.get("sglang_head") == expected_head
            and numeric.get("verified") is True
            and numeric.get("min_exact", 0) >= 0.999
            and numeric.get("max_relative_l2", math.inf) <= 1e-3
            and numeric.get("live_records") == 344
            and numeric.get("live_layers") == 43
        ):
            raise ValueError("preflight/head/direct-numeric contract failed")
        return payload
    except (AttributeError, TypeError, ValueError) as error:
        raise ArtifactError("infrastructure", f"preflight failed: {error}") from error


def _check_power_input(preflight: dict) -> tuple[dict, str]:
    expected_assets = _load_json(QUALITY_SOURCE / "expected_assets.json", "infrastructure")
    try:
        spec = expected_assets["phase2_v4_power_input"]
        relative_path = spec["path"]
        expected_sha = spec["sha256"]
        if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
            raise ValueError("power input path must be relative to quality source")
        if not _is_sha256(expected_sha):
            raise ValueError("power input SHA is not frozen")
        path = _require_file(QUALITY_SOURCE / relative_path, "infrastructure")
        observed_sha = sha256(path)
        if observed_sha != expected_sha:
            raise ValueError("power input SHA mismatch")
        manifest_power = preflight.get("power_analysis", {})
        if (
            manifest_power.get("verified") is not True
            or manifest_power.get("actual_sha256") != expected_sha
            or manifest_power.get("expected_sha256") != expected_sha
        ):
            raise ValueError("preflight did not attest the frozen power input")
        # Preflight has already recomputed the 10k x 10k analysis directly
        # from the bounded raw D/S/P files.  Re-running it at every checkpoint
        # would be expensive and would not add independence; consume only the
        # current preflight process' attestation, never a manifest summary.
        assessment = manifest_power.get("assessment")
        if (
            not isinstance(assessment, dict)
            or assessment.get("schema") != "phase2-v4-raw-power-assessment-v2"
            or assessment.get("raw_manifest_sha256") != expected_sha
            or assessment.get("overall_pass") is not True
        ):
            error = assessment.get("error") if isinstance(assessment, dict) else assessment
            raise ValueError(f"power assessment NO-GO: {error}")
        return assessment, observed_sha
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError(
            "infrastructure",
            f"Phase 2 v4 power input unavailable/NO-GO: {error}",
        ) from error


def _collect_free_run_info(qdir: str) -> dict[str, object]:
    info: dict[str, object] = {"blocking": False, "tags": {}}
    for tag in ("d1", "s1", "f1"):
        scope = _scope_for_tag(tag)
        path = Path(qdir) / f"free-run-info-{tag}.json"
        if not path.is_file():
            continue
        try:
            payload = _load_json(path, scope)
            if payload.get("protocol") != PROTOCOL or payload.get("kind") != "free-run-info" or payload.get("tag") != tag:
                raise ValueError("free-run protocol/kind/tag mismatch")
            info["tags"][tag] = {
                "sha256": sha256(path),
                "comparisons": payload.get("info", {}),
            }
        except (ArtifactError, AttributeError, ValueError) as error:
            info["tags"][tag] = {"error": str(error)}
    return info


def _validate_free_run_artifact(
    qdir: str, tag: str, receipt: dict, targets: dict
) -> None:
    if tag not in {"d1", "s1", "f1"}:
        return
    scope = _scope_for_tag(tag)
    path = Path(qdir) / f"free-run-info-{tag}.json"
    payload = _load_json(path, scope)
    try:
        validate_free_run(payload)
        expected_mode = RECEIPT_CONTRACT[tag][0]
        if payload.get("mode") != expected_mode or payload.get("tag") != tag:
            raise ValueError("free-run mode/tag mismatch")
        if payload.get("targets_sha256") != sha256(Path(qdir) / TARGETS_FILE):
            raise ValueError("free-run targets SHA mismatch")
        if payload.get("path_config_receipt_sha256") != receipt["path_config_sha256"]:
            raise ValueError("free-run path-config SHA mismatch")
        expected_prompts = [
            {
                "prompt_id": row["prompt_id"],
                "prompt_sha256": row["prompt_sha256"],
                "prompt_ids": row["prompt_ids"],
            }
            for row in targets["rows"][:16]
        ]
        if payload.get("prompts") != expected_prompts:
            raise ValueError("free-run prompts do not match frozen first 16")
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ArtifactError(scope, f"malformed free-run-info-{tag}: {error}") from error


def _artifact_manifest(qdir: str) -> dict[str, str]:
    return {
        path.name: sha256(path)
        for path in sorted(Path(qdir).iterdir())
        if path.suffix in {".json", ".csv"} and path.name != "quality-gate-verdict.json"
    }


def _atomic_json_no_clobber(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=1, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    except FileExistsError as error:
        raise ArtifactError("infrastructure", f"refusing to replace frozen output {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _emit_report(
    qdir: str,
    recorder: GateRecorder,
    *,
    filename: str,
    status: str,
    failure: str | None,
    control_bands: dict[str, object] | None = None,
    power: dict[str, object] | None = None,
) -> int:
    report = {
        "protocol": "phase2-v4-fixed-margin-cluster-bootstrap",
        "status": status,
        "all_pass": status == "PASS",
        "failure": failure,
        "gates": recorder.gates,
        "fixed_caps": CAPS.__dict__,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "gate_quantile": 0.95,
            "report_interval": [0.025, 0.975],
            "units": {"teacher": "prompt", "gsm8k": "question"},
        },
        "windows": [window.__dict__ for window in WINDOW_VIEWS],
        "control_bands_information_only": control_bands or {},
        "power_assessment": power or {},
        "free_run_information_only": _collect_free_run_info(qdir),
        "manifest": _artifact_manifest(qdir),
    }
    output = Path(qdir) / filename
    _atomic_json_no_clobber(output, report)
    print(f"QUALITY_GATE_VERDICT|status={status}|report={filename}|sha256={sha256(output)}", flush=True)
    return 0 if status == "PASS" else (1 if status == "FAIL" else 2)


def _load_family(qdir: str, tags: Sequence[str], targets: dict) -> tuple[dict[str, list[dict]], dict[str, dict[int, tuple[str, int]]]]:
    scores, gsm = {}, {}
    for tag in tags:
        receipt = _check_receipt(qdir, tag)
        scores[tag] = load_score(tag, qdir, targets, receipt)
        gsm[tag] = load_gsm8k(tag, qdir)
        _validate_free_run_artifact(qdir, tag, receipt, targets)
    return scores, gsm


def _validate_ci_gate_coverage(recorder: GateRecorder) -> None:
    observed = {
        gate["gate"]
        for gate in recorder.gates
        if isinstance(gate.get("detail"), dict) and "bootstrap" in gate["detail"]
    }
    if observed != EXPECTED_POWER_GATE_NAMES:
        raise QualityContractError(
            f"CI gate matrix incomplete: missing={sorted(EXPECTED_POWER_GATE_NAMES-observed)} extra={sorted(observed-EXPECTED_POWER_GATE_NAMES)}"
        )


def _checkpoint(qdir: str, expected_head: str, checkpoint: str) -> int:
    recorder = GateRecorder()
    output = "deepep-env-checkpoint.json" if checkpoint == "deepep-env" else "split-aa-freeze.json"
    try:
        preflight = _check_preflight(qdir, expected_head)
        power, power_sha = _check_power_input(preflight)
        _check_receipt(qdir, "t-s")
        targets = load_targets(qdir)
        scores, gsm = _load_family(qdir, DEEP_TAGS, targets)
        bands = {"deepep": _control_admission(recorder, scores, DEEP_TAGS, "deepep")}
        _record_gsm_span(recorder, gsm, DEEP_TAGS, "deepep", "environment")
        if checkpoint == "split-aa-freeze":
            premature = []
            for tag in FUSED_TAGS:
                for pattern in (
                    f"teacher512-{tag}.json",
                    f"gsm8k-{tag}.csv",
                    f"gsm8k-{tag}.json",
                    f"free-run-info-{tag}.json",
                    f"path-config-receipt-{tag}-bundle.json",
                    f"path-receipt-{tag}-bundle.json",
                ):
                    if (Path(qdir) / pattern).exists():
                        premature.append(pattern)
            if premature:
                raise ArtifactError(
                    "infrastructure",
                    f"candidate artifacts exist before split freeze: {sorted(premature)}",
                )
            split_scores, split_gsm = _load_family(qdir, SPLIT_TAGS, targets)
            scores.update(split_scores)
            gsm.update(split_gsm)
            bands["split"] = _control_admission(recorder, scores, SPLIT_TAGS, "split")
            _record_gsm_span(recorder, gsm, SPLIT_TAGS, "split", "environment")
        status = "PASS" if recorder.all_pass("environment") else "INVALID"
        power = {**power, "input_sha256": power_sha, "targets_sha256": sha256(Path(qdir) / TARGETS_FILE)}
        return _emit_report(
            qdir,
            recorder,
            filename=output,
            status=status,
            failure=None if status == "PASS" else "control admission failed",
            control_bands=bands,
            power=power,
        )
    except (ArtifactError, QualityContractError) as error:
        recorder.add("0-checkpoint-contract", False, {"error": str(error)}, gate_class="environment")
        return _emit_report(qdir, recorder, filename=output, status="INVALID", failure=str(error))


def _check_frozen_checkpoints(qdir: str) -> None:
    deep = _load_json(Path(qdir) / "deepep-env-checkpoint.json", "infrastructure")
    split_path = Path(qdir) / "split-aa-freeze.json"
    split = _load_json(split_path, "infrastructure")
    expected_split_sha = os.environ.get("SPLIT_AA_FREEZE_SHA")
    if deep.get("status") != "PASS" or split.get("status") != "PASS":
        raise ArtifactError("infrastructure", "control checkpoint did not PASS")
    if not _is_sha256(expected_split_sha) or sha256(split_path) != expected_split_sha:
        raise ArtifactError("infrastructure", "split checkpoint SHA is absent or changed")
    frozen_manifest = split.get("manifest")
    if not isinstance(frozen_manifest, dict) or not frozen_manifest:
        raise ArtifactError("infrastructure", "split checkpoint lacks control manifest")
    for name, frozen_sha in frozen_manifest.items():
        current = Path(qdir) / name
        if not _is_sha256(frozen_sha) or not current.is_file() or sha256(current) != frozen_sha:
            raise ArtifactError(
                "infrastructure", f"control artifact changed after split freeze: {name}"
            )


def _final(qdir: str, expected_head: str) -> int:
    recorder = GateRecorder()
    control_bands: dict[str, object] = {}
    power: dict[str, object] = {}
    target_path_hit = False
    controls_stable = False
    try:
        preflight = _check_preflight(qdir, expected_head)
        power, power_sha = _check_power_input(preflight)
        _check_frozen_checkpoints(qdir)
        _check_receipt(qdir, "t-s")
        targets = load_targets(qdir)
        deep_scores, deep_gsm = _load_family(qdir, DEEP_TAGS, targets)
        split_scores, split_gsm = _load_family(qdir, SPLIT_TAGS, targets)
        scores = {**deep_scores, **split_scores}
        gsm = {**deep_gsm, **split_gsm}
        control_bands = {
            "deepep": _control_admission(recorder, scores, DEEP_TAGS, "deepep"),
            "split": _control_admission(recorder, scores, SPLIT_TAGS, "split"),
        }
        _record_gsm_span(recorder, gsm, DEEP_TAGS, "deepep", "environment")
        _record_gsm_span(recorder, gsm, SPLIT_TAGS, "split", "environment")
        controls_stable = recorder.all_pass("environment")
        if not controls_stable:
            return _emit_report(
                qdir,
                recorder,
                filename="quality-gate-verdict.json",
                status="INVALID",
                failure="control environment admission failed",
                control_bands=control_bands,
                power={**power, "input_sha256": power_sha},
            )

        _check_receipt_header(qdir, "f1")
        target_path_hit = True
        f1_receipt = _check_receipt(qdir, "f1")
        scores["f1"] = load_score("f1", qdir, targets, f1_receipt)
        gsm["f1"] = load_gsm8k("f1", qdir)
        _validate_free_run_artifact(qdir, "f1", f1_receipt, targets)
        for tag in ("f2", "f3"):
            receipt = _check_receipt(qdir, tag)
            scores[tag] = load_score(tag, qdir, targets, receipt)
            gsm[tag] = load_gsm8k(tag, qdir)
        _check_exact_receipt_set(qdir)
        _candidate_score_gates(recorder, scores)
        _gsm8k_candidate_gates(recorder, gsm)
        _validate_ci_gate_coverage(recorder)
        status = "PASS" if recorder.all_pass() else "FAIL"
        return _emit_report(
            qdir,
            recorder,
            filename="quality-gate-verdict.json",
            status=status,
            failure=None if status == "PASS" else "candidate quality gate failed",
            control_bands=control_bands,
            power={**power, "input_sha256": power_sha},
        )
    except (ArtifactError, QualityContractError) as error:
        scope = error.scope if isinstance(error, ArtifactError) else "infrastructure"
        status = classify_failure(scope=scope, target_path_hit=target_path_hit, controls_stable=controls_stable)
        recorder.add(
            "0-artifact-contract",
            False,
            {"scope": scope, "error": str(error)},
            gate_class="candidate" if status == "FAIL" else "environment",
        )
        return _emit_report(
            qdir,
            recorder,
            filename="quality-gate-verdict.json",
            status=status,
            failure=str(error),
            control_bands=control_bands,
            power=power,
        )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    qdir = arguments[0] if arguments else DEFAULT_QDIR
    expected_head = arguments[1] if len(arguments) > 1 else ""
    checkpoint = arguments[2] if len(arguments) > 2 else "final"
    if checkpoint in {"deepep-env", "split-aa-freeze"}:
        return _checkpoint(qdir, expected_head, checkpoint)
    if checkpoint != "final":
        print(f"unknown checkpoint {checkpoint}", file=sys.stderr)
        return 2
    return _final(qdir, expected_head)


if __name__ == "__main__":
    raise SystemExit(main())
