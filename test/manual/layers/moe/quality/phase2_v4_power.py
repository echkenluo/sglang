"""Raw, candidate-blind power analysis for the Phase 2 v4 quality gate.

The formal asset is a manifest over independent DeepEP (D1--D3), split
baseline (S1--S3), and held-out split pseudo-candidate (P1--P3) sessions.
Reported rates are deliberately not an input: every count and digest is
recomputed from raw Teacher512 and GSM8K rows before the fused runs exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
from typing import Callable, Iterable, Mapping, Sequence


EXPECTED_PROMPTS = 128
EXPECTED_QUESTIONS = 1314
EXPECTED_TARGET_TOKENS = 512
REPEATS_PER_MODE = 3
BOOTSTRAP_REPLICATES = 10_000
MONTE_CARLO_REPLICATES = 10_000
BOOTSTRAP_SEED = 196_944_573
MONTE_CARLO_SEED = 196_944_574
POWER_INPUT_SCHEMA = "phase2-v4-raw-power-input-v2"
POWER_ASSESSMENT_SCHEMA = "phase2-v4-raw-power-assessment-v2"
RAW_DIRECTORY_NAME = "phase2-v4-power-raw"
CI_GATE_COUNT = 70
SPAN_GATE_COUNT = 3
BLOCKING_GATE_COUNT = 73
EFFECT_SCENARIO_COUNT = 22


@dataclass(frozen=True)
class ScoreWindow:
    name: str
    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start


WINDOW_VIEWS = (
    ScoreWindow("t000_063", 0, 64),
    ScoreWindow("t064_127", 64, 128),
    ScoreWindow("t128_255", 128, 256),
    ScoreWindow("t256_511", 256, 512),
    ScoreWindow("t000_511", 0, 512),
)


@dataclass(frozen=True)
class V4Caps:
    control_mean_abs_logprob: float = 0.02
    control_p95_abs_logprob: float = 0.10
    control_top1_flip: float = 0.02
    deepep_prefix_mean_abs_logprob: float = 0.08
    deepep_prefix_p95_abs_logprob: float = 0.30
    deepep_prefix_top1_flip: float = 0.02
    fused_self_mean_abs_logprob: float = 0.02
    fused_self_p95_abs_logprob: float = 0.10
    fused_self_top1_flip: float = 0.02
    fused_split_mean_abs_logprob: float = 0.02
    fused_split_p95_abs_logprob: float = 0.10
    fused_split_top1_flip: float = 0.02
    directional_delta_nll: float = 0.02
    directional_delta_target_error: float = 0.02
    gsm8k_delta_error: float = 0.01
    gsm8k_accuracy_span: float = 0.01


CAPS = V4Caps()


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    report_lower025: float
    gate_upper95: float
    report_upper975: float
    clusters: int
    cluster_kind: str
    replicates: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UpperGateResult:
    passed: bool
    margin: float
    bootstrap: BootstrapResult

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "margin": self.margin,
            "bootstrap": self.bootstrap.to_dict(),
        }


@dataclass(frozen=True)
class SimulationConfig:
    """Execution parameters; the formal CLI/preflight never accepts overrides."""

    outer_trials: int = MONTE_CARLO_REPLICATES
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    outer_chunk: int = 32


FORMAL_SIMULATION = SimulationConfig()
FORMAL_OUTER_CHUNK = FORMAL_SIMULATION.outer_chunk


class QualityContractError(ValueError):
    """An input cannot satisfy the frozen Phase 2 v4 contract."""


class PowerBackendError(RuntimeError):
    """The mandatory vectorized backend is unavailable or not accelerated."""


def nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise QualityContractError("cannot take a quantile of an empty sample")
    if not 0 <= quantile <= 1:
        raise QualityContractError(f"invalid quantile {quantile}")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _validate_cluster_values(
    values: Iterable[float], *, cluster_kind: str, expected_clusters: int
) -> list[float]:
    if cluster_kind not in {"prompt", "question"}:
        raise QualityContractError(
            "bootstrap cluster_kind must be 'prompt' or 'question'; "
            f"token-level resampling is forbidden, got {cluster_kind!r}"
        )
    required = EXPECTED_PROMPTS if cluster_kind == "prompt" else EXPECTED_QUESTIONS
    if expected_clusters != required:
        raise QualityContractError(
            f"{cluster_kind} bootstrap requires {required} clusters, "
            f"got contract value {expected_clusters}"
        )
    sample = [float(value) for value in values]
    if len(sample) != required:
        raise QualityContractError(
            f"{cluster_kind} bootstrap requires exactly {required} cluster "
            f"values, got {len(sample)}"
        )
    if not all(math.isfinite(value) for value in sample):
        raise QualityContractError("bootstrap cluster values must all be finite")
    return sample


def _scalar_bootstrap_means(
    values: Sequence[float], *, replicates: int, seed: int
) -> list[float]:
    if replicates < 100:
        raise QualityContractError("bootstrap requires at least 100 replicates")
    count = len(values)
    return [
        math.fsum(value * weight for value, weight in zip(values, weights)) / count
        for weights in iter_bootstrap_count_rows(count, replicates, seed)
    ]


def iter_bootstrap_count_rows(
    cluster_count: int, replicates: int, seed: int
):
    """Yield exact ``random.Random.randrange`` bootstrap multiplicities.

    ``randrange`` deliberately retains CPython's rejection-sampling path; a
    NumPy RNG is not an interchangeable source for the frozen evaluator.
    """

    if cluster_count < 1 or replicates < 1:
        raise QualityContractError("bootstrap count matrix dimensions must be positive")
    rng = random.Random(seed)
    for _ in range(replicates):
        counts = [0] * cluster_count
        for _ in range(cluster_count):
            counts[rng.randrange(cluster_count)] += 1
        yield counts


def upper_margin_predicate(value: float, margin: float) -> bool:
    """Frozen one-sided predicate shared by scalar and vector evaluators."""

    return value <= margin or abs(value - margin) <= 1e-12


def exact_margin_injection(
    values: Sequence[float], margin: float
) -> tuple[float, float, bool]:
    """Return shift, injected estimand, and the degenerate-source marker."""

    if not values or margin <= 0 or not all(math.isfinite(value) for value in values):
        raise QualityContractError("effect injection requires finite values and margin>0")
    source_mean = statistics.fmean(values)
    source_sd = math.sqrt(
        statistics.fmean((float(value) - source_mean) ** 2 for value in values)
    )
    degenerate = source_sd <= 1e-15
    target = margin * 1.25 if degenerate else margin
    return target - source_mean, target, degenerate


def clustered_bootstrap_mean(
    values: Iterable[float],
    *,
    cluster_kind: str,
    expected_clusters: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapResult:
    """Scalar reference implementation used by the production evaluator."""

    sample = _validate_cluster_values(
        values, cluster_kind=cluster_kind, expected_clusters=expected_clusters
    )
    bootstrap_means = _scalar_bootstrap_means(
        sample, replicates=replicates, seed=seed
    )
    return BootstrapResult(
        estimate=statistics.fmean(sample),
        report_lower025=nearest_rank(bootstrap_means, 0.025),
        gate_upper95=nearest_rank(bootstrap_means, 0.95),
        report_upper975=nearest_rank(bootstrap_means, 0.975),
        clusters=len(sample),
        cluster_kind=cluster_kind,
        replicates=replicates,
        seed=seed,
    )


def assess_upper_noninferiority(
    values: Iterable[float],
    *,
    margin: float,
    cluster_kind: str,
    expected_clusters: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> UpperGateResult:
    if margin < 0:
        raise QualityContractError("gate margin must be non-negative")
    result = clustered_bootstrap_mean(
        values,
        cluster_kind=cluster_kind,
        expected_clusters=expected_clusters,
        replicates=replicates,
        seed=seed,
    )
    passed = upper_margin_predicate(result.gate_upper95, margin)
    return UpperGateResult(passed=passed, margin=margin, bootstrap=result)


def validate_repeat_tags(tags: Iterable[str], family: str) -> tuple[str, ...]:
    expected = tuple(f"{family}{index}" for index in range(1, 4))
    observed = tuple(sorted(tags))
    if observed != expected:
        raise QualityContractError(
            f"{family} requires three independent repeats {expected}, got {observed}"
        )
    return expected


def classify_failure(
    *, scope: str, target_path_hit: bool, controls_stable: bool
) -> str:
    if scope not in {"candidate", "control", "infrastructure"}:
        raise QualityContractError(f"unknown failure scope {scope!r}")
    if scope == "candidate" and target_path_hit and controls_stable:
        return "FAIL"
    return "INVALID"


def validate_window_views(windows: Sequence[ScoreWindow] = WINDOW_VIEWS) -> None:
    expected = ((0, 64), (64, 128), (128, 256), (256, 512), (0, 512))
    observed = tuple((window.start, window.stop) for window in windows)
    if observed != expected:
        raise QualityContractError(
            f"frozen score windows changed: {observed} != {expected}"
        )


def validate_window_coverage(names: Iterable[str], label: str) -> None:
    expected = {window.name for window in WINDOW_VIEWS}
    observed_list = list(names)
    observed = set(observed_list)
    if observed != expected or len(observed_list) != len(expected):
        raise QualityContractError(
            f"{label} must cover all five score windows exactly once; "
            f"missing={sorted(expected-observed)} extra={sorted(observed-expected)}"
        )


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_git_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_power_gate_names() -> set[str]:
    names: set[str] = set()
    for window in WINDOW_VIEWS:
        for metric in ("mean-abs-logprob", "p95-abs-logprob", "top1-flip"):
            names.add(f"1-split-{window.name}-{metric}-admission")
            if window == WINDOW_VIEWS[0]:
                names.add(f"1-deepep-{window.name}-{metric}-admission")
        for prefix in ("2-fused-self", "3-fused-split"):
            for metric in ("mean-abs-logprob", "p95-abs-logprob", "top1-flip"):
                names.add(f"{prefix}-{window.name}-{metric}")
        for reference in ("split", "deepep"):
            names.add(f"4-fused-{reference}-{window.name}-delta-nll")
            names.add(f"4-fused-{reference}-{window.name}-delta-target-error")
    names.update(
        {
            "5-gsm8k-fused-split-1pp",
            "5-gsm8k-fused-deepep-1pp",
        }
    )
    return names


EXPECTED_POWER_GATE_NAMES = frozenset(_required_power_gate_names())
EFFECT_GATE_NAMES = tuple(
    sorted(
        name
        for name in EXPECTED_POWER_GATE_NAMES
        if name.startswith("4-") or name.startswith("5-gsm8k-fused-")
    )
)
SPAN_GATE_NAMES = (
    "5-gsm8k-deepep-accuracy-span",
    "5-gsm8k-split-accuracy-span",
    "5-gsm8k-fused-accuracy-span",
)
if len(EXPECTED_POWER_GATE_NAMES) != CI_GATE_COUNT:
    raise RuntimeError("internal CI gate count changed")
if len(EFFECT_GATE_NAMES) != EFFECT_SCENARIO_COUNT:
    raise RuntimeError("internal effect scenario count changed")


RAW_TAGS = tuple(f"{family}{index}" for family in "dsp" for index in range(1, 4))
RAW_CONTRACT = {
    "t-s": ("split", "target-build"),
    **{f"d{i}": ("deepep", "bundle") for i in range(1, 4)},
    **{f"s{i}": ("split", "bundle") for i in range(1, 4)},
    **{f"p{i}": ("split", "bundle") for i in range(1, 4)},
}


def _required_raw_filenames() -> frozenset[str]:
    names = {
        "targets-freeze.json",
        "path-config-receipt-t-s-target-build.json",
        "path-receipt-t-s-target-build.json",
    }
    for tag in RAW_TAGS:
        names.update(
            {
                f"teacher512-{tag}.json",
                f"gsm8k-{tag}.csv",
                f"gsm8k-{tag}.json",
                f"path-config-receipt-{tag}-bundle.json",
                f"path-receipt-{tag}-bundle.json",
            }
        )
    return frozenset(names)


REQUIRED_RAW_FILENAMES = _required_raw_filenames()
REQUIRED_SOURCE_SHA_KEYS = REQUIRED_RAW_FILENAMES
FORBIDDEN_REPORTED_KEYS = frozenset(
    {"families", "rates", "passes", "overall_pass", "single_gate_zero_effect"}
)
PROVENANCE_KEYS = frozenset(
    {
        "sglang_head",
        "mok_head",
        "dataset_sha256",
        "gsm8k_sha256",
        "tokenizer_sha256",
        "generator_sha256",
        "evaluator_sha256",
        "numpy_version",
        "openblas_version",
        "host_class",
    }
)


def validate_power_input(payload: object) -> dict:
    """Validate only manifest schema; raw bytes require a manifest path."""

    if not isinstance(payload, dict):
        raise QualityContractError("power input must be a JSON object")
    expected_keys = {
        "schema",
        "candidate_blind",
        "control_only",
        "source_root",
        "artifacts",
        "provenance",
    }
    if set(payload) != expected_keys:
        extra = sorted(set(payload) - expected_keys)
        missing = sorted(expected_keys - set(payload))
        raise QualityContractError(
            f"raw power manifest key set mismatch: missing={missing} extra={extra}"
        )
    if FORBIDDEN_REPORTED_KEYS.intersection(payload):
        raise QualityContractError("reported power rates are forbidden input")
    if payload.get("schema") != POWER_INPUT_SCHEMA:
        raise QualityContractError(f"power input schema must be {POWER_INPUT_SCHEMA}")
    if payload.get("candidate_blind") is not True:
        raise QualityContractError("power input must set candidate_blind=true")
    if payload.get("control_only") is not True:
        raise QualityContractError("power input must set control_only=true")
    source_root = payload.get("source_root")
    if source_root != RAW_DIRECTORY_NAME:
        raise QualityContractError(
            f"source_root must be the dedicated relative directory {RAW_DIRECTORY_NAME}"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != REQUIRED_RAW_FILENAMES:
        raise QualityContractError("artifact set must exactly match the raw D/S/P contract")
    if not all(_is_sha256(value) for value in artifacts.values()):
        raise QualityContractError("every raw artifact requires a full SHA256")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
        raise QualityContractError("provenance key set mismatch")
    for key, value in provenance.items():
        if key.endswith("_head"):
            if not _is_git_sha(value):
                raise QualityContractError(f"{key} must be a full git SHA")
        elif key in {"numpy_version", "openblas_version", "host_class"}:
            if not isinstance(value, str) or not value or len(value) > 128:
                raise QualityContractError(f"{key} must be a bounded nonempty string")
        elif not _is_sha256(value):
            raise QualityContractError(f"{key} must be a full SHA256")
    return payload


def _load_json(path: Path, label: str) -> object:
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise QualityContractError(f"malformed {label}: {error}") from error


def _ids_sha256(values: Sequence[int]) -> str:
    raw = json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _validate_targets(path: Path, provenance: Mapping[str, str]) -> dict:
    payload = _load_json(path, "targets-freeze.json")
    try:
        if not isinstance(payload, dict) or set(payload) != {
            "protocol", "seed", "dataset_sha256", "tokenizer_sha256",
            "generator_sha256", "request_batching", "rows",
        }:
            raise ValueError("target key set mismatch")
        if payload["protocol"] != "phase2-v4" or payload["seed"] != 196_944_571:
            raise ValueError("target protocol/seed mismatch")
        for target_key, provenance_key in (
            ("dataset_sha256", "dataset_sha256"),
            ("tokenizer_sha256", "tokenizer_sha256"),
            ("generator_sha256", "generator_sha256"),
        ):
            if payload[target_key] != provenance[provenance_key]:
                raise ValueError(f"target {target_key} provenance mismatch")
        if payload["request_batching"] != {
            "schema": "phase2-v4-request-batching-v1",
            "tokenize_endpoint": "/v1/tokenize",
            "tokenize_batch_size": 128,
            "tokenize_response_identity": "server-request-order-contract",
            "generate_endpoint": "/generate",
            "generate_batch_size": 16,
            "generate_response_identity": "request-order-and-rid-echo",
        }:
            raise ValueError("target batching contract mismatch")
        rows = payload["rows"]
        if not isinstance(rows, list) or len(rows) != EXPECTED_PROMPTS:
            raise ValueError("target row count mismatch")
        seen_ids: set[int] = set()
        seen_prompts: set[str] = set()
        seen_text: set[str] = set()
        for offset, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {
                "prompt_id", "text_sha", "prompt_ids", "prompt_len",
                "prompt_sha256", "target_ids", "target_sha256",
            }:
                raise ValueError(f"target row {offset} key set mismatch")
            prompt_id = row["prompt_id"]
            prompt_ids = row["prompt_ids"]
            target_ids = row["target_ids"]
            if (
                not isinstance(prompt_id, int) or isinstance(prompt_id, bool)
                or prompt_id < 0 or prompt_id in seen_ids
            ):
                raise ValueError(f"invalid/duplicate prompt_id at {offset}")
            if (
                not isinstance(row["prompt_len"], int)
                or not 256 <= row["prompt_len"] <= 2048
                or len(prompt_ids) != row["prompt_len"]
                or len(target_ids) != EXPECTED_TARGET_TOKENS
            ):
                raise ValueError(f"target token shape mismatch at {offset}")
            if not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in prompt_ids + target_ids):
                raise ValueError(f"invalid target token at {offset}")
            if _ids_sha256(prompt_ids) != row["prompt_sha256"] or _ids_sha256(target_ids) != row["target_sha256"]:
                raise ValueError(f"target ID SHA mismatch at {offset}")
            if not _is_sha256(row["text_sha"]):
                raise ValueError(f"target text SHA invalid at {offset}")
            if row["prompt_sha256"] in seen_prompts or row["text_sha"] in seen_text:
                raise ValueError(f"duplicate target source at {offset}")
            seen_ids.add(prompt_id)
            seen_prompts.add(row["prompt_sha256"])
            seen_text.add(row["text_sha"])
        return payload
    except (KeyError, TypeError, ValueError) as error:
        raise QualityContractError(f"invalid targets-freeze.json: {error}") from error


def _validate_path_and_receipt(
    root: Path, tag: str, targets_sha: str, artifact_shas: Mapping[str, str]
) -> tuple[str, str]:
    mode, stage = RAW_CONTRACT[tag]
    path_name = f"path-config-receipt-{tag}-{stage}.json"
    receipt_name = f"path-receipt-{tag}-{stage}.json"
    path_payload = _load_json(root / path_name, path_name)
    receipt = _load_json(root / receipt_name, receipt_name)
    expected_free_run = 0
    try:
        if not isinstance(path_payload, dict) or set(path_payload) != {
            "schema", "mode", "tag", "stage", "port", "free_run", "startup_ok",
            "prefill_backend", "cuda_graph", "radix_cache", "overlap_schedule",
            "flashinfer_autotune", "attention_backend",
        }:
            raise ValueError("path-config key set mismatch")
        if (
            path_payload["schema"] != "phase2-v4-path-config-v1"
            or path_payload["mode"] != mode or path_payload["tag"] != tag
            or path_payload["stage"] != stage
            or not isinstance(path_payload["port"], int)
            or isinstance(path_payload["port"], bool)
            or not 1 <= path_payload["port"] <= 65535
            or path_payload["free_run"] != expected_free_run
            or path_payload["startup_ok"] != 1
            or path_payload["prefill_backend"] != "disabled"
            or path_payload["cuda_graph"] is not False
            or path_payload["radix_cache"] is not False
            or path_payload["overlap_schedule"] is not False
            or path_payload["flashinfer_autotune"] is not False
            or path_payload["attention_backend"] != "dsv4"
        ):
            raise ValueError("path-config content mismatch")
        if not isinstance(receipt, dict) or set(receipt) != {
            "schema", "mode", "tag", "stage", "rc", "path_ok", "free_run",
            "path_config_sha256", "client_rc", "outputs",
        }:
            raise ValueError("receipt key set mismatch")
        if (
            receipt["schema"] != "phase2-v4-session-receipt-v1"
            or receipt["mode"] != mode or receipt["tag"] != tag
            or receipt["stage"] != stage or receipt["rc"] != 0
            or receipt["path_ok"] != 1 or receipt["free_run"] != 0
            or receipt["path_config_sha256"] != artifact_shas[path_name]
        ):
            raise ValueError("receipt header mismatch")
        if stage == "target-build":
            expected_rc = {"target": 0, "gsm8k": None, "teacher512": None, "free_run_info": None}
            expected_outputs = {"targets_freeze": targets_sha, "gsm8k_csv": None, "gsm8k_json": None, "teacher512": None, "free_run_info": None}
        else:
            expected_rc = {"target": None, "gsm8k": 0, "teacher512": 0, "free_run_info": None}
            expected_outputs = {
                "targets_freeze": targets_sha,
                "gsm8k_csv": artifact_shas[f"gsm8k-{tag}.csv"],
                "gsm8k_json": artifact_shas[f"gsm8k-{tag}.json"],
                "teacher512": artifact_shas[f"teacher512-{tag}.json"],
                "free_run_info": None,
            }
        if receipt["client_rc"] != expected_rc or receipt["outputs"] != expected_outputs:
            raise ValueError("receipt client/output graph mismatch")
        return mode, receipt["path_config_sha256"]
    except (KeyError, TypeError, ValueError) as error:
        raise QualityContractError(f"invalid {receipt_name}: {error}") from error


def _validate_teacher(
    path: Path, tag: str, mode: str, targets: dict, targets_sha: str,
    path_config_sha: str,
) -> list[dict]:
    payload = _load_json(path, path.name)
    try:
        if not isinstance(payload, dict) or set(payload) != {
            "protocol", "mode", "tag", "targets_sha256",
            "path_config_receipt_sha256", "request_batching", "rows",
        }:
            raise ValueError("teacher key set mismatch")
        if (
            payload["protocol"] != "phase2-v4" or payload["mode"] != mode
            or payload["tag"] != tag or payload["targets_sha256"] != targets_sha
            or payload["path_config_receipt_sha256"] != path_config_sha
            or payload["request_batching"] != {
                "schema": "phase2-v4-request-batching-v1",
                "generate_endpoint": "/generate",
                "generate_batch_size": 16,
                "response_identity": "request-order-and-rid-echo",
            }
        ):
            raise ValueError("teacher header mismatch")
        rows = payload["rows"]
        if not isinstance(rows, list) or len(rows) != EXPECTED_PROMPTS:
            raise ValueError("teacher row count mismatch")
        for offset, (row, target) in enumerate(zip(rows, targets["rows"])):
            if not isinstance(row, dict) or set(row) != {
                "prompt_id", "prompt_sha256", "target_sha256", "token_ids",
                "logprobs", "top1_ids",
            }:
                raise ValueError(f"teacher row {offset} key set mismatch")
            if (
                row["prompt_id"] != target["prompt_id"]
                or row["prompt_sha256"] != target["prompt_sha256"]
                or row["target_sha256"] != target["target_sha256"]
                or row["token_ids"] != target["target_ids"]
            ):
                raise ValueError(f"teacher target alignment mismatch at {offset}")
            if (
                not isinstance(row["logprobs"], list)
                or len(row["logprobs"]) != EXPECTED_TARGET_TOKENS
                or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in row["logprobs"])
            ):
                raise ValueError(f"teacher logprob invalid at {offset}")
            if (
                not isinstance(row["top1_ids"], list)
                or len(row["top1_ids"]) != EXPECTED_TARGET_TOKENS
                or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in row["top1_ids"])
            ):
                raise ValueError(f"teacher top1 invalid at {offset}")
        return rows
    except (KeyError, TypeError, ValueError) as error:
        raise QualityContractError(f"invalid {path.name}: {error}") from error


def _validate_gsm(
    root: Path, tag: str, provenance: Mapping[str, str]
) -> list[int]:
    csv_path = root / f"gsm8k-{tag}.csv"
    summary_path = root / f"gsm8k-{tag}.json"
    summary = _load_json(summary_path, summary_path.name)
    try:
        rows: dict[int, int] = {}
        with open(csv_path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["idx", "pred", "gold", "correct", "text_sha256"]:
                raise ValueError("GSM8K CSV header mismatch")
            for row in reader:
                index = int(row["idx"])
                correct = int(row["correct"])
                if index in rows or correct not in (0, 1) or not _is_sha256(row["text_sha256"]):
                    raise ValueError(f"invalid GSM8K row {index}")
                rows[index] = correct
        if set(rows) != set(range(EXPECTED_QUESTIONS)):
            raise ValueError("GSM8K question set mismatch")
        correct_total = sum(rows.values())
        accuracy = correct_total / EXPECTED_QUESTIONS
        if not isinstance(summary, dict) or set(summary) != {
            "schema", "tag", "n", "correct", "accuracy", "dataset_sha256",
            "csv_sha256", "request_batching",
        }:
            raise ValueError("GSM8K summary key set mismatch")
        if (
            summary["schema"] != "phase2-v4-gsm8k-v1" or summary["tag"] != tag
            or summary["n"] != EXPECTED_QUESTIONS or summary["correct"] != correct_total
            or not math.isclose(float(summary["accuracy"]), accuracy, abs_tol=1e-15)
            or summary["dataset_sha256"] != provenance["gsm8k_sha256"]
            or summary["csv_sha256"] != _sha256(csv_path)
            or summary["request_batching"] != {
                "schema": "phase2-v4-request-batching-v1",
                "generate_endpoint": "/generate", "generate_batch_size": 16,
                "response_identity": "request-order-and-rid-echo",
            }
        ):
            raise ValueError("GSM8K summary mismatch")
        return [rows[index] for index in range(EXPECTED_QUESTIONS)]
    except (KeyError, TypeError, ValueError) as error:
        raise QualityContractError(f"invalid GSM8K {tag}: {error}") from error


@dataclass(frozen=True)
class RawControlAssets:
    manifest_sha256: str
    source_digest: str
    artifact_shas: Mapping[str, str]
    provenance: Mapping[str, str]
    teachers: Mapping[str, Sequence[dict]]
    gsm_correct: Mapping[str, Sequence[int]]


def load_raw_control_assets(manifest_path: str | Path) -> RawControlAssets:
    manifest = Path(manifest_path)
    if manifest.is_symlink() or not manifest.is_file():
        raise QualityContractError("power manifest must be a regular non-symlink file")
    payload = validate_power_input(_load_json(manifest, manifest.name))
    root = manifest.parent / payload["source_root"]
    if root.is_symlink() or not root.is_dir():
        raise QualityContractError("raw source_root must be a regular non-symlink directory")
    if root.resolve().parent != manifest.parent.resolve():
        raise QualityContractError("raw source_root escapes the manifest directory")
    observed = {entry.name for entry in root.iterdir()}
    if observed != REQUIRED_RAW_FILENAMES:
        raise QualityContractError(
            f"raw directory file set mismatch: missing={sorted(REQUIRED_RAW_FILENAMES-observed)} "
            f"extra={sorted(observed-REQUIRED_RAW_FILENAMES)}"
        )
    for name in REQUIRED_RAW_FILENAMES:
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root.resolve():
            raise QualityContractError(f"raw artifact is not a bounded regular file: {name}")
        observed_sha = _sha256(path)
        if observed_sha != payload["artifacts"][name]:
            raise QualityContractError(f"raw artifact SHA mismatch: {name}")
    current_evaluator_sha = _sha256(Path(__file__))
    if payload["provenance"]["evaluator_sha256"] != current_evaluator_sha:
        raise QualityContractError("power evaluator SHA does not match manifest provenance")
    generator = manifest.parent / "logprob_client.py"
    if generator.is_symlink() or not generator.is_file() or _sha256(generator) != payload["provenance"]["generator_sha256"]:
        raise QualityContractError("generator script SHA does not match manifest provenance")

    targets_path = root / "targets-freeze.json"
    targets = _validate_targets(targets_path, payload["provenance"])
    targets_sha = payload["artifacts"]["targets-freeze.json"]
    _validate_path_and_receipt(root, "t-s", targets_sha, payload["artifacts"])
    teachers: dict[str, Sequence[dict]] = {}
    gsm: dict[str, Sequence[int]] = {}
    for tag in RAW_TAGS:
        mode, path_config_sha = _validate_path_and_receipt(
            root, tag, targets_sha, payload["artifacts"]
        )
        teachers[tag] = _validate_teacher(
            root / f"teacher512-{tag}.json", tag, mode, targets,
            targets_sha, path_config_sha,
        )
        gsm[tag] = _validate_gsm(root, tag, payload["provenance"])
    source_digest = hashlib.sha256(
        json.dumps(payload["artifacts"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return RawControlAssets(
        manifest_sha256=_sha256(manifest),
        source_digest=source_digest,
        artifact_shas=dict(payload["artifacts"]),
        provenance=dict(payload["provenance"]),
        teachers=teachers,
        gsm_correct=gsm,
    )


def _per_prompt_nll(rows: Sequence[dict], window: ScoreWindow) -> list[float]:
    return [-statistics.fmean(float(value) for value in row["logprobs"][window.start:window.stop]) for row in rows]


def _per_prompt_target_error(rows: Sequence[dict], window: ScoreWindow) -> list[float]:
    return [sum(a != b for a, b in zip(row["top1_ids"][window.start:window.stop], row["token_ids"][window.start:window.stop])) / window.width for row in rows]


def _pair_metrics(left: Sequence[dict], right: Sequence[dict], window: ScoreWindow) -> dict[str, list[float]]:
    result = {"mean": [], "p95": [], "flip": []}
    for lhs, rhs in zip(left, right):
        deltas = [abs(float(a) - float(b)) for a, b in zip(lhs["logprobs"][window.start:window.stop], rhs["logprobs"][window.start:window.stop])]
        result["mean"].append(statistics.fmean(deltas))
        result["p95"].append(nearest_rank(deltas, 0.95))
        result["flip"].append(sum(a != b for a, b in zip(lhs["top1_ids"][window.start:window.stop], rhs["top1_ids"][window.start:window.stop])) / window.width)
    return result


def _cluster_reduce(samples: Sequence[Sequence[float]], reducer: Callable[[Iterable[float]], float]) -> list[float]:
    if not samples or any(len(sample) != EXPECTED_PROMPTS for sample in samples):
        raise QualityContractError("prompt-cluster samples have inconsistent shapes")
    return [float(reducer(sample[index] for sample in samples)) for index in range(EXPECTED_PROMPTS)]


def _precompute_gate_vectors(raw: RawControlAssets) -> tuple[dict[str, list[float]], dict[str, list[list[int]]], dict[str, float]]:
    scores = raw.teachers
    gates: dict[str, list[float]] = {}
    margins: dict[str, float] = {}
    for window in WINDOW_VIEWS:
        for label, tags in (("deepep", ("d1", "d2", "d3")), ("split", ("s1", "s2", "s3"))):
            pairs = tuple(itertools.combinations(tags, 2))
            pair_metrics = [_pair_metrics(scores[a], scores[b], window) for a, b in pairs]
            for metric_name, key, split_margin, deep_margin in (
                ("mean-abs-logprob", "mean", CAPS.control_mean_abs_logprob, CAPS.deepep_prefix_mean_abs_logprob),
                ("p95-abs-logprob", "p95", CAPS.control_p95_abs_logprob, CAPS.deepep_prefix_p95_abs_logprob),
                ("top1-flip", "flip", CAPS.control_top1_flip, CAPS.deepep_prefix_top1_flip),
            ):
                if label == "deepep" and window != WINDOW_VIEWS[0]:
                    continue
                name = f"1-{label}-{window.name}-{metric_name}-admission"
                gates[name] = _cluster_reduce([item[key] for item in pair_metrics], statistics.median)
                margins[name] = deep_margin if label == "deepep" else split_margin

        for prefix, left_tags, right_tags, caps in (
            ("2-fused-self", ("p1", "p2", "p3"), None, (CAPS.fused_self_mean_abs_logprob, CAPS.fused_self_p95_abs_logprob, CAPS.fused_self_top1_flip)),
            ("3-fused-split", ("p1", "p2", "p3"), ("s1", "s2", "s3"), (CAPS.fused_split_mean_abs_logprob, CAPS.fused_split_p95_abs_logprob, CAPS.fused_split_top1_flip)),
        ):
            pairs = tuple(itertools.combinations(left_tags, 2)) if right_tags is None else tuple(itertools.product(left_tags, right_tags))
            pair_metrics = [_pair_metrics(scores[a], scores[b], window) for a, b in pairs]
            for metric_name, key, margin in zip(("mean-abs-logprob", "p95-abs-logprob", "top1-flip"), ("mean", "p95", "flip"), caps):
                name = f"{prefix}-{window.name}-{metric_name}"
                gates[name] = _cluster_reduce([item[key] for item in pair_metrics], statistics.median)
                margins[name] = margin

        family_metrics: dict[str, dict[str, list[float]]] = {}
        for family, tags in (("fused", ("p1", "p2", "p3")), ("split", ("s1", "s2", "s3")), ("deepep", ("d1", "d2", "d3"))):
            family_metrics[family] = {
                "nll": _cluster_reduce([_per_prompt_nll(scores[tag], window) for tag in tags], statistics.fmean),
                "target_error": _cluster_reduce([_per_prompt_target_error(scores[tag], window) for tag in tags], statistics.fmean),
            }
        for reference in ("split", "deepep"):
            for metric, margin in (("nll", CAPS.directional_delta_nll), ("target_error", CAPS.directional_delta_target_error)):
                name = f"4-fused-{reference}-{window.name}-delta-{metric.replace('_', '-')}"
                gates[name] = [a - b for a, b in zip(family_metrics["fused"][metric], family_metrics[reference][metric])]
                margins[name] = margin

    gsm = raw.gsm_correct
    family_error = {
        family: [statistics.fmean(1 - gsm[tag][index] for tag in tags) for index in range(EXPECTED_QUESTIONS)]
        for family, tags in (("fused", ("p1", "p2", "p3")), ("split", ("s1", "s2", "s3")), ("deepep", ("d1", "d2", "d3")))
    }
    for reference in ("split", "deepep"):
        name = f"5-gsm8k-fused-{reference}-1pp"
        gates[name] = [a - b for a, b in zip(family_error["fused"], family_error[reference])]
        margins[name] = CAPS.gsm8k_delta_error
    spans = {
        "5-gsm8k-deepep-accuracy-span": [list(gsm[tag]) for tag in ("d1", "d2", "d3")],
        "5-gsm8k-split-accuracy-span": [list(gsm[tag]) for tag in ("s1", "s2", "s3")],
        "5-gsm8k-fused-accuracy-span": [list(gsm[tag]) for tag in ("p1", "p2", "p3")],
    }
    if set(gates) != EXPECTED_POWER_GATE_NAMES or len(gates) != CI_GATE_COUNT:
        raise QualityContractError("precomputed CI gate set is not exactly 70")
    return gates, spans, margins


def _require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise PowerBackendError(
            "formal raw power requires NumPy linked to OpenBLAS or MKL; no scalar fallback"
        ) from error
    import contextlib
    import io
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        np.show_config()
    configuration = capture.getvalue()
    if "openblas" not in configuration.lower():
        raise PowerBackendError("formal raw power requires NumPy linked to OpenBLAS")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise PowerBackendError("formal raw power requires OPENBLAS_NUM_THREADS=1")
    runtime_capture = io.StringIO()
    with contextlib.redirect_stdout(runtime_capture):
        show_runtime = getattr(np, "show_runtime", None)
        if show_runtime is not None:
            show_runtime()
    runtime = runtime_capture.getvalue()
    import re
    runtime_threads = re.findall(
        r"num_threads['\"]?\s*[:=]\s*([0-9]+)", runtime,
        flags=re.IGNORECASE,
    )
    if runtime_threads and any(value != "1" for value in runtime_threads):
        raise PowerBackendError(
            "OpenBLAS runtime thread count is not the frozen value 1"
        )
    backend_text = configuration + "\n" + runtime
    versions = re.findall(
        r"OpenBLAS\s+([0-9]+(?:\.[0-9]+)+)", backend_text,
        flags=re.IGNORECASE,
    )
    if not versions:
        versions = re.findall(
            r"(?:internal_api|name)['\"]?\s*[:=]\s*['\"]?[^\n}]*openblas"
            r"[\s\S]{0,400}?version['\"]?\s*[:=]\s*['\"]?"
            r"([0-9]+(?:\.[0-9]+)+)",
            backend_text,
            flags=re.IGNORECASE,
        )
    if not versions:
        raise PowerBackendError("cannot freeze the OpenBLAS runtime version")
    return np, configuration + "\n" + runtime, versions[-1]


def bootstrap_count_matrix(cluster_count: int, replicates: int, seed: int, np):
    """Count matrix exactly matching ``random.Random`` scalar resampling."""

    result = np.zeros((replicates, cluster_count), dtype=np.uint16)
    for replicate, row in enumerate(
        iter_bootstrap_count_rows(cluster_count, replicates, seed)
    ):
        result[replicate] = row
    return result


def batch_upper_decisions(vectors, margins, counts, np):
    """Apply the scalar U95 gate to rows; also return means for shift reuse."""

    vectors = np.asarray(vectors, dtype=np.float64)
    margins = np.asarray(margins, dtype=np.float64)
    counts = np.asarray(counts)
    if vectors.ndim != 2 or counts.ndim != 2 or vectors.shape[1] != counts.shape[1]:
        raise QualityContractError("batch bootstrap shape mismatch")
    if margins.shape != (vectors.shape[0],):
        raise QualityContractError("batch margin shape mismatch")
    means = vectors @ counts.T.astype(np.float64, copy=False) / vectors.shape[1]
    rank = math.ceil(0.95 * counts.shape[0])
    within = (means <= margins[:, None]) | (
        np.abs(means - margins[:, None]) <= 1e-12
    )
    passed = np.count_nonzero(within, axis=1) >= rank
    return passed, means


def verify_scalar_batch_equivalence(np) -> str:
    """Fail closed unless vectorized decisions equal the scalar reference."""

    fixtures = (
        ([0.0] * 7 + [0.14], 0.02),
        ([0.02] * 8, 0.02),
        ([0.02 - 1e-9] * 8, 0.02),
        ([0.02 + 1e-9] * 8, 0.02),
        ([-0.02, 0.02] * 4, 0.02),
    )
    replicates = 200
    counts = bootstrap_count_matrix(8, replicates, BOOTSTRAP_SEED, np)
    vectors = np.asarray([values for values, _margin in fixtures], dtype=np.float64)
    margins = np.asarray([margin for _values, margin in fixtures], dtype=np.float64)
    batch, _means = batch_upper_decisions(vectors, margins, counts, np)
    scalar = []
    scalar_means = []
    for values, margin in fixtures:
        means = _scalar_bootstrap_means(
            values, replicates=replicates, seed=BOOTSTRAP_SEED
        )
        scalar_means.append(means)
        upper = nearest_rank(means, 0.95)
        scalar.append(upper_margin_predicate(upper, margin))
    maximum_mean_difference = float(
        np.max(np.abs(np.asarray(scalar_means, dtype=np.float64) - _means))
    )
    if maximum_mean_difference > 1e-11:
        raise PowerBackendError(
            "batch bootstrap means differ from count-weighted scalar evaluator"
        )
    if list(bool(item) for item in batch) != scalar:
        raise PowerBackendError("batch bootstrap is not decision-equivalent to scalar evaluator")
    return hashlib.sha256(
        bytes(int(item) for item in scalar)
        + f"|maxdiff={maximum_mean_difference:.17g}".encode()
    ).hexdigest()


def joint_outer_draws(trials: int, np):
    """Draw prompt/question clusters jointly from one frozen Python stream."""

    rng = random.Random(MONTE_CARLO_SEED)
    prompt = np.empty((trials, EXPECTED_PROMPTS), dtype=np.uint16)
    question = np.empty((trials, EXPECTED_QUESTIONS), dtype=np.uint16)
    for trial in range(trials):
        prompt[trial] = [rng.randrange(EXPECTED_PROMPTS) for _ in range(EXPECTED_PROMPTS)]
        question[trial] = [rng.randrange(EXPECTED_QUESTIONS) for _ in range(EXPECTED_QUESTIONS)]
    return prompt, question


def _digest_array(array) -> str:
    header = f"{array.dtype.str}|{','.join(str(value) for value in array.shape)}|".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _rate_record(count: int, trials: int, *, key: str, threshold: float, comparison: str) -> dict[str, object]:
    rate = count / trials
    passed = rate >= threshold if comparison == "ge" else rate <= threshold
    return {"trials": trials, key: count, key.replace("count", "rate"): rate, "threshold": threshold, "pass": passed}


def compute_power_assessment(
    manifest_path: str | Path, *, simulation: SimulationConfig = FORMAL_SIMULATION
) -> dict[str, object]:
    """Recompute the complete raw power verdict.

    ``simulation`` exists only for unit tests/calibration.  The CLI and
    preflight call this function without an override, and the manifest has no
    field capable of changing formal trial counts or seeds.
    """

    if simulation.outer_trials < 1 or simulation.bootstrap_replicates < 100 or simulation.outer_chunk < 1:
        raise QualityContractError("invalid internal simulation configuration")
    raw = load_raw_control_assets(manifest_path)
    np, blas_configuration, openblas_version = _require_numpy()
    host_class = os.environ.get("PHASE2_POWER_HOST_CLASS", "")
    if not host_class or host_class != raw.provenance["host_class"]:
        raise PowerBackendError(
            "PHASE2_POWER_HOST_CLASS must match the frozen raw manifest"
        )
    if np.__version__ != raw.provenance["numpy_version"]:
        raise PowerBackendError("NumPy version differs from frozen provenance")
    if openblas_version != raw.provenance["openblas_version"]:
        raise PowerBackendError("OpenBLAS version differs from frozen provenance")
    equivalence_digest = verify_scalar_batch_equivalence(np)
    gates, spans, margins = _precompute_gate_vectors(raw)
    teacher_names = tuple(sorted(name for name in gates if not name.startswith("5-gsm8k")))
    gsm_names = tuple(sorted(name for name in gates if name.startswith("5-gsm8k")))
    gate_order = teacher_names + gsm_names
    if len(teacher_names) != 68 or len(gsm_names) != 2 or len(gate_order) != CI_GATE_COUNT:
        raise QualityContractError("CI gate partition must be 68 teacher + 2 GSM")

    Wp = bootstrap_count_matrix(EXPECTED_PROMPTS, simulation.bootstrap_replicates, BOOTSTRAP_SEED, np)
    Wq = bootstrap_count_matrix(EXPECTED_QUESTIONS, simulation.bootstrap_replicates, BOOTSTRAP_SEED, np)
    Wp_float = Wp.astype(np.float64)
    Wq_float = Wq.astype(np.float64)
    prompt_draws, question_draws = joint_outer_draws(
        simulation.outer_trials, np
    )

    teacher_source = np.asarray([gates[name] for name in teacher_names], dtype=np.float64)
    gsm_source = np.asarray([gates[name] for name in gsm_names], dtype=np.float64)
    teacher_margins = np.asarray([margins[name] for name in teacher_names], dtype=np.float64)
    gsm_margins = np.asarray([margins[name] for name in gsm_names], dtype=np.float64)
    effect_shifts: dict[str, float] = {}
    effect_targets: dict[str, float] = {}
    degenerate_effects: set[str] = set()
    for name in EFFECT_GATE_NAMES:
        source = np.asarray(gates[name], dtype=np.float64)
        shift, target, degenerate = exact_margin_injection(
            gates[name], margins[name]
        )
        if degenerate:
            degenerate_effects.add(name)
        if (name.endswith("target-error") or name.startswith("5-gsm8k")) and (
            float((source + shift).min()) < -1.0 - 1e-12
            or float((source + shift).max()) > 1.0 + 1e-12
        ):
            raise QualityContractError(f"exact-margin injection leaves valid support: {name}")
        effect_shifts[name] = shift
        effect_targets[name] = target

    ci_rejections = np.zeros(CI_GATE_COUNT, dtype=np.int64)
    span_failures = np.zeros(SPAN_GATE_COUNT, dtype=np.int64)
    family_passes = {"teacher68": 0, "gsm5": 0, "full73": 0}
    effect_gate_rejections = {name: 0 for name in EFFECT_GATE_NAMES}
    effect_family_failures = {name: 0 for name in EFFECT_GATE_NAMES}
    digest = hashlib.sha256()
    digest_header = {
        "schema": POWER_ASSESSMENT_SCHEMA,
        "manifest_sha256": raw.manifest_sha256,
        "gate_order": gate_order,
        "span_order": SPAN_GATE_NAMES,
        "effect_order": EFFECT_GATE_NAMES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "monte_carlo_seed": MONTE_CARLO_SEED,
        "trials": simulation.outer_trials,
        "bootstrap_replicates": simulation.bootstrap_replicates,
        "outer_chunk": simulation.outer_chunk,
        "outer_rng": "python-random-MT19937-randrange-joint",
        "inner_rng": "python-random-MT19937-count-matrix",
        "numpy_version": np.__version__,
        "openblas_version": openblas_version,
        "host_class": host_class,
    }
    digest.update(json.dumps(digest_header, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    total_bits = 0
    total_ones = 0

    span_arrays = [np.asarray(spans[name], dtype=np.float64) for name in SPAN_GATE_NAMES]
    teacher_effect_index = {name: teacher_names.index(name) for name in EFFECT_GATE_NAMES if name in teacher_names}
    gsm_effect_index = {name: gsm_names.index(name) for name in EFFECT_GATE_NAMES if name in gsm_names}

    for start in range(0, simulation.outer_trials, simulation.outer_chunk):
        stop = min(simulation.outer_trials, start + simulation.outer_chunk)
        prompt_index = prompt_draws[start:stop].astype(np.intp, copy=False)
        question_index = question_draws[start:stop].astype(np.intp, copy=False)
        chunk = stop - start

        teacher_vectors = teacher_source[:, prompt_index].transpose(1, 0, 2).reshape(chunk * len(teacher_names), EXPECTED_PROMPTS)
        teacher_caps = np.tile(teacher_margins, chunk)
        teacher_pass_flat, teacher_means_flat = batch_upper_decisions(teacher_vectors, teacher_caps, Wp_float, np)
        teacher_pass = teacher_pass_flat.reshape(chunk, len(teacher_names))
        teacher_means = teacher_means_flat.reshape(chunk, len(teacher_names), simulation.bootstrap_replicates)

        gsm_vectors = gsm_source[:, question_index].transpose(1, 0, 2).reshape(chunk * len(gsm_names), EXPECTED_QUESTIONS)
        gsm_caps = np.tile(gsm_margins, chunk)
        gsm_pass_flat, gsm_means_flat = batch_upper_decisions(gsm_vectors, gsm_caps, Wq_float, np)
        gsm_pass = gsm_pass_flat.reshape(chunk, len(gsm_names))
        gsm_means = gsm_means_flat.reshape(chunk, len(gsm_names), simulation.bootstrap_replicates)
        ci_pass = np.concatenate((teacher_pass, gsm_pass), axis=1)
        ci_rejections += np.count_nonzero(~ci_pass, axis=0)

        span_pass_columns = []
        for source in span_arrays:
            sampled = source[:, question_index]
            accuracies = sampled.mean(axis=2).transpose(1, 0)
            span = accuracies.max(axis=1) - accuracies.min(axis=1)
            span_pass_columns.append(span <= CAPS.gsm8k_accuracy_span + 1e-12)
        span_pass = np.column_stack(span_pass_columns)
        span_failures += np.count_nonzero(~span_pass, axis=0)
        teacher_family = np.all(teacher_pass, axis=1)
        gsm_family = np.all(gsm_pass, axis=1) & np.all(span_pass, axis=1)
        full_family = teacher_family & gsm_family
        family_passes["teacher68"] += int(np.count_nonzero(teacher_family))
        family_passes["gsm5"] += int(np.count_nonzero(gsm_family))
        family_passes["full73"] += int(np.count_nonzero(full_family))

        effect_bits = []
        required_rank = math.ceil(0.95 * simulation.bootstrap_replicates)
        for name in EFFECT_GATE_NAMES:
            if name in teacher_effect_index:
                index = teacher_effect_index[name]
                shifted = teacher_means[:, index, :] + effect_shifts[name]
                injected_pass = np.count_nonzero(
                    (shifted <= margins[name])
                    | (np.abs(shifted - margins[name]) <= 1e-12), axis=1
                ) >= required_rank
                other_pass = np.all(
                    np.delete(teacher_pass, index, axis=1), axis=1
                )
            else:
                index = gsm_effect_index[name]
                shifted = gsm_means[:, index, :] + effect_shifts[name]
                injected_pass = np.count_nonzero(
                    (shifted <= margins[name])
                    | (np.abs(shifted - margins[name]) <= 1e-12), axis=1
                ) >= required_rank
                other_pass = np.all(np.delete(gsm_pass, index, axis=1), axis=1) & np.all(span_pass, axis=1)
            rejected = ~injected_pass
            family_failed = ~(other_pass & injected_pass)
            effect_gate_rejections[name] += int(np.count_nonzero(rejected))
            effect_family_failures[name] += int(np.count_nonzero(family_failed))
            effect_bits.extend((rejected, family_failed))

        packed_columns = [ci_pass, span_pass, teacher_family[:, None], gsm_family[:, None], full_family[:, None]]
        packed_columns.extend(column[:, None] for column in effect_bits)
        bits = np.concatenate(packed_columns, axis=1)
        total_bits += int(bits.size)
        total_ones += int(np.count_nonzero(bits))
        digest.update(np.packbits(bits, axis=None, bitorder="little").tobytes())

    single_gates = {}
    for index, name in enumerate(gate_order):
        count = int(ci_rejections[index])
        single_gates[name] = _rate_record(count, simulation.outer_trials, key="rejection_count", threshold=0.05, comparison="le")
    span_records = {
        name: _rate_record(int(span_failures[index]), simulation.outer_trials, key="failure_count", threshold=1.0, comparison="le")
        for index, name in enumerate(SPAN_GATE_NAMES)
    }
    families = {
        name: _rate_record(count, simulation.outer_trials, key="pass_count", threshold=0.80, comparison="ge")
        for name, count in family_passes.items()
    }
    effects = {}
    for name in EFFECT_GATE_NAMES:
        gate_record = _rate_record(effect_gate_rejections[name], simulation.outer_trials, key="rejection_count", threshold=0.80, comparison="ge")
        family_record = _rate_record(effect_family_failures[name], simulation.outer_trials, key="failure_count", threshold=0.80, comparison="ge")
        effects[name] = {
            "margin": margins[name],
            "injected_estimand": effect_targets[name],
            "shift": effect_shifts[name],
            "degenerate_source": name in degenerate_effects,
            "injection_policy": (
                "margin-plus-25pct" if name in degenerate_effects else "exact-margin"
            ),
            "gate": gate_record,
            "family": family_record,
            "pass": gate_record["pass"] and family_record["pass"],
        }

    popcounts = {
        "ci_pass": simulation.outer_trials * CI_GATE_COUNT - int(ci_rejections.sum()),
        "span_pass": simulation.outer_trials * SPAN_GATE_COUNT - int(span_failures.sum()),
        "family_pass": sum(family_passes.values()),
        "effect_gate_rejection": sum(effect_gate_rejections.values()),
        "effect_family_failure": sum(effect_family_failures.values()),
    }
    expected_bits = simulation.outer_trials * (
        CI_GATE_COUNT + SPAN_GATE_COUNT + 3 + 2 * EFFECT_SCENARIO_COUNT
    )
    if total_bits != expected_bits or total_ones != sum(popcounts.values()):
        raise PowerBackendError("pass-vector bit/popcount accounting mismatch")
    overall = all(record["pass"] for record in single_gates.values()) and all(record["pass"] for record in families.values()) and all(record["pass"] for record in effects.values())
    return {
        "schema": POWER_ASSESSMENT_SCHEMA,
        "candidate_blind": True,
        "control_only": True,
        "pseudo_f": {"source": ["p1", "p2", "p3"], "implementation": "split", "independent_from": ["s1", "s2", "s3"]},
        "raw_manifest_sha256": raw.manifest_sha256,
        "raw_source_digest": raw.source_digest,
        "source_shas": dict(raw.artifact_shas),
        "provenance": dict(raw.provenance),
        "contract": {
            "ci_gates": CI_GATE_COUNT, "span_gates": SPAN_GATE_COUNT,
            "blocking_gates": BLOCKING_GATE_COUNT,
            "effect_scenarios": EFFECT_SCENARIO_COUNT,
            "outer_trials": simulation.outer_trials,
            "bootstrap_replicates": simulation.bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "monte_carlo_seed": MONTE_CARLO_SEED,
            "outer_rng": "python-random-MT19937-randrange-joint",
            "inner_rng": "python-random-MT19937-count-matrix",
            "outer_chunk": simulation.outer_chunk,
            "formal_outer_chunk": FORMAL_OUTER_CHUNK,
        },
        "gate_order": list(gate_order),
        "span_order": list(SPAN_GATE_NAMES),
        "effect_order": list(EFFECT_GATE_NAMES),
        "single_ci_gates": single_gates,
        "span_gates_information_and_family": span_records,
        "zero_degradation_families": families,
        "exact_margin_effects": effects,
        "pass_vector": {
            "sha256": digest.hexdigest(), "bits": total_bits,
            "ones": total_ones, "popcounts": popcounts,
        },
        "rng_digests": {
            "prompt_outer": _digest_array(prompt_draws),
            "question_outer": _digest_array(question_draws),
            "prompt_inner_counts": _digest_array(Wp),
            "question_inner_counts": _digest_array(Wq),
            "scalar_batch_equivalence": equivalence_digest,
        },
        "backend": {
            "numpy": np.__version__, "openblas": openblas_version,
            "openblas_num_threads": 1, "host_class": host_class,
            "blas": blas_configuration.strip(),
        },
        "overall_pass": overall,
        "error": None,
    }


def assess_power_input(payload: object) -> dict[str, object]:
    """Reject legacy self-reported payloads; formal assessment needs raw bytes."""

    try:
        validate_power_input(payload)
    except QualityContractError as error:
        message = str(error)
    else:
        message = "raw manifest validated structurally; use compute_power_assessment(path)"
    return {
        "schema": POWER_ASSESSMENT_SCHEMA,
        "candidate_blind": False,
        "control_only": False,
        "overall_pass": False,
        "error": message,
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
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    except FileExistsError as error:
        raise QualityContractError(f"refusing to replace power assessment {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not 1 <= len(arguments) <= 2:
        print("usage: phase2_v4_power.py INPUT [ASSESSMENT_OUT]", file=sys.stderr)
        return 2
    try:
        assessment = compute_power_assessment(arguments[0])
    except (OSError, QualityContractError, PowerBackendError) as error:
        assessment = {
            "schema": POWER_ASSESSMENT_SCHEMA,
            "candidate_blind": False,
            "control_only": False,
            "overall_pass": False,
            "error": f"{type(error).__name__}: {error}",
        }
    if len(arguments) == 2:
        try:
            _atomic_json_no_clobber(Path(arguments[1]), assessment)
        except (OSError, QualityContractError) as error:
            print(f"POWER_OUTPUT_ERROR|{error}", file=sys.stderr)
            return 2
    print(f"PHASE2_V4_POWER|overall_pass={int(bool(assessment['overall_pass']))}|error={assessment.get('error')}", flush=True)
    return 0 if assessment["overall_pass"] else 2


validate_window_views()


if __name__ == "__main__":
    raise SystemExit(main())
