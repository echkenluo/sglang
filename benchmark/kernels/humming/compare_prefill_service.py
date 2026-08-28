#!/usr/bin/env python3
"""Validate and compare a four-leg Humming cold-prefill service group."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
SUMMARY_METRICS = ("median_seconds", "p95_seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_leg(path: Path) -> dict[str, Any]:
    leg = json.loads(path.read_text())
    if leg.get("state") != "MEASURED":
        raise ValueError(f"{path} is not a MEASURED service leg")
    if not isinstance(leg.get("summaries"), dict) or not leg["summaries"]:
        raise ValueError(f"{path} has no latency summaries")
    leg["_source_path"] = str(path)
    leg["_source_sha256"] = file_sha256(path)
    return leg


def input_contract(leg: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "prompt_tokens": item["prompt_tokens"],
                "input_ids_sha256": item["input_ids_sha256"],
            }
            for item in leg["inputs"]
        ],
        key=lambda item: item["prompt_tokens"],
    )


def response_hashes(leg: dict[str, Any]) -> dict[str, list[str]]:
    hashes: dict[str, set[str]] = {}
    for measurement in leg["measurements"]:
        prompt_tokens = str(measurement["prompt_tokens"])
        hashes.setdefault(prompt_tokens, set()).add(measurement["response_sha256"])
    return {key: sorted(value) for key, value in hashes.items()}


def improvement_pct(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def compare_candidate(
    candidate: dict[str, Any],
    *,
    a1: dict[str, Any],
    baseline_reference: dict[str, dict[str, float]],
) -> dict[str, Any]:
    versus_a1 = {}
    versus_baseline_reference = {}
    for prompt_tokens in baseline_reference:
        versus_a1[prompt_tokens] = {
            metric: improvement_pct(
                a1["summaries"][prompt_tokens][metric],
                candidate["summaries"][prompt_tokens][metric],
            )
            for metric in SUMMARY_METRICS
        }
        versus_baseline_reference[prompt_tokens] = {
            metric: improvement_pct(
                baseline_reference[prompt_tokens][metric],
                candidate["summaries"][prompt_tokens][metric],
            )
            for metric in SUMMARY_METRICS
        }
    return {
        "variant": candidate["variant"],
        "summaries": candidate["summaries"],
        "improvement_pct_versus_a1": versus_a1,
        "improvement_pct_versus_baseline_reference": versus_baseline_reference,
    }


def candidate_decision(comparison: dict[str, Any], prompt_tokens: list[str]) -> str:
    improvements = comparison["improvement_pct_versus_baseline_reference"]
    median_gains = [improvements[key]["median_seconds"] for key in prompt_tokens]
    p95_gains = [improvements[key]["p95_seconds"] for key in prompt_tokens]
    largest_shape = max(prompt_tokens, key=int)
    largest_median_gain = improvements[largest_shape]["median_seconds"]

    if (
        all(gain > 0 for gain in median_gains)
        and all(gain >= 0 for gain in p95_gains)
        and largest_median_gain >= 1.0
    ):
        return "SERVICE_GO"
    if (
        all(gain > 0 for gain in median_gains)
        and largest_median_gain >= 0.3
        and all(gain >= -1.0 for gain in p95_gains)
    ):
        return "WEAK_SIGNAL"
    return "SERVICE_NO_GO"


def compare_group(
    a1: dict[str, Any],
    candidates: list[dict[str, Any]],
    a2: dict[str, Any],
    *,
    drift_threshold_pct: float,
) -> dict[str, Any]:
    legs = [a1, *candidates, a2]
    if len({leg["variant"] for leg in legs}) != len(legs):
        raise ValueError("every service leg must have a unique variant")
    if any(leg["contract"] != a1["contract"] for leg in legs[1:]):
        raise ValueError("service leg contracts do not match")
    if any(input_contract(leg) != input_contract(a1) for leg in legs[1:]):
        raise ValueError("service leg input artifacts do not match")
    if any(set(leg["summaries"]) != set(a1["summaries"]) for leg in legs[1:]):
        raise ValueError("service leg prompt shapes do not match")

    prompt_tokens = sorted(a1["summaries"], key=int)
    hashes_by_leg = {leg["variant"]: response_hashes(leg) for leg in legs}
    response_valid = all(
        len(hashes_by_leg[leg["variant"]][key]) == 1
        and hashes_by_leg[leg["variant"]][key] == hashes_by_leg[a1["variant"]][key]
        for leg in legs
        for key in prompt_tokens
    )

    baseline_drift_pct = {}
    baseline_reference = {}
    for key in prompt_tokens:
        baseline_drift_pct[key] = {
            metric: abs(
                improvement_pct(
                    a1["summaries"][key][metric], a2["summaries"][key][metric]
                )
            )
            for metric in SUMMARY_METRICS
        }
        baseline_reference[key] = {
            metric: (a1["summaries"][key][metric] + a2["summaries"][key][metric]) / 2.0
            for metric in SUMMARY_METRICS
        }

    median_drift_valid = all(
        baseline_drift_pct[key]["median_seconds"] < drift_threshold_pct
        for key in prompt_tokens
    )
    if not response_valid:
        state = "INVALID_RESPONSE_MISMATCH"
    elif not median_drift_valid:
        state = "INVALID_DRIFT"
    else:
        state = "VALID"

    candidate_comparisons = [
        compare_candidate(candidate, a1=a1, baseline_reference=baseline_reference)
        for candidate in candidates
    ]
    for comparison in candidate_comparisons:
        comparison["decision"] = (
            candidate_decision(comparison, prompt_tokens)
            if state == "VALID"
            else "UNANSWERABLE_INVALID_GROUP"
        )

    return {
        "format_version": FORMAT_VERSION,
        "state": state,
        "drift_threshold_pct": drift_threshold_pct,
        "response_hashes_by_leg": hashes_by_leg,
        "baseline_drift_pct": baseline_drift_pct,
        "baseline_reference": baseline_reference,
        "baseline_legs": {
            "a1": {"variant": a1["variant"], "summaries": a1["summaries"]},
            "a2": {"variant": a2["variant"], "summaries": a2["summaries"]},
        },
        "candidates": candidate_comparisons,
        "sources": [
            {
                "variant": leg["variant"],
                "path": leg["_source_path"],
                "sha256": leg["_source_sha256"],
            }
            for leg in legs
        ],
    }


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--a2", type=Path, required=True)
    parser.add_argument("--drift-threshold-pct", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.drift_threshold_pct <= 0:
        raise ValueError("drift threshold must be positive")
    result = compare_group(
        load_leg(args.a1),
        [load_leg(path) for path in args.candidate],
        load_leg(args.a2),
        drift_threshold_pct=args.drift_threshold_pct,
    )
    atomic_write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
