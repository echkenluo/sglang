#!/usr/bin/env python3
"""Single-boot paired service gate for indexed Humming W2 grid tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
RUNTIME_KEY = "humming_indexed_w2_runtime_num_sms"
CANDIDATES = (4096, 5120)


def canonical_json_sha256(data: Any) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_input_ids(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    input_ids = payload.get("input_ids") if isinstance(payload, dict) else payload
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError(f"{path} must contain a non-empty input_ids list")
    if not all(type(token) is int and token >= 0 for token in input_ids):
        raise ValueError(f"{path} contains a non-integer or negative input id")
    return {
        "path": str(path),
        "input_ids": input_ids,
        "prompt_tokens": len(input_ids),
        "input_ids_sha256": canonical_json_sha256(input_ids),
        "source_metadata": (
            payload.get("metadata") if isinstance(payload, dict) else None
        ),
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": statistics.median(values),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def selector_order(round_index: int) -> list[int]:
    return [0, 4096, 0, 5120, 0] if round_index % 2 == 0 else [0, 5120, 0, 4096, 0]


def build_round_schedule(
    prompt_tokens: list[int], rounds: int, seed: int
) -> list[dict[str, int]]:
    if rounds <= 0 or not prompt_tokens:
        raise ValueError("rounds and prompt token shapes must be positive")
    rng = random.Random(seed)
    schedule = []
    for round_index in range(rounds):
        shape_order = list(prompt_tokens)
        rng.shuffle(shape_order)
        for prompt_token_count in shape_order:
            schedule.append(
                {"round_index": round_index, "prompt_tokens": prompt_token_count}
            )
    return schedule


def request_json(
    url: str,
    *,
    payload: Any | None = None,
    encoded_payload: bytes | None = None,
    timeout: int,
) -> Any:
    if payload is not None and encoded_payload is not None:
        raise ValueError("payload and encoded_payload are mutually exclusive")
    data = encoded_payload
    if payload is not None:
        data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        raw = response.read()
    return json.loads(raw) if raw else None


def set_selector(base_url: str, num_sms: int, timeout: int) -> dict[str, Any]:
    updated = request_json(
        f"{base_url}/set_internal_state",
        payload={"server_args": {RUNTIME_KEY: num_sms}},
        timeout=timeout,
    )
    if not isinstance(updated, list) or not updated or not all(updated):
        raise RuntimeError(f"selector update {num_sms} failed: {updated!r}")
    server_info = request_json(f"{base_url}/server_info", timeout=timeout)
    internal_states = server_info.get("internal_states") or []
    rank_values = [state.get(RUNTIME_KEY) for state in internal_states]
    if not rank_values or any(value != num_sms for value in rank_values):
        raise RuntimeError(
            f"selector readback {num_sms} disagrees across ranks: {rank_values!r}"
        )
    return {
        "requested": num_sms,
        "updated": updated,
        "rank_values": rank_values,
        "startup_time": server_info.get("startup_time"),
    }


def run_generate(
    base_url: str,
    item: dict[str, Any],
    *,
    request_id: str,
    timeout: int,
) -> dict[str, Any]:
    body = {
        "input_ids": item["input_ids"],
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "rid": request_id,
    }
    encoded_body = json.dumps(body).encode()
    started_ns = time.perf_counter_ns()
    response = request_json(
        f"{base_url}/generate", encoded_payload=encoded_body, timeout=timeout
    )
    elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1e9
    if not isinstance(response, dict):
        raise RuntimeError("/generate did not return a JSON object")
    meta = response.get("meta_info") or {}
    if meta.get("prompt_tokens") != item["prompt_tokens"]:
        raise RuntimeError(
            f"prompt token mismatch: {meta.get('prompt_tokens')!r} != "
            f"{item['prompt_tokens']}"
        )
    if meta.get("cached_tokens") != 0:
        raise RuntimeError(
            f"request was not cold: cached_tokens={meta.get('cached_tokens')!r}"
        )
    if meta.get("completion_tokens") != 1:
        raise RuntimeError(
            f"completion token mismatch: {meta.get('completion_tokens')!r}"
        )
    signature = {
        "text": response.get("text"),
        "output_ids": meta.get("output_ids"),
        "finish_reason": meta.get("finish_reason"),
        "completion_tokens": meta.get("completion_tokens"),
    }
    return {
        "elapsed_seconds": elapsed_seconds,
        "cached_tokens": meta.get("cached_tokens"),
        "response_sha256": canonical_json_sha256(signature),
        "response_signature": signature,
    }


def analyze_measurements(
    measurements: list[dict[str, Any]], prompt_tokens: list[int], rounds: int
) -> dict[str, Any]:
    shape_results = {}
    errors = []
    for shape in prompt_tokens:
        shape_rows = [row for row in measurements if row["prompt_tokens"] == shape]
        hashes = sorted({row["response_sha256"] for row in shape_rows})
        if len(hashes) != 1:
            errors.append(f"shape {shape} has {len(hashes)} response hashes")
        candidate_results = {}
        for candidate in CANDIDATES:
            paired_improvements = []
            bracket_drifts = []
            candidate_latencies = []
            bracket_latencies = []
            for round_index in range(rounds):
                rows = sorted(
                    [row for row in shape_rows if row["round_index"] == round_index],
                    key=lambda row: row["position"],
                )
                if [row["selector"] for row in rows] != selector_order(round_index):
                    errors.append(f"shape {shape} round {round_index} order mismatch")
                    continue
                position = next(
                    index
                    for index, row in enumerate(rows)
                    if row["selector"] == candidate
                )
                left = rows[position - 1]["elapsed_seconds"]
                current = rows[position]["elapsed_seconds"]
                right = rows[position + 1]["elapsed_seconds"]
                bracket = (left + right) / 2.0
                bracket_latencies.append(bracket)
                candidate_latencies.append(current)
                paired_improvements.append((bracket - current) / bracket * 100.0)
                bracket_drifts.append(abs(right - left) / bracket * 100.0)
            if len(paired_improvements) != rounds:
                errors.append(
                    f"shape {shape} candidate {candidate} has "
                    f"{len(paired_improvements)}/{rounds} pairs"
                )
                continue
            drift_summary = summarize(bracket_drifts)
            drift_valid = (
                drift_summary["median"] < 1.0
                and drift_summary["p95"] < 2.0
                and sum(value > 3.0 for value in bracket_drifts) <= 1
            )
            if not drift_valid:
                errors.append(f"shape {shape} candidate {candidate} drift gate failed")
            p95_gain = (
                (
                    percentile(bracket_latencies, 0.95)
                    - percentile(candidate_latencies, 0.95)
                )
                / percentile(bracket_latencies, 0.95)
                * 100.0
            )
            candidate_results[str(candidate)] = {
                "paired_improvement_pct": summarize(paired_improvements),
                "bracket_drift_pct": drift_summary,
                "brackets_over_3pct": sum(value > 3.0 for value in bracket_drifts),
                "candidate_latency_seconds": summarize(candidate_latencies),
                "bracket_latency_seconds": summarize(bracket_latencies),
                "p95_improvement_pct": p95_gain,
            }
        shape_results[str(shape)] = {
            "response_hashes": hashes,
            "candidates": candidate_results,
        }

    valid = not errors
    candidate_decisions = {}
    ordered_shapes = sorted(prompt_tokens)
    for candidate in CANDIDATES:
        if not valid:
            candidate_decisions[str(candidate)] = "UNANSWERABLE_INVALID_GROUP"
            continue
        per_shape = [
            shape_results[str(shape)]["candidates"][str(candidate)]
            for shape in ordered_shapes
        ]
        median_gains = [row["paired_improvement_pct"]["median"] for row in per_shape]
        p95_gains = [row["p95_improvement_pct"] for row in per_shape]
        largest_gain = median_gains[-1]
        if (
            all(gain > 0 for gain in median_gains)
            and largest_gain >= 1.0
            and all(gain >= 0 for gain in p95_gains)
        ):
            decision = "SERVICE_GO"
        elif (
            all(gain > 0 for gain in median_gains)
            and largest_gain >= 0.3
            and all(gain >= -1.0 for gain in p95_gains)
        ):
            decision = "WEAK_SIGNAL"
        else:
            decision = "SERVICE_NO_GO"
        candidate_decisions[str(candidate)] = decision

    decisions = set(candidate_decisions.values())
    if not valid:
        decision = "INVALID"
    elif "SERVICE_GO" in decisions:
        decision = "SERVICE_GO"
    elif "WEAK_SIGNAL" in decisions:
        decision = "WEAK_SIGNAL"
    else:
        decision = "SERVICE_NO_GO"
    return {
        "state": "VALID" if valid else "INVALID",
        "decision": decision,
        "errors": errors,
        "candidate_decisions": candidate_decisions,
        "shapes": shape_results,
    }


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--input-ids-json", type=Path, nargs="+", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    inputs = [load_input_ids(path) for path in args.input_ids_json]
    by_shape = {item["prompt_tokens"]: item for item in inputs}
    if len(by_shape) != len(inputs):
        raise ValueError("every input artifact must have a distinct prompt length")
    if args.warmups < 0 or args.rounds <= 0:
        raise ValueError("warmups must be non-negative and rounds must be positive")

    result: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "state": "RUNNING",
        "contract": {
            "single_boot": True,
            "radix_cache_disabled": True,
            "flush_cache_per_request": False,
            "selectors": [0, *CANDIDATES],
            "warmups_per_selector_shape": args.warmups,
            "rounds_per_shape": args.rounds,
            "seed": args.seed,
        },
        "inputs": [
            {key: value for key, value in item.items() if key != "input_ids"}
            for item in inputs
        ],
        "warmups": [],
        "measurements": [],
    }
    try:
        initial_info = request_json(f"{base_url}/server_info", timeout=args.timeout)
        result["server_info_before"] = initial_info
        for selector in (0, *CANDIDATES):
            for item in inputs:
                for warmup_index in range(args.warmups):
                    receipt = set_selector(base_url, selector, args.timeout)
                    response = run_generate(
                        base_url,
                        item,
                        request_id=(
                            f"humming-w2-inservice-warmup-{selector}-"
                            f"{item['prompt_tokens']}-{warmup_index}"
                        ),
                        timeout=args.timeout,
                    )
                    result["warmups"].append(
                        {
                            "selector": selector,
                            "prompt_tokens": item["prompt_tokens"],
                            "warmup_index": warmup_index,
                            "selector_receipt": receipt,
                            **response,
                        }
                    )
        set_selector(base_url, 0, args.timeout)

        schedule = build_round_schedule(list(by_shape), args.rounds, args.seed)
        for scheduled in schedule:
            round_index = scheduled["round_index"]
            shape = scheduled["prompt_tokens"]
            item = by_shape[shape]
            for position, selector in enumerate(selector_order(round_index)):
                receipt = set_selector(base_url, selector, args.timeout)
                response = run_generate(
                    base_url,
                    item,
                    request_id=(
                        f"humming-w2-inservice-formal-{shape}-{round_index}-"
                        f"{position}-{selector}"
                    ),
                    timeout=args.timeout,
                )
                result["measurements"].append(
                    {
                        "round_index": round_index,
                        "position": position,
                        "selector": selector,
                        "prompt_tokens": shape,
                        "input_ids_sha256": item["input_ids_sha256"],
                        "selector_receipt": receipt,
                        **response,
                    }
                )

        final_info = request_json(f"{base_url}/server_info", timeout=args.timeout)
        result["server_info_after"] = final_info
        if final_info.get("startup_time") != initial_info.get("startup_time"):
            raise RuntimeError(
                "server startup_time changed during the single-boot group"
            )
        result["analysis"] = analyze_measurements(
            result["measurements"], list(by_shape), args.rounds
        )
        result["state"] = result["analysis"]["state"]
        result["decision"] = result["analysis"]["decision"]
    except Exception as exc:
        result["state"] = "INVALID"
        result["decision"] = "INVALID"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            result["final_selector_receipt"] = set_selector(base_url, 0, args.timeout)
        except Exception as exc:
            result["state"] = "INVALID"
            result["decision"] = "INVALID"
            result["selector_restore_error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(args.out, result)

    print(
        json.dumps(
            {
                "state": result["state"],
                "decision": result["decision"],
                "out": str(args.out),
            },
            indent=2,
        )
    )
    if result["state"] != "VALID":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
