#!/usr/bin/env python3
"""Run a fixed-input, cold-cache Humming prefill service leg.

This benchmark is intentionally serial.  It flushes the radix cache before
every request, sends the same exact input-id artifacts to every server leg,
and records one-token request latency plus response hashes.  A separate server
restart is required for each tuning variant so the cached Humming config and
CUDA graph state cannot leak across variants.
"""

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


def canonical_json_sha256(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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


def build_measurement_schedule(
    inputs: list[dict[str, Any]], repeats: int, seed: int
) -> list[dict[str, Any]]:
    if not inputs or repeats <= 0:
        raise ValueError("inputs and repeats must be non-empty and positive")
    rng = random.Random(seed)
    schedule = []
    for round_index in range(repeats):
        round_inputs = list(inputs)
        rng.shuffle(round_inputs)
        for order_index, item in enumerate(round_inputs):
            schedule.append(
                {
                    "round_index": round_index,
                    "order_index": order_index,
                    "input": item,
                }
            )
    return schedule


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min_seconds": min(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "p90_seconds": percentile(values, 0.90),
        "p95_seconds": percentile(values, 0.95),
        "max_seconds": max(values),
    }


def request_json(
    url: str,
    *,
    payload: Any | None = None,
    encoded_payload: bytes | None = None,
    method: str | None = None,
    timeout: int,
    expect_json: bool = True,
) -> Any:
    if payload is not None and encoded_payload is not None:
        raise ValueError("payload and encoded_payload are mutually exclusive")
    data = encoded_payload
    if payload is not None:
        data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        raw = response.read()
    if not raw:
        return None
    return json.loads(raw) if expect_json else raw.decode()


def run_request(
    base_url: str,
    item: dict[str, Any],
    *,
    request_id: str,
    timeout: int,
) -> dict[str, Any]:
    request_json(
        f"{base_url}/flush_cache",
        payload={},
        method="POST",
        timeout=timeout,
        expect_json=False,
    )
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
    meta_info = response.get("meta_info") or {}
    prompt_tokens = meta_info.get("prompt_tokens")
    if prompt_tokens is not None and prompt_tokens != item["prompt_tokens"]:
        raise RuntimeError(
            f"server reported {prompt_tokens} prompt tokens for "
            f"{item['prompt_tokens']}-token input"
        )
    response_signature = {
        "text": response.get("text"),
        "output_ids": meta_info.get("output_ids"),
        "finish_reason": meta_info.get("finish_reason"),
        "completion_tokens": meta_info.get("completion_tokens"),
    }
    return {
        "elapsed_seconds": elapsed_seconds,
        "response_sha256": canonical_json_sha256(response_signature),
        "response_signature": response_signature,
        "server_prompt_tokens": prompt_tokens,
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
    parser.add_argument("--variant", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    base_url = args.base_url.rstrip("/")
    inputs = [load_input_ids(path) for path in args.input_ids_json]
    if len({item["prompt_tokens"] for item in inputs}) != len(inputs):
        raise ValueError("every input artifact must have a distinct prompt length")

    server_info = request_json(f"{base_url}/server_info", timeout=args.timeout)
    warmup_measurements = []
    for warmup_index in range(args.warmups):
        for item in inputs:
            result = run_request(
                base_url,
                item,
                request_id=(
                    f"humming-prefill-{args.variant}-warmup-{warmup_index}-"
                    f"{item['prompt_tokens']}"
                ),
                timeout=args.timeout,
            )
            warmup_measurements.append(
                {
                    "warmup_index": warmup_index,
                    "prompt_tokens": item["prompt_tokens"],
                    **result,
                }
            )

    measurements = []
    schedule = build_measurement_schedule(inputs, args.repeats, args.seed)
    for schedule_item in schedule:
        item = schedule_item["input"]
        round_index = schedule_item["round_index"]
        result = run_request(
            base_url,
            item,
            request_id=(
                f"humming-prefill-{args.variant}-formal-{round_index}-"
                f"{item['prompt_tokens']}"
            ),
            timeout=args.timeout,
        )
        measurements.append(
            {
                "round_index": round_index,
                "order_index": schedule_item["order_index"],
                "prompt_tokens": item["prompt_tokens"],
                "input_ids_sha256": item["input_ids_sha256"],
                **result,
            }
        )

    summaries = {}
    for item in inputs:
        prompt_tokens = item["prompt_tokens"]
        samples = [
            measurement["elapsed_seconds"]
            for measurement in measurements
            if measurement["prompt_tokens"] == prompt_tokens
        ]
        summaries[str(prompt_tokens)] = summarize(samples)

    result = {
        "format_version": FORMAT_VERSION,
        "state": "MEASURED",
        "variant": args.variant,
        "base_url": base_url,
        "contract": {
            "cold_cache": "POST /flush_cache before every timed request",
            "serial_requests": True,
            "max_new_tokens": 1,
            "temperature": 0,
            "warmups_per_shape": args.warmups,
            "measured_repeats_per_shape": args.repeats,
            "measurement_order_seed": args.seed,
        },
        "inputs": [
            {key: value for key, value in item.items() if key != "input_ids"}
            for item in inputs
        ],
        "server_info": server_info,
        "warmup_measurements": warmup_measurements,
        "measurements": measurements,
        "summaries": summaries,
    }
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "state": result["state"],
                "variant": args.variant,
                "summaries": summaries,
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
