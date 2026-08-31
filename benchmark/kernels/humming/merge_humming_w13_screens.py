#!/usr/bin/env python3
"""Merge deterministic W13 correctness shards into a survivor ID list."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def atomic_write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _require_equal(name: str, values: list[Any]) -> Any:
    first = canonical_json(values[0])
    if any(canonical_json(value) != first for value in values[1:]):
        raise ValueError(f"screen shards disagree on {name}")
    return values[0]


def merge_screens(payloads: list[dict], sources: list[str]) -> dict:
    if not payloads:
        raise ValueError("at least one screen shard is required")
    layers = []
    for payload in payloads:
        if payload.get("state") != "SCREENED":
            raise ValueError("screen shard top-level state is not SCREENED")
        layer = payload.get("sublayers", {}).get("w13")
        if not layer or layer.get("state") != "SCREENED":
            raise ValueError("screen shard W13 state is not SCREENED")
        if layer.get("rejection_policy") != "filter" or not layer.get(
            "correctness_only"
        ):
            raise ValueError("screen shard is not a filter-mode correctness screen")
        layers.append(layer)

    shard_count = _require_equal(
        "candidate_shard_count",
        [layer["candidate_shard_count"] for layer in layers],
    )
    if shard_count != len(payloads):
        raise ValueError("screen shard file count does not match shard count")
    shard_indices = [layer["candidate_shard_index"] for layer in layers]
    if sorted(shard_indices) != list(range(shard_count)):
        raise ValueError("screen shard indices do not cover the full range")

    common_top_fields = (
        "capture_sha256",
        "model_config_sha256",
        "tp_size",
        "shape_m",
        "humming_version",
        "torch_version",
        "cuda_version",
        "device_name",
        "device_capability",
        "w13_sampler",
    )
    common = {
        field: _require_equal(field, [payload.get(field) for payload in payloads])
        for field in common_top_fields
    }
    normalized_parameters = []
    for payload in payloads:
        parameters = dict(payload.get("parameters", {}))
        parameters.pop("candidate_shard_index", None)
        normalized_parameters.append(parameters)
    common["screen_parameters"] = _require_equal(
        "parameters excluding shard index", normalized_parameters
    )
    if common["screen_parameters"].get("route_split") != "train":
        raise ValueError("screen shards must use the train route split")
    if not common["screen_parameters"].get("correctness_only"):
        raise ValueError("screen shards must be correctness-only")
    if common["screen_parameters"].get("candidate_rejection_policy") != "filter":
        raise ValueError("screen shards must use the filter rejection policy")
    common_layer_fields = (
        "candidate_source",
        "candidate_universe_count",
        "candidate_universe_ids",
        "candidate_universe_sha256",
        "heuristic_id",
        "heuristic_config",
        "correctness_gate",
        "route_points",
    )
    common_layer = {
        field: _require_equal(field, [layer.get(field) for layer in layers])
        for field in common_layer_fields
    }
    universe_ids = common_layer["candidate_universe_ids"]
    if len(universe_ids) != common_layer["candidate_universe_count"]:
        raise ValueError("candidate universe count does not match its ID list")
    if len(universe_ids) != len(set(universe_ids)):
        raise ValueError("candidate universe IDs are not unique")
    heuristic_id = common_layer["heuristic_id"]
    if heuristic_id not in universe_ids:
        raise ValueError("heuristic is absent from candidate universe")

    covered: list[str] = []
    valid_ids = {heuristic_id}
    rejected = []
    shard_receipts = []
    for source, payload, layer in zip(sources, payloads, layers, strict=True):
        selected = layer["selected_candidate_ids"]
        if heuristic_id not in selected:
            raise ValueError("screen shard does not contain the heuristic")
        nonheuristic = [item for item in selected if item != heuristic_id]
        covered.extend(nonheuristic)
        layer_valid = {item["config_id"] for item in layer["valid_candidates"]}
        layer_rejected = {item["config_id"] for item in layer["rejected"]}
        if heuristic_id not in layer_valid:
            raise ValueError("heuristic failed a screen shard")
        if layer_valid & layer_rejected:
            raise ValueError("candidate is both valid and rejected in one shard")
        if layer_valid | layer_rejected != set(selected):
            raise ValueError("screen shard does not account for every candidate")
        valid_ids.update(layer_valid)
        rejected.extend(layer["rejected"])
        shard_receipts.append(
            {
                "source": source,
                "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "shard_index": layer["candidate_shard_index"],
                "selected_count": len(selected),
                "valid_count": len(layer_valid),
                "rejected_count": len(layer_rejected),
            }
        )

    expected_covered = [item for item in universe_ids if item != heuristic_id]
    if (
        sorted(covered) != sorted(expected_covered)
        or len(covered) != len(set(covered))
    ):
        raise ValueError("non-heuristic candidate shards are incomplete or overlapping")
    survivor_ids = [item for item in universe_ids if item in valid_ids]
    return {
        "format_version": 1,
        "state": "MERGED",
        **common,
        **common_layer,
        "candidate_ids": survivor_ids,
        "survivor_count": len(survivor_ids),
        "rejected_count": len(rejected),
        "rejected": rejected,
        "shards": sorted(shard_receipts, key=lambda item: item["shard_index"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [json.loads(path.read_text()) for path in args.screen]
    result = merge_screens(payloads, [str(path) for path in args.screen])
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("state", "survivor_count", "rejected_count")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
