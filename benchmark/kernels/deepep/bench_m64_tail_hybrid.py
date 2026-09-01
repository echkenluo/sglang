#!/usr/bin/env python3
"""Screen row-wise warp kernels for ExpertLane's final partial M64 tiles.

The control executes the current M64 WGMMA W13 -> fused SwiGLU/FP8 quant ->
W2 chain.  The candidate reuses the graph-safe warp masked chain and only
processes real tail rows.  This is a tail-only operator screen, not a serving
benchmark and not an integrated hybrid scheduler.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

os.environ.setdefault("SGLANG_OPT_USE_JIT_EP_ACTIVATION", "1")

from sglang.kernels.ops.attention.dsv4 import (  # noqa: E402
    silu_and_mul_masked_post_quant,
)
from sglang.srt.layers.moe.moe_runner.triton_utils import (  # noqa: E402
    warp_decode_moe,
)


E = 64
MAX_M = 64
HIDDEN = 4096
INTER = 2048
GROUP = 128
SWIGLU_LIMIT = 10.0


@dataclass(frozen=True)
class Geometry:
    padded_tokens: int
    sample_weight: int
    tail_experts: int
    tail_rows: int
    bin_counts: tuple[int, int, int, int, int]


GEOMETRIES = (
    Geometry(1024, 172, 55, 1179, (17, 9, 14, 9, 6)),
    Geometry(2048, 344, 54, 1250, (16, 9, 12, 10, 7)),
    Geometry(4096, 860, 55, 1356, (15, 9, 13, 10, 8)),
)
BIN_RANGES = ((1, 8), (9, 16), (17, 32), (33, 48), (49, 63))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def build_tail_rows(geometry: Geometry, seed: int) -> list[int]:
    values: list[int] = []
    for count, (lower, upper) in zip(geometry.bin_counts, BIN_RANGES):
        midpoint = (lower + upper) // 2
        values.extend([midpoint] * count)
    if len(values) != geometry.tail_experts:
        raise ValueError("tail histogram does not match tail expert count")

    delta = geometry.tail_rows - sum(values)
    direction = 1 if delta >= 0 else -1
    remaining = abs(delta)
    while remaining:
        changed = False
        for index, value in enumerate(values):
            bin_index = next(
                i
                for i, (lower, upper) in enumerate(BIN_RANGES)
                if lower <= value <= upper
            )
            lower, upper = BIN_RANGES[bin_index]
            candidate = value + direction
            if lower <= candidate <= upper:
                values[index] = candidate
                remaining -= 1
                changed = True
                if not remaining:
                    break
        if not changed:
            raise ValueError("tail total is infeasible for the frozen histogram")

    random.Random(seed).shuffle(values)
    values.extend([0] * (E - len(values)))
    random.Random(seed + 1).shuffle(values)
    if sum(values) != geometry.tail_rows:
        raise AssertionError("tail row synthesis lost rows")
    return values


def capture_graph(fn: Callable[[], torch.Tensor]) -> torch.cuda.CUDAGraph:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def time_graph(
    graph: torch.cuda.CUDAGraph, *, groups: int, replays_per_group: int
) -> list[float]:
    samples: list[float] = []
    for _ in range(groups):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays_per_group):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / replays_per_group)
    return samples


def summarize(samples: list[float]) -> dict[str, object]:
    return {
        "sample_count": len(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def relative_improvement(control: float, candidate: float) -> float:
    return 100.0 * (control - candidate) / control


def block_scaled_reference(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    rows: int,
    output_columns: int,
) -> torch.Tensor:
    output = torch.zeros(
        (rows, output_columns), device=a.device, dtype=torch.float32
    )
    for block in range(a.shape[-1] // GROUP):
        start = block * GROUP
        stop = start + GROUP
        partial = a[:rows, start:stop].float() @ b[:output_columns, start:stop].float().T
        row_scale = a_scale[:rows, block, None]
        weight_scale = b_scale[: math.ceil(output_columns / GROUP), block]
        weight_scale = weight_scale.repeat_interleave(GROUP)[:output_columns]
        output.add_(partial * row_scale * weight_scale[None, :])
    return output


def correctness_gate(
    tensors: dict[str, torch.Tensor], tail_rows: list[int]
) -> dict[str, object]:
    candidate = tensors["candidate_out"]
    control = tensors["control_out"]
    active = [expert for expert, rows in enumerate(tail_rows) if rows]
    experts = active[:2]
    output_columns = 256
    stats: dict[str, dict[str, float]] = {}
    candidate_parts = []
    control_parts = []
    reference_parts = []

    for expert in experts:
        rows = tail_rows[expert]
        gate_up = block_scaled_reference(
            tensors["x"][expert],
            tensors["x_scale"][expert],
            tensors["w13"][expert],
            tensors["w13_scale"][expert],
            rows,
            2 * INTER,
        )
        gate = gate_up[:, :INTER].clamp(max=SWIGLU_LIMIT)
        up = gate_up[:, INTER:].clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        activated = F.silu(gate) * up
        reference = block_scaled_reference(
            activated,
            torch.ones((rows, INTER // GROUP), device=activated.device),
            tensors["w2"][expert],
            tensors["w2_scale"][expert],
            rows,
            output_columns,
        )
        reference_parts.append(reference)
        candidate_parts.append(candidate[expert, :rows, :output_columns].float())
        control_parts.append(control[expert, :rows, :output_columns].float())

    reference = torch.cat(reference_parts)
    reference_max = float(reference.abs().max().clamp_min(1e-6))
    for name, values in (
        ("candidate", torch.cat(candidate_parts)),
        ("control", torch.cat(control_parts)),
    ):
        error = (values - reference).abs()
        stats[name] = {
            "max_abs": float(error.max()),
            "mean_abs": float(error.mean()),
            "maxnorm_relative": float(error.max()) / reference_max,
            "finite_fraction": float(torch.isfinite(values).float().mean()),
        }
    passed = all(
        arm["finite_fraction"] == 1.0 and arm["maxnorm_relative"] < 0.05
        for arm in stats.values()
    )
    return {
        "passed": passed,
        "experts": experts,
        "output_columns": output_columns,
        "stats": stats,
    }


def run_geometry(
    mok_ext,
    warp_ext,
    weights: dict[str, torch.Tensor],
    geometry: Geometry,
    *,
    groups: int,
    replays_per_group: int,
    pair_slots: int,
) -> dict[str, object]:
    device = torch.device("cuda", 0)
    seed = 20260901 + geometry.padded_tokens
    tail_rows = build_tail_rows(geometry, seed)
    activation_launch_factor = math.ceil(sum(tail_rows) / MAX_M)
    masked_m = torch.tensor(tail_rows, dtype=torch.int32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)

    x = torch.randn((E, MAX_M, HIDDEN), generator=generator, device=device)
    x = x.clamp(-3, 3).to(torch.float8_e4m3fn)
    x_scale = torch.rand(
        (E, MAX_M, HIDDEN // GROUP), generator=generator, device=device
    ) * 0.09 + 0.01

    gate_up = torch.empty((E, MAX_M, 2 * INTER), dtype=torch.bfloat16, device=device)
    down_input = torch.empty(
        (E, MAX_M, INTER), dtype=torch.float8_e4m3fn, device=device
    )
    down_scale = torch.empty(
        (E, MAX_M, INTER // GROUP), dtype=torch.float32, device=device
    )
    control_out = torch.empty(
        (E, MAX_M, HIDDEN), dtype=torch.bfloat16, device=device
    )
    pairs = torch.empty((1 + E * MAX_M,), dtype=torch.int32, device=device)
    candidate_buf = torch.empty(
        (E, MAX_M, INTER), dtype=torch.bfloat16, device=device
    )
    candidate_out = torch.empty_like(control_out)

    def control() -> torch.Tensor:
        mok_ext.sm90_fp8_block_grouped_pipelined_out_test(
            x,
            weights["w13"],
            x_scale,
            weights["w13_scale"],
            masked_m,
            gate_up,
        )
        silu_and_mul_masked_post_quant(
            gate_up,
            down_input,
            down_scale,
            GROUP,
            masked_m,
            # This benchmark packs only the final partial tile of every
            # expert into the T=64 dimension.  The production `topk` launch
            # multiplier is therefore no longer sufficient: size the static
            # graph grid from the actual number of valid tail rows.
            topk=activation_launch_factor,
            swiglu_limit=SWIGLU_LIMIT,
        )
        mok_ext.sm90_fp8_block_grouped_pipelined_out_test(
            down_input,
            weights["w2"],
            down_scale,
            weights["w2_scale"],
            masked_m,
            control_out,
        )
        return control_out

    def candidate() -> torch.Tensor:
        warp_ext.moe_warp_masked_fp8(
            x,
            x_scale,
            weights["w13"],
            weights["w13_scale"],
            weights["w2"],
            weights["w2_scale"],
            masked_m,
            pairs,
            candidate_buf,
            candidate_out,
            SWIGLU_LIMIT,
            pair_slots,
        )
        return candidate_out

    control()
    candidate()
    torch.cuda.synchronize()
    correctness = correctness_gate(
        {
            "x": x,
            "x_scale": x_scale,
            "w13": weights["w13"],
            "w13_scale": weights["w13_scale"],
            "w2": weights["w2"],
            "w2_scale": weights["w2_scale"],
            "control_out": control_out,
            "candidate_out": candidate_out,
        },
        tail_rows,
    )
    if not correctness["passed"]:
        return {
            "padded_tokens": geometry.padded_tokens,
            "state": "INVALID_CORRECTNESS",
            "correctness": correctness,
        }

    control_graph = capture_graph(control)
    candidate_graph = capture_graph(candidate)
    legs: list[tuple[str, list[float]]] = []
    for arm, graph in (
        ("control_a1", control_graph),
        ("candidate_b1", candidate_graph),
        ("candidate_b2", candidate_graph),
        ("control_a2", control_graph),
    ):
        legs.append(
            (
                arm,
                time_graph(
                    graph, groups=groups, replays_per_group=replays_per_group
                ),
            )
        )
    summaries = {arm: summarize(samples) for arm, samples in legs}
    control_ms = statistics.mean(
        [
            summaries["control_a1"]["median_ms"],
            summaries["control_a2"]["median_ms"],
        ]
    )
    candidate_ms = statistics.mean(
        [
            summaries["candidate_b1"]["median_ms"],
            summaries["candidate_b2"]["median_ms"],
        ]
    )
    control_p95 = statistics.mean(
        [summaries["control_a1"]["p95_ms"], summaries["control_a2"]["p95_ms"]]
    )
    candidate_p95 = statistics.mean(
        [
            summaries["candidate_b1"]["p95_ms"],
            summaries["candidate_b2"]["p95_ms"],
        ]
    )
    return {
        "padded_tokens": geometry.padded_tokens,
        "state": "COMPLETE",
        "sample_weight": geometry.sample_weight,
        "tail_experts": geometry.tail_experts,
        "tail_rows": geometry.tail_rows,
        "activation_launch_factor": activation_launch_factor,
        "pair_slots": pair_slots,
        "masked_m": tail_rows,
        "correctness": correctness,
        "legs": summaries,
        "control_median_ms": control_ms,
        "candidate_median_ms": candidate_ms,
        "median_improvement_percent": relative_improvement(control_ms, candidate_ms),
        "control_p95_ms": control_p95,
        "candidate_p95_ms": candidate_p95,
        "p95_improvement_percent": relative_improvement(control_p95, candidate_p95),
        "control_drift_percent": relative_improvement(
            summaries["control_a1"]["median_ms"],
            summaries["control_a2"]["median_ms"],
        ),
        "candidate_drift_percent": relative_improvement(
            summaries["candidate_b1"]["median_ms"],
            summaries["candidate_b2"]["median_ms"],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mok-source", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=12)
    parser.add_argument("--replays-per-group", type=int, default=10)
    parser.add_argument("--pair-slots", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.groups < 3 or args.replays_per_group < 1:
        raise ValueError("groups must be >=3 and replays-per-group must be >=1")
    if args.pair_slots < 1:
        raise ValueError("pair-slots must be >=1")

    sys.path.insert(0, str(args.mok_source))
    from mok import _C as mok_ext

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("this benchmark requires SM90")
    warp_ext = warp_decode_moe._load_ext()
    if warp_ext is None:
        raise RuntimeError("warp masked extension failed to load")
    required_mok = "sm90_fp8_block_grouped_pipelined_out_test"
    if not hasattr(mok_ext, required_mok):
        raise RuntimeError(f"MoK extension lacks {required_mok}")

    generator = torch.Generator(device=device).manual_seed(20260901)
    w13 = torch.randn((E, 2 * INTER, HIDDEN), generator=generator, device=device)
    w13 = w13.clamp(-3, 3).to(torch.float8_e4m3fn)
    w2 = torch.randn((E, HIDDEN, INTER), generator=generator, device=device)
    w2 = w2.clamp(-3, 3).to(torch.float8_e4m3fn)
    w13_scale = torch.rand(
        (E, 2 * INTER // GROUP, HIDDEN // GROUP),
        generator=generator,
        device=device,
    ) * 0.09 + 0.01
    w2_scale = torch.rand(
        (E, HIDDEN // GROUP, INTER // GROUP),
        generator=generator,
        device=device,
    ) * 0.09 + 0.01
    weights = {
        "w13": w13,
        "w13_scale": w13_scale,
        "w2": w2,
        "w2_scale": w2_scale,
    }

    results = [
        run_geometry(
            mok_ext,
            warp_ext,
            weights,
            geometry,
            groups=args.groups,
            replays_per_group=args.replays_per_group,
            pair_slots=args.pair_slots,
        )
        for geometry in GEOMETRIES
    ]
    if any(result["state"] != "COMPLETE" for result in results):
        disposition = "INVALID_CORRECTNESS"
    else:
        total_weight = sum(result["sample_weight"] for result in results)
        weighted_control = sum(
            result["sample_weight"] * result["control_median_ms"]
            for result in results
        ) / total_weight
        weighted_candidate = sum(
            result["sample_weight"] * result["candidate_median_ms"]
            for result in results
        ) / total_weight
        weighted_improvement = relative_improvement(
            weighted_control, weighted_candidate
        )
        primary_pass = weighted_improvement >= 5.0
        geometry_pass = all(
            result["median_improvement_percent"] >= 0.0
            and result["p95_improvement_percent"] >= -3.0
            and abs(result["control_drift_percent"]) <= 2.0
            for result in results
        )
        disposition = (
            "TAIL_OPERATOR_SIGNAL"
            if primary_pass and geometry_pass
            else "TAIL_OPERATOR_NO_GO"
        )

    payload = {
        "schema": "expertlane-m64-tail-hybrid-screen.v1",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "shape": {
            "experts": E,
            "max_m": MAX_M,
            "hidden": HIDDEN,
            "inter": INTER,
            "group_size": GROUP,
        },
        "timing": {
            "groups": args.groups,
            "replays_per_group": args.replays_per_group,
            "order": "control-candidate-candidate-control",
            "cuda_graph": True,
            "pair_slots": args.pair_slots,
        },
        "results": results,
        "disposition": disposition,
        "gate": {
            "weighted_median_improvement_percent_min": 5.0,
            "per_geometry_median_regression_percent_max": 0.0,
            "per_geometry_p95_regression_percent_max": 3.0,
            "control_drift_percent_max": 2.0,
        },
        "scope_note": (
            "Tail-only operator screen. Full M64 tiles, route split metadata, "
            "dispatch/combine, and service scheduling are not timed."
        ),
    }
    if all(result["state"] == "COMPLETE" for result in results):
        total_weight = sum(result["sample_weight"] for result in results)
        payload["weighted"] = {
            "control_median_ms": sum(
                result["sample_weight"] * result["control_median_ms"]
                for result in results
            )
            / total_weight,
            "candidate_median_ms": sum(
                result["sample_weight"] * result["candidate_median_ms"]
                for result in results
            )
            / total_weight,
        }
        payload["weighted"]["median_improvement_percent"] = relative_improvement(
            payload["weighted"]["control_median_ms"],
            payload["weighted"]["candidate_median_ms"],
        )

    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n")
    print(encoded)
    return 0 if disposition == "TAIL_OPERATOR_SIGNAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
