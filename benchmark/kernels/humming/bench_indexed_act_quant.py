#!/usr/bin/env python3
"""Benchmark indexed Humming SwiGLU plus per-token FP8 quantization."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from humming import ops
from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    act_and_mul_quant_fp8_per_token,
    act_and_mul_triton,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[6, 12, 24, 48, 96, 192, 384, 768, 1536, 196608],
    )
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = math.ceil(quantile * len(ordered)) - 1
    return ordered[max(index, 0)]


def time_us(fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "p95_us": percentile(samples, 0.95),
    }


def benchmark_shape(
    shape_m: int,
    intermediate: int,
    swiglu_limit: float,
    repeats: int,
) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(20260830 + shape_m)
    gate_up = (
        torch.randn(
            shape_m,
            2 * intermediate,
            generator=generator,
            device="cuda",
        )
        * 3.0
    ).to(torch.bfloat16)
    activation = torch.empty(
        shape_m, intermediate, device="cuda", dtype=torch.bfloat16
    )
    baseline_q = torch.empty(
        shape_m, intermediate, device="cuda", dtype=torch.float8_e4m3fn
    )
    fused_q = torch.empty_like(baseline_q)

    def activation_fn():
        act_and_mul_triton(
            gateup_output=gate_up,
            down_input=activation,
            config={},
            activation="silu",
            swiglu_limit=swiglu_limit,
        )

    def quant_fn():
        return ops.quant_input(
            activation,
            dtype="float8e4m3",
            outputs=baseline_q,
            group_size=None,
        )

    def baseline_fn():
        activation_fn()
        return quant_fn()

    def fused_fn():
        return act_and_mul_quant_fp8_per_token(
            gate_up, fused_q, swiglu_limit=swiglu_limit
        )

    for _ in range(10):
        baseline_fn()
        fused_fn()
    torch.cuda.synchronize()

    _, baseline_scale = baseline_fn()
    fused_scale = fused_fn()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        fused_scale, baseline_scale, rtol=5e-3, atol=1e-7
    )
    reference = activation.float()
    fused_dequant = fused_q.float() * fused_scale
    if not bool(((fused_dequant - reference).abs() <= fused_scale * 17.0 + 1e-4).all()):
        raise AssertionError(f"fused dequantization gate failed for M={shape_m}")

    iterations = 500
    if shape_m >= 1024:
        iterations = 50
    if shape_m >= 65536:
        iterations = 10

    baseline_samples = []
    fused_samples = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            baseline_samples.append(time_us(baseline_fn, iterations))
            fused_samples.append(time_us(fused_fn, iterations))
        else:
            fused_samples.append(time_us(fused_fn, iterations))
            baseline_samples.append(time_us(baseline_fn, iterations))

    activation_samples = [time_us(activation_fn, iterations) for _ in range(repeats)]
    quant_samples = [time_us(quant_fn, iterations) for _ in range(repeats)]
    baseline_summary = summarize(baseline_samples)
    fused_summary = summarize(fused_samples)
    baseline_median = float(baseline_summary["median_us"])
    fused_median = float(fused_summary["median_us"])

    return {
        "shape_m": shape_m,
        "intermediate": intermediate,
        "iterations": iterations,
        "baseline": baseline_summary,
        "fused": fused_summary,
        "activation": summarize(activation_samples),
        "quant": summarize(quant_samples),
        "fused_delta_pct": (fused_median / baseline_median - 1.0) * 100.0,
        "scale_max_abs_diff": float((fused_scale - baseline_scale).abs().max()),
        "q_byte_equal_ratio": float(
            (fused_q.view(torch.uint8) == baseline_q.view(torch.uint8)).float().mean()
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    result = {
        "contract": {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "m_values": args.m_values,
            "intermediate": args.intermediate,
            "swiglu_limit": args.swiglu_limit,
            "repeats": args.repeats,
            "baseline": "act_and_mul_triton + humming.ops.quant_input(group_size=None)",
            "candidate": "act_and_mul_quant_fp8_per_token",
        },
        "points": [
            benchmark_shape(
                shape_m,
                args.intermediate,
                args.swiglu_limit,
                args.repeats,
            )
            for shape_m in args.m_values
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
