#!/usr/bin/env python3
"""Compare MoK and DeepGEMM FP8 block GEMMs against one FP32 reference.

This is a diagnostic, not a quality-gate substitute.  It deliberately feeds
both kernels the same quantized tensors and logical row-major scale values.
Only DeepGEMM receives the TMA-aligned view required by its H20 production
path.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

os.environ.setdefault("MOK_SM90_EXPERIMENTAL", "1")
os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "1")

import torch


BLOCK = 128


@dataclass(frozen=True)
class ErrorMetrics:
    mean_abs: float
    p95_abs: float
    max_abs: float
    relative_l2: float
    exact_bf16_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument(
        "--active-experts",
        type=int,
        help="experts receiving rows; defaults to min(experts, m / 64)",
    )
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", help="optional JSON output path")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.experts < 1:
        raise ValueError("experts must be positive")
    if args.m < 64 or args.m % 64:
        raise ValueError("m must be at least 64 and divisible by 64")
    if args.n < BLOCK or args.n % BLOCK:
        raise ValueError(f"n must be divisible by {BLOCK}")
    if args.k < BLOCK or args.k % BLOCK:
        raise ValueError(f"k must be divisible by {BLOCK}")
    active_experts = (
        args.active_experts
        if args.active_experts is not None
        else min(args.experts, args.m // 64)
    )
    if active_experts < 1 or active_experts > args.experts:
        raise ValueError("active-experts must be in [1, experts]")
    if active_experts > args.m // 64:
        raise ValueError("each active expert must receive at least one 64-row tile")
    args.active_experts = active_experts


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    fp8 = torch.float8_e4m3fn
    a = (
        torch.randn((args.m, args.k), generator=generator, device=device)
        .clamp(-3, 3)
        .to(fp8)
    )
    b = (
        torch.randn(
            (args.experts, args.n, args.k), generator=generator, device=device
        )
        .clamp(-3, 3)
        .to(fp8)
    )
    a_scale = (
        torch.rand(
            (args.m, args.k // BLOCK), generator=generator, device=device
        )
        * 0.09
        + 0.01
    )
    b_scale = (
        torch.rand(
            (args.experts, args.n // BLOCK, args.k // BLOCK),
            generator=generator,
            device=device,
        )
        * 0.09
        + 0.01
    )
    # Production dispatch pads every active expert segment to a 64-row tile.
    # Keep segments contiguous; the remaining experts are intentionally empty.
    num_tiles = args.m // 64
    tile_experts = (
        torch.arange(num_tiles, dtype=torch.int64, device=device)
        .mul_(args.active_experts)
        .div_(num_tiles, rounding_mode="floor")
        .to(torch.int32)
    )
    m_indices = torch.repeat_interleave(tile_experts, 64)
    return a, b, a_scale, b_scale, m_indices


def fp32_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape
    _, n, _ = b.shape
    output = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for expert in range(b.shape[0]):
        rows = torch.where(m_indices == expert)[0]
        if rows.numel() == 0:
            continue
        accumulator = torch.zeros(
            (rows.numel(), n), dtype=torch.float32, device=a.device
        )
        for kb in range(k // BLOCK):
            reduction_slice = slice(kb * BLOCK, (kb + 1) * BLOCK)
            partial = (
                a[rows, reduction_slice].float()
                @ b[expert, :, reduction_slice].float().T
            )
            weight_scale = b_scale[expert, :, kb].repeat_interleave(BLOCK)
            accumulator.add_(
                partial * a_scale[rows, kb, None] * weight_scale[None, :]
            )
        output[rows] = accumulator
    return output


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> ErrorMetrics:
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    error = (actual_f32 - reference_f32).abs().flatten()
    relative_l2 = torch.linalg.vector_norm(actual_f32 - reference_f32) / torch.linalg.vector_norm(
        reference_f32
    ).clamp_min(1e-12)
    return ErrorMetrics(
        mean_abs=error.mean().item(),
        p95_abs=torch.quantile(error, 0.95).item(),
        max_abs=error.max().item(),
        relative_l2=relative_l2.item(),
        exact_bf16_fraction=(actual == reference).float().mean().item(),
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("this audit requires SM90")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")

    from mok.functional import grouped_gemm_fp8_block_out
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.moe.ep_moe.kernels import tma_align_input_scale

    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        raise RuntimeError("SGLang JIT DeepGEMM is not enabled")

    a, b, a_scale, b_scale, m_indices = make_inputs(args)
    mok_output = torch.empty((args.m, args.n), dtype=torch.bfloat16, device="cuda")
    deepgemm_output = torch.empty_like(mok_output)

    grouped_gemm_fp8_block_out(
        a, b, a_scale, b_scale, m_indices, mok_output
    )
    deepgemm_a_scale = (
        tma_align_input_scale(a_scale)
        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES
        else a_scale
    )
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
        (a, deepgemm_a_scale),
        (b, b_scale),
        deepgemm_output,
        m_indices,
    )
    reference_fp32 = fp32_reference(a, b, a_scale, b_scale, m_indices)
    reference_bf16 = reference_fp32.to(torch.bfloat16)
    torch.cuda.synchronize()

    result = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "shape": {
            "experts": args.experts,
            "active_experts": args.active_experts,
            "m": args.m,
            "n": args.n,
            "k": args.k,
        },
        "seed": args.seed,
        "reference": "FP32 sum of K128 FP32 matmuls, then BF16 cast",
        "mok_vs_reference_bf16": asdict(
            error_metrics(mok_output, reference_bf16)
        ),
        "deepgemm_vs_reference_bf16": asdict(
            error_metrics(deepgemm_output, reference_bf16)
        ),
        "mok_vs_deepgemm": asdict(
            error_metrics(mok_output, deepgemm_output)
        ),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
