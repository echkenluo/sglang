# SPDX-License-Identifier: Apache-2.0
"""Row-count based backend choice for block-FP8 dense linears on SM89 (L20).

With SGLANG_FORCE_FP8_MARLIN=1 every block-FP8 dense linear runs the Marlin
W8A16 kernel. On L20 that kernel is the best choice only for very few rows.
Between 24 and 128 rows it is 2x to 13x slower than a BF16 cuBLAS GEMM on the
dequantized weight (it needs two launches at 96 rows and its blocked GEMM is
compute-inefficient), and above a few hundred rows the Triton W8A8 block-FP8
kernel is 1.5x to 2x faster than both.

This module keeps the Marlin weight and adds up to two more copies per layer:
a BF16 dequantized weight and the original block-FP8 weight. `choose_backend`
picks one per call from the row count. Row counts are static inside a CUDA
graph, so the choice costs nothing at replay time.

The thresholds come from a single-GPU microbenchmark on L20
(bench-l20-dense-gemm-backends.py, 2026-09-21). Shapes that are not in the
table keep the plain Marlin path and use no extra memory.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

MARLIN = "marlin"
BF16 = "bf16"
TRITON = "triton"

# (input_size_per_partition, output_size_per_partition)
#   -> (marlin_max_rows, bf16_max_rows)
# rows <= marlin_max_rows           : Marlin W8A16
# marlin_max_rows < rows <= bf16_max: BF16 cuBLAS on the dequantized weight
# rows > bf16_max_rows              : Triton W8A8 block FP8
# DeepSeek-V4-Flash at TP8. Decode and DSpark verify never exceed 96 rows, so
# they never reach the W8A8 kernel and keep weight-only numerics.
_L20_POLICY: Dict[Tuple[int, int], Tuple[int, int]] = {
    (4096, 1536): (16, 512),  # attention wqkv_a (fused wq_a + wkv)
    (1024, 4096): (16, 128),  # attention wq_b, wo_b
    (1024, 8192): (16, 128),  # indexer wq_b
    (4096, 512): (0, 1024),  # shared expert gate_up
    (256, 4096): (1, 256),  # shared expert down
}

_extra_bytes = 0


def enabled_backends() -> Tuple[str, ...]:
    raw = envs.SGLANG_SM89_FP8_LINEAR_DISPATCH.get() or ""
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    for name in names:
        if name not in (BF16, TRITON):
            raise ValueError(
                f"SGLANG_SM89_FP8_LINEAR_DISPATCH accepts 'bf16' and 'triton', got {name!r}"
            )
    return names


def choose_backend(
    rows: int, marlin_max_rows: int, bf16_max_rows: int, has_bf16: bool, has_triton: bool
) -> str:
    if rows <= marlin_max_rows:
        return MARLIN
    if rows <= bf16_max_rows:
        return BF16 if has_bf16 else MARLIN
    if has_triton:
        return TRITON
    # Without the Triton copy BF16 is still at least as fast as Marlin here.
    return BF16 if has_bf16 else MARLIN


def prepare_layer(layer: torch.nn.Module, weight_block_size) -> None:
    """Attach the extra weight copies. Call before the Marlin repack consumes
    `layer.weight` and `layer.weight_scale_inv`."""
    global _extra_bytes

    backends = enabled_backends()
    if not backends:
        return
    if getattr(layer, "bias", None) is not None:
        return
    key = (layer.input_size_per_partition, layer.output_size_per_partition)
    thresholds = _L20_POLICY.get(key)
    if thresholds is None:
        return

    weight = layer.weight.data
    scale_inv = layer.weight_scale_inv.data.to(torch.float32)
    assert weight.shape == (key[1], key[0]), (weight.shape, key)

    if BF16 in backends:
        from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

        layer.sm89_bf16_weight = block_quant_dequant(
            weight, scale_inv, list(weight_block_size), torch.bfloat16
        ).contiguous()
        _extra_bytes += layer.sm89_bf16_weight.numel() * 2
    if TRITON in backends:
        layer.sm89_fp8_weight = weight.clone()
        layer.sm89_fp8_scale_inv = scale_inv.clone()
        layer.sm89_fp8_block_size = list(weight_block_size)
        _extra_bytes += weight.numel() + scale_inv.numel() * 4

    layer.sm89_dense_thresholds = thresholds
    logger.info(
        "SM89 dense GEMM dispatch: K=%d N=%d thresholds=%s backends=%s, "
        "extra weight memory so far %.2f GiB",
        key[0],
        key[1],
        thresholds,
        ",".join(backends),
        _extra_bytes / 2**30,
    )


def apply(
    layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Return the output, or None when the caller should run Marlin."""
    thresholds = getattr(layer, "sm89_dense_thresholds", None)
    if thresholds is None or bias is not None:
        return None
    rows = x.numel() // x.shape[-1]
    backend = choose_backend(
        rows,
        thresholds[0],
        thresholds[1],
        hasattr(layer, "sm89_bf16_weight"),
        hasattr(layer, "sm89_fp8_weight"),
    )
    if backend == BF16:
        return F.linear(x, layer.sm89_bf16_weight)
    if backend == TRITON:
        from sglang.srt.layers.quantization.fp8_utils import (
            triton_w8a8_block_fp8_linear,
        )

        return triton_w8a8_block_fp8_linear(
            input=x,
            weight=layer.sm89_fp8_weight,
            block_size=layer.sm89_fp8_block_size,
            weight_scale=layer.sm89_fp8_scale_inv,
            input_scale=None,
            bias=None,
        )
    return None
