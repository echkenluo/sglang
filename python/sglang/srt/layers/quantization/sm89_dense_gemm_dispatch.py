# SPDX-License-Identifier: Apache-2.0
"""Backend choice for block-FP8 dense linears on SM89 (L20).

With SGLANG_FORCE_FP8_MARLIN=1 every block-FP8 dense linear runs the Marlin
W8A16 kernel. On L20 that kernel is only competitive below about 16 rows.
Between 24 and 128 rows it is 2x to 13x slower than a BF16 cuBLAS GEMM on the
dequantized weight (it needs two launches at 96 rows and its blocked GEMM is
compute-inefficient), and above a few hundred rows the Triton W8A8 block-FP8
kernel is 1.5x to 2x faster than both.

With the tuned launch configs in kernels/ops/quantization/configs
(device_name=NVIDIA_L20, block_shape=[128, 128]) the Triton W8A8 kernel,
activation quantization included, is the fastest or within 1 us of the fastest
backend from 1 to 4096 rows on all five shapes (43 distinct weight copies per
backend, bench-l20-dense-gemm-backends.py --layers 43, 2026-09-21; wqkv_a at 6
rows: Triton 11.5 us, Marlin 13.3 us, BF16 23.5 us; at 96 rows: 14.9, 63.4 and
23.9 us). The built-in policy therefore sends every row count to Triton. Such
a layer skips the Marlin repack and runs on the loaded FP8 weight, so it needs
no extra weight copy; the earlier mixed policy cost 14% of the KV capacity.
W8A8 with 128-wide activation groups is also the numeric format the
checkpoint was trained in, Marlin W8A16 is the deviation.

The untuned Triton default (BLOCK_SIZE_M=64) is 2x to 3x slower below 128
rows, so the policy is only valid together with the tuned config files.

`choose_backend` picks one backend per call from the row count. Row counts are
static inside a CUDA graph, so the choice costs nothing at replay time. Mixed
policies stay available through SGLANG_SM89_FP8_LINEAR_POLICY: a layer that
keeps Marlin for small row counts pays one extra FP8 copy for Triton, and a
BF16 copy only if its policy has a BF16 range. Shapes that are not in the
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
# DeepSeek-V4-Flash at TP8. (0, 0) is Marlin-free: Triton at every row count.
_L20_POLICY: Dict[Tuple[int, int], Tuple[int, int]] = {
    (4096, 1536): (0, 0),  # attention wqkv_a (fused wq_a + wkv)
    (1024, 4096): (0, 0),  # attention wq_b, wo_b
    (1024, 8192): (0, 0),  # indexer wq_b
    (4096, 512): (0, 0),  # shared expert gate_up
    (256, 4096): (0, 0),  # shared expert down
}

_extra_bytes = 0
_logged_shapes = set()
_logged_gib = [0]


def enabled_backends() -> Tuple[str, ...]:
    raw = envs.SGLANG_SM89_FP8_LINEAR_DISPATCH.get() or ""
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    for name in names:
        if name not in (BF16, TRITON):
            raise ValueError(
                f"SGLANG_SM89_FP8_LINEAR_DISPATCH accepts 'bf16' and 'triton', got {name!r}"
            )
    return names


def policy_table() -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Built-in table plus SGLANG_SM89_FP8_LINEAR_POLICY overrides.

    Override syntax: "K:N:marlin_max_rows:bf16_max_rows" entries joined by ";",
    for example "4096:1536:0:0;4096:512:0:0".
    """
    table = dict(_L20_POLICY)
    raw = envs.SGLANG_SM89_FP8_LINEAR_POLICY.get() or ""
    for entry in (part.strip() for part in raw.split(";")):
        if not entry:
            continue
        fields = entry.split(":")
        if len(fields) != 4 or not all(f.strip().isdigit() for f in fields):
            raise ValueError(
                "SGLANG_SM89_FP8_LINEAR_POLICY entries are "
                f"'K:N:marlin_max_rows:bf16_max_rows', got {entry!r}"
            )
        k, n, marlin_max_rows, bf16_max_rows = (int(f) for f in fields)
        table[(k, n)] = (marlin_max_rows, bf16_max_rows)
    return table


def choose_backend(
    rows: int, marlin_max_rows: int, bf16_max_rows: int, has_bf16: bool, has_triton: bool
) -> str:
    if rows <= marlin_max_rows:
        return MARLIN
    if rows <= bf16_max_rows and has_bf16:
        return BF16
    if has_triton:
        return TRITON
    # Without the Triton copy BF16 is still at least as fast as Marlin here.
    return BF16 if has_bf16 else MARLIN


def prepare_layer(layer: torch.nn.Module, weight_block_size) -> bool:
    """Attach what the policy needs. Call before the Marlin repack consumes
    `layer.weight` and `layer.weight_scale_inv`.

    Returns True when the layer never uses Marlin; the caller must then skip
    the Marlin repack and leave `layer.weight` as loaded."""
    global _extra_bytes

    backends = enabled_backends()
    if not backends:
        return False
    if getattr(layer, "bias", None) is not None:
        return False
    key = (layer.input_size_per_partition, layer.output_size_per_partition)
    thresholds = policy_table().get(key)
    if thresholds is None:
        return False
    marlin_max_rows, bf16_max_rows = thresholds

    weight = layer.weight.data
    scale_inv = layer.weight_scale_inv.data.to(torch.float32)
    assert weight.shape == (key[1], key[0]), (weight.shape, key)
    no_marlin = marlin_max_rows <= 0 and TRITON in backends

    if BF16 in backends and bf16_max_rows > marlin_max_rows:
        from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

        layer.sm89_bf16_weight = block_quant_dequant(
            weight, scale_inv, list(weight_block_size), torch.bfloat16
        ).contiguous()
        _extra_bytes += layer.sm89_bf16_weight.numel() * 2
    if TRITON in backends:
        # The Marlin repack replaces layer.weight, so a layer that keeps Marlin
        # needs its own FP8 copy; a Marlin-free layer uses the loaded weight.
        layer.sm89_fp8_weight = weight if no_marlin else weight.clone()
        layer.sm89_fp8_scale_inv = scale_inv if no_marlin else scale_inv.clone()
        layer.sm89_fp8_block_size = list(weight_block_size)
        if not no_marlin:
            _extra_bytes += weight.numel() + scale_inv.numel() * 4

    layer.sm89_dense_thresholds = thresholds
    layer.sm89_no_marlin = no_marlin
    # One line per shape, then one line per GiB, instead of one per layer.
    if key not in _logged_shapes or _extra_bytes // 2**30 > _logged_gib[0]:
        _logged_shapes.add(key)
        _logged_gib[0] = _extra_bytes // 2**30
        logger.info(
            "SM89 dense GEMM dispatch: K=%d N=%d thresholds=%s backends=%s "
            "marlin=%s, extra weight memory so far %.2f GiB",
            key[0],
            key[1],
            thresholds,
            ",".join(backends),
            "no" if no_marlin else "yes",
            _extra_bytes / 2**30,
        )
    return no_marlin


def apply(
    layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Return the output, or None when the caller should run Marlin."""
    thresholds = getattr(layer, "sm89_dense_thresholds", None)
    if thresholds is None:
        return None
    if bias is not None:
        assert not layer.sm89_no_marlin, "Marlin-free SM89 dense layers carry no bias"
        return None
    rows = x.numel() // x.shape[-1]
    if rows == 0 and layer.sm89_no_marlin:
        # A Marlin-free layer has no Marlin weight to fall back to.
        return x.new_empty((*x.shape[:-1], layer.sm89_fp8_weight.shape[0]))
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
