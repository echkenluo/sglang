"""Warp-decode fast path for small-batch FP8 block-scale MoE decode.

Port of Cursor's warp-decode design (one warp per output scalar) to H20/SM90
with DeepSeek-style [128,128] FP8 block-scale weights and BF16 activations.
Based on the ZelinMa557/warp_decode BF16 Hopper reproduction; the FP8 variants
dequantize weights in registers, so no activation quantization kernel runs.

Enable with SGLANG_WARP_DECODE_MOE=1. Forward batches with more than
SGLANG_WARP_DECODE_MAX_TOKENS tokens (default 2) fall back to the triton
fused_moe path (expert-centric grouped GEMM wins at large per-expert batch).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("SGLANG_WARP_DECODE_MOE", "0") == "1"
# Default crossover from H20 V4-Flash TP4 measurements (2026-08-19):
# microbench wins at 1-2 tokens (+27%/+9%); E2E four-arm run showed c4
# slightly negative (+4% TPOT), so the default threshold is 2.
_MAX_TOKENS = int(os.environ.get("SGLANG_WARP_DECODE_MAX_TOKENS", "2"))
_VERBOSE = os.environ.get("SGLANG_WARP_DECODE_LOG", "0") == "1"
_MASKED_ENABLED = os.environ.get("SGLANG_WARP_DECODE_MASKED", "0") == "1"
# masked path safety gate: expected_m is a host-side per-graph constant from
# the LL dispatcher; decode captures sit at 1-2, prefill never reaches the
# masked runner under deepep-mode auto.
_MASKED_MAX_EXPECTED_M = int(
    os.environ.get("SGLANG_WARP_DECODE_MASKED_MAX_EXPECTED_M", "16"))
# pair_slots sweep on H20 V4 EP4 shapes (2026-08-19): 8/16/32/64 gave
# 0.030/0.032/0.038/0.047 ms at 2 copies and 0.446/0.423/0.421/0.428 ms at 48;
# 16 is within a few percent of best across the whole decode range.
_PAIR_SLOTS = int(os.environ.get("SGLANG_WARP_DECODE_PAIR_SLOTS", "16"))

_ext = None
_ext_failed = False
_logged_active = False
_logged_reject: set = set()


def _load_ext():
    global _ext, _ext_failed
    if _ext is not None or _ext_failed:
        return _ext
    try:
        from torch.utils.cpp_extension import load

        csrc = Path(__file__).resolve().parent / "warp_decode_csrc"
        _ext = load(
            name="sglang_warp_decode_ext",
            sources=[
                str(csrc / "binding_sglang.cpp"),
                str(csrc / "moe_gate_up_fp8_blockscale.cu"),
                str(csrc / "moe_down_fp8_blockscale.cu"),
                str(csrc / "moe_warp_masked.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-gencode=arch=compute_90,code=sm_90",
            ],
            verbose=_VERBOSE,
        )
        logger.info("warp-decode MoE extension built and loaded")
    except Exception as e:
        _ext_failed = True
        logger.warning(
            "warp-decode MoE extension build failed; falling back to triton: %s", e
        )
    return _ext


_hit_counters_enabled_cache: Optional[bool] = None


def _hit_counters_enabled() -> bool:
    global _hit_counters_enabled_cache
    if _hit_counters_enabled_cache is None:
        from sglang.srt.environ import envs

        _hit_counters_enabled_cache = envs.SGLANG_MOE_PATH_HIT_COUNTERS.get()
    return _hit_counters_enabled_cache


def _reject(reason: str) -> bool:
    if _VERBOSE and reason not in _logged_reject:
        _logged_reject.add(reason)
        logger.info("warp-decode path rejected (once per reason): %s", reason)
    return False


def should_use(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    moe_runner_config,
    b1: Optional[torch.Tensor],
    b2: Optional[torch.Tensor],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
) -> bool:
    if not _ENABLED:
        return False
    if hidden_states.shape[0] > _MAX_TOKENS:
        return False
    if not use_fp8_w8a8 or block_shape != [128, 128]:
        return _reject("requires use_fp8_w8a8 with block_shape [128,128]")
    if use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16 or per_channel_quant:
        return _reject("other quant modes active")
    if b1 is not None or b2 is not None or w1_zp is not None or w2_zp is not None:
        return _reject("bias/zero-point not supported")
    cfg = moe_runner_config
    if cfg.activation != "silu" or not cfg.is_gated:
        return _reject(f"activation {cfg.activation} / is_gated {cfg.is_gated}")
    # NOTE: cfg.gate_up_interleaved defaults to True and DeepSeek does not
    # override it, but the flag is only consumed by the gemm1_alpha
    # (GPT-OSS swiglu) branch; with gemm1_alpha/limit None the triton path
    # itself treats w13 as split gate/up halves, which is what our kernel
    # assumes. So we gate on gemm1_alpha/limit below, not on this flag.
    if cfg.apply_router_weight_on_input or cfg.no_combine:
        return _reject("apply_router_weight_on_input / no_combine")
    if cfg.gemm1_alpha is not None or cfg.gemm1_clamp_limit is not None:
        return _reject("gemm1 alpha/clamp")
    # swiglu_limit (DeepSeek V4 clamp) is supported: folded into the gate_up
    # kernel epilogue (gate: min(g, limit); up: clamp(u, +/-limit)).
    if (
        cfg.num_experts is not None
        and cfg.num_local_experts is not None
        and cfg.num_experts != cfg.num_local_experts
    ):
        return _reject("filter_expert (num_experts != num_local_experts)")
    if hidden_states.dtype != torch.bfloat16:
        return _reject(f"hidden dtype {hidden_states.dtype}")
    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return _reject("weights not fp8 e4m3fn")
    if w1_scale is None or w2_scale is None or w1_scale.dim() != 3:
        return _reject("missing 3D block scales")
    if w1_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
        return _reject("scales not fp32")
    hidden = hidden_states.shape[1]
    inter = w2.shape[2]
    if hidden % 512 != 0 or inter % 128 != 0:
        return _reject(f"shape not supported (hidden={hidden}, inter={inter})")
    if _load_ext() is None:
        return False
    return True


_logged_masked_active = False


def masked_should_use(
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    masked_m: torch.Tensor,
    expected_m: int,
    quant_info,
    swiglu_limit: Optional[float],
    running_state: dict,
) -> bool:
    if not _MASKED_ENABLED:
        return False
    if expected_m > _MASKED_MAX_EXPECTED_M:
        return False
    if running_state.get("down_gemm_overlap_args") is not None:
        return _reject("masked: down-gemm overlap (SBO) active")
    if getattr(quant_info, "is_fp4_experts", False):
        return _reject("masked: fp4 experts")
    if hidden_states.dtype != torch.float8_e4m3fn or hidden_states.dim() != 3:
        return _reject(f"masked: recv dtype/dim {hidden_states.dtype}")
    if hidden_states_scale is None or hidden_states_scale.dtype != torch.float32:
        return _reject("masked: recv scales missing or not fp32 (ue8m0?)")
    E, max_m, hidden = hidden_states.shape
    if hidden_states_scale.shape != (E, max_m, (hidden + 127) // 128):
        return _reject(f"masked: scale shape {tuple(hidden_states_scale.shape)}")
    w13, w2 = quant_info.w13_weight, quant_info.w2_weight
    w13s, w2s = quant_info.w13_scale, quant_info.w2_scale
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return _reject("masked: weights not fp8")
    if w13s is None or w2s is None or w13s.dim() != 3 or w13s.dtype != torch.float32:
        return _reject("masked: weight scales not 3D fp32")
    inter = w2.shape[2]
    if hidden % 512 != 0 or inter % 256 != 0:
        return _reject(f"masked: shape (hidden={hidden}, inter={inter})")
    if masked_m.dtype != torch.int32 or max_m >= 65536 or E > 1024:
        return _reject("masked: masked_m dtype or size limits")
    if _load_ext() is None:
        return False
    return True


def masked_run(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    masked_m: torch.Tensor,
    quant_info,
    swiglu_limit: Optional[float],
) -> torch.Tensor:
    global _logged_masked_active
    ext = _load_ext()

    E, max_m, hidden = hidden_states.shape
    inter = quant_info.w2_weight.shape[2]
    device = hidden_states.device

    x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    xs = (hidden_states_scale
          if hidden_states_scale.is_contiguous()
          else hidden_states_scale.contiguous())
    mm = masked_m if masked_m.is_contiguous() else masked_m.contiguous()

    pairs = torch.empty(1 + E * max_m, dtype=torch.int32, device=device)
    buf = torch.empty(E, max_m, inter, dtype=torch.bfloat16, device=device)
    out = torch.empty(E, max_m, hidden, dtype=torch.bfloat16, device=device)

    limit = float("inf") if swiglu_limit is None else float(swiglu_limit)

    if not _logged_masked_active:
        _logged_masked_active = True
        logger.info(
            "warp-decode MASKED path ACTIVE: E=%d max_m=%d hidden=%d inter=%d "
            "limit=%s pair_slots=%d",
            E, max_m, hidden, inter, swiglu_limit, _PAIR_SLOTS,
        )

    # Canary-only coexistence counter. The enabled flag is cached once per
    # process so the off-path cost in the decode hot loop stays at a global
    # read; lazy imports keep this module free of the heavy mok chain.
    if _hit_counters_enabled():
        from sglang.srt.layers.dp_attention import get_is_extend_in_batch
        from sglang.srt.layers.moe.moe_runner.mok_fp8_native import _note_path_hit

        _note_path_hit(
            "warp_masked",
            mode="extend" if get_is_extend_in_batch() else "decode",
            num_tokens=None,
        )

    ext.moe_warp_masked_fp8(
        x, xs,
        quant_info.w13_weight, quant_info.w13_scale,
        quant_info.w2_weight, quant_info.w2_scale,
        mm, pairs, buf, out, limit, _PAIR_SLOTS,
    )
    return out


def run_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    moe_runner_config,
) -> torch.Tensor:
    global _logged_active
    ext = _load_ext()

    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    if not ids.is_contiguous():
        ids = ids.contiguous()

    scale = topk_weights
    if scale.dtype != torch.float32:
        scale = scale.float()
    rsf = moe_runner_config.routed_scaling_factor
    if rsf is not None and rsf != 1.0:
        scale = scale * rsf
    if not scale.is_contiguous():
        scale = scale.contiguous()

    x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()

    if not _logged_active:
        _logged_active = True
        logger.info(
            "warp-decode MoE path ACTIVE: tokens=%d topk=%d E=%d hidden=%d inter=%d "
            "rsf=%s max_tokens=%d",
            x.shape[0], ids.shape[1], w1.shape[0], x.shape[1], w2.shape[2],
            rsf, _MAX_TOKENS,
        )

    limit = moe_runner_config.swiglu_limit
    if limit is None:
        limit = float("inf")
    buf = ext.moe_gate_up_fp8_blockscale(x, w1, w1_scale, ids, float(limit))
    out = ext.moe_down_fp8_blockscale(buf, w2, w2_scale, ids, scale, x.shape[0])

    if moe_runner_config.inplace:
        hidden_states.copy_(out)
        return hidden_states
    return out
