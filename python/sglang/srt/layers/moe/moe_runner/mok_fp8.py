"""Feature-gated Mixture-of-Kittens FP8 block-scale expert compute."""

from __future__ import annotations

import functools
from typing import Optional

import torch


def shape_contract_error(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    masked_m: torch.Tensor,
) -> Optional[str]:
    """Return the first unsupported static tensor-contract reason."""

    if hidden_states.dim() != 3:
        return "hidden_states must have shape [E,max_m,H]"
    experts, max_m, hidden = hidden_states.shape
    if w13_weight.dim() != 3 or w2_weight.dim() != 3:
        return "expert weights must be rank-3"
    if w13_weight.shape[0] != experts or w2_weight.shape[0] != experts:
        return "hidden_states and weights must have the same expert count"
    gate_up = w13_weight.shape[1]
    intermediate = w2_weight.shape[2]
    if gate_up != 2 * intermediate:
        return "w13 output must be twice the w2 reduction dimension"
    if w13_weight.shape[2] != hidden:
        return "w13 reduction dimension must equal hidden size"
    if w2_weight.shape[1] != hidden:
        return "w2 output dimension must equal hidden size"
    if max_m < 64 or max_m % 64 != 0:
        return "max_m must be positive and divisible by 64"
    if hidden % 128 != 0 or intermediate % 128 != 0:
        return "hidden and intermediate dimensions must be divisible by 128"

    fp8_dtype = torch.float8_e4m3fn
    if any(
        tensor.dtype != fp8_dtype
        for tensor in (hidden_states, w13_weight, w2_weight)
    ):
        return "hidden_states and weights must use float8_e4m3fn"
    if any(
        tensor.dtype != torch.float32
        for tensor in (hidden_states_scale, w13_scale, w2_scale)
    ):
        return "block scales must use float32"
    if masked_m.dtype != torch.int32:
        return "masked_m must use int32"

    expected_shapes = (
        (experts, max_m, hidden // 128),
        (experts, gate_up // 128, hidden // 128),
        (experts, hidden // 128, intermediate // 128),
        (experts,),
    )
    actual_shapes = (
        tuple(hidden_states_scale.shape),
        tuple(w13_scale.shape),
        tuple(w2_scale.shape),
        tuple(masked_m.shape),
    )
    if actual_shapes != expected_shapes:
        return (
            "invalid block-scale/mask shapes: "
            f"expected={expected_shapes}, actual={actual_shapes}"
        )

    tensors = (
        hidden_states,
        hidden_states_scale,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        masked_m,
    )
    if not all(tensor.is_contiguous() for tensor in tensors):
        return "all MoK FP8 inputs must be contiguous"
    return None


def runtime_contract_error(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    masked_m: torch.Tensor,
) -> Optional[str]:
    error = shape_contract_error(
        hidden_states,
        hidden_states_scale,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        masked_m,
    )
    if error is not None:
        return error
    tensors = (
        hidden_states,
        hidden_states_scale,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        masked_m,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        return "all MoK FP8 inputs must be CUDA tensors"
    if not all(tensor.device == hidden_states.device for tensor in tensors):
        return "all MoK FP8 inputs must be on the same CUDA device"
    if torch.cuda.get_device_capability(hidden_states.device) != (9, 0):
        return "MoK FP8 block-scale backend currently requires SM90"
    if torch.cuda.is_current_stream_capturing():
        return "CUDA graph capture is not supported by the MoK FP8 canary"
    return None


@functools.lru_cache(maxsize=1)
def _extension():
    try:
        from mok import _C
    except ImportError as exc:
        raise RuntimeError(
            "SGLANG_OPT_USE_MOK_FP8_EXPERT_MLP is enabled, but the MoK "
            "extension is not importable"
        ) from exc
    if not hasattr(_C, "fp8_block_grouped_pipelined_out"):
        raise RuntimeError(
            "The loaded MoK extension does not expose "
            "fp8_block_grouped_pipelined_out"
        )
    return _C


def grouped_gemm_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    masked_m: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    return _extension().fp8_block_grouped_pipelined_out(
        input,
        weight,
        input_scale,
        weight_scale,
        masked_m,
        output,
    )
