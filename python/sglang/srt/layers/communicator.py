# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import (
    attention_tensor_model_parallel_all_reduce,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    moe_tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.device_communicators.tokenweave_fused import (
    _tokenweave_fusion_enabled,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa.utils import (
    is_nsa_enable_prefill_cp,
    nsa_use_prefill_cp,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    attn_tp_reduce_scatter_tensor,
    dp_gather_partial,
    dp_reduce_scatter_tensor,
    dp_scatter,
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_allocation_symmetric,
    is_dp_attention_enabled,
)
from sglang.srt.layers.flashinfer_comm_fusion import is_flashinfer_allreduce_unavailable
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_flashinfer_available,
    is_gfx95_supported,
    is_hip,
    is_npu,
    is_sm90_supported,
    is_sm100_supported,
)

_is_cuda = is_cuda()
_is_flashinfer_available = is_flashinfer_available()
_is_sm90_supported = _is_cuda and is_sm90_supported()
_is_sm100_supported = _is_cuda and is_sm100_supported()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
_is_gfx95_supported = is_gfx95_supported()
_is_npu = is_npu()
_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()

if _use_aiter and _is_gfx95_supported:
    from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant

    from sglang.srt.layers.quantization.rocm_mxfp4_utils import fused_rms_mxfp4_quant
elif _is_npu:
    from sglang.srt.hardware_backend.npu.cmo import prepare_weight_cache


# TODO: According to the discussion in https://github.com/flashinfer-ai/flashinfer/issues/1223#issuecomment-3047256465
# We set the max token num to 128 for allreduce fusion with min-latency case(use_oneshot=True).
FUSE_ALLREDUCE_MAX_BATCH_SIZE = 2048


def apply_flashinfer_allreduce_fusion(batch_size: int):
    return (
        # NOTE: flashinfer 0.6.1 caused performance regression on sm100 for allreduce fusion
        # Ref: https://github.com/sgl-project/sglang/issues/17237
        (_is_sm90_supported or _is_sm100_supported)
        and _is_flashinfer_available
        and batch_size > 0
        and batch_size <= FUSE_ALLREDUCE_MAX_BATCH_SIZE
        and not is_dp_attention_enabled()
        and get_global_server_args().enable_flashinfer_allreduce_fusion
        and not is_flashinfer_allreduce_unavailable()
        # FlashInfer's TRT-LLM allreduce backend creates its own NCCL communicator
        # which doesn't support PyTorch sub-process groups used by context parallelism
        and get_global_server_args().attn_cp_size <= 1
    )


def apply_aiter_all_reduce_fusion(input_tensor: torch.Tensor):
    n = input_tensor.shape[-1]
    total_bytes = input_tensor.numel() * input_tensor.element_size()
    return (
        _use_aiter
        and total_bytes > 0
        and n <= 16384
        and total_bytes < 8 * 1024 * 8192
        and get_tensor_model_parallel_world_size() != 6
        and not is_dp_attention_enabled()
        and get_global_server_args().enable_aiter_allreduce_fusion
    )


class ScatterMode(Enum):
    """
    Suppose we have TP=4, DP=2, enable-dp-attention, and the system handles seq a,b,c,d
    Model input/output: [ab, ab, cd, cd] for four ranks respectively
    SCATTERED: [a, b, c, d]
    TP_ATTN_FULL: [ab, ab, cd, cd], i.e. all ranks inside a TP attn group have full data of the group
    FULL: [abcd, abcd, abcd, abcd]
    """

    SCATTERED = auto()
    TP_ATTN_FULL = auto()
    FULL = auto()

    @staticmethod
    def model_input_output():
        """The scatter mode for model forward pass input and output data"""
        if is_nsa_enable_prefill_cp():
            return ScatterMode.SCATTERED

        return ScatterMode.TP_ATTN_FULL


# FP8-quantized all-gather for the input-scattered (RS+AG) path.
# AG transfers final values (not partial sums), so quantize-transfer-dequant
# is numerically safe; wire bytes for the hidden states are halved.
# SGLANG_FP8_AG=1        enable (packed single-collective path)
# SGLANG_FP8_AG_V1=1     legacy two-collective path (payload + scale gathered
#                        separately) for A/B comparison
# SGLANG_FP8_AG_SIDE     all (default) | mlp | attn — quantize only that AG site
_ENABLE_FP8_AG = get_bool_env_var("SGLANG_FP8_AG")
_FP8_AG_V1 = get_bool_env_var("SGLANG_FP8_AG_V1")
_FP8_AG_SIDE = os.environ.get("SGLANG_FP8_AG_SIDE", "all").lower()
_FP8_AG_GROUP_SIZE = 128


@triton.jit
def _fp8_ag_dequant_kernel(
    q_ptr,
    s_ptr,
    out_ptr,
    q_row_stride,
    s_row_stride,
    out_row_stride,
    GROUP: tl.constexpr,
):
    t = tl.program_id(0)
    g = tl.program_id(1)
    offs = g * GROUP + tl.arange(0, GROUP)
    q = tl.load(q_ptr + t * q_row_stride + offs).to(tl.float32)
    s = tl.load(s_ptr + t * s_row_stride + g)
    tl.store(out_ptr + t * out_row_stride + offs, q * s)


_FP8_AG_ENGAGED_LOGGED = False


def _tp_all_gather_hidden_states(hidden_states, forward_batch, side="all"):
    total_tokens = forward_batch.input_ids.shape[0]
    _tw_comm = getattr(get_tp_group(), "tokenweave_comm", None)
    if _tw_comm is not None and getattr(
        _tw_comm, "native_scattered_enabled", False
    ):
        # N2b: when hidden_states is the rank slice a fused scattered claim
        # just handed out, the full normed rows already sit in the symm
        # region on every rank (kernel multicast AG) -- hand out that view
        # and skip this gather entirely.
        _tw_full = _tw_comm.native_scattered_full_view(hidden_states, total_tokens)
        if _tw_full is not None:
            return _tw_full
    if (
        _ENABLE_FP8_AG
        and (_FP8_AG_SIDE == "all" or _FP8_AG_SIDE == side)
        and hidden_states.shape[0] > 0
        and hidden_states.shape[-1] % _FP8_AG_GROUP_SIZE == 0
    ):
        global _FP8_AG_ENGAGED_LOGGED
        if not _FP8_AG_ENGAGED_LOGGED:
            _FP8_AG_ENGAGED_LOGGED = True
            logging.info(
                f"[FP8-AG] engaged: side={side} tokens={total_tokens} "
                f"hidden={hidden_states.shape[-1]}"
            )
        # Packing overhead beats the saved collective launch only at larger
        # gathers (L1: v2 wins >=4k total tokens, loses at ~1k).
        if _FP8_AG_V1 or total_tokens < 4096:
            return _tp_all_gather_hidden_states_fp8(hidden_states, total_tokens)
        return _tp_all_gather_hidden_states_fp8_packed(hidden_states, total_tokens)
    output = hidden_states.new_empty((total_tokens, hidden_states.shape[-1]))
    get_tp_group().all_gather_into_tensor(output, hidden_states)
    return output


def _tp_all_gather_hidden_states_fp8_packed(hidden_states, total_tokens):
    """Single-collective variant: fp8 payload and fp32 scales are packed into
    one byte row per token, gathered once, then dequantized by a fused kernel.

    Note: a fused quant-into-packed-layout triton kernel was benchmarked and
    showed no gain over quant + slice copies (the copies are not a bottleneck);
    keeping the simpler form."""
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    hidden = hidden_states.shape[-1]
    scale_bytes = (hidden // _FP8_AG_GROUP_SIZE) * 4
    row_bytes = hidden + scale_bytes
    x_q, x_s = sglang_per_token_group_quant_fp8(
        hidden_states.contiguous(), _FP8_AG_GROUP_SIZE
    )
    local_tokens = x_q.shape[0]
    packed = torch.empty(
        (local_tokens, row_bytes), dtype=torch.uint8, device=x_q.device
    )
    packed[:, :hidden] = x_q.view(torch.uint8)
    packed[:, hidden:] = x_s.view(torch.uint8)
    packed_full = torch.empty(
        (total_tokens, row_bytes), dtype=torch.uint8, device=x_q.device
    )
    get_tp_group().all_gather_into_tensor(packed_full, packed)
    q_full = packed_full.view(torch.float8_e4m3fn)[:, :hidden]
    s_full = (
        packed_full[:, hidden:]
        .contiguous()
        .view(torch.float32)
        .view(total_tokens, hidden // _FP8_AG_GROUP_SIZE)
    )
    output = torch.empty(
        (total_tokens, hidden), dtype=hidden_states.dtype, device=x_q.device
    )
    _fp8_ag_dequant_kernel[(total_tokens, hidden // _FP8_AG_GROUP_SIZE)](
        q_full,
        s_full,
        output,
        packed_full.stride(0),
        s_full.stride(0),
        output.stride(0),
        GROUP=_FP8_AG_GROUP_SIZE,
    )
    return output


def _tp_all_gather_hidden_states_fp8(hidden_states, total_tokens):
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    x_q, x_s = sglang_per_token_group_quant_fp8(
        hidden_states.contiguous(), _FP8_AG_GROUP_SIZE
    )
    tp_group = get_tp_group()
    q_full = torch.empty(
        (total_tokens, x_q.shape[-1]), dtype=x_q.dtype, device=x_q.device
    )
    s_full = torch.empty(
        (total_tokens, x_s.shape[-1]), dtype=x_s.dtype, device=x_s.device
    )
    # NCCL has no fp8 dtype; gather the quantized payload as raw bytes.
    tp_group.all_gather_into_tensor(q_full.view(torch.uint8), x_q.view(torch.uint8))
    tp_group.all_gather_into_tensor(s_full, x_s)
    output = (
        q_full.to(hidden_states.dtype)
        .view(total_tokens, -1, _FP8_AG_GROUP_SIZE)
        .mul_(s_full.unsqueeze(-1).to(hidden_states.dtype))
        .view(total_tokens, x_q.shape[-1])
    )
    return output


class AttentionInputs:

    def __init__(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        qkv_latent_func: Callable,
    ):
        self.hidden_states_local = hidden_states
        self.forward_batch = forward_batch
        self.qkv_latent_func = qkv_latent_func
        self.hidden_states_ = None
        self.qkv_latent_ = None

    def tp_all_gather_hidden_states(self, hidden_states, forward_batch):
        return _tp_all_gather_hidden_states(hidden_states, forward_batch, side="attn")

    def fetch_qkv_latent(self):
        if self.qkv_latent_ is not None:
            return self.qkv_latent_
        assert self.qkv_latent_func is not None
        self.qkv_latent_ = self.qkv_latent_func(
            self.hidden_states_local, self.forward_batch
        )
        if get_attn_tp_context().input_scattered:
            self.qkv_latent_ = self.tp_all_gather_hidden_states(
                self.qkv_latent_, self.forward_batch
            )
        return self.qkv_latent_

    def fetch_hidden_states(self):
        if self.hidden_states_ is not None:
            return self.hidden_states_
        self.hidden_states_ = self.hidden_states_local
        if get_attn_tp_context().input_scattered:
            self.hidden_states_ = self.tp_all_gather_hidden_states(
                self.hidden_states_, self.forward_batch
            )
        return self.hidden_states_
    

class MLPInputs:

    def __init__(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        mlp_latent_func: Callable,
    ):
        self.hidden_states_local = hidden_states
        self.forward_batch = forward_batch
        self.mlp_latent_func = mlp_latent_func
        self.hidden_states_ = None
        self.mlp_latent_ = None

    def tp_all_gather_hidden_states(self, hidden_states, forward_batch):
        return _tp_all_gather_hidden_states(hidden_states, forward_batch, side="mlp")

    def fetch_mlp_latent(self, last_layer=False):
        if self.mlp_latent_ is not None:
            return self.mlp_latent_
        assert self.mlp_latent_func is not None
        self.mlp_latent_ = self.mlp_latent_func(
            self.hidden_states_local, self.forward_batch
        )
        if get_attn_tp_context().input_scattered and not last_layer:
            self.mlp_latent_ = self.tp_all_gather_hidden_states(
                self.mlp_latent_, self.forward_batch
            )
        return self.mlp_latent_

    def fetch_hidden_states(self):
        if self.hidden_states_ is not None:
            return self.hidden_states_
        self.hidden_states_ = self.hidden_states_local
        if get_attn_tp_context().input_scattered:
            self.hidden_states_ = self.tp_all_gather_hidden_states(
                self.hidden_states_, self.forward_batch
            )
        return self.hidden_states_


class AttnTpContext:
    def __init__(self):
        self.allow_input_scattered = False
        self.input_scattered_ = False
        self.attn_inputs_: Optional[AttentionInputs] = None
        self.mlp_inputs_: Optional[MLPInputs] = None

    def init_context(self, q_lora_rank, is_nsa):
        self.allow_input_scattered = (
            get_global_server_args().enable_attn_tp_input_scattered
            and (_is_cuda or _is_npu)
            and q_lora_rank is not None
            and not is_nsa
            and get_tensor_model_parallel_world_size() > 1
            and not is_dp_attention_enabled()
            and get_moe_a2a_backend().is_none()
            and not enable_moe_dense_fully_dp()
            and get_global_server_args().disable_piecewise_cuda_graph
            and get_global_server_args().speculative_algorithm != "EAGLE3"
        )
        if get_global_server_args().enable_attn_tp_input_scattered:
            if not self.allow_input_scattered:
                logging.info(
                    "attn_tp_input_scattered is not enabled while other conditions are not met"
                )
            else:
                logging.info("attn_tp_input_scattered is enabled")

    def use_input_scattered(self, forward_batch: ForwardBatch):
        return (
            self.allow_input_scattered
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend()
            and forward_batch.input_ids is not None
            and not forward_batch.can_run_tbo
        )

    @property
    def input_scattered(self):
        return self.input_scattered_

    def set_attn_inputs(self, attn_inputs: AttentionInputs):
        self.attn_inputs_ = attn_inputs

    def set_mlp_inputs(self, mlp_inputs: MLPInputs):
        self.mlp_inputs_ = mlp_inputs

    def fetch_qkv_latent(self):
        assert self.attn_inputs_ is not None
        return self.attn_inputs_.fetch_qkv_latent()
    
    def fetch_mlp_latent(self, last_layer=False):
        assert self.mlp_inputs_ is not None
        return self.mlp_inputs_.fetch_mlp_latent(last_layer=last_layer)

    def fetch_hidden_states(self):
        assert self.attn_inputs_ is not None
        return self.attn_inputs_.fetch_hidden_states()

    @contextmanager
    def maybe_input_scattered(self, forward_batch: ForwardBatch):
        flag = self.use_input_scattered(forward_batch)
        old_flag = self.input_scattered
        self.input_scattered_ = flag
        yield
        self.input_scattered_ = old_flag
        self.attn_inputs_ = None
        self.mlp_inputs_ = None


ATTN_TP_CONTEXT = AttnTpContext()


def get_attn_tp_context():
    return ATTN_TP_CONTEXT


@dataclass
class _LayerModeComputationContext:
    num_layers: int
    layer_id: int
    is_layer_sparse: bool
    is_previous_layer_sparse: Optional[bool]
    is_next_layer_sparse: Optional[bool]

    def previous_layer(self):
        assert self.is_previous_layer_sparse is not None
        return _LayerModeComputationContext(
            num_layers=self.num_layers,
            layer_id=self.layer_id - 1,
            is_layer_sparse=self.is_previous_layer_sparse,
            is_previous_layer_sparse=None,
            is_next_layer_sparse=self.is_layer_sparse,
        )


@dataclass
class LayerScatterModes:
    layer_input_mode: ScatterMode
    attn_mode: ScatterMode
    # Can be further split into e.g. mlp_input_mode and mlp_output_mode if needed
    mlp_mode: ScatterMode
    middle_residual_mode: ScatterMode
    layer_output_mode: ScatterMode

    @classmethod
    def init_new(cls, **kwargs):
        context = _LayerModeComputationContext(**kwargs)
        return cls(
            layer_input_mode=cls._compute_layer_input_mode(context),
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=cls._compute_mlp_mode(context),
            middle_residual_mode=cls._compute_middle_residual_mode(context),
            layer_output_mode=cls._compute_layer_output_mode(context),
        )

    @classmethod
    def _compute_layer_input_mode(cls, context: _LayerModeComputationContext):
        if context.layer_id == 0:
            return ScatterMode.model_input_output()
        return cls._compute_layer_output_mode(context.previous_layer())

    @classmethod
    def _compute_mlp_mode(cls, context: _LayerModeComputationContext):
        if context.is_layer_sparse:
            return (
                ScatterMode.SCATTERED
                if (
                    # Token dispatch/combine will be handled outside of LayerCommunicator for these modes.
                    not get_moe_a2a_backend().is_none()
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                )
                else ScatterMode.FULL
            )
        else:
            return (
                ScatterMode.SCATTERED
                if enable_moe_dense_fully_dp()
                else ScatterMode.FULL
            )

    @classmethod
    def _should_gather_for_tbo(cls, context: _LayerModeComputationContext):
        return (
            not context.is_layer_sparse
            and context.is_next_layer_sparse
            and enable_moe_dense_fully_dp()
            and get_global_server_args().enable_two_batch_overlap
        )

    @classmethod
    def _compute_middle_residual_mode(cls, context: _LayerModeComputationContext):
        mlp_mode = cls._compute_mlp_mode(context)
        if mlp_mode == ScatterMode.SCATTERED:
            return ScatterMode.SCATTERED
        if mlp_mode == ScatterMode.FULL:
            return ScatterMode.TP_ATTN_FULL
        raise NotImplementedError

    @classmethod
    def _compute_layer_output_mode(cls, context: _LayerModeComputationContext):
        mlp_mode = cls._compute_mlp_mode(context)
        if context.layer_id == context.num_layers - 1:
            return ScatterMode.model_input_output()
        if mlp_mode == ScatterMode.SCATTERED:
            if cls._should_gather_for_tbo(context):
                return ScatterMode.TP_ATTN_FULL
            return ScatterMode.SCATTERED
        if mlp_mode == ScatterMode.FULL:
            return ScatterMode.TP_ATTN_FULL
        raise NotImplementedError


def enable_moe_dense_fully_dp():
    return get_global_server_args().moe_dense_tp_size == 1


def _tokenweave_native_rs_norm_scattered(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_module: torch.nn.Module,
    post_residual_addition: Optional[torch.Tensor] = None,
    site: str = "?",
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Try the TokenWeave native fused RS+residual-add+RMSNorm+AG at a
    scattered-flow site, replacing _tp_reduce_scatter plus the norm that
    follows it. Engages only when the producer GEMM direct-wrote the armed
    symm region (data_ptr identity, checked inside the communicator).
    Returns None to run the incumbent composed path."""
    if residual is None or post_residual_addition is not None:
        return None
    comm = getattr(get_tp_group(), "tokenweave_comm", None)
    if comm is None or not getattr(comm, "native_scattered_enabled", False):
        return None
    weight = getattr(norm_module, "weight", None)
    eps = getattr(norm_module, "variance_epsilon", None)
    if weight is None or eps is None:
        return None
    return comm.native_fused_rs_norm_scattered(
        hidden_states, residual, weight, eps, site=site
    )


class LayerCommunicator:
    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        # Reduce scatter requires skipping all-reduce in model code after MoE/MLP, so only enable for models which have that implemented. Remove flag once done for all models that use LayerCommunicator.
        allow_reduce_scatter: bool = False,
        is_last_layer: bool = False,
        qkv_latent_func: Optional[Callable] = None,
        mlp_latent_func: Optional[Callable] = None,
    ):
        self.layer_scatter_modes = layer_scatter_modes
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.allow_reduce_scatter = allow_reduce_scatter
        self.is_last_layer = is_last_layer
        self.qkv_latent_func = qkv_latent_func
        self.mlp_latent_func = mlp_latent_func

        self._context = CommunicateContext.init_new()
        self._post_init_communicate()
        self._speculative_algo = SpeculativeAlgorithm.from_string(
            get_global_server_args().speculative_algorithm
        )

    def _post_init_communicate(self):
        self._communicate_simple_fn = CommunicateSimpleFn.get_fn(
            input_mode=self.layer_scatter_modes.layer_input_mode,
            output_mode=self.layer_scatter_modes.attn_mode,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = (
            CommunicateWithAllReduceAndLayerNormFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.attn_mode,
                residual_input_mode=self.layer_scatter_modes.layer_input_mode,
                hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,
                residual_output_mode=self.layer_scatter_modes.middle_residual_mode,
                context=self._context,
            )
        )
        self._communicate_summable_tensor_pair_fn = (
            CommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,
                residual_input_mode=self.layer_scatter_modes.middle_residual_mode,
                output_mode=self.layer_scatter_modes.layer_output_mode,
                context=self._context,
            )
        )

    def prepare_attn_and_capture_last_layer_outputs(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        hidden_states, residual = self.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            post_residual_addition=post_residual_addition,
        )
        if captured_last_layer_outputs is not None:
            gathered_last_layer_output = self._communicate_simple_fn(
                hidden_states=residual,
                forward_batch=forward_batch,
                context=self._context,
            )
            if gathered_last_layer_output is residual:
                # Clone to avoid modifying the original residual by Custom RMSNorm inplace operation
                gathered_last_layer_output = residual.clone()
            captured_last_layer_outputs.append(gathered_last_layer_output)
        return hidden_states, residual

    def prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        quant_format: str = "",
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        _tw_normed = False
        if get_attn_tp_context().input_scattered:
            if (
                residual is not None
                and hidden_states.shape[0] == residual.shape[0]
                and hidden_states.shape[0] > 0
            ):
                # A fused GatherRS MoE tail already produced the
                # scattered-reduced shard (rows match the scattered
                # residual); skip the RS. Mirrors the prepare_mlp skip for
                # the fused GemmRS o_proj.
                pass
            else:
                _tw = _tokenweave_native_rs_norm_scattered(
                    hidden_states,
                    residual,
                    self.input_layernorm,
                    post_residual_addition,
                    site="attn",
                )
                if _tw is not None:
                    # Fused kernel consumed the site (RS+add+norm+AG); the
                    # norm block below must not run again.
                    hidden_states, residual = _tw
                    _tw_normed = True
                else:
                    hidden_states, residual = self._tp_reduce_scatter(
                        hidden_states,
                        residual,
                    )
        if _tw_normed:
            pass
        elif hidden_states.shape[0] == 0:
            residual = hidden_states
        else:
            if (
                residual is not None
                and hasattr(hidden_states, "_sglang_needs_allreduce_fusion")
                and hidden_states._sglang_needs_allreduce_fusion
            ):
                if (
                    apply_aiter_all_reduce_fusion(hidden_states)
                    or apply_flashinfer_allreduce_fusion(hidden_states.shape[0])
                ) and hasattr(self.input_layernorm, "forward_with_allreduce_fusion"):
                    hidden_states, residual = (
                        self.input_layernorm.forward_with_allreduce_fusion(
                            hidden_states, residual, use_attn_tp_group=True
                        )
                    )
                else:
                    hidden_states = moe_tensor_model_parallel_all_reduce(hidden_states)
                    hidden_states, residual = self.input_layernorm(
                        hidden_states, residual
                    )
            else:
                if residual is None:
                    residual = hidden_states

                    if _use_aiter and _is_gfx95_supported and ("mxfp4" in quant_format):
                        hidden_states, *_, _ = fused_rms_mxfp4_quant(
                            hidden_states,
                            self.input_layernorm.weight,
                            self.input_layernorm.variance_epsilon,
                            None,
                            None,
                            None,
                            None,
                        )
                    elif _use_aiter and _is_gfx95_supported and ("fp8" in quant_format):

                        hidden_states, _, _, _res = fused_rms_fp8_group_quant(
                            hidden_states,
                            self.input_layernorm.weight,
                            self.input_layernorm.variance_epsilon,
                            inp2=None,
                            inp2_weight=None,
                            inp2_epsilon=None,
                            group_size=128,
                            dtype_quant=torch.float8_e4m3fn,
                            res1=None,
                            output_unquantized_inp1=False,
                        )

                    else:
                        hidden_states = self.input_layernorm(hidden_states)
                else:

                    if _use_aiter and _is_gfx95_supported and ("mxfp4" in quant_format):
                        hidden_states, *_, residual = fused_rms_mxfp4_quant(
                            hidden_states,
                            self.input_layernorm.weight,
                            self.input_layernorm.variance_epsilon,
                            None,
                            None,
                            None,
                            residual,
                        )
                    elif _use_aiter and _is_gfx95_supported and ("fp8" in quant_format):
                        # RMSNorm + FP8 per-group quant
                        # return hidden_states：
                        #   out_fp8  : FP8 activation →  a8w8 GEMM
                        #   out_bs   : block-scale →  gemm_a8w8_blockscale.x_scale
                        hidden_states, _, _, residual = fused_rms_fp8_group_quant(
                            hidden_states,
                            self.input_layernorm.weight,
                            self.input_layernorm.variance_epsilon,
                            inp2=None,
                            inp2_weight=None,
                            inp2_epsilon=None,
                            group_size=128,
                            dtype_quant=torch.float8_e4m3fn,
                            res1=residual,
                            output_unquantized_inp1=False,
                        )
                    else:
                        hidden_states, residual = self.input_layernorm(
                            hidden_states,
                            residual,
                            post_residual_addition,
                        )

        hidden_states = self._communicate_simple_fn(
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            context=self._context,
        )
        if self.qkv_latent_func is not None:
            attn_inputs = AttentionInputs(
                hidden_states, forward_batch, self.qkv_latent_func
            )
            get_attn_tp_context().set_attn_inputs(attn_inputs)
        return hidden_states, residual

    def _tp_reduce_scatter(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states
        assert (
            hidden_states.shape[0] % self._context.tp_size == 0
        ), f"Expected total tokens {hidden_states.shape[0]} % tp_size {self._context.tp_size} to be 0"
        local_tokens = hidden_states.shape[0] // self._context.tp_size
        output = hidden_states.new_empty(local_tokens, *hidden_states.shape[1:])
        get_tp_group().reduce_scatter_tensor(output, hidden_states)
        # if residual is not None:
        #     residual = residual.tensor_split(self._context.tp_size)[
        #         self._context.tp_rank
        #     ]
        return output, residual

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        cache=None,
    ):
        if cache is not None:
            self._context.cache = cache
        from sglang.srt.layers import fo_overlap

        if fo_overlap.pending():
            # FlashOverlap already all-reduced o_proj inside the GEMM; restore
            # tile order fused with the residual-add + rmsnorm.
            hidden_states, residual = fo_overlap.consume(
                hidden_states, residual, self.post_attention_layernorm
            )
            if self.mlp_latent_func is not None:
                mlp_inputs = MLPInputs(
                    hidden_states, forward_batch, self.mlp_latent_func
                )
                get_attn_tp_context().set_mlp_inputs(mlp_inputs)
            return hidden_states, residual
        if get_attn_tp_context().input_scattered and not self.is_last_layer:
            _tw_normed = False
            if (
                residual is not None
                and hidden_states.shape[0] == residual.shape[0]
                and hidden_states.shape[0] > 0
            ):
                # A fused GemmRS o_proj already produced the scattered-reduced
                # shard (rows match the scattered residual); skip the RS.
                pass
            else:
                _tw = _tokenweave_native_rs_norm_scattered(
                    hidden_states,
                    residual,
                    self.post_attention_layernorm,
                    site="mlp",
                )
                if _tw is not None:
                    hidden_states, residual = _tw
                    _tw_normed = True
                else:
                    hidden_states, residual = self._tp_reduce_scatter(
                        hidden_states,
                        residual,
                    )
            if _tw_normed:
                pass
            elif hidden_states.shape[0] == 0:
                residual = hidden_states
            else:
                hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        else:
            hidden_states, residual = self._communicate_with_all_reduce_and_layer_norm_fn(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
                layernorm=self.post_attention_layernorm,
                context=self._context,
            )
        if self.mlp_latent_func is not None:
            mlp_inputs = MLPInputs(
                hidden_states, forward_batch, self.mlp_latent_func
            )
            get_attn_tp_context().set_mlp_inputs(mlp_inputs)
        return hidden_states, residual

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return self._communicate_summable_tensor_pair_fn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            context=self._context,
            allow_reduce_scatter=self.allow_reduce_scatter,
        )

    def should_use_reduce_scatter(self, forward_batch: ForwardBatch):
        if not self.allow_reduce_scatter:
            return False
        if (
            self._communicate_summable_tensor_pair_fn
            is CommunicateSummableTensorPairFn._scatter_hidden_states
            and forward_batch.dp_padding_mode.is_max_len()
        ):
            return True
        if nsa_use_prefill_cp(forward_batch):
            return True
        if get_attn_tp_context().input_scattered and not self.is_last_layer:
            return True
        return False

    # NOTE: This function will cause torch recompilation
    def should_fuse_mlp_allreduce_with_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        if (
            is_dp_attention_enabled()
            and self._speculative_algo is not None
            and self._speculative_algo.is_eagle()
        ):
            return False

        if get_attn_tp_context().input_scattered:
            return False

        batch_size = (
            forward_batch.input_ids.shape[0]
            if hasattr(forward_batch, "input_ids")
            else 0
        )

        return (
            (
                apply_flashinfer_allreduce_fusion(batch_size)
                or (
                    _use_aiter
                    and batch_size > 0
                    and get_tensor_model_parallel_world_size() != 6
                    and get_global_server_args().enable_aiter_allreduce_fusion
                )
            )
            and (not self.is_last_layer)
            and (self._context.tp_size > 1)
        )


@dataclass
class CommunicateContext:
    process_group_sizes: Dict[ScatterMode, int]
    attn_tp_rank: int
    attn_tp_size: int
    attn_dp_size: int
    attn_cp_rank: int
    attn_cp_size: int
    tp_size: int
    cache = None
    tp_rank: int

    def is_same_group_size(self, a: ScatterMode, b: ScatterMode):
        return self.process_group_sizes[a] == self.process_group_sizes[b]

    @classmethod
    def init_new(cls):
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        attn_dp_size = get_attention_dp_size()
        attn_cp_size = get_attention_cp_size()
        attn_cp_rank = get_attention_cp_rank()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        process_group_sizes = {
            ScatterMode.SCATTERED: 1,
            ScatterMode.TP_ATTN_FULL: attn_tp_size,
            # TODO: support --moe-dense-tp-size > 1
            # With context parallel enabled, we should exclude
            # the attn_cp_size from the total tp_size
            ScatterMode.FULL: tp_size // attn_cp_size,
        }
        return cls(
            process_group_sizes=process_group_sizes,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=attn_tp_size,
            attn_dp_size=attn_dp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )


class CommunicateSimpleFn:
    @staticmethod
    def get_fn(
        input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(input_mode, output_mode):
            return CommunicateSimpleFn._trivial

        if (input_mode == ScatterMode.SCATTERED) and (
            output_mode == ScatterMode.TP_ATTN_FULL
        ):
            if _use_ag_after_qlora:
                return CommunicateSimpleFn._trivial
            return CommunicateSimpleFn._scattered_to_tp_attn_full

        raise NotImplementedError(f"{input_mode=} {output_mode=}")

    @staticmethod
    def _trivial(
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ) -> torch.Tensor:
        return hidden_states

    @staticmethod
    def _scattered_to_tp_attn_full(
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(hidden_states, tuple):
            gathered_hidden_states = []
            for local_hidden_states in hidden_states:
                with use_symmetric_memory(
                    get_tp_group(),
                    disabled=not is_allocation_symmetric(),
                ):
                    output = torch.empty(
                        (
                            local_hidden_states.shape[0] * context.attn_tp_size,
                            *local_hidden_states.shape[1:],
                        ),
                        dtype=local_hidden_states.dtype,
                        device=local_hidden_states.device,
                    )
                attn_tp_all_gather_into_tensor(
                    output,
                    local_hidden_states,
                )
                gathered_hidden_states.append(output)
            return tuple(gathered_hidden_states)

        hidden_states, local_hidden_states = (
            get_local_dp_buffer(),
            hidden_states,
        )
        attn_tp_all_gather_into_tensor(
            hidden_states,
            local_hidden_states,
        )
        return hidden_states


class CommunicateWithAllReduceAndLayerNormFn:
    """Besides communication, needs to
    1. All reduce in tp_attn_group on hidden_states
    2. Apply layer norm
    """

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        hidden_states_output_mode: ScatterMode,
        residual_output_mode: ScatterMode,
        context: CommunicateContext,
    ):

        if (
            context.is_same_group_size(
                hidden_states_input_mode, hidden_states_output_mode
            )
            and context.is_same_group_size(residual_input_mode, residual_output_mode)
            and context.attn_tp_size == 1
        ):
            return CommunicateWithAllReduceAndLayerNormFn._simple

        if (
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (
                residual_input_mode in [ScatterMode.SCATTERED, ScatterMode.TP_ATTN_FULL]
            )
            and (hidden_states_output_mode == ScatterMode.FULL)
            and (residual_output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return partial(
                CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,
            )

        if (
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (
                residual_input_mode in [ScatterMode.SCATTERED, ScatterMode.TP_ATTN_FULL]
            )
            and (hidden_states_output_mode == ScatterMode.SCATTERED)
            and (residual_output_mode == ScatterMode.SCATTERED)
        ):
            return partial(
                CommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,
            )

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {hidden_states_output_mode=} {residual_output_mode=}"
        )

    @staticmethod
    def _simple(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
    ):
        # TODO move these `if shape != 0` into LayerNorm itself
        if hidden_states.shape[0] != 0:
            hidden_states, residual = layernorm(hidden_states, residual)
        return hidden_states, residual

    @staticmethod
    def _gather_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
    ):
        if get_attn_tp_context().input_scattered:
            return CommunicateWithAllReduceAndLayerNormFn._tp_all_reduce_with_scattered_residual(
                hidden_states,
                residual,
                layernorm,
                context,
            )

        if residual_input_mode == ScatterMode.SCATTERED and context.attn_tp_size > 1:
            residual, local_residual = (
                get_local_dp_buffer(),
                residual,
            )
            attn_tp_all_gather_into_tensor(residual, local_residual)
        if context.attn_dp_size != 1:
            # Perform layernorm on smaller data before comm. Only valid when attn_tp_size is 1 (tp_size == dp_size)
            use_layer_norm_before_gather = context.attn_tp_size == 1
            if use_layer_norm_before_gather and hidden_states.shape[0] != 0:
                with use_symmetric_memory(
                    get_tp_group(),
                    disabled=not is_allocation_symmetric(),
                ):
                    hidden_states, residual = layernorm(hidden_states, residual)
            elif context.attn_tp_rank == 0:
                hidden_states += residual

            hidden_states, local_hidden_states = (
                get_global_dp_buffer(),
                hidden_states,
            )
            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)

            if not use_layer_norm_before_gather:
                dp_scatter(residual, hidden_states, forward_batch)
                if hidden_states.shape[0] != 0:
                    hidden_states = layernorm(hidden_states)
        else:
            handled = False
            if (
                apply_aiter_all_reduce_fusion(hidden_states)
                or apply_flashinfer_allreduce_fusion(hidden_states.shape[0])
                # TokenWeave (ladder4 stage A): route into the layernorm's
                # fused path so the generic dispatcher gets a shot; on a miss
                # the layernorm fork reproduces the not-handled branch below.
                or _tokenweave_fusion_enabled()
            ) and hasattr(layernorm, "forward_with_allreduce_fusion"):
                hidden_states, residual = layernorm.forward_with_allreduce_fusion(
                    hidden_states, residual, use_attn_tp_group=True
                )
                handled = True

            if not handled:
                if not forward_batch.useFlux:
                    hidden_states = attention_tensor_model_parallel_all_reduce(
                        hidden_states
                    )
                if _is_npu and context.cache is not None:
                    _ = prepare_weight_cache(hidden_states, context.cache)
                hidden_states, residual = layernorm(hidden_states, residual)
        return hidden_states, residual

    @staticmethod
    def _scatter_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
    ):
        input_hidden_states = hidden_states
        hidden_states = hidden_states.tensor_split(context.attn_tp_size)[
            context.attn_tp_rank
        ]
        attn_tp_reduce_scatter_tensor(hidden_states, input_hidden_states)
        if residual_input_mode == ScatterMode.TP_ATTN_FULL:
            residual = residual.tensor_split(context.attn_tp_size)[context.attn_tp_rank]
        if hidden_states.shape[0] != 0:
            hidden_states, residual = layernorm(hidden_states, residual)
        return hidden_states, residual

    @staticmethod
    def _tp_all_reduce_with_scattered_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states

        scattered_states = hidden_states.tensor_split(context.tp_size)[context.tp_rank]
        scattered_states += residual
        residual = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = layernorm(residual)
        return hidden_states, residual


class CommunicateSummableTensorPairFn:
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""

    @classmethod
    def execute(
        cls,
        hidden_states_input_mode,
        residual_input_mode,
        output_mode,
        context,
        **kwargs,
    ):
        return cls.get_fn(
            hidden_states_input_mode=hidden_states_input_mode,
            residual_input_mode=residual_input_mode,
            output_mode=output_mode,
            context=context,
        )(context=context, **kwargs)

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(
            hidden_states_input_mode, output_mode
        ) and context.is_same_group_size(residual_input_mode, output_mode):
            return CommunicateSummableTensorPairFn._trivial

        if (
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.TP_ATTN_FULL)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return CommunicateSummableTensorPairFn._scatter_hidden_states

        if (
            (hidden_states_input_mode == ScatterMode.SCATTERED)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return CommunicateSummableTensorPairFn._gather

        if (
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (residual_input_mode == ScatterMode.TP_ATTN_FULL)
            and (output_mode == ScatterMode.SCATTERED)
        ):
            return CommunicateSummableTensorPairFn._scatter

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

    @staticmethod
    def _trivial(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        return hidden_states, residual

    @staticmethod
    def _scatter_hidden_states(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        allow_reduce_scatter: bool = False,
    ):
        hidden_states, global_hidden_states = (
            get_local_dp_buffer(),
            hidden_states,
        )
        if allow_reduce_scatter and forward_batch.dp_padding_mode.is_max_len():
            # When using padding, all_reduce is skipped after MLP and MOE and reduce scatter is used here instead.
            dp_reduce_scatter_tensor(hidden_states, global_hidden_states)
        else:
            dp_scatter(hidden_states, global_hidden_states, forward_batch)
        return hidden_states, residual

    @staticmethod
    def _gather(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        hidden_states += residual
        residual = None
        hidden_states, local_hidden_states = (
            get_local_dp_buffer(),
            hidden_states,
        )
        attn_tp_all_gather_into_tensor(
            hidden_states,
            local_hidden_states,
        )
        return hidden_states, residual

    @staticmethod
    def _scatter(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ):
        assert residual is None, "not yet handled residual!=None"
        tensor_list = list(hidden_states.tensor_split(context.attn_tp_size))
        hidden_states = tensor_list[context.attn_tp_rank]
        return hidden_states, residual
