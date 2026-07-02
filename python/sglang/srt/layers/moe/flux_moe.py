"""Flux fused AllGather + grouped GEMM1 for the MoE FFN (M3 slice 1).

Home: the PK scattered residual flow (input_scattered), TP-only (ep_size==1),
bf16/fp16 unquantized expert weights, non-last layer. GEMM1 (gate/up) runs as
flux AGScatter consuming the PRE-GATHER scattered shard; the token all-gather
happens inside the fused grouped GEMM. Activation, per-expert GEMM2 and the
token unsort/weighted-sum stay on the sglang side (unfused; GatherRS replaces
them in slice 2). The returned tensor keeps the exact contract of
``FusedMoE``'s output on this path: (ntokens_global, hidden) TP-partial sums
(w2 is sharded on the contraction dim) that the existing downstream reduction
(EP-AR skip + communicator reduce-scatter) consumes unchanged.

Slice-1 accepted costs (documented, addressed later):
- The router gate still runs on the gathered tokens from ``fetch_mlp_latent``,
  so that all-gather still happens. The big win here is fusing the
  GEMM1-feeding gather; removing the gate-AG (all-gather only topk metadata)
  is a later micro-opt.
- GEMM2 is a python loop of per-expert ``torch.mm`` over the expert-sorted
  segments, including one GPU->CPU sync for the segment sizes.

NOTE (first GPU validation will resolve): flux's MoeArguments/DistEnvTPWithEP
ffn sharding semantics cannot be verified without a GPU run. Both
interpretations are implemented behind ``SGLANG_FLUX_MOE_FFN_MODE``:
- "shard" (default): MoeArguments.ffn_hidden = the per-rank per-direction
  shard size (w13_weight.shape[1] // 2), i.e. ffn_tp_size==1 semantics.
- "full": MoeArguments.ffn_hidden = shard * tp_size (the full intermediate
  size as physically materialized, robust to FusedMoE padding); flux then
  derives the shard through its env's ffn-TP division.

Other defensive levers / assumptions:
- ``SGLANG_FLUX_MOE_CONTIG_W=1`` copies the gate/up weight groups to
  contiguous tensors each forward (slow; correctness-validation lever in case
  flux rejects or mishandles the non-contiguous dim-1 slices of w13_weight).
- Op construction is collective across the TP group; the gate conditions are
  rank-uniform (shapes/env/batch geometry), so all ranks reach the lazy ctor
  on the same forward.
- Any exception in the flux path disables it on that block instance (warning
  logged once per instance) and falls back to the normal experts path.
"""

import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

logger = logging.getLogger(__name__)

_FFN_MODE = os.environ.get("SGLANG_FLUX_MOE_FFN_MODE", "shard")
_CONTIG_W = os.environ.get("SGLANG_FLUX_MOE_CONTIG_W", "0") == "1"
# Mirrors MAX_M used by the attention-side AGCook/GemmRS glue in unquant.py.
_MAX_NTOKENS = 65536

_ENGAGED_LOGGED = False


def _make_dist_env(flux_mod, tp_group):
    """Build DistEnvTPWithEP for ep_size==1, trying the plausible signatures."""
    attempts = (
        ("ep_group=None", lambda: flux_mod.DistEnvTPWithEP(tp_group, 1, None)),
        ("no ep_group arg", lambda: flux_mod.DistEnvTPWithEP(tp_group, 1)),
        ("ep_group=tp_group", lambda: flux_mod.DistEnvTPWithEP(tp_group, 1, tp_group)),
    )
    errors = []
    for desc, ctor in attempts:
        try:
            return ctor(), desc
        except Exception as exc:  # noqa: BLE001 -- try every known signature
            errors.append(f"{desc}: {exc}")
    raise RuntimeError("DistEnvTPWithEP construction failed: " + "; ".join(errors))


def _make_moe_args(flux_mod, max_ntokens, hidden, ffn_hidden, nexperts, topk, dtype):
    try:
        return (
            flux_mod.MoeArguments(
                max_ntokens=max_ntokens,
                hidden=hidden,
                ffn_hidden=ffn_hidden,
                nexperts=nexperts,
                topk=topk,
                input_dtype=dtype,
                output_dtype=dtype,
            ),
            "kwargs",
        )
    except TypeError:
        return (
            flux_mod.MoeArguments(
                max_ntokens, hidden, ffn_hidden, nexperts, topk, dtype, dtype
            ),
            "positional",
        )


def _scatter_metadata(
    topk_ids: torch.Tensor, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expert-sort metadata in flux's format.

    Reuses sglang's existing MoE preprocess (stable sort by expert id).
    ``deepep_run_moe_deep_preprocess`` is the current in-tree equivalent of
    the old ``run_moe_ep_preproess``; the two coincide because standard TopK
    on this path emits only valid expert ids (no -1 entries to drop):
    - src2dst reshaped (ntokens, topk), int32  == flux scatter_index
    - diff(seg_indptr), int32                  == flux splits (GLOBAL counts)
    """
    from sglang.srt.layers.moe.ep_moe.kernels import deepep_run_moe_deep_preprocess

    _, src2dst, seg_indptr = deepep_run_moe_deep_preprocess(topk_ids, num_experts)
    splits_gpu = torch.diff(seg_indptr).to(torch.int32)
    scatter_index = src2dst.view(topk_ids.shape).to(torch.int32)
    return splits_gpu, scatter_index


class FluxMoeAGScatter:
    """Wraps a flux GemmGroupedV2/V3 AGScatter op for one MoE layer."""

    def __init__(
        self,
        tp_group,
        tp_size: int,
        num_experts: int,
        topk: int,
        hidden: int,
        inter_shard: int,
        dtype: torch.dtype,
        ffn_mode: str,
    ):
        import flux  # lazy: only reached when the flux path is gated on

        if ffn_mode not in ("shard", "full"):
            raise RuntimeError(f"invalid SGLANG_FLUX_MOE_FFN_MODE={ffn_mode!r}")
        ffn_hidden = inter_shard if ffn_mode == "shard" else inter_shard * tp_size

        env, env_desc = _make_dist_env(flux, tp_group)
        moe_args, args_desc = _make_moe_args(
            flux, _MAX_NTOKENS, hidden, ffn_hidden, num_experts, topk, dtype
        )
        if torch.cuda.get_device_capability()[0] >= 9:
            op_cls_name = "GemmGroupedV3AGScatter"
        else:
            op_cls_name = "GemmGroupedV2AGScatterOp"
        op_cls = getattr(flux, op_cls_name, None)
        if op_cls is None:
            raise RuntimeError(f"flux lacks {op_cls_name}")
        try:
            self.op = op_cls(env, moe_args)
        except TypeError:
            self.op = op_cls(tp_env=env, moe_args=moe_args)

        self.key = (num_experts, topk, hidden, inter_shard, dtype, ffn_mode)
        self.ctor_desc = (
            f"op={op_cls_name} env=({env_desc}) args=({args_desc}) "
            f"ffn_mode={ffn_mode} ffn_hidden={ffn_hidden}"
        )

    def _run_gemm1(
        self,
        inputs_shard: torch.Tensor,
        weight_groups: List[torch.Tensor],
        splits_gpu: torch.Tensor,
        scatter_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fmw = getattr(self.op, "forward_multiple_weights", None)
        if fmw is None:
            raise RuntimeError("flux AGScatter op lacks forward_multiple_weights")
        try:
            outs = fmw(
                inputs_shard=inputs_shard,
                weights=weight_groups,
                splits_gpu=splits_gpu,
                scatter_index=scatter_index,
                output_scale=None,
                outputs_buf=None,
                fast_accum=False,
            )
        except TypeError:
            outs = fmw(inputs_shard, weight_groups, splits_gpu, scatter_index)
        if isinstance(outs, torch.Tensor) or len(outs) != 2:
            raise RuntimeError(
                f"unexpected forward_multiple_weights output type/arity: {type(outs)}"
            )
        return outs[0], outs[1]

    def forward(
        self,
        hidden_shard: torch.Tensor,
        topk_ids_global: torch.Tensor,
        topk_weights_global: torch.Tensor,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> torch.Tensor:
        """AGScatter GEMM1 -> silu*mul -> per-expert GEMM2 -> unsort+combine.

        Returns (ntokens_global, hidden) TP-partial sums, token order matching
        the gathered order used for routing (flux's internal all-gather and
        sglang's fetch AG both concatenate rank shards in rank order).
        """
        ntokens, topk = topk_ids_global.shape
        num_experts = w13_weight.shape[0]
        inter_shard = w13_weight.shape[1] // 2

        splits_gpu, scatter_index = _scatter_metadata(topk_ids_global, num_experts)

        # w13 stacks [gate; up] along dim 1; dim-1 slices are non-contiguous
        # views (per-expert 2D blocks are contiguous). See module NOTE.
        gate_w = w13_weight[:, :inter_shard, :]
        up_w = w13_weight[:, inter_shard:, :]
        if _CONTIG_W:
            gate_w = gate_w.contiguous()
            up_w = up_w.contiguous()

        gate_out, up_out = self._run_gemm1(
            hidden_shard, [gate_w, up_w], splits_gpu, scatter_index
        )
        total_rows = scatter_index.numel()
        if gate_out.shape[0] != total_rows:
            if gate_out.shape[0] > total_rows:
                # flux may hand back max-ntokens-sized buffers.
                gate_out = gate_out[:total_rows]
                up_out = up_out[:total_rows]
            else:
                raise RuntimeError(
                    f"GEMM1 rows {gate_out.shape[0]} < expected {total_rows}"
                )

        # sglang's SiluAndMul expects a concatenated tensor; flux returns the
        # groups separately, so apply silu*mul directly.
        act = F.silu(gate_out) * up_out

        # GEMM2: plain matmul per expert segment of the expert-sorted rows
        # (seg boundaries = splits cumsum). Unfused on purpose in slice 1;
        # GatherRS takes over in slice 2. The .tolist() is a GPU->CPU sync.
        down = torch.empty(
            (total_rows, w2_weight.shape[1]),
            dtype=act.dtype,
            device=act.device,
        )
        start = 0
        for expert_id, count in enumerate(splits_gpu.tolist()):
            if count == 0:
                continue
            end = start + count
            torch.mm(act[start:end], w2_weight[expert_id].t(), out=down[start:end])
            start = end

        # Unsort back to token order and apply the routing weights: for token
        # t, out[t] = sum_k w[t,k] * down[scatter_index[t,k]]. bf16 accumulate
        # over topk matches the existing post_reorder kernel's numerics.
        gathered = down[scatter_index.view(-1).long()].view(ntokens, topk, -1)
        weights = topk_weights_global.to(gathered.dtype).reshape(ntokens, topk, 1)
        return (gathered * weights).sum(dim=1)


def _get_or_build_op(moe_block, w13_weight, topk, hidden):
    inter_shard = w13_weight.shape[1] // 2
    key = (
        w13_weight.shape[0],
        topk,
        hidden,
        inter_shard,
        w13_weight.dtype,
        _FFN_MODE,
    )
    op = getattr(moe_block, "_flux_moe_op", None)
    if op is not None and op.key == key:
        return op
    op = FluxMoeAGScatter(
        tp_group=get_tp_group().device_group,
        tp_size=get_tensor_model_parallel_world_size(),
        num_experts=w13_weight.shape[0],
        topk=topk,
        hidden=hidden,
        inter_shard=inter_shard,
        dtype=w13_weight.dtype,
        ffn_mode=_FFN_MODE,
    )
    moe_block._flux_moe_op = op
    return op


def try_flux_moe_forward(
    moe_block, hidden_states: torch.Tensor, topk_output
) -> Optional[torch.Tensor]:
    """Run the MoE FFN through flux AGScatter; None means "use the normal path".

    The caller (Qwen3MoeSparseMoeBlock.forward_normal) already checked
    SGLANG_FLUX_MOE=1 and not-last-layer. ``hidden_states`` is the gathered
    (ntokens_global, hidden) tensor the gate ran on; the flux GEMM1 instead
    consumes the pre-gather scattered shard. Never raises: hard failures
    disable the path on this block instance and fall back.
    """
    if getattr(moe_block, "_flux_moe_disabled", False):
        return None

    from sglang.srt.layers.communicator import get_attn_tp_context

    ctx = get_attn_tp_context()
    topk_ids = getattr(topk_output, "topk_ids", None)
    topk_weights = getattr(topk_output, "topk_weights", None)
    w13_weight = getattr(moe_block.experts, "w13_weight", None)
    w2_weight = getattr(moe_block.experts, "w2_weight", None)
    if (
        not ctx.input_scattered
        or ctx.mlp_inputs_ is None
        or moe_block.ep_size != 1
        or topk_ids is None
        or topk_weights is None
        or w13_weight is None
        or w2_weight is None
    ):
        return None

    local = ctx.mlp_inputs_.hidden_states_local
    tp_size = get_tensor_model_parallel_world_size()
    hidden = hidden_states.shape[-1]
    if (
        tp_size <= 1
        or local is None
        or local.dim() != 2
        or local.shape[0] == 0
        # The scattered shard must be exactly this rank's slice of the
        # gathered rows (also excludes the last layer, which keeps full rows).
        or local.shape[0] * tp_size != hidden_states.shape[0]
        or hidden_states.shape[0] > _MAX_NTOKENS
        or hidden_states.dtype not in (torch.bfloat16, torch.float16)
        or local.dtype != hidden_states.dtype
        # bf16/fp16 unquantized expert weights only (excludes quantized MoE).
        or w13_weight.dtype != hidden_states.dtype
        or w13_weight.dim() != 3
        or w2_weight.dim() != 3
        or w13_weight.shape[1] % 2 != 0
        or w13_weight.shape[2] != hidden
        or w2_weight.shape[1] != hidden
        or w2_weight.shape[2] != w13_weight.shape[1] // 2
        or w2_weight.shape[0] != w13_weight.shape[0]
        or topk_ids.dim() != 2
        or topk_ids.shape[0] != hidden_states.shape[0]
    ):
        return None

    try:
        op = _get_or_build_op(moe_block, w13_weight, topk_ids.shape[1], hidden)
        out = op.forward(local, topk_ids, topk_weights, w13_weight, w2_weight)
    except Exception as exc:  # noqa: BLE001 -- defensive: flux API glue is
        # unvalidated until the first GPU round; never take the server down.
        moe_block._flux_moe_disabled = True
        logger.warning(
            "[FLUX-MOE] disabled: layer=%s reason=%s",
            getattr(moe_block, "layer_id", "?"),
            exc,
            exc_info=True,
        )
        return None

    global _ENGAGED_LOGGED
    if not _ENGAGED_LOGGED:
        _ENGAGED_LOGGED = True
        logger.info(
            "[FLUX-MOE] engaged: layer=%s tokens=%d experts=%d topk=%d %s",
            getattr(moe_block, "layer_id", "?"),
            hidden_states.shape[0],
            w13_weight.shape[0],
            topk_ids.shape[1],
            op.ctor_desc,
        )
    return out
