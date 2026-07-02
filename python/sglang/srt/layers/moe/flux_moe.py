"""Flux fused MoE FFN under the PK scattered flow (M3 slices 1+2).

Home: the PK scattered residual flow (input_scattered), TP-only (ep_size==1),
bf16/fp16 unquantized expert weights, non-last layer.

Slice 1 (GEMM1): gate/up run as flux AGScatter consuming the PRE-GATHER
scattered shard; the token all-gather happens inside the fused grouped GEMM.

Slice 2 (GEMM2 tail, SGLANG_FLUX_MOE_TAIL=gather_rs): GEMM2 + topk gather +
routing-weight scaling + weighted reduction + cross-rank ReduceScatter run
fused inside flux GatherRS, returning the (ntokens/tp, hidden) scattered
REDUCED rows directly (the downstream RS is skipped via a shape check in
prepare_attn). Default tail stays "torch" (slice-1 behavior): silu*mul +
per-expert ``torch.mm`` loop + unsort/weighted-sum, returning
(ntokens_global, hidden) TP-partial sums for the existing downstream
reduction (EP-AR skip + communicator reduce-scatter), exactly like
``FusedMoE``'s output on this path.

gather_rs runs an UNCONDITIONAL cross-rank GroupBarrier + nvshmem ring
reduce-scatter inside every forward, so it deadlocks on the tiny batches the
serving warmup drives (2 tokens over 128 experts -> most experts empty; a
regime flux's own tests never cover). It therefore engages only when the
GLOBAL token count >= SGLANG_FLUX_MOE_GATHER_RS_MIN_TOKENS (default 256),
falling back to the torch tail below that. The threshold is compared against
the gathered global count, identical on all ranks, so the choice is
rank-uniform and never desyncs the collective.

Known remaining costs (documented, addressed later):
- The router gate still runs on the gathered tokens from ``fetch_mlp_latent``,
  so that all-gather still happens. The big win here is fusing the
  GEMM1-feeding gather; removing the gate-AG (all-gather only topk metadata)
  is a later micro-opt.
- The torch tail's GEMM2 is a python loop of per-expert ``torch.mm`` over the
  expert-sorted segments, including one GPU->CPU sync for the segment sizes
  (both gone under gather_rs).

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
- Layered fallback: a gather_rs tail failure disables only that tail
  (globally, "[FLUX-MOE] gather_rs disabled") and drops back to the torch
  tail with AGScatter still engaged; any other exception in the flux path
  disables flux on that block instance ("[FLUX-MOE] disabled") and falls
  back to the normal experts path.
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
_MAX_NTOKENS = int(os.environ.get("SGLANG_FLUX_MOE_MAX_NTOKENS", "8192"))
# M3 slice 2: tail implementation selector.
# - "torch" (default): the slice-1 tail (silu*mul + python per-expert GEMM2
#   loop + unsort/combine); output = (ntokens, hidden) TP-partial sums for the
#   downstream communicator RS.
# - "gather_rs": flux GemmGroupedV2/V3 GatherRS, fusing grouped GEMM2 + topk
#   gather + fp32 per-row scaling + weighted reduce + cross-rank
#   ReduceScatter; output = (ntokens/tp, hidden) scattered REDUCED rows (the
#   next layer's prepare_attn detects the row match and skips its RS).
_TAIL_MODE = os.environ.get("SGLANG_FLUX_MOE_TAIL", "torch")
# Minimum GLOBAL token count to engage the gather_rs tail; below it, fall back
# to the (correct, slower) torch tail. flux GatherRS runs an UNCONDITIONAL
# cross-rank GroupBarrier + nvshmem ring reduce-scatter inside every
# forward_gather_rs, and flux's own tests only ever exercise huge M
# (ntokens=32768) over few experts with an even round-robin split -- tiny M
# and zero-count experts (our 128-expert model on the 2-token serving warmup:
# ~16 sorted rows over 128 experts, ~112 empty) are untested and deadlock the
# collective. The threshold MUST be compared against the global token count
# (identical on all ranks) so the torch/gather_rs choice is rank-uniform;
# deciding from a per-rank local count would desync the collective and hang.
# Default 256: comfortably above serving warmup batches, and slice-2's win is
# on large prefill chunks anyway. Set to 0 to force gather_rs always (repro).
_GATHER_RS_MIN_TOKENS = int(
    os.environ.get("SGLANG_FLUX_MOE_GATHER_RS_MIN_TOKENS", "256")
)

_ENGAGED_LOGGED = False
_GATHER_RS_ENGAGED_LOGGED = False
_GATHER_RS_SKIP_SMALL_LOGGED = False
# Global, not per-layer: the flux ops are cached per shape and shared by all
# layers, so a gather_rs failure is structural (API/workspace), not per-layer.
# Disabling falls back to the torch tail only; AGScatter GEMM1 stays engaged
# (layered fallback with a distinct log reason).
_GATHER_RS_DISABLED = False


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

        # Kept for the lazily-built companion GatherRS tail op (slice 2),
        # which must live on the same group.
        self.tp_group = tp_group
        self.tp_size = tp_size
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
        return flux_moe_compute(
            self,
            hidden_shard,
            topk_ids_global,
            topk_weights_global,
            w13_weight,
            w2_weight,
        )


def _moe_tail(
    gate_out: torch.Tensor,
    up_out: torch.Tensor,
    splits_gpu: torch.Tensor,
    scatter_index: torch.Tensor,
    topk_weights_global: torch.Tensor,
    w2_weight: torch.Tensor,
) -> torch.Tensor:
    """Post-GEMM1 tail: silu*mul -> per-expert GEMM2 -> unsort + weighted sum.

    Pure function of tensors so the numeric probe can feed it reference GEMM1
    outputs; the serving path goes through here too (via flux_moe_compute).
    """
    ntokens, topk = scatter_index.shape
    total_rows = scatter_index.numel()
    assert gate_out.shape[0] == total_rows and up_out.shape[0] == total_rows

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


class FluxMoeGatherRS:
    """Wraps flux GemmGroupedV2/V3 GatherRS for the MoE GEMM2 tail (slice 2).

    Fuses grouped GEMM2 + topk gather + fp32 per-row scaling + weighted
    reduction + cross-rank ReduceScatter. Signatures verified against the
    deployed flux checkout (src/pybind/gemm_grouped_v2_gather_rs.cc and
    gemm_grouped_v3_gather_rs.cc): V2 takes the tp_group and output_dtype (no
    rank/world args) and accepts GPU splits; V3 takes rank/world and requires
    CPU splits (one D2H sync per forward).
    """

    def __init__(
        self,
        tp_group,
        tp_size: int,
        num_experts: int,
        topk: int,
        n_dim: int,
        max_m: int,
        dtype: torch.dtype,
    ):
        import flux  # lazy: only reached when the flux path is gated on

        # topk is a template parameter of the v2 gather kernel; instantiated
        # set verified in topk_gather_rs_v2.cu on the deployed checkout.
        if topk not in (1, 2, 3, 4, 5, 6, 8, 10):
            raise RuntimeError(
                f"gather_rs kernel not instantiated for topk={topk} "
                f"(supported: 1-6, 8, 10)"
            )
        self._v3 = torch.cuda.get_device_capability()[0] >= 9
        # The v2 reduce kernel tiles N at 1024 (kTileSizeN); flux auto-heals
        # n_split to n_dim/1024, which only works when n_dim divides evenly.
        if not self._v3 and n_dim % 1024 != 0:
            raise RuntimeError(
                f"gather_rs (v2) requires hidden % 1024 == 0, got {n_dim}"
            )
        if self._v3:
            op_cls = getattr(flux, "GemmGroupedV3GatherRS", None)
            if op_cls is None:
                raise RuntimeError("flux lacks GemmGroupedV3GatherRS")
            rank = torch.distributed.get_rank(tp_group)
            self.op = op_cls(
                total_num_experts=num_experts,
                max_m=max_m,
                n_dim=n_dim,
                topk=topk,
                rank=rank,
                world_size=tp_size,
                tp_world_size=tp_size,  # ffn TP == full TP (ep_size == 1)
                ep_world_size=1,
                max_input_groups=1,
            )
            op_cls_name = "GemmGroupedV3GatherRS"
        else:
            op_cls = getattr(flux, "GemmGroupedV2GatherRSOp", None)
            if op_cls is None:
                raise RuntimeError("flux lacks GemmGroupedV2GatherRSOp")
            self.op = op_cls(
                tp_group=tp_group,
                total_num_experts=num_experts,
                max_m=max_m,
                n_dim=n_dim,
                topk=topk,
                output_dtype=dtype,
                tp_world_size=tp_size,  # ffn TP == full TP (ep_size == 1)
                ep_world_size=1,
                max_input_groups=1,
            )
            op_cls_name = "GemmGroupedV2GatherRSOp"

        self.tp_size = tp_size
        self.max_m = max_m
        self.ctor_desc = (
            f"op={op_cls_name} max_m={max_m} n_dim={n_dim} topk={topk} "
            f"tp={tp_size} dtype={dtype}"
        )

    def run(
        self,
        act_sorted: torch.Tensor,
        w2_weight: torch.Tensor,
        splits_gpu: torch.Tensor,
        routing_idx: torch.Tensor,
        output_vec_scale: torch.Tensor,
    ) -> torch.Tensor:
        if self._v3:
            # V3 asserts splits on CPU int32 (one D2H sync per forward).
            return self.op.forward_gather_rs(
                input=act_sorted,
                weight=w2_weight,
                splits_cpu=splits_gpu.to("cpu", torch.int32),
                routing_idx=routing_idx,
                output_vec_scale=output_vec_scale,
            )
        # V2 accepts GPU splits (copies internally when given CPU): no sync.
        return self.op.forward_gather_rs(
            input=act_sorted,
            weight=w2_weight,
            splits=splits_gpu,
            scatter_idx=routing_idx,
            output_vec_scale=output_vec_scale,
        )


def _get_or_build_gather_rs(op: "FluxMoeAGScatter", w2_weight, topk):
    # Cached on the (shape-deduped, globally shared) AGScatter op so there is
    # exactly one GatherRS workspace per shape -- same nvshmem-heap lesson as
    # _OP_CACHE.
    grs = getattr(op, "_gather_rs", None)
    if grs is not None:
        return grs
    grs = FluxMoeGatherRS(
        tp_group=op.tp_group,
        tp_size=op.tp_size,
        num_experts=w2_weight.shape[0],
        topk=topk,
        n_dim=w2_weight.shape[1],
        max_m=_MAX_NTOKENS * topk,
        dtype=w2_weight.dtype,
    )
    op._gather_rs = grs
    logger.info("[FLUX-MOE] gather_rs op constructed: %s", grs.ctor_desc)
    return grs


def _gather_rs_tail(
    op: "FluxMoeAGScatter",
    gate_out: torch.Tensor,
    up_out: torch.Tensor,
    splits_gpu: torch.Tensor,
    scatter_index: torch.Tensor,
    topk_weights_global: torch.Tensor,
    w2_weight: torch.Tensor,
) -> torch.Tensor:
    """Slice-2 tail: silu*mul stays torch; GEMM2 + topk gather + weighted
    reduce + cross-rank ReduceScatter run fused inside flux GatherRS.

    CONTRACT DIFFERENCE vs _moe_tail: returns (ntokens/tp, hidden) token-
    SHARDED, FULLY REDUCED rows -- flux reduce-scatters with the same
    rank-block layout as sglang's _tp_reduce_scatter, so this equals what the
    next layer's prepare_attn RS would have produced (and that RS is skipped
    via its shape check).
    """
    ntokens, topk = scatter_index.shape
    total_rows = scatter_index.numel()
    assert gate_out.shape[0] == total_rows and up_out.shape[0] == total_rows
    grs = _get_or_build_gather_rs(op, w2_weight, topk)
    if ntokens % grs.tp_size != 0:
        raise RuntimeError(f"gather_rs needs ntokens % tp == 0, got {ntokens}")
    if total_rows > grs.max_m:
        raise RuntimeError(f"gather_rs rows {total_rows} > max_m {grs.max_m}")

    act = F.silu(gate_out) * up_out

    # Routing weights as a per-SORTED-row fp32 vector; flux applies them in
    # fp32 during the gather (one rounding fewer than the torch tail's bf16
    # combine). scatter_index maps flat (t,k) -> sorted row.
    flat_s2d = scatter_index.view(-1)
    output_vec_scale = torch.empty(
        total_rows, dtype=torch.float32, device=act.device
    )
    output_vec_scale[flat_s2d.long()] = topk_weights_global.reshape(-1).to(
        torch.float32
    )

    out = grs.run(act, w2_weight, splits_gpu, flat_s2d, output_vec_scale)
    expected_rows = ntokens // grs.tp_size
    if out.shape[0] != expected_rows or out.shape[1] != w2_weight.shape[1]:
        raise RuntimeError(
            f"gather_rs output {tuple(out.shape)} != "
            f"({expected_rows}, {w2_weight.shape[1]})"
        )
    return out


def flux_moe_compute(
    op: "FluxMoeAGScatter",
    hidden_shard: torch.Tensor,
    topk_ids_global: torch.Tensor,
    topk_weights_global: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    capture: Optional[dict] = None,
) -> torch.Tensor:
    """AGScatter GEMM1 -> silu*mul -> GEMM2 tail (torch or flux GatherRS).

    Single compute path shared by serving (FluxMoeAGScatter.forward via
    try_flux_moe_forward) and the numeric probe
    (scripts/probe_flux_moe_numerics.py). Token order matches the gathered
    order used for routing (flux's internal all-gather and sglang's fetch AG
    both concatenate rank shards in rank order).

    Return contract depends on the tail (SGLANG_FLUX_MOE_TAIL):
    - torch tail: (ntokens_global, hidden) TP-partial sums, reduced by the
      downstream communicator RS as before.
    - gather_rs tail: (ntokens_global/tp, hidden) scattered REDUCED rows;
      forward_normal returns them as-is and the next layer's prepare_attn
      skips its RS on the row match. gather_rs failures disable that tail
      globally and fall back to the torch tail (AGScatter stays engaged).

    ``capture``, when given, receives the expert-sorted intermediates:
    gate_out, up_out (post row-slice), splits_gpu, scatter_index.
    """
    global _GATHER_RS_DISABLED, _GATHER_RS_ENGAGED_LOGGED
    global _GATHER_RS_SKIP_SMALL_LOGGED

    num_experts = w13_weight.shape[0]
    inter_shard = w13_weight.shape[1] // 2
    # Rank-uniform tail choice: topk_ids_global is the gathered routing (same
    # global token count on every rank), so this comparison never desyncs the
    # gather_rs collective. Decided BEFORE the _scatter_metadata .item() sync
    # so a below-threshold batch takes the torch tail with no risk of blocking
    # on a prior rank's pending collective.
    ntokens_global = topk_ids_global.shape[0]
    use_gather_rs = (
        _TAIL_MODE == "gather_rs"
        and not _GATHER_RS_DISABLED
        and ntokens_global >= _GATHER_RS_MIN_TOKENS
    )
    if (
        _TAIL_MODE == "gather_rs"
        and ntokens_global < _GATHER_RS_MIN_TOKENS
        and not _GATHER_RS_SKIP_SMALL_LOGGED
    ):
        _GATHER_RS_SKIP_SMALL_LOGGED = True
        logger.info(
            "[FLUX-MOE] gather_rs skipped: tokens=%d < min=%d (torch tail; "
            "flux GatherRS deadlocks its cross-rank barrier on tiny batches)",
            ntokens_global,
            _GATHER_RS_MIN_TOKENS,
        )

    splits_gpu, scatter_index = _scatter_metadata(topk_ids_global, num_experts)

    # w13 stacks [gate; up] along dim 1; dim-1 slices are non-contiguous
    # views (per-expert 2D blocks are contiguous). See module NOTE.
    gate_w = w13_weight[:, :inter_shard, :]
    up_w = w13_weight[:, inter_shard:, :]
    if _CONTIG_W:
        gate_w = gate_w.contiguous()
        up_w = up_w.contiguous()

    gate_out, up_out = op._run_gemm1(
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

    if capture is not None:
        capture.update(
            gate_out=gate_out,
            up_out=up_out,
            splits_gpu=splits_gpu,
            scatter_index=scatter_index,
        )

    if use_gather_rs:
        try:
            out = _gather_rs_tail(
                op,
                gate_out,
                up_out,
                splits_gpu,
                scatter_index,
                topk_weights_global,
                w2_weight,
            )
        except Exception as exc:  # noqa: BLE001 -- layered fallback: keep the
            # AGScatter GEMM1 engaged, only the fused tail drops out.
            # NOTE: a mid-forward failure here is only collective-safe if it
            # raises on ALL ranks (shape/threshold checks do; the gather_rs
            # collective itself, once entered, would already have hung). The
            # rank-uniform threshold above keeps every rank on the same branch.
            _GATHER_RS_DISABLED = True
            logger.warning(
                "[FLUX-MOE] gather_rs disabled: %s (falling back to the "
                "torch tail; AGScatter stays engaged)",
                exc,
                exc_info=True,
            )
        else:
            if not _GATHER_RS_ENGAGED_LOGGED:
                _GATHER_RS_ENGAGED_LOGGED = True
                logger.info(
                    "[FLUX-MOE] gather_rs engaged: tokens=%d rows=%d %s",
                    scatter_index.shape[0],
                    scatter_index.numel(),
                    op._gather_rs.ctor_desc,
                )
            return out

    return _moe_tail(
        gate_out, up_out, splits_gpu, scatter_index, topk_weights_global, w2_weight
    )


# One op per shape shared by ALL layers: the flux op holds nvshmem workspace
# buffers (not layer state), and per-layer instances exhaust the symmetric
# heap after a handful of layers (nvshmem_malloc -> nullptr at flux_shm.cc:117).
_OP_CACHE: dict = {}


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
    op = _OP_CACHE.get(key)
    if op is not None:
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
    _OP_CACHE[key] = op
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
