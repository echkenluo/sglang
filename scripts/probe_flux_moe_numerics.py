#!/usr/bin/env python3
"""Two-rank numeric probe: localize the flux-MoE teacher-forced logprob shift.

Usage (inside the serving container; PYTHONPATH must include the patched
sglang python/ dir and flux):

    torchrun --nproc_per_node=2 scripts/probe_flux_moe_numerics.py

On identical deterministic inputs (Qwen3-30B-A3B TP2 geometry: n=512,
hidden=2048, E=128, inter_shard=384, topk=8, bf16) it compares:

  Path B  the EXACT serving compute: flux_moe.flux_moe_compute — the same
          code path try_flux_moe_forward runs — including the flux AGScatter
          GEMM1 (ffn_mode="full", CONTIG_W honored, default 1 = the winning
          serving config).
  Path A  pure-torch fp32 reference on the full token tensor, expert-sorted
          with the same _scatter_metadata; plus a bf16 variant that mirrors
          the serving REFERENCE path's numerics (fused_experts triton: bf16
          GEMMs w/ fp32 accum, silu*mul computed in fp32 and rounded once,
          routing weights applied in FP32 before the bf16 store, fp32-accum
          sum over topk). That bf16 variant is the noise floor: deviations
          within ~3x of it are expected mixed-precision noise; beyond it,
          something computes differently.

Stages reported (max_abs / mean_abs / cosine, worst across ranks):
  (i)   flux gate_out        vs fp32 reference (expert-sorted)
  (ii)  flux up_out          vs fp32 reference (expert-sorted)
  (iii) flux final           vs fp32 reference final
  (iv)  flux final           vs bf16 reference final (serving-mirror)
  (v)   _moe_tail fed the REFERENCE GEMM1 outputs vs fp32 reference final
        (isolates the sglang-side tail: silu*mul, GEMM2 loop, unsort+combine)
plus a metadata oracle check (direct per-expert accumulation from topk_ids,
independent of the sort machinery) and, if GEMM1 deviates grossly, an
explicit gather-ORDERING hypothesis test (original rank-contiguous order vs
rank-rotated order) to catch a flux-internal all-gather layout mismatch.

Verdict: the first stage exceeding 3x the bf16 noise floor, mapped to
  (a) AGScatter GEMM1 + scatter metadata
  (b) the sglang-side tail
  (c) neither: genuine mixed-precision accumulation difference (quantified)
Exit code is nonzero iff any stage exceeds its threshold (or the metadata
oracle fails), so a wrapper can sentinel it. Rank 0 prints; metrics are
all-reduced so the exit code is rank-consistent.
"""

import os
import sys

# Must be set before sglang.srt.layers.moe.flux_moe is imported (module-level
# read). Default to the validated serving config; env still overrides.
os.environ.setdefault("SGLANG_FLUX_MOE_CONTIG_W", "1")

import torch
import torch.distributed as dist
import torch.nn.functional as F

SEED = 1234
N_TOKENS = 512
HIDDEN = 2048
N_EXPERTS = 128
INTER_SHARD = 384  # TP2 shard of Qwen3-30B-A3B moe_intermediate_size=768
TOPK = 8
TP = 2
DTYPE = torch.bfloat16
THRESH_MULT = 3.0
EPS = 1e-8
GROSS_MULT = 100.0  # (i) beyond this x floor => run the ordering hypothesis


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a64, b64 = a.double().reshape(-1), b.double().reshape(-1)
    d = (a64 - b64).abs()
    cos = F.cosine_similarity(a64.unsqueeze(0), b64.unsqueeze(0)).item()
    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(), cos=cos)


def seg_mm(xs: torch.Tensor, weights: torch.Tensor, splits_list) -> torch.Tensor:
    """Per-expert segment matmul over expert-sorted rows: xs[s:e] @ w[e].T."""
    out = xs.new_empty((xs.shape[0], weights.shape[1]))
    start = 0
    for expert_id, count in enumerate(splits_list):
        if count == 0:
            continue
        end = start + count
        torch.mm(xs[start:end], weights[expert_id].t(), out=out[start:end])
        start = end
    return out


def main() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == TP, f"launch with --nproc_per_node={TP} (got world={world})"
    dev = torch.device("cuda", local_rank)

    import flux

    flux.init_flux_shm(dist.group.WORLD)

    from sglang.srt.layers.moe import flux_moe
    from sglang.srt.layers.moe.flux_moe import (
        FluxMoeAGScatter,
        _moe_tail,
        _scatter_metadata,
        flux_moe_compute,
    )

    def p0(*args):
        if rank == 0:
            print(*args, flush=True)

    def reduce_worst(m: dict) -> dict:
        t = torch.tensor(
            [m["max_abs"], m["mean_abs"], -m["cos"]], device=dev, dtype=torch.float64
        )
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return dict(max_abs=t[0].item(), mean_abs=t[1].item(), cos=-t[2].item())

    def gather_scalar(v: float):
        t = torch.tensor([v], device=dev, dtype=torch.float64)
        outs = [torch.zeros_like(t) for _ in range(world)]
        dist.all_gather(outs, t)
        return [o.item() for o in outs]

    # ---- deterministic identical inputs on both ranks (CPU generator) ------
    g = torch.Generator().manual_seed(SEED)
    x_full = torch.randn(N_TOKENS, HIDDEN, generator=g).to(dev, DTYPE)
    # Weights scaled ~N(0, 1/fan_in) so activations stay O(1): precision
    # comparisons are then in the realistic regime instead of scale-dominated.
    w13 = (
        torch.randn(N_EXPERTS, 2 * INTER_SHARD, HIDDEN, generator=g) * HIDDEN**-0.5
    ).to(dev, DTYPE)
    w2 = (
        torch.randn(N_EXPERTS, HIDDEN, INTER_SHARD, generator=g) * INTER_SHARD**-0.5
    ).to(dev, DTYPE)
    router_logits = torch.randn(N_TOKENS, N_EXPERTS, generator=g).to(dev)
    probs = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, TOPK, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    local = x_full.chunk(TP, dim=0)[rank].contiguous()

    # ---- shared expert-sort metadata (the serving function) ----------------
    splits_gpu, scatter_index = _scatter_metadata(topk_ids, N_EXPERTS)
    splits_list = splits_gpu.tolist()
    flat_s2d = scatter_index.view(-1).long()  # (t*topk+k) -> sorted row
    total = flat_s2d.numel()
    row_src = torch.empty(total, dtype=torch.long, device=dev)
    row_src[flat_s2d] = torch.arange(total, device=dev)  # sorted row -> (t,k)
    token_of_row = row_src // TOPK
    w_sorted = topk_weights.reshape(-1)[row_src]  # fp32 weight per sorted row

    gate_w = w13[:, :INTER_SHARD, :]
    up_w = w13[:, INTER_SHARD:, :]

    # ---- Path A: fp32 reference (truth) ------------------------------------
    xs32 = x_full.float()[token_of_row]
    ref_gate32 = seg_mm(xs32, gate_w.float(), splits_list)
    ref_up32 = seg_mm(xs32, up_w.float(), splits_list)
    ref_act32 = F.silu(ref_gate32) * ref_up32
    ref_down32 = seg_mm(ref_act32, w2.float(), splits_list)
    ref_out32 = (
        (ref_down32 * w_sorted.unsqueeze(1))[flat_s2d]
        .view(N_TOKENS, TOPK, HIDDEN)
        .sum(dim=1)
    )

    # ---- Path A': bf16 noise floor (serving fused_experts numerics mirror):
    # bf16 GEMMs (cublas fp32 accum), silu*mul computed fp32 + one bf16 round,
    # routing weight applied in FP32 then bf16 store, fp32-accum topk sum.
    xs16 = x_full[token_of_row]
    ref_gate16 = seg_mm(xs16, gate_w, splits_list)
    ref_up16 = seg_mm(xs16, up_w, splits_list)
    ref_act16 = (F.silu(ref_gate16.float()) * ref_up16.float()).to(DTYPE)
    ref_down16 = seg_mm(ref_act16, w2, splits_list)
    ref_out16 = (
        (ref_down16.float() * w_sorted.unsqueeze(1))
        .to(DTYPE)[flat_s2d]
        .view(N_TOKENS, TOPK, HIDDEN)
        .float()
        .sum(dim=1)
        .to(DTYPE)
    )

    # ---- metadata oracle: direct per-expert accumulation from topk_ids,
    # independent of the sort machinery (catches scatter_index/splits bugs
    # that would otherwise cancel out between Path B and the sorted refs).
    oracle = torch.zeros(N_TOKENS, HIDDEN, dtype=torch.float32, device=dev)
    x32 = x_full.float()
    for e in range(N_EXPERTS):
        t_idx, k_idx = (topk_ids == e).nonzero(as_tuple=True)
        if t_idx.numel() == 0:
            continue
        h = x32[t_idx]
        d = (
            F.silu(h @ gate_w[e].float().t()) * (h @ up_w[e].float().t())
        ) @ w2[e].float().t()
        oracle.index_add_(0, t_idx, d * topk_weights[t_idx, k_idx].unsqueeze(1))
    m_oracle = reduce_worst(metrics(ref_out32, oracle))

    # ---- Path B: the serving compute (shared code path) ---------------------
    op = FluxMoeAGScatter(
        tp_group=dist.group.WORLD,
        tp_size=TP,
        num_experts=N_EXPERTS,
        topk=TOPK,
        hidden=HIDDEN,
        inter_shard=INTER_SHARD,
        dtype=DTYPE,
        ffn_mode="full",
    )
    cap = {}
    flux_out = flux_moe_compute(op, local, topk_ids, topk_weights, w13, w2, capture=cap)

    # ---- Path B tail isolation: the ACTUAL serving tail fed exact GEMM1 ----
    tail_out = _moe_tail(
        ref_gate32.to(DTYPE),
        ref_up32.to(DTYPE),
        splits_gpu,
        scatter_index,
        topk_weights,
        w2,
    )

    # ---- stage metrics (worst across ranks) ---------------------------------
    m_gate = reduce_worst(metrics(cap["gate_out"], ref_gate32))
    m_up = reduce_worst(metrics(cap["up_out"], ref_up32))
    m_fin32 = reduce_worst(metrics(flux_out, ref_out32))
    m_fin16 = reduce_worst(metrics(flux_out, ref_out16))
    m_tail = reduce_worst(metrics(tail_out, ref_out32))
    fg = metrics(ref_gate16, ref_gate32)
    fu = metrics(ref_up16, ref_up32)
    floor_g1 = reduce_worst(
        dict(
            max_abs=max(fg["max_abs"], fu["max_abs"]),
            mean_abs=max(fg["mean_abs"], fu["mean_abs"]),
            cos=min(fg["cos"], fu["cos"]),
        )
    )
    floor_fin = reduce_worst(metrics(ref_out16, ref_out32))

    thr_g1 = THRESH_MULT * floor_g1["mean_abs"] + EPS
    thr_fin = THRESH_MULT * floor_fin["mean_abs"] + EPS
    g1_mean = max(m_gate["mean_abs"], m_up["mean_abs"])
    oracle_fail = m_oracle["mean_abs"] > 1e-3 or m_oracle["max_abs"] > 1e-2
    exceed_g1 = g1_mean > thr_g1
    exceed_tail = m_tail["mean_abs"] > thr_fin
    exceed_fin = m_fin32["mean_abs"] > thr_fin

    # ---- ordering hypothesis (only meaningful if GEMM1 is grossly off):
    # does flux's internal gather use rank-contiguous [shard0;shard1] order
    # (== what scatter_index assumes, since topk_ids came from x_full in
    # original order), or a rank-ROTATED [local;remote] layout?
    ordering_note = ""
    if g1_mean > GROSS_MULT * (floor_g1["mean_abs"] + EPS):
        chunks = list(x_full.float().chunk(TP, dim=0))
        rot = torch.cat([chunks[(rank + i) % TP] for i in range(TP)], dim=0)
        rot_gate32 = seg_mm(rot[token_of_row], gate_w.float(), splits_list)
        m_rot = metrics(cap["gate_out"], rot_gate32)["mean_abs"]
        m_orig = metrics(cap["gate_out"], ref_gate32)["mean_abs"]
        origs = gather_scalar(m_orig)
        rots = gather_scalar(m_rot)
        per_rank = "; ".join(
            f"rank{r}: orig={origs[r]:.3e} rotated={rots[r]:.3e}"
            for r in range(world)
        )
        # rank0's rotation is the identity; only ranks>0 discriminate.
        rotated_match = any(
            rots[r] < origs[r] / 10.0 for r in range(1, world)
        )
        if rotated_match:
            ordering_note = (
                f" ORDERING MISMATCH: gate_out matches the rank-ROTATED gather "
                f"order ({per_rank}) — flux's internal AG layout differs from "
                f"the rank-contiguous order scatter_index assumes."
            )
        else:
            ordering_note = (
                f" ordering check: no rank-rotated match ({per_rank}) — "
                f"misorder is not the cause."
            )

    # ---- report -------------------------------------------------------------
    def fmt(name, m):
        return (
            f"{name:<50s} max_abs={m['max_abs']:.6e} "
            f"mean_abs={m['mean_abs']:.6e} cos={m['cos']:.9f}"
        )

    p0("=" * 100)
    p0(
        f"probe config: n={N_TOKENS} hidden={HIDDEN} E={N_EXPERTS} "
        f"inter_shard={INTER_SHARD} topk={TOPK} tp={TP} dtype={DTYPE} "
        f"seed={SEED} CONTIG_W={flux_moe._CONTIG_W}"
    )
    p0(f"flux op: {op.ctor_desc}")
    p0("-" * 100)
    p0(fmt("metadata oracle (sorted-ref vs direct, fp32)", m_oracle))
    p0(fmt("noise floor GEMM1 (bf16 ref vs fp32 ref)", floor_g1))
    p0(fmt("noise floor final (bf16 ref vs fp32 ref)", floor_fin))
    p0("-" * 100)
    p0(fmt("(i)   flux gate_out vs fp32 ref", m_gate))
    p0(fmt("(ii)  flux up_out   vs fp32 ref", m_up))
    p0(fmt("(iii) flux final    vs fp32 ref", m_fin32))
    p0(fmt("(iv)  flux final    vs bf16 ref (serving mirror)", m_fin16))
    p0(fmt("(v)   serving tail(ref GEMM1) vs fp32 ref", m_tail))
    p0("-" * 100)

    if oracle_fail:
        verdict = (
            f"VERDICT: (a) SCATTER METADATA broken — sorted-machinery reference "
            f"deviates from the direct oracle (mean_abs={m_oracle['mean_abs']:.3e}); "
            f"fix _scatter_metadata before trusting any other stage."
        )
    elif exceed_g1:
        verdict = (
            f"VERDICT: (a) flux AGScatter GEMM1 deviates first — "
            f"gate/up mean_abs={g1_mean:.3e} > 3x bf16 floor "
            f"({floor_g1['mean_abs']:.3e}, thr={thr_g1:.3e}).{ordering_note}"
        )
    elif exceed_tail:
        verdict = (
            f"VERDICT: (b) sglang-side tail deviates first (fed exact GEMM1) — "
            f"mean_abs={m_tail['mean_abs']:.3e} > 3x bf16 floor "
            f"({floor_fin['mean_abs']:.3e}, thr={thr_fin:.3e}); prime suspects: "
            f"bf16 topk-weight combine and bf16 silu*mul (reference keeps both "
            f"in fp32)."
        )
    elif exceed_fin:
        verdict = (
            f"VERDICT: (a+b) compounded — GEMM1 and tail each pass 3x floor but "
            f"end-to-end exceeds it (mean_abs={m_fin32['mean_abs']:.3e} > "
            f"thr={thr_fin:.3e})."
        )
    else:
        ratio = m_fin32["mean_abs"] / (floor_fin["mean_abs"] + EPS)
        verdict = (
            f"VERDICT: (c) within 3x bf16 noise floor — genuine mixed-precision "
            f"accumulation difference. flux-vs-fp32 mean_abs="
            f"{m_fin32['mean_abs']:.3e} = {ratio:.2f}x floor "
            f"({floor_fin['mean_abs']:.3e}); flux vs serving-mirror bf16 "
            f"mean_abs={m_fin16['mean_abs']:.3e}."
        )
    p0(verdict)
    p0("=" * 100)

    failed = oracle_fail or exceed_g1 or exceed_tail or exceed_fin
    dist.barrier()
    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
