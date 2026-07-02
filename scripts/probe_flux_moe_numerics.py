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
  (vi)  slice-2 end-to-end (flux GEMM1 + fused GatherRS tail: GEMM2 + topk
        gather + fp32 weight scaling + reduce + cross-rank ReduceScatter)
        vs the fp32 reference SCATTERED rows
  (vii) same slice-2 output vs the bf16 reference SCATTERED rows
  (viii-x) slice-1 batched GEMM2 recovery (torch tail, SGLANG_FLUX_MOE_GEMM2):
        (viii) grouped GEMM2 (torch._grouped_mm) vs the per-expert loop, GEMM2
        stage only -- must match to rounding (same FLOPs); (ix)/(x) grouped-
        and loop-GEMM2 tails vs fp32 ref -- both at the final noise floor.
        Skipped if torch._grouped_mm is unavailable.
  (xi)  load-time weight repack (SGLANG_FLUX_MOE_LOAD_REPACK): the prebuilt
        contiguous gate/up path vs the per-forward-slice path -- must be
        BIT-IDENTICAL (max_abs == 0), since the repack is the same
        .contiguous() done once at load instead of every forward.
  (xii) gate-local routing (SGLANG_FLUX_MOE_GATE_LOCAL): routing computed on
        each rank's SCATTERED shard + metadata-all-gathered vs routing on the
        gathered full tokens -- topk_ids / splits / scatter_index must be
        BIT-IDENTICAL (the gate is a replicated per-token fn, so routing is
        gather-invariant; this also validates the metadata-AG row order).
Both ranks hold IDENTICAL weight shards, so GatherRS's cross-rank reduction
doubles the single-shard partial: the scattered references are
(2 * reference).chunk(TP)[rank]. A missing or rank-misordered reduction is
still visible (1x values / swapped rank blocks).
plus a metadata oracle check (direct per-expert accumulation from topk_ids,
independent of the sort machinery) and, if GEMM1 deviates grossly, an
explicit gather-ORDERING hypothesis test (original rank-contiguous order vs
rank-rotated order) to catch a flux-internal all-gather layout mismatch.

Verdict: the first stage exceeding 3x the bf16 noise floor, mapped to
  (a) AGScatter GEMM1 + scatter metadata
  (b) the sglang-side torch tail / (b/gather_rs) the fused GatherRS tail
  (c) neither: genuine mixed-precision accumulation difference (quantified)
  (d) batched grouped GEMM2 diverges from the loop (perf path unsafe)
  (e) load-time repack changes output (must be bit-identical)
  (f) gate-local routing differs from gate-gathered (must be bit-identical)
Exit code is nonzero iff any stage exceeds its threshold (or the metadata
oracle fails), so a wrapper can sentinel it. Rank 0 prints; metrics are
all-reduced so the exit code is rank-consistent.

SGLANG_FLUX_MOE_MAX_NTOKENS (default 8192) also applies here; it sizes both
flux workspaces (GatherRS max_m = max_ntokens * topk) and must be set before
launch if the nvshmem heap is tight.

HANG-LOCALIZATION MODE (SGLANG_FLUX_MOE_PROBE_NS): when set, the probe skips
the numeric comparison and instead runs the real gather_rs tail across a
comma-separated batch-size sweep, printing an [ENTER]/[OK] heartbeat per case
so a deadlock pins to one batch shape (reproduces the serving hang without a
server). Suffix a size with "c" for concentrated routing (all tokens pick
experts [0, topk) -> most experts get 0 tokens). Example:

    SGLANG_FLUX_MOE_PROBE_NS="2,4,8,8c,512" \
        torchrun --nproc_per_node=2 scripts/probe_flux_moe_numerics.py

A case that hangs freezes right after its [ENTER] line; the first such n (or
the first concentrated case) is the trigger. The sweep calls _gather_rs_tail
directly, so it is NOT gated by SGLANG_FLUX_MOE_GATHER_RS_MIN_TOKENS -- that
is deliberate, so the collective is actually entered at tiny M.

MULTI-LAYER INTERLEAVE MODE (SGLANG_FLUX_MOE_PROBE_LAYERS): replays L
iterations of [AGScatter GEMM1 -> gather_rs tail] on one fixed batch
(SGLANG_FLUX_MOE_PROBE_N, default 4096), a server-free repro of the 48-layer
serving interleave that Path C's single composition does not exercise. Prints
a per-iteration heartbeat so a deadlock pins to the iteration index
(distinguishes accumulated-state/heap growth from a single-op bug). Example:

    SGLANG_FLUX_MOE_PROBE_LAYERS=48 SGLANG_FLUX_MOE_PROBE_N=4096 \
        torchrun --nproc_per_node=2 scripts/probe_flux_moe_numerics.py
"""

import os
import sys

# Must be set before sglang.srt.layers.moe.flux_moe is imported (module-level
# read). Default to the validated serving config; env still overrides.
os.environ.setdefault("SGLANG_FLUX_MOE_CONTIG_W", "1")
# Path B must be the torch-tail pipeline (full-row output) regardless of the
# launcher's env; the gather_rs tail is exercised explicitly as Path C.
os.environ["SGLANG_FLUX_MOE_TAIL"] = "torch"

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
        _gather_rs_tail,
        _gemm2_grouped,
        _gemm2_loop,
        _get_prebuilt_gate_up,
        _moe_tail,
        _scatter_metadata,
        flux_moe_compute,
        gather_local_routing,
    )

    def _combine(down, scatter_index, topk_weights_global):
        # The tail's vectorized unsort + topk-weighted combine (shared by both
        # GEMM2 backends; mirrors _moe_tail's tail so Path D isolates GEMM2).
        nt, tk = scatter_index.shape
        gathered = down[scatter_index.view(-1).long()].view(nt, tk, -1)
        w = topk_weights_global.to(gathered.dtype).reshape(nt, tk, 1)
        return (gathered * w).sum(dim=1)

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

    def make_op(inter_shard=INTER_SHARD):
        return FluxMoeAGScatter(
            tp_group=dist.group.WORLD,
            tp_size=TP,
            num_experts=N_EXPERTS,
            topk=TOPK,
            hidden=HIDDEN,
            inter_shard=inter_shard,
            dtype=DTYPE,
            ffn_mode="full",
        )

    # ---- tiny-batch hang localization (SGLANG_FLUX_MOE_PROBE_NS) -----------
    # Runs the REAL gather_rs tail across a sweep of batch sizes, printing a
    # heartbeat before/after each so a hang pins to one n. This reproduces the
    # serving deadlock (2-token warmup) without a server. "Nc" (e.g. "8c")
    # forces concentrated routing: all tokens pick experts [0, topk), so every
    # other expert gets 0 tokens -- the zero-count-expert regime that flux's
    # own tests (even round-robin over few experts, huge M) never exercise.
    probe_ns_env = os.environ.get("SGLANG_FLUX_MOE_PROBE_NS", "")
    if probe_ns_env:
        cases = []
        for tok in probe_ns_env.split(","):
            tok = tok.strip()
            if not tok:
                continue
            concentrated = tok.endswith("c")
            cases.append((int(tok[:-1] if concentrated else tok), concentrated))
        p0("=" * 100)
        p0(f"HANG-LOCALIZATION SWEEP: cases={cases} (Nc = concentrated routing)")
        p0("Each case: build inputs -> barrier -> gather_rs tail -> sync -> "
           "barrier. A hang stops right after the last '[ENTER]' line.")
        p0("-" * 100)
        sweep_op = make_op()
        gsweep = torch.Generator().manual_seed(SEED)
        # Small fan-in-scaled weights so activations stay finite; values are
        # irrelevant to a collective deadlock, only shapes/counts matter.
        w13_s = (
            torch.randn(N_EXPERTS, 2 * INTER_SHARD, HIDDEN, generator=gsweep)
            * HIDDEN**-0.5
        ).to(dev, DTYPE)
        w2_s = (
            torch.randn(N_EXPERTS, HIDDEN, INTER_SHARD, generator=gsweep)
            * INTER_SHARD**-0.5
        ).to(dev, DTYPE)
        for n_tok, concentrated in cases:
            tag = f"n={n_tok}{' concentrated' if concentrated else ''}"
            if n_tok % TP != 0:
                p0(f"[SKIP ] {tag}: not divisible by tp={TP} (gate rejects it)")
                continue
            xs = torch.randn(n_tok, HIDDEN, generator=gsweep).to(dev, DTYPE)
            if concentrated:
                tids = (
                    torch.arange(TOPK, dtype=torch.int32).repeat(n_tok, 1).to(dev)
                )
                tw = torch.full((n_tok, TOPK), 1.0 / TOPK, device=dev)
            else:
                pr = torch.softmax(
                    torch.randn(n_tok, N_EXPERTS, generator=gsweep).to(dev).float(),
                    dim=-1,
                )
                tw, tids = torch.topk(pr, TOPK, dim=-1)
                tw = (tw / tw.sum(-1, keepdim=True)).contiguous()
                tids = tids.to(torch.int32).contiguous()
            loc = xs.chunk(TP, dim=0)[rank].contiguous()
            sp, si, _ = _scatter_metadata(tids, N_EXPERTS)
            nz = int((sp > 0).sum().item())
            gate_out, up_out = sweep_op._run_gemm1(
                loc, [w13_s[:, :INTER_SHARD, :], w13_s[:, INTER_SHARD:, :]], sp, si
            )
            dist.barrier()
            p0(f"[ENTER] {tag}: rows={si.numel()} nonzero_experts={nz}/{N_EXPERTS}")
            try:
                out = _gather_rs_tail(sweep_op, gate_out, up_out, sp, si, tw, w2_s)
                torch.cuda.synchronize()
                dist.barrier()
                p0(f"[ OK  ] {tag}: out={tuple(out.shape)}")
            except Exception as exc:  # noqa: BLE001
                dist.barrier()
                p0(f"[ERROR] {tag}: {type(exc).__name__}: {exc}")
        p0("=" * 100)
        p0("SWEEP COMPLETE (reached end without hang). A case that hung would "
           "have frozen after its [ENTER]; the first such n is the trigger.")
        dist.barrier()
        dist.destroy_process_group()
        return 0

    # ---- multi-layer interleave repro (SGLANG_FLUX_MOE_PROBE_LAYERS) -------
    # Path C proved ONE AGScatter->GatherRS composition works at n=512. The
    # serving hang is at the FULL 48-layer loop where AGScatter (nvshmem
    # collective) and GatherRS (nvshmem TEAM_WORLD barrier + ring RS) interleave
    # 48x. This mode replays exactly that: L iterations of
    # [AGScatter GEMM1 -> gather_rs tail] on a fixed batch, printing a
    # heartbeat per iter so a hang pins to the iteration index (isolating
    # accumulated-state / heap-growth from a single-op bug). Set
    # SGLANG_FLUX_MOE_PROBE_LAYERS=48 (and optionally SGLANG_FLUX_MOE_PROBE_N to
    # the batch size, default 4096 = the batch that hung serving).
    probe_layers = int(os.environ.get("SGLANG_FLUX_MOE_PROBE_LAYERS", "0"))
    if probe_layers > 0:
        n_tok = int(os.environ.get("SGLANG_FLUX_MOE_PROBE_N", "4096"))
        assert n_tok % TP == 0, f"SGLANG_FLUX_MOE_PROBE_N={n_tok} must be % tp"
        p0("=" * 100)
        p0(f"MULTI-LAYER INTERLEAVE REPRO: layers={probe_layers} n={n_tok} "
           f"(each iter = AGScatter GEMM1 -> gather_rs tail, like one decoder "
           f"MoE). A hang freezes right after the last '[iter K]' line.")
        p0("-" * 100)
        li_op = make_op()
        gli = torch.Generator().manual_seed(SEED)
        w13_l = (
            torch.randn(N_EXPERTS, 2 * INTER_SHARD, HIDDEN, generator=gli)
            * HIDDEN**-0.5
        ).to(dev, DTYPE)
        w2_l = (
            torch.randn(N_EXPERTS, HIDDEN, INTER_SHARD, generator=gli)
            * INTER_SHARD**-0.5
        ).to(dev, DTYPE)
        xs = torch.randn(n_tok, HIDDEN, generator=gli).to(dev, DTYPE)
        pr = torch.softmax(
            torch.randn(n_tok, N_EXPERTS, generator=gli).to(dev).float(), dim=-1
        )
        tw, tids = torch.topk(pr, TOPK, dim=-1)
        tw = (tw / tw.sum(-1, keepdim=True)).contiguous()
        tids = tids.to(torch.int32).contiguous()
        loc = xs.chunk(TP, dim=0)[rank].contiguous()
        sp, si, _ = _scatter_metadata(tids, N_EXPERTS)
        w13_groups = [w13_l[:, :INTER_SHARD, :], w13_l[:, INTER_SHARD:, :]]
        for it in range(probe_layers):
            dist.barrier()
            p0(f"[iter {it}] enter")
            try:
                g_out, u_out = li_op._run_gemm1(loc, w13_groups, sp, si)
                out = _gather_rs_tail(li_op, g_out, u_out, sp, si, tw, w2_l)
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                dist.barrier()
                p0(f"[iter {it}] ERROR {type(exc).__name__}: {exc}")
                break
        else:
            dist.barrier()
            p0(f"[ OK  ] all {probe_layers} iters completed, out="
               f"{tuple(out.shape)}")
        p0("=" * 100)
        p0("If this froze at some [iter K], the interleave accumulates state "
           "that deadlocks (hypothesis 1: heap/barrier). If it completed but "
           "serving still hangs, the trigger is the attention/prepare_attn "
           "collectives BETWEEN the MoE ops, not the MoE ops alone.")
        dist.barrier()
        dist.destroy_process_group()
        return 0

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
    splits_gpu, scatter_index, seg_indptr = _scatter_metadata(topk_ids, N_EXPERTS)
    splits_list = splits_gpu.tolist()
    flat_s2d = scatter_index.view(-1).long()  # (t*topk+k) -> sorted row
    total = flat_s2d.numel()
    row_src = torch.empty(total, dtype=torch.long, device=dev)
    row_src[flat_s2d] = torch.arange(total, device=dev)  # sorted row -> (t,k)
    token_of_row = row_src // TOPK
    w_sorted = topk_weights.reshape(-1)[row_src]  # fp32 weight per sorted row

    gate_w = w13[:, :INTER_SHARD, :]
    up_w = w13[:, INTER_SHARD:, :]

    # ---- gate-local routing identity (SGLANG_FLUX_MOE_GATE_LOCAL) -----------
    # Prove that routing computed on each rank's SCATTERED shard, then metadata-
    # all-gathered, is BIT-IDENTICAL to routing computed on the gathered full
    # tokens. Uses a real per-token gate (replicated matrix) so the comparison
    # is meaningful: gate_router(x) = softmax(x @ Wg.T) -> renorm topk.
    Wg = (torch.randn(N_EXPERTS, HIDDEN, generator=g) * HIDDEN**-0.5).to(dev, DTYPE)

    def _route(h):
        pr = torch.softmax((h @ Wg.t()).float(), dim=-1)
        tw, tid = torch.topk(pr, TOPK, dim=-1)
        tw = (tw / tw.sum(-1, keepdim=True)).contiguous()
        return tid.to(torch.int32).contiguous(), tw.contiguous()

    # gathered-gate: route all tokens at once.
    ids_gathered, w_gathered = _route(x_full)
    # local-gate + metadata-AG: route this rank's shard, all-gather metadata.
    ids_loc, w_loc = _route(local)
    ids_ag, w_ag = gather_local_routing(ids_loc, w_loc, TP)
    # scatter metadata from each; splits/scatter_index must match exactly.
    sp_gathered, si_gathered, seg_gathered = _scatter_metadata(ids_gathered, N_EXPERTS)
    sp_ag, si_ag, seg_ag = _scatter_metadata(ids_ag, N_EXPERTS)
    m_ids = int((ids_ag != ids_gathered).sum().item())
    m_w = (w_ag - w_gathered).abs().max().item()
    m_splits = int((sp_ag != sp_gathered).sum().item())
    m_scatter = int((si_ag != si_gathered).sum().item())
    gate_local_bad = bool(m_ids or m_splits or m_scatter or m_w > 0.0)

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

    # ---- load-time repack equivalence: the prebuilt contiguous gate/up must
    # give byte-identical output to the per-forward slice path (it's the same
    # .contiguous() done once). Reuses a throwaway block-like holder.
    class _Blk:  # minimal stand-in for the sparse-moe block attr cache
        pass

    gate_up = _get_prebuilt_gate_up(_Blk(), w13)
    flux_out_repack = flux_moe_compute(
        op, local, topk_ids, topk_weights, w13, w2, gate_up=gate_up
    )
    m_repack = reduce_worst(metrics(flux_out_repack, flux_out))

    # ---- Path B tail isolation: the ACTUAL serving tail fed exact GEMM1 ----
    tail_out = _moe_tail(
        ref_gate32.to(DTYPE),
        ref_up32.to(DTYPE),
        splits_gpu,
        scatter_index,
        topk_weights,
        w2,
    )

    # ---- Path C: slice-2 end-to-end (flux GEMM1 + fused GatherRS tail) -----
    # The serving gather_rs code path (collective; both ranks construct the
    # GatherRS op through the same builder). Output: (N_TOKENS/TP, HIDDEN)
    # scattered REDUCED rows. Raises on failure (strict: no silent fallback
    # to the torch tail here, unlike serving).
    grs_out = _gather_rs_tail(
        op, cap["gate_out"], cap["up_out"], splits_gpu, scatter_index,
        topk_weights, w2,
    )
    # Identical weight shards on both ranks => cross-rank sum = 2x the
    # single-shard partial (see module docstring).
    ref_rs32 = (2.0 * ref_out32).chunk(TP, dim=0)[rank]
    ref_rs16 = (2.0 * ref_out16.float()).to(DTYPE).chunk(TP, dim=0)[rank]

    # ---- Path D: batched grouped GEMM2 torch tail (slice-1 perf recovery) --
    # Runs BOTH GEMM2 backends explicitly on the SAME silu*mul activation (fed
    # the exact fp32 reference GEMM1, like Path B) so the comparison isolates
    # GEMM2: grouped (torch._grouped_mm, GPU offsets) MUST match the loop to
    # rounding, and both must sit at the bf16 final noise floor vs fp32. If
    # torch._grouped_mm is unavailable this path is skipped (reported).
    act_ref = (F.silu(ref_gate32) * ref_up32).to(DTYPE)
    have_grouped = hasattr(torch, "_grouped_mm")
    grouped_err = None
    if have_grouped:
        down_loop = _gemm2_loop(act_ref, w2, splits_gpu)
        try:
            down_grp = _gemm2_grouped(act_ref, w2, seg_indptr)
        except Exception as exc:  # noqa: BLE001 -- report a layout rejection
            # cleanly instead of crashing the whole probe.
            grouped_err = f"{type(exc).__name__}: {exc}"
            have_grouped = False
        else:
            grp_out = _combine(down_grp, scatter_index, topk_weights)
            loop_out = _combine(down_loop, scatter_index, topk_weights)
            m_grp_vs_loop = reduce_worst(metrics(down_grp, down_loop))  # GEMM2
            m_grp_fin = reduce_worst(metrics(grp_out, ref_out32))
            m_loop_fin = reduce_worst(metrics(loop_out, ref_out32))

    # ---- stage metrics (worst across ranks) ---------------------------------
    m_gate = reduce_worst(metrics(cap["gate_out"], ref_gate32))
    m_up = reduce_worst(metrics(cap["up_out"], ref_up32))
    m_fin32 = reduce_worst(metrics(flux_out, ref_out32))
    m_fin16 = reduce_worst(metrics(flux_out, ref_out16))
    m_tail = reduce_worst(metrics(tail_out, ref_out32))
    m_grs32 = reduce_worst(metrics(grs_out, ref_rs32))
    m_grs16 = reduce_worst(metrics(grs_out, ref_rs16))
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
    floor_rs = reduce_worst(metrics(ref_rs16, ref_rs32))

    thr_g1 = THRESH_MULT * floor_g1["mean_abs"] + EPS
    thr_fin = THRESH_MULT * floor_fin["mean_abs"] + EPS
    thr_rs = THRESH_MULT * floor_rs["mean_abs"] + EPS
    g1_mean = max(m_gate["mean_abs"], m_up["mean_abs"])
    oracle_fail = m_oracle["mean_abs"] > 1e-3 or m_oracle["max_abs"] > 1e-2
    exceed_g1 = g1_mean > thr_g1
    exceed_tail = m_tail["mean_abs"] > thr_fin
    exceed_grs = m_grs32["mean_abs"] > thr_rs
    exceed_fin = m_fin32["mean_abs"] > thr_fin
    # Load-time repack must be EXACTLY equal to the per-forward slice path
    # (same .contiguous(), done once) -- bit-identical, so tolerance is 0.
    exceed_repack = m_repack["max_abs"] != 0.0
    # Path D thresholds: grouped GEMM2 must equal the loop essentially exactly
    # (same FLOPs, both bf16 cublas) and sit at the final noise floor vs fp32.
    exceed_grp = False
    if have_grouped:
        exceed_grp = (
            m_grp_vs_loop["mean_abs"] > thr_fin  # grouped vs loop divergence
            or m_grp_fin["mean_abs"] > thr_fin  # grouped vs fp32 ref
        )

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
        f"seed={SEED} CONTIG_W={flux_moe._CONTIG_W} "
        f"gather_rs_min_tokens={flux_moe._GATHER_RS_MIN_TOKENS} "
        f"gemm2_mode={flux_moe._GEMM2_MODE} "
        f"has_grouped_mm={flux_moe._HAS_GROUPED_MM} "
        f"load_repack={flux_moe._LOAD_REPACK} "
        f"gate_local={flux_moe._GATE_LOCAL}"
    )
    p0(f"flux op: {op.ctor_desc}")
    p0(f"flux gather_rs op: {op._gather_rs.ctor_desc}")
    p0("-" * 100)
    p0(fmt("metadata oracle (sorted-ref vs direct, fp32)", m_oracle))
    p0(fmt("noise floor GEMM1 (bf16 ref vs fp32 ref)", floor_g1))
    p0(fmt("noise floor final (bf16 ref vs fp32 ref)", floor_fin))
    p0(fmt("noise floor scattered (bf16 rs-ref vs fp32 rs-ref)", floor_rs))
    p0("-" * 100)
    p0(fmt("(i)   flux gate_out vs fp32 ref", m_gate))
    p0(fmt("(ii)  flux up_out   vs fp32 ref", m_up))
    p0(fmt("(iii) flux final    vs fp32 ref", m_fin32))
    p0(fmt("(iv)  flux final    vs bf16 ref (serving mirror)", m_fin16))
    p0(fmt("(v)   serving tail(ref GEMM1) vs fp32 ref", m_tail))
    p0(fmt("(vi)  gather_rs slice-2 vs fp32 scattered ref", m_grs32))
    p0(fmt("(vii) gather_rs slice-2 vs bf16 scattered ref", m_grs16))
    if have_grouped:
        p0(fmt("(viii) grouped GEMM2 vs loop GEMM2 (GEMM2 only)", m_grp_vs_loop))
        p0(fmt("(ix)  grouped-GEMM2 tail vs fp32 ref", m_grp_fin))
        p0(fmt("(x)   loop-GEMM2 tail   vs fp32 ref", m_loop_fin))
    else:
        why = grouped_err or "torch._grouped_mm unavailable"
        p0(f"(viii-x) grouped GEMM2: SKIPPED ({why}) -- serving auto mode "
           "would use the loop backend")
    p0(fmt("(xi)  load-repack path vs per-forward-slice (must be 0)", m_repack))
    p0(
        f"{'(xii) gate-local vs gate-gathered routing (must be 0)':<50s} "
        f"ids_mismatch={m_ids} splits_mismatch={m_splits} "
        f"scatter_mismatch={m_scatter} w_max_abs={m_w:.3e}"
    )
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
    elif exceed_grs:
        verdict = (
            f"VERDICT: (b/gather_rs) fused GatherRS tail deviates (GEMM1 and "
            f"torch tail pass) — mean_abs={m_grs32['mean_abs']:.3e} > 3x "
            f"scattered bf16 floor ({floor_rs['mean_abs']:.3e}, "
            f"thr={thr_rs:.3e}); suspects: vec_scale/routing_idx mapping, "
            f"reduce-scatter layout, GatherRS accumulation."
        )
    elif exceed_fin:
        verdict = (
            f"VERDICT: (a+b) compounded — GEMM1 and tail each pass 3x floor but "
            f"end-to-end exceeds it (mean_abs={m_fin32['mean_abs']:.3e} > "
            f"thr={thr_fin:.3e})."
        )
    elif exceed_grp:
        verdict = (
            f"VERDICT: (d) batched grouped GEMM2 DIVERGES — grouped-vs-loop "
            f"mean_abs={m_grp_vs_loop['mean_abs']:.3e}, grouped-vs-fp32="
            f"{m_grp_fin['mean_abs']:.3e} (thr={thr_fin:.3e}); "
            f"torch._grouped_mm offsets/layout wrong. Serving must use "
            f"SGLANG_FLUX_MOE_GEMM2=loop until fixed."
        )
    elif exceed_repack:
        verdict = (
            f"VERDICT: (e) load-time repack CHANGES output — repack-vs-slice "
            f"max_abs={m_repack['max_abs']:.3e} != 0 (must be bit-identical; "
            f"the prebuilt gate/up is the same .contiguous() done once). Bug "
            f"in _get_prebuilt_gate_up slicing/caching."
        )
    elif gate_local_bad:
        verdict = (
            f"VERDICT: (f) gate-local routing DIFFERS from gate-gathered — "
            f"ids_mismatch={m_ids} splits_mismatch={m_splits} "
            f"scatter_mismatch={m_scatter} w_max_abs={m_w:.3e} (must be 0). "
            f"Local-shard routing must equal full-token routing; check the "
            f"gate is replicated + the metadata-AG row order. Serving must use "
            f"SGLANG_FLUX_MOE_GATE_LOCAL=0 until fixed."
        )
    else:
        ratio = m_fin32["mean_abs"] / (floor_fin["mean_abs"] + EPS)
        ratio_rs = m_grs32["mean_abs"] / (floor_rs["mean_abs"] + EPS)
        grp_note = ""
        if have_grouped:
            grp_note = (
                f"; grouped GEMM2 == loop (mean_abs={m_grp_vs_loop['mean_abs']:.3e}"
                f", grouped-vs-fp32={m_grp_fin['mean_abs']:.3e} vs loop "
                f"{m_loop_fin['mean_abs']:.3e}) -- perf path numerically safe"
            )
        verdict = (
            f"VERDICT: (c) within 3x bf16 noise floor — genuine mixed-precision "
            f"accumulation difference. flux-vs-fp32 mean_abs="
            f"{m_fin32['mean_abs']:.3e} = {ratio:.2f}x floor "
            f"({floor_fin['mean_abs']:.3e}); flux vs serving-mirror bf16 "
            f"mean_abs={m_fin16['mean_abs']:.3e}; gather_rs slice-2 vs fp32 "
            f"scattered mean_abs={m_grs32['mean_abs']:.3e} = {ratio_rs:.2f}x "
            f"scattered floor ({floor_rs['mean_abs']:.3e})" + grp_note + "."
        )
    p0(verdict)
    p0("=" * 100)

    failed = (
        oracle_fail
        or exceed_g1
        or exceed_tail
        or exceed_grs
        or exceed_fin
        or exceed_grp
        or exceed_repack
        or gate_local_bad
    )
    dist.barrier()
    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
