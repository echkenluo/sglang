#!/usr/bin/env python3
"""Two-rank numeric probe: validate the M2-s1 flux-fp8 dense GLUE (T5 path).

Usage (inside the serving container; PYTHONPATH must include the patched
sglang python/ dir and flux):

    SGLANG_FLUX_FP8=1 SGLANG_USE_FUSED_OVERLAP=1 \
        torchrun --nproc_per_node=2 scripts/probe_flux_fp8_dense.py

This is the fp8 sibling of scripts/probe_flux_moe_numerics.py. It targets one
DENSE decoder layer's two flux joints on TP2 (L20 / SM89) with synthetic
weights + inputs:

  qkv_proj joint  flux AGKernel  (AGCook: AG fp8 payload + fp8 GEMM -> bf16)
  o_proj  joint  flux GemmRS    (GemmRS: fp8 GEMM -> bf16 partials + RS)

The s1 route (m2-design-flux-fp8.md sections 2-3) rides the ONLY instantiated
SM89 dense fp8 flux kernels: per-tensor scale_a x per-tensor scale_b, absmax
epilogue. The design gate for s1 is GLUE CORRECTNESS -- not end-model
accuracy. Per-tensor weight quant is known-lossy (that is recorded by
m2_gate.sh, not gated). So this probe checks that flux's fused fp8 GEMM+comm
produces the SAME numbers as a plain-torch implementation of the IDENTICAL
per-tensor math, and that flux is no further from an fp32 anchor than that
same-math reference is.

Three legs per joint, on deterministic identical inputs on both ranks:

  (i)  FLUX leg     -- through the T5 code path: the real
       Fp8LinearMethod._forward_flux_fp8 / _flux_fp8_prepare_weight (imported
       from sglang.srt.layers.quantization.fp8, NOT reimplemented here),
       driving a real flux AGKernel / GemmRS op.
  (ii) REFERENCE leg -- the IDENTICAL per-tensor math in plain torch:
       - qkv: rank-uniform amax (MAX-allreduce) -> scaled_fp8_quant (the SAME
         kernel flux's path calls) -> torch all_gather of the fp8 payload ->
         dequant matmul (input_scale * weight_scale * (A_q @ W_q.T)).
       - o_proj: LOCAL per-tensor scale -> scaled_fp8_quant -> dequant matmul
         -> torch reduce_scatter of the bf16 partials.
       Same scale, same quant kernel, same weight bytes as the flux leg, so
       the fp8 payload is BITWISE identical by construction; the reference
       differs from flux only in HOW the collective + GEMM are executed.
  (iii) FP32 anchor -- no quantization: torch fp32 matmul + fp32 collectives.
       The truth the two fp8 legs are scored against (design section 3 c-class
       verdict basis: whoever is closer to fp32 is right).

Anchors (the four divergence points, m2-design section 3 table):
  (1) post-quant  fp8 payload + scale : flux-leg quant == reference-leg quant,
      BITWISE (max_abs of the fp8 payloads reinterpreted as bytes == 0, and
      the scales are identical). Proves the quantization the flux path applies
      is exactly the reference per-tensor quant -- and that no view/stride
      corruption (the .contiguous() scar) mangled the payload.
  (2) post-AG     gathered fp8 order  : flux-internal AG layout == torch
      all_gather rank-contiguous [shard0; shard1; ...] order, BITWISE on the
      fp8 payload. Catches a flux AG permutation (the M3 ordering scar).
      Only meaningful on the qkv (AGKernel) joint.
  (3) post-GEMM   bf16 output         : |flux - fp32| <= 1.1 * |reference -
      fp32|, using a NOISE FLOOR mirror -- the reference GEMM is run TWICE
      with a reordered-summation perturbation (K split + regrouped) and the
      spread between them is the accumulation-order noise floor; the 1.1x is
      a small multiplicative slack on top of the reference's own distance to
      fp32. flux must not be materially further from fp32 than the same-math
      torch reference is.
  (4) post-RS     scattered bf16      : flux-internal RS == torch
      reduce_scatter of the SAME bf16 partials, within the reduction-order
      noise floor. Only meaningful on the o_proj (GemmRS) joint.

The five hard-won requirements (each a scar; see task-6-brief.md):
  1. sglang TP-group init -- NOT bare torchrun. The T5 fp8 path reads the
     global get_tp_group()/get_tensor_model_parallel_world_size() singletons
     (fp8.py:882-903), and flux.init_flux_shm is driven INSIDE the sglang
     GroupCoordinator ctor, gated on SGLANG_USE_FUSED_OVERLAP=1
     (parallel_state.py:328-337). So we call init_distributed_environment +
     initialize_model_parallel(tensor_model_parallel_size=2) with that env
     set, exactly the construction the serving path uses -- a bare
     dist.init_process_group would leave _TP unset and flux_shm uninit'd
     (M3 stage-xii lesson).
  2. fp32 anchoring -- leg (iii), the truth both fp8 legs are scored against.
  3. noise-floor mirroring -- anchor (3)/(4) thresholds come from the
     reference run's own reorder-summation spread, not a magic constant.
  4. anchor placement -- one verdict at each of the four flow divergence
     points (post-quant / post-AG / post-GEMM / post-RS).
  5. .contiguous() self-care -- every low-level weight/payload VIEW handed to
     a collective or reinterpreted as bytes is forced contiguous first (the
     M3 harness hit a non-contiguous weight view 3x).

Output: an aligned per-anchor verdict table between ===M-START===/===M-END===,
then a final sentinel line: M2PROBE-OK (all anchors pass) or
M2PROBE-FAIL <anchor> (the first anchor that failed). Exit code is nonzero
iff any anchor failed; metrics are all-reduced so the exit code is
rank-consistent. Rank 0 prints.

NVSHMEM is NOT needed -- the dense GemmRS/AGKernel SM89 ops are cudaIPC
collectives (unlike the MoE AGScatter/GatherRS which are nvshmem); but the
flux env prerequisite the bf16 flux line needs
(SGLANG_USE_FUSED_OVERLAP=1 for the flux_shm construction gate) is required
and asserted at startup.

Shapes default to a Qwen3-32B-ish dense layer TP2 shard; override via env:
  SGLANG_FP8_PROBE_M       (default 512)  tokens (must be % tp == 0)
  SGLANG_FP8_PROBE_HIDDEN  (default 4096) model hidden = qkv K = o_proj N
  SGLANG_FP8_PROBE_QKV_N   (default 5120) qkv fused out (Q+K+V), full N
  SGLANG_FP8_PROBE_SEED    (default 1234)
"""

import os
import sys

# Must be set BEFORE any sglang.srt.distributed import: the flux_shm init in
# the GroupCoordinator ctor is gated on this (parallel_state.py:328-337), and
# fp8.py reads SGLANG_FLUX_FP8 at module import. Default them on so a bare
# `torchrun ... probe.py` still works; an explicit env still wins.
os.environ.setdefault("SGLANG_FLUX_FP8", "1")
os.environ.setdefault("SGLANG_USE_FUSED_OVERLAP", "1")

import torch
import torch.distributed as dist
import torch.nn.functional as F

SEED = int(os.environ.get("SGLANG_FP8_PROBE_SEED", "1234"))
M = int(os.environ.get("SGLANG_FP8_PROBE_M", "512"))
HIDDEN = int(os.environ.get("SGLANG_FP8_PROBE_HIDDEN", "4096"))
QKV_N = int(os.environ.get("SGLANG_FP8_PROBE_QKV_N", "5120"))
TP = 2
DTYPE = torch.bfloat16
EPS = 1e-8
# Anchor (3)/(4): flux must be within this multiple of the same-math torch
# reference's OWN distance to fp32 (plus the reorder noise floor). 1.1x is a
# small slack over the reference (both do the identical per-tensor math; flux
# only differs in kernel accumulation order / fused epilogue).
POSTGEMM_SLACK = 1.1


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a64, b64 = a.double().reshape(-1), b.double().reshape(-1)
    d = (a64 - b64).abs()
    cos = F.cosine_similarity(a64.unsqueeze(0), b64.unsqueeze(0)).item()
    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(), cos=cos)


def fp8_bytes(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret an fp8 tensor's payload as uint8 for a BITWISE compare.

    .contiguous() first (scar 5): a non-contiguous fp8 view would view_as the
    wrong bytes. E4M3 has no negative-zero surprise for a byte compare -- we
    compare the raw storage, so two payloads match iff every code point is
    identical.
    """
    return t.contiguous().view(torch.uint8)


def main() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    # ---- scar 1: initialize sglang's distributed env the SERVING way -------
    # init_distributed_environment builds torch's WORLD + sglang's _WORLD;
    # initialize_model_parallel builds the _TP GroupCoordinator, whose ctor
    # (parallel_state.py:328-337) calls flux.init_flux_shm(device_group)
    # BECAUSE SGLANG_USE_FUSED_OVERLAP=1. This is exactly how launch_server
    # brings the flux comm up -- a bare dist.init_process_group would leave
    # get_tp_group() unset (the T5 fp8 path calls it) and flux_shm uninit'd.
    from sglang.srt.distributed import (
        get_tensor_model_parallel_world_size,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=TP)

    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == TP, f"launch with --nproc_per_node={TP} (got world={world})"
    assert (
        get_tensor_model_parallel_world_size() == TP
    ), "sglang TP group not initialized to TP=2"
    assert (
        os.environ.get("SGLANG_USE_FUSED_OVERLAP") == "1"
    ), "SGLANG_USE_FUSED_OVERLAP=1 is required (flux_shm construction gate)"
    assert (
        os.environ.get("SGLANG_FLUX_FP8") == "1"
    ), "SGLANG_FLUX_FP8=1 is required (T5 fp8 path is gated on it)"
    assert M % TP == 0, f"M={M} must be divisible by tp={TP}"
    assert QKV_N % TP == 0, f"QKV_N={QKV_N} must be divisible by tp={TP}"
    assert HIDDEN % TP == 0, f"HIDDEN={HIDDEN} must be divisible by tp={TP}"

    tp_group = get_tp_group()

    # Import the REAL T5 pieces we drive (not reimplemented): the method whose
    # _forward_flux_fp8 / _flux_fp8_prepare_weight IS the flux leg, plus the
    # op builder and the exact quant kernel + fp8 dtype the path uses.
    from sglang.srt.layers.quantization.fp8 import (
        Fp8LinearMethod,
        _flux_fp8_get_op,
    )
    from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, scaled_fp8_quant

    fp8_max = torch.finfo(fp8_dtype).max

    def p0(*args):
        if rank == 0:
            print(*args, flush=True)

    def reduce_worst(m: dict) -> dict:
        t = torch.tensor(
            [m["max_abs"], m["mean_abs"], -m["cos"]], device=dev, dtype=torch.float64
        )
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return dict(max_abs=t[0].item(), mean_abs=t[1].item(), cos=-t[2].item())

    def reduce_max_int(v: int) -> int:
        t = torch.tensor([int(v)], device=dev, dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t.item())

    # ---- a method host that runs the REAL T5 methods -----------------------
    # Bind the four unbound Fp8LinearMethod methods onto a light host carrying
    # exactly the flags they read. This executes the genuine T5 code
    # (_forward_flux_fp8 -> _flux_fp8_prepare_weight -> op.forward) with a real
    # flux op -- not a reimplementation. s1's validation base is the non-block
    # runtime-fp8 weight path (design section 2: --quantization fp8 on a bf16
    # checkpoint), so block_quant=False.
    class MethodHost:
        use_marlin = False
        use_mxfp8 = False
        block_quant = False

    host = MethodHost()
    host._forward_flux_fp8 = Fp8LinearMethod._forward_flux_fp8.__get__(host)
    host._flux_fp8_prepare_weight = Fp8LinearMethod._flux_fp8_prepare_weight.__get__(
        host
    )

    # ---- a layer stub carrying the attributes the T5 path reads ------------
    class LayerStub:
        pass

    def make_layer(kind, n, k, weight_kn):
        """A layer whose weight is the runtime-fp8 (non-block) stored form.

        process_weights_after_loading leaves the fp8 weight TRANSPOSED as
        [K, N] with a per-tensor scalar scale (design section 2 s1 weight
        side). _flux_fp8_prepare_weight's else-branch reads exactly
        layer.weight.data ([K,N]) + layer.weight_scale.data (scalar), dequants
        to [N,K] fp32, and per-tensor requantizes for flux. We build that
        stored form here from a synthetic bf16 weight.
        """
        lyr = LayerStub()
        lyr.output_size_per_partition = n
        lyr.input_size_per_partition = k
        lyr.orig_dtype = DTYPE
        lyr._flux_fp8_fuse = kind
        lyr._flux_fp8_prefix = f"probe.{kind}"
        lyr._flux_fp8_weight = None
        lyr._flux_fp8_weight_scale = None
        # Runtime fp8 weight quant of the synthetic [N,K] bf16 weight, stored
        # transposed [K,N] with a per-tensor scalar (the create_weights form).
        w_q_nk, w_s = scaled_fp8_quant(weight_kn.to(torch.float32).contiguous())
        lyr.weight = torch.nn.Parameter(
            w_q_nk.t().contiguous(), requires_grad=False  # [K,N] stored view
        )
        lyr.weight_scale = torch.nn.Parameter(
            w_s.to(torch.float32).reshape(1), requires_grad=False
        )
        # Build the flux op now (also caches it on the layer).
        _flux_fp8_get_op(lyr, kind)
        return lyr, w_q_nk, w_s

    # ---- deterministic identical inputs on both ranks (CPU generator) ------
    g = torch.Generator().manual_seed(SEED)
    # Weights ~N(0, 1/fan_in) so activations stay O(1): the precision compare
    # is then in the realistic regime, not scale-dominated.
    # qkv: full activation x_full [M, HIDDEN]; each rank owns a scattered M/tp
    # row-shard; the AGKernel gathers to full M and multiplies by the local
    # column-shard of the qkv weight [QKV_N/tp, HIDDEN].
    x_full = torch.randn(M, HIDDEN, generator=g).to(dev, DTYPE)
    qkv_w_full = (
        torch.randn(QKV_N, HIDDEN, generator=g) * HIDDEN**-0.5
    ).to(dev, DTYPE)
    qkv_n_shard = QKV_N // TP
    qkv_w_shard = qkv_w_full[rank * qkv_n_shard : (rank + 1) * qkv_n_shard, :]
    # o_proj: activation is full M rows x K/tp column-shard; weight is
    # [HIDDEN, HIDDEN/tp] local column-shard; GemmRS sums partials over ranks.
    o_k_shard = HIDDEN // TP
    attn_out_full = torch.randn(M, HIDDEN, generator=g).to(dev, DTYPE)
    attn_out_shard = attn_out_full[:, rank * o_k_shard : (rank + 1) * o_k_shard]
    o_w_full = (torch.randn(HIDDEN, HIDDEN, generator=g) * HIDDEN**-0.5).to(dev, DTYPE)
    o_w_shard = o_w_full[:, rank * o_k_shard : (rank + 1) * o_k_shard]

    x_local = x_full.chunk(TP, dim=0)[rank].contiguous()  # scattered M/tp rows

    # ========================================================================
    # JOINT 1: qkv_proj  (flux AGKernel: AG fp8 payload + fp8 GEMM -> bf16)
    # ========================================================================
    qkv_layer, qkv_wq_nk, qkv_ws = make_layer(
        "ag_gemm", qkv_n_shard, HIDDEN, qkv_w_shard
    )

    # ---- FLUX leg (i): the real T5 path ------------------------------------
    qkv_flux = host._forward_flux_fp8(qkv_layer, x_local, None, "ag_gemm")

    # ---- REFERENCE leg (ii): identical per-tensor math in plain torch ------
    # Mirror fp8.py:897-905 EXACTLY: rank-uniform amax (MAX-allreduce) -> the
    # SAME scaled_fp8_quant static call -> torch all_gather of the fp8 payload
    # -> dequant matmul with the flux-requantized weight (read back off the
    # layer so the weight bytes are byte-identical to the flux leg).
    amax = x_local.contiguous().view(-1, HIDDEN).abs().amax().float().reshape(1)
    dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=tp_group.device_group)
    x_s = (amax / fp8_max).clamp(min=1e-12)
    x_q_local, _ = scaled_fp8_quant(x_local.view(-1, HIDDEN), x_s)
    # Anchor (2) prep: rank-contiguous all_gather of the fp8 payload.
    gathered = [torch.empty_like(x_q_local) for _ in range(TP)]
    dist.all_gather(gathered, x_q_local.contiguous(), group=tp_group.device_group)
    x_q_full = torch.cat(gathered, dim=0).contiguous()  # [M, HIDDEN] fp8
    w_flux = qkv_layer._flux_fp8_weight  # [N/tp, HIDDEN] fp8 (flux-requantized)
    w_s_flux = qkv_layer._flux_fp8_weight_scale
    qkv_ref = (
        x_s
        * w_s_flux
        * (x_q_full.to(torch.float32) @ w_flux.to(torch.float32).t())
    ).to(DTYPE)

    # ---- FP32 anchor (iii): no quant, torch fp32 mm on gathered full x -----
    x_full_f32 = x_full.float()
    w_shard_f32 = qkv_w_shard.float()
    qkv_fp32 = x_full_f32 @ w_shard_f32.t()

    # ---- noise floor: reference GEMM re-run with reordered K summation -----
    # Split K in half and add the two partials in the opposite grouping; the
    # spread vs the single-shot reference is the accumulation-order floor.
    kh = HIDDEN // 2
    qkv_fp32_reorder = (
        x_full_f32[:, kh:] @ w_shard_f32[:, kh:].t()
        + x_full_f32[:, :kh] @ w_shard_f32[:, :kh].t()
    )
    qkv_floor = reduce_worst(metrics(qkv_fp32_reorder, qkv_fp32))

    # ---- anchor (1): flux-leg quant == reference-leg quant, BITWISE --------
    # The flux leg's _forward_flux_fp8 (ag_gemm) computes its AG payload as
    # scaled_fp8_quant(x_2d, x_s) with x_s = (MAX-allreduced amax)/fp8_max
    # (fp8.py:897-905). The reference leg above (x_q_local / x_s) reproduces
    # that formula line-for-line, so it IS leg-(i)'s quantization. This anchor
    # confirms that reproduction is exact and stable in TWO independent ways:
    #   - scale: recompute x_s from scratch (fresh amax + allreduce) and demand
    #     it be BITWISE identical (the rank-uniform scale is the load-bearing
    #     agreement; a per-rank scale drift here would silently corrupt the AG
    #     dequant, scar-class bug);
    #   - payload: re-quantize with the recomputed scale and demand the fp8
    #     bytes be identical -- a contiguity/stride slip (scar 5) between the
    #     [M/tp,HIDDEN] view and its .contiguous() form would flip payload
    #     bytes even at a fixed scale.
    amax_r = x_local.contiguous().view(-1, HIDDEN).abs().amax().float().reshape(1)
    dist.all_reduce(amax_r, op=dist.ReduceOp.MAX, group=tp_group.device_group)
    x_s_recompute = (amax_r / fp8_max).clamp(min=1e-12)
    x_q_recompute, _ = scaled_fp8_quant(x_local.view(-1, HIDDEN), x_s_recompute)
    q1_bytes = int((fp8_bytes(x_q_recompute) != fp8_bytes(x_q_local)).sum().item())
    q1_scale = float((x_s_recompute - x_s).abs().max().item())
    # Reduce the local badness so the verdict/exit is rank-consistent (either
    # rank tripping fails both): 1 if this rank saw a byte or scale mismatch.
    a1_bad = reduce_max_int(1 if (q1_bytes != 0 or q1_scale != 0.0) else 0) != 0

    # ---- anchor (2): flux AG order == torch rank-contiguous order ----------
    # The AGKernel gathers the fp8 payload internally. Its output GEMM used the
    # rank-contiguous [shard0; shard1] gather order iff flux's result matches
    # the reference built on x_q_full (rank-contiguous). A rank-ROTATED layout
    # would show up as a large post-GEMM error that the ROTATED reference
    # explains better. We test the ordering hypothesis directly on the fp32
    # inputs (order-only, quant-free) so it is a clean permutation check.
    chunks = list(x_full_f32.chunk(TP, dim=0))
    rot = torch.cat([chunks[(rank + i) % TP] for i in range(TP)], dim=0)
    qkv_fp32_rot = rot @ w_shard_f32.t()
    m_orig = metrics(qkv_flux, qkv_fp32)["mean_abs"]
    m_rot = metrics(qkv_flux, qkv_fp32_rot)["mean_abs"]
    # rank0's rotation is the identity, so only ranks>0 discriminate; gather
    # both so rank0 can decide.
    origs = [torch.zeros(1, device=dev, dtype=torch.float64) for _ in range(TP)]
    rots = [torch.zeros(1, device=dev, dtype=torch.float64) for _ in range(TP)]
    dist.all_gather(origs, torch.tensor([m_orig], device=dev, dtype=torch.float64))
    dist.all_gather(rots, torch.tensor([m_rot], device=dev, dtype=torch.float64))
    ao = [o.item() for o in origs]
    ar = [r.item() for r in rots]
    # order is WRONG if any rank>0 matches the rotated layout markedly better.
    a2_bad = any(ar[r] < ao[r] / 10.0 for r in range(1, TP))

    # ---- anchor (3): |flux - fp32| <= slack * |ref - fp32| + floor ---------
    m_qkv_flux_fp32 = reduce_worst(metrics(qkv_flux, qkv_fp32))
    m_qkv_ref_fp32 = reduce_worst(metrics(qkv_ref, qkv_fp32))
    m_qkv_flux_ref = reduce_worst(metrics(qkv_flux, qkv_ref))
    qkv_budget = POSTGEMM_SLACK * m_qkv_ref_fp32["mean_abs"] + qkv_floor["mean_abs"]
    a3_qkv_bad = m_qkv_flux_fp32["mean_abs"] > qkv_budget

    # ========================================================================
    # JOINT 2: o_proj  (flux GemmRS: fp8 GEMM -> bf16 partials + RS)
    # ========================================================================
    o_layer, o_wq_nk, o_ws = make_layer("gemm_rs", HIDDEN, o_k_shard, o_w_shard)

    # ---- FLUX leg (i): the real T5 path ------------------------------------
    o_flux = host._forward_flux_fp8(o_layer, attn_out_shard, None, "gemm_rs")

    # ---- REFERENCE leg (ii): identical per-tensor math in plain torch ------
    # Mirror fp8.py:882-888: LOCAL per-tensor dynamic scale -> the SAME
    # scaled_fp8_quant -> dequant matmul (each rank's partial is fully
    # dequantized: input_scale * weight_scale * (A_q @ W_q.T)) -> torch
    # reduce_scatter of the bf16 partials (RS stays bf16 -- numeric-gate law).
    a_local = attn_out_shard.contiguous().view(-1, o_k_shard)
    a_q, a_s = scaled_fp8_quant(a_local)  # dynamic per-tensor, LOCAL scale
    ow_flux = o_layer._flux_fp8_weight  # [HIDDEN, K/tp] fp8
    ow_s_flux = o_layer._flux_fp8_weight_scale
    o_partial = (
        a_s * ow_s_flux * (a_q.to(torch.float32) @ ow_flux.to(torch.float32).t())
    ).to(DTYPE)  # [M, HIDDEN] this rank's partial, bf16
    o_scattered_ref = o_partial.new_empty((M // TP, HIDDEN))
    dist.reduce_scatter_tensor(
        o_scattered_ref, o_partial.contiguous(), group=tp_group.device_group
    )

    # ---- FP32 anchor (iii): fp32 partial + fp32 reduce_scatter -------------
    a_local_f32 = attn_out_shard.float()
    ow_shard_f32 = o_w_shard.float()
    o_partial_fp32 = a_local_f32 @ ow_shard_f32.t()  # [M, HIDDEN] fp32 partial
    o_scattered_fp32 = o_partial_fp32.new_empty((M // TP, HIDDEN))
    dist.reduce_scatter_tensor(
        o_scattered_fp32, o_partial_fp32.contiguous(), group=tp_group.device_group
    )

    # ---- noise floor: reduction re-run with reordered K summation ----------
    kh2 = o_k_shard // 2
    o_partial_fp32_reorder = (
        a_local_f32[:, kh2:] @ ow_shard_f32[:, kh2:].t()
        + a_local_f32[:, :kh2] @ ow_shard_f32[:, :kh2].t()
    )
    o_scat_fp32_reorder = o_partial_fp32_reorder.new_empty((M // TP, HIDDEN))
    dist.reduce_scatter_tensor(
        o_scat_fp32_reorder,
        o_partial_fp32_reorder.contiguous(),
        group=tp_group.device_group,
    )
    o_floor = reduce_worst(metrics(o_scat_fp32_reorder, o_scattered_fp32))

    # ---- anchor (4): flux RS == torch reduce_scatter, within floor ---------
    # Both the flux GemmRS and the reference reduce the SAME per-rank bf16
    # partials; the difference must sit within the reduction-order noise floor.
    m_o_flux_ref = reduce_worst(metrics(o_flux, o_scattered_ref))
    a4_bad = m_o_flux_ref["mean_abs"] > POSTGEMM_SLACK * o_floor["mean_abs"] + EPS

    # ---- anchor (3) for o_proj: |flux - fp32| <= slack*|ref - fp32| + floor -
    m_o_flux_fp32 = reduce_worst(metrics(o_flux, o_scattered_fp32))
    m_o_ref_fp32 = reduce_worst(metrics(o_scattered_ref, o_scattered_fp32))
    o_budget = POSTGEMM_SLACK * m_o_ref_fp32["mean_abs"] + o_floor["mean_abs"]
    a3_o_bad = m_o_flux_fp32["mean_abs"] > o_budget

    # ========================================================================
    # report
    # ========================================================================
    def fmt(name, m):
        return (
            f"{name:<52s} max_abs={m['max_abs']:.6e} "
            f"mean_abs={m['mean_abs']:.6e} cos={m['cos']:.9f}"
        )

    def verdict_line(name, bad):
        return f"{name:<52s} {'FAIL' if bad else 'PASS'}"

    p0("=" * 100)
    p0(
        f"probe config: M={M} hidden={HIDDEN} qkv_N={QKV_N} tp={TP} "
        f"dtype={DTYPE} seed={SEED} fp8_dtype={fp8_dtype} fp8_max={fp8_max} "
        f"postgemm_slack={POSTGEMM_SLACK}"
    )
    p0(f"qkv AGKernel op:  {getattr(qkv_layer._flux_fp8_op, 'ctor_desc', '?')}")
    p0(f"o_proj GemmRS op: {getattr(o_layer._flux_fp8_op, 'ctor_desc', '?')}")
    p0("-" * 100)
    p0("[JOINT 1] qkv_proj  (AGKernel: AG fp8 payload + fp8 GEMM)")
    p0(fmt("  noise floor (fp32 reorder vs fp32)", qkv_floor))
    p0(fmt("  (i)  flux  vs fp32", m_qkv_flux_fp32))
    p0(fmt("  (ii) ref   vs fp32", m_qkv_ref_fp32))
    p0(fmt("  (i)  flux  vs (ii) ref", m_qkv_flux_ref))
    p0(
        f"  {'anchor1 quant bitwise: payload_byte_mismatch':<50s} "
        f"={q1_bytes} scale_abs_diff={q1_scale:.3e}"
    )
    p0(f"  anchor2 AG order: per-rank orig={ao} rotated={ar}")
    p0(f"  anchor3 budget (post-GEMM) = {qkv_budget:.6e}")
    p0("-" * 100)
    p0("[JOINT 2] o_proj  (GemmRS: fp8 GEMM -> bf16 partials + RS)")
    p0(fmt("  noise floor (fp32 reorder vs fp32, scattered)", o_floor))
    p0(fmt("  (i)  flux  vs fp32 (scattered)", m_o_flux_fp32))
    p0(fmt("  (ii) ref   vs fp32 (scattered)", m_o_ref_fp32))
    p0(fmt("  (i)  flux  vs (ii) ref (scattered)", m_o_flux_ref))
    p0(f"  anchor3 budget (post-GEMM) = {o_budget:.6e}")
    p0(f"  anchor4 budget (post-RS)   = {POSTGEMM_SLACK * o_floor['mean_abs'] + EPS:.6e}")
    p0("-" * 100)
    p0("VERDICT TABLE (per anchor, worst across ranks):")
    p0(verdict_line("  anchor1  post-quant  fp8 payload+scale bitwise", a1_bad))
    p0(verdict_line("  anchor2  post-AG     gathered fp8 order", a2_bad))
    p0(
        verdict_line(
            "  anchor3  post-GEMM   |flux-fp32| <= 1.1x|ref-fp32|",
            a3_qkv_bad or a3_o_bad,
        )
    )
    p0(verdict_line("  anchor4  post-RS     flux RS == torch RS", a4_bad))
    p0("=" * 100)

    # First failing anchor (ordered) drives the sentinel.
    first_fail = None
    if a1_bad:
        first_fail = "anchor1-post-quant"
    elif a2_bad:
        first_fail = "anchor2-post-AG"
    elif a3_qkv_bad or a3_o_bad:
        which = "qkv" if a3_qkv_bad else "o_proj"
        first_fail = f"anchor3-post-GEMM-{which}"
    elif a4_bad:
        first_fail = "anchor4-post-RS"

    if first_fail is None:
        p0("M2PROBE-OK")
    else:
        p0(f"M2PROBE-FAIL {first_fail}")

    dist.barrier()
    dist.destroy_process_group()
    return 0 if first_fail is None else 1


if __name__ == "__main__":
    sys.exit(main())
