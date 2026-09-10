# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""F28 first-layer mHC broadcast, adapted from commit 0d01ff31997b."""

import math

# This module is imported only inside the enabled runtime path. Import the real
# DSL here: copying mhc.T would retain its stale lazy proxy after mhc loads it.
import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.layernorm.mhc import _compute_num_split_for_mhc_pre


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_broadcast_with_norm_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    residual_out,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 1,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
):
    ENABLE_PDL = False  # SM89 has no programmatic dependent launch.
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    residual_out: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0

        if ENABLE_PDL:
            T.pdl_sync()

        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        # rms scale
        rms[0] = T.rsqrt(rms[0] / hidden_size + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )

            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared(hidden_block, T.bfloat16)
                xl = T.alloc_fragment(hidden_block, T.float32)
                T.copy(residual[i, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                for i_hc, i1_h in T.Parallel(hc_mult, hidden_block):
                    residual_out[i, i_hc, i0_h * hidden_block + i1_h] = xs[i1_h]

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i1_h]

                for i1_h in T.Parallel(hidden_block):
                    sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                    output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                w_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(norm_weight[i0_h * hidden_block], w_shared)
                T.copy(w_shared, w_local)

                ol = T.alloc_fragment(hidden_block, T.float32)
                for i1_h in T.Parallel(hidden_block):
                    ol[i1_h] = (
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()


def mhc_pre_broadcast(
    residual,
    fn_broadcast,
    hc_scale,
    hc_base,
    rms_eps,
    hc_eps,
    post_mult,
    sinkhorn_repeat,
    norm_weight,
    norm_eps,
):
    """Produce completed contiguous residual and pre output from a 2D embedding."""
    if torch.cuda.get_device_capability(residual.device) != (8, 9):
        raise RuntimeError("F28 broadcast adapter requires SM89")
    assert residual.ndim == 2 and residual.dtype == torch.bfloat16
    tokens, hidden = residual.shape
    assert fn_broadcast.shape == (24, hidden) and fn_broadcast.dtype == torch.float32
    residual_out = torch.empty(
        tokens, 4, hidden, device=residual.device, dtype=residual.dtype
    )
    post = torch.empty(tokens, 4, device=residual.device, dtype=torch.float32)
    comb = torch.empty(tokens, 16, device=residual.device, dtype=torch.float32)
    output = torch.empty_like(residual)
    if tokens:
        n_splits = _compute_num_split_for_mhc_pre(tokens, hidden)
        mul = torch.empty(
            n_splits, tokens, 24, device=residual.device, dtype=torch.float32
        )
        sqr = torch.empty(n_splits, tokens, device=residual.device, dtype=torch.float32)
        tf32_hc_prenorm_gemm_triton(residual, fn_broadcast, mul, sqr, n_splits)
        mhc_pre_big_fuse_broadcast_with_norm_tilelang(
            mul,
            sqr,
            hc_scale,
            hc_base,
            residual,
            residual_out,
            post,
            comb,
            output,
            norm_weight,
            hidden,
            rms_eps,
            hc_eps,
            hc_eps,
            post_mult,
            sinkhorn_repeat,
            norm_eps,
            n_splits=n_splits,
        )
    return residual_out, post.unsqueeze(-1), comb.view(tokens, 4, 4), output


# F28 masked SM89 GEMM fallback, from the same fixed source revision.
@triton.jit
def _tf32_hc_prenorm_gemm_kernel(
    x_ptr,
    fn_ptr,
    out_ptr,
    sqrsum_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_fnn: tl.constexpr,
    stride_fnk: tl.constexpr,
    stride_outs,
    stride_outm: tl.constexpr,
    stride_outn: tl.constexpr,
    stride_sqs,
    stride_sqm: tl.constexpr,
    NUM_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    split_k = tl.cdiv(K, NUM_SPLIT)
    split_begin = pid_s * split_k
    split_end = tl.minimum(split_begin + split_k, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k0 in tl.range(0, split_k, BLOCK_K):
        k = split_begin + k0 + offs_k
        k_mask = k < split_end
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        fn = tl.load(
            fn_ptr + offs_n[None, :] * stride_fnn + k[:, None] * stride_fnk,
            mask=(offs_n[None, :] < N) & k_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(x, fn, input_precision="tf32", out_dtype=tl.float32)
        sq += tl.sum(x * x, axis=1)

    tl.store(
        out_ptr
        + pid_s * stride_outs
        + offs_m[:, None] * stride_outm
        + offs_n[None, :] * stride_outn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

    if pid_n == 0:
        tl.store(
            sqrsum_ptr + pid_s * stride_sqs + offs_m * stride_sqm,
            sq,
            mask=offs_m < M,
        )


def tf32_hc_prenorm_gemm_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> None:
    assert x.dim() == 2
    assert fn.dim() == 2
    assert out.dim() == 3
    assert sqrsum.dim() == 2

    m, k = x.shape
    n = fn.shape[0]
    assert fn.shape[1] == k
    assert out.shape == (num_split, m, n)
    assert sqrsum.shape == (num_split, m)

    if m == 0:
        return

    block_m = 16
    block_n = triton.next_power_of_2(n)
    block_n = min(max(block_n, 16), 32)
    block_k = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), num_split)
    _tf32_hc_prenorm_gemm_kernel[grid](
        x,
        fn,
        out,
        sqrsum,
        m,
        k,
        n,
        x.stride(0),
        x.stride(1),
        fn.stride(0),
        fn.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sqrsum.stride(0),
        sqrsum.stride(1),
        num_split,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
