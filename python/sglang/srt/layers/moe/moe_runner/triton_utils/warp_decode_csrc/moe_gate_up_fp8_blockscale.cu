// warp_decode: gate_up kernel for FP8-E4M3 block-scale MoE decode (DeepSeek-style
// [128,128] weight blocks, e.g. DeepSeek-V4-Flash w8a8-block-fp8 checkpoints).
//
// Differences vs the BF16 kernel (moe_gate_up_3d_batched.cu):
//   * W is fp8 e4m3, dequantized to float in registers (weight * block_scale).
//   * Activations stay BF16 end-to-end -- no activation quantization needed.
//   * VEC is 16 fp8 per uint4 load (vs 8 bf16), so hidden must be % 512.
//
// Layout:
//   X   : [B, hidden]                    bf16
//   W   : [E, 2*inter, hidden]           fp8 e4m3  (rows [0,inter)=gate, [inter,2i)=up)
//   Ws  : [E, ceil(2*inter/128), ceil(hidden/128)]  fp32 block scales (weight_scale_inv)
//   ids : [B, topk]                      int32
//   out : [B * topk, inter]              bf16      (== silu(gate) * up, fused)
//
// Scale handling: each lane's 16-element chunk lies inside exactly one
// 128-wide k-block (16 | 128 and chunk starts are 16-aligned), so the lane
// accumulates a per-chunk partial dot product and folds the block scale in
// once per chunk:  acc += scale[n_block][k_block] * partial.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace warp_decode {

constexpr int GU8_WARPS_PER_CTA   = 8;
constexpr int GU8_THREADS_PER_CTA = GU8_WARPS_PER_CTA * 32;
constexpr int GU8_VEC             = 16;  // 16 fp8 == 16 B == uint4
constexpr int GU8_XVEC            = 8;   // 8 bf16 == 16 B == uint4

__device__ __forceinline__ float gu8_silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float gu8_warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

__global__ void moe_gate_up_fp8_blockscale_kernel(
    const __nv_bfloat16* __restrict__ X,        // [B, hidden]
    const uint8_t*       __restrict__ W,        // [E, 2*inter, hidden] fp8 e4m3
    const float*         __restrict__ Ws,       // [E, RS, CS]
    const int*           __restrict__ topk_ids, // [B, topk]
    __nv_bfloat16*       __restrict__ out,      // [B*topk, inter]
    int B, int E, int topk, int hidden, int inter,
    int RS, int CS,                             // scale rows/cols per expert
    float swiglu_limit)                         // +INF disables the clamp
{
    const int flat_tok = blockIdx.x;                 // in [0, B*topk)
    const int token_id = flat_tok / topk;
    const int slot     = flat_tok - token_id * topk;

    const int warp_id  = threadIdx.x >> 5;
    const int lane_id  = threadIdx.x & 31;

    const int neuron_n = blockIdx.y * GU8_WARPS_PER_CTA + warp_id;
    if (neuron_n >= inter) return;

    const int expert_id = topk_ids[token_id * topk + slot];

    // ---- 1. Stage x[token] into shared memory (shared across all 8 warps) ----
    extern __shared__ __nv_bfloat16 smem_x[];  // hidden bf16 elements
    const __nv_bfloat16* __restrict__ x_ptr = X + (size_t)token_id * hidden;

    #pragma unroll 1
    for (int i = threadIdx.x * GU8_XVEC; i < hidden; i += GU8_THREADS_PER_CTA * GU8_XVEC) {
        *reinterpret_cast<uint4*>(&smem_x[i]) =
            *reinterpret_cast<const uint4*>(&x_ptr[i]);
    }
    __syncthreads();

    // ---- 2. Warp streams its two fp8 weight rows, dequant + dot with x -------
    const size_t expert_stride = (size_t)(2 * inter) * hidden;
    const uint8_t* __restrict__ gate_ptr =
        W + (size_t)expert_id * expert_stride + (size_t)neuron_n * hidden;
    const uint8_t* __restrict__ up_ptr =
        gate_ptr + (size_t)inter * hidden;

    // Scale rows for this neuron's gate row (n) and up row (n + inter).
    const float* __restrict__ sg_row =
        Ws + ((size_t)expert_id * RS + (neuron_n >> 7)) * CS;
    const float* __restrict__ su_row =
        Ws + ((size_t)expert_id * RS + ((neuron_n + inter) >> 7)) * CS;

    float acc_gate = 0.f;
    float acc_up   = 0.f;

    #pragma unroll 1
    for (int k = lane_id * GU8_VEC; k < hidden; k += 32 * GU8_VEC) {
        uint4 gv = *reinterpret_cast<const uint4*>(&gate_ptr[k]);
        uint4 uv = *reinterpret_cast<const uint4*>(&up_ptr[k]);
        uint4 xv0 = *reinterpret_cast<const uint4*>(&smem_x[k]);
        uint4 xv1 = *reinterpret_cast<const uint4*>(&smem_x[k + GU8_XVEC]);

        const __nv_fp8x2_storage_t* g2 =
            reinterpret_cast<const __nv_fp8x2_storage_t*>(&gv);
        const __nv_fp8x2_storage_t* u2 =
            reinterpret_cast<const __nv_fp8x2_storage_t*>(&uv);
        const __nv_bfloat16* xh0 = reinterpret_cast<const __nv_bfloat16*>(&xv0);
        const __nv_bfloat16* xh1 = reinterpret_cast<const __nv_bfloat16*>(&xv1);

        float part_gate = 0.f;
        float part_up   = 0.f;

        #pragma unroll
        for (int i = 0; i < GU8_VEC / 2; ++i) {
            float2 gf = __half22float2(
                __half2(__nv_cvt_fp8x2_to_halfraw2(g2[i], __NV_E4M3)));
            float2 uf = __half22float2(
                __half2(__nv_cvt_fp8x2_to_halfraw2(u2[i], __NV_E4M3)));
            const __nv_bfloat16* xh = (2 * i < GU8_XVEC) ? xh0 : xh1;
            const int base = (2 * i) & (GU8_XVEC - 1);
            float x0 = __bfloat162float(xh[base]);
            float x1 = __bfloat162float(xh[base + 1]);
            part_gate += x0 * gf.x + x1 * gf.y;
            part_up   += x0 * uf.x + x1 * uf.y;
        }

        const int kb = k >> 7;  // k is 16-aligned; chunk lies in one 128-block
        acc_gate += sg_row[kb] * part_gate;
        acc_up   += su_row[kb] * part_up;
    }

    acc_gate = gu8_warp_reduce_sum(acc_gate);
    acc_up   = gu8_warp_reduce_sum(acc_up);

    if (lane_id == 0) {
        // DeepSeek V4 swiglu clamp (gate: upper only; up: symmetric).
        // swiglu_limit == +INF makes both ops identity.
        acc_gate = fminf(acc_gate, swiglu_limit);
        acc_up   = fminf(fmaxf(acc_up, -swiglu_limit), swiglu_limit);
        float y = gu8_silu_f32(acc_gate) * acc_up;
        out[(size_t)flat_tok * inter + neuron_n] = __float2bfloat16(y);
    }
}

void launch_moe_gate_up_fp8_blockscale(
    const __nv_bfloat16* X,
    const uint8_t*       W,
    const float*         Ws,
    const int*           topk_ids,
    __nv_bfloat16*       out,
    int B, int E, int topk, int hidden, int inter,
    int RS, int CS,
    cudaStream_t stream,
    float swiglu_limit)
{
    dim3 grid(B * topk, (inter + GU8_WARPS_PER_CTA - 1) / GU8_WARPS_PER_CTA);
    dim3 block(GU8_THREADS_PER_CTA);
    size_t smem_bytes = (size_t)hidden * sizeof(__nv_bfloat16);

    moe_gate_up_fp8_blockscale_kernel<<<grid, block, smem_bytes, stream>>>(
        X, W, Ws, topk_ids, out, B, E, topk, hidden, inter, RS, CS,
        swiglu_limit);
}

}  // namespace warp_decode
