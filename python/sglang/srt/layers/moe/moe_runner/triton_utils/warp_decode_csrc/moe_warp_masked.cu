// warp_decode: masked-layout kernels for the DeepEP low-latency decode path.
//
// Replaces (DeepGEMM masked GEMM0 -> varlen silu+quant -> masked GEMM1) with
// two warp-decode kernels operating directly on the LL dispatch layout:
//
//   x      : [E, max_m, hidden]              fp8 e4m3 (LL recv buffer)
//   xs     : [E, max_m, ceil(hidden/128)]    fp32 per-token-group scales
//   w13    : [E, 2*inter, hidden]            fp8 e4m3 (+ [E,R,C] block scales)
//   w2     : [E, hidden, inter]              fp8 e4m3 (+ [E,R,C] block scales)
//   buf    : [E, max_m, inter]               bf16 (intermediate, silu fused)
//   out    : [E, max_m, hidden]              bf16 (LL combine applies topk
//                                            weights afterwards, so no topk
//                                            fold here)
//   masked_m : [E] int32 (valid token count per local expert)
//
// Work is compacted into (expert, token) pairs by a tiny single-CTA kernel;
// the compute kernels grid-stride over the pair list so their grids stay
// static for CUDA-graph capture regardless of the dynamic masked_m values.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace warp_decode {

// ---------------------------------------------------------------------------
// pair compaction: masked_m -> pairs[0]=count, pairs[1+i] = (e << 16) | m
// ---------------------------------------------------------------------------

__global__ void compact_pairs_kernel(
    const int* __restrict__ masked_m,   // [E]
    int*       __restrict__ pairs,      // [1 + E*max_m]
    int E, int max_m)
{
    __shared__ int offsets[1024];
    const int tid = threadIdx.x;

    int cnt = 0;
    if (tid < E) {
        cnt = masked_m[tid];
        if (cnt < 0) cnt = 0;
        if (cnt > max_m) cnt = max_m;
    }
    offsets[tid] = cnt;
    __syncthreads();

    // simple exclusive scan over E entries by thread 0 (E <= 1024, decode-tiny)
    __shared__ int total;
    if (tid == 0) {
        int run = 0;
        for (int e = 0; e < E; ++e) {
            int c = offsets[e];
            offsets[e] = run;
            run += c;
        }
        total = run;
        pairs[0] = run;
    }
    __syncthreads();

    if (tid < E) {
        int base = offsets[tid];
        for (int m = 0; m < cnt; ++m) {
            pairs[1 + base + m] = (tid << 16) | m;
        }
    }
    (void)total;
}

void launch_compact_pairs(
    const int* masked_m, int* pairs, int E, int max_m, cudaStream_t stream)
{
    compact_pairs_kernel<<<1, 1024, 0, stream>>>(masked_m, pairs, E, max_m);
}

// ---------------------------------------------------------------------------
// gate_up masked: fp8 x (per-token-group scales) x fp8 w13 -> bf16 buf
// ---------------------------------------------------------------------------

constexpr int GUM_WARPS_PER_CTA   = 8;
constexpr int GUM_THREADS_PER_CTA = GUM_WARPS_PER_CTA * 32;
constexpr int GUM_VEC             = 16;  // 16 fp8 == uint4

__device__ __forceinline__ float gum_silu_f32(float v) {
    return v / (1.0f + __expf(-v));
}

__device__ __forceinline__ float gum_warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

__global__ void moe_gate_up_masked_fp8_kernel(
    const uint8_t* __restrict__ X,        // [E, max_m, hidden] fp8
    const float*   __restrict__ Xs,       // [E, max_m, CSX]
    const uint8_t* __restrict__ W,        // [E, 2*inter, hidden] fp8
    const float*   __restrict__ Ws,       // [E, RS, CS]
    const int*     __restrict__ pairs,    // [1 + cap]
    __nv_bfloat16* __restrict__ buf,      // [E, max_m, inter]
    int E, int max_m, int hidden, int inter,
    int RS, int CS, int CSX,
    float swiglu_limit)
{
    extern __shared__ __nv_bfloat16 smem_x[];  // hidden bf16

    const int warp_id  = threadIdx.x >> 5;
    const int lane_id  = threadIdx.x & 31;
    const int neuron_n = blockIdx.y * GUM_WARPS_PER_CTA + warp_id;
    const int count    = pairs[0];

    for (int p = blockIdx.x; p < count; p += gridDim.x) {
        const int packed = pairs[1 + p];
        const int e = packed >> 16;
        const int m = packed & 0xFFFF;

        // ---- stage + dequantize x[e, m] into bf16 smem ----
        const uint8_t* __restrict__ x_ptr =
            X + ((size_t)e * max_m + m) * hidden;
        const float* __restrict__ xs_row =
            Xs + ((size_t)e * max_m + m) * CSX;

        for (int i = threadIdx.x * GUM_VEC; i < hidden;
             i += GUM_THREADS_PER_CTA * GUM_VEC) {
            uint4 xv = *reinterpret_cast<const uint4*>(&x_ptr[i]);
            const float sx = xs_row[i >> 7];
            const __nv_fp8x2_storage_t* x2 =
                reinterpret_cast<const __nv_fp8x2_storage_t*>(&xv);
            #pragma unroll
            for (int j = 0; j < GUM_VEC / 2; ++j) {
                float2 xf = __half22float2(
                    __half2(__nv_cvt_fp8x2_to_halfraw2(x2[j], __NV_E4M3)));
                smem_x[i + 2 * j]     = __float2bfloat16(sx * xf.x);
                smem_x[i + 2 * j + 1] = __float2bfloat16(sx * xf.y);
            }
        }
        __syncthreads();

        if (neuron_n < inter) {
            const size_t expert_stride = (size_t)(2 * inter) * hidden;
            const uint8_t* __restrict__ gate_ptr =
                W + (size_t)e * expert_stride + (size_t)neuron_n * hidden;
            const uint8_t* __restrict__ up_ptr = gate_ptr + (size_t)inter * hidden;

            const float* __restrict__ sg_row =
                Ws + ((size_t)e * RS + (neuron_n >> 7)) * CS;
            const float* __restrict__ su_row =
                Ws + ((size_t)e * RS + ((neuron_n + inter) >> 7)) * CS;

            float acc_gate = 0.f;
            float acc_up   = 0.f;

            #pragma unroll 1
            for (int k = lane_id * GUM_VEC; k < hidden; k += 32 * GUM_VEC) {
                uint4 gv  = *reinterpret_cast<const uint4*>(&gate_ptr[k]);
                uint4 uv  = *reinterpret_cast<const uint4*>(&up_ptr[k]);
                uint4 xv0 = *reinterpret_cast<const uint4*>(&smem_x[k]);
                uint4 xv1 = *reinterpret_cast<const uint4*>(&smem_x[k + 8]);

                const __nv_fp8x2_storage_t* g2 =
                    reinterpret_cast<const __nv_fp8x2_storage_t*>(&gv);
                const __nv_fp8x2_storage_t* u2 =
                    reinterpret_cast<const __nv_fp8x2_storage_t*>(&uv);
                const __nv_bfloat16* xh0 =
                    reinterpret_cast<const __nv_bfloat16*>(&xv0);
                const __nv_bfloat16* xh1 =
                    reinterpret_cast<const __nv_bfloat16*>(&xv1);

                float part_gate = 0.f;
                float part_up   = 0.f;

                #pragma unroll
                for (int i = 0; i < GUM_VEC / 2; ++i) {
                    float2 gf = __half22float2(
                        __half2(__nv_cvt_fp8x2_to_halfraw2(g2[i], __NV_E4M3)));
                    float2 uf = __half22float2(
                        __half2(__nv_cvt_fp8x2_to_halfraw2(u2[i], __NV_E4M3)));
                    const __nv_bfloat16* xh = (2 * i < 8) ? xh0 : xh1;
                    const int base = (2 * i) & 7;
                    float x0 = __bfloat162float(xh[base]);
                    float x1 = __bfloat162float(xh[base + 1]);
                    part_gate += x0 * gf.x + x1 * gf.y;
                    part_up   += x0 * uf.x + x1 * uf.y;
                }

                const int kb = k >> 7;
                acc_gate += sg_row[kb] * part_gate;
                acc_up   += su_row[kb] * part_up;
            }

            acc_gate = gum_warp_reduce_sum(acc_gate);
            acc_up   = gum_warp_reduce_sum(acc_up);

            if (lane_id == 0) {
                acc_gate = fminf(acc_gate, swiglu_limit);
                acc_up   = fminf(fmaxf(acc_up, -swiglu_limit), swiglu_limit);
                buf[((size_t)e * max_m + m) * inter + neuron_n] =
                    __float2bfloat16(gum_silu_f32(acc_gate) * acc_up);
            }
        }
        __syncthreads();  // smem reused by the next pair
    }
}

void launch_moe_gate_up_masked_fp8(
    const uint8_t* X, const float* Xs,
    const uint8_t* W, const float* Ws,
    const int* pairs,
    __nv_bfloat16* buf,
    int E, int max_m, int hidden, int inter,
    int RS, int CS, int CSX,
    float swiglu_limit,
    int pair_slots,
    cudaStream_t stream)
{
    dim3 grid(pair_slots, (inter + GUM_WARPS_PER_CTA - 1) / GUM_WARPS_PER_CTA);
    dim3 block(GUM_THREADS_PER_CTA);
    size_t smem_bytes = (size_t)hidden * sizeof(__nv_bfloat16);
    moe_gate_up_masked_fp8_kernel<<<grid, block, smem_bytes, stream>>>(
        X, Xs, W, Ws, pairs, buf, E, max_m, hidden, inter, RS, CS, CSX,
        swiglu_limit);
}

// ---------------------------------------------------------------------------
// down masked: bf16 buf x fp8 w2 -> bf16 out (no topk fold; combine does it)
// ---------------------------------------------------------------------------

constexpr int DM_WARP_SIZE       = 32;
constexpr int DM_WARPS_PER_CTA   = 8;
constexpr int DM_THREADS_PER_CTA = DM_WARPS_PER_CTA * DM_WARP_SIZE;
constexpr int DM_OUTS            = 4;   // measured sweet spot with OCT loads

__device__ __forceinline__ float dm_warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

__global__ void moe_down_masked_fp8_kernel(
    const __nv_bfloat16* __restrict__ buf,     // [E, max_m, inter]
    const uint8_t*       __restrict__ W,       // [E, hidden, inter] fp8
    const float*         __restrict__ Ws,      // [E, RS, CS]
    const int*           __restrict__ pairs,   // [1 + cap]
    __nv_bfloat16*       __restrict__ out,     // [E, max_m, hidden]
    int E, int max_m, int hidden, int inter,
    int RS, int CS,
    int pair_slots)
{
    const size_t global_tid  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t global_warp = global_tid / DM_WARP_SIZE;
    const int    lane_id     = threadIdx.x & (DM_WARP_SIZE - 1);

    const int groups_per_token = hidden / DM_OUTS;
    const int slot      = (int)(global_warp / groups_per_token);
    const int group_idx = (int)(global_warp - (size_t)slot * groups_per_token);
    if (slot >= pair_slots) return;
    const int h_base = group_idx * DM_OUTS;
    const int count  = pairs[0];
    const int oblocks = inter >> 8;

    for (int p = slot; p < count; p += pair_slots) {
        const int packed = pairs[1 + p];
        const int e = packed >> 16;
        const int m = packed & 0xFFFF;

        const __nv_bfloat16* __restrict__ in_ptr =
            buf + ((size_t)e * max_m + m) * inter;
        const uint8_t* w_rows =
            W + (size_t)e * hidden * inter + (size_t)h_base * inter;
        const float* __restrict__ ws_row =
            Ws + ((size_t)e * RS + (h_base >> 7)) * CS;

        float acc[DM_OUTS];
        #pragma unroll
        for (int j = 0; j < DM_OUTS; ++j) acc[j] = 0.f;

        #pragma unroll 1
        for (int ob = 0; ob < oblocks; ++ob) {
            const int e0 = ob * 256 + lane_id * 8;

            uint2 wq[DM_OUTS];
            #pragma unroll
            for (int j = 0; j < DM_OUTS; ++j) {
                wq[j] = *reinterpret_cast<const uint2*>(
                    &w_rows[(size_t)j * inter + e0]);
            }
            const uint4 xq = *reinterpret_cast<const uint4*>(&in_ptr[e0]);
            const float s_blk = ws_row[e0 >> 7];

            const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xq);
            float xf[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) xf[i] = __bfloat162float(xh[i]);

            #pragma unroll
            for (int j = 0; j < DM_OUTS; ++j) {
                const __nv_fp8x2_storage_t* w2 =
                    reinterpret_cast<const __nv_fp8x2_storage_t*>(&wq[j]);
                float oct = 0.f;
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float2 wf = __half22float2(
                        __half2(__nv_cvt_fp8x2_to_halfraw2(w2[q], __NV_E4M3)));
                    oct += xf[2 * q] * wf.x + xf[2 * q + 1] * wf.y;
                }
                acc[j] += s_blk * oct;
            }
        }

        #pragma unroll
        for (int j = 0; j < DM_OUTS; ++j) acc[j] = dm_warp_reduce_sum(acc[j]);

        if (lane_id == 0) {
            __nv_bfloat16* out_ptr =
                out + ((size_t)e * max_m + m) * hidden + h_base;
            #pragma unroll
            for (int j = 0; j < DM_OUTS; ++j) {
                out_ptr[j] = __float2bfloat16(acc[j]);
            }
        }

        #pragma unroll
        for (int j = 0; j < DM_OUTS; ++j) acc[j] = 0.f;
    }
}

void launch_moe_down_masked_fp8(
    const __nv_bfloat16* buf,
    const uint8_t* W, const float* Ws,
    const int* pairs,
    __nv_bfloat16* out,
    int E, int max_m, int hidden, int inter,
    int RS, int CS,
    int pair_slots,
    cudaStream_t stream)
{
    const size_t total_warps = (size_t)pair_slots * (hidden / DM_OUTS);
    const size_t ctas = (total_warps + DM_WARPS_PER_CTA - 1) / DM_WARPS_PER_CTA;
    moe_down_masked_fp8_kernel<<<dim3((unsigned)ctas), dim3(DM_THREADS_PER_CTA),
                                 0, stream>>>(
        buf, W, Ws, pairs, out, E, max_m, hidden, inter, RS, CS, pair_slots);
}

}  // namespace warp_decode
