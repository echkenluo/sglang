// warp_decode: down kernel for FP8-E4M3 block-scale MoE decode (DeepSeek-style
// [128,128] weight blocks).
//
// v2 notes (H20 tuning):
//   * The fp8 dequant chain (cvt fp8x2->half2 -> float2) between each weight
//     load and its use made ptxas serialize the OUTS row loads into a
//     dependency chain (SASS showed consecutive LDGs reusing one register),
//     collapsing memory-level parallelism ~5x. Fix: stage all OUTS row loads
//     into a register array first, then convert+FMA in a second unrolled loop.
//   * OUTS_PER_WARP is a template param selected at launch: at tiny B the
//     OUTS=8 grid (B*hidden/64 CTAs) starves H20's 78 SMs, so the launcher
//     drops to OUTS=4/2 to keep >=2 CTAs per SM when B is small.
//
// Layout:
//   buffer     : [B, topk, inter]     bf16   (gate_up output)
//   W          : [E, hidden, inter]   fp8 e4m3
//   Ws         : [E, ceil(hidden/128), ceil(inter/128)]  fp32 block scales
//   topk_ids   : [B, topk]            int32
//   topk_scale : [B, topk]            fp32
//   out        : [B, hidden]          bf16
//
// The OUTS-row group always sits inside one 128-row scale block (OUTS | 128),
// so all OUTS rows share the same scale row.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace warp_decode {

constexpr int D8_WARP_SIZE       = 32;
constexpr int D8_WARPS_PER_CTA   = 8;
constexpr int D8_THREADS_PER_CTA = D8_WARPS_PER_CTA * D8_WARP_SIZE;
constexpr int D8_QUAD            = 4;    // 4 fp8 == uint32; 4 bf16 == uint2

__device__ __forceinline__ float d8_warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

template<int OUTS, bool OCT>
__global__ void moe_down_fp8_blockscale_kernel(
    const __nv_bfloat16* __restrict__ buffer,     // [B, topk, inter]
    const uint8_t*       __restrict__ W,          // [E, hidden, inter] fp8
    const float*         __restrict__ Ws,         // [E, RS, CS]
    const int*           __restrict__ topk_ids,   // [B, topk]
    const float*         __restrict__ topk_scale, // [B, topk]
    __nv_bfloat16*       __restrict__ out,        // [B, hidden]
    int B, int E, int topk, int hidden, int inter,
    int RS, int CS)
{
    const size_t global_tid  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t global_warp = global_tid / D8_WARP_SIZE;
    const int    lane_id     = threadIdx.x & (D8_WARP_SIZE - 1);

    const int    groups_per_token = hidden / OUTS;
    const size_t total_warps      = (size_t)B * groups_per_token;
    if (global_warp >= total_warps) return;

    const int token_id  = (int)(global_warp / groups_per_token);
    const int group_idx = (int)(global_warp - (size_t)token_id * groups_per_token);
    const int h_base    = group_idx * OUTS;

    // 32 lanes x 4 fp8 quads == 128 elems == one scale block per iteration.
    const int kblocks = inter >> 7;

    float acc[OUTS];
    #pragma unroll
    for (int j = 0; j < OUTS; ++j) acc[j] = 0.f;

    #pragma unroll 1
    for (int s = 0; s < topk; ++s) {
        const int   expert_id = topk_ids  [token_id * topk + s];
        const float scale_tk  = topk_scale[token_id * topk + s];

        const __nv_bfloat16* __restrict__ in_ptr =
            buffer + ((size_t)token_id * topk + s) * inter;

        const uint8_t* w_base =
            W + (size_t)expert_id * hidden * inter + (size_t)h_base * inter;
        const uint32_t* w_ptrs[OUTS];
        #pragma unroll
        for (int j = 0; j < OUTS; ++j) {
            w_ptrs[j] = reinterpret_cast<const uint32_t*>(w_base + (size_t)j * inter);
        }

        const float* __restrict__ ws_row =
            Ws + ((size_t)expert_id * RS + (h_base >> 7)) * CS;

        float local[OUTS];
        #pragma unroll
        for (int j = 0; j < OUTS; ++j) local[j] = 0.f;

        if (OCT) {
            // 8-fp8-per-load path (inter % 256 == 0): halves the load count
            // and iteration count vs the quad path, doubling in-flight bytes.
            const uint8_t* w_rows = reinterpret_cast<const uint8_t*>(w_ptrs[0]);
            const int oblocks = inter >> 8;
            #pragma unroll 1
            for (int ob = 0; ob < oblocks; ++ob) {
                const int e0 = ob * 256 + lane_id * 8;   // element offset

                // ---- stage: issue all loads back-to-back ----
                uint2 wq[OUTS];
                #pragma unroll
                for (int j = 0; j < OUTS; ++j) {
                    wq[j] = *reinterpret_cast<const uint2*>(
                        &w_rows[(size_t)j * inter + e0]);
                }
                const uint4 xq = *reinterpret_cast<const uint4*>(&in_ptr[e0]);
                const float s_blk = ws_row[e0 >> 7];

                // ---- compute after all loads are in flight ----
                const __nv_bfloat16* xh =
                    reinterpret_cast<const __nv_bfloat16*>(&xq);
                float xf[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) xf[i] = __bfloat162float(xh[i]);

                #pragma unroll
                for (int j = 0; j < OUTS; ++j) {
                    const __nv_fp8x2_storage_t* w2 =
                        reinterpret_cast<const __nv_fp8x2_storage_t*>(&wq[j]);
                    float oct = 0.f;
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        float2 wf = __half22float2(
                            __half2(__nv_cvt_fp8x2_to_halfraw2(w2[p], __NV_E4M3)));
                        oct += xf[2 * p] * wf.x + xf[2 * p + 1] * wf.y;
                    }
                    local[j] += s_blk * oct;
                }
            }
        } else {
        #pragma unroll 1
        for (int kb = 0; kb < kblocks; ++kb) {
            const int q = kb * D8_WARP_SIZE + lane_id;   // quad index

            // ---- stage: issue all loads back-to-back (keep MLP high) ----
            uint32_t wq[OUTS];
            #pragma unroll
            for (int j = 0; j < OUTS; ++j) wq[j] = w_ptrs[j][q];
            const uint2 xq = *reinterpret_cast<const uint2*>(&in_ptr[q * D8_QUAD]);
            const float s_blk = ws_row[kb];

            // ---- compute: dequant + FMA after all loads are in flight ----
            const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xq);
            const float x0 = __bfloat162float(xh[0]);
            const float x1 = __bfloat162float(xh[1]);
            const float x2 = __bfloat162float(xh[2]);
            const float x3 = __bfloat162float(xh[3]);

            #pragma unroll
            for (int j = 0; j < OUTS; ++j) {
                const __nv_fp8x2_storage_t* w2 =
                    reinterpret_cast<const __nv_fp8x2_storage_t*>(&wq[j]);
                float2 w01 = __half22float2(
                    __half2(__nv_cvt_fp8x2_to_halfraw2(w2[0], __NV_E4M3)));
                float2 w23 = __half22float2(
                    __half2(__nv_cvt_fp8x2_to_halfraw2(w2[1], __NV_E4M3)));
                float quad = x0 * w01.x + x1 * w01.y + x2 * w23.x + x3 * w23.y;
                local[j] += s_blk * quad;
            }
        }
        }

        #pragma unroll
        for (int j = 0; j < OUTS; ++j) acc[j] += scale_tk * local[j];
    }

    #pragma unroll
    for (int j = 0; j < OUTS; ++j) acc[j] = d8_warp_reduce_sum(acc[j]);

    if (lane_id == 0) {
        __nv_bfloat16* out_ptr = out + (size_t)token_id * hidden + h_base;
        #pragma unroll
        for (int j = 0; j < OUTS; ++j) {
            out_ptr[j] = __float2bfloat16(acc[j]);
        }
    }
}

template<int OUTS>
static void launch_down_fp8(
    const __nv_bfloat16* buffer, const uint8_t* W, const float* Ws,
    const int* topk_ids, const float* topk_scale, __nv_bfloat16* out,
    int B, int E, int topk, int hidden, int inter, int RS, int CS,
    cudaStream_t stream)
{
    const size_t total_warps = (size_t)B * (hidden / OUTS);
    const size_t ctas = (total_warps + D8_WARPS_PER_CTA - 1) / D8_WARPS_PER_CTA;
    if (inter % 256 == 0) {
        moe_down_fp8_blockscale_kernel<OUTS, true><<<dim3((unsigned)ctas),
            dim3(D8_THREADS_PER_CTA), 0, stream>>>(
            buffer, W, Ws, topk_ids, topk_scale, out,
            B, E, topk, hidden, inter, RS, CS);
    } else {
        moe_down_fp8_blockscale_kernel<OUTS, false><<<dim3((unsigned)ctas),
            dim3(D8_THREADS_PER_CTA), 0, stream>>>(
            buffer, W, Ws, topk_ids, topk_scale, out,
            B, E, topk, hidden, inter, RS, CS);
    }
}

void launch_moe_down_fp8_blockscale(
    const __nv_bfloat16* buffer,
    const uint8_t*       W,
    const float*         Ws,
    const int*           topk_ids,
    const float*         topk_scale,
    __nv_bfloat16*       out,
    int B, int E, int topk, int hidden, int inter,
    int RS, int CS,
    cudaStream_t stream,
    int outs /* 2, 4, 8, or -1 = heuristic */)
{
    if (outs != 2 && outs != 4 && outs != 8) {
        if (inter % 256 == 0) {
            // OCT path: 8-elem loads already double per-lane input reuse, and
            // measured sweeps on H20 (V4 shapes) show OUTS=4 wins at every
            // batch size (better CTA count at tiny B, no reuse loss at large B).
            outs = 4;
        } else {
            // Quad path: shrink OUTS until the grid can put ~2 CTAs on each
            // of H20's 78 SMs; input reuse falls with OUTS, prefer largest.
            outs = 8;
            while (outs > 2 &&
                   (size_t)B * (hidden / outs) / D8_WARPS_PER_CTA < 156) {
                outs >>= 1;
            }
        }
    }
    if (hidden % outs != 0) outs = 2;  // hidden % 2 enforced host-side

    switch (outs) {
        case 8: launch_down_fp8<8>(buffer, W, Ws, topk_ids, topk_scale, out,
                                   B, E, topk, hidden, inter, RS, CS, stream); break;
        case 4: launch_down_fp8<4>(buffer, W, Ws, topk_ids, topk_scale, out,
                                   B, E, topk, hidden, inter, RS, CS, stream); break;
        default: launch_down_fp8<2>(buffer, W, Ws, topk_ids, topk_scale, out,
                                    B, E, topk, hidden, inter, RS, CS, stream); break;
    }
}

}  // namespace warp_decode
