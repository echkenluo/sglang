"""FlashOverlap GEMM+AllReduce overlap for attention o_proj (research prototype).

Replaces o_proj GEMM + the communicator's AllReduce with FlashOverlap's
signal-based overlapped kernel; the tile-reordered output is restored by a
fused add+reorder+rmsnorm kernel at the prepare_mlp site.

Activation: SGLANG_FO=1, fp16 weights, TP>1, gathered M == SGLANG_FO_M
(the offline-tuned shape). Non-matching batches fall through unchanged.
"""
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

_ENABLE = os.environ.get("SGLANG_FO", "0") == "1"
_FO_LIB = os.environ.get(
    "SGLANG_FO_LIB", "/work/FlashOverlap/build/lib/libst_pybinding.so"
)
_FO_CFG = os.environ.get(
    "SGLANG_FO_CFG", "/work/FlashOverlap/configs/m4096n2048k1024_l20.json"
)
_FO_M = int(os.environ.get("SGLANG_FO_M", "4096"))

_state = None
_failed = False
_pending = False

_CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warpReduceSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

__device__ float blockReduceSum(float v, float* shared) {
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;
  v = warpReduceSum(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  v = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
  if (wid == 0) v = warpReduceSum(v);
  return v;
}

// One block per logical row. Input is FlashOverlap tile-sequential layout
// (rldn=1 addressing); gathers, adds residual, writes residual_out and the
// rmsnorm output, both in logical row-major order.
__global__ void fused_add_reorder_rmsnorm_kernel(
    const half* __restrict__ x, const half* __restrict__ res,
    const half* __restrict__ w, half* __restrict__ out,
    half* __restrict__ res_out, const int* __restrict__ RA,
    int bs, int dim, int BM, int BN, int ldn, float eps) {
  int bid = blockIdx.x;
  int j = threadIdx.x << 3;
  if (j >= dim) return;

  int old_index = bid / BM * ldn + j / BN;
  int new_index = RA[old_index];
  long src = ((long)new_index * BM + bid % BM) * BN + j % BN;

  half2 xv[4], rv[4], wv[4];
  *(float4*)(&xv[0]) = *(const float4*)(&x[src]);
  *(float4*)(&rv[0]) = *(const float4*)(&res[(long)bid * dim + j]);
  *(float4*)(&wv[0]) = *(const float4*)(&w[j]);

  float ps = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float sx = __half2float(xv[i].x) + __half2float(rv[i].x);
    float sy = __half2float(xv[i].y) + __half2float(rv[i].y);
    xv[i].x = __float2half(sx);
    xv[i].y = __float2half(sy);
    ps += sx * sx + sy * sy;
  }
  *(float4*)(&res_out[(long)bid * dim + j]) = *(float4*)(&xv[0]);

  __shared__ float sh[WARP_SIZE];
  ps = blockReduceSum(ps, sh);
  __shared__ float scale;
  if (threadIdx.x == 0) scale = rsqrtf(ps / (float)dim + eps);
  __syncthreads();
  float s = scale;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    xv[i].x = __hmul(__float2half(__half2float(xv[i].x) * s), wv[i].x);
    xv[i].y = __hmul(__float2half(__half2float(xv[i].y) * s), wv[i].y);
  }
  *(float4*)(&out[(long)bid * dim + j]) = *(float4*)(&xv[0]);
}

void fused_add_reorder_rmsnorm(
    torch::Tensor x, torch::Tensor res, torch::Tensor w, torch::Tensor out,
    torch::Tensor res_out, torch::Tensor ra, int64_t BM, int64_t BN,
    double eps) {
  int bs = res.size(0), dim = res.size(1);
  int ldn = dim / BN;
  fused_add_reorder_rmsnorm_kernel<<<bs, dim / 8, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<half*>(res.data_ptr<at::Half>()),
      reinterpret_cast<half*>(w.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<half*>(res_out.data_ptr<at::Half>()),
      ra.data_ptr<int>(), bs, dim, (int)BM, (int)BN, ldn, (float)eps);
}
"""

_CPP_SRC = (
    "void fused_add_reorder_rmsnorm(torch::Tensor x, torch::Tensor res, "
    "torch::Tensor w, torch::Tensor out, torch::Tensor res_out, "
    "torch::Tensor ra, int64_t BM, int64_t BN, double eps);"
)


def _div_up(a, b):
    return (a + b - 1) // b


def _reorder_indices(s, hint):
    new_order = [-1] * s
    for i, e in enumerate(hint):
        new_order[e] = i
    hint_set = set(hint)
    rem = [x for x in range(s) if x not in hint_set]
    for i, e in enumerate(rem, start=len(hint)):
        new_order[e] = i
    return torch.tensor(new_order, dtype=torch.int, device="cuda")


class _FOState:
    pass


def _init_state(weight):
    import torch.distributed as dist

    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        get_tp_group,
    )
    from torch.utils.cpp_extension import load_inline

    st = _FOState()
    torch.ops.load_library(_FO_LIB)
    rank = get_tensor_model_parallel_rank()
    world = get_tensor_model_parallel_world_size()
    cfg = json.load(open(_FO_CFG))

    group = get_tp_group().device_group
    nid = torch.ops.flashoverlap_op.generate_nccl_id() if rank == 0 else None
    obj = [nid]
    dist.broadcast_object_list(obj, src=dist.get_global_rank(group, 0), group=group)
    nid = obj[0]

    impl = torch.classes.flashoverlap_class.OverlapImpl()
    impl.nccl_init(rank, world, nid)
    impl.cutlass_init()
    impl.overlap_init()

    n_out = weight.shape[0]
    bm, bn = cfg["BM"], cfg["BN"]
    tm, tn = _div_up(_FO_M, bm), _div_up(n_out, bn)
    st.impl = impl
    st.algo = cfg["Algo"]
    st.bm, st.bn = bm, bn
    st.counter = torch.zeros((1, tn), dtype=torch.int, device="cuda")
    ra = _reorder_indices(tm * tn, cfg["hint"])
    st.ra2d = ra.reshape((tm, tn))
    st.ra_flat = ra.contiguous()
    st.cseg_cpu = torch.tensor(cfg["cSeg"], dtype=torch.int32)
    st.cseg_gpu = st.cseg_cpu.cuda()
    st.mod = load_inline(
        name="sgl_fo_farr",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["fused_add_reorder_rmsnorm"],
        verbose=False,
    )
    logger.info("[FO] initialized: M=%d N=%d algo=%s BM=%d BN=%d", _FO_M, n_out, st.algo, bm, bn)
    return st


def maybe_o_proj(x, weight):
    """Returns tile-reordered full-sum o_proj output, or None to fall through."""
    global _state, _failed, _pending
    if (
        not _ENABLE
        or _failed
        or x.dtype != torch.float16
        or x.shape[0] != _FO_M
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    if _state is None:
        try:
            _state = _init_state(weight)
        except Exception:
            logger.exception("[FO] init failed; disabled")
            _failed = True
            return None
    st = _state
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float16, device=x.device)
    st.impl.gemm_allreduce_overlap(
        x.contiguous(), weight, out, st.counter, st.ra2d, 1,
        st.cseg_cpu, st.cseg_gpu, st.algo, False,
    )
    _pending = True
    return out


def pending():
    return _pending


def consume(hidden_states, residual, norm_module):
    """Fused add+reorder+rmsnorm; returns (normed, new_residual) in logical order."""
    global _pending
    st = _state
    out = torch.empty_like(residual)
    res_out = torch.empty_like(residual)
    st.mod.fused_add_reorder_rmsnorm(
        hidden_states, residual, norm_module.weight, out, res_out,
        st.ra_flat, st.bm, st.bn, norm_module.variance_epsilon,
    )
    _pending = False
    return out, res_out
