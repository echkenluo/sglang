// Torch bindings for the warp-decode FP8 block-scale MoE kernels (sglang embed).
#include <torch/extension.h>
#include <limits>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

namespace warp_decode {

void launch_moe_gate_up_fp8_blockscale(
    const __nv_bfloat16* X,
    const uint8_t*       W,
    const float*         Ws,
    const int*           topk_ids,
    __nv_bfloat16*       out,
    int B, int E, int topk, int hidden, int inter,
    int RS, int CS,
    cudaStream_t stream,
    float swiglu_limit);

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
    int outs);

void launch_compact_pairs(
    const int* masked_m, int* pairs, int E, int max_m, cudaStream_t stream);

void launch_moe_gate_up_masked_fp8(
    const uint8_t* X, const float* Xs,
    const uint8_t* W, const float* Ws,
    const int* pairs,
    __nv_bfloat16* buf,
    int E, int max_m, int hidden, int inter,
    int RS, int CS, int CSX,
    float swiglu_limit,
    int pair_slots,
    cudaStream_t stream);

void launch_moe_down_masked_fp8(
    const __nv_bfloat16* buf,
    const uint8_t* W, const float* Ws,
    const int* pairs,
    __nv_bfloat16* out,
    int E, int max_m, int hidden, int inter,
    int RS, int CS,
    int pair_slots,
    cudaStream_t stream);

}  // namespace warp_decode

namespace {

#define CHECK_CUDA(x)   TORCH_CHECK((x).is_cuda(),       #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x)   TORCH_CHECK((x).scalar_type() == at::kBFloat16,      #x " must be bf16")
#define CHECK_F32(x)    TORCH_CHECK((x).scalar_type() == at::kFloat,         #x " must be fp32")
#define CHECK_I32(x)    TORCH_CHECK((x).scalar_type() == at::kInt,           #x " must be int32")
#define CHECK_FP8(x)    TORCH_CHECK((x).scalar_type() == at::kFloat8_e4m3fn, #x " must be fp8 e4m3fn")

torch::Tensor moe_gate_up_fp8_blockscale(
    torch::Tensor x,           // [B, hidden]           bf16
    torch::Tensor w,           // [E, 2*inter, hidden]  fp8 e4m3
    torch::Tensor ws,          // [E, RS, CS]           fp32 block scales
    torch::Tensor topk_ids,    // [B, topk]             int32
    double        swiglu_limit)  // +inf disables
{
    CHECK_CUDA(x);        CHECK_CONTIG(x);        CHECK_BF16(x);
    CHECK_CUDA(w);        CHECK_CONTIG(w);        CHECK_FP8(w);
    CHECK_CUDA(ws);       CHECK_CONTIG(ws);       CHECK_F32(ws);
    CHECK_CUDA(topk_ids); CHECK_CONTIG(topk_ids); CHECK_I32(topk_ids);

    TORCH_CHECK(x.dim() == 2 && w.dim() == 3 && ws.dim() == 3 && topk_ids.dim() == 2);

    const int B      = x.size(0);
    const int hidden = x.size(1);
    const int E      = w.size(0);
    const int two_i  = w.size(1);
    TORCH_CHECK(two_i % 2 == 0, "w.size(1) must be 2 * inter");
    const int inter  = two_i / 2;
    TORCH_CHECK(w.size(2) == hidden, "w.size(2) must == hidden");
    TORCH_CHECK(topk_ids.size(0) == B);
    const int topk   = topk_ids.size(1);

    const int RS = ws.size(1);
    const int CS = ws.size(2);
    TORCH_CHECK(ws.size(0) == E);
    TORCH_CHECK(RS == (two_i  + 127) / 128, "ws rows must be ceil(2*inter/128)");
    TORCH_CHECK(CS == (hidden + 127) / 128, "ws cols must be ceil(hidden/128)");

    TORCH_CHECK(hidden % 512 == 0, "hidden must be a multiple of 512");
    TORCH_CHECK(inter  % 128 == 0, "inter must be a multiple of 128");

    auto out = torch::empty({B * topk, inter}, x.options().dtype(at::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();

    warp_decode::launch_moe_gate_up_fp8_blockscale(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const uint8_t*>(w.data_ptr()),
        ws.data_ptr<float>(),
        topk_ids.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        B, E, topk, hidden, inter, RS, CS,
        stream.stream(), static_cast<float>(swiglu_limit));

    return out;
}

torch::Tensor moe_down_fp8_blockscale(
    torch::Tensor buffer,       // [B*topk, inter]        bf16
    torch::Tensor w,            // [E, hidden, inter]     fp8 e4m3
    torch::Tensor ws,           // [E, RS, CS]            fp32 block scales
    torch::Tensor topk_ids,     // [B, topk]              int32
    torch::Tensor topk_scale,   // [B, topk]              fp32
    int64_t       B_arg,
    int64_t       outs)         // 2/4/8 or -1 = heuristic
{
    CHECK_CUDA(buffer);     CHECK_CONTIG(buffer);     CHECK_BF16(buffer);
    CHECK_CUDA(w);          CHECK_CONTIG(w);          CHECK_FP8(w);
    CHECK_CUDA(ws);         CHECK_CONTIG(ws);         CHECK_F32(ws);
    CHECK_CUDA(topk_ids);   CHECK_CONTIG(topk_ids);   CHECK_I32(topk_ids);
    CHECK_CUDA(topk_scale); CHECK_CONTIG(topk_scale); CHECK_F32(topk_scale);

    TORCH_CHECK(buffer.dim() == 2 && w.dim() == 3 && ws.dim() == 3
                && topk_ids.dim() == 2 && topk_scale.dim() == 2);

    const int B      = static_cast<int>(B_arg);
    const int inter  = buffer.size(1);
    const int E      = w.size(0);
    const int hidden = w.size(1);
    TORCH_CHECK(w.size(2) == inter, "w.size(2) must == inter");
    TORCH_CHECK(topk_ids.size(0) == B);
    const int topk   = topk_ids.size(1);
    TORCH_CHECK(topk_scale.size(0) == B && topk_scale.size(1) == topk);
    TORCH_CHECK(buffer.size(0) == B * topk, "buffer rows must be B*topk");

    const int RS = ws.size(1);
    const int CS = ws.size(2);
    TORCH_CHECK(ws.size(0) == E);
    TORCH_CHECK(RS == (hidden + 127) / 128, "ws rows must be ceil(hidden/128)");
    TORCH_CHECK(CS == (inter  + 127) / 128, "ws cols must be ceil(inter/128)");

    TORCH_CHECK(inter  % 128 == 0, "inter must be a multiple of 128");
    TORCH_CHECK(hidden %   8 == 0, "hidden must be a multiple of OUTS_PER_WARP");

    auto out = torch::empty({B, hidden}, buffer.options().dtype(at::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();

    warp_decode::launch_moe_down_fp8_blockscale(
        reinterpret_cast<const __nv_bfloat16*>(buffer.data_ptr()),
        reinterpret_cast<const uint8_t*>(w.data_ptr()),
        ws.data_ptr<float>(),
        topk_ids.data_ptr<int>(),
        topk_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        B, E, topk, hidden, inter, RS, CS,
        stream.stream(), static_cast<int>(outs));

    return out;
}


torch::Tensor moe_warp_masked_fp8(
    torch::Tensor x,          // [E, max_m, hidden]  fp8 e4m3 (LL recv)
    torch::Tensor xs,         // [E, max_m, CSX]     fp32 per-token-group scales
    torch::Tensor w13,        // [E, 2*inter, hidden] fp8
    torch::Tensor w13s,       // [E, RS13, CS13]     fp32
    torch::Tensor w2,         // [E, hidden, inter]  fp8
    torch::Tensor w2s,        // [E, RS2, CS2]       fp32
    torch::Tensor masked_m,   // [E]                 int32
    torch::Tensor pairs,      // [1 + E*max_m]       int32 workspace
    torch::Tensor buf,        // [E, max_m, inter]   bf16 workspace
    torch::Tensor out,        // [E, max_m, hidden]  bf16 output
    double        swiglu_limit,
    int64_t       pair_slots)
{
    CHECK_CUDA(x);   CHECK_CONTIG(x);   CHECK_FP8(x);
    CHECK_CUDA(xs);  CHECK_CONTIG(xs);  CHECK_F32(xs);
    CHECK_CUDA(w13); CHECK_CONTIG(w13); CHECK_FP8(w13);
    CHECK_CUDA(w13s);CHECK_CONTIG(w13s);CHECK_F32(w13s);
    CHECK_CUDA(w2);  CHECK_CONTIG(w2);  CHECK_FP8(w2);
    CHECK_CUDA(w2s); CHECK_CONTIG(w2s); CHECK_F32(w2s);
    CHECK_CUDA(masked_m); CHECK_CONTIG(masked_m); CHECK_I32(masked_m);
    CHECK_CUDA(pairs);    CHECK_CONTIG(pairs);    CHECK_I32(pairs);
    CHECK_CUDA(buf); CHECK_CONTIG(buf); CHECK_BF16(buf);
    CHECK_CUDA(out); CHECK_CONTIG(out); CHECK_BF16(out);

    const int E      = x.size(0);
    const int max_m  = x.size(1);
    const int hidden = x.size(2);
    const int two_i  = w13.size(1);
    const int inter  = two_i / 2;
    const int RS  = w13s.size(1);
    const int CS  = w13s.size(2);
    const int RS2 = w2s.size(1);
    const int CS2 = w2s.size(2);
    const int CSX = xs.size(2);

    TORCH_CHECK(w13.size(0) == E && w2.size(0) == E && masked_m.size(0) == E);
    TORCH_CHECK(w13.size(2) == hidden && w2.size(1) == hidden && w2.size(2) == inter);
    TORCH_CHECK(xs.size(0) == E && xs.size(1) == max_m);
    TORCH_CHECK(CSX == (hidden + 127) / 128, "xs cols must be ceil(hidden/128)");
    TORCH_CHECK(RS == (two_i + 127) / 128 && CS == (hidden + 127) / 128);
    TORCH_CHECK(RS2 == (hidden + 127) / 128 && CS2 == (inter + 127) / 128);
    TORCH_CHECK(buf.size(0) == E && buf.size(1) == max_m && buf.size(2) == inter);
    TORCH_CHECK(out.size(0) == E && out.size(1) == max_m && out.size(2) == hidden);
    TORCH_CHECK(pairs.numel() >= 1 + E * max_m, "pairs workspace too small");
    TORCH_CHECK(hidden % 512 == 0, "hidden must be a multiple of 512");
    TORCH_CHECK(inter % 256 == 0, "inter must be a multiple of 256 (OCT loads)");
    TORCH_CHECK(hidden % 4 == 0);
    TORCH_CHECK(E <= 1024, "compact_pairs assumes E <= 1024");
    TORCH_CHECK(max_m < 65536, "pair packing assumes max_m < 2^16");
    TORCH_CHECK(pair_slots >= 1);

    auto stream = at::cuda::getCurrentCUDAStream();

    warp_decode::launch_compact_pairs(
        masked_m.data_ptr<int>(), pairs.data_ptr<int>(), E, max_m,
        stream.stream());

    warp_decode::launch_moe_gate_up_masked_fp8(
        reinterpret_cast<const uint8_t*>(x.data_ptr()),
        xs.data_ptr<float>(),
        reinterpret_cast<const uint8_t*>(w13.data_ptr()),
        w13s.data_ptr<float>(),
        pairs.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(buf.data_ptr()),
        E, max_m, hidden, inter, RS, CS, CSX,
        static_cast<float>(swiglu_limit),
        static_cast<int>(pair_slots),
        stream.stream());

    warp_decode::launch_moe_down_masked_fp8(
        reinterpret_cast<const __nv_bfloat16*>(buf.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2.data_ptr()),
        w2s.data_ptr<float>(),
        pairs.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        E, max_m, hidden, inter, RS2, CS2,
        static_cast<int>(pair_slots),
        stream.stream());

    return out;
}

}  // anonymous namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_gate_up_fp8_blockscale", &moe_gate_up_fp8_blockscale,
          "Warp-decode FP8-blockscale gate_up + silu_and_mul (fused, bf16 activations)",
          pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("ws"),
          pybind11::arg("topk_ids"),
          pybind11::arg("swiglu_limit") = std::numeric_limits<double>::infinity());
    m.def("moe_warp_masked_fp8", &moe_warp_masked_fp8,
          "Warp-decode masked MoE (DeepEP LL layout): compact + gate_up + down",
          pybind11::arg("x"), pybind11::arg("xs"), pybind11::arg("w13"),
          pybind11::arg("w13s"), pybind11::arg("w2"), pybind11::arg("w2s"),
          pybind11::arg("masked_m"), pybind11::arg("pairs"), pybind11::arg("buf"),
          pybind11::arg("out"),
          pybind11::arg("swiglu_limit") = std::numeric_limits<double>::infinity(),
          pybind11::arg("pair_slots") = 64);
    m.def("moe_down_fp8_blockscale", &moe_down_fp8_blockscale,
          "Warp-decode FP8-blockscale down projection + topk-weighted sum",
          pybind11::arg("buffer"), pybind11::arg("w"), pybind11::arg("ws"),
          pybind11::arg("topk_ids"), pybind11::arg("topk_scale"),
          pybind11::arg("B_arg"), pybind11::arg("outs") = -1);
}
