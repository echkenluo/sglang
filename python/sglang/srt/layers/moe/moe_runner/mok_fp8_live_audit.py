"""Opt-in real-layer arithmetic diagnostic, never a performance measurement.

The caller owns the MoK workspace lease throughout this function. Compare
actual fused intermediates and a W13 component replay with SGLang's production
contiguous grouped DeepGEMM entry. This is a common-input arithmetic reference,
not an independent routing oracle or an end-to-end quality gate.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path

import torch


_SEEN = set()


def grouped_row_plan(indices, experts, alignment=128):
    """Map arbitrary expert rows into independently padded contiguous groups.

    Returns source rows (-1 means new zero padding), grouped expert indices,
    and the inverse map restoring every original row exactly once.
    Pure Python so the reference's layout contract can be tested without CUDA.
    """
    if experts <= 0 or alignment <= 0 or not indices:
        raise ValueError("nonempty rows, positive expert count/alignment required")
    groups = [[] for _ in range(experts)]
    for row, expert in enumerate(indices):
        if type(expert) is not int or not 0 <= expert < experts:
            raise ValueError("active row has invalid expert index")
        groups[expert].append(row)
    source, grouped = [], []
    inverse = [-1] * len(indices)
    for expert, rows in enumerate(groups):
        start = len(source)
        for offset, row in enumerate(rows):
            inverse[row] = start + offset
        padded = (len(rows) + alignment - 1) // alignment * alignment
        source.extend(rows + [-1] * (padded - len(rows)))
        grouped.extend([expert] * padded)
    assert sorted(x for x in source if x >= 0) == list(range(len(indices)))
    assert all(source[inverse[row]] == row for row in range(len(indices)))
    return source, grouped, inverse


def comparison(actual, reference):
    """Full-row sufficient statistics plus aggregate metrics; no threshold waiver."""
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    # FP64 reductions avoid statistics depending on TF32/matmul settings.
    x, y = actual.double(), reference.double()
    assert bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
    delta = x - y
    rows = {
        "error_squared": (delta * delta).sum(-1).cpu(),
        "reference_squared": (y * y).sum(-1).cpu(),
        "max_abs": delta.abs().amax(-1).cpu(),
        "exact_elements": (actual.view(torch.uint8) == reference.view(torch.uint8))
        .reshape(actual.shape[0], actual.shape[1], actual.element_size())
        .all(-1).sum(-1).cpu(),
    }
    errors = rows["error_squared"].sqrt()
    norms = rows["reference_squared"].sqrt()
    relative = torch.where(norms > 0, errors / norms, errors)
    total_norm = float(rows["reference_squared"].sum().sqrt())
    return {
        "rows": actual.shape[0], "columns": actual.shape[1],
        "exact_fraction": float(rows["exact_elements"].sum()) / actual.numel(),
        "max_row_relative_l2": float(relative.max()),
        "relative_l2": float(rows["error_squared"].sum().sqrt()) / max(total_norm, 1e-30),
        "max_abs": float(rows["max_abs"].max()),
    }, rows


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_warprole_layer(layer, state, schedule, output, variant, input_tokens):
    import torch.distributed as dist
    from mok import _C
    from sglang.kernels.ops.attention.dsv4 import silu_and_mul_contig_post_quant_dynamic
    from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers import deep_gemm_wrapper

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("MoK live arithmetic audit requires eager execution")
    if input_tokens < int(os.environ.get("SGLANG_MOK_LIVE_AUDIT_MIN_TOKENS", "1024")):
        return
    rank, layer_id = dist.get_rank(), int(layer.layer_id)
    directory = Path(os.environ["SGLANG_MOK_LIVE_AUDIT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    # This completion receipt applies to every eligible Python invocation.
    torch.cuda.synchronize()
    with (directory / f"rank{rank}-coverage.jsonl").open("a") as stream:
        stream.write(json.dumps({"layer": layer_id, "rank": rank, "variant": variant,
                                 "input_tokens": input_tokens,
                                 "kernel_completed": True}) + "\n")
    if layer_id in _SEEN:
        return

    n = int(schedule.num_tokens.item())
    assert 0 < n <= state.capacity and n % 64 == 0
    indices = state.m_indices[:n].cpu().tolist()
    source, grouped, inverse = grouped_row_plan(indices, layer.num_local_experts)
    device = output.device
    source_gpu = torch.tensor(source, dtype=torch.long, device=device)
    inverse_gpu = torch.tensor(inverse, dtype=torch.long, device=device)
    grouped_gpu = torch.tensor(grouped, dtype=torch.int32, device=device)
    valid = source_gpu >= 0
    gather = source_gpu.clamp_min(0)
    # Preserve output bytes across all diagnostic launches.
    original_output = output.clone()

    def pack(tensor, scale):
        packed = tensor[:n].view(torch.uint8)[gather].contiguous()
        packed[~valid] = 0
        packed = packed.view(tensor.dtype)
        packed_scale = scale[:n][gather].contiguous()
        packed_scale[~valid] = 1.0
        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            packed_scale = tma_align_input_scale(packed_scale)
        return packed, packed_scale

    def grouped_gemm(tensor, scale, weight, weight_scale):
        result = torch.empty((len(source), weight.shape[1]), dtype=torch.bfloat16, device=device)
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            pack(tensor, scale), (weight, weight_scale), result, grouped_gpu
        )
        return result

    def unpad(tensor):
        # CUDA indexing is not implemented for all float8 variants.
        if tensor.dtype == torch.float8_e4m3fn:
            return tensor.view(torch.uint8)[inverse_gpu].contiguous().view(tensor.dtype)
        return tensor[inverse_gpu].contiguous()

    w13 = torch.empty((state.capacity, layer.w13_weight.shape[1]),
                      dtype=torch.bfloat16, device=device)
    getattr(_C, f"fp8_block_warprole_gemm_{variant}_out")(
        state.routed_x, layer.w13_weight, state.routed_x_scale,
        layer.w13_weight_scale_inv, state.m_indices, schedule.num_tokens, w13
    )
    dg_gate = grouped_gemm(state.routed_x, state.routed_x_scale,
                           layer.w13_weight, layer.w13_weight_scale_inv)
    dg_hidden = torch.empty((len(source), state.hidden.shape[1]),
                            dtype=state.hidden.dtype, device=device)
    dg_scale = torch.empty((len(source), state.hidden_scale.shape[1]),
                           dtype=torch.float32, device=device)
    active = torch.tensor([len(source)], dtype=torch.int32, device=device)
    silu_and_mul_contig_post_quant_dynamic(
        input=dg_gate, output=dg_hidden, output_scale=dg_scale,
        active_tokens=active, quant_group_size=128, scale_ue8m0=False,
        transposed=False, swiglu_limit=layer.moe_runner_config.swiglu_limit, swizzle=False,
    )
    dg_down_same_hidden = grouped_gemm(state.hidden, state.hidden_scale,
                                       layer.w2_weight, layer.w2_weight_scale_inv)
    # Complete DeepGEMM expert path, carrying its own W13/activation differences.
    hidden_original_order, scale_original_order = unpad(dg_hidden), unpad(dg_scale)
    dg_down_pipeline = grouped_gemm(hidden_original_order, scale_original_order,
                                    layer.w2_weight, layer.w2_weight_scale_inv)
    pairs = {
        "w13_component": (w13[:n], unpad(dg_gate)),
        "fused_activation": (state.hidden[:n], hidden_original_order),
        "activation_scale": (state.hidden_scale[:n], scale_original_order),
        "w2_same_input": (state.routed_y[:n], unpad(dg_down_same_hidden)),
        "expert_pipeline": (state.routed_y[:n], unpad(dg_down_pipeline)),
    }
    # Deterministic first/last real route per active expert. Full-row statistics
    # cover padding too; saved raw samples only cover real source routes.
    peers = schedule.peer_rank[:n].cpu()
    samples = []
    for expert in range(layer.num_local_experts):
        rows = [i for i, e in enumerate(indices) if e == expert and int(peers[i]) >= 0]
        if rows:
            samples.extend(sorted({rows[0], rows[-1]}))
    assert samples, "no real routes in audited layer"
    sample_gpu = torch.tensor(samples, dtype=torch.long, device=device)

    def sample(tensor):
        return tensor.view(torch.uint8)[sample_gpu].contiguous().view(tensor.dtype).cpu()

    snapshots = {"samples": torch.tensor(samples), "indices": torch.tensor(indices),
                 "group_source": torch.tensor(source), "group_inverse": torch.tensor(inverse),
                 "peer_rank": peers, "peer_token_idx": schedule.peer_token_idx[:n].cpu(),
                 "routed_x": sample(state.routed_x), "routed_x_scale": sample(state.routed_x_scale),
                 "candidate_output": output.cpu(), "statistics": {}, "pairs": {}}
    results = {}
    for name, (actual, reference) in pairs.items():
        results[name], snapshots["statistics"][name] = comparison(actual, reference)
        snapshots["pairs"][name] = {"actual": sample(actual), "reference": sample(reference)}
    assert torch.equal(output, original_output), "diagnostic changed candidate output"
    import deep_gemm
    receipt = {
        "schema": 1, "rank": rank, "layer": layer_id, "variant": variant,
        "input_tokens": input_tokens, "active_rows_including_padding": n,
        "real_routes": int((peers >= 0).sum()), "deepgemm_padded_rows": len(source),
        "sample_rows": samples, "metrics": results,
        "reference": "SGLang contiguous grouped DeepGEMM; same live weights/quantized inputs",
        "reference_alignment": 128, "recipe_a": None, "recipe_b": None,
        "w13_scope": "component replay; fused kernel does not materialize gate/up",
        "output_preserved": True, "verdict": "DIAGNOSTIC_ONLY_NOT_QUALITY_OR_PERFORMANCE_GO",
        "component_thresholds": {"exact_fraction_min": .999, "max_row_relative_l2_max": .001},
        "component_gates": {name: results[name]["exact_fraction"] >= .999
                            and results[name]["max_row_relative_l2"] <= .001
                            for name in ("w13_component", "w2_same_input")},
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "deepgemm_version": deep_gemm.__version__,
        "source_sha256": {"audit": _sha(__file__), "mok_extension": _sha(_C.__file__),
                          "deepgemm_binary": _sha(Path(deep_gemm.__file__).parent / "_C.so"),
                          "grouped_wrapper": _sha(inspect.getsourcefile(deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig)),
                          "activation": _sha(inspect.getsourcefile(silu_and_mul_contig_post_quant_dynamic))},
    }
    stem = f"rank{rank}-layer{layer_id}"
    tensor_path = directory / f"{stem}.pt"
    if tensor_path.exists() or (directory / f"{stem}.json").exists():
        raise RuntimeError("refusing to overwrite an existing live audit")
    torch.save(snapshots, tensor_path)
    receipt["tensors_sha256"] = _sha(tensor_path)
    (directory / f"{stem}.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print("MOK_LIVE_AUDIT " + json.dumps({"rank": rank, "layer": layer_id,
          "active_rows": n, "component_gates": receipt["component_gates"]}), flush=True)
    _SEEN.add(layer_id)
    # Diagnostic/JIT costs differ across ranks. Do not let a faster rank enter
    # the next MoK spin protocol while another is compiling its reference.
    dist.barrier(group=get_tp_group().device_group)
