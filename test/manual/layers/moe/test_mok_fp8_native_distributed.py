"""Four-rank correctness gate for the full-native MoK FP8 path."""

import os
from types import SimpleNamespace

os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
os.environ["SGLANG_MASKED_GEMM_FAST_ACT"] = "0"

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
)
from sglang.srt.layers.moe.moe_runner import mok_fp8_native
from sglang.srt.layers.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)


def _fp8_random(shape, generator, device):
    return (
        torch.randn(
            shape,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        .clamp(-3, 3)
        .to(torch.float8_e4m3fn)
    )


def _fake_layer(
    *,
    rank: int,
    world_size: int,
    hidden: int,
    intermediate: int,
    local_experts: int,
    topk: int = 2,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
):
    fp8_method_type = type("Fp8MoEMethod", (), {})
    quant_method = fp8_method_type()
    quant_method.quant_config = SimpleNamespace(weight_block_size=[128, 128])
    quant_method.is_fp4_expert = False
    quant_method.with_bias = False
    runner_config = MoeRunnerConfig(
        num_experts=local_experts * world_size,
        num_local_experts=local_experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        layer_id=4,
        top_k=topk,
        activation="silu",
        is_gated=True,
        swiglu_limit=10,
    )
    return SimpleNamespace(
        quant_method=quant_method,
        moe_runner_config=runner_config,
        w13_weight=w13,
        w13_weight_scale_inv=w13_scale,
        w2_weight=w2,
        w2_weight_scale_inv=w2_scale,
        num_local_experts=local_experts,
        num_experts=local_experts * world_size,
        moe_ep_size=world_size,
        moe_ep_rank=rank,
        layer_id=4,
    )


def test_mok_fp8_native_full_chain_matches_deepgemm():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    assert torch.cuda.get_device_capability(device) == (9, 0)
    dist.init_process_group("nccl")
    try:
        _run_mok_fp8_native_full_chain_test(device)
    finally:
        # Stop CPU polling before ProcessGroup/CUDA teardown.  Leaving the
        # daemon alive here can race PyTorch interpreter finalization and
        # abort an otherwise successful distributed test.
        mok_fp8_native.shutdown_trap_watchdog()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_mok_fp8_native_full_chain_test(device):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4

    from mok import functional as mok_functional

    mok_fp8_native.get_tp_group = lambda: SimpleNamespace(device_group=dist.group.WORLD)
    generator = torch.Generator(device=device).manual_seed(20260821)
    # An unaligned token count exercises padding, invalid route slots, active-row
    # compute, and the caller-owned combine-buffer clearing contract together.
    tokens = int(os.environ.get("MOK_TEST_TOKENS", "257"))
    topk = 2
    local_experts, hidden, intermediate = 2, 256, 256
    hidden_states = torch.randn(
        (tokens, hidden),
        generator=torch.Generator(device=device).manual_seed(20260900 + rank),
        device=device,
        dtype=torch.bfloat16,
    ).clamp(-2, 2)
    w13 = _fp8_random((local_experts, 2 * intermediate, hidden), generator, device)
    w2 = _fp8_random((local_experts, hidden, intermediate), generator, device)
    w13_scale = (
        torch.rand(
            (local_experts, 2 * intermediate // 128, hidden // 128),
            generator=generator,
            device=device,
        )
        * 0.09
        + 0.01
    )
    w2_scale = (
        torch.rand(
            (local_experts, hidden // 128, intermediate // 128),
            generator=generator,
            device=device,
        )
        * 0.09
        + 0.01
    )
    token = torch.arange(tokens, device=device, dtype=torch.int64)
    topk_ids = torch.stack(
        (
            (token + rank) % (local_experts * world_size),
            (token * 3 + 5 + rank) % (local_experts * world_size),
        ),
        dim=1,
    )
    topk_weights = (
        torch.tensor([0.625, 0.375], dtype=torch.float32, device=device)
        .expand(tokens, -1)
        .contiguous()
    )
    topk_output = SimpleNamespace(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    layer = _fake_layer(
        rank=rank,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        local_experts=local_experts,
        w13=w13,
        w13_scale=w13_scale,
        w2=w2,
        w2_scale=w2_scale,
    )

    actual = mok_fp8_native.maybe_run_mok_fp8_native(layer, hidden_states, topk_output)
    assert actual is not None

    if os.environ.get("MOK_TEST_CUDA_GRAPH") == "1":
        # The eager call above creates every shape-dependent route workspace
        # and JIT kernel before capture.  Capture and replay must then remain
        # entirely device driven, including the routed-row count and EP
        # barriers, on all four ranks.
        eager_actual = actual.clone()
        torch.cuda.synchronize(device)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_actual = mok_fp8_native.maybe_run_mok_fp8_native(
                layer, hidden_states, topk_output
            )
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(
            captured_actual, eager_actual, rtol=0, atol=0
        )

        # Keep every captured address and shape stable, but route every token
        # to rank 0 and change the activations/weights in place.  This changes
        # the replay-time active row count from a balanced distribution to
        # T * topk * world_size on rank 0 and zero on the other ranks.  The
        # graph must consume the device-side count produced by this replay,
        # rather than a host value frozen during capture.
        hidden_states.mul_(0.5)
        topk_ids[:, 0].fill_(0)
        topk_ids[:, 1].fill_(1)
        topk_weights[:, 0].fill_(0.25)
        topk_weights[:, 1].fill_(0.75)
        dynamic_eager = mok_fp8_native.maybe_run_mok_fp8_native(
            layer, hidden_states, topk_output
        ).clone()
        torch.cuda.synchronize(device)
        assert not torch.equal(dynamic_eager, eager_actual)

        dist.barrier()
        graph.replay()
        torch.cuda.synchronize(device)
        actual = captured_actual.clone()
        torch.testing.assert_close(actual, dynamic_eager, rtol=0, atol=0)
        print(
            f"MOK_CUDA_GRAPH_DYNAMIC|rank={rank}|T={tokens}|"
            f"expected_active_rows="
            f"{tokens * topk * world_size if rank == 0 else 0}",
            flush=True,
        )

    input_fp8, input_scale = sglang_per_token_group_quant_fp8(
        hidden_states,
        128,
        column_major_scales=False,
        scale_tma_aligned=False,
        scale_ue8m0=False,
    )
    route_local_experts = (topk_ids % local_experts).reshape(-1)
    flat_route_input = (
        input_fp8[:, None, :]
        .expand(tokens, topk, hidden)
        .reshape(tokens * topk, hidden)
        .contiguous()
    )
    flat_route_scale = (
        input_scale[:, None, :]
        .expand(tokens, topk, hidden // 128)
        .reshape(tokens * topk, hidden // 128)
        .contiguous()
    )
    # DeepGemm's contiguous grouped kernel requires aligned expert segment
    # bases.  Build an independent M256-padded reference layout rather than
    # placing expert 1 immediately after an unaligned expert-0 row count.
    route_counts = torch.bincount(route_local_experts, minlength=local_experts)
    padded_route_counts = (
        torch.div(route_counts + 255, 256, rounding_mode="floor") * 256
    )
    reference_rows = int(padded_route_counts.sum().item())
    route_input = torch.zeros(
        (reference_rows, hidden), dtype=input_fp8.dtype, device=device
    )
    route_input_scale = torch.zeros(
        (reference_rows, hidden // 128),
        dtype=input_scale.dtype,
        device=device,
    )
    m_indices = torch.repeat_interleave(
        torch.arange(local_experts, dtype=torch.int32, device=device),
        padded_route_counts,
    )
    route_positions = torch.empty(tokens * topk, dtype=torch.int64, device=device)
    expert_base = 0
    for expert in range(local_experts):
        source_rows = torch.nonzero(
            route_local_experts == expert, as_tuple=False
        ).flatten()
        destination_rows = expert_base + torch.arange(
            source_rows.numel(), dtype=torch.int64, device=device
        )
        route_input[destination_rows] = flat_route_input[source_rows]
        route_input_scale[destination_rows] = flat_route_scale[source_rows]
        route_positions[source_rows] = destination_rows
        expert_base += int(padded_route_counts[expert].item())
    runner_config = layer.moe_runner_config
    core = DeepGemmRunnerCore(runner_config)
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        use_fp8=True,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[128, 128],
    )
    running_state = {
        "all_tokens": reference_rows,
        "hidden_states_device": device,
        "hidden_states_dtype": torch.float8_e4m3fn,
        "hidden_states_shape": (reference_rows, hidden),
    }
    with envs.SGLANG_OPT_USE_MOK_FP8_EXPERT_MLP.override(False):
        sorted_reference = core._run_contiguous_gemm(
            DeepGemmRunnerInput(
                hidden_states=route_input,
                hidden_states_scale=route_input_scale,
                use_masked_gemm=False,
                m_indices=m_indices,
            ),
            quant_info,
            running_state,
        )
    route_reference = sorted_reference[route_positions]
    reference = (
        route_reference.view(tokens, topk, hidden).float() * topk_weights[:, :, None]
    ).sum(dim=1)

    error = (actual.float() - reference).abs()
    metrics = torch.stack(
        (
            error.max() / reference.abs().max().clamp_min(1e-6),
            torch.linalg.vector_norm(actual.float() - reference)
            / torch.linalg.vector_norm(reference).clamp_min(1e-6),
        )
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    print(
        f"MOK_NATIVE_FULL_CHAIN|rank={rank}|"
        f"T={tokens}|rel_maxnorm={metrics[0].item():.8f}|"
        f"relative_l2={metrics[1].item():.8f}|"
        f"actual_norm={torch.linalg.vector_norm(actual.float()).item():.6f}|"
        f"reference_norm={torch.linalg.vector_norm(reference).item():.6f}|"
        f"actual_zero_rows={(actual == 0).all(dim=1).sum().item()}|"
        f"reference_zero_rows={(reference == 0).all(dim=1).sum().item()}",
        flush=True,
    )
    mok_functional.clear_workspace_cache()
    dist.barrier()
    assert metrics[0].item() < 0.05
    assert metrics[1].item() < 0.05
