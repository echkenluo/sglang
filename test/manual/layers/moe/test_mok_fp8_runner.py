"""Manual SM90 correctness gate for the SGLang MoK FP8 runner path."""

import os
import threading

os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
os.environ["SGLANG_MASKED_GEMM_FAST_ACT"] = "0"

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner import mok_fp8_native
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.mok_fp8_native import (
    _accept_runtime_contract,
    _capacity_factor_from_global_counts,
    _conservative_route_capacity_factor,
    _route_padding_config,
    native_shape_contract_error,
)


class _NoTrapFunctional:
    polled = threading.Event()

    @staticmethod
    def format_trap_record(workspace):
        _NoTrapFunctional.polled.set()
        return None


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


def test_mok_fp8_native_static_contract():
    tokens, topk = 17, 3
    local_experts, ep_size = 2, 4
    hidden, intermediate = 256, 256
    hidden_states = torch.empty((tokens, hidden), dtype=torch.bfloat16)
    topk_ids = torch.zeros((tokens, topk), dtype=torch.int64)
    topk_weights = torch.empty((tokens, topk), dtype=torch.float32)
    w13 = torch.empty(
        (local_experts, 2 * intermediate, hidden), dtype=torch.float8_e4m3fn
    )
    w2 = torch.empty((local_experts, hidden, intermediate), dtype=torch.float8_e4m3fn)
    w13_scale = torch.empty(
        (local_experts, 2 * intermediate // 128, hidden // 128),
        dtype=torch.float32,
    )
    w2_scale = torch.empty(
        (local_experts, hidden // 128, intermediate // 128),
        dtype=torch.float32,
    )

    args = (
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w13_scale,
        w2,
        w2_scale,
    )
    kwargs = dict(
        num_local_experts=local_experts,
        num_global_experts=local_experts * ep_size,
        ep_size=ep_size,
    )
    assert native_shape_contract_error(*args, **kwargs) is None
    assert "global experts" in native_shape_contract_error(
        *args, **(kwargs | {"num_global_experts": 7})
    )
    assert "block scales" in native_shape_contract_error(
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w13_scale.to(torch.bfloat16),
        w2,
        w2_scale,
        **kwargs,
    )


def test_mok_fp8_native_strict_contract_skips_collective():
    assert _accept_runtime_contract(None, None, None, strict=True)
    with pytest.raises(RuntimeError, match="strict full-native MoK contract"):
        _accept_runtime_contract("unsupported shape", None, None, strict=True)


def test_mok_fp8_native_trap_watchdog_shutdown_is_idempotent_and_restartable():
    mok_fp8_native.shutdown_trap_watchdog()
    first_workspace = object()
    second_workspace = object()
    first_thread = None
    try:
        _NoTrapFunctional.polled.clear()
        mok_fp8_native._register_trap_watchdog(first_workspace, _NoTrapFunctional)
        first_thread = mok_fp8_native._TRAP_WATCHDOG_THREAD
        assert first_thread is not None
        assert first_thread.is_alive()
        assert first_thread.daemon
        assert _NoTrapFunctional.polled.wait(timeout=1)

        mok_fp8_native.shutdown_trap_watchdog()
        assert not first_thread.is_alive()
        assert mok_fp8_native._TRAP_WATCHDOG_THREAD is None
        assert not mok_fp8_native._TRAP_WATCHDOG_STARTED
        assert not mok_fp8_native._TRAP_WATCHDOG_ENTRIES
        assert not mok_fp8_native._TRAP_WATCHDOG_STOP.is_set()

        # A second call is a no-op, and a later registration gets a fresh
        # poller rather than inheriting the stopped Event/thread.
        mok_fp8_native.shutdown_trap_watchdog()
        _NoTrapFunctional.polled.clear()
        mok_fp8_native._register_trap_watchdog(second_workspace, _NoTrapFunctional)
        second_thread = mok_fp8_native._TRAP_WATCHDOG_THREAD
        assert second_thread is not None
        assert second_thread is not first_thread
        assert second_thread.is_alive()
        assert _NoTrapFunctional.polled.wait(timeout=1)
    finally:
        mok_fp8_native.shutdown_trap_watchdog()


def test_mok_fp8_native_capacity_includes_expert_padding():
    # V4 has 64 local experts. Even light balanced traffic consumes one padded
    # segment per nonempty expert.
    balanced = torch.full((256,), 48, dtype=torch.int64)
    assert (
        _capacity_factor_from_global_counts(
            balanced,
            base_rows=512 * 6,
            num_local_experts=64,
            ep_size=4,
        )
        == 6
    )
    assert (
        _capacity_factor_from_global_counts(
            balanced,
            base_rows=512 * 6,
            num_local_experts=64,
            ep_size=4,
            expert_padding=64,
        )
        == 2
    )

    concentrated = torch.zeros((256,), dtype=torch.int64)
    concentrated[0] = 4 * 512 * 6
    assert (
        _capacity_factor_from_global_counts(
            concentrated,
            base_rows=512 * 6,
            num_local_experts=64,
            ep_size=4,
        )
        == 4
    )

    conservative = _conservative_route_capacity_factor(
        base_rows=512 * 6,
        num_local_experts=64,
        ep_size=4,
        expert_padding=64,
    )
    assert conservative == 6
    assert (
        _capacity_factor_from_global_counts(
            concentrated,
            base_rows=512 * 6,
            num_local_experts=64,
            ep_size=4,
            expert_padding=64,
        )
        <= conservative
    )


@pytest.mark.parametrize(
    ("num_tokens", "topk", "expected_tokens", "expected_chunk_bytes"),
    [
        (1, 6, 2, 16),
        (2, 6, 2, 16),
        (3, 6, 4, 16),
        (4, 6, 4, 16),
        (1, 3, 4, 16),
        (5, 6, 256, 1024),
        (257, 6, 512, 1024),
    ],
)
def test_mok_fp8_native_route_padding(
    num_tokens, topk, expected_tokens, expected_chunk_bytes
):
    padded_tokens, chunk_bytes = _route_padding_config(num_tokens, topk)
    assert padded_tokens == expected_tokens
    assert chunk_bytes == expected_chunk_bytes
    assert padded_tokens % 2 == 0
    assert padded_tokens * topk * 4 % chunk_bytes == 0


@pytest.mark.parametrize(("num_tokens", "topk"), [(0, 6), (1, 0)])
def test_mok_fp8_native_route_padding_rejects_invalid_counts(num_tokens, topk):
    with pytest.raises(ValueError, match="must be positive"):
        _route_padding_config(num_tokens, topk)


@pytest.mark.parametrize("base_rows", [12, 24, 512, 3072, 6144, 12288])
def test_mok_fp8_native_conservative_capacity_bounds_routes(base_rows):
    num_local_experts, ep_size, expert_padding = 64, 4, 64
    bound = _conservative_route_capacity_factor(
        base_rows=base_rows,
        num_local_experts=num_local_experts,
        ep_size=ep_size,
        expert_padding=expert_padding,
    )
    assert bound * base_rows % 256 == 0
    generator = torch.Generator().manual_seed(20260816 + base_rows)
    for _ in range(32):
        destination = int(torch.randint(ep_size, (), generator=generator))
        counts = torch.zeros(num_local_experts * ep_size, dtype=torch.int64)
        routed_experts = torch.randint(
            num_local_experts,
            (ep_size * base_rows,),
            generator=generator,
        )
        counts[
            destination * num_local_experts : (destination + 1) * num_local_experts
        ] = torch.bincount(routed_experts, minlength=num_local_experts)
        actual = _capacity_factor_from_global_counts(
            counts,
            base_rows=base_rows,
            num_local_experts=num_local_experts,
            ep_size=ep_size,
            expert_padding=expert_padding,
        )
        assert actual <= bound


def test_mok_fp8_masked_runner_matches_deepgemm():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda", 0)
    if torch.cuda.get_device_capability(device) != (9, 0):
        pytest.skip("MoK FP8 runner currently requires SM90")
    try:
        from mok import _C
    except ImportError:
        pytest.skip("MoK extension is not importable")
    if not hasattr(_C, "fp8_block_grouped_pipelined_out"):
        pytest.skip("MoK extension does not expose the stable FP8 grouped API")

    generator = torch.Generator(device=device).manual_seed(20260819)
    experts, max_m, hidden, intermediate = 2, 64, 512, 512
    valid_rows = (64, 32)
    hidden_states = _fp8_random((experts, max_m, hidden), generator, device)
    w13 = _fp8_random((experts, 2 * intermediate, hidden), generator, device)
    w2 = _fp8_random((experts, hidden, intermediate), generator, device)
    hidden_scale = (
        torch.rand(
            (experts, max_m, hidden // 128),
            generator=generator,
            device=device,
        )
        * 0.09
        + 0.01
    )
    w13_scale = (
        torch.rand(
            (experts, 2 * intermediate // 128, hidden // 128),
            generator=generator,
            device=device,
        )
        * 0.09
        + 0.01
    )
    w2_scale = (
        torch.rand(
            (experts, hidden // 128, intermediate // 128),
            generator=generator,
            device=device,
        )
        * 0.09
        + 0.01
    )
    masked_m = torch.tensor(valid_rows, dtype=torch.int32, device=device)

    config = MoeRunnerConfig(
        num_experts=experts,
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=2,
        activation="silu",
        is_gated=True,
        swiglu_limit=10,
    )
    core = DeepGemmRunnerCore(config)
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        use_fp8=True,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[128, 128],
    )
    running_state = {"hidden_states_device": device}

    def runner_input():
        return DeepGemmRunnerInput(
            hidden_states=hidden_states.clone(),
            hidden_states_scale=hidden_scale.clone(),
            use_masked_gemm=True,
            masked_m=masked_m,
            expected_m=48,
        )

    with (
        envs.SGLANG_OPT_MOK_FP8_MIN_EXPECTED_M.override(0),
        envs.SGLANG_OPT_USE_MOK_FP8_EXPERT_MLP.override(True),
    ):
        reason = core._mok_fp8_unsupported_reason(
            runner_input(), quant_info, running_state
        )
        assert reason is None, reason
        mok_output = core._run_masked_gemm(runner_input(), quant_info, running_state)
    with envs.SGLANG_OPT_USE_MOK_FP8_EXPERT_MLP.override(False):
        deepgemm_output = core._run_masked_gemm(
            runner_input(), quant_info, running_state
        )
    torch.cuda.synchronize(device)

    actual = torch.cat(
        [mok_output[expert, :rows] for expert, rows in enumerate(valid_rows)]
    ).float()
    reference = torch.cat(
        [deepgemm_output[expert, :rows] for expert, rows in enumerate(valid_rows)]
    ).float()
    error = (actual - reference).abs()
    rel_maxnorm = error.max() / reference.abs().max().clamp_min(1e-6)
    relative_l2 = torch.linalg.vector_norm(
        actual - reference
    ) / torch.linalg.vector_norm(reference).clamp_min(1e-6)
    assert rel_maxnorm.item() < 0.05, rel_maxnorm.item()
    assert relative_l2.item() < 0.05, relative_l2.item()


def test_mok_fp8_contiguous_runner_matches_deepgemm():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda", 0)
    if torch.cuda.get_device_capability(device) != (9, 0):
        pytest.skip("MoK FP8 runner currently requires SM90")
    try:
        from mok import _C
    except ImportError:
        pytest.skip("MoK extension is not importable")
    if not hasattr(_C, "fp8_block_grouped_contiguous_out"):
        pytest.skip("MoK extension does not expose the contiguous FP8 API")

    generator = torch.Generator(device=device).manual_seed(20260820)
    experts, hidden, intermediate = 2, 512, 512
    rows = (128, 256)
    total_m = sum(rows)
    hidden_states = _fp8_random((total_m, hidden), generator, device)
    w13 = _fp8_random((experts, 2 * intermediate, hidden), generator, device)
    w2 = _fp8_random((experts, hidden, intermediate), generator, device)
    hidden_scale = (
        torch.rand((total_m, hidden // 128), generator=generator, device=device) * 0.09
        + 0.01
    )
    w13_scale = (
        torch.rand(
            (experts, 2 * intermediate // 128, hidden // 128),
            generator=generator,
            device=device,
        )
        * 0.09
        + 0.01
    )
    w2_scale = (
        torch.rand(
            (experts, hidden // 128, intermediate // 128),
            generator=generator,
            device=device,
        )
        * 0.09
        + 0.01
    )
    m_indices = torch.repeat_interleave(
        torch.arange(experts, dtype=torch.int32, device=device),
        torch.tensor(rows, dtype=torch.int64, device=device),
    )

    config = MoeRunnerConfig(
        num_experts=experts,
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=2,
        activation="silu",
        is_gated=True,
        swiglu_limit=10,
    )
    core = DeepGemmRunnerCore(config)
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        use_fp8=True,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[128, 128],
    )
    running_state = {
        "all_tokens": total_m,
        "hidden_states_device": device,
        "hidden_states_dtype": torch.float8_e4m3fn,
        "hidden_states_shape": (total_m, hidden),
    }

    def runner_input():
        return DeepGemmRunnerInput(
            hidden_states=hidden_states.clone(),
            hidden_states_scale=hidden_scale.clone(),
            use_masked_gemm=False,
            m_indices=m_indices,
        )

    with (
        envs.SGLANG_OPT_MOK_FP8_MIN_EXPECTED_M.override(0),
        envs.SGLANG_OPT_USE_MOK_FP8_EXPERT_MLP.override(True),
    ):
        reason = core._mok_fp8_contiguous_unsupported_reason(
            runner_input(), quant_info, running_state
        )
        assert reason is None, reason
        mok_output = core._run_contiguous_gemm(
            runner_input(), quant_info, running_state
        )
    with envs.SGLANG_OPT_USE_MOK_FP8_EXPERT_MLP.override(False):
        deepgemm_output = core._run_contiguous_gemm(
            runner_input(), quant_info, running_state
        )
    torch.cuda.synchronize(device)

    actual = mok_output.float()
    reference = deepgemm_output.float()
    error = (actual - reference).abs()
    rel_maxnorm = error.max() / reference.abs().max().clamp_min(1e-6)
    relative_l2 = torch.linalg.vector_norm(
        actual - reference
    ) / torch.linalg.vector_norm(reference).clamp_min(1e-6)
    assert rel_maxnorm.item() < 0.05, rel_maxnorm.item()
    assert relative_l2.item() < 0.05, relative_l2.item()
