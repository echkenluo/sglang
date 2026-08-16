"""Manual SM90 correctness gate for the SGLang MoK FP8 runner path."""

import os

os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
os.environ["SGLANG_MASKED_GEMM_FAST_ACT"] = "0"

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
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
    hidden_states = _fp8_random(
        (experts, max_m, hidden), generator, device
    )
    w13 = _fp8_random(
        (experts, 2 * intermediate, hidden), generator, device
    )
    w2 = _fp8_random((experts, hidden, intermediate), generator, device)
    hidden_scale = torch.rand(
        (experts, max_m, hidden // 128),
        generator=generator,
        device=device,
    ) * 0.09 + 0.01
    w13_scale = torch.rand(
        (experts, 2 * intermediate // 128, hidden // 128),
        generator=generator,
        device=device,
    ) * 0.09 + 0.01
    w2_scale = torch.rand(
        (experts, hidden // 128, intermediate // 128),
        generator=generator,
        device=device,
    ) * 0.09 + 0.01
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
        mok_output = core._run_masked_gemm(
            runner_input(), quant_info, running_state
        )
    with envs.SGLANG_OPT_USE_MOK_FP8_EXPERT_MLP.override(False):
        deepgemm_output = core._run_masked_gemm(
            runner_input(), quant_info, running_state
        )
    torch.cuda.synchronize(device)

    actual = torch.cat(
        [mok_output[expert, :rows] for expert, rows in enumerate(valid_rows)]
    ).float()
    reference = torch.cat(
        [
            deepgemm_output[expert, :rows]
            for expert, rows in enumerate(valid_rows)
        ]
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
    hidden_states = _fp8_random(
        (total_m, hidden), generator, device
    )
    w13 = _fp8_random(
        (experts, 2 * intermediate, hidden), generator, device
    )
    w2 = _fp8_random((experts, hidden, intermediate), generator, device)
    hidden_scale = torch.rand(
        (total_m, hidden // 128), generator=generator, device=device
    ) * 0.09 + 0.01
    w13_scale = torch.rand(
        (experts, 2 * intermediate // 128, hidden // 128),
        generator=generator,
        device=device,
    ) * 0.09 + 0.01
    w2_scale = torch.rand(
        (experts, hidden // 128, intermediate // 128),
        generator=generator,
        device=device,
    ) * 0.09 + 0.01
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
