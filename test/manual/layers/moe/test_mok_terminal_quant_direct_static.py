"""Dependency-free source contracts for direct terminal FP8 quantization."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FP8_KERNEL = ROOT / "python/sglang/srt/layers/quantization/fp8_kernel.py"
TERMINAL_RUNNER = (
    ROOT / "python/sglang/srt/layers/moe/moe_runner/mok_fp8_native.py"
)
JIT_V2 = ROOT / "python/sglang/jit_kernel/per_token_group_quant_8bit_v2.py"
AOT_SCHEMA = ROOT / "sgl-kernel/csrc/common_extension.cc"


def _function_source(path: Path, name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                segment = ast.get_source_segment(source, node)
                assert segment is not None
                return segment
    raise AssertionError(f"function {name} not found in {path}")


def test_quant_out_reuses_the_allocating_numerical_dispatch():
    allocating = _function_source(
        FP8_KERNEL, "sglang_per_token_group_quant_fp8"
    )
    caller_owned = _function_source(
        FP8_KERNEL, "sglang_per_token_group_quant_fp8_out"
    )
    helper = _function_source(
        FP8_KERNEL, "_run_sglang_per_token_group_quant_fp8_out"
    )
    shared_call = "_run_sglang_per_token_group_quant_fp8_out("
    assert shared_call in allocating
    assert shared_call in caller_owned
    assert "torch.empty(" not in caller_owned
    for dispatch in (
        "sgl_per_token_group_quant_8bit(",
        "sgl_per_token_group_quant_8bit_jit_v2(",
        "sgl_per_token_group_quant_8bit_jit(",
        "sgl_per_token_group_quant_fp8(",
    ):
        assert dispatch in helper


def test_quant_backends_declare_caller_owned_outputs_mutated():
    jit = JIT_V2.read_text()
    assert 'mutates_args=["output_q", "output_s"]' in jit
    schema = AOT_SCHEMA.read_text()
    assert (
        "sgl_per_token_group_quant_8bit(Tensor input, Tensor! output_q, "
        "Tensor! output_s"
    ) in schema
    assert (
        "sgl_per_token_group_quant_8bit_v2(Tensor input, Tensor! output_q, "
        "Tensor! output_s"
    ) in schema


def test_terminal_quantization_is_inside_the_workspace_lease():
    runner = _function_source(TERMINAL_RUNNER, "_run_terminal_eager")
    ordered = (
        "\n    validate_sglang_per_token_group_quant_fp8_out(",
        "\n    mok_functional.acquire_megakernel_fp8_block_from_topk_lease(",
        "\n    sglang_per_token_group_quant_fp8_out(",
        "\n    return mok_functional."
        "megakernel_fp8_block_from_topk_preloaded_leased(",
    )
    positions = [runner.find(needle) for needle in ordered]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
    for forbidden in (
        "sglang_per_token_group_quant_fp8(",
        "megakernel_fp8_block_from_topk(",
        "workspace.x_buffer.copy_(",
        "workspace.x_scale_buffer.copy_(",
        "release_workspace_lease(",
    ):
        assert forbidden not in runner
