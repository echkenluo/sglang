"""Dependency-free source contracts for direct terminal FP8 quantization."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FP8_KERNEL = ROOT / "python/sglang/srt/layers/quantization/fp8_kernel.py"
TERMINAL_RUNNER = (
    ROOT / "python/sglang/srt/layers/moe/moe_runner/mok_fp8_native.py"
)
JIT_V2 = ROOT / "python/sglang/jit_kernel/per_token_group_quant_8bit_v2.py"
JIT_V1 = ROOT / "python/sglang/jit_kernel/per_token_group_quant_8bit.py"
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
    for jit_path in (JIT_V1, JIT_V2):
        jit = jit_path.read_text()
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
    jit_v2 = JIT_V2.read_text()
    assert "_jit_module(input.dtype, output_q.dtype, use_pdl)" in jit_v2


def test_terminal_quantization_is_inside_the_workspace_lease():
    runner = _function_source(TERMINAL_RUNNER, "_run_terminal_eager")
    ordered = (
        "\n    prewarm_receipt = prewarm_sglang_per_token_group_quant_fp8_out(",
        "\n        mok_functional.acquire_megakernel_fp8_block_from_topk_lease(",
        "\n        launch_sglang_per_token_group_quant_fp8_out_prevalidated(",
        "\n        return mok_functional."
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
    assert "except BaseException as error:" in runner
    assert "_fatal_terminal_transaction_failure(" in runner


def test_quant_prewarm_materializes_before_capture_and_returns_receipt():
    prewarm = _function_source(
        FP8_KERNEL, "prewarm_sglang_per_token_group_quant_fp8_out"
    )
    assert "validate_sglang_per_token_group_quant_fp8_out(" in prewarm
    assert "torch.cuda.is_current_stream_capturing()" in prewarm
    assert "must be prewarmed before capture" in prewarm
    assert "_jit_per_token_group_quant_8bit_v2_module(" in prewarm
    assert "_jit_per_token_group_quant_8bit_module(" in prewarm
    ordered = (
        "scratch_q = torch.empty_like(x_q)",
        "_run_sglang_per_token_group_quant_fp8_out(",
        "_synchronize_fp8_out_prewarm(x)",
        "_FP8_OUT_PREWARM_RECEIPTS[contract] = receipt",
    )
    positions = [prewarm.find(needle) for needle in ordered]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
    assert "resolved_use_pdl=receipt.use_pdl" in prewarm

    launch = _function_source(
        FP8_KERNEL,
        "launch_sglang_per_token_group_quant_fp8_out_prevalidated",
    )
    assert "validate_sglang_per_token_group_quant_fp8_out(" not in launch
    assert "_run_sglang_per_token_group_quant_fp8_out(" in launch
    assert "resolved_backend=contract.backend" in launch
    assert "resolved_use_pdl=prewarm_receipt.use_pdl" in launch


def test_post_acquire_failure_endpoint_is_cpu_only_and_unconditional():
    fatal = _function_source(
        TERMINAL_RUNNER, "_fatal_terminal_transaction_failure"
    )
    assert "format_terminal_transaction_failure(" in fatal
    assert "except BaseException:" in fatal
    assert "os._exit(70)" in fatal
    for forbidden in (
        "release_workspace_lease(",
        "torch.cuda",
        "synchronize(",
    ):
        assert forbidden not in fatal
