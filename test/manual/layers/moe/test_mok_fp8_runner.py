"""Manual SM90 correctness gate for the SGLang MoK FP8 runner path."""

import inspect
import logging
import os
import threading
from types import SimpleNamespace

os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
os.environ["SGLANG_MASKED_GEMM_FAST_ACT"] = "0"

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
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
    _pad_eager_inputs,
    _route_padding_config,
    _terminal_graph_context_error,
    native_shape_contract_error,
    native_terminal_contract_error,
    terminal_deepep_outer_context,
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


def _terminal_contract_fixture():
    layer = DeepEPMoE.__new__(DeepEPMoE)
    torch.nn.Module.__init__(layer)
    layer.moe_ep_size = 4
    layer.num_experts = 256
    layer.num_local_experts = 64
    layer.num_fused_shared_experts = 0
    layer._has_fused_shared = False
    layer.reduce_results = False
    layer.deprecate_flag = False
    layer.quant_method = SimpleNamespace(load_up_proj_weight_first=False)
    layer.w13_weight = torch.nn.Parameter(
        torch.empty((64, 4096, 4096), dtype=torch.float8_e4m3fn, device="meta"),
        requires_grad=False,
    )
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.empty((64, 32, 32), dtype=torch.float32, device="meta"),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.empty((64, 4096, 2048), dtype=torch.float8_e4m3fn, device="meta"),
        requires_grad=False,
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.empty((64, 32, 16), dtype=torch.float32, device="meta"),
        requires_grad=False,
    )
    hidden_states = torch.empty((2, 4096), dtype=torch.bfloat16)
    topk_output = SimpleNamespace(
        topk_ids=torch.zeros((2, 6), dtype=torch.int64),
        topk_weights=torch.zeros((2, 6), dtype=torch.float32),
    )
    return layer, hidden_states, topk_output


def test_mok_fp8_terminal_exact_contract(monkeypatch):
    monkeypatch.setattr(
        mok_fp8_native,
        "native_runtime_contract_error",
        lambda layer, hidden_states, topk_output: None,
    )
    monkeypatch.setattr(
        mok_fp8_native, "_terminal_deepep_backend_is_active", lambda: True
    )
    layer, hidden_states, topk_output = _terminal_contract_fixture()
    with terminal_deepep_outer_context():
        assert (
            native_terminal_contract_error(layer, hidden_states, topk_output) is None
        )

        layer.num_fused_shared_experts = 1
        assert "shared-expert fusion disabled" in native_terminal_contract_error(
            layer, hidden_states, topk_output
        )
        layer.num_fused_shared_experts = 0
        layer.quant_method.load_up_proj_weight_first = True
        assert "gate-then-up" in native_terminal_contract_error(
            layer, hidden_states, topk_output
        )
        layer.quant_method.load_up_proj_weight_first = False
        layer.moe_ep_size = 8
        assert "EP4" in native_terminal_contract_error(
            layer, hidden_states, topk_output
        )


def test_mok_fp8_terminal_outer_contract_is_fail_closed(monkeypatch):
    monkeypatch.setattr(
        mok_fp8_native,
        "native_runtime_contract_error",
        lambda layer, hidden_states, topk_output: None,
    )
    layer, hidden_states, topk_output = _terminal_contract_fixture()

    monkeypatch.setattr(
        mok_fp8_native, "_terminal_deepep_backend_is_active", lambda: True
    )
    assert "forward_deepep outer semantics" in native_terminal_contract_error(
        layer, hidden_states, topk_output
    )

    with terminal_deepep_outer_context():
        monkeypatch.setattr(
            mok_fp8_native, "_terminal_deepep_backend_is_active", lambda: False
        )
        assert "--moe-a2a-backend deepep" in native_terminal_contract_error(
            layer, hidden_states, topk_output
        )
        monkeypatch.setattr(
            mok_fp8_native, "_terminal_deepep_backend_is_active", lambda: True
        )
        layer.reduce_results = True
        assert "must not be all-reduced" in native_terminal_contract_error(
            layer, hidden_states, topk_output
        )

    with terminal_deepep_outer_context():
        assert "DeepEPMoE production layer" in native_terminal_contract_error(
            SimpleNamespace(), hidden_states, topk_output
        )


def test_mok_fp8_terminal_deepep_override_cannot_fall_back(monkeypatch):
    layer = DeepEPMoE.__new__(DeepEPMoE)
    torch.nn.Module.__init__(layer)
    layer.deprecate_flag = False
    calls = []
    marker = object()

    def terminal_base_forward(self, hidden_states, topk_output):
        calls.append((hidden_states, topk_output))
        return marker

    monkeypatch.setattr(FusedMoE, "forward_impl", terminal_base_forward)
    with envs.SGLANG_OPT_MOK_FP8_NATIVE_TERMINAL.override(True):
        assert layer.forward("hidden", "topk") is marker
        assert layer.forward_impl("hidden-impl", "topk-impl") is marker
    assert calls == [("hidden", "topk"), ("hidden-impl", "topk-impl")]


def test_mok_fp8_terminal_rejects_split_entry_points():
    from sglang.srt.models import deepseek_v2

    layer = DeepEPMoE.__new__(DeepEPMoE)
    torch.nn.Module.__init__(layer)
    moe = deepseek_v2.DeepseekV2MoE.__new__(deepseek_v2.DeepseekV2MoE)
    torch.nn.Module.__init__(moe)

    with envs.SGLANG_OPT_MOK_FP8_NATIVE_TERMINAL.override(True):
        for call, name in (
            (lambda: layer.dispatch("hidden", "topk"), "dispatch"),
            (lambda: layer.run_moe_core("dispatch-output"), "run_moe_core"),
            (
                lambda: layer.forward_deferred_finalize("hidden", "topk"),
                "forward_deferred_finalize",
            ),
        ):
            with pytest.raises(RuntimeError, match=name):
                call()
        with pytest.raises(RuntimeError, match="SBO/TBO"):
            moe.op_gate(object())


def test_mok_fp8_terminal_selects_only_deepep_outer(monkeypatch):
    from sglang.srt.layers.moe import mega_moe
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.models import deepseek_v2

    moe = deepseek_v2.DeepseekV2MoE.__new__(deepseek_v2.DeepseekV2MoE)
    torch.nn.Module.__init__(moe)
    moe._enable_a2a_moe = True
    calls = []
    graph_modes = []
    marker = object()
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    monkeypatch.setattr(mega_moe, "should_use_mega_moe", lambda *args: False)
    monkeypatch.setattr(
        deepseek_v2,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(is_deepep=lambda: True),
    )
    monkeypatch.setattr(
        mok_fp8_native,
        "_terminal_graph_context_error",
        lambda forward_mode=None: graph_modes.append(forward_mode),
    )

    def forward_deepep(self, hidden_states, forward_batch, input_ids_global=None):
        calls.append((hidden_states, forward_batch, input_ids_global))
        return marker

    monkeypatch.setattr(deepseek_v2.DeepseekV2MoE, "forward_deepep", forward_deepep)
    with envs.SGLANG_OPT_MOK_FP8_NATIVE_TERMINAL.override(True):
        assert (
            moe.forward(
                "hidden",
                forward_batch=forward_batch,
                input_ids_global="input-ids",
            )
            is marker
        )
    assert calls == [("hidden", forward_batch, "input-ids")]
    assert graph_modes == [ForwardMode.DECODE]

    moe._enable_a2a_moe = False
    with (
        envs.SGLANG_OPT_MOK_FP8_NATIVE_TERMINAL.override(True),
        pytest.raises(RuntimeError, match="requires the DeepSeekV2 DeepEP outer"),
    ):
        moe.forward("hidden", forward_batch=forward_batch)


def _concurrency_server_args(**overrides):
    values = dict(
        enable_pdmux=False,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("enable_pdmux", "PDMux concurrent graph streams"),
        ("enable_two_batch_overlap", "two-batch overlap (TBO)"),
        ("enable_single_batch_overlap", "single-batch overlap (SBO)"),
    ],
)
def test_mok_fp8_terminal_rejects_concurrent_modes(
    monkeypatch, field, message
):
    from sglang.srt import runtime_context
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        full_decode_cuda_graph_mode,
        model_capture_mode,
    )

    server_args = _concurrency_server_args(**{field: True})
    monkeypatch.setattr(runtime_context, "get_server_args", lambda: server_args)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with (
        envs.SGLANG_OPT_MOK_FP8_NATIVE_PREFILL_GRAPH.override(False),
        envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_DISPATCH_GEMM.override(False),
        envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE.override(False),
    ):
        assert message in mok_fp8_native._terminal_mode_config_error()
        with model_capture_mode(), full_decode_cuda_graph_mode():
            assert message in _terminal_graph_context_error(ForwardMode.DECODE)


def test_mok_fp8_terminal_concurrency_config_is_fail_closed(monkeypatch):
    from sglang.srt import runtime_context

    def no_server_args():
        raise ValueError("Global server args is not set yet!")

    monkeypatch.setattr(runtime_context, "get_server_args", no_server_args)
    assert mok_fp8_native._terminal_concurrency_mode_error() is None

    monkeypatch.setattr(
        runtime_context,
        "get_server_args",
        lambda: SimpleNamespace(enable_pdmux=False),
    )
    with pytest.raises(AttributeError, match="enable_two_batch_overlap"):
        mok_fp8_native._terminal_concurrency_mode_error()


def test_mok_fp8_terminal_allows_only_full_decode_graph(monkeypatch):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        enable_breakable_cuda_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        enable_tc_piecewise_cuda_graph,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        full_decode_cuda_graph_mode,
        model_capture_mode,
    )

    monkeypatch.setattr(
        mok_fp8_native, "_terminal_concurrency_mode_error", lambda: None
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with enable_tc_piecewise_cuda_graph():
        assert "tc_piecewise" in _terminal_graph_context_error(ForwardMode.DECODE)
    with enable_breakable_cuda_graph():
        assert "breakable" in _terminal_graph_context_error(ForwardMode.DECODE)
    with model_capture_mode():
        assert "DecodeCudaGraphRunner Full" in _terminal_graph_context_error(
            ForwardMode.DECODE
        )
        with full_decode_cuda_graph_mode():
            assert _terminal_graph_context_error(ForwardMode.DECODE) is None
            assert (
                "DeepEP ForwardMode.DECODE outer"
                in _terminal_graph_context_error()
            )
            for unsupported_mode in (
                ForwardMode.EXTEND,
                ForwardMode.TARGET_VERIFY,
                ForwardMode.DLLM_EXTEND,
            ):
                assert (
                    "supports only ForwardMode.DECODE"
                    in _terminal_graph_context_error(unsupported_mode)
                )
            with terminal_deepep_outer_context(ForwardMode.DECODE):
                assert _terminal_graph_context_error() is None
            assert (
                "DeepEP ForwardMode.DECODE outer"
                in _terminal_graph_context_error()
            )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert "external active CUDA graph" in _terminal_graph_context_error(
        ForwardMode.DECODE
    )


def test_mok_fp8_terminal_full_decode_scope_excludes_draft_runner():
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_full_decode_cuda_graph_mode,
    )

    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.backend = FullCudaGraphBackend.__new__(FullCudaGraphBackend)
    server_args = _concurrency_server_args()
    runner.model_runner = SimpleNamespace(
        is_draft_worker=False, server_args=server_args
    )
    assert not get_is_full_decode_cuda_graph_mode()
    with runner._capture_mode_scope():
        assert get_is_full_decode_cuda_graph_mode()
    assert not get_is_full_decode_cuda_graph_mode()

    runner.model_runner.is_draft_worker = True
    with runner._capture_mode_scope():
        assert not get_is_full_decode_cuda_graph_mode()

    runner.model_runner.is_draft_worker = False
    for field in (
        "enable_two_batch_overlap",
        "enable_single_batch_overlap",
        "enable_pdmux",
    ):
        setattr(server_args, field, True)
        with runner._capture_mode_scope():
            assert not get_is_full_decode_cuda_graph_mode()
        setattr(server_args, field, False)

    del runner.model_runner.is_draft_worker
    with pytest.raises(AttributeError, match="is_draft_worker"):
        with runner._capture_mode_scope():
            pass

    runner.model_runner = SimpleNamespace(
        is_draft_worker=False,
        server_args=SimpleNamespace(
            enable_two_batch_overlap=False,
            enable_single_batch_overlap=False,
        ),
    )
    with pytest.raises(AttributeError, match="enable_pdmux"):
        with runner._capture_mode_scope():
            pass


def test_mok_fp8_terminal_active_receipt_is_once_per_layer(caplog, monkeypatch):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        full_decode_cuda_graph_mode,
        model_capture_mode,
    )

    monkeypatch.setattr(mok_fp8_native, "_REPORTED_ACTIVE", False)
    monkeypatch.setattr(mok_fp8_native, "_REPORTED_TERMINAL_LAYER_IDS", set())
    monkeypatch.setattr(
        mok_fp8_native, "_terminal_concurrency_mode_error", lambda: None
    )
    workspace = SimpleNamespace(schedule_capacity=4608)

    def report(layer_id, *, terminal_mode, num_tokens=1):
        layer = SimpleNamespace(
            layer_id=layer_id,
            num_local_experts=64,
            deprecate_flag=False,
        )
        mok_fp8_native._report_active(
            layer,
            num_tokens,
            2,
            8,
            workspace,
            True,
            terminal_mode=terminal_mode,
        )

    with caplog.at_level(logging.INFO, logger=mok_fp8_native.__name__):
        report(3, terminal_mode=True)
        report(3, terminal_mode=True, num_tokens=2)
        with (
            model_capture_mode(),
            full_decode_cuda_graph_mode(),
            terminal_deepep_outer_context(ForwardMode.DECODE),
        ):
            report(7, terminal_mode=True)
        report(11, terminal_mode=False)
        report(12, terminal_mode=False)

    receipts = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("MoK full-native FP8 active:")
    ]
    terminal_receipts = [
        message for message in receipts if "terminal=true" in message
    ]
    nonterminal_receipts = [
        message for message in receipts if "terminal=False" in message
    ]
    assert len(terminal_receipts) == 2
    assert len(nonterminal_receipts) == 1
    assert (
        "layer_id=3 class=SimpleNamespace deprecate_flag=false"
        in terminal_receipts[0]
    )
    assert "T=1 padded=2" in terminal_receipts[0]
    assert "capacity=4608" in terminal_receipts[0]
    assert "layer_id=7" in terminal_receipts[1]
    assert all("T=2" not in message for message in terminal_receipts)
    assert "terminal_graph=false eager_only=true" in terminal_receipts[0]
    assert "terminal_graph=true eager_only=false" in terminal_receipts[1]


def test_mok_fp8_terminal_eager_padding_is_route_stable():
    hidden_states = torch.arange(3 * 4, dtype=torch.bfloat16).view(3, 4)
    topk_ids = torch.arange(18, dtype=torch.int64).view(3, 6)
    topk_weights = torch.arange(18, dtype=torch.float32).view(3, 6)
    padded_hidden, padded_ids, padded_weights = _pad_eager_inputs(
        hidden_states,
        SimpleNamespace(topk_ids=topk_ids, topk_weights=topk_weights),
        padded_tokens=4,
    )
    assert torch.equal(padded_hidden[:3], hidden_states)
    assert torch.equal(padded_ids[:3], topk_ids.to(torch.int32))
    assert torch.equal(padded_weights[:3], topk_weights)
    assert torch.count_nonzero(padded_hidden[3]) == 0
    assert torch.all(padded_ids[3] == -1)
    assert torch.count_nonzero(padded_weights[3]) == 0


def test_mok_fp8_terminal_eager_source_has_no_intermediate_path():
    terminal_source = inspect.getsource(mok_fp8_native._run_terminal_eager)
    ordered = (
        "\n    prewarm_receipt = prewarm_sglang_per_token_group_quant_fp8_out(",
        "\n        mok_functional.acquire_megakernel_fp8_block_from_topk_lease(",
        "\n        launch_sglang_per_token_group_quant_fp8_out_prevalidated(",
        "\n        return mok_functional."
        "megakernel_fp8_block_from_topk_preloaded_leased(",
    )
    positions = [terminal_source.find(needle) for needle in ordered]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
    for forbidden in (
        "torch.cat(",
        ".clone(",
        "sglang_per_token_group_quant_fp8(",
        "megakernel_fp8_block_from_topk(",
        "workspace.x_buffer.copy_(",
        "workspace.x_scale_buffer.copy_(",
        "dispatch_gemm_fused_fp8_block(",
        "gemm_combine_fused_fp8_block(",
        "release_workspace_lease(",
    ):
        assert forbidden not in terminal_source
    assert "except BaseException as error:" in terminal_source
    assert "_fatal_terminal_transaction_failure(" in terminal_source

    adapter_source = inspect.getsource(mok_fp8_native.maybe_run_mok_fp8_native)
    terminal_return = adapter_source.find("return output[:num_tokens]")
    graph_branch = adapter_source.find("use_graph = (")
    assert 0 <= terminal_return < graph_branch

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    layer_forward_source = inspect.getsource(FusedMoE.forward)
    terminal_dispatch = layer_forward_source.find(
        "SGLANG_OPT_MOK_FP8_NATIVE_TERMINAL"
    )
    piecewise_dispatch = layer_forward_source.find(
        "is_in_tc_piecewise_cuda_graph"
    )
    assert 0 <= terminal_dispatch < piecewise_dispatch

    deepep_forward_source = inspect.getsource(DeepEPMoE.forward)
    terminal_dispatch = deepep_forward_source.find(
        "SGLANG_OPT_MOK_FP8_NATIVE_TERMINAL"
    )
    piecewise_dispatch = deepep_forward_source.find(
        "is_in_tc_piecewise_cuda_graph"
    )
    assert 0 <= terminal_dispatch < piecewise_dispatch

    deepep_impl_source = inspect.getsource(DeepEPMoE.forward_impl)
    terminal_dispatch = deepep_impl_source.find(
        "SGLANG_OPT_MOK_FP8_NATIVE_TERMINAL"
    )
    legacy_dispatch = deepep_impl_source.find("self.dispatcher.dispatch")
    assert 0 <= terminal_dispatch < legacy_dispatch


class _InjectedTerminalFatalExit(BaseException):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


@pytest.mark.parametrize(
    "injected",
    (RuntimeError("quant-runtime"), ValueError("quant-value")),
    ids=("runtime-error", "non-runtime-error"),
)
def test_mok_terminal_post_acquire_quant_failure_is_fatal(
    monkeypatch, injected
):
    from sglang.srt.layers.quantization import fp8_kernel

    events = []
    receipt = (
        "jit_v2",
        "cpu",
        None,
        torch.bfloat16,
        torch.float8_e4m3fn,
        128,
        True,
        False,
    )

    def prewarm(*args, **kwargs):
        events.append("prewarm")
        return receipt

    def fail_quant(*args, **kwargs):
        events.append("quant")
        raise injected

    monkeypatch.setattr(
        fp8_kernel,
        "prewarm_sglang_per_token_group_quant_fp8_out",
        prewarm,
    )
    monkeypatch.setattr(
        fp8_kernel,
        "launch_sglang_per_token_group_quant_fp8_out_prevalidated",
        fail_quant,
    )
    monkeypatch.setattr(
        mok_fp8_native, "_REPORTED_TERMINAL_QUANT_PREWARM", set()
    )

    class FakeMoKFunctional:
        @staticmethod
        def acquire_megakernel_fp8_block_from_topk_lease(*args, **kwargs):
            events.append("acquire")

        @staticmethod
        def megakernel_fp8_block_from_topk_preloaded_leased(*args, **kwargs):
            events.append("terminal")
            raise AssertionError("terminal must not run after quant failure")

        @staticmethod
        def format_terminal_transaction_failure(workspace, phase, error):
            events.append(f"format:{phase}:{type(error).__name__}")
            return "MOK_TERMINAL_TRANSACTION_FATAL|injected=1"

    def fatal_exit(code):
        events.append(f"exit:{code}")
        raise _InjectedTerminalFatalExit(code)

    monkeypatch.setattr(logging, "shutdown", lambda: events.append("shutdown"))
    monkeypatch.setattr(os, "_exit", fatal_exit)
    workspace = SimpleNamespace(
        x_buffer=torch.empty((2, 4096), dtype=torch.float8_e4m3fn),
        x_scale_buffer=torch.empty((2, 32), dtype=torch.float32),
    )
    layer = SimpleNamespace(
        w13_weight=object(),
        w13_weight_scale_inv=object(),
        w2_weight=object(),
        w2_weight_scale_inv=object(),
        moe_runner_config=SimpleNamespace(swiglu_limit=10.0),
    )
    padded_hidden = torch.empty((2, 4096), dtype=torch.bfloat16)
    padded_ids = torch.zeros((2, 6), dtype=torch.int32)
    padded_weights = torch.zeros((2, 6), dtype=torch.float32)

    with pytest.raises(_InjectedTerminalFatalExit) as fatal:
        mok_fp8_native._run_terminal_eager(
            layer,
            workspace,
            FakeMoKFunctional,
            object(),
            padded_hidden,
            padded_ids,
            padded_weights,
        )
    assert fatal.value.code == 70
    assert events == [
        "prewarm",
        "acquire",
        "quant",
        f"format:quant:{type(injected).__name__}",
        "shutdown",
        "exit:70",
    ]


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
