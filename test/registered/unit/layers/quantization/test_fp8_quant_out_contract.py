import inspect

import pytest
import torch

from sglang.srt.layers.quantization import fp8_kernel


def _make_quant_tensors(rows: int = 4, hidden: int = 256):
    x = torch.empty((rows, hidden), dtype=torch.bfloat16)
    x_q = torch.empty((rows, hidden), dtype=fp8_kernel.fp8_dtype)
    x_s = torch.empty((rows, hidden // 128), dtype=torch.float32)
    return x, x_q, x_s


def test_fp8_quant_out_contract_accepts_terminal_layout():
    fp8_kernel.validate_sglang_per_token_group_quant_fp8_out(
        *_make_quant_tensors(), 128
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda x, x_q, x_s: (x[:, ::2], x_q, x_s), "x must be contiguous"),
        (
            lambda x, x_q, x_s: (x, x_q[:, :-1], x_s),
            "x_q must match x shape",
        ),
        (
            lambda x, x_q, x_s: (x, x_q, x_s[:, :1]),
            "x_s must be float32 with shape",
        ),
        (
            lambda x, x_q, x_s: (x, x_q, x_q.view(torch.float32)),
            "x_s must be float32 with shape",
        ),
    ],
)
def test_fp8_quant_out_contract_rejects_invalid_layout(mutate, match):
    with pytest.raises(ValueError, match=match):
        fp8_kernel.validate_sglang_per_token_group_quant_fp8_out(
            *mutate(*_make_quant_tensors()), 128
        )


def test_fp8_quant_out_reuses_allocating_path_dispatch():
    allocating = inspect.getsource(fp8_kernel.sglang_per_token_group_quant_fp8)
    caller_owned = inspect.getsource(
        fp8_kernel.sglang_per_token_group_quant_fp8_out
    )
    helper = "_run_sglang_per_token_group_quant_fp8_out("
    assert helper in allocating
    assert helper in caller_owned
    assert "torch.empty(" not in caller_owned


def test_fp8_quant_out_prewarm_receipt_materializes_once(monkeypatch):
    x, x_q, x_s = _make_quant_tensors()
    materialized = []
    monkeypatch.setattr(
        fp8_kernel,
        "enable_sgl_per_token_group_quant_8bit",
        True,
        raising=False,
    )
    monkeypatch.setattr(fp8_kernel, "_is_musa", False)
    monkeypatch.setattr(fp8_kernel, "is_arch_support_pdl", lambda: False)
    monkeypatch.setattr(
        fp8_kernel,
        "_jit_per_token_group_quant_8bit_v2_module",
        lambda *args: materialized.append(args),
        raising=False,
    )
    monkeypatch.setattr(fp8_kernel, "_FP8_OUT_PREWARM_RECEIPTS", set())

    first = fp8_kernel.prewarm_sglang_per_token_group_quant_fp8_out(
        x, x_q, x_s, 128
    )
    second = fp8_kernel.prewarm_sglang_per_token_group_quant_fp8_out(
        x, x_q, x_s, 128
    )
    assert first == second
    assert first[0] == "jit_v2"
    assert first in fp8_kernel._FP8_OUT_PREWARM_RECEIPTS
    assert len(materialized) == 1


def test_fp8_quant_prevalidated_launch_does_not_revalidate(monkeypatch):
    x, x_q, x_s = _make_quant_tensors()
    receipt = ("jit_v2", "cpu", None, x.dtype, x_q.dtype, 128, True, False)
    launched = []
    monkeypatch.setattr(
        fp8_kernel, "_FP8_OUT_PREWARM_RECEIPTS", {receipt}
    )
    monkeypatch.setattr(
        fp8_kernel,
        "validate_sglang_per_token_group_quant_fp8_out",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("post-acquire validation is forbidden")
        ),
    )
    monkeypatch.setattr(
        fp8_kernel,
        "_run_sglang_per_token_group_quant_fp8_out",
        lambda *args, **kwargs: launched.append((args, kwargs)),
    )
    fp8_kernel.launch_sglang_per_token_group_quant_fp8_out_prevalidated(
        x, x_q, x_s, 128, receipt
    )
    assert len(launched) == 1


def test_fp8_quant_cold_prewarm_is_rejected_during_capture(monkeypatch):
    device = type("FakeDevice", (), {"type": "cuda", "index": 0})()
    x = type(
        "FakeTensor",
        (),
        {"device": device, "dtype": torch.bfloat16, "is_cuda": True},
    )()
    x_q = type(
        "FakeTensor",
        (),
        {"device": device, "dtype": torch.float8_e4m3fn},
    )()
    x_s = object()
    monkeypatch.setattr(
        fp8_kernel,
        "validate_sglang_per_token_group_quant_fp8_out",
        lambda *args: None,
    )
    monkeypatch.setattr(
        fp8_kernel,
        "enable_sgl_per_token_group_quant_8bit",
        True,
        raising=False,
    )
    monkeypatch.setattr(fp8_kernel, "_is_musa", False)
    monkeypatch.setattr(fp8_kernel, "is_arch_support_pdl", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: True
    )
    monkeypatch.setattr(fp8_kernel, "_FP8_OUT_PREWARM_RECEIPTS", set())

    with pytest.raises(RuntimeError, match="prewarmed before capture"):
        fp8_kernel.prewarm_sglang_per_token_group_quant_fp8_out(
            x, x_q, x_s, 128
        )
    assert not fp8_kernel._FP8_OUT_PREWARM_RECEIPTS
