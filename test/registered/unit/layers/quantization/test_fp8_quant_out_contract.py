import inspect

import pytest
import torch

from sglang.srt.layers.quantization import fp8_kernel


def _make_quant_tensors(rows: int = 4, hidden: int = 256):
    x = torch.empty((rows, hidden), dtype=torch.bfloat16)
    x_q = torch.empty((rows, hidden), dtype=fp8_kernel.fp8_dtype)
    x_s = torch.empty((rows, hidden // 128), dtype=torch.float32)
    return x, x_q, x_s


def _make_receipt(x, x_q, x_s):
    contract = fp8_kernel._FP8OutLaunchContract(
        backend="jit_v2",
        device_type=x.device.type,
        device_index=x.device.index,
        input_dtype=x.dtype,
        output_dtype=x_q.dtype,
        scale_dtype=x_s.dtype,
        group_size=128,
        resolved_v2=True,
        input_shape=tuple(x.shape),
        input_stride=tuple(x.stride()),
        output_shape=tuple(x_q.shape),
        output_stride=tuple(x_q.stride()),
        scale_shape=tuple(x_s.shape),
        scale_stride=tuple(x_s.stride()),
        eps=1e-10,
    )
    return fp8_kernel._FP8OutPrewarmReceipt(contract, False)


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
    events = []
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
        lambda *args: events.append("module"),
        raising=False,
    )
    monkeypatch.setattr(fp8_kernel, "_FP8_OUT_PREWARM_RECEIPTS", {})

    def launch(input_tensor, scratch_q, scratch_s, *args, **kwargs):
        assert not fp8_kernel._FP8_OUT_PREWARM_RECEIPTS
        assert scratch_q.untyped_storage().data_ptr() not in {
            x_q.untyped_storage().data_ptr(),
            x_s.untyped_storage().data_ptr(),
        }
        assert scratch_s.untyped_storage().data_ptr() not in {
            x_q.untyped_storage().data_ptr(),
            x_s.untyped_storage().data_ptr(),
        }
        events.append("launch")

    def synchronize(input_tensor):
        assert not fp8_kernel._FP8_OUT_PREWARM_RECEIPTS
        events.append("sync")

    monkeypatch.setattr(
        fp8_kernel, "_run_sglang_per_token_group_quant_fp8_out", launch
    )
    monkeypatch.setattr(
        fp8_kernel, "_synchronize_fp8_out_prewarm", synchronize
    )

    first = fp8_kernel.prewarm_sglang_per_token_group_quant_fp8_out(
        x, x_q, x_s, 128
    )
    second = fp8_kernel.prewarm_sglang_per_token_group_quant_fp8_out(
        x, x_q, x_s, 128
    )
    assert first == second
    assert first.contract.backend == "jit_v2"
    assert fp8_kernel._FP8_OUT_PREWARM_RECEIPTS[first.contract] == first
    assert events == ["module", "launch", "sync"]


def test_fp8_quant_prevalidated_launch_uses_exact_receipt(monkeypatch):
    x, x_q, x_s = _make_quant_tensors()
    receipt = _make_receipt(x, x_q, x_s)
    launched = []
    monkeypatch.setattr(
        fp8_kernel,
        "_FP8_OUT_PREWARM_RECEIPTS",
        {receipt.contract: receipt},
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
        x, x_q, x_s, receipt
    )
    assert len(launched) == 1
    assert launched[0][0][3:5] == (128, 1e-10)
    assert launched[0][1]["resolved_backend"] == "jit_v2"
    assert launched[0][1]["resolved_use_pdl"] is False

    wrong_shape = torch.empty((5, 256), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="does not match"):
        fp8_kernel.launch_sglang_per_token_group_quant_fp8_out_prevalidated(
            wrong_shape,
            torch.empty((5, 256), dtype=fp8_kernel.fp8_dtype),
            torch.empty((5, 2), dtype=torch.float32),
            receipt,
        )


def test_fp8_quant_cold_prewarm_is_rejected_during_capture(monkeypatch):
    device = type("FakeDevice", (), {"type": "cuda", "index": 0})()
    x = type(
        "FakeTensor",
        (),
        {
            "device": device,
            "dtype": torch.bfloat16,
            "is_cuda": True,
            "shape": (4, 256),
            "stride": lambda self: (256, 1),
        },
    )()
    x_q = type(
        "FakeTensor",
        (),
        {
            "device": device,
            "dtype": torch.float8_e4m3fn,
            "shape": (4, 256),
            "stride": lambda self: (256, 1),
        },
    )()
    x_s = type(
        "FakeTensor",
        (),
        {
            "device": device,
            "dtype": torch.float32,
            "shape": (4, 2),
            "stride": lambda self: (2, 1),
        },
    )()
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
    monkeypatch.setattr(fp8_kernel, "_FP8_OUT_PREWARM_RECEIPTS", {})
    monkeypatch.setattr(
        torch,
        "empty_like",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cold capture must not allocate prewarm scratch")
        ),
    )

    with pytest.raises(RuntimeError, match="prewarmed before capture"):
        fp8_kernel.prewarm_sglang_per_token_group_quant_fp8_out(
            x, x_q, x_s, 128
        )
    assert not fp8_kernel._FP8_OUT_PREWARM_RECEIPTS
