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
