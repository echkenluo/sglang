import torch

from sglang.srt.layers.moe.moe_runner.mok_fp8 import (
    runtime_contract_error,
    shape_contract_error,
)


def make_inputs():
    experts, max_m, hidden, intermediate = 2, 64, 128, 128
    return (
        torch.empty(
            (experts, max_m, hidden), dtype=torch.float8_e4m3fn
        ),
        torch.empty((experts, max_m, hidden // 128), dtype=torch.float32),
        torch.empty(
            (experts, 2 * intermediate, hidden), dtype=torch.float8_e4m3fn
        ),
        torch.empty(
            (experts, 2 * intermediate // 128, hidden // 128),
            dtype=torch.float32,
        ),
        torch.empty(
            (experts, hidden, intermediate), dtype=torch.float8_e4m3fn
        ),
        torch.empty(
            (experts, hidden // 128, intermediate // 128),
            dtype=torch.float32,
        ),
        torch.empty((experts,), dtype=torch.int32),
    )


def test_mok_fp8_shape_contract_accepts_v4_layout():
    assert shape_contract_error(*make_inputs()) is None


def test_mok_fp8_shape_contract_rejects_bad_scale_shape():
    inputs = list(make_inputs())
    inputs[3] = inputs[3][:, :1]
    assert "invalid block-scale/mask shapes" in shape_contract_error(*inputs)


def test_mok_fp8_shape_contract_rejects_noncontiguous_input():
    inputs = list(make_inputs())
    inputs[0] = torch.empty(
        (2, 128, 64), dtype=torch.float8_e4m3fn
    ).transpose(1, 2)
    assert shape_contract_error(*inputs) == "all MoK FP8 inputs must be contiguous"


def test_mok_fp8_runtime_contract_rejects_cpu_tensors():
    assert runtime_contract_error(*make_inputs()) == (
        "all MoK FP8 inputs must be CUDA tensors"
    )
