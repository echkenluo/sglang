"""SM90 correctness gate for device-count contiguous SwiGLU quantization."""

import pytest
import torch

from sglang.jit_kernel.dsv4 import (
    silu_and_mul_contig_post_quant,
    silu_and_mul_contig_post_quant_dynamic,
)


@pytest.mark.parametrize("active_rows", [0, 384])
def test_silu_and_mul_contig_post_quant_dynamic(active_rows: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda", 0)
    if torch.cuda.get_device_capability(device) != (9, 0):
        pytest.skip("device-count MoK activation currently requires SM90")

    capacity, hidden = 512, 256
    generator = torch.Generator(device=device).manual_seed(20260822)
    input = torch.randn(
        (capacity, 2 * hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).clamp(-3, 3)
    output = torch.full(
        (capacity, hidden),
        1.0,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    scale = torch.full(
        (capacity, hidden // 128),
        101.0,
        dtype=torch.float32,
        device=device,
    )
    active_tokens = torch.tensor([active_rows], dtype=torch.int32, device=device)

    silu_and_mul_contig_post_quant_dynamic(
        input=input,
        output=output,
        output_scale=scale,
        active_tokens=active_tokens,
        quant_group_size=128,
        swiglu_limit=10,
    )

    if active_rows:
        reference_output = torch.empty(
            (active_rows, hidden), dtype=torch.float8_e4m3fn, device=device
        )
        reference_scale = torch.empty(
            (active_rows, hidden // 128), dtype=torch.float32, device=device
        )
        silu_and_mul_contig_post_quant(
            input=input[:active_rows],
            output=reference_output,
            output_scale=reference_scale,
            quant_group_size=128,
            swiglu_limit=10,
        )
        torch.testing.assert_close(
            output[:active_rows].view(torch.uint8),
            reference_output.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            scale[:active_rows], reference_scale, rtol=0, atol=0
        )

    torch.testing.assert_close(
        output[active_rows:].view(torch.uint8),
        torch.full_like(output[active_rows:], 1.0).view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        scale[active_rows:],
        torch.full_like(scale[active_rows:], 101.0),
        rtol=0,
        atol=0,
    )
