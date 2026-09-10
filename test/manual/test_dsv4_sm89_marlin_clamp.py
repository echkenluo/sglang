# SPDX-License-Identifier: Apache-2.0
"""Fused F28-style clamp/SiLU keeps BF16 activation rounding."""

import unittest

import torch
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
    sm89_swiglu_clamp,
    swiglu_limit_func,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "requires SM89",
)
class TestMarlinClamp(unittest.TestCase):
    def test_finite_bf16_range_and_graph(self):
        gate = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .short()
            .view(torch.bfloat16)
        )
        gate = gate[torch.isfinite(gate)].reshape(-1, 256)
        for value in (-11.0, -1.0, 0.25, 1.0, 11.0):
            x = torch.cat((gate, torch.full_like(gate, value)), 1).contiguous()
            out = torch.empty_like(gate)
            ref = torch.empty_like(gate)
            sm89_swiglu_clamp(out, x, 10.0)
            swiglu_limit_func(ref, x, 10.0)
            torch.testing.assert_close(out, ref, atol=0.001, rtol=0.01)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                sm89_swiglu_clamp(out, x, 10.0)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            sm89_swiglu_clamp(out, x, 10.0)
        for _ in range(3):
            x.normal_(0, 12)
            graph.replay()
            swiglu_limit_func(ref, x, 10.0)
            torch.testing.assert_close(out, ref, atol=0.001, rtol=0.01)


if __name__ == "__main__":
    unittest.main()
