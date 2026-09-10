# SPDX-License-Identifier: Apache-2.0
"""Validate existing FP8 Marlin linear for L20 block-128 serving shapes."""

import unittest

import torch
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "requires SM89",
)
class TestFp8Marlin(unittest.TestCase):
    def test_block_scales_and_graph(self):
        torch.manual_seed(20260910)
        torch.backends.cuda.matmul.allow_tf32 = False
        for n, k in (
            (512, 4096),
            (1024, 4096),
            (1536, 4096),
            (4096, 1024),
            (4096, 12288),
            (4096, 256),
            (8192, 1024),
        ):
            layer = torch.nn.Module()
            layer.output_size_per_partition, layer.input_size_per_partition = n, k
            layer.orig_dtype = torch.bfloat16
            layer.weight_block_size = [128, 128]
            weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
            scales = (
                (torch.rand(n // 128, k // 128, device="cuda") * 0.02 + 0.01)
                .bfloat16()
                .float()
            )
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
            dense = weight.float() * scales.repeat_interleave(128, 0).repeat_interleave(
                128, 1
            )
            # marlin_template.h scale() uses BF16 __hmul2 before the MMA.
            # Compare the implementation with that rounded W8A16 contract;
            # the unrounded FP8*FP32-scale oracle is a separate precision delta.
            unrounded_dense = dense
            dense = dense.bfloat16().float()
            prepare_fp8_layer_for_marlin(layer, size_k_first=False)
            for rows in (1, 6, 16, 128, 513):
                x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)

                def call():
                    return apply_fp8_marlin_linear(
                        x, layer.weight, layer.weight_scale, layer.workspace, n, k, None
                    )

                def check(out):
                    ref = (x.float() @ dense.T).bfloat16().float()
                    self.assertTrue(torch.isfinite(out).all().item())
                    self.assertLess(
                        (out.float() - ref).norm().item() / ref.norm().item(), 0.002
                    )
                    torch.testing.assert_close(out.float(), ref, atol=0.02, rtol=0.02)

                eager = call()
                check(eager)
                unrounded_ref = x.float() @ unrounded_dense.T
                print(
                    f"FP8_PRECISION n={n} k={k} rows={rows} unrounded_weight_l2="
                    f"{((eager.float()-unrounded_ref).norm()/unrounded_ref.norm()).item():.8g}",
                    flush=True,
                )
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        call()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = call()
                for update in range(3):
                    x.normal_()
                    graph.replay()
                    check(out)
                print(
                    f"FP8_MARLIN n={n} k={k} rows={rows} graph_updates=3 passed",
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()
