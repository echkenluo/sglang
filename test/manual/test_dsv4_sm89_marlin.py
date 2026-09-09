# SPDX-License-Identifier: Apache-2.0
"""Check MXFP4 Marlin repack and routed MoE against dequantized weights."""

import unittest

import torch
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "requires SM89 GPU",
)
class TestSm89Mxfp4Marlin(unittest.TestCase):
    def weights(self, experts, rows, cols):
        codes = torch.randint(
            16, (experts, rows, cols), device="cuda", dtype=torch.uint8
        )
        packed = (codes[..., ::2] | (codes[..., 1::2] << 4)).view(torch.int8)
        exponents = torch.randint(
            119, 124, (experts, rows, cols // 32), device="cuda", dtype=torch.uint8
        )
        values = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )
        dense = values[codes.long()] * torch.pow(
            2.0, exponents.float() - 127
        ).repeat_interleave(32, -1)
        return packed, exponents.view(torch.float8_e8m0fnu), dense.bfloat16()

    def test_actual_tp8_shapes_and_routing(self):
        torch.manual_seed(20260909)
        experts, hidden, intermediate, topk = 8, 4096, 256, 6
        layer = torch.nn.Module()
        w13, s13, ref13 = self.weights(experts, 2 * intermediate, hidden)
        w2, s2, ref2 = self.weights(experts, hidden, intermediate)
        for name, value in (
            ("w13_weight", w13),
            ("w2_weight", w2),
            ("w13_weight_scale_inv", s13),
            ("w2_weight_scale_inv", s2),
        ):
            layer.register_parameter(
                name, torch.nn.Parameter(value, requires_grad=False)
            )
        layer.orig_dtype = torch.bfloat16
        prepare_moe_mxfp4_layer_for_marlin(layer)
        for tokens in (1, 7, 16, 128, 513):
            with self.subTest(tokens=tokens):
                x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 4
                logits = torch.randn(tokens, experts, device="cuda")
                weights, ids = torch.topk(torch.softmax(logits, -1), topk, dim=-1)
                weights = (weights / weights.sum(-1, keepdim=True) * 1.5).contiguous()
                ids = ids.int().contiguous()
                out = fused_marlin_moe(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                    logits,
                    weights,
                    ids,
                    workspace=layer.workspace,
                    num_bits=4,
                    clamp_limit=10.0,
                )
                ref = torch.zeros_like(x, dtype=torch.float32)
                for expert in range(experts):
                    token_ids, slots = torch.where(ids == expert)
                    if token_ids.numel() == 0:
                        continue
                    gate_up = (
                        (x[token_ids].float() @ ref13[expert].float().T)
                        .bfloat16()
                    )
                    gate, up = gate_up.chunk(2, -1)
                    # The production clamp path uses separate BF16 PyTorch
                    # operations: SiLU rounds before the multiplication.
                    activation = (
                        torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
                    )
                    # marlin_template.h stores the GEMM accumulator in BF16,
                    # converts each routing weight to BF16, then uses __hmul2.
                    # Preserve both rounding boundaries in the dense oracle.
                    value = (activation.float() @ ref2[expert].float().T).bfloat16()
                    route = weights[token_ids, slots, None].bfloat16()
                    value = (value * route).float()
                    ref.index_add_(0, token_ids, value)
                # The expert sum is stored into a BF16 output buffer.
                ref = ref.bfloat16().float()
                self.assertTrue(torch.isfinite(out).all().item())
                relative_l2 = ((out.float() - ref).norm() / ref.norm()).item()
                print(
                    f"tokens={tokens} relative_l2={relative_l2:.8f} "
                    f"max_abs={(out.float() - ref).abs().max().item():.8f}",
                    flush=True,
                )
                self.assertLessEqual(relative_l2, 0.02)
                torch.testing.assert_close(out.float(), ref, atol=0.1, rtol=0.05)


if __name__ == "__main__":
    unittest.main()
