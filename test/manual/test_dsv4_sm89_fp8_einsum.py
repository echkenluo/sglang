# SPDX-License-Identifier: Apache-2.0
"""Manual L20 operator gate: actual TP8 wo_a shape and CUDA Graph replay."""

import unittest

import torch

from sglang.kernels.ops.dsv4_sm89_fp8_einsum import sm89_fp8_einsum


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "requires SM89 GPU",
)
class TestSm89Fp8Einsum(unittest.TestCase):
    def make_inputs(self, tokens, groups=1, rank=1024, hidden=4096):
        torch.manual_seed(20260909 + tokens)
        a = torch.randn(tokens, groups, hidden, device="cuda").clamp(-4, 4)
        b = torch.randn(groups, rank, hidden, device="cuda").clamp(-4, 4)
        # Nonuniform power-of-two scales exercise both scale axes; these are
        # exact BF16 multipliers, so a dense FP32 reference is independent of
        # the kernel's per-K-block dot accumulation order.
        sa = torch.pow(2.0, torch.randint(-4, 1, (tokens, groups, hidden // 128), device="cuda")).float()
        sb = torch.pow(2.0, torch.randint(-4, 1, (groups, rank // 128, hidden // 128), device="cuda")).float()
        return a.to(torch.float8_e4m3fn), sa, b.to(torch.float8_e4m3fn), sb

    def check_result(self, a, sa, b, sb, out):
        aa = a.float() * sa.repeat_interleave(128, -1)
        bb = b.float() * sb.repeat_interleave(128, -2).repeat_interleave(128, -1)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            ref = torch.einsum("tgd,grd->tgr", aa, bb)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        self.assertTrue(torch.isfinite(out).all().item())
        if not out.numel():
            return
        rel_l2 = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
        self.assertLessEqual(rel_l2, 0.004)
        self.assertLessEqual((out.float() - ref).abs().max().item(), 0.02 * ref.abs().max().item() + 0.001)

    def test_actual_tp8_and_padding_boundaries(self):
        for tokens in (0, 1, 7, 8, 9, 16, 17, 511, 512, 513, 4096):
            with self.subTest(tokens=tokens):
                a, sa, b, sb = self.make_inputs(tokens)
                out = torch.empty(tokens, 1, 1024, device="cuda", dtype=torch.bfloat16)
                sm89_fp8_einsum(a, sa, b, sb, out)
                self.check_result(a, sa, b, sb, out)

    def test_strides_and_groups(self):
        a, sa, b, sb = self.make_inputs(17, groups=2, rank=256, hidden=128)
        # Exercise the stride contract without assuming scale groups are
        # aligned to 16 elements (untrue for the supported small hidden size).
        a = a.transpose(0, 1).contiguous().transpose(0, 1)
        sa = sa.transpose(0, 1).contiguous().transpose(0, 1)
        out = torch.empty(17, 2, 256, device="cuda", dtype=torch.bfloat16)
        sm89_fp8_einsum(a, sa, b, sb, out)
        self.check_result(a, sa, b, sb, out)

    def test_graph_replay_reads_fresh_inputs(self):
        a, sa, b, sb = self.make_inputs(16)
        out = torch.empty(16, 1, 1024, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                sm89_fp8_einsum(a, sa, b, sb, out)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sm89_fp8_einsum(a, sa, b, sb, out)
        for multiplier in (0.5, 2.0, 0.25):
            sa.mul_(multiplier)
            graph.replay()
            self.check_result(a, sa, b, sb, out)


if __name__ == "__main__":
    unittest.main()
