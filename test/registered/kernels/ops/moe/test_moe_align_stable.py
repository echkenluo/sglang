"""Stable Marlin packing: independent CPU oracle, repeat and Graph checks."""

import unittest

import torch

from sglang.kernels.ops.moe.moe_align_stable import moe_align_block_size_stable
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def oracle(ids, experts, block):
    groups = [[] for _ in range(experts)]
    flat = ids.flatten().cpu().tolist()
    for slot, expert in enumerate(flat):
        groups[expert].append(slot)
    packed, blocks = [], []
    for expert, slots in enumerate(groups):
        padding = (-len(slots)) % block
        packed.extend(slots + [len(flat)] * padding)
        blocks.extend([expert] * ((len(slots) + padding) // block))
    return torch.tensor(packed, dtype=torch.int32), torch.tensor(blocks, dtype=torch.int32)


def check(ids, experts, block, result):
    packed, blocks = oracle(ids, experts, block)
    actual, actual_experts, count = result
    assert int(count.item()) == len(packed)
    torch.testing.assert_close(actual[: len(packed)].cpu(), packed, atol=0, rtol=0)
    torch.testing.assert_close(actual_experts[: len(blocks)].cpu(), blocks, atol=0, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestStableMarlinAlignment(unittest.TestCase):
    def test_layout_and_repeat(self):
        torch.manual_seed(20260912)
        cases = [(8, 1, 8), (8, 7, 16), (8, 128, 32), (8, 513, 64),
                 (256, 5, 8), (256, 6, 8), (256, 128, 8),
                 (256, 513, 16), (256, 4096, 64), (256, 513, 48)]
        for experts, tokens, block in cases:
            for dtype in (torch.int32, torch.int64):
                with self.subTest(experts=experts, tokens=tokens, block=block, dtype=str(dtype)):
                    ids = torch.randint(experts, (tokens, 6), device="cuda", dtype=dtype)
                    for _ in range(5):
                        check(ids, experts, block, moe_align_block_size_stable(ids, block, experts))

    def test_empty_and_concentrated(self):
        for experts, tokens, block in ((8, 0, 8), (256, 0, 48),
                                        (256, 513, 16), (256, 4096, 64)):
            with self.subTest(experts=experts, tokens=tokens, block=block):
                ids = torch.full((tokens, 6), experts - 1, device="cuda", dtype=torch.int32)
                check(ids, experts, block, moe_align_block_size_stable(ids, block, experts))

    def test_graph_changed_routes_and_restore(self):
        torch.manual_seed(20260913)
        for experts, tokens, block in ((256, 5, 8), (256, 6, 8), (256, 513, 16),
                                        (256, 4096, 64), (8, 513, 64)):
            with self.subTest(experts=experts, tokens=tokens, block=block):
                original = torch.randint(experts, (tokens, 6), device="cuda", dtype=torch.int32)
                static = original.clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        moe_align_block_size_stable(static, block, experts)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = moe_align_block_size_stable(static, block, experts)
                for _ in range(5):
                    graph.replay()
                    check(static, experts, block, captured)
                for changed in ((original + 3) % experts, torch.full_like(original, experts - 1), original):
                    static.copy_(changed)
                    graph.replay()
                    check(changed, experts, block, captured)


if __name__ == "__main__":
    unittest.main()
