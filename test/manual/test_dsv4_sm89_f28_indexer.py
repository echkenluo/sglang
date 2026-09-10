# SPDX-License-Identifier: Apache-2.0
"""SM89 paged indexer: independent oracle, real page layout and Graph replay."""
import unittest

import torch
from sglang.srt.layers.attention.dsv4.f28_mqa import sglang_paged_mqa_logits


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9), "requires SM89")
class TestF28Indexer(unittest.TestCase):
    def test_paged_logits_and_graph_updates(self):
        torch.manual_seed(20260910)
        torch.backends.cuda.matmul.allow_tf32 = False
        h, d, page, pages = 64, 128, 64, 19
        keys = (torch.randn(pages, page, d, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        scales = torch.rand(pages, page, device="cuda") * 0.02 + 0.01
        packed = torch.cat((keys.view(torch.uint8).reshape(pages, -1), scales.view(torch.uint8).reshape(pages, -1)), 1)
        # Storage offset is deliberately nonzero; views must preserve it.
        storage = torch.zeros(packed.numel() + page * 132, device="cuda", dtype=torch.uint8)
        cache = storage[page * 132:].view(pages, page, 1, 132)
        cache.copy_(packed.view_as(cache))
        for rows in (1, 6, 16, 80, 513):
            q = torch.randn(rows, 1, h, d, device="cuda").to(torch.float8_e4m3fn)
            weights = torch.randn(rows, h, device="cuda") * 0.03
            tables = torch.randint(pages, (rows, 5), device="cuda", dtype=torch.int32)
            lengths = torch.randint(0, 318, (rows, 1), device="cuda", dtype=torch.int32)
            def call():
                return sglang_paged_mqa_logits(q, cache, weights, lengths, tables, None, 317, False)
            def check(out):
                # FP32 oracle avoids using either engine's implementation.
                k = keys[tables.long()].float().reshape(rows, 320, d)[:, :317]
                scale = scales[tables.long()].reshape(rows, 320)[:, :317]
                dots = torch.bmm(q[:, 0].float(), k.transpose(1, 2))
                ref = (dots.relu() * weights[:, :, None]).sum(1) * scale
                valid = torch.arange(317, device="cuda")[None] < lengths
                self.assertTrue(torch.isneginf(out[~valid]).all().item())
                torch.testing.assert_close(out[valid], ref[valid], atol=0.002, rtol=0.002)
            check(call())
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for _ in range(3):
                    call()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = call()
            for update in range(3):
                q.copy_(torch.randn(q.shape, device="cuda").to(q.dtype))
                tables.copy_(torch.randint(pages, tables.shape, device="cuda", dtype=torch.int32))
                lengths.copy_(torch.randint(0, 318, lengths.shape, device="cuda", dtype=torch.int32))
                graph.replay()
                check(captured)
            print(f"INDEXER rows={rows} graph_updates=3 passed", flush=True)


if __name__ == "__main__":
    unittest.main()
