# SPDX-License-Identifier: Apache-2.0
"""Community SM89 decode candidate: numeric and Graph gates, no timing claim."""

import unittest

import test_dsv4_sm89_flashinfer as fixture
import torch

from sglang.kernels.ops.attention.flash_mla_sm120_triton import (
    flash_mla_sparse_decode_triton,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "requires an isolated SM89 GPU and the pinned SGLang runtime",
)
class TestCommunityDecode(unittest.TestCase):
    # Reuse the independently constructed footer layout and dense FP32 oracle.
    packed_cache = fixture.TestDsv4FlashInferSm89.packed_cache
    case = fixture.TestDsv4FlashInferSm89.case
    check_reference = fixture.TestDsv4FlashInferSm89.check_reference

    @staticmethod
    def invoke(kwargs):
        args = dict(kwargs)
        if "extra_indices_in_kvcache" in args:
            args["extra_indices"] = args.pop("extra_indices_in_kvcache")
        return flash_mla_sparse_decode_triton(**args)

    def test_h8_footer_layouts_and_speculative_widths(self):
        for tokens in (1, 5, 6, 10, 12, 40, 48, 80, 96):
            for extra in (0, 2, 64):
                topk = 192 if tokens in (5, 10, 40, 80) else 128
                with self.subTest(tokens=tokens, extra_page=extra, topk=topk):
                    kwargs, scopes = self.case(tokens, 256, extra, topk)
                    for dtype in (torch.uint8, torch.float8_e4m3fn):
                        args = dict(kwargs)
                        args["k_cache"] = args["k_cache"].view(dtype)
                        if "extra_k_cache" in args:
                            args["extra_k_cache"] = args["extra_k_cache"].view(dtype)
                        out, lse = self.invoke(args)
                        self.assertEqual(tuple(out.shape), (tokens, 1, 8, 512))
                        self.assertEqual(tuple(lse.shape), (tokens, 8, 1))
                        self.check_reference(kwargs, scopes, out, lse)

    def test_all_invalid_cache_with_finite_sink(self):
        kwargs, _ = self.case(6, 256, 2)
        kwargs["indices"].fill_(-1)
        kwargs["extra_indices_in_kvcache"].fill_(-1)
        kwargs["topk_length"].zero_()
        kwargs["extra_topk_length"].zero_()
        kwargs["k_cache"].fill_(255)
        kwargs["extra_k_cache"].fill_(255)
        out, lse = self.invoke(kwargs)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        expected_lse = kwargs["attn_sink"][None, :, None].expand(6, 8, 1)
        torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)

    def test_graph_replay_and_changed_query(self):
        for tokens, extra in ((5, 2), (6, 64)):
            with self.subTest(tokens=tokens, extra_page=extra):
                kwargs, scopes = self.case(tokens, 256, extra, 192)
                query_before = kwargs["q"].clone()
                cache_before = kwargs["k_cache"].clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.invoke(kwargs)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output, lse = self.invoke(kwargs)
                graph.replay()
                first, first_lse = output.clone(), lse.clone()
                for _ in range(3):
                    graph.replay()
                    self.assertTrue(torch.equal(output, first))
                    self.assertTrue(torch.equal(lse, first_lse))
                self.check_reference(kwargs, scopes, output, lse)
                self.assertTrue(torch.equal(kwargs["q"], query_before))
                self.assertTrue(torch.equal(kwargs["k_cache"], cache_before))
                kwargs["q"].neg_()
                graph.replay()
                self.check_reference(kwargs, scopes, output, lse)
                kwargs["q"].copy_(query_before)
                graph.replay()
                self.assertTrue(torch.equal(output, first))
                self.assertTrue(torch.equal(lse, first_lse))


if __name__ == "__main__":
    unittest.main()
