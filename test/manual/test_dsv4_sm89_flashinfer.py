# SPDX-License-Identifier: Apache-2.0
"""L20 adapter gate against dense attention on known footer-packed KV."""

import unittest

import torch
from sglang.kernels.ops.attention.dsv4_flashinfer_sm89 import Dsv4FlashInferSm89


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "requires SM89 GPU and the pinned F28 FlashInfer wheel",
)
class TestDsv4FlashInferSm89(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = Dsv4FlashInferSm89()

    def packed_cache(self, pages, page_size):
        # Use the actual SGLang page alignment, leaving sentinel padding that
        # catches a mistaken contiguous copy or page-stride reconstruction.
        width = page_size * 584
        stride = ((width + 575) // 576) * 576
        raw = torch.full((pages, stride), 211, device="cuda", dtype=torch.uint8)
        payload = raw[:, : page_size * 576].view(pages, page_size, 576)
        footer = raw[:, page_size * 576 : width].view(pages, page_size, 8)
        nope = torch.randn(pages, page_size, 448, device="cuda").to(torch.float8_e4m3fn)
        exponent = torch.randint(
            124, 128, (pages, page_size, 7), device="cuda", dtype=torch.uint8
        )
        rope = torch.randn(pages, page_size, 64, device="cuda", dtype=torch.bfloat16)
        payload[..., :448] = nope.view(torch.uint8)
        payload[..., 448:] = rope.view(torch.uint8)
        footer[..., :7] = exponent
        footer[..., 7] = 0
        scales = torch.pow(2.0, exponent.float() - 127).repeat_interleave(64, -1)
        known = torch.cat(((nope.float() * scales).bfloat16(), rope), -1)
        return raw[:, :width].view(pages, page_size, 1, 584), known.flatten(0, 1)

    def case(self, tokens, page_size, extra_page_size, main_topk=128):
        torch.manual_seed(20260909 + tokens + page_size)
        cache, kv = self.packed_cache(8, page_size)
        q = torch.randn(tokens, 1, 8, 512, device="cuda", dtype=torch.bfloat16)
        # Noncontiguous index row strides are common after prefix slicing.
        ids = torch.randint(
            kv.shape[0], (tokens, main_topk + 128), device="cuda", dtype=torch.int32
        )[:, :main_topk]
        ids[:, 3] = -1
        lengths = torch.arange(tokens, device="cuda", dtype=torch.int32) % main_topk + 1
        sink = torch.linspace(-2, 2, 8, device="cuda")
        kwargs = {
            "q": q,
            "k_cache": cache,
            "head_dim_v": 512,
            "softmax_scale": 512**-0.5,
            "indices": ids[:, None],
            "topk_length": lengths,
            "attn_sink": sink,
        }
        scopes = [(kv, ids, lengths)]
        if extra_page_size:
            extra, ev = self.packed_cache(8, extra_page_size)
            ei = torch.randint(
                ev.shape[0], (tokens, 512), device="cuda", dtype=torch.int32
            )
            ei[:, 5] = -1
            el = torch.arange(tokens, device="cuda", dtype=torch.int32) % 513
            kwargs.update(
                extra_k_cache=extra,
                extra_indices_in_kvcache=ei[:, None],
                extra_topk_length=el,
            )
            scopes.append((ev, ei, el))
        return kwargs, scopes

    def check_reference(self, kwargs, scopes, out, lse):
        q = kwargs["q"].squeeze(1).float()
        sink = kwargs["attn_sink"]
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            # Bound reference memory independently of prefill length.
            for start in range(0, q.shape[0], 16):
                end = start + 16
                values, masks = [], []
                for kv, ids, lengths in scopes:
                    selected = ids[start:end]
                    values.append(kv[selected.clamp_min(0).long()].float())
                    masks.append(
                        (selected >= 0)
                        & (
                            torch.arange(ids.shape[-1], device="cuda")[None]
                            < lengths[start:end, None]
                        )
                    )
                value = torch.cat(values, 1)
                valid = torch.cat(masks, 1)
                logits = (
                    torch.einsum("thd,tkd->thk", q[start:end], value)
                    * kwargs["softmax_scale"]
                )
                logits.masked_fill_(~valid[:, None], float("-inf"))
                logits_with_sink = torch.cat(
                    (logits, sink[None, :, None].expand(logits.shape[0], -1, -1)), -1
                )
                prob = torch.softmax(logits_with_sink, -1)[..., :-1]
                ref = torch.einsum("thk,tkd->thd", prob, value)
                # Match the fork's published operator-test tolerance; not a
                # model-quality claim or a bit-exactness requirement.
                torch.testing.assert_close(
                    out[start:end, 0].float(), ref, atol=0.05, rtol=0.05
                )
                self.assertTrue(torch.isfinite(lse[start:end]).all().item())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    def test_layouts_lengths_and_dispatch_boundaries(self):
        for tokens, page, extra, main_topk in (
            (1, 64, 0, 128),
            (7, 256, 2, 128),
            (16, 64, 64, 128),
            (64, 256, 64, 128),
            (65, 64, 2, 128),
            (513, 256, 64, 128),
            # Optional page-128 geometry and full 4K chunks. Actual CUDA
            # storage pages are 256; the logical SWA window is 128.
            (1, 128, 0, 128),
            (7, 128, 2, 128),
            (16, 128, 64, 128),
            (64, 128, 64, 128),
            (65, 128, 2, 128),
            (513, 128, 64, 128),
            (4096, 128, 0, 128),
            (4096, 128, 2, 128),
            (4096, 128, 64, 128),
            # DSpark draft uses padded 192-wide indices (128 context + 5
            # active draft entries); 16 requests produce 80 query rows.
            (5, 128, 0, 192),
            (80, 128, 0, 192),
        ):
            with self.subTest(
                tokens=tokens, page=page, extra=extra, main_topk=main_topk
            ):
                kwargs, scopes = self.case(tokens, page, extra, main_topk)
                for dtype in (torch.uint8, torch.float8_e4m3fn):
                    with self.subTest(storage_view=dtype):
                        for key in ("k_cache", "extra_k_cache"):
                            if key not in kwargs:
                                continue
                            original = kwargs[key]
                            kwargs[key] = original.view(dtype)
                            packed = self.adapter._packed_cache_bytes(kwargs[key])
                            self.assertEqual(packed.data_ptr(), original.data_ptr())
                            self.assertEqual(packed.stride(), original.stride())
                            self.assertTrue(
                                torch.equal(packed, original.view(torch.uint8))
                            )
                        out, lse = self.adapter(**kwargs)
                        self.check_reference(kwargs, scopes, out, lse)

    def test_native_heads_match_padded_valid_heads(self):
        from sglang.srt.models.deepseek_v4 import MqaAttentionBase

        model = MqaAttentionBase.__new__(MqaAttentionBase)
        torch.nn.Module.__init__(model)
        model.attn_tp_size, model.attn_tp_rank = 8, 3
        model.n_heads, model.n_local_heads = 64, 8
        model.attn_sink = torch.arange(64, device="cuda", dtype=torch.float32)
        for enabled, expected in ((False, 64), (True, 8)):
            model._sm89_flashinfer_native_heads = enabled
            model._attn_sink_local = None
            self.assertEqual(model._attention_compute_heads(), expected)
            sink = model._local_attn_sink()
            self.assertEqual(sink.numel(), expected)
            torch.testing.assert_close(sink[:8], model.attn_sink[24:32])
        for tokens, extra in ((1, 0), (16, 2), (65, 64), (4096, 64)):
            with self.subTest(tokens=tokens, extra=extra):
                kwargs, scopes = self.case(tokens, 256, extra)
                native, lse = self.adapter(**kwargs)
                self.check_reference(kwargs, scopes, native, lse)
                padded = dict(kwargs)
                q = torch.full(
                    (tokens, 1, 64, 512),
                    float("nan"),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                q[:, :, :8] = kwargs["q"]
                sink = torch.zeros(64, device="cuda")
                sink[:8] = kwargs["attn_sink"]
                padded.update(q=q, attn_sink=sink)
                old, _ = self.adapter(**padded)
                torch.testing.assert_close(native, old[:, :, :8], atol=0.05, rtol=0.05)

    def test_graph_replay(self):
        kwargs, scopes = self.case(16, 256, 2)
        # Match the real KVPool API, including its reinterpretation of footer
        # bytes as FP8 values. No arithmetic is valid on this storage view.
        for key in ("k_cache", "extra_k_cache"):
            kwargs[key] = kwargs[key].view(torch.float8_e4m3fn)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.adapter(**kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, lse = self.adapter(**kwargs)
        for multiplier in (0.5, 2.0, 0.25):
            kwargs["q"].mul_(multiplier)
            graph.replay()
            self.check_reference(kwargs, scopes, out, lse)


if __name__ == "__main__":
    unittest.main()
