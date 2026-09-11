"""L20-only checks for community prefill and SGLang's compressed/SWA indices."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.sparse_prefill_bf16_sm89 import (
    sparse_prefill_bf16_sm89,
)
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    combine_topk_swa_indices,
)


def reference(q, kv, indices, lengths, sink):
    # Independent FP32 softmax, including the zero-valued sink key.
    kv = kv.reshape(-1, 512).float()
    indices = indices.reshape(q.shape[0], -1)
    rows = []
    for i in range(q.shape[0]):
        ids = indices[i, : max(0, min(int(lengths[i]), indices.shape[1]))]
        ids = ids[(ids >= 0) & (ids < kv.shape[0])].long()
        values = kv[ids]
        scores = q[i].float() @ values.T / 512**0.5
        scores = torch.cat((scores, sink[:, None]), dim=1)
        probs = scores.softmax(dim=1)[:, :-1]
        rows.append((probs @ values).to(torch.bfloat16))
    return torch.stack(rows)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "Requires an SM89 GPU",
)
class TestCommunityPrefill(unittest.TestCase):
    def check_attention(self, q, kv, ids, lengths, sink):
        actual = sparse_prefill_bf16_sm89(q, kv, ids, lengths, sink, 512**-0.5)
        expected = reference(q, kv, ids, lengths, sink)
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)

    def test_h8_and_h64_boundaries(self):
        torch.manual_seed(7301)
        for heads in (8, 64):
            for topk in (128, 256, 640, 1152):
                with self.subTest(heads=heads, topk=topk):
                    q = torch.randn(7, heads, 512, device="cuda", dtype=torch.bfloat16)
                    kv = torch.randn(1536, 1, 512, device="cuda", dtype=torch.bfloat16)
                    ids = torch.randint(0, 1536, (7, topk), device="cuda", dtype=torch.int32)
                    ids[2, :] = -1
                    ids[3, 1] = 1540
                    lengths = torch.tensor(
                        [-1, 0, 65, 127, 128, topk - 1, topk + 1],
                        device="cuda", dtype=torch.int32,
                    )
                    sink = torch.linspace(-5, 5, heads, device="cuda")
                    self.check_attention(q, kv, ids, lengths, sink)

    def test_c4_c128_and_swa_chunk_indices(self):
        torch.manual_seed(7302)
        # Two independent requests, chunks crossing positions 128 and 256.
        seqs, queries = [131, 259], [5, 7]
        window = 128
        gather = [min(s, n + window - 1) for s, n in zip(seqs, queries)]
        device = "cuda"

        def ints(values):
            return torch.tensor(values, dtype=torch.int32, device=device)

        for ratio in (0, 4, 128):
            with self.subTest(ratio=ratio):
                compression = max(1, ratio)
                topk = 0 if ratio == 0 else (512 if ratio == 4 else 128)
                compressed = [s // compression if topk else 0 for s in seqs]
                comp_base = [0, compressed[0]]
                swa_base = [sum(compressed), sum(compressed) + gather[0]]
                raw = torch.arange(max(1, topk), device=device, dtype=torch.int32)
                raw = raw.repeat(sum(queries), 1)
                ids, lengths = combine_topk_swa_indices(
                    raw, ints([100, 105, 112]), ints(seqs), ints(gather),
                    ints(comp_base), ints(swa_base), window, compression, topk,
                )
                expected = torch.full_like(ids, -1)
                expected_lengths = []
                row = 0
                for request, (seq, count) in enumerate(zip(seqs, queries)):
                    for pos in range(seq - count, seq):
                        n = min((pos + 1) // compression, topk)
                        sw = min(pos + 1, window)
                        entries = list(range(comp_base[request], comp_base[request] + n))
                        start = swa_base[request] + pos - sw + 1 - (seq - gather[request])
                        entries += list(range(start, start + sw))
                        expected[row, :len(entries)] = ints(entries)
                        expected_lengths.append(len(entries))
                        row += 1
                torch.testing.assert_close(ids, expected, atol=0, rtol=0)
                torch.testing.assert_close(lengths, ints(expected_lengths), atol=0, rtol=0)
                q = torch.randn(sum(queries), 8, 512, device=device, dtype=torch.bfloat16)
                kv = torch.randn(sum(compressed) + sum(gather), 1, 512,
                                 device=device, dtype=torch.bfloat16)
                sink = torch.randn(8, device=device)
                self.check_attention(q, kv, ids, lengths, sink)


if __name__ == "__main__":
    unittest.main()
