# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual nonpaged SGLang adapter against independent FP32 logits."""

from types import SimpleNamespace
import unittest

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "requires SM89",
)
class TestF28Nonpaged(unittest.TestCase):
    def test_actual_nonpaged_adapter(self):
        torch.manual_seed(20260910)
        torch.backends.cuda.matmul.allow_tf32 = False
        with envs.SGLANG_DSV4_SM89_F28_INDEXER.override(True):
            for rows, count in ((1, 317), (6, 1024), (513, 317), (1024, 4096)):
                q = torch.randn(rows, 64, 128, device="cuda").to(torch.float8_e4m3fn)
                k = torch.randn(count, 128, device="cuda").to(torch.float8_e4m3fn)
                scales = torch.rand(count, device="cuda") * 0.02 + 0.01
                weights = torch.randn(rows, 64, device="cuda") * 0.03
                starts = torch.randint(
                    0, count // 2, (rows,), device="cuda", dtype=torch.int32
                )
                ends = torch.randint(
                    count // 2, count + 1, (rows,), device="cuda", dtype=torch.int32
                )
                pool = SimpleNamespace(
                    get_index_k_scale_buffer=lambda **kwargs: (
                        k.view(torch.uint8),
                        scales.view(torch.uint8).view(count, 4),
                    )
                )
                plan = SimpleNamespace(
                    gather_seq_lens=ends[-1:],
                    page_table=None,
                    seq_len_sum=count,
                    max_seq_len=count,
                    max_seqlen_k=count,
                    query_rows=rows,
                    ks=starts,
                    ke=ends,
                )

                def call():
                    return C4IndexerBackendMixin._forward_nonpaged_indexer(
                        q_indexer=q,
                        weights=weights,
                        c4_indexer=SimpleNamespace(layer_id=2),
                        token_to_kv_pool=pool,
                        plan=plan,
                    )

                def check(out):
                    # Chunk the oracle to avoid allocating rows*heads*keys at once.
                    for lo in range(0, rows, 16):
                        hi = min(rows, lo + 16)
                        dots = torch.einsum("mhd,nd->mhn", q[lo:hi].float(), k.float())
                        ref = (dots.relu() * weights[lo:hi, :, None]).sum(1) * scales
                        pos = torch.arange(count, device="cuda")[None]
                        mask = (pos >= starts[lo:hi, None]) & (pos < ends[lo:hi, None])
                        self.assertTrue(torch.isneginf(out[lo:hi][~mask]).all().item())
                        torch.testing.assert_close(
                            out[lo:hi][mask], ref[mask], atol=0.002, rtol=0.002
                        )

                check(call())
                # The model's nonpaged gather admission remains eager-only.
                # The packed-cache-free kernel itself also supports capture.
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        call()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = call()
                for _ in range(3):
                    q.copy_(torch.randn(q.shape, device="cuda").to(q.dtype))
                    graph.replay()
                    check(out)
                print(
                    f"NONPAGED rows={rows} keys={count} graph_updates=3 passed",
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()
