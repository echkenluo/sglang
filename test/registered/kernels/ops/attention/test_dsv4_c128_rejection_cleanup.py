"""Check rejected DSpark slots across C128 boundaries and CUDA graph replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="base-b-test-1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestC128RejectionCleanup(unittest.TestCase):
    def test_rejected_rows_and_replay_updated_acceptance(self):
        ring, width, drafts = 128, 1032, 6
        state = torch.full((5 * ring, width), 7.0, device="cuda")
        unrelated = torch.full_like(state, 11.0)
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.compress_state_pools = [
            SimpleNamespace(ratio=128, ring_size=ring,
                            kv_score_buffer=SimpleNamespace(kv_score=state)),
            SimpleNamespace(ratio=4, ring_size=ring,
                            kv_score_buffer=SimpleNamespace(kv_score=unrelated)),
            None,
        ]
        slots = torch.tensor([1, 3, 2], device="cuda")
        prefixes = torch.tensor([126, 255, 127], device="cuda")
        accepted = torch.tensor([1, 6, 3], device="cuda")

        def call():
            pool.clear_unaccepted_c128_draft_states(slots, prefixes, accepted, drafts)

        def check(lengths, accepts):
            expected = torch.full_like(state, 7.0)
            for slot, prefix, keep in zip([1, 3, 2], lengths, accepts):
                for offset in range(keep, drafts):
                    row = slot * ring + (prefix + offset) % ring
                    expected[row, :width // 2] = 0
                    expected[row, width // 2:] = -torch.inf
            torch.testing.assert_close(state, expected, atol=0, rtol=0)
            self.assertTrue((unrelated == 11).all().item())

        with patch("sglang.srt.mem_cache.deepseek_v4_memory_pool.ONLINE_C128", False):
            call()
            check([126, 255, 127], [1, 6, 3])
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                call()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                call()
            for lengths, accepts in [([127, 128, 256], [6, 1, 2]),
                                     ([383, 0, 511], [2, 4, 6])]:
                state.fill_(7)
                prefixes.copy_(torch.tensor(lengths, device="cuda"))
                accepted.copy_(torch.tensor(accepts, device="cuda"))
                graph.replay()
                check(lengths, accepts)


if __name__ == "__main__":
    unittest.main()
