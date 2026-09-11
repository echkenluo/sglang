"""SM89 exact budget/tie and CUDA Graph checks for community DSpark scheduling."""

import unittest

import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    schedule_verify_lens_topk_triton,
)
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkScheduleConfig


def reference(confidence, budget, cfg):
    # CPU enumeration is independent of the parallel rank/select implementation.
    survival = confidence.float().cpu().cumprod(dim=1)
    candidates = []
    for request, row in enumerate(survival.tolist()):
        for position, probability in enumerate(row[: cfg.resolved_max_verify_len()]):
            if probability >= cfg.survival_eps:
                candidates.append((-probability, position, request))
    candidates.sort()
    counts = [cfg.min_verify_len] * confidence.shape[0]
    for _, _, request in candidates[: max(budget, 0)]:
        counts[request] += 1
    return torch.tensor(
        [min(max(c, max(cfg.min_verify_len, 1)), cfg.resolved_max_verify_len()) for c in counts],
        dtype=torch.int32, device=confidence.device,
    )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "Requires SM89",
)
class TestCommunitySchedule(unittest.TestCase):
    def test_exact_budget_and_ties(self):
        torch.manual_seed(7311)
        cfg = DSparkScheduleConfig(gamma=5, survival_eps=0.25)
        for bs in (0, 1, 2, 4, 8, 16, 32, 64):
            # Quarter increments keep cumulative products exactly representable,
            # including ties and the survival-epsilon boundary.
            inputs = [torch.randint(0, 5, (bs, 5), device="cuda").float() / 4]
            inputs += [torch.full((bs, 5), v, device="cuda") for v in (0.0, 0.5, 1.0)]
            for variant, confidence in enumerate(inputs):
                for budget in sorted({-1, 0, 1, bs, bs * 5 - 1, bs * 5, bs * 5 + 3}):
                    with self.subTest(bs=bs, variant=variant, budget=budget):
                        actual = schedule_verify_lens_topk_triton(
                            confidence=confidence, budget=budget, cfg=cfg,
                        )
                        torch.testing.assert_close(
                            actual, reference(confidence, budget, cfg), atol=0, rtol=0,
                        )

    def test_graph_updates(self):
        torch.manual_seed(7312)
        cfg = DSparkScheduleConfig(gamma=5, survival_eps=0.25)
        for bs, budget in ((1, 0), (1, 3), (4, 7), (16, 80)):
            with self.subTest(bs=bs, budget=budget):
                confidence = torch.randint(0, 5, (bs, 5), device="cuda").float() / 4
                for _ in range(3):
                    schedule_verify_lens_topk_triton(confidence=confidence, budget=budget, cfg=cfg)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = schedule_verify_lens_topk_triton(
                        confidence=confidence, budget=budget, cfg=cfg,
                    )
                for value in (0.0, 1.0, 0.5):
                    confidence.fill_(value)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        output, reference(confidence, budget, cfg), atol=0, rtol=0,
                    )


if __name__ == "__main__":
    unittest.main()
