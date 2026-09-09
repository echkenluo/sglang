"""CPU tests of the live audit's independent layout and metric contracts."""

import importlib.util
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[5] / "python/sglang/srt/layers/moe/moe_runner/mok_fp8_live_audit.py"
spec = importlib.util.spec_from_file_location("mok_live_audit_under_test", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class LiveAuditContractTest(unittest.TestCase):
    def test_regroup_unsorted_with_empty_expert_and_padding(self):
        indices = [2, 0, 2, 3, 0, 2, 3]
        source, grouped, inverse = audit.grouped_row_plan(indices, 5, 4)
        self.assertEqual(len(source), 12)
        self.assertEqual(grouped, [0] * 4 + [2] * 4 + [3] * 4)
        self.assertEqual([source[i] for i in inverse], list(range(7)))
        for position, row in enumerate(source):
            if row >= 0:
                self.assertEqual(grouped[position], indices[row])

    def test_m64_to_production_m128_retains_every_row(self):
        indices = [3] * 64 + [0] * 192 + [2] * 128
        source, grouped, inverse = audit.grouped_row_plan(indices, 4)
        self.assertEqual(len(source), 512)
        self.assertEqual(source.count(-1), 128)
        for start in range(0, len(grouped), 128):
            self.assertEqual(len(set(grouped[start:start + 128])), 1)
        self.assertEqual([source[i] for i in inverse], list(range(len(indices))))

    def test_invalid_active_expert_is_rejected(self):
        for indices in ([], [-1], [4], [True]):
            with self.assertRaises(ValueError):
                audit.grouped_row_plan(indices, 4)

    def test_metrics_preserve_zero_row_and_bitwise_distinctions(self):
        actual = torch.tensor([[3.0, 4.0], [0.0, -0.0]], dtype=torch.bfloat16)
        reference = torch.tensor([[0.0, 4.0], [0.0, 0.0]], dtype=torch.bfloat16)
        metrics, rows = audit.comparison(actual, reference)
        self.assertEqual(metrics["exact_fraction"], .5)
        self.assertEqual(metrics["max_row_relative_l2"], .75)
        self.assertEqual(metrics["relative_l2"], .75)
        self.assertEqual(rows["error_squared"].tolist(), [9.0, 0.0])
        actual[1, 0] = 2
        metrics, _ = audit.comparison(actual, reference)
        self.assertEqual(metrics["max_row_relative_l2"], 2)


if __name__ == "__main__":
    unittest.main()
