"""CPU tests of the live audit's independent layout and metric contracts."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


SOURCE = Path(__file__).resolve().parents[5] / "python/sglang/srt/layers/moe/moe_runner/mok_fp8_live_audit.py"
spec = importlib.util.spec_from_file_location("mok_live_audit_under_test", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class LiveAuditContractTest(unittest.TestCase):
    def test_capture_allocates_only_first_eligible_call(self):
        layer = SimpleNamespace(layer_id=5, w13_weight=torch.empty(2, 8))
        state = SimpleNamespace(capacity=12, hidden=torch.empty(12, 4))
        with patch.dict(os.environ, {"SGLANG_MOK_LIVE_AUDIT_W13": "1",
                                    "SGLANG_MOK_LIVE_AUDIT_DIR": "/unused",
                                    "SGLANG_MOK_LIVE_AUDIT_MIN_TOKENS": "256"}), \
                patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), \
                patch.object(audit, "_SEEN", set()):
            self.assertIsNone(audit.prepare_w13_capture(layer, state, "c2s4", 255))
            capture = audit.prepare_w13_capture(layer, state, "c2s4", 256)
            self.assertEqual(capture.shape, (12, 8))
            self.assertEqual(capture.dtype, torch.bfloat16)
            self.assertTrue(torch.isnan(capture).all())
            audit._SEEN.add(5)
            self.assertIsNone(audit.prepare_w13_capture(layer, state, "c2s4", 256))

    def test_capture_rejects_unbound_or_unsupported_mode(self):
        with patch.dict(os.environ, {"SGLANG_MOK_LIVE_AUDIT_W13": "0"}):
            self.assertIsNone(audit.prepare_w13_capture(None, None, "c1s6", 256))
        for env in ({"SGLANG_MOK_LIVE_AUDIT_W13": "yes"},
                    {"SGLANG_MOK_LIVE_AUDIT_W13": "1", "SGLANG_MOK_LIVE_AUDIT_DIR": ""},
                    {"SGLANG_MOK_LIVE_AUDIT_W13": "1", "SGLANG_MOK_LIVE_AUDIT_DIR": "/unused"}):
            with patch.dict(os.environ, env), self.assertRaises(ValueError):
                audit.prepare_w13_capture(None, None, "c1s6", 256)
        with patch.dict(os.environ, {"SGLANG_MOK_LIVE_AUDIT_W13": "1",
                                    "SGLANG_MOK_LIVE_AUDIT_DIR": "/unused"}), \
                patch.object(torch.cuda, "is_current_stream_capturing", return_value=True), \
                self.assertRaises(ValueError):
            audit.prepare_w13_capture(None, None, "c2s4", 256)

    def test_capture_sampling_keeps_prior_tail_even_outside_current_top8(self):
        indices = [28] * 20; peers = [1] * 19 + [-1]
        slots = list(range(20)); slots[10] = 397 * 6 + 1
        rows = {"error_squared": torch.arange(20, 0, -1, dtype=torch.float64),
                "reference_squared": torch.ones(20, dtype=torch.float64)}
        selected, why = audit.capture_sample_rows(indices, peers, slots, rows, 0, 39)
        self.assertEqual(why["top8_current_rank_layer"], list(range(8)))
        self.assertEqual(why["previous_worst_match_rows"], [10])
        self.assertEqual(selected, list(range(8)) + [10, 18])
        slots[11] = slots[10]
        with self.assertRaises(ValueError):
            audit.capture_sample_rows(indices, peers, slots, rows, 0, 39)

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
