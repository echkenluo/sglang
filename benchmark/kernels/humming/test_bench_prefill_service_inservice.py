import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_prefill_service_inservice import (
    analyze_measurements,
    build_round_schedule,
    selector_order,
    set_selector,
)


def make_measurements(candidate_scale=0.98, drift_scale=1.0):
    rows = []
    for shape in (32768, 65536):
        for round_index in range(12):
            for position, selector in enumerate(selector_order(round_index)):
                latency = 2.0 if shape == 32768 else 4.0
                if selector:
                    latency *= candidate_scale
                elif position == 2:
                    latency *= drift_scale
                rows.append(
                    {
                        "prompt_tokens": shape,
                        "round_index": round_index,
                        "position": position,
                        "selector": selector,
                        "elapsed_seconds": latency,
                        "response_sha256": f"same-{shape}",
                    }
                )
    return rows


class InServicePrefillBenchTest(unittest.TestCase):
    @patch("bench_prefill_service_inservice.request_json")
    def test_selector_requires_all_rank_readbacks(self, request_json):
        request_json.side_effect = [
            [True, True, True, True],
            {
                "internal_states": [
                    {"humming_indexed_w2_runtime_num_sms": 4096} for _ in range(4)
                ],
                "startup_time": {"boot": 1},
            },
        ]
        receipt = set_selector("http://server", 4096, 10)
        self.assertEqual(receipt["rank_values"], [4096] * 4)

    @patch("bench_prefill_service_inservice.request_json")
    def test_selector_rejects_partial_update(self, request_json):
        request_json.return_value = [True, False, True, True]
        with self.assertRaises(RuntimeError):
            set_selector("http://server", 5120, 10)

    def test_schedule_is_balanced_and_deterministic(self):
        first = build_round_schedule([32768, 65536], 12, 7)
        second = build_round_schedule([32768, 65536], 12, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(sum(row["prompt_tokens"] == 32768 for row in first), 12)

    def test_valid_go_uses_bracketed_pairs(self):
        result = analyze_measurements(make_measurements(), [32768, 65536], 12)
        self.assertEqual(result["state"], "VALID")
        self.assertEqual(result["decision"], "SERVICE_GO")
        self.assertEqual(result["candidate_decisions"]["4096"], "SERVICE_GO")

    def test_response_mismatch_invalidates_whole_group(self):
        rows = make_measurements()
        rows[0]["response_sha256"] = "different"
        result = analyze_measurements(rows, [32768, 65536], 12)
        self.assertEqual(result["state"], "INVALID")
        self.assertEqual(result["decision"], "INVALID")

    def test_bracket_drift_invalidates_whole_group(self):
        result = analyze_measurements(
            make_measurements(drift_scale=1.05), [32768, 65536], 12
        )
        self.assertEqual(result["state"], "INVALID")

    def test_flat_candidate_is_service_no_go(self):
        result = analyze_measurements(
            make_measurements(candidate_scale=1.0), [32768, 65536], 12
        )
        self.assertEqual(result["state"], "VALID")
        self.assertEqual(result["decision"], "SERVICE_NO_GO")


if __name__ == "__main__":
    unittest.main()
