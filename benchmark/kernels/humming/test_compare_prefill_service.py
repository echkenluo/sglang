import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_prefill_service import compare_group


def make_leg(variant, median_32k, median_64k, p95_32k, p95_64k):
    measurements = [
        {"prompt_tokens": 32768, "response_sha256": "hash-32k"},
        {"prompt_tokens": 65536, "response_sha256": "hash-64k"},
    ]
    return {
        "state": "MEASURED",
        "variant": variant,
        "contract": {"cold_cache": True, "repeats": 8},
        "inputs": [
            {"prompt_tokens": 32768, "input_ids_sha256": "input-32k"},
            {"prompt_tokens": 65536, "input_ids_sha256": "input-64k"},
        ],
        "measurements": measurements,
        "summaries": {
            "32768": {
                "median_seconds": median_32k,
                "p95_seconds": p95_32k,
            },
            "65536": {
                "median_seconds": median_64k,
                "p95_seconds": p95_64k,
            },
        },
        "_source_path": f"/{variant}.json",
        "_source_sha256": f"sha-{variant}",
    }


class ComparePrefillServiceTest(unittest.TestCase):
    def test_valid_group_can_produce_service_go(self):
        a1 = make_leg("a1", 10.0, 20.0, 11.0, 22.0)
        candidate = make_leg("candidate", 9.7, 19.5, 10.8, 21.5)
        a2 = make_leg("a2", 10.1, 20.2, 11.1, 22.2)

        result = compare_group(a1, [candidate], a2, drift_threshold_pct=2.0)

        self.assertEqual(result["state"], "VALID")
        self.assertEqual(result["candidates"][0]["decision"], "SERVICE_GO")

    def test_baseline_drift_invalidates_candidate_decisions(self):
        a1 = make_leg("a1", 10.0, 20.0, 11.0, 22.0)
        candidate = make_leg("candidate", 9.0, 18.0, 10.0, 20.0)
        a2 = make_leg("a2", 11.0, 21.0, 12.0, 23.0)

        result = compare_group(a1, [candidate], a2, drift_threshold_pct=2.0)

        self.assertEqual(result["state"], "INVALID_DRIFT")
        self.assertEqual(
            result["candidates"][0]["decision"], "UNANSWERABLE_INVALID_GROUP"
        )

    def test_response_mismatch_invalidates_group(self):
        a1 = make_leg("a1", 10.0, 20.0, 11.0, 22.0)
        candidate = make_leg("candidate", 9.7, 19.5, 10.8, 21.5)
        candidate = copy.deepcopy(candidate)
        candidate["measurements"][1]["response_sha256"] = "different"
        a2 = make_leg("a2", 10.1, 20.2, 11.1, 22.2)

        result = compare_group(a1, [candidate], a2, drift_threshold_pct=2.0)

        self.assertEqual(result["state"], "INVALID_RESPONSE_MISMATCH")

    def test_contract_mismatch_fails_closed(self):
        a1 = make_leg("a1", 10.0, 20.0, 11.0, 22.0)
        candidate = make_leg("candidate", 9.7, 19.5, 10.8, 21.5)
        candidate["contract"] = {"cold_cache": False, "repeats": 8}
        a2 = make_leg("a2", 10.1, 20.2, 11.1, 22.2)

        with self.assertRaisesRegex(ValueError, "contracts do not match"):
            compare_group(a1, [candidate], a2, drift_threshold_pct=2.0)


if __name__ == "__main__":
    unittest.main()
