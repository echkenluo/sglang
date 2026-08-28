import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_prefill_service import (
    build_measurement_schedule,
    load_input_ids,
    percentile,
    summarize,
)


class BenchPrefillServiceTest(unittest.TestCase):
    def test_load_input_ids_validates_and_hashes_exact_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(
                json.dumps(
                    {
                        "input_ids": [1, 2, 3],
                        "metadata": {"encoding_spec": "dsv4"},
                    }
                )
            )
            actual = load_input_ids(path)

        self.assertEqual(actual["prompt_tokens"], 3)
        self.assertEqual(len(actual["input_ids_sha256"]), 64)
        self.assertEqual(actual["source_metadata"], {"encoding_spec": "dsv4"})

    def test_load_input_ids_rejects_bool_and_negative_tokens(self):
        for input_ids in ([1, True], [1, -1]):
            with (
                self.subTest(input_ids=input_ids),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "input.json"
                path.write_text(json.dumps({"input_ids": input_ids}))
                with self.assertRaises(ValueError):
                    load_input_ids(path)

    def test_schedule_is_deterministic_and_balanced(self):
        inputs = [{"prompt_tokens": 32768}, {"prompt_tokens": 65536}]
        first = build_measurement_schedule(inputs, repeats=5, seed=7)
        second = build_measurement_schedule(inputs, repeats=5, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(
            [item["input"]["prompt_tokens"] for item in first].count(32768), 5
        )
        self.assertEqual(
            [item["input"]["prompt_tokens"] for item in first].count(65536), 5
        )

    def test_summary_uses_interpolated_percentiles(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile(values, 0.5), 2.5)
        actual = summarize(values)
        self.assertEqual(actual["count"], 4)
        self.assertEqual(actual["median_seconds"], 2.5)
        self.assertAlmostEqual(actual["p90_seconds"], 3.7)


if __name__ == "__main__":
    unittest.main()
