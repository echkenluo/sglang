import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_humming_w13_screens import merge_screens


def make_payload(shard_index, selected, valid, rejected):
    universe = ["h", "a", "b", "c", "d"]
    return {
        "state": "SCREENED",
        "capture_sha256": "capture",
        "model_config_sha256": "model",
        "tp_size": 4,
        "shape_m": 196608,
        "humming_version": "0.1.13",
        "torch_version": "torch",
        "cuda_version": "cuda",
        "device_name": "H20",
        "device_capability": [9, 0],
        "w13_sampler": {"source_sha256": "sampler"},
        "parameters": {
            "route_split": "train",
            "correctness_only": True,
            "candidate_rejection_policy": "filter",
            "candidate_shard_count": 2,
            "candidate_shard_index": shard_index,
            "seed": 20260831,
        },
        "sublayers": {
            "w13": {
                "state": "SCREENED",
                "rejection_policy": "filter",
                "correctness_only": True,
                "candidate_shard_count": 2,
                "candidate_shard_index": shard_index,
                "candidate_source": "official_humming_0.1.13_sampler",
                "candidate_universe_count": len(universe),
                "candidate_universe_ids": universe,
                "candidate_universe_sha256": "universe",
                "heuristic_id": "h",
                "heuristic_config": {"id": "h"},
                "correctness_gate": {"rtol": 0.01, "atol": 0.2},
                "route_points": [{"layer_id": 1}],
                "selected_candidate_ids": selected,
                "valid_candidates": [
                    {"config_id": item, "config": {"id": item}}
                    for item in valid
                ],
                "rejected": [
                    {"config_id": item, "phase": "correctness"}
                    for item in rejected
                ],
            }
        },
    }


class MergeHummingW13ScreensTest(unittest.TestCase):
    def test_merge_requires_full_disjoint_coverage_and_preserves_order(self):
        payloads = [
            make_payload(0, ["h", "a", "c"], ["h", "a", "c"], []),
            make_payload(1, ["h", "b", "d"], ["h", "d"], ["b"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            sources = []
            for index, payload in enumerate(payloads):
                path = Path(directory) / f"screen-{index}.json"
                path.write_text(json.dumps(payload))
                sources.append(str(path))
            result = merge_screens(payloads, sources)
        self.assertEqual(result["state"], "MERGED")
        self.assertEqual(result["candidate_ids"], ["h", "a", "c", "d"])
        self.assertEqual(result["survivor_count"], 4)
        self.assertEqual(result["rejected_count"], 1)

    def test_merge_rejects_overlapping_shards(self):
        payloads = [
            make_payload(0, ["h", "a", "c"], ["h", "a", "c"], []),
            make_payload(1, ["h", "a", "d"], ["h", "a", "d"], []),
        ]
        with tempfile.TemporaryDirectory() as directory:
            sources = []
            for index, payload in enumerate(payloads):
                path = Path(directory) / f"screen-{index}.json"
                path.write_text(json.dumps(payload))
                sources.append(str(path))
            with self.assertRaisesRegex(ValueError, "incomplete or overlapping"):
                merge_screens(copy.deepcopy(payloads), sources)

    def test_replicated_merge_intersects_cross_gpu_survivors(self):
        payloads = [
            make_payload(
                0, ["h", "a", "b", "c", "d"], ["h", "a", "c", "d"], ["b"]
            ),
            make_payload(
                1, ["h", "a", "b", "c", "d"], ["h", "a", "b", "d"], ["c"]
            ),
        ]
        for index, payload in enumerate(payloads):
            payload["parameters"]["seed"] = 20260902 + index
            payload["parameters"]["replicate_candidate_universe"] = True
            payload["sublayers"]["w13"][
                "candidate_selection"
            ] = "replicated_full_universe"
        with tempfile.TemporaryDirectory() as directory:
            sources = []
            for index, payload in enumerate(payloads):
                path = Path(directory) / f"screen-{index}.json"
                path.write_text(json.dumps(payload))
                sources.append(str(path))
            result = merge_screens(payloads, sources, "replicated")
        self.assertEqual(result["candidate_ids"], ["h", "a", "d"])
        self.assertEqual(result["rejected_count"], 2)
        self.assertEqual(result["rejection_observation_count"], 2)
        self.assertEqual(result["screen_seeds"], {"0": 20260902, "1": 20260903})


if __name__ == "__main__":
    unittest.main()
