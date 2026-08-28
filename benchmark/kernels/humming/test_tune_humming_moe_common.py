import array
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

# The local documentation workspace intentionally has no CUDA/PyTorch stack.
# Stub imports let us exercise the pure manifest/config-selection helpers; GPU
# behavior is validated in the H20 container.
torch_stub = types.ModuleType("torch")
torch_stub.nn = types.SimpleNamespace(Module=object)
sys.modules.setdefault("torch", torch_stub)
for name in (
    "humming",
    "humming.config",
    "humming.kernel",
    "humming.kernel.humming",
    "humming.layer",
    "humming.schema",
    "humming.tune",
    "humming.testing",
    "humming.testing.tuning",
    "sglang",
    "sglang.srt",
    "sglang.srt.layers",
    "sglang.srt.layers.moe",
    "sglang.srt.layers.moe.fused_moe_triton",
):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["humming.config"].ComputeConfig = object
sys.modules["humming.config"].GemmType = object
sys.modules["humming.config"].TuningConfig = object
sys.modules["humming.kernel.humming"].HummingKernel = object
sys.modules["humming.layer"].HummingMethod = object
sys.modules["humming.schema"].BaseInputSchema = object
sys.modules["humming.schema"].BaseWeightSchema = object
sys.modules["humming.schema"].HummingInputSchema = object
sys.modules["humming.tune"].get_heuristics_config = object
sys.modules["sglang.srt.layers.moe.fused_moe_triton"].moe_align_block_size = object

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_humming_moe import (  # noqa: E402
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    cap_candidate_configs,
    choose_representative_points,
    deduplicate_configs,
    load_capture,
    percentile,
    persistent_grid_values,
    w13_schedule_grid,
)


class TuneHummingCommonTest(unittest.TestCase):
    def test_default_gate_matches_humming_moe_tests(self):
        self.assertEqual(DEFAULT_RTOL, 0.01)
        self.assertEqual(DEFAULT_ATOL, 0.2)

    def test_persistent_grid_sweep_bounds_shape_specific_value(self):
        self.assertEqual(
            persistent_grid_values(2048, 8192),
            [
                2048,
                3072,
                3584,
                4096,
                4608,
                5120,
                5632,
                6144,
                8192,
                10240,
                12288,
                16384,
            ],
        )

    def test_w13_schedule_grid_preserves_exact_shape_geometry(self):
        heuristic = {
            "block_shape": [64, 128, 128],
            "warp_shape": [64, 16, 128],
            "num_stages": 4,
            "num_ctas_per_sm": 2,
            "num_sms": 78,
            "use_stream_k": True,
            "use_f16_accum": False,
        }

        candidates = w13_schedule_grid(heuristic)

        self.assertEqual(len(candidates), 24)
        self.assertEqual(
            sorted({config["num_sms"] for config in candidates}),
            [39, 52, 78, 104, 156],
        )
        self.assertEqual({config["num_stages"] for config in candidates}, {3, 4})
        self.assertEqual({config["num_ctas_per_sm"] for config in candidates}, {1, 2})
        self.assertTrue(
            all(
                config["block_shape"] == heuristic["block_shape"]
                for config in candidates
            )
        )
        self.assertTrue(
            all(
                config["warp_shape"] == heuristic["warp_shape"] for config in candidates
            )
        )
        self.assertEqual(
            {config["num_sms"] for config in candidates if not config["use_stream_k"]},
            {78},
        )

    def test_w13_schedule_grid_rejects_unusable_heuristic(self):
        with self.assertRaisesRegex(ValueError, "positive num_sms"):
            w13_schedule_grid({"num_sms": 0, "num_stages": 4})
        with self.assertRaisesRegex(ValueError, "at least three stages"):
            w13_schedule_grid({"num_sms": 78, "num_stages": 2})

    def test_candidate_cap_supports_full_ladder_and_smoke(self):
        configs = [{"id": index} for index in range(10)]
        self.assertEqual(cap_candidate_configs(configs, 4), configs[:4])
        self.assertEqual(cap_candidate_configs(configs, 1), configs[:1])
        self.assertEqual(cap_candidate_configs(configs, 0), configs)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            cap_candidate_configs(configs, -1)

    def test_deduplicate_configs_normalizes_tuples(self):
        configs = [
            {"block_shape": (64, 128, 128)},
            {"block_shape": [64, 128, 128]},
        ]
        self.assertEqual(
            deduplicate_configs(configs), [{"block_shape": [64, 128, 128]}]
        )

    def test_representative_points_cover_extremes(self):
        points = [
            {
                "chunk_index": 0,
                "layer_id": index,
                "max_expert_rows": index,
                "active_experts": 8,
            }
            for index in range(9)
        ]
        selected = choose_representative_points(points, 3)
        self.assertEqual([point["layer_id"] for point in selected], [0, 4, 8])

    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(percentile([5, 1, 3, 2, 4], 0.95), 5)
        self.assertEqual(percentile([5, 1, 3, 2, 4], 0.5), 3)

    def test_load_capture_checks_raw_hash_and_selects_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = array.array("i", [0, 1, 1, 0])
            raw = values.tobytes()
            (root / "routes.i32").write_bytes(raw)
            import hashlib

            manifest = {
                "state": "CAPTURED",
                "raw_dtype": "little-endian-int32",
                "raw_shape": [2, 1, 2],
                "raw_file": "routes.i32",
                "routed_payload_sha256": hashlib.sha256(raw).hexdigest(),
                "points": [
                    {
                        "chunk_index": 0,
                        "layer_id": 0,
                        "valid_shape_m": 4,
                        "max_expert_rows": 2,
                        "active_experts": 2,
                    }
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            _, loaded, points, shape_m = load_capture(path, None, 1)
        self.assertEqual(loaded, values)
        self.assertEqual(len(points), 1)
        self.assertEqual(shape_m, 4)

    def test_load_capture_rejects_invalid_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "INVALID.json").write_text('{"state":"INVALID"}\n')
            manifest = root / "manifest.json"
            manifest.write_text('{"state":"CAPTURED"}\n')
            with self.assertRaisesRegex(ValueError, "INVALID marker"):
                load_capture(manifest, None, 1)


if __name__ == "__main__":
    unittest.main()
