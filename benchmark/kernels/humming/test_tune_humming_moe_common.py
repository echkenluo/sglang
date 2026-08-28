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
sys.modules["humming.kernel.humming"].HummingKernel = object
sys.modules["humming.layer"].HummingMethod = object
sys.modules["humming.schema"].BaseInputSchema = object
sys.modules["humming.schema"].BaseWeightSchema = object
sys.modules["humming.schema"].HummingInputSchema = object
sys.modules["humming.testing.tuning"].sample_test_tuning_configs = object
sys.modules[
    "sglang.srt.layers.moe.fused_moe_triton"
].moe_align_block_size = object

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_humming_moe import (  # noqa: E402
    choose_representative_points,
    deduplicate_configs,
    load_capture,
    percentile,
)


class TuneHummingCommonTest(unittest.TestCase):
    def test_deduplicate_configs_normalizes_tuples(self):
        configs = [
            {"block_shape": (64, 128, 128)},
            {"block_shape": [64, 128, 128]},
        ]
        self.assertEqual(deduplicate_configs(configs), [{"block_shape": [64, 128, 128]}])

    def test_representative_points_cover_extremes(self):
        points = [
            {"chunk_index": 0, "layer_id": index, "max_expert_rows": index, "active_experts": 8}
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


if __name__ == "__main__":
    unittest.main()
