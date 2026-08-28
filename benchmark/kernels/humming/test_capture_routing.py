import base64
import array
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_routing import (
    chunk_ranges,
    decode_routed_experts,
    load_input_ids,
    load_model_shape,
    summarize_capture,
)


class CaptureRoutingTest(unittest.TestCase):
    def test_decode_and_summarize_preserve_layer_distribution(self):
        routed = array.array(
            "i",
            [
                0, 0, 1, 2, 2, 3,
                0, 0, 1, 3, 2, 2,
                0, 0, 2, 3, 3, 3,
            ],
        )
        encoded = base64.b64encode(routed.tobytes()).decode("ascii")
        decoded, digest = decode_routed_experts(
            encoded, num_tokens=3, num_layers=3, top_k=2
        )
        self.assertEqual(decoded, routed)
        self.assertEqual(len(digest), 64)

        points = summarize_capture(
            decoded,
            num_tokens=3,
            num_layers=3,
            top_k=2,
            first_moe_layer=1,
            num_hash_layers=0,
            num_experts=4,
            chunk_size=2,
        )
        self.assertEqual(len(points), 4)
        self.assertEqual(points[0]["valid_shape_m"], 4)
        self.assertEqual(points[0]["expert_counts"], [0, 2, 1, 1])
        self.assertEqual(points[2]["valid_shape_m"], 2)
        self.assertEqual(points[2]["expert_counts"], [0, 0, 1, 1])

    def test_decode_rejects_wrong_shape(self):
        encoded = base64.b64encode(array.array("i", [0, 0, 0]).tobytes()).decode()
        with self.assertRaisesRegex(ValueError, "byte length"):
            decode_routed_experts(
                encoded, num_tokens=2, num_layers=2, top_k=1
            )

    def test_summarize_rejects_invalid_expert(self):
        routed = array.array("i", [0, 4])
        with self.assertRaisesRegex(ValueError, "outside"):
            summarize_capture(
                routed,
                num_tokens=1,
                num_layers=2,
                top_k=1,
                first_moe_layer=1,
                num_hash_layers=0,
                num_experts=4,
                chunk_size=1,
            )

    def test_chunk_ranges_keep_remainder(self):
        self.assertEqual(chunk_ranges(9, 4), [(0, 4), (4, 8), (8, 9)])

    def test_formal_input_requires_file(self):
        with self.assertRaisesRegex(ValueError, "formal capture requires"):
            load_input_ids(None, 8, False)

    def test_model_shape_uses_text_config_and_dense_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "text_config": {
                            "num_hidden_layers": 7,
                            "num_experts_per_tok": 3,
                            "n_routed_experts": 8,
                            "first_k_dense_replace": 2,
                            "n_hash_layers": 3,
                        }
                    }
                )
            )
            shape = load_model_shape(path)
        self.assertEqual(shape["num_layers"], 7)
        self.assertEqual(shape["first_moe_layer"], 2)
        self.assertEqual(shape["top_k"], 3)
        self.assertEqual(shape["num_hash_layers"], 3)

    def test_model_shape_treats_null_dense_prefix_as_v4_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "num_hidden_layers": 43,
                        "num_experts_per_tok": 6,
                        "n_routed_experts": 256,
                        "first_k_dense_replace": None,
                    }
                )
            )
            shape = load_model_shape(path)
        self.assertEqual(shape["first_moe_layer"], 0)

    def test_summarize_rejects_uncaptured_hash_layer_sentinel(self):
        routed = array.array("i", [0, 1, 0, 2])
        with self.assertRaisesRegex(ValueError, "uncaptured device-cache sentinel"):
            summarize_capture(
                routed,
                num_tokens=2,
                num_layers=2,
                top_k=1,
                first_moe_layer=0,
                num_hash_layers=1,
                num_experts=4,
                chunk_size=2,
            )


if __name__ == "__main__":
    unittest.main()
