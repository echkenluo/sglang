import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.mok_fp8_native import (
    _conservative_route_capacity_factor,
    _route_padding_config,
    native_shape_contract_error,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _make_shape_inputs():
    tokens, hidden, intermediate = 3, 256, 256
    local_experts, ep_size, topk = 2, 4, 2
    return {
        "hidden_states": torch.empty((tokens, hidden), dtype=torch.bfloat16),
        "topk_ids": torch.empty((tokens, topk), dtype=torch.int32),
        "topk_weights": torch.empty((tokens, topk), dtype=torch.float32),
        "w13_weight": torch.empty(
            (local_experts, 2 * intermediate, hidden), dtype=torch.float8_e4m3fn
        ),
        "w13_scale": torch.empty(
            (local_experts, 2 * intermediate // 128, hidden // 128),
            dtype=torch.float32,
        ),
        "w2_weight": torch.empty(
            (local_experts, hidden, intermediate), dtype=torch.float8_e4m3fn
        ),
        "w2_scale": torch.empty(
            (local_experts, hidden // 128, intermediate // 128),
            dtype=torch.float32,
        ),
        "num_local_experts": local_experts,
        "num_global_experts": local_experts * ep_size,
        "ep_size": ep_size,
    }


class TestMoKSplitContract(unittest.TestCase):
    def test_v4_shape_contract(self):
        self.assertIsNone(native_shape_contract_error(**_make_shape_inputs()))

    def test_shape_contract_rejects_wrong_scale(self):
        inputs = _make_shape_inputs()
        inputs["w13_scale"] = inputs["w13_scale"][:, :1]
        self.assertIn(
            "invalid expert block-scale shapes",
            native_shape_contract_error(**inputs),
        )

    def test_decode_and_prefill_padding_are_distinct(self):
        self.assertEqual(_route_padding_config(1, 8), (2, 16))
        self.assertEqual(_route_padding_config(4, 8), (4, 16))
        self.assertEqual(_route_padding_config(5, 8), (256, 1024))

    def test_capacity_factor_covers_concentrated_routes(self):
        self.assertEqual(
            _conservative_route_capacity_factor(
                base_rows=2048,
                num_local_experts=2,
                ep_size=4,
                expert_padding=64,
            ),
            5,
        )

    def test_feature_is_opt_in(self):
        self.assertFalse(envs.SGLANG_OPT_USE_MOK_FP8_NATIVE.default)
        self.assertFalse(envs.SGLANG_OPT_MOK_FP8_NATIVE_STRICT.default)
        self.assertFalse(envs.SGLANG_OPT_MOK_FP8_NATIVE_PREFILL_GRAPH.default)


if __name__ == "__main__":
    unittest.main()
