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

    def _gate_call(self, tokens, extend, min_tokens):
        from unittest import mock

        from sglang.srt.layers.moe.moe_runner import mok_fp8_native

        hidden = torch.empty((tokens, 256), dtype=torch.bfloat16)
        sentinel = RuntimeError("gate passed")
        with envs.SGLANG_OPT_MOK_MIN_TOKENS.override(min_tokens), mock.patch.object(
            mok_fp8_native, "get_is_extend_in_batch", return_value=extend
        ), mock.patch.object(mok_fp8_native, "get_tp_group", side_effect=sentinel):
            try:
                result = mok_fp8_native.maybe_run_mok_fp8_native(
                    layer=mock.Mock(), hidden_states=hidden, topk_output=mock.Mock()
                )
            except RuntimeError as exc:
                self.assertIs(exc, sentinel)
                return "passed_gate"
        self.assertIsNone(result)
        return "gated_off"

    def test_min_tokens_gate_rejects_decode_even_when_large(self):
        self.assertEqual(self._gate_call(4096, extend=False, min_tokens=256), "gated_off")

    def test_min_tokens_gate_boundary_255_256_257(self):
        self.assertEqual(self._gate_call(255, extend=True, min_tokens=256), "gated_off")
        self.assertEqual(self._gate_call(256, extend=True, min_tokens=256), "passed_gate")
        self.assertEqual(self._gate_call(257, extend=True, min_tokens=256), "passed_gate")

    def test_min_tokens_gate_defaults_off_and_skips_forward_context(self):
        from unittest import mock

        from sglang.srt.layers.moe.moe_runner import mok_fp8_native

        hidden = torch.empty((8, 256), dtype=torch.bfloat16)
        sentinel = RuntimeError("gate bypassed")
        # With the default min_tokens=0 the gate must not touch the forward
        # context at all: get_is_extend_in_batch would raise AssertionError
        # if consulted, so reaching the get_tp_group sentinel proves bypass.
        with mock.patch.object(
            mok_fp8_native,
            "get_is_extend_in_batch",
            side_effect=AssertionError("must not be called"),
        ), mock.patch.object(mok_fp8_native, "get_tp_group", side_effect=sentinel):
            with self.assertRaises(RuntimeError) as ctx:
                mok_fp8_native.maybe_run_mok_fp8_native(
                    layer=mock.Mock(), hidden_states=hidden, topk_output=mock.Mock()
                )
        self.assertIs(ctx.exception, sentinel)
        self.assertEqual(envs.SGLANG_OPT_MOK_MIN_TOKENS.default, 0)

    def test_hit_counter_records_and_logs_thresholds(self):
        from sglang.srt.layers.moe.moe_runner import mok_fp8_native

        with envs.SGLANG_MOE_PATH_HIT_COUNTERS.override(True):
            mok_fp8_native._HIT_COUNTS.clear()
            emitted = []
            for i in range(501):
                line = mok_fp8_native._note_path_hit(
                    "mok", mode="extend", num_tokens=512
                )
                if line:
                    emitted.append((i, line))
        self.assertEqual(
            mok_fp8_native._HIT_COUNTS[("mok", "extend", "ge256")], 501
        )
        # first hit and every 500th emit a line
        self.assertEqual([i for i, _ in emitted], [0, 499])
        self.assertIn("path=mok mode=extend bucket=ge256", emitted[0][1])

    def test_hit_counter_buckets_and_na(self):
        from sglang.srt.layers.moe.moe_runner import mok_fp8_native

        with envs.SGLANG_MOE_PATH_HIT_COUNTERS.override(True):
            mok_fp8_native._HIT_COUNTS.clear()
            mok_fp8_native._note_path_hit("mok", mode="extend", num_tokens=255)
            mok_fp8_native._note_path_hit("warp_masked", mode="decode", num_tokens=None)
        self.assertEqual(
            mok_fp8_native._HIT_COUNTS[("mok", "extend", "lt256")], 1
        )
        self.assertEqual(
            mok_fp8_native._HIT_COUNTS[("warp_masked", "decode", "na")], 1
        )

    def test_hit_counter_disabled_is_noop(self):
        from sglang.srt.layers.moe.moe_runner import mok_fp8_native

        mok_fp8_native._HIT_COUNTS.clear()
        self.assertIsNone(
            mok_fp8_native._note_path_hit("mok", mode="decode", num_tokens=8)
        )
        self.assertEqual(len(mok_fp8_native._HIT_COUNTS), 0)
        self.assertFalse(envs.SGLANG_MOE_PATH_HIT_COUNTERS.default)


if __name__ == "__main__":
    unittest.main()
