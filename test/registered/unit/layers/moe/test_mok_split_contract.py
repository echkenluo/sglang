import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.mok_fp8_native import (
    _admit_workspace_geometry,
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
        self.assertEqual(envs.SGLANG_OPT_MOK_MAX_TOKENS.default, 4096)
        self.assertEqual(envs.SGLANG_OPT_MOK_MAX_SEQUENCE_TOKENS.default, 16384)
        self.assertEqual(envs.SGLANG_OPT_MOK_WORKSPACE_CACHE_CAP.default, 6)

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

    def test_max_tokens_gate_falls_back_before_runtime_contract(self):
        from unittest import mock

        from sglang.srt.layers.moe.moe_runner import mok_fp8_native

        hidden = torch.empty((16385, 256), dtype=torch.bfloat16)
        with (
            envs.SGLANG_OPT_MOK_MIN_TOKENS.override(256),
            envs.SGLANG_OPT_MOK_MAX_TOKENS.override(16384),
            mock.patch.object(
                mok_fp8_native, "get_is_extend_in_batch", return_value=True
            ),
            mock.patch.object(
                mok_fp8_native,
                "get_tp_group",
                side_effect=AssertionError("must fall back before group lookup"),
            ),
        ):
            self.assertIsNone(
                mok_fp8_native.maybe_run_mok_fp8_native(
                    layer=mock.Mock(), hidden_states=hidden, topk_output=mock.Mock()
                )
            )

    def test_max_sequence_gate_catches_chunked_long_request(self):
        from unittest import mock

        from sglang.srt.layers.moe.moe_runner import mok_fp8_native

        # A 32K request reaches the model in <=16K chunks. The aggregate-token
        # gate alone therefore cannot protect the unsafe long-sequence region.
        hidden = torch.empty((16384, 256), dtype=torch.bfloat16)
        with (
            envs.SGLANG_OPT_MOK_MIN_TOKENS.override(256),
            envs.SGLANG_OPT_MOK_MAX_TOKENS.override(16384),
            envs.SGLANG_OPT_MOK_MAX_SEQUENCE_TOKENS.override(16384),
            mock.patch.object(
                mok_fp8_native, "get_is_extend_in_batch", return_value=True
            ),
            mock.patch.object(
                mok_fp8_native, "get_max_sequence_length", return_value=32768
            ),
            mock.patch.object(
                mok_fp8_native,
                "get_tp_group",
                side_effect=AssertionError("must fall back before group lookup"),
            ),
        ):
            self.assertIsNone(
                mok_fp8_native.maybe_run_mok_fp8_native(
                    layer=mock.Mock(), hidden_states=hidden, topk_output=mock.Mock()
                )
            )

    def test_workspace_geometry_cap_keeps_hits_and_rejects_only_misses(self):
        from unittest import mock

        from sglang.srt.layers.moe.moe_runner import mok_fp8_native

        group = mock.Mock(group_name="ep")
        base = {
            "group": group,
            "device": torch.device("cuda", 0),
            "hidden_size": 7168,
            "topk": 8,
            "num_local_experts": 64,
            "schedule_capacity_factor": 5,
        }
        with mok_fp8_native._WORKSPACE_GEOMETRY_LOCK:
            mok_fp8_native._WORKSPACE_GEOMETRIES.clear()
        with envs.SGLANG_OPT_MOK_WORKSPACE_CACHE_CAP.override(2):
            self.assertTrue(_admit_workspace_geometry(num_local_tokens=256, **base))
            self.assertTrue(_admit_workspace_geometry(num_local_tokens=512, **base))
            # A hit at the cap remains admitted: it must not trigger a clear.
            self.assertTrue(_admit_workspace_geometry(num_local_tokens=256, **base))
            self.assertFalse(_admit_workspace_geometry(num_local_tokens=768, **base))
        with mok_fp8_native._WORKSPACE_GEOMETRY_LOCK:
            self.assertEqual(len(mok_fp8_native._WORKSPACE_GEOMETRIES), 2)
            mok_fp8_native._WORKSPACE_GEOMETRIES.clear()

    def test_prefill_power_of_two_bucketing_bounds_workspace_geometries(self):
        from sglang.srt.layers.moe.moe_runner.mok_fp8_native import (
            _route_padding_config,
        )

        # The default retains the exact M256 geometry.
        self.assertEqual(_route_padding_config(1280, 8), (1280, 1024))
        self.assertEqual(_route_padding_config(3072, 8), (3072, 1024))
        # The opt-in maps scheduler-dependent prefill shapes to a bounded set.
        self.assertEqual(
            _route_padding_config(1280, 8, prefill_pow2_bucket=True),
            (2048, 1024),
        )
        self.assertEqual(
            _route_padding_config(3072, 8, prefill_pow2_bucket=True),
            (4096, 1024),
        )
        self.assertEqual(
            _route_padding_config(4096, 8, prefill_pow2_bucket=True),
            (4096, 1024),
        )
        # Decode-sized inputs retain the existing even-token contract.
        self.assertEqual(
            _route_padding_config(3, 8, prefill_pow2_bucket=True), (4, 16)
        )

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

    def test_runtime_contract_tolerates_interleave_flag_on_split_checkpoints(self):
        # V4 loads separate w1/w3 shards: _load_w13 places [gate; up] halves
        # unconditionally, and the deepgemm silu stage consumes halves, so a
        # default-True gate_up_interleaved config flag must NOT reject the
        # contract (audited 2026-08-25 after the CANARY3 strict raise). On
        # CPU the contract should instead fall through to the CUDA-tensor
        # requirement -- reaching it proves the interleave check is gone.
        from unittest import mock

        from sglang.srt.layers.moe.moe_runner import mok_fp8_native

        shapes = _make_shape_inputs()

        class Fp8MoEMethod:
            pass

        qm = Fp8MoEMethod()
        qm.quant_config = mock.Mock(weight_block_size=[128, 128])
        qm.is_fp4_expert = False
        qm.with_bias = False
        layer = mock.Mock()
        layer.quant_method = qm
        layer.moe_runner_config = mock.Mock(
            activation="silu", is_gated=True, swiglu_limit=10,
            apply_router_weight_on_input=False, no_combine=False,
            gate_up_interleaved=True,
        )
        layer.moe_tp_size = 1
        layer._dwdp_bound = False
        layer.use_triton_kernels = False
        layer.w13_weight = shapes["w13_weight"]
        layer.w13_weight_scale_inv = shapes["w13_scale"]
        layer.w2_weight = shapes["w2_weight"]
        layer.w2_weight_scale_inv = shapes["w2_scale"]
        layer.num_local_experts = shapes["num_local_experts"]
        layer.num_experts = shapes["num_global_experts"]
        layer.moe_ep_size = shapes["ep_size"]
        topk = mock.Mock(
            topk_ids=shapes["topk_ids"], topk_weights=shapes["topk_weights"]
        )

        with mock.patch.object(
            mok_fp8_native.deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False
        ), mock.patch(
            "sglang.srt.layers.moe.utils.is_sbo_enabled", return_value=False
        ), mock.patch(
            "sglang.srt.layers.moe.utils.is_tbo_enabled", return_value=False
        ):
            reason = mok_fp8_native.native_runtime_contract_error(
                layer, shapes["hidden_states"], topk
            )
        self.assertEqual(reason, "all full-native tensors must be CUDA tensors")

    def test_function_level_lazy_imports_resolve(self):
        # _run_native_core lazily imports kernels at first execution;
        # py_compile and this CPU suite never execute those lines, which is
        # how a v0.5.15->v0.5.17 module move (fp8_kernel) survived to a live
        # scheduler crash (CANARY5). Pin resolution of every function-level
        # `from sglang...` import in the module.
        import ast
        import importlib
        import inspect

        from sglang.srt.layers.moe.moe_runner import mok_fp8_native

        tree = ast.parse(inspect.getsource(mok_fp8_native))
        lazy = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.ImportFrom) and sub.module and \
                            sub.module.startswith("sglang"):
                        lazy.append((sub.module, [a.name for a in sub.names]))
        self.assertTrue(lazy, "expected at least one lazy sglang import")
        for module, names in lazy:
            mod = importlib.import_module(module)
            for name in names:
                self.assertTrue(
                    hasattr(mod, name), f"{module} lacks {name}"
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
