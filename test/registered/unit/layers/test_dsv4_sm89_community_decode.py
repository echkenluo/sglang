# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch checks; GPU cache/numerical/Graph validation is separate."""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
BACKEND = ROOT / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
KERNEL = ROOT / "python/sglang/kernels/ops/attention/flash_mla_sm120_triton.py"
spec = importlib.util.spec_from_file_location(
    "community_decode_ci_registry", ROOT / "python/sglang/test/ci/ci_register.py"
)
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)
registry.register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def dispatch(enabled, mode, ratio=4, sm120=False, flashinfer=True):
    # Execute the actual dispatch statement, substituting only GPU operators.
    tree = ast.parse(BACKEND.read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and n.orelse
        and isinstance(n.test, ast.BoolOp)
        and isinstance(n.test.values[0], ast.Attribute)
        and n.test.values[0].attr == "_sm89_sparse_decode"
    )
    calls = []

    def operator(name):
        def call(**kwargs):
            calls.append((name, kwargs))
            return name, None

        return call

    paths = {
        "sglang.kernels.ops.attention.flash_mla_sm120_triton": SimpleNamespace(
            flash_mla_sparse_decode_triton=operator("community")
        ),
        "sglang.kernels.ops.attention.flash_mla_sm120": SimpleNamespace(
            flash_mla_with_kvcache_sm120=operator("sm120")
        ),
        "sglang.kernels.ops.attention.nsa_triton_decode": SimpleNamespace(
            triton_fp8_attention_fwd=operator("nsa")
        ),
    }
    state = SimpleNamespace(
        _sm89_sparse_decode=enabled,
        _sm89_sparse_decode_seen=set(),
        _sm89_flashinfer=operator("flashinfer") if flashinfer else None,
        head_dim_v=512,
        softmax_scale=512**-0.5,
    )
    mode_object = SimpleNamespace(
        is_decode_or_idle=lambda: mode in ("decode", "idle"),
        is_target_verify=lambda: mode == "verify",
        is_draft_extend_v2=lambda: mode == "draft",
    )
    namespace = dict(
        self=state,
        forward_batch=SimpleNamespace(forward_mode=mode_object),
        _is_sm120=sm120,
        _is_sm89=not sm120,
        _is_xpu=False,
        q=SimpleNamespace(shape=(6, 1, 8, 512)),
        swa_k_cache=object(),
        swa_page_indices=object(),
        swa_topk_lengths=object(),
        attn_sink=object(),
        extra_k_cache=object() if ratio else None,
        extra_indices=object() if ratio else None,
        extra_topk_lengths=object() if ratio else None,
        flashmla_metadata=object(),
        compress_ratio=ratio,
        logger=SimpleNamespace(info=lambda *args: None),
    )
    with patch.dict(sys.modules, paths):
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(BACKEND), "exec"),
            namespace,
        )
    return calls, namespace


class TestCommunityDecodeDispatch(unittest.TestCase):
    def test_continuation_and_cache_arguments(self):
        for mode in ("decode", "idle", "verify", "draft"):
            for ratio in (0, 4, 128):
                with self.subTest(mode=mode, ratio=ratio):
                    calls, n = dispatch(True, mode, ratio)
                    self.assertEqual(len(calls), 1)
                    name, kwargs = calls[0]
                    self.assertEqual(name, "community")
                    for key in ("q", "attn_sink", "extra_k_cache", "extra_indices"):
                        self.assertIs(kwargs[key], n[key])
                    self.assertIs(kwargs["extra_topk_length"], n["extra_topk_lengths"])
                    self.assertIs(kwargs["indices"], n["swa_page_indices"])
                    self.assertIs(kwargs["topk_length"], n["swa_topk_lengths"])
                    self.assertEqual(kwargs["head_dim_v"], 512)
                    self.assertEqual(n["self"]._sm89_sparse_decode_seen, {ratio})

    def test_regular_prefill_stays_on_original_route(self):
        self.assertEqual(dispatch(True, "prefill")[0][0][0], "flashinfer")

    def test_disabled_keeps_original_continuation(self):
        for mode in ("prefill", "decode", "verify", "draft"):
            self.assertEqual(dispatch(False, mode)[0][0][0], "flashinfer")

    def test_disabled_preserves_nsa_fallback(self):
        self.assertEqual(dispatch(False, "decode", flashinfer=False)[0][0][0], "nsa")

    def test_sm120_original_dispatch(self):
        self.assertEqual(dispatch(False, "decode", sm120=True)[0][0][0], "sm120")

    def test_non_sm89_rejected_at_initialization(self):
        tree = ast.parse(BACKEND.read_text())
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and any(
                isinstance(v, ast.Constant)
                and v.value == "SGLANG_DSV4_SM89_SPARSE_DECODE requires SM89 GPUs"
                for v in ast.walk(n)
            )
        )
        # Select the innermost guard rather than the enclosing class/method.
        with self.assertRaisesRegex(ValueError, "requires SM89"):
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), str(BACKEND), "exec"),
                {"self": SimpleNamespace(_sm89_sparse_decode=True), "_is_sm89": False},
            )

    def test_fixed_sm89_and_other_device_configs(self):
        node = next(
            n
            for n in ast.parse(KERNEL.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "_get_sparse_decode_configs"
        )
        namespace = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            "triton": SimpleNamespace(Config=lambda values, **kwargs: (values, kwargs)),
        }
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(KERNEL), "exec"),
            namespace,
        )
        choose = namespace["_get_sparse_decode_configs"]
        self.assertEqual(
            choose((8, 9)), [({"BLOCK_T": 32}, {"num_warps": 8, "num_stages": 2})]
        )
        for capability in ((9, 0), (12, 0), None):
            self.assertEqual(len(choose(capability)), 3)

    def test_environment_default_disabled(self):
        tree = ast.parse((ROOT / "python/sglang/srt/environ.py").read_text())
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "SGLANG_DSV4_SM89_SPARSE_DECODE"
                for t in n.targets
            )
        )
        self.assertEqual(ast.literal_eval(node.value.args[0]), False)


if __name__ == "__main__":
    unittest.main()
