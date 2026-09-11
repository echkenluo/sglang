# SPDX-License-Identifier: Apache-2.0
"""CPU routing tests; these do not replace GPU Graph replay correctness tests."""
import ast
import contextlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

ROOT = Path(__file__).resolve().parents[4]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The registry and bucket validator are stdlib-only. Avoid importing the GPU
# runtime merely to check the startup/routing boundary on a CPU machine.
register_cpu_ci = load_file(
    "short_graph_ci_registry", ROOT / "python/sglang/test/ci/ci_register.py"
).register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")
validate = load_file(
    "short_graph_bucket_validator",
    ROOT / "python/sglang/srt/layers/dsv4_short_prefill_graph.py",
).validate_short_prefill_graph_buckets


def routing_context(backend="breakable", buckets=(32,), enabled=True, tp=8, dp=False):
    source = ROOT / "python/sglang/srt/layers/communicator.py"
    tree = ast.parse(source.read_text())
    names = {"_allow_dsv4_scattered_with_short_prefill_graph", "AttnTpContext"}
    # Exercise the production methods, while replacing their CUDA/runtime
    # dependencies with configuration-only objects.
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    flags = {
        "SGLANG_DSV4_TP_INPUT_SCATTERED": True,
        "SGLANG_DSV4_SHORT_PREFILL_GRAPH_WITH_COMM": enabled,
        "SGLANG_DSV4_TP_INPUT_SCATTERED_MIN_TOKENS": 512,
        "SGLANG_DSV4_TP_SCATTER_TBO": False,
    }
    envs = SimpleNamespace(
        **{key: SimpleNamespace(get=lambda v=value: v) for key, value in flags.items()}
    )
    namespace = {
        "envs": envs,
        "_is_cuda": True,
        "_is_npu": False,
        "get_exec": lambda: SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(backend=backend, bs=buckets)
                )
            )
        ),
        "get_parallel": lambda: SimpleNamespace(
            enable_attn_tp_input_scattered=True, tp_size=tp
        ),
        "get_spec": lambda: SimpleNamespace(speculative_algorithm="DSPARK"),
        "is_dp_attention_enabled": lambda: dp,
        "get_moe_a2a_backend": lambda: SimpleNamespace(is_none=lambda: True),
        "enable_moe_dense_fully_dp": lambda: False,
        "check_cuda_graph_backend": lambda phase, expected: backend == expected,
        "validate_short_prefill_graph_buckets": validate,
        "Backend": SimpleNamespace(DISABLED="disabled", TC_PIECEWISE="tc_piecewise"),
        "Phase": SimpleNamespace(PREFILL="prefill"),
        "logging": SimpleNamespace(info=lambda *args: None),
        "contextmanager": contextlib.contextmanager,
        "ForwardBatch": object,
        "AttentionInputs": object,
    }
    exec(compile(tree, str(source), "exec"), namespace)
    context = namespace["AttnTpContext"]()
    context.init_context(q_lora_rank=512, is_dsa=True)
    return context


def batch(tokens, extend=True, verify=False, tbo=False):
    return SimpleNamespace(
        input_ids=SimpleNamespace(shape=(tokens,)),
        forward_mode=SimpleNamespace(
            is_extend=lambda: extend, is_target_verify=lambda: verify
        ),
        can_run_tbo=tbo,
    )


def replay_eligible(tokens, enabled, prefix=0, verify=False):
    source = (
        ROOT
        / "python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py"
    )
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "can_replay_locally"
    )
    namespace = {
        "Optional": Optional,
        "Backend": SimpleNamespace(BREAKABLE="breakable"),
        "envs": SimpleNamespace(
            SGLANG_DSV4_SHORT_PREFILL_GRAPH_WITH_COMM=SimpleNamespace(
                get=lambda: enabled
            )
        ),
        "_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR": 2,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    runner = SimpleNamespace(
        _is_full_backend=False,
        prefill_backend_name="breakable",
        has_mha_companion_layers=False,
        max_num_tokens=256,
        capture_num_tokens=[21, 32, 64, 128, 256],
        _pad_to_bucket=lambda n, buckets: next(b for b in buckets if b >= n),
        _uses_eager_prefill_tail=lambda: True,
    )
    return namespace["can_replay_locally"](
        runner,
        batch_size=1,
        num_tokens=tokens,
        input_embeds=None,
        replace_embeds=None,
        prefix_lens=[prefix],
        is_target_verify=verify,
        capture_hidden_mode=None,
        return_logprob=True,
    )


class TestShortPrefillGraph(unittest.TestCase):
    def test_experimental_replay_requires_captured_token_shape(self):
        for prefix in (0, 4096):
            for tokens in (21, 32, 64, 128, 256):
                with self.subTest(prefix=prefix, tokens=tokens):
                    self.assertTrue(replay_eligible(tokens, True, prefix))
            for tokens in (20, 22, 33, 65, 129, 255, 257, 512):
                with self.subTest(prefix=prefix, tokens=tokens):
                    self.assertFalse(replay_eligible(tokens, True, prefix))
        self.assertFalse(replay_eligible(32, True, verify=True))

    def test_default_replay_retains_existing_padding_policy(self):
        for tokens in (20, 22, 33, 65, 129, 255):
            with self.subTest(tokens=tokens):
                self.assertTrue(replay_eligible(tokens, False))
        self.assertFalse(replay_eligible(257, False))
        self.assertFalse(replay_eligible(32, False, verify=True))

    def test_graph_and_communication_ranges_are_disjoint(self):
        context = routing_context()
        self.assertTrue(context.allow_input_scattered)
        for tokens in (1, 16, 32, 33, 128, 511):
            with self.subTest(tokens=tokens):
                self.assertFalse(context.use_input_scattered(batch(tokens)))
        for tokens in (512, 513, 4096, 32768):
            with self.subTest(tokens=tokens):
                self.assertTrue(context.use_input_scattered(batch(tokens)))

    def test_default_gate_preserves_old_behavior(self):
        self.assertFalse(routing_context(enabled=False).allow_input_scattered)
        context = routing_context(backend="disabled", enabled=False)
        self.assertTrue(context.use_input_scattered(batch(512)))
        self.assertFalse(context.use_input_scattered(batch(32)))

    def test_other_routing_exclusions_remain(self):
        context = routing_context()
        for kwargs in ({"extend": False}, {"verify": True}, {"tbo": True}):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(context.use_input_scattered(batch(4096, **kwargs)))
        self.assertFalse(routing_context(tp=1).allow_input_scattered)
        self.assertFalse(routing_context(dp=True).allow_input_scattered)

    def test_unsafe_or_unresolved_buckets_fail_at_startup(self):
        for buckets in (None, (), (512,), (32, 1024), (0,), (-1,), (True,), (32.0,)):
            with self.subTest(buckets=buckets), self.assertRaises(ValueError):
                routing_context(buckets=buckets)
        for backend in ("full", "tc_piecewise"):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                routing_context(backend=backend)
        for threshold in (0, -1, True, 512.0):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                validate("breakable", [32], threshold)
        validate("breakable", [32, 128, 511], 512)


if __name__ == "__main__":
    unittest.main()
