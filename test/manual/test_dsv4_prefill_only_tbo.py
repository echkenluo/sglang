"""Prefill-only TBO: predicate and decode cuda-graph bucket alignment.

The DSV4 TP-scattered chunk pipeline overlaps micro-batches of a prefill chunk
only. Everything --enable-two-batch-overlap adds for the sake of a graph
captured TBO *decode* must stay off there, and the first casualty was the
cuda-graph bucket list: the even-bs alignment drops the bs=1 bucket, so a
single-request decode replays the bs=2 graph and pays for a row it does not
have.

The functions under test live in modules that import torch / orjson, which the
serving image has and a dev box usually does not. Rather than skip, load the
two definitions straight out of the shipped source and run them against stubs
for the handful of names they touch -- the assertions below then hold against
the real code, not a copy of it.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMON_PY = _REPO_ROOT / "python/sglang/srt/utils/common.py"
_BASE_RUNNER_PY = (
    _REPO_ROOT / "python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py"
)


def _load_functions(source_path, names, namespace):
    """Exec the named top-level functions of `source_path` into `namespace`."""
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    wanted = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    missing = set(names) - set(wanted)
    if missing:
        raise AssertionError(f"{source_path.name} no longer defines {sorted(missing)}")
    # `from __future__ import annotations` keeps ServerArgs / ModelRunner hints
    # unevaluated, so the stub namespace needs no type-only names.
    lazy_annotations = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.Module(
        body=[lazy_annotations] + [wanted[name] for name in names], type_ignores=[]
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def _make_server_args(
    *,
    enable_two_batch_overlap=True,
    input_scattered=True,
    scatter_tbo=True,
    enable_dp_attention=False,
    moe_a2a_backend="none",
    decode_bs=None,
    torch_compile_max_bs=0,
):
    """A ServerArgs stand-in carrying only the fields the functions read."""
    return SimpleNamespace(
        enable_two_batch_overlap=enable_two_batch_overlap,
        enable_dp_attention=enable_dp_attention,
        moe_a2a_backend=moe_a2a_backend,
        torch_compile_max_bs=torch_compile_max_bs,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(
                bs=list(decode_bs if decode_bs is not None else [1, 2, 4, 8, 12])
            )
        ),
        _envs=SimpleNamespace(input_scattered=input_scattered, scatter_tbo=scatter_tbo),
    )


def _make_envs():
    """Stub `envs`, reading the flags off the ServerArgs stand-in under test."""

    class _Flag:
        def __init__(self, name):
            self.name = name

        def get(self):
            return _ACTIVE_ENVS[self.name]

    return SimpleNamespace(
        SGLANG_DSV4_TP_INPUT_SCATTERED=_Flag("input_scattered"),
        SGLANG_DSV4_TP_SCATTER_TBO=_Flag("scatter_tbo"),
    )


_ACTIVE_ENVS = {"input_scattered": True, "scatter_tbo": True}


def _bind_envs(server_args):
    _ACTIVE_ENVS["input_scattered"] = server_args._envs.input_scattered
    _ACTIVE_ENVS["scatter_tbo"] = server_args._envs.scatter_tbo


def _ceil_align(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


# Namespace shared by both loaded modules: get_batch_sizes_to_capture calls the
# real get_cuda_graph_batch_size_alignment / get_cuda_graph_max_batch_size.
_NS = {
    "envs": _make_envs(),
    "ceil_align": _ceil_align,
    # attn_tp / attn_cp do not shape this deployment's alignment (TP8, no DP
    # attention, no a2a backend -> no gathered buffer, cp_size 1).
    "get_parallel": lambda: SimpleNamespace(attn_tp_size=8, attn_cp_size=1),
    "require_gathered_buffer": lambda server_args: False,
    "get_flags": lambda: SimpleNamespace(
        capture=SimpleNamespace(enable_torch_compile=False)
    ),
}
_load_functions(
    _COMMON_PY,
    ["is_dsv4_prefill_only_tbo", "get_cuda_graph_batch_size_alignment"],
    _NS,
)
_NS["get_cuda_graph_max_batch_size"] = lambda server_args, max_batch_size: _ceil_align(
    max_batch_size, _NS["get_cuda_graph_batch_size_alignment"](server_args)
)
_load_functions(_BASE_RUNNER_PY, ["get_batch_sizes_to_capture"], _NS)

is_dsv4_prefill_only_tbo = _NS["is_dsv4_prefill_only_tbo"]
get_cuda_graph_batch_size_alignment = _NS["get_cuda_graph_batch_size_alignment"]
get_batch_sizes_to_capture = _NS["get_batch_sizes_to_capture"]


class TestPrefillOnlyTboPredicate(unittest.TestCase):
    """Only the DSV4 TP-scattered chunk pipeline is prefill-only TBO.

    Every other TBO deployment (DP attention, an EP a2a backend) does overlap
    decode, so it must keep the decode-side machinery.
    """

    def _check(self, expected, **kwargs):
        server_args = _make_server_args(**kwargs)
        _bind_envs(server_args)
        self.assertEqual(is_dsv4_prefill_only_tbo(server_args), expected)

    def test_scattered_chunk_pipeline_is_prefill_only(self):
        self._check(True)

    def test_false_without_tbo(self):
        self._check(False, enable_two_batch_overlap=False)

    def test_false_without_the_scatter_tbo_opt_in(self):
        self._check(False, scatter_tbo=False)

    def test_false_without_the_scattered_prefill_path(self):
        self._check(False, input_scattered=False)

    def test_false_under_dp_attention(self):
        """The DP path populates a split index for decode batches too."""
        self._check(False, enable_dp_attention=True)

    def test_false_with_an_ep_a2a_backend(self):
        """EP TBO overlaps the dispatch/combine of a decode batch."""
        self._check(False, moe_a2a_backend="deepep")


class TestDecodeGraphBucketAlignment(unittest.TestCase):
    """bs=1 must keep its own decode graph under prefill-only TBO.

    The x2 alignment exists so a captured TBO decode graph can split its rows
    into two equal micro-batches. Prefill-only TBO captures no such graph, so
    paying the alignment only costs the bs=1 bucket -- and with it ~9% of
    single-request decode, which then replays the bs=2 graph.
    """

    def _alignment(self, **kwargs):
        server_args = _make_server_args(**kwargs)
        _bind_envs(server_args)
        return get_cuda_graph_batch_size_alignment(server_args)

    def test_alignment_is_one_under_prefill_only_tbo(self):
        self.assertEqual(self._alignment(), 1)

    def test_alignment_matches_tbo_off(self):
        self.assertEqual(
            self._alignment(), self._alignment(enable_two_batch_overlap=False)
        )

    def test_alignment_is_two_for_ordinary_tbo(self):
        """Unchanged for the DP / EP deployments that do overlap decode."""
        self.assertEqual(self._alignment(enable_dp_attention=True), 2)
        self.assertEqual(self._alignment(moe_a2a_backend="deepep"), 2)

    def _capture_bs(self, *, captured_req_width, **kwargs):
        server_args = _make_server_args(**kwargs)
        _bind_envs(server_args)
        model_runner = SimpleNamespace(
            server_args=server_args,
            req_to_token_pool=SimpleNamespace(size=256),
        )
        capture_bs, _ = get_batch_sizes_to_capture(model_runner, captured_req_width)
        return capture_bs

    def test_plain_decode_keeps_the_bs_one_bucket(self):
        self.assertIn(1, self._capture_bs(captured_req_width=1))

    def test_dspark_verify_keeps_the_odd_buckets(self):
        """DSpark verifies 6 tokens per request; the buckets are per request."""
        capture_bs = self._capture_bs(captured_req_width=6, decode_bs=list(range(1, 9)))
        self.assertEqual(capture_bs, list(range(1, 9)))

    def test_ordinary_tbo_still_drops_the_odd_buckets(self):
        capture_bs = self._capture_bs(
            captured_req_width=1, decode_bs=list(range(1, 9)), enable_dp_attention=True
        )
        self.assertEqual(capture_bs, [2, 4, 6, 8])

    def test_bucket_list_matches_tbo_off(self):
        """The whole point: decode buckets are byte-identical to a non-TBO run."""
        self.assertEqual(
            self._capture_bs(captured_req_width=6, decode_bs=list(range(1, 9))),
            self._capture_bs(
                captured_req_width=6,
                decode_bs=list(range(1, 9)),
                enable_two_batch_overlap=False,
            ),
        )


if __name__ == "__main__":
    unittest.main()
