"""Offline contract tests for per-request spec-decoding usage export.

Loads protocol.py / usage_processor.py standalone via importlib (no torch, no
GPU, no sglang package import) so this runs on any dev box. Covers:
- meta_info -> SpecDecodingDetails mapping (values and absence semantics)
- non-streaming and streaming aggregation paths
- JSON shape: field absent-as-null when no verify ran (backward compat)
"""

import importlib.util
import pathlib
import sys
import types

HERE = pathlib.Path(__file__).resolve()
OPENAI_DIR = HERE.parents[1] / "srt" / "entrypoints" / "openai"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap():
    for pkg in (
        "sglang",
        "sglang.srt",
        "sglang.srt.entrypoints",
        "sglang.srt.entrypoints.openai",
    ):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = []  # mark as package for submodule registration
            sys.modules[pkg] = mod
    if "sglang.utils" not in sys.modules:
        utils_stub = types.ModuleType("sglang.utils")
        utils_stub.convert_json_schema_to_str = lambda schema: str(schema)
        sys.modules["sglang.utils"] = utils_stub
    protocol = _load(
        "sglang.srt.entrypoints.openai.protocol", OPENAI_DIR / "protocol.py"
    )
    processor = _load(
        "sglang.srt.entrypoints.openai.usage_processor",
        OPENAI_DIR / "usage_processor.py",
    )
    return protocol, processor


protocol, processor = _bootstrap()
UsageProcessor = processor.UsageProcessor


def _meta(completion=100, verify=25, correct=60, proposed=375, **extra):
    m = {
        "prompt_tokens": 1000,
        "completion_tokens": completion,
        "spec_verify_ct": verify,
        "spec_num_correct_drafts": correct,
        "spec_num_proposed_drafts": proposed,
    }
    m.update(extra)
    return m


def test_spec_details_values():
    d = UsageProcessor._spec_details_from_metas([_meta()])
    assert d is not None
    assert abs(d.accept_length - 100 / 25) < 1e-9
    assert abs(d.accept_rate - 60 / 375) < 1e-9
    assert d.verify_ct == 25


def test_no_verify_means_none():
    assert UsageProcessor._spec_details_from_metas([{"completion_tokens": 5}]) is None
    assert UsageProcessor._spec_details_from_metas([]) is None


def test_response_usage_carries_spec_details():
    usage = UsageProcessor.calculate_response_usage(
        [{"meta_info": _meta()}], n_choices=1
    )
    assert usage.sglang_spec_details is not None
    assert usage.sglang_spec_details.verify_ct == 25
    assert usage.completion_tokens == 100


def test_response_usage_without_spec_is_null_field():
    usage = UsageProcessor.calculate_response_usage(
        [{"meta_info": {"prompt_tokens": 10, "completion_tokens": 5}}], n_choices=1
    )
    assert usage.sglang_spec_details is None
    payload = usage.model_dump()
    assert payload["sglang_spec_details"] is None


def test_streaming_usage_aggregates_last_chunk_metas():
    spec_metas = {0: _meta(completion=80, verify=20, correct=40, proposed=300)}
    usage = UsageProcessor.calculate_streaming_usage(
        prompt_tokens={0: 1000},
        reasoning_tokens={0: 10},
        completion_tokens={0: 80},
        cached_tokens={0: 0},
        n_choices=1,
        spec_metas=spec_metas,
    )
    d = usage.sglang_spec_details
    assert d is not None
    assert abs(d.accept_length - 4.0) < 1e-9
    assert d.verify_ct == 20


def test_streaming_usage_default_is_backward_compatible():
    usage = UsageProcessor.calculate_streaming_usage(
        prompt_tokens={0: 10},
        reasoning_tokens={0: 0},
        completion_tokens={0: 4},
        cached_tokens={0: 0},
        n_choices=1,
    )
    assert usage.sglang_spec_details is None


def test_usageinfo_dump_shape_for_responses_path():
    """Non-streaming /v1/responses dumps UsageInfo directly; nested details must
    serialize as a plain dict with the three keys."""
    usage = protocol.UsageInfo(
        prompt_tokens=10,
        completion_tokens=4,
        total_tokens=14,
        sglang_spec_details=UsageProcessor._spec_details_from_metas([_meta()]),
    )
    payload = usage.model_dump()
    d = payload["sglang_spec_details"]
    assert set(d) == {"accept_length", "accept_rate", "verify_ct"}
    assert d["verify_ct"] == 25


def test_multi_choice_aggregation():
    metas = [
        _meta(completion=50, verify=10, correct=30, proposed=150),
        _meta(completion=30, verify=10, correct=10, proposed=150),
    ]
    d = UsageProcessor._spec_details_from_metas(metas)
    assert d.verify_ct == 20
    assert abs(d.accept_length - 80 / 20) < 1e-9
    assert abs(d.accept_rate - 40 / 300) < 1e-9


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
