"""CPU-only contract tests for opt-in NGRAM token-position tracing."""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "srt"
    / "speculative"
    / "ngram_accept_trace.py"
)
SPEC = importlib.util.spec_from_file_location("ngram_accept_trace_under_test", MODULE_PATH)
TRACE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE_MODULE)

TRACE_KEY = TRACE_MODULE.TRACE_KEY
align_ngram_accept_trace_prefix = TRACE_MODULE.align_ngram_accept_trace_prefix
append_ngram_accept_trace = TRACE_MODULE.append_ngram_accept_trace
trace_invalid_reason = TRACE_MODULE.trace_invalid_reason


@dataclass
class FakeReq:
    rid: str = "trace-test"
    output_ids: list[int] = field(default_factory=list)
    customized_info: object = None


def test_disabled_is_strict_noop():
    req = FakeReq(output_ids=[7], customized_info=None)
    before = dict(req.__dict__)
    assert not append_ngram_accept_trace(req, [8, 9], 1, enabled=False)
    assert req.__dict__ == before


def test_prefill_alignment_and_multi_step_append():
    req = FakeReq()
    req.output_ids.append(10)
    assert align_ngram_accept_trace_prefix(req, enabled=True)
    assert req.customized_info[TRACE_KEY] == [None]

    assert append_ngram_accept_trace(req, [11, 12, 13], 2, enabled=True)
    assert req.customized_info[TRACE_KEY] == [None, 1, 1, 0]

    req.output_ids.extend([11, 12, 13])
    assert append_ngram_accept_trace(req, [14, 15], 1, enabled=True)
    assert req.customized_info[TRACE_KEY] == [None, 1, 1, 0, 1, 0]


def test_grammar_truncation_caps_correct_drafts_to_retained_tokens():
    req = FakeReq(output_ids=[10])
    assert append_ngram_accept_trace(req, [11], 7, enabled=True)
    assert req.customized_info[TRACE_KEY] == [None, 1]


def test_zero_and_negative_correct_counts_are_target_tokens():
    req = FakeReq()
    assert append_ngram_accept_trace(req, [1, 2], -4, enabled=True)
    assert req.customized_info[TRACE_KEY] == [0, 0]


def test_unrelated_customized_info_is_preserved():
    req = FakeReq(output_ids=[1], customized_info={"other": ["x"]})
    assert append_ngram_accept_trace(req, [2], 1, enabled=True)
    assert req.customized_info["other"] == ["x"]
    assert req.customized_info[TRACE_KEY] == [None, 1]


def test_collision_fails_open_without_mutating_existing_info():
    original = {TRACE_KEY: [1], "other": ["x", "y"]}
    req = FakeReq(output_ids=[1, 2], customized_info=original)
    assert not append_ngram_accept_trace(req, [3], 1, enabled=True)
    assert req.customized_info is original
    assert trace_invalid_reason(req)

    # Once invalid, later calls remain no-ops and do not replace user data.
    assert not append_ngram_accept_trace(req, [4], 1, enabled=True)
    assert req.customized_info is original


def test_stop_trim_uses_existing_absolute_slice_contract():
    req = FakeReq(output_ids=[1])
    append_ngram_accept_trace(req, [2, 3, 4], 2, enabled=True)
    req.output_ids.extend([2, 3, 4])

    # output_streamer slices both output_ids and customized_info by the same
    # absolute [send_offset:finished_len] range.
    send_offset, finished_len = 1, 3
    assert req.output_ids[send_offset:finished_len] == [2, 3]
    assert req.customized_info[TRACE_KEY][send_offset:finished_len] == [1, 1]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__)
