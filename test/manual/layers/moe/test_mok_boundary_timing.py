"""CPU lifecycle tests; these do not establish CUDA event timing accuracy."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/utils/mok_boundary_timing.py"
with patch.dict(os.environ, {"SGLANG_MOK_BOUNDARY_TIMING_DIR": ""}):
    spec = importlib.util.spec_from_file_location("boundary_timing_test", SOURCE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)


class FakeCuda:
    capturing = False

    def __init__(self):
        self.syncs = 0
        self.events = 0
        self.recorded = []

    def is_current_stream_capturing(self):
        return self.capturing

    def current_stream(self, device):
        return types.SimpleNamespace(cuda_stream=71)

    def Event(self, *, enable_timing):
        self.events += 1
        owner = self

        class Event:
            def record(self, stream):
                owner.recorded.append(stream.cuda_stream)

            def elapsed_time(self, other):
                if owner.syncs == 0:
                    raise AssertionError("elapsed_time before drain synchronization")
                return 1.25

        return Event()

    def synchronize(self):
        self.syncs += 1


class BoundaryTimingTest(unittest.TestCase):
    def setUp(self):
        self.cuda = FakeCuda()
        self.owner = types.SimpleNamespace(layer_id=7)
        self.hidden = types.SimpleNamespace(shape=(1024, 4096), device=types.SimpleNamespace(index=0))
        self.recorder = m.Recorder(self.cuda, 16)
        self.patches = [patch.object(m, "DIRECTORY", "/diagnostic"),
                        patch.object(m, "_recorder", self.recorder),
                        patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=self.cuda)})]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_disabled_returns_identical_function(self):
        def original(owner, hidden):
            return hidden
        with patch.object(m, "DIRECTORY", ""):
            self.assertIs(m.boundary("full_moe")(original), original)

    def test_nested_boundaries_preserve_output_and_do_not_sync_until_drain(self):
        @m.boundary("shared_expert")
        def inner(owner, hidden):
            return hidden

        @m.boundary("full_moe")
        def outer(owner, hidden):
            return inner(owner, hidden)

        self.assertIs(outer(self.owner, self.hidden), self.hidden)
        self.assertEqual(self.cuda.syncs, 0)
        payload = self.recorder.drain()
        self.assertEqual(self.cuda.syncs, 1)
        self.assertEqual([r["parent"] for r in payload["rows"]], [None, 0])
        self.assertEqual([r["cuda_elapsed_ms"] for r in payload["rows"]], [1.25, 1.25])
        self.assertFalse(payload["intervals_are_additive"])
        self.assertTrue(payload["complete_eager_recording"])
        outer(self.owner, self.hidden)
        self.assertEqual(self.cuda.events, 4, "reuse events after draining warmup")

    def test_none_exception_and_parent_reset_are_recorded(self):
        @m.boundary("mok_native_call")
        def fallback(owner, hidden):
            return None

        @m.boundary("full_moe")
        def failed(owner, hidden):
            raise ValueError("fixture failure")

        self.assertIsNone(fallback(self.owner, self.hidden))
        with self.assertRaisesRegex(ValueError, "fixture failure"):
            failed(self.owner, self.hidden)
        rows = self.recorder.drain()["rows"]
        self.assertEqual([r["outcome"] for r in rows], ["none", "exception"])
        self.assertEqual([r["parent"] for r in rows], [None, None])

    def test_capture_and_capacity_omissions_are_explicit(self):
        @m.boundary("full_moe")
        def call(owner, hidden):
            return hidden

        self.cuda.capturing = True
        self.assertIs(call(self.owner, self.hidden), self.hidden)
        self.cuda.capturing = False
        self.recorder.limit = 1
        call(self.owner, self.hidden)
        call(self.owner, self.hidden)
        payload = self.recorder.drain()
        self.assertEqual(payload["skipped"], {"cuda_graph_capture": 1, "record_capacity": 1})
        self.assertFalse(payload["complete_eager_recording"])
        self.assertEqual(len(payload["rows"]), 1)

    def test_active_scope_cannot_be_drained(self):
        item = self.recorder.begin("full_moe", self.owner, self.hidden)
        with self.assertRaisesRegex(RuntimeError, "active scope"):
            self.recorder.drain()
        self.assertEqual(self.cuda.syncs, 0)
        self.recorder.finish(item, "returned_output")
        self.assertEqual(len(self.recorder.drain()["rows"]), 1)

    def test_keyword_arguments_and_small_inputs_preserved(self):
        @m.boundary("mok_native_call")
        def call(layer, hidden_states, *, option):
            self.assertEqual(option, 3)
            return hidden_states

        self.assertIs(call(layer=self.owner, hidden_states=self.hidden, option=3), self.hidden)
        small = types.SimpleNamespace(shape=(1, 4096))
        self.assertIs(call(layer=self.owner, hidden_states=small, option=3), small)
        self.assertEqual(len(self.recorder.drain()["rows"]), 1)


if __name__ == "__main__":
    unittest.main()
