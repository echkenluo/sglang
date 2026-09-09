"""Real CUDA checks for the recorder; not full-model or service evidence."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest

import torch

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/utils/mok_boundary_timing.py"


@unittest.skipUnless(torch.cuda.is_available(), "requires an owned CUDA device")
class BoundaryTimingCudaTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("boundary_cuda_test", SOURCE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        root = os.environ.get("MOK_BOUNDARY_CUDA_TEST_OUTPUT")
        if root:
            self.directory = Path(root) / self._testMethodName
            self.directory.mkdir(parents=True, exist_ok=False)
        else:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            self.directory = Path(temporary.name)
        self.m.DIRECTORY = str(self.directory)
        self.m.MIN_TOKENS = 256
        self.m.MAX_RECORDS = 32
        self.m._recorder = self.m.Recorder(torch.cuda, 32)
        self.owner = types.SimpleNamespace(layer_id=7)
        self.x = torch.ones((256, 256), device="cuda", dtype=torch.float32)

    def test_joined_stream_boundaries_and_real_export(self):
        main = torch.cuda.current_stream()
        alternate = torch.cuda.Stream()

        @self.m.boundary("shared_expert")
        def shared(owner, hidden_states):
            value = None
            for _ in range(16):
                value = hidden_states @ hidden_states
            return value

        @self.m.boundary("routed_moe")
        def routed(owner, hidden_states):
            value = None
            for _ in range(16):
                value = hidden_states @ hidden_states
            return value

        @self.m.boundary("full_moe")
        def full(owner, hidden_states):
            alternate.wait_stream(main)
            with torch.cuda.stream(alternate):
                shared_output = shared(owner, hidden_states)
            routed_output = routed(owner, hidden_states)
            main.wait_stream(alternate)
            return routed_output + shared_output

        # Materialize CUDA events and GEMM paths before the checked call.
        full(self.owner, self.x)
        self.m.flush_if_enabled()
        reference_start = torch.cuda.Event(enable_timing=True)
        reference_end = torch.cuda.Event(enable_timing=True)
        reference_start.record(main)
        result = full(self.owner, self.x)
        reference_end.record(main)
        self.m.flush_if_enabled()
        self.assertTrue(torch.equal(result, torch.full_like(result, 512)))
        files = sorted(self.directory.glob("rank*-flush*.json"))
        self.assertEqual(len(files), 2)
        payload = json.loads(files[1].read_text())
        self.assertEqual(len(payload["rows"]), 3)
        scopes = {row["label"]: row for row in payload["rows"]}
        outer = scopes["full_moe"]
        self.assertGreater(outer["cuda_elapsed_ms"], 0)
        reference_ms = reference_start.elapsed_time(reference_end)
        self.assertLessEqual(outer["cuda_elapsed_ms"], reference_ms + 0.001)
        self.assertEqual(outer["stream"], main.cuda_stream)
        self.assertEqual(scopes["shared_expert"]["stream"], alternate.cuda_stream)
        for label in ("shared_expert", "routed_moe"):
            self.assertEqual(scopes[label]["parent"], outer["sequence"])
            self.assertEqual(scopes[label]["cuda_reference_sequence"], outer["sequence"])
            self.assertGreaterEqual(scopes[label]["cuda_start_ms"], -0.001)
            self.assertLessEqual(scopes[label]["cuda_end_ms"], outer["cuda_end_ms"] + 0.001)
            self.assertAlmostEqual(scopes[label]["cuda_end_ms"] - scopes[label]["cuda_start_ms"],
                                   scopes[label]["cuda_elapsed_ms"], delta=0.002)
            self.assertGreater(scopes[label]["cuda_elapsed_ms"], 0)
            self.assertLessEqual(scopes[label]["cuda_elapsed_ms"], outer["cuda_elapsed_ms"] + 0.001)
        self.assertEqual(payload["schema"], "mok-boundary-timing-v2")
        self.assertEqual(outer["cuda_start_ms"], 0.0)
        self.assertEqual(outer["cuda_reference_sequence"], outer["sequence"])
        self.assertTrue(payload["complete_eager_recording"])
        self.assertFalse(payload["intervals_are_additive"])
        (self.directory / "reference.json").write_text(json.dumps({
            "reference_ms": reference_ms, "observed_full_moe_ms": outer["cuda_elapsed_ms"],
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "verdict": "CUDA_RECORDER_ONLY_NOT_MODEL_COVERAGE_OR_OVERHEAD_GATE",
        }, indent=2) + "\n")

    def test_serial_stream_intervals_do_not_report_overlap(self):
        main = torch.cuda.current_stream()
        alternate = torch.cuda.Stream()

        @self.m.boundary("shared_expert")
        def shared(owner, hidden_states):
            value = None
            for _ in range(16):
                value = hidden_states @ hidden_states
            return value

        @self.m.boundary("routed_moe")
        def routed(owner, hidden_states):
            value = None
            for _ in range(16):
                value = hidden_states @ hidden_states
            return value

        @self.m.boundary("full_moe")
        def full(owner, hidden_states):
            alternate.wait_stream(main)
            with torch.cuda.stream(alternate):
                shared_output = shared(owner, hidden_states)
            # This fixture forces shared completion before routed begins.
            main.wait_stream(alternate)
            routed_output = routed(owner, hidden_states)
            return shared_output + routed_output

        full(self.owner, self.x)
        self.m.flush_if_enabled()
        result = full(self.owner, self.x)
        self.m.flush_if_enabled()
        self.assertTrue(torch.equal(result, torch.full_like(result, 512)))
        payload = json.loads(sorted(self.directory.glob("rank*-flush*.json"))[1].read_text())
        scopes = {row["label"]: row for row in payload["rows"]}
        shared_row, routed_row, outer = (scopes[k] for k in ("shared_expert", "routed_moe", "full_moe"))
        self.assertEqual(outer["cuda_start_ms"], 0.0)
        self.assertGreater(shared_row["cuda_end_ms"], shared_row["cuda_start_ms"])
        self.assertGreater(routed_row["cuda_end_ms"], routed_row["cuda_start_ms"])
        self.assertLessEqual(shared_row["cuda_end_ms"], routed_row["cuda_start_ms"] + 0.001)
        self.assertLessEqual(routed_row["cuda_end_ms"], outer["cuda_end_ms"] + 0.001)
        self.assertTrue(all(r["cuda_reference_sequence"] == outer["sequence"] for r in scopes.values()))
        overlap = max(0.0, min(shared_row["cuda_end_ms"], routed_row["cuda_end_ms"])
                      - max(shared_row["cuda_start_ms"], routed_row["cuda_start_ms"]))
        self.assertLessEqual(overlap, 0.001)
        (self.directory / "reference.json").write_text(json.dumps({
            "forced_serial": True, "observed_interval_overlap_ms": overlap,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "verdict": "CUDA_TIMELINE_ONLY_NOT_MODEL_OVERLAP_OR_OVERHEAD_GATE",
        }, indent=2) + "\n")

    def test_capture_is_counted_and_replay_is_not_claimed(self):
        @self.m.boundary("full_moe")
        def call(owner, hidden_states):
            return hidden_states + hidden_states

        call(self.owner, self.x)
        self.m.flush_if_enabled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = call(self.owner, self.x)
        graph.replay()
        torch.cuda.synchronize()
        self.m.flush_if_enabled()
        self.assertTrue(torch.equal(result, torch.full_like(result, 2)))
        payload = json.loads(sorted(self.directory.glob("rank*-flush*.json"))[1].read_text())
        self.assertEqual(payload["rows"], [])
        self.assertEqual(payload["skipped"], {"cuda_graph_capture": 1})
        self.assertFalse(payload["complete_eager_recording"])
        self.assertFalse(payload["graph_replay_measured"])


if __name__ == "__main__":
    unittest.main()
