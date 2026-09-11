"""CPU tests for diagnostic event ordering and failure preservation."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'python/sglang/srt/utils/mok_fault_progress.py'


class Event:
    def __init__(self):
        self.complete = False
        self.error = None
        self.records = 0
        self.queries = 0

    def record(self, stream):
        self.records += 1

    def query(self):
        self.queries += 1
        if self.error:
            raise self.error
        return self.complete


class Cuda:
    def __init__(self):
        self.events = []
        self.capturing = False

    def Event(self, **kwargs):
        assert kwargs == {'enable_timing': False}
        result = Event()
        self.events.append(result)
        return result

    def current_stream(self, device):
        return SimpleNamespace(cuda_stream=17)

    def is_current_stream_capturing(self):
        return self.capturing

    def synchronize(self, *args):
        raise AssertionError('recorder must not synchronize')


class Hidden:
    shape = (24576, 4096)
    dtype = 'torch.bfloat16'
    device = SimpleNamespace(index=0)

    def stride(self):
        return (4096, 1)

    def data_ptr(self):
        return 1234

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: 24576 * 4096 * 2)


class TestProgress(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {'SGLANG_MOK_FAULT_PROGRESS_DIR': ''}):
            spec = importlib.util.spec_from_file_location('progress_test', SOURCE)
            self.mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.mod)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cuda = Cuda()
        torch = ModuleType('torch')
        dist = ModuleType('torch.distributed')
        dist.is_initialized = lambda: True
        dist.get_rank = lambda: 2
        torch.cuda = self.cuda
        torch.distributed = dist
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'torch': torch, 'torch.distributed': dist}).start()
        self.addCleanup(self.close_recorder)

    def close_recorder(self):
        if self.mod._recorder is not None:
            os.close(self.mod._recorder.fd)

    def arm(self, fn):
        self.mod.DIRECTORY = self.tmp.name
        return self.mod.boundary('attention', minimum_rows=24576, tensor_argument='x')(fn)

    def rows(self):
        path = next(Path(self.tmp.name).glob('*.jsonl'))
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_warmup_input_is_cpu_only_and_exact(self):
        self.mod.DIRECTORY = self.tmp.name
        self.mod.record_warmup_input(3, [2, 4, 6])
        path = next(Path(self.tmp.name).glob('warmup-input-*.json'))
        row = json.loads(path.read_text())
        self.assertEqual(row['input_ids'], [2, 4, 6])
        self.assertEqual(row['sampling_params']['max_new_tokens'], 1)
        self.assertFalse(self.cuda.events)

    def test_disabled_is_identity(self):
        def fn(owner, x):
            return x
        self.assertIs(self.mod.boundary('attention', minimum_rows=24576)(fn), fn)

    def test_small_and_capture_skip(self):
        fn = self.arm(lambda owner, x: x)
        small = Hidden()
        small.shape = (1, 4096)
        self.assertIs(fn(None, small), small)
        self.cuda.capturing = True
        self.assertIsInstance(fn(None, Hidden()), Hidden)
        self.assertIsNone(self.mod._recorder)
        self.assertFalse(self.cuda.events)

    def test_submission_is_not_completion(self):
        fn = self.arm(lambda owner, x: 'ok')
        self.assertEqual(fn(SimpleNamespace(layer_id=3), Hidden()), 'ok')
        self.assertEqual(len(self.cuda.events), 2)
        self.assertNotIn('gpu_event_complete', [r['kind'] for r in self.rows()])
        for event in self.cuda.events:
            event.complete = True
        self.mod.poll_if_enabled()
        done = [r for r in self.rows() if r['kind'] == 'gpu_event_complete']
        self.assertEqual([r['phase'] for r in done], ['before', 'after'])
        self.assertTrue(all(r['layer_id'] == 3 for r in done))
        self.assertEqual(self.mod._recorder.pending, [])

    def test_original_exception_has_no_after_cuda_call(self):
        def fail(owner, x):
            raise ValueError('original failure')
        with self.assertRaisesRegex(ValueError, 'original failure'):
            self.arm(fail)(SimpleNamespace(layer_id=2), Hidden())
        self.assertEqual(len(self.cuda.events), 1)
        self.assertEqual(self.cuda.events[0].queries, 0)
        self.assertEqual(self.rows()[-1]['kind'], 'host_exception')
        self.assertTrue(self.mod._recorder.failed)

    def test_query_failure_stops_future_submissions(self):
        fn = self.arm(lambda owner, x: 'ok')
        fn(SimpleNamespace(layer_id=0), Hidden())
        self.cuda.events[0].error = RuntimeError('CUDA failed')
        with self.assertRaisesRegex(RuntimeError, 'CUDA failed'):
            fn(SimpleNamespace(layer_id=1), Hidden())
        self.assertEqual(len(self.cuda.events), 2)
        self.assertEqual(self.rows()[-1]['kind'], 'query_error')
        queries = self.cuda.events[0].queries
        with self.assertRaisesRegex(RuntimeError, 'already failed'):
            self.mod.poll_if_enabled()
        self.assertEqual(self.cuda.events[0].queries, queries)

    def test_selected_sync_is_only_for_named_layer(self):
        from unittest.mock import Mock
        stream = SimpleNamespace(cuda_stream=17, synchronize=Mock())
        self.cuda.current_stream = lambda device: stream
        self.mod.SYNC_SCOPE = 'attention:3'
        fn = self.arm(lambda owner, x: 'ok')
        fn(SimpleNamespace(layer_id=2), Hidden())
        stream.synchronize.assert_not_called()
        fn(SimpleNamespace(layer_id=3), Hidden())
        stream.synchronize.assert_called_once()
        stream.synchronize.side_effect = RuntimeError('selected GPU failure')
        with self.assertRaisesRegex(RuntimeError, 'selected GPU failure'):
            fn(SimpleNamespace(layer_id=3), Hidden())
        self.assertEqual(self.rows()[-1]['kind'], 'selected_sync_error')
        self.assertTrue(self.mod._recorder.failed)

    def test_replay_rejects_modified_tokens(self):
        self.mod.DIRECTORY = self.tmp.name
        self.mod.record_warmup_input(3, [2, 4, 6])
        original = next(Path(self.tmp.name).glob('warmup-input-*.json'))
        target = Path(self.tmp.name) / 'warmup-input-3.json'
        target.write_bytes(original.read_bytes())
        self.mod.REPLAY_DIRECTORY = self.tmp.name
        self.assertEqual(self.mod.replay_warmup_input(3, [1, 1, 1]), [2, 4, 6])
        row = json.loads(target.read_text());row['input_ids'][0] = 3
        target.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.mod.replay_warmup_input(3, [1, 1, 1])


if __name__ == '__main__':
    unittest.main()
