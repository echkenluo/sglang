import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.entrypoints.warmup import _prefill_warmup_sizes, prefill_shapes
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillShapeWarmup(unittest.IsolatedAsyncioTestCase):
    def test_explicit_sizes_and_default(self):
        self.assertEqual(_prefill_warmup_sizes("8192, 1024,4096,1024"), [1024, 4096, 8192])
        default = _prefill_warmup_sizes("")
        self.assertEqual((default[0], default[-1], len(default)), (64, 32768, 18))
        for value in ("0", "1024,-1", "1024,", "1.5", " "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _prefill_warmup_sizes(value)

    async def test_finishes_each_request_before_next_shape(self):
        steps = []
        async def generate(req, request):
            size = len(req.input_ids)
            steps.append((size, "start"))
            yield {}
            steps.append((size, "complete"))
        with envs.SGLANG_PREFILL_WARMUP_SIZES.override("2048,1024"), mock.patch(
            "sglang.srt.entrypoints.warmup.tqdm.tqdm", side_effect=lambda sizes, **kwargs: sizes
        ):
            await prefill_shapes("null", SimpleNamespace(generate_request=generate))
        self.assertEqual(steps, [(1024, "start"), (1024, "complete"), (2048, "start"), (2048, "complete")])

    async def test_late_failure_prevents_next_shape(self):
        started = []
        async def generate(req, request):
            started.append(len(req.input_ids))
            yield {}
            raise RuntimeError("warmup failed after initial yield")
        with envs.SGLANG_PREFILL_WARMUP_SIZES.override("1024,2048"), mock.patch(
            "sglang.srt.entrypoints.warmup.tqdm.tqdm", side_effect=lambda sizes, **kwargs: sizes
        ), self.assertRaisesRegex(RuntimeError, "after initial yield"):
            await prefill_shapes("null", SimpleNamespace(generate_request=generate))
        self.assertEqual(started, [1024])


if __name__ == "__main__":
    unittest.main()
