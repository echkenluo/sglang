import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.entrypoints.warmup import (
    _prefill_warmup_batches,
    _prefill_warmup_sizes,
    prefill_shapes,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillShapeWarmup(unittest.IsolatedAsyncioTestCase):
    def test_batch_shape_configuration(self):
        self.assertEqual(_prefill_warmup_batches(""), [])
        self.assertEqual(
            _prefill_warmup_batches("3x4096, 2x4096,3x4096"),
            [(3, 4096), (2, 4096)],
        )
        for value in ("2", "2x0", "-1x4096", "2x4096,", "2x3x4", " "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _prefill_warmup_batches(value)

    async def test_batches_finish_after_single_sequence_sweep(self):
        steps = []

        async def generate(req, request):
            rows = (
                req.input_ids
                if isinstance(req.input_ids[0], list)
                else [req.input_ids]
            )
            shape = (len(rows), len(rows[0]))
            self.assertTrue(all(len(row) == shape[1] for row in rows))
            self.assertEqual(req.sampling_params["max_new_tokens"], 1)
            steps.append((shape, "start"))
            yield {}
            steps.append((shape, "complete"))

        with (
            envs.SGLANG_PREFILL_WARMUP_SIZES.override("16"),
            envs.SGLANG_PREFILL_WARMUP_BATCHES.override("2x8,3x8"),
        ):
            await prefill_shapes("null", SimpleNamespace(generate_request=generate))
        self.assertEqual(
            steps,
            [
                ((1, 16), "start"), ((1, 16), "complete"),
                ((2, 8), "start"), ((2, 8), "complete"),
                ((3, 8), "start"), ((3, 8), "complete"),
            ],
        )

    async def test_batch_failure_does_not_start_later_shape(self):
        shapes = []

        async def generate(req, request):
            batched = isinstance(req.input_ids[0], list)
            shapes.append(len(req.input_ids) if batched else 1)
            yield {}
            if batched:
                raise RuntimeError("batch failed after first yield")

        with (
            envs.SGLANG_PREFILL_WARMUP_SIZES.override("16"),
            envs.SGLANG_PREFILL_WARMUP_BATCHES.override("2x8,3x8"),
            self.assertRaisesRegex(RuntimeError, "batch failed"),
        ):
            await prefill_shapes("null", SimpleNamespace(generate_request=generate))
        self.assertEqual(shapes, [1, 2])

    async def test_disaggregated_batches_fail_before_any_request(self):
        manager = SimpleNamespace(generate_request=mock.Mock())
        with (
            envs.SGLANG_PREFILL_WARMUP_BATCHES.override("2x8"),
            self.assertRaisesRegex(ValueError, "standalone"),
        ):
            await prefill_shapes("prefill", manager)
        manager.generate_request.assert_not_called()

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
