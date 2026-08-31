import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt import runtime_context
from sglang.srt.managers.io_struct import SetInternalStateReq
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestSchedulerHummingW2Runtime(unittest.TestCase):
    def setUp(self):
        runtime_context.reset_context()

    def tearDown(self):
        runtime_context.reset_context()

    def _new_scheduler(self, *, idle=True):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.is_fully_idle = MagicMock(return_value=idle)
        scheduler.spec_algorithm = MagicMock()
        scheduler.spec_algorithm.is_none.return_value = True
        scheduler.metrics_reporter = MagicMock()
        scheduler.metrics_reporter.spec_total_num_forward_ct = 0
        scheduler.metrics_reporter.spec_total_num_accept_tokens = 0
        return scheduler

    def test_idle_scheduler_publishes_frozen_candidate(self):
        scheduler = self._new_scheduler()
        output = scheduler.set_internal_state(
            SetInternalStateReq(
                server_args={"humming_indexed_w2_runtime_num_sms": 5120}
            )
        )
        self.assertTrue(output.updated)
        self.assertEqual(
            runtime_context.get_flags().moe.humming_indexed_w2_runtime_num_sms,
            5120,
        )

    def test_busy_scheduler_rejects_switch(self):
        scheduler = self._new_scheduler(idle=False)
        output = scheduler.set_internal_state(
            SetInternalStateReq(
                server_args={"humming_indexed_w2_runtime_num_sms": 4096}
            )
        )
        self.assertFalse(output.updated)
        self.assertEqual(
            runtime_context.get_flags().moe.humming_indexed_w2_runtime_num_sms,
            0,
        )

    def test_unfrozen_candidate_is_rejected(self):
        scheduler = self._new_scheduler()
        output = scheduler.set_internal_state(
            SetInternalStateReq(
                server_args={"humming_indexed_w2_runtime_num_sms": 3072}
            )
        )
        self.assertFalse(output.updated)
        self.assertEqual(
            runtime_context.get_flags().moe.humming_indexed_w2_runtime_num_sms,
            0,
        )


if __name__ == "__main__":
    unittest.main()
