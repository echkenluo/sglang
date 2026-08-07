import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.model_executor.forward_batch_info as forward_batch_info
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


class TestFluxEngagementPolicy(unittest.TestCase):
    @staticmethod
    def make_batch(mode: ForwardMode, tokens: int) -> SimpleNamespace:
        return SimpleNamespace(
            forward_mode=mode,
            input_ids=SimpleNamespace(shape=(tokens,)),
            useFlux=False,
        )

    def apply_policy(
        self,
        mode: ForwardMode,
        tokens: int,
        *,
        enabled: int = 1,
        min_tokens: int = 0,
        prefill_only: int = 0,
    ) -> bool:
        batch = self.make_batch(mode, tokens)
        env = {
            "SGLANG_USE_FUSED_OVERLAP": str(enabled),
            "SGLANG_FLUX_MIN_TOKENS": str(min_tokens),
            "SGLANG_FLUX_PREFILL_ONLY": str(prefill_only),
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(
                forward_batch_info,
                "get_tensor_model_parallel_world_size",
                return_value=4,
            ),
        ):
            ForwardBatch.useFluxFunc(batch)
        return batch.useFlux

    def test_default_policy_preserves_divisible_mixed_engagement(self):
        self.assertTrue(self.apply_policy(ForwardMode.MIXED, 8))

    def test_prefill_threshold_engages_large_pure_extend(self):
        self.assertTrue(
            self.apply_policy(
                ForwardMode.EXTEND,
                4096,
                min_tokens=1024,
                prefill_only=1,
            )
        )

    def test_prefill_threshold_rejects_small_extend(self):
        self.assertFalse(
            self.apply_policy(
                ForwardMode.EXTEND,
                8,
                min_tokens=1024,
                prefill_only=1,
            )
        )

    def test_prefill_only_rejects_mixed_and_speculative_modes(self):
        for mode in (ForwardMode.MIXED, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                self.assertFalse(
                    self.apply_policy(
                        mode,
                        4096,
                        min_tokens=1024,
                        prefill_only=1,
                    )
                )

    def test_tp_alignment_and_master_switch_remain_required(self):
        self.assertFalse(self.apply_policy(ForwardMode.EXTEND, 4097))
        self.assertFalse(self.apply_policy(ForwardMode.EXTEND, 4096, enabled=0))

    def test_ineligible_batch_clears_stale_engagement_state(self):
        batch = self.make_batch(ForwardMode.EXTEND, 8)
        batch.useFlux = True
        env = {
            "SGLANG_USE_FUSED_OVERLAP": "1",
            "SGLANG_FLUX_MIN_TOKENS": "1024",
            "SGLANG_FLUX_PREFILL_ONLY": "1",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(
                forward_batch_info,
                "get_tensor_model_parallel_world_size",
                return_value=4,
            ),
        ):
            ForwardBatch.useFluxFunc(batch)
        self.assertFalse(batch.useFlux)


if __name__ == "__main__":
    unittest.main()
