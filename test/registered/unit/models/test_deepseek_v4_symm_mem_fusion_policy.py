"""CPU policy tests for the DeepSeek-V4 symmetric-memory CP fusion."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed import parallel_state
import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.distributed.device_communicators.torch_symm_mem import (
    TorchSymmMemCommunicator,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekV4SymmMemFusionPolicy(unittest.TestCase):
    def setUp(self):
        self.comm = SimpleNamespace(disabled=False, world_size=4)
        self.tp_group = SimpleNamespace(
            torch_symm_mem_comm=self.comm,
            world_size=4,
            all_gather_into_tensor=self._all_gather_into_tensor,
        )
        self.forward_batch = SimpleNamespace()
        gate_up = SimpleNamespace(weight=object(), weight_scale_inv=object())
        shared = SimpleNamespace(gate_up_proj=gate_up)
        runner_config = SimpleNamespace(inplace=True)
        experts = SimpleNamespace(moe_runner_config=runner_config)
        self.mlp = SimpleNamespace(
            experts=experts,
            shared_experts=shared,
            shared_experts_is_fp8=True,
            _shared_expert_tp1=False,
            num_fused_shared_experts=0,
            _fuse_shared_experts_inside_sbo=False,
            top_k=8,
            n_shared_experts=1,
        )
        self.hidden_states = torch.empty((4, 6144), dtype=torch.bfloat16)

    def _all_gather_into_tensor(self, output, value):
        output.copy_(value.repeat(self.tp_group.world_size, 1))

    def _get_comm(self):
        with (
            envs.SGLANG_OPT_USE_TORCH_SYMM_MEM_FUSED_KERNEL.override(True),
            patch.object(deepseek_v4, "get_is_capture_mode", return_value=False),
            patch.object(
                deepseek_v4,
                "get_tp_group",
                return_value=self.tp_group,
            ),
            patch.object(
                deepseek_v4,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=4, attn_cp_size=4),
            ),
        ):
            return deepseek_v4._get_cp_fused_symm_mem_comm(
                self.mlp, self.hidden_states, self.forward_batch
            )

    def test_admits_complete_tp4_cp4_contract(self):
        self.assertIs(self._get_comm(), self.comm)

    def test_rejects_missing_fp8_scale(self):
        self.mlp.shared_experts.gate_up_proj.weight_scale_inv = None
        self.assertIsNone(self._get_comm())

    def test_rejects_nonidentical_tp_and_cp_groups(self):
        self.comm.world_size = 8
        self.assertIsNone(self._get_comm())

    def test_rejects_cross_rank_signature_mismatch(self):
        def mismatched_gather(output, value):
            output.copy_(value.repeat(self.tp_group.world_size, 1))
            output[-1, 1] += 1

        self.tp_group.all_gather_into_tensor = mismatched_gather
        self.assertIsNone(self._get_comm())

    def test_scope_resets_after_exception(self):
        comm = object.__new__(TorchSymmMemCommunicator)
        comm.use_cp = False
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with comm.cp_fused_scope():
                self.assertTrue(comm.use_cp)
                raise RuntimeError("test failure")
        self.assertFalse(comm.use_cp)

    def test_scope_rejects_nesting_and_still_resets(self):
        comm = object.__new__(TorchSymmMemCommunicator)
        comm.use_cp = False
        with comm.cp_fused_scope():
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with comm.cp_fused_scope():
                    pass
        self.assertFalse(comm.use_cp)

    def test_fused_only_communicator_is_limited_to_tp_group(self):
        with envs.SGLANG_OPT_USE_TORCH_SYMM_MEM_FUSED_KERNEL.override(True):
            self.assertEqual(
                parallel_state._torch_symm_mem_comm_mode(False, "tp"),
                (True, True),
            )
            for group_name in ("attn_cp", "moe_tp", "dcp", "world"):
                self.assertEqual(
                    parallel_state._torch_symm_mem_comm_mode(False, group_name),
                    (False, False),
                )

    def test_explicit_allreduce_keeps_generic_buffer_policy(self):
        with envs.SGLANG_OPT_USE_TORCH_SYMM_MEM_FUSED_KERNEL.override(True):
            self.assertEqual(
                parallel_state._torch_symm_mem_comm_mode(True, "tp"),
                (True, False),
            )
            self.assertEqual(
                parallel_state._torch_symm_mem_comm_mode(True, "attn_cp"),
                (True, False),
            )


if __name__ == "__main__":
    unittest.main()
