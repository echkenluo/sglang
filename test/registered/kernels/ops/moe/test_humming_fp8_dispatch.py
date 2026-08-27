"""SM90 component CI for Humming FP8 DeepEP dispatch and fused SiLU quant."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.dsv4.moe import (
    silu_and_mul_masked_post_quant,
)
from sglang.kernels.ops.moe.ep_moe_kernels import moe_permute_with_scale
from sglang.srt.batch_overlap import single_batch_overlap as sbo
from sglang.srt.layers.moe.moe_runner import humming as humming_runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputFormat
from sglang.srt.layers.quantization import humming_utils
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _silu_reference(gate_up: torch.Tensor, swiglu_limit: float | None) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    if swiglu_limit is not None:
        gate = gate.clamp_max(swiglu_limit)
        up = up.clamp(-swiglu_limit, swiglu_limit)
    gate = gate.float()
    up = up.float()
    return gate * torch.sigmoid(gate) * up


class TestHummingFp8Dispatch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() != 90:
            raise unittest.SkipTest("Humming FP8 dispatch requires SM90")

    def test_fp8_dispatch_preserves_group_scales(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        hidden_states = torch.randn(
            5,
            256,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).to(torch.float8_e4m3fn)
        hidden_states_scale = torch.arange(10, device="cuda", dtype=torch.float32).view(
            5, 2
        )
        topk_ids = torch.tensor(
            [[2, 0], [1, -1], [0, 2], [1, 0], [-1, 2]],
            device="cuda",
            dtype=torch.int32,
        )

        output, output_scale, src2dst, expert_offsets = moe_permute_with_scale(
            inputs=hidden_states,
            input_scale=hidden_states_scale,
            topk_ids=topk_ids,
            num_experts=3,
            is_ep=True,
        )

        for token in range(topk_ids.size(0)):
            for slot in range(topk_ids.size(1)):
                source_index = token * topk_ids.size(1) + slot
                destination_index = int(src2dst[source_index])
                if topk_ids[token, slot] < 0:
                    self.assertLess(destination_index, 0)
                    continue
                self.assertTrue(
                    torch.equal(
                        output[destination_index].view(torch.uint8),
                        hidden_states[token].view(torch.uint8),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        output_scale[destination_index], hidden_states_scale[token]
                    )
                )

        self.assertTrue(
            torch.equal(
                expert_offsets,
                torch.tensor([0, 3, 5, 8], device="cuda", dtype=torch.int32),
            )
        )

        dispatcher = SimpleNamespace(
            quant_config={"existing_key": "preserved"},
            set_quant_config=MagicMock(),
        )
        layer = SimpleNamespace(
            dispatcher=dispatcher,
            humming_metas={
                "w13": SimpleNamespace(
                    a_dtype=humming_runner.dtypes.float8e4m3,
                    input_scale_group_size=128,
                )
            },
        )
        with (
            patch.object(
                humming_utils,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_deepep=lambda: True),
            ),
            patch.object(
                humming_utils,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(deepep_dispatcher_output_dtype="auto")
                ),
            ),
            patch.object(
                humming_utils.envs.SGLANG_DEEPEP_BF16_DISPATCH,
                "get",
                return_value=False,
            ),
        ):
            self.assertTrue(humming_utils.configure_humming_deepep_dispatch(layer))

        self.assertTrue(layer._humming_uses_deepep_fp8_dispatch)
        dispatcher.set_quant_config.assert_called_once_with(
            {"existing_key": "preserved", "dispatcher_output_dtype": "fp8"}
        )
        humming_runner._validate_deepep_dispatch_input(output, output_scale, layer)

    def test_silu_fused_masked_quant(self):
        num_experts = 8
        max_tokens = 17
        intermediate = 768
        num_groups = intermediate // 128

        generator = torch.Generator(device="cuda").manual_seed(1)
        gate_up = (
            torch.randn(
                num_experts,
                max_tokens,
                2 * intermediate,
                generator=generator,
                device="cuda",
            )
            * 3.0
        ).to(torch.bfloat16)
        expert_num_tokens = torch.tensor(
            [17, 0, 7, 10, 17, 1, 8, 5], device="cuda", dtype=torch.int32
        )

        for swiglu_limit in (None, 10.0):
            with self.subTest(swiglu_limit=swiglu_limit):
                runner = humming_runner.HummingRunnerCore(
                    MoeRunnerConfig(
                        num_experts=num_experts,
                        num_local_experts=num_experts,
                        intermediate_size_per_partition=intermediate,
                        top_k=8,
                        activation="silu",
                        is_gated=True,
                        swiglu_limit=swiglu_limit,
                        gate_up_interleaved=False,
                    )
                )
                runner.layer = SimpleNamespace(
                    humming_metas={
                        "w2": SimpleNamespace(
                            a_dtype=humming_runner.dtypes.float8e4m3,
                            input_scale_group_size=128,
                        )
                    }
                )
                activation_output = torch.full(
                    (num_experts * max_tokens, intermediate),
                    float("nan"),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                buffers = {
                    "gate_up_output": gate_up.flatten(0, 1),
                    "activation_output": activation_output,
                }

                def run_poisoned_kernel(*args, **kwargs):
                    args[1].view(torch.uint8).fill_(0x7F)
                    args[2].fill_(float("nan"))
                    return silu_and_mul_masked_post_quant(*args, **kwargs)

                with (
                    patch.object(
                        runner,
                        "apply_activation",
                        side_effect=AssertionError("SiLU fused path was not selected"),
                    ) as fallback_activation,
                    patch(
                        "sglang.kernels.ops.attention.dsv4.moe."
                        "silu_and_mul_masked_post_quant",
                        side_effect=run_poisoned_kernel,
                    ) as fused_silu,
                ):
                    down_input, down_scale = runner._grouped_masked_act_quant(
                        gate_up, expert_num_tokens, buffers
                    )

                fused_silu.assert_called_once()
                fallback_activation.assert_not_called()
                call = fused_silu.call_args
                self.assertIs(call.args[0], gate_up)
                self.assertIs(call.args[4], expert_num_tokens)
                self.assertEqual(call.args[3], 128)
                self.assertEqual(call.kwargs["topk"], 8)
                self.assertEqual(call.kwargs["swiglu_limit"], swiglu_limit)
                self.assertFalse(call.kwargs["scale_ue8m0"])
                self.assertEqual(call.args[1].data_ptr(), down_input.data_ptr())
                self.assertEqual(call.args[2].data_ptr(), down_scale.data_ptr())
                self.assertTrue(bool(activation_output.isnan().all()))

                actual = down_input.view(num_experts, max_tokens, intermediate)
                actual_scale = down_scale.view(num_experts, max_tokens, num_groups)
                reference = _silu_reference(gate_up, swiglu_limit)
                grouped_reference = reference.view(
                    num_experts, max_tokens, num_groups, 128
                )
                expected_scale = (
                    grouped_reference.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
                )

                for expert in range(num_experts):
                    valid = int(expert_num_tokens[expert])
                    if valid:
                        torch.testing.assert_close(
                            actual_scale[expert, :valid],
                            expected_scale[expert, :valid],
                            rtol=5e-3,
                            atol=1e-6,
                        )
                        expanded_scale = actual_scale[expert, :valid].repeat_interleave(
                            128, dim=-1
                        )
                        dequantized = actual[expert, :valid].float() * expanded_scale
                        self.assertTrue(
                            bool(
                                (
                                    (dequantized - reference[expert, :valid]).abs()
                                    <= expanded_scale * 17.0 + 1e-4
                                ).all()
                            )
                        )

                    if valid < max_tokens:
                        self.assertTrue(
                            bool(
                                (actual[expert, valid:].view(torch.uint8) == 0x7F).all()
                            )
                        )
                        self.assertTrue(
                            bool(actual_scale[expert, valid:].isnan().all())
                        )

    def test_silu_fused_masked_quant_can_be_disabled(self):
        num_experts = 8
        max_tokens = 4
        intermediate = 768
        gate_up = torch.empty(
            num_experts,
            max_tokens,
            2 * intermediate,
            device="cuda",
            dtype=torch.bfloat16,
        )
        expert_num_tokens = torch.full(
            (num_experts,), max_tokens, device="cuda", dtype=torch.int32
        )
        activation_output = torch.empty(
            num_experts * max_tokens,
            intermediate,
            device="cuda",
            dtype=torch.bfloat16,
        )
        buffers = {
            "gate_up_output": gate_up.flatten(0, 1),
            "activation_output": activation_output,
        }
        expected_down_input = torch.empty_like(activation_output)
        expected_down_scale = torch.empty(
            num_experts * max_tokens,
            intermediate // 128,
            device="cuda",
            dtype=torch.float32,
        )

        runner = humming_runner.HummingRunnerCore(
            MoeRunnerConfig(
                num_experts=num_experts,
                num_local_experts=num_experts,
                intermediate_size_per_partition=intermediate,
                top_k=8,
                activation="silu",
                is_gated=True,
                swiglu_limit=None,
                gate_up_interleaved=False,
            )
        )
        runner.layer = SimpleNamespace(
            humming_metas={
                "w2": SimpleNamespace(
                    a_dtype=humming_runner.dtypes.float8e4m3,
                    input_scale_group_size=128,
                )
            }
        )

        with (
            patch.object(
                humming_runner.envs.SGLANG_HUMMING_FUSED_MASKED_ACT_QUANT,
                "get",
                return_value=False,
            ),
            patch.object(runner, "apply_activation") as fallback_activation,
            patch.object(
                humming_runner.HummingMethod,
                "may_quant_input",
                return_value=(expected_down_input, expected_down_scale),
            ) as fallback_quant,
            patch(
                "sglang.kernels.ops.attention.dsv4.moe."
                "silu_and_mul_masked_post_quant"
            ) as fused_silu,
        ):
            down_input, down_scale = runner._grouped_masked_act_quant(
                gate_up, expert_num_tokens, buffers
            )

        fallback_activation.assert_called_once_with(
            inputs=buffers["gate_up_output"], outputs=activation_output
        )
        fallback_quant.assert_called_once_with(
            layer=runner.layer,
            inputs=activation_output,
            quanted_input=None,
            sublayer_name="w2",
        )
        fused_silu.assert_not_called()
        self.assertIs(down_input, expected_down_input)
        self.assertIs(down_scale, expected_down_scale)


class TestHummingSboContract(unittest.TestCase):
    def test_grouped_masked_output_signal_metadata(self):
        runner = humming_runner.HummingRunnerCore(
            MoeRunnerConfig(
                num_experts=64,
                num_local_experts=64,
                intermediate_size_per_partition=2048,
                top_k=6,
                activation="silu",
                is_gated=True,
                swiglu_limit=None,
                gate_up_interleaved=False,
            )
        )
        runner.layer = SimpleNamespace(
            humming_metas={"w2": SimpleNamespace(shape_n=4096)}
        )
        configs = {
            "w2_tuning_config": [
                [
                    0,
                    1 << 30,
                    {
                        "block_shape": [8, 128, 128],
                        "use_tma": False,
                        "use_stream_k": True,
                    },
                ]
            ]
        }

        self.assertEqual(
            runner.get_grouped_masked_output_signal_meta(configs, 256),
            (8, 32, True, "legacy"),
        )

        configs["w2_tuning_config"][0][2]["multi_cast_size_a"] = 2
        with self.assertRaisesRegex(RuntimeError, "without multicast"):
            runner.get_grouped_masked_output_signal_meta(configs, 256)

    def test_auto_normal_dispatch_uses_serial_event_contract(self):
        hidden_states = SimpleNamespace(shape=(64, 32, 7168), device="cuda")
        dispatch_output = SimpleNamespace(
            hidden_states=hidden_states,
            format=DispatchOutputFormat.DEEPEP_NORMAL,
        )
        wait_event = object()
        humming_backend = SimpleNamespace(is_humming=lambda: True)

        with (
            patch.object(
                sbo.SboFlags,
                "enable_combine_down_gemm_two_stream_overlap",
                return_value=True,
            ),
            patch.object(
                sbo.SboFlags,
                "enable_combine_shared_two_stream_overlap",
                return_value=True,
            ),
            patch.object(sbo, "get_moe_runner_backend", return_value=humming_backend),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=132),
            ),
            patch.object(torch.cuda, "Event", return_value=wait_event),
            patch.object(torch, "zeros") as zeros,
        ):
            combine, down_gemm, meta = sbo.compute_overlap_args(
                dispatch_output, alt_stream=object()
            )

        zeros.assert_not_called()
        self.assertFalse(combine.overlap)
        self.assertIsNone(combine.signal)
        self.assertIsNone(down_gemm)
        self.assertIs(meta["record_event_after_down"], wait_event)

    def test_ll_dispatch_allocates_for_humming_min_block_m(self):
        hidden_states = SimpleNamespace(shape=(64, 32, 7168), device="cuda")
        dispatch_output = SimpleNamespace(
            hidden_states=hidden_states,
            format=DispatchOutputFormat.DEEPEP_LL,
        )
        wait_event = object()
        signal = object()
        humming_backend = SimpleNamespace(is_humming=lambda: True)

        with (
            patch.object(
                sbo.SboFlags,
                "enable_combine_down_gemm_two_stream_overlap",
                return_value=True,
            ),
            patch.object(
                sbo.SboFlags,
                "enable_combine_shared_two_stream_overlap",
                return_value=True,
            ),
            patch.object(sbo, "get_moe_runner_backend", return_value=humming_backend),
            patch.object(
                sbo.envs.SGLANG_HUMMING_ENABLE_SBO, "get", return_value=True
            ),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=132),
            ),
            patch.object(torch.cuda, "Event", return_value=wait_event),
            patch.object(torch, "zeros", return_value=signal) as zeros,
        ):
            combine, down_gemm, meta = sbo.compute_overlap_args(
                dispatch_output, alt_stream=object()
            )

        zeros.assert_called_once_with(256, dtype=torch.int32, device="cuda")
        self.assertTrue(combine.overlap)
        self.assertIs(combine.signal, signal)
        self.assertIs(down_gemm.signal, signal)
        self.assertIs(down_gemm.start_event, wait_event)
        self.assertNotIn("record_event_after_down", meta)


if __name__ == "__main__":
    unittest.main()
