# SPDX-License-Identifier: Apache-2.0
"""Check F28 broadcast algebra, Graph updates and the BF16 entry-RS guard."""

from contextlib import ExitStack, nullcontext
from unittest.mock import patch
import statistics
import unittest

import torch
import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.f28_mhc_broadcast import mhc_pre_broadcast


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "requires SM89",
)
class TestMhcBroadcast(unittest.TestCase):
    def test_numerics_and_graph(self):
        torch.manual_seed(20260910)
        with ExitStack() as stack:
            for name, value in (
                ("is_dsa_prefill_cp_round_robin_split", lambda: False),
                ("use_symmetric_memory", lambda *a, **kw: nullcontext()),
                ("is_allocation_symmetric", lambda: False),
                ("get_tp_group", lambda: None),
            ):
                stack.enter_context(patch.object(mhc, name, value))
            stack.enter_context(envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(False))
            h = 4096
            fn = torch.randn(24, 4 * h, device="cuda") * 0.01
            broadcast_fn = fn.view(24, 4, h).sum(1).contiguous()
            scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
            base = torch.zeros(24, device="cuda")
            norm = torch.ones(h, device="cuda", dtype=torch.bfloat16)
            for rows in (0, 1, 6, 16, 80, 512, 4096):
                x = torch.randn(rows, h, device="cuda", dtype=torch.bfloat16) * 0.1

                def call():
                    return mhc_pre_broadcast(
                        x, broadcast_fn, scale, base, 1e-6, 1e-6, 2.0, 20, norm, 1e-6
                    )

                def baseline():
                    expanded = x[:, None].repeat(1, 4, 1)
                    return expanded, *mhc.mhc_pre(
                        expanded,
                        fn,
                        scale,
                        base,
                        1e-6,
                        1e-6,
                        1e-6,
                        2.0,
                        20,
                        norm_weight=norm,
                        norm_eps=1e-6,
                    )

                def check(out):
                    ref = baseline()
                    torch.testing.assert_close(out[0], ref[0], atol=0, rtol=0)
                    for i in (1, 2):
                        torch.testing.assert_close(
                            out[i], ref[i], atol=0.001, rtol=0.001
                        )
                    torch.testing.assert_close(out[3], ref[3], atol=0.02, rtol=0.02)

                if rows == 0:
                    self.assertEqual(call()[0].shape, (0, 4, h))
                    continue
                check(call())
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        call()
                        baseline()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = call()
                for _ in range(3):
                    x.normal_(0, 0.1)
                    fn.normal_(0, 0.01)
                    broadcast_fn.copy_(fn.view(24, 4, h).sum(1))
                    graph.replay()
                    check(out)

                def measure(f):
                    values = []
                    for _ in range(5):
                        start, end = (
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        )
                        start.record()
                        for _ in range(20):
                            f()
                        end.record()
                        end.synchronize()
                        values.append(start.elapsed_time(end) / 20)
                    return statistics.median(values)

                print(
                    f"BROADCAST rows={rows} baseline_ms={measure(baseline):.6f} candidate_ms={measure(call):.6f} graph_updates=3 passed",
                    flush=True,
                )

    def test_entry_reduce_scatter_cannot_enable_fp8(self):
        import sglang.srt.models.deepseek_v4 as model

        class Group:
            world_size = 8

            def reduce_scatter_tensor(self, out, x):
                out.copy_(x[: out.shape[0]])

        x = torch.ones(512, 4096, device="cuda", dtype=torch.bfloat16)
        with (
            patch.object(model, "get_tp_group", return_value=Group()),
            patch.object(
                model,
                "_dsv4_tp_fp8_a2a_reduce_scatter",
                side_effect=AssertionError("FP8 entry forbidden"),
            ),
            envs.SGLANG_DSV4_TP_SCATTER_FP8_RS.override(True),
            envs.SGLANG_DSV4_TP_SCATTER_PRESERVE_AR.override(False),
        ):
            out = model._dsv4_tp_reduce_scatter_rows(x, allow_fp8=False)
            torch.testing.assert_close(out, x[:64], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
