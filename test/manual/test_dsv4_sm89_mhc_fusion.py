# SPDX-License-Identifier: Apache-2.0
"""L20 fused mHC boundary including deferred residual and changing Graph inputs."""
from contextlib import ExitStack, nullcontext
from unittest.mock import patch
import unittest

import torch
import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.srt.environ import envs


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9), "requires SM89")
class TestMhcFusion(unittest.TestCase):
    def test_fusion_graph_and_aux_boundary(self):
        torch.manual_seed(20260910)
        with ExitStack() as stack:
            for name, value in (("is_dsa_prefill_cp_round_robin_split", lambda: False), ("use_symmetric_memory", lambda *a, **kw: nullcontext()), ("is_allocation_symmetric", lambda: False), ("get_tp_group", lambda: None)):
                stack.enter_context(patch.object(mhc, name, value))
            stack.enter_context(envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(False))
            h, mult = 4096, 4
            fn = torch.randn(24, h * mult, device="cuda") * .01
            scale = torch.tensor([.5, .25, .25], device="cuda")
            base = torch.zeros(24, device="cuda")
            norm = torch.ones(h, dtype=torch.bfloat16, device="cuda")
            args = (fn, scale, base, 1e-6, 1e-6, 1e-6, 2., 20)
            kwargs = dict(norm_weight=norm, norm_eps=1e-6)
            for rows in (1, 6, 16, 32, 33, 80, 128, 513):
                x = torch.randn(rows, h, device="cuda", dtype=torch.bfloat16) * .1
                residual = torch.randn(rows, mult, h, device="cuda", dtype=torch.bfloat16) * .1
                post = torch.rand(rows, mult, 1, device="cuda")
                comb = torch.rand(rows, mult, mult, device="cuda") * .25
                def call():
                    return mhc.mhc_fused_post_pre(x, residual, post, comb, *args, **kwargs)
                def check(out):
                    completed = mhc.mhc_post(x, residual, post, comb)
                    ref = (completed, *mhc.mhc_pre(completed, *args, **kwargs))
                    # Deferred state must complete before DSpark aux mean.
                    torch.testing.assert_close(out[0], ref[0], atol=0, rtol=0)
                    torch.testing.assert_close(out[0].float().mean(-2), completed.float().mean(-2), atol=0, rtol=0)
                    for i in (1, 2):
                        torch.testing.assert_close(out[i], ref[i], atol=.001, rtol=.001)
                    torch.testing.assert_close(out[3], ref[3], atol=.02, rtol=.02)
                check(call())
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        call()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = call()
                for update in range(3):
                    x.normal_(0, .1)
                    residual.normal_(0, .1)
                    graph.replay()
                    check(out)
                print(f"MHC_FUSION rows={rows} graph_updates=3 passed", flush=True)


if __name__ == "__main__":
    unittest.main()
