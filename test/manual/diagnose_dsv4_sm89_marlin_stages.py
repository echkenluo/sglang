"""Locate FP4 MoE gate failures without changing the frozen test or its tolerances.

Run on SM89 with --output PATH. Uses the original seed, weight generation,
shapes and routed inputs; records both FP32 and FP64 dense stage oracles.
This is diagnostic evidence, not a replacement acceptance test.
"""

import argparse
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from test_dsv4_sm89_marlin import TestSm89Mxfp4Marlin

from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)


def difference(actual, expected):
    a, b = actual.float(), expected.float()
    delta = (a - b).abs()
    bad = delta > 0.1 + 0.05 * b.abs()
    coords = bad.nonzero()[:16]
    return {
        "finite": bool(torch.isfinite(a).all()),
        "relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-20)),
        "max_abs": float(delta.max()),
        "unequal": int((a != b).sum()),
        "outside_frozen_elementwise_gate": int(bad.sum()),
        "examples": [{"index": c.tolist(), "actual": float(a[tuple(c)]),
                      "expected": float(b[tuple(c)])} for c in coords],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (8, 9)
    module = importlib.import_module(
        'sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe'
    )
    torch.manual_seed(20260909)
    fixture = TestSm89Mxfp4Marlin()
    experts, hidden, intermediate, topk = 8, 4096, 256, 6
    layer = torch.nn.Module()
    w13, s13, ref13 = fixture.weights(experts, 2 * intermediate, hidden)
    w2, s2, ref2 = fixture.weights(experts, hidden, intermediate)
    for name, value in [('w13_weight', w13), ('w2_weight', w2),
                        ('w13_weight_scale_inv', s13), ('w2_weight_scale_inv', s2)]:
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    layer.orig_dtype = torch.bfloat16
    prepare_moe_mxfp4_layer_for_marlin(layer)
    report = {'seed': 20260909, 'device': torch.cuda.get_device_name(),
              'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
              'contract': 'Original BF16 stage boundaries; atol=.1 rtol=.05 L2<=.02 unchanged',
              'cases': []}
    original_gemm = module.moe_wna16_marlin_gemm
    original_activation = module.swiglu_limit_func
    for tokens in (1, 7, 16, 128, 513):
        x = torch.randn(tokens, hidden, device='cuda', dtype=torch.bfloat16) * 4
        logits = torch.randn(tokens, experts, device='cuda')
        weights, ids = torch.topk(torch.softmax(logits, -1), topk, dim=-1)
        weights = (weights / weights.sum(-1, keepdim=True) * 1.5).contiguous()
        ids = ids.int().contiguous()
        captured = []
        activation = []

        def gemm(*a, **kw):
            out = original_gemm(*a, **kw)
            captured.append(out.clone())
            return out

        def act(out, *a, **kw):
            result = original_activation(out, *a, **kw)
            activation.append(out.clone())
            return result

        # Probe the original unfused clamp used by the failing frozen gate.
        with patch.object(module, 'moe_wna16_marlin_gemm', gemm), \
             patch.object(module, 'swiglu_limit_func', act), \
             patch.object(module.envs.SGLANG_DSV4_SM89_MARLIN_CLAMP, 'get', return_value=False):
            output = module.fused_marlin_moe(
                x, layer.w13_weight, layer.w2_weight, layer.w13_weight_scale,
                layer.w2_weight_scale, logits, weights, ids,
                workspace=layer.workspace, num_bits=4, clamp_limit=10.0)
        assert len(captured) == 2 and len(activation) == 1
        actual1 = captured[0].view(tokens, topk, 2 * intermediate)
        actual_act = activation[0].view(tokens, topk, intermediate)
        actual2 = captured[1].view(tokens, topk, hidden)
        case = {'tokens': tokens, 'oracles': {}}
        for dtype, label in [(torch.float32, 'fp32'), (torch.float64, 'fp64')]:
            expected1 = torch.empty_like(actual1)
            expected_act = torch.empty_like(actual_act)
            expected2 = torch.empty_like(actual2)
            isolated2 = torch.empty_like(actual2)
            for expert in range(experts):
                rows, slots = torch.where(ids == expert)
                if rows.numel() == 0:
                    continue
                gu = (x[rows].to(dtype) @ ref13[expert].to(dtype).T).bfloat16()
                gate, up = gu.chunk(2, -1)
                dense_act = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
                route = weights[rows, slots, None].bfloat16()
                expected1[rows, slots] = gu
                expected_act[rows, slots] = dense_act
                expected2[rows, slots] = (
                    (dense_act.to(dtype) @ ref2[expert].to(dtype).T).bfloat16() * route)
                isolated2[rows, slots] = (
                    (actual_act[rows, slots].to(dtype) @ ref2[expert].to(dtype).T).bfloat16() * route)
            isolated_act = torch.nn.functional.silu(actual1[..., :intermediate].clamp(max=10)) \
                * actual1[..., intermediate:].clamp(-10, 10)
            dense_out = expected2.float().sum(1).bfloat16()
            case['oracles'][label] = {
                'gemm1': difference(actual1, expected1),
                'activation_chained': difference(actual_act, expected_act),
                'activation_same_input': difference(actual_act, isolated_act),
                'gemm2_chained': difference(actual2, expected2),
                'gemm2_same_input': difference(actual2, isolated2),
                'sum_same_input': difference(output, actual2.float().sum(1).bfloat16()),
                'end_to_end': difference(output, dense_out),
            }
        report['cases'].append(case)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(case), flush=True)


if __name__ == '__main__':
    main()
