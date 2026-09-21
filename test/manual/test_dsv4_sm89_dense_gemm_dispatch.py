# SPDX-License-Identifier: Apache-2.0
"""Row-count dispatch of block-FP8 dense linears on L20 (SM89).

Why this matters: DSpark verify runs 6, 24, 48 or 96 rows per step. Marlin is
2x to 13x slower than BF16 cuBLAS between 24 and 128 rows, so those rows must
leave Marlin, while decode must never reach the W8A8 kernel (its activation
quantization would change decode numerics for no measurable gain).
"""

import os
import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import sm89_dense_gemm_dispatch as dispatch
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

SHAPES = tuple(dispatch._L20_POLICY)  # (k, n)
DECODE_ROWS = (1, 2, 4, 6, 8, 12, 16, 24, 48, 96)


class TestPolicy(unittest.TestCase):
    def test_decode_rows_never_use_w8a8(self):
        for thresholds in dispatch._L20_POLICY.values():
            for rows in DECODE_ROWS:
                self.assertNotEqual(
                    dispatch.choose_backend(rows, *thresholds, True, True),
                    dispatch.TRITON,
                )

    def test_mid_rows_leave_marlin(self):
        for thresholds in dispatch._L20_POLICY.values():
            for rows in (24, 48, 96):
                self.assertEqual(
                    dispatch.choose_backend(rows, *thresholds, True, True),
                    dispatch.BF16,
                )

    def test_missing_copies_fall_back(self):
        # Without a BF16 copy the mid rows go to Triton, not back to the slow
        # Marlin range; this is what lets a deployment drop the BF16 copy.
        self.assertEqual(dispatch.choose_backend(96, 16, 128, False, True), dispatch.TRITON)
        self.assertEqual(dispatch.choose_backend(4096, 16, 128, False, True), dispatch.TRITON)
        self.assertEqual(dispatch.choose_backend(4096, 16, 128, True, False), dispatch.BF16)
        self.assertEqual(dispatch.choose_backend(4096, 16, 128, False, False), dispatch.MARLIN)


class TestPolicyOverride(unittest.TestCase):
    def test_override_replaces_single_shapes_only(self):
        with mock.patch.dict(
            os.environ, {"SGLANG_SM89_FP8_LINEAR_POLICY": "4096:1536:0:0; 1024:4096:8:64"}
        ):
            table = dispatch.policy_table()
        self.assertEqual(table[(4096, 1536)], (0, 0))
        self.assertEqual(table[(1024, 4096)], (8, 64))
        self.assertEqual(table[(4096, 512)], dispatch._L20_POLICY[(4096, 512)])

    def test_malformed_override_is_rejected(self):
        with mock.patch.dict(os.environ, {"SGLANG_SM89_FP8_LINEAR_POLICY": "4096:1536:0"}):
            with self.assertRaises(ValueError):
                dispatch.policy_table()

    def test_marlin_free_policy_sends_every_row_count_to_triton(self):
        for rows in (1, 6, 24, 96, 4096):
            self.assertEqual(dispatch.choose_backend(rows, 0, 0, False, True), dispatch.TRITON)


def _make_layer(n, k):
    layer = torch.nn.Module()
    layer.output_size_per_partition, layer.input_size_per_partition = n, k
    layer.orig_dtype = torch.bfloat16
    weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    scales = (torch.rand(n // 128, k // 128, device="cuda") * 0.02 + 0.01).float()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
    layer.input_scale = None
    dense = weight.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return layer, dense


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "requires SM89",
)
class TestDispatchOnGpu(unittest.TestCase):
    def _method(self):
        quant_config = mock.Mock()
        quant_config.weight_block_size = [128, 128]
        quant_config.use_mxfp8 = False
        method = Fp8LinearMethod.__new__(Fp8LinearMethod)
        method.quant_config = quant_config
        method.use_marlin = True
        method.use_mxfp8 = False
        method.block_quant = True
        return method

    def test_every_backend_matches_the_dequantized_reference(self):
        torch.manual_seed(20260921)
        with mock.patch.dict(
            os.environ, {"SGLANG_SM89_FP8_LINEAR_DISPATCH": "bf16,triton"}
        ):
            for k, n in SHAPES:
                layer, dense = _make_layer(n, k)
                method = self._method()
                layer.weight_block_size = [128, 128]
                dispatch.prepare_layer(layer, [128, 128])
                from sglang.srt.layers.quantization.marlin_utils_fp8 import (
                    prepare_fp8_layer_for_marlin,
                )

                prepare_fp8_layer_for_marlin(layer, False)
                thresholds = dispatch._L20_POLICY[(k, n)]
                seen = set()
                for rows in (1, 6, 24, 96, 300, 2048):
                    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
                    backend = dispatch.choose_backend(rows, *thresholds, True, True)
                    seen.add(backend)
                    out = method.apply(layer, x, None)
                    ref = x.float() @ dense.T
                    rel = ((out.float() - ref).norm() / ref.norm()).item()
                    # W8A8 quantizes activations to FP8, so it gets a looser bound.
                    bound = 0.03 if backend == dispatch.TRITON else 0.005
                    print(f"k={k} n={n} rows={rows} backend={backend} rel_l2={rel:.3e}")
                    self.assertEqual(out.shape, (rows, n))
                    self.assertLess(rel, bound, (k, n, rows, backend))
                self.assertIn(dispatch.BF16, seen)
                self.assertIn(dispatch.TRITON, seen)

    def test_marlin_free_layer_uses_loaded_weight_and_no_copy(self):
        # The point of the Marlin-free mode is zero extra weight memory: the
        # Triton kernel must run on the tensor the loader produced.
        torch.manual_seed(20260921)
        env = {
            "SGLANG_SM89_FP8_LINEAR_DISPATCH": "triton",
            "SGLANG_SM89_FP8_LINEAR_POLICY": "4096:1536:0:0",
        }
        with mock.patch.dict(os.environ, env):
            layer, dense = _make_layer(1536, 4096)
            loaded = layer.weight.data_ptr()
            method = self._method()
            self.assertTrue(dispatch.prepare_layer(layer, [128, 128]))
            self.assertEqual(layer.sm89_fp8_weight.data_ptr(), loaded)
            self.assertFalse(hasattr(layer, "sm89_bf16_weight"))
            for rows in (1, 6, 24, 96, 2048):
                x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
                out = method.apply(layer, x, None)
                ref = x.float() @ dense.T
                rel = ((out.float() - ref).norm() / ref.norm()).item()
                print(f"marlin-free rows={rows} rel_l2={rel:.3e}")
                self.assertLess(rel, 0.03, rows)
            # Zero-row calls must not fall through to the absent Marlin weight.
            empty = method.apply(layer, x[:0], None)
            self.assertEqual(tuple(empty.shape), (0, 1536))

    def test_disabled_by_default_adds_no_weights(self):
        with mock.patch.dict(os.environ, {"SGLANG_SM89_FP8_LINEAR_DISPATCH": ""}):
            layer, _ = _make_layer(4096, 1024)
            self.assertFalse(dispatch.prepare_layer(layer, [128, 128]))
            self.assertFalse(hasattr(layer, "sm89_dense_thresholds"))
            self.assertFalse(hasattr(layer, "sm89_bf16_weight"))


if __name__ == "__main__":
    unittest.main()
