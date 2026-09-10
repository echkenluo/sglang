"""CPU coverage/failure tests; no numerical quality or GPU execution claim."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/utils/mok_path_audit.py"
with patch.dict(os.environ, {"SGLANG_MOK_PATH_AUDIT_DIR": ""}):
    spec = importlib.util.spec_from_file_location("path_audit_test", SOURCE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)


class PathAuditTest(unittest.TestCase):
    def setUp(self):
        self.recorder = m.Recorder()
        self.policy = {"native": True, "warprole": True, "variant": "c2s4",
                       "min_tokens": 256, "max_tokens": 0, "max_sequence_tokens": 0,
                       "strict": True, "prefill_graph": False, "workspace_cap": 12}
        self.eligibility = "eligible"
        self.hidden = types.SimpleNamespace(shape=(512, 4096), device=types.SimpleNamespace(index=0))
        self.model = types.SimpleNamespace(start_layer=0, end_layer=3)
        self.owner = types.SimpleNamespace(layer_id=0)
        self.layout = {"attn_tp_size": 4, "attn_tp_rank": 0, "attn_dp_size": 1,
                       "attn_cp_size": 1, "moe_ep_size": 4, "backend": "deepep",
                       "tp_attention_scatter": True}
        self.patches = [patch.object(m, "DIRECTORY", "/audit"),
                        patch.object(m, "_recorder", self.recorder),
                        patch.object(m, "_flush_index", 0),
                        patch.object(m, "runtime_policy", side_effect=lambda: dict(self.policy)),
                        patch.object(m, "runtime_layout", side_effect=lambda: dict(self.layout)),
                        patch.object(m, "batch_policy", side_effect=lambda *args: ("extend", self.eligibility))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def invoke(self, case="success"):
        @m.scope("core")
        def core(layer):
            if case == "core_exception":
                raise RuntimeError("device launch failed")
            return self.hidden

        @m.scope("native")
        def native(layer, hidden_states):
            if case == "hidden_fallback" or self.eligibility != "eligible":
                return None
            if case == "native_exception":
                raise RuntimeError("native failed")
            if case != "missing_core":
                core(layer)
            return hidden_states

        @m.scope("outer")
        def outer(layer, hidden_states):
            if self.policy["native"]:
                # Errors swallowed by an outer fallback must still invalidate.
                try:
                    native(layer=layer, hidden_states=hidden_states)
                except RuntimeError:
                    if case != "swallow":
                        raise
            if case == "outer_exception":
                raise RuntimeError("output handback failed")
            return hidden_states

        @m.scope("model")
        def model(owner, input_ids):
            for layer_id in range(owner.end_layer):
                if case == "missing_layer" and layer_id == 1:
                    continue
                self.owner.layer_id = layer_id
                outer(self.owner, self.hidden)
            return input_ids

        inputs = types.SimpleNamespace(shape=(self.hidden.shape[0] * 4,))
        result = model(self.model, inputs)
        self.assertIs(result, inputs)
        return self.hidden

    def test_disabled_returns_original_function(self):
        def function(owner):
            return owner
        with patch.object(m, "DIRECTORY", ""):
            for label in ("model", "outer", "native", "core"):
                self.assertIs(m.scope(label)(function), function)

    def test_complete_model_checks_every_layer_and_retains_no_tensors(self):
        self.assertIs(self.invoke(), self.hidden)
        self.assertIs(self.invoke(), self.hidden)
        payload = self.recorder.snapshot()
        json.dumps(payload)
        self.assertEqual(payload["errors"], {})
        self.assertEqual(payload["models"]["returned"], 2)
        self.assertEqual(len(payload["rows"]), 3)
        for row in payload["rows"]:
            self.assertEqual([row[k] for k in ("entered", "returned", "native_returned", "core_returned")], [2] * 4)
            self.assertEqual(row["tokens"], 1024)

    def test_expected_short_prefill_fallback_is_not_hidden_fallback(self):
        self.eligibility = "short_prefill"
        self.invoke()
        payload = self.recorder.snapshot()
        self.assertEqual(payload["errors"], {})
        for row in payload["rows"]:
            self.assertEqual((row["native_none"], row["core_entered"]), (1, 0))

    def test_deepep_control_requires_no_native_calls(self):
        self.policy["native"] = False
        self.invoke()
        self.assertEqual(self.recorder.snapshot()["errors"], {})
        self.assertTrue(all(r["native_entered"] == 0 for r in self.recorder.snapshot()["rows"]))

    def test_hidden_fallback_and_missing_core_are_rejected(self):
        for case in ("hidden_fallback", "missing_core"):
            self.invoke(case)
        self.assertEqual(self.recorder.errors["outer_path_completion"], 6)

    def test_missing_model_layer_is_rejected_even_if_all_observed_calls_return(self):
        self.invoke("missing_layer")
        self.assertEqual(self.recorder.errors, {"model_layer_coverage": 1})
        self.assertEqual(self.recorder.models["returned"], 1)

    def test_exceptions_propagate_and_contexts_restore(self):
        for case in ("core_exception", "native_exception", "outer_exception"):
            with self.assertRaises(RuntimeError):
                self.invoke(case)
            self.assertIsNone(m._model.get())
            self.assertIsNone(m._outer.get())
            self.assertEqual(self.recorder.active, 0)
        self.assertEqual(self.recorder.models["exception"], 3)

    def test_policy_change_is_not_silently_merged(self):
        self.invoke()
        self.policy["warprole"] = False
        self.invoke()
        self.assertGreater(self.recorder.errors["policy_changed"], 0)

    def test_active_snapshot_is_forbidden(self):
        self.recorder.active = 1
        with self.assertRaises(RuntimeError):
            self.recorder.snapshot()

    def test_tensor_split_uses_local_rows_not_model_rows(self):
        self.assertEqual(m.local_moe_tokens(4096, self.layout), 1024)
        for rank in range(4):
            self.layout['attn_tp_rank'] = rank
            self.assertEqual(m.local_moe_tokens(1025, self.layout), 257 if rank == 0 else 256)
            self.assertEqual(m.local_moe_tokens(1, self.layout), 1 if rank == 0 else 0)
        self.invoke()
        snapshot = self.recorder.snapshot()
        self.assertEqual(snapshot['errors'], {})
        self.assertEqual(snapshot['model_batches'], [{'mode': 'extend', 'model_tokens': 2048,
            'local_tokens': 512, 'eligibility': 'eligible', 'calls': 1}])

    def test_wrong_local_rows_are_still_rejected(self):
        with patch.object(m, 'local_moe_tokens', return_value=513):
            self.invoke()
        self.assertEqual(self.recorder.errors['model_moe_token_mismatch'], 3)

    def test_cp_and_dp_layouts_are_not_silently_accepted(self):
        for key in ('attn_cp_size', 'attn_dp_size'):
            layout = dict(self.layout); layout[key] = 2
            with self.assertRaises(ValueError):
                m.local_moe_tokens(512, layout)

    def test_flush_sync_failure_publishes_nothing_and_success_is_cumulative(self):
        syncs = []
        cuda = types.SimpleNamespace(current_device=lambda: 0,
            synchronize=lambda device: syncs.append(device),
            get_device_properties=lambda device: types.SimpleNamespace(uuid="GPU-fixture"))
        dist = types.SimpleNamespace(is_initialized=lambda: True, get_rank=lambda: 2)
        torch = types.ModuleType("torch"); torch.cuda = cuda; torch.distributed = dist
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, "DIRECTORY", tmp), \
             patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}):
            m.flush_if_enabled()
            self.invoke()
            with patch.object(cuda, "synchronize", side_effect=RuntimeError("async CUDA error")):
                with self.assertRaises(RuntimeError):
                    m.flush_if_enabled()
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 1)
            m.flush_if_enabled()
            files = sorted(Path(tmp).glob("*.json"))
            before, after = [json.loads(p.read_text()) for p in files]
            self.assertEqual(before["models"]["entered"], 0)
            self.assertEqual(after["models"]["returned"], 1)
            self.assertEqual(after["flush_index"], 1)
            self.assertEqual(before["session_id"], after["session_id"])
            self.assertEqual(syncs, [0, 0])
            self.assertTrue(all(p.stat().st_mode & 0o222 == 0 for p in files))
            self.assertFalse(list(Path(tmp).glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
