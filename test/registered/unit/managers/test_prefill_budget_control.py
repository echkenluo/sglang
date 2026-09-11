"""Exercise the actual scheduler control methods without importing CUDA extensions.

Only dependency boundaries are substituted. Distributed execution and actual
PrefillAdder batching are checked by the subsequent serving experiment.
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest


class Ready:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def scheduler_methods(context, peer_ready):
    source = Path(__file__).resolve().parents[4] / "python/sglang/srt/managers/scheduler.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ("_prefill_budget_update_error", "set_internal_state")]
    assert len(methods) == 2
    calls = []

    def all_reduce(ready, *, op, group):
        calls.append((op, group))
        ready.value = min(ready.value, peer_ready)

    namespace = {
        "SetInternalStateReq": SimpleNamespace,
        "SetInternalStateReqOutput": SimpleNamespace,
        "DisaggregationMode": SimpleNamespace(NULL="null"),
        "get_context": lambda: context,
        "logging": logging,
        "logger": logging.getLogger(__name__),
        "torch": SimpleNamespace(
            int32="int32", tensor=lambda values, **kwargs: Ready(values[0]),
            distributed=SimpleNamespace(all_reduce=all_reduce, ReduceOp=SimpleNamespace(MIN="MIN"))),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    cls = type("ControlMethods", (), {node.name: namespace[node.name] for node in methods})
    return cls, calls


class TestPrefillBudgetControl(unittest.TestCase):
    def make(self, *, idle=True, peer_ready=1):
        values = {"max_prefill_tokens": 32768}
        updates = []

        def override(*, source, **kwargs):
            updates.append((source, kwargs))
            values.update(kwargs)

        cls, votes = scheduler_methods(SimpleNamespace(override=override), peer_ready)
        obj = cls()
        obj.max_prefill_tokens = obj._startup_max_prefill_tokens = 32768
        obj.enable_overlap = False
        obj.ps = SimpleNamespace(pp_size=1, dp_size=1, attn_cp_size=1)
        obj.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        obj.disaggregation_mode = "null"
        obj.is_fully_idle = lambda: idle
        obj.tp_group = SimpleNamespace(cpu_group="tp-cpu")
        obj.metrics_reporter = SimpleNamespace(spec_total_num_accept_tokens=0, spec_total_num_forward_ct=0)
        return obj, values, updates, votes

    def update(self, obj, **values):
        return obj.set_internal_state(SimpleNamespace(server_args=values)).updated

    def test_restore_original_capacity_after_reduction(self):
        obj, config, updates, votes = self.make()
        for budget in (16384, 8192, 32768):
            self.assertTrue(self.update(obj, max_prefill_tokens=budget))
            self.assertEqual(obj.max_prefill_tokens, budget)
            self.assertEqual(config["max_prefill_tokens"], budget)
            self.assertEqual(obj._startup_max_prefill_tokens, 32768)
        self.assertEqual(votes, [("MIN", "tp-cpu")] * 3)
        self.assertEqual(len(updates), 3)

    def test_invalid_capacity_and_types_do_not_mutate(self):
        for budget in (0, -1, 32769, True, 16384.0, "16384", None):
            with self.subTest(budget=budget):
                obj, config, updates, votes = self.make()
                self.assertFalse(self.update(obj, max_prefill_tokens=budget))
                self.assertEqual((obj.max_prefill_tokens, config["max_prefill_tokens"]), (32768, 32768))
                self.assertEqual(updates, [])
                self.assertEqual(len(votes), 1)

    def test_local_or_remote_busy_rejects_every_change(self):
        for idle, peer in ((False, 1), (True, 0)):
            obj, config, updates, votes = self.make(idle=idle, peer_ready=peer)
            self.assertFalse(self.update(obj, max_prefill_tokens=16384))
            self.assertEqual((obj.max_prefill_tokens, config["max_prefill_tokens"]), (32768, 32768))
            self.assertEqual(updates, [])
            self.assertEqual(len(votes), 1)

    def test_mixed_invalid_request_is_atomic(self):
        obj, config, updates, votes = self.make()
        self.assertFalse(self.update(obj, max_prefill_tokens=16384, unsupported=True))
        self.assertEqual((obj.max_prefill_tokens, config["max_prefill_tokens"]), (32768, 32768))
        self.assertEqual(updates, [])
        self.assertEqual(len(votes), 1)

    def test_existing_control_does_not_require_budget_vote(self):
        obj, config, updates, votes = self.make(idle=False, peer_ready=0)
        self.assertTrue(self.update(obj, speculative_accept_threshold_single=0.5))
        self.assertEqual(config["speculative_accept_threshold_single"], 0.5)
        self.assertEqual(obj.max_prefill_tokens, 32768)
        self.assertEqual(len(updates), 1)
        self.assertEqual(votes, [])

    def test_unsupported_execution_modes_reject(self):
        for field in ("overlap", "pp", "dp", "cp", "speculation", "disaggregation"):
            with self.subTest(field=field):
                obj, config, updates, _ = self.make()
                if field == "overlap": obj.enable_overlap = True
                elif field in ("pp", "dp", "cp"):
                    setattr(obj.ps, {"pp": "pp_size", "dp": "dp_size", "cp": "attn_cp_size"}[field], 2)
                elif field == "speculation": obj.spec_algorithm.is_none = lambda: False
                else: obj.disaggregation_mode = "prefill"
                self.assertFalse(self.update(obj, max_prefill_tokens=16384))
                self.assertEqual(config["max_prefill_tokens"], 32768)
                self.assertEqual(updates, [])


if __name__ == "__main__":
    unittest.main()
