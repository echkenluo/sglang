"""Exercise the actual watchdog functions with CPU fakes and real threads.

AST loading isolates these functions from optional serving/CUDA imports; no
watchdog implementation is copied into this test. This is a teardown ordering
test, not a reproduction of the service's native abort.
"""
import ast
import logging
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import Mock


def load_watchdog():
    root = Path(__file__).resolve().parents[4]
    source = root / "python/sglang/srt/layers/moe/moe_runner/mok_fp8_native.py"
    tree = ast.parse(source.read_text())
    names = {"_trap_watchdog_loop", "_register_trap_watchdog",
             "_die_if_trapped", "shutdown_trap_watchdog"}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id.startswith("_TRAP_WATCHDOG_")
                   for t in targets):
                nodes.append(node)
    ns = {"threading": threading, "Optional": Optional,
          "torch": SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock())),
          "logging": Mock(), "logger": logging.getLogger(__name__),
          "os": SimpleNamespace(_exit=Mock(side_effect=RuntimeError("trap exit")))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
    return ns


class WatchdogShutdownTest(unittest.TestCase):
    def setUp(self):
        self.ns = load_watchdog()

    def tearDown(self):
        self.ns["_TRAP_WATCHDOG_STOP"].set()
        thread = self.ns["_TRAP_WATCHDOG_THREAD"]
        if thread:
            thread.join(3)
            self.assertFalse(thread.is_alive())

    def test_unused_path_never_touches_cuda(self):
        self.ns["shutdown_trap_watchdog"]()
        self.ns["torch"].cuda.synchronize.assert_not_called()

    def test_shutdown_joins_inflight_tensor_reader_before_dropping_entries(self):
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        def read(workspace):
            if threading.current_thread().name == "mok-trap-watchdog":
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test reader timed out")
            return None
        self.ns["_register_trap_watchdog"](object(), SimpleNamespace(format_trap_record=read))
        self.assertTrue(entered.wait(1))
        errors = []
        def close():
            try:
                self.ns["shutdown_trap_watchdog"]()
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()
        closer = threading.Thread(target=close)
        closer.start()
        try:
            self.assertTrue(self.ns["_TRAP_WATCHDOG_STOP"].wait(1))
            self.assertFalse(done.is_set())
            self.assertTrue(self.ns["_TRAP_WATCHDOG_ENTRIES"])
        finally:
            release.set()
            closer.join(2)
        self.assertTrue(done.is_set())
        self.assertEqual(errors, [])
        self.assertEqual(self.ns["_TRAP_WATCHDOG_ENTRIES"], [])
        self.assertFalse(self.ns["_TRAP_WATCHDOG_THREAD"].is_alive())
        self.ns["shutdown_trap_watchdog"]()
        self.ns["torch"].cuda.synchronize.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "shut down"):
            self.ns["_register_trap_watchdog"](object(), SimpleNamespace())

    def test_monitor_remains_enabled_during_gpu_drain(self):
        observed = threading.Event()
        def read(workspace):
            observed.set()
            return None
        self.ns["_register_trap_watchdog"](object(), SimpleNamespace(format_trap_record=read))
        self.assertTrue(observed.wait(1))
        def drain():
            self.assertFalse(self.ns["_TRAP_WATCHDOG_STOP"].is_set())
            self.assertTrue(self.ns["_TRAP_WATCHDOG_THREAD"].is_alive())
            observed.clear()
            self.assertTrue(observed.wait(1))
        self.ns["torch"].cuda.synchronize.side_effect = drain
        self.ns["shutdown_trap_watchdog"]()

    def test_final_trap_check_keeps_existing_fatal_exit(self):
        reader = SimpleNamespace(format_trap_record=lambda workspace: "MOK_TRAP|code=1")
        self.ns["_TRAP_WATCHDOG_STARTED"] = True
        self.ns["_TRAP_WATCHDOG_ENTRIES"].append((object(), reader))
        with self.assertRaisesRegex(RuntimeError, "trap exit"):
            self.ns["shutdown_trap_watchdog"]()
        self.ns["os"]._exit.assert_called_once_with(70)
        self.assertFalse(self.ns["_TRAP_WATCHDOG_STOP"].is_set())


if __name__ == "__main__":
    unittest.main()
