"""CPU checks of DSpark capture ordering without importing the serving stack."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


class TestScatteredAux(unittest.TestCase):
    def test_tp_rows_are_restored_without_fp8_or_padding(self):
        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/models/deepseek_v4.py"
        )
        tree = ast.parse(source.read_text())
        names = {"_dsv4_tp_all_gather_rows", "_dsv4_dspark_aux_hidden_state"}
        functions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.assertEqual(len(functions), 2)
        context = SimpleNamespace(input_scattered=True)
        group = SimpleNamespace(world_size=8)
        envs = SimpleNamespace(
            SGLANG_DSV4_FP8_AG_SITE=SimpleNamespace(get=lambda: "all"),
            SGLANG_DSV4_TP_SCATTER_FP8_AG=SimpleNamespace(get=lambda: True),
        )
        namespace = {
            "torch": torch, "Optional": __import__("typing").Optional,
            "get_attn_tp_context": lambda: context,
            "get_tp_group": lambda: group, "envs": envs,
            "_DSV4_TP_SCATTER_STATS": {"dspark_aux_ag": 0},
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        capture = namespace["_dsv4_dspark_aux_hidden_state"]
        for real_rows in (1, 7, 8, 9, 513, 4096):
            with self.subTest(real_rows=real_rows):
                full = torch.arange(real_rows * 4 * 8, dtype=torch.float32)
                full = (full.reshape(real_rows, 4, 8) % 31).to(torch.bfloat16)
                reference = full.mean(dim=1)
                padded = torch.cat([full, torch.zeros((-real_rows % 8, 4, 8), dtype=full.dtype)])
                shards = list(padded.chunk(8))
                gathered = torch.cat([shard.mean(dim=1) for shard in shards])
                namespace["_DSV4_TP_SCATTER_REAL_ROWS"] = real_rows
                for rank, shard in enumerate(shards):
                    def gather(output, local, rank=rank):
                        torch.testing.assert_close(local, shards[rank].mean(dim=1), rtol=0, atol=0)
                        output.copy_(gathered)
                    group.all_gather_into_tensor = gather
                    context.input_scattered = True
                    actual = capture(shard)
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, torch.bfloat16)
                context.input_scattered = False
                torch.testing.assert_close(capture(full), reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
