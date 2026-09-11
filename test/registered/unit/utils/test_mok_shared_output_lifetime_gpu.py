"""Isolate the real forward_deepep handback under delayed consumer execution.

No model weights are required. The routed/shared arithmetic is replaced with
small deterministic tensors; the production method, stream joins, output
ownership, and return path are executed from the supplied source file.
The unsafe baseline is observed through allocator state without reusing its
freed block, so the test does not intentionally execute an illegal access.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def forward_from(path):
    source = path.read_bytes()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2MoE")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_deepep")
    fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(torch=torch, is_in_breakable_cuda_graph=lambda: False,
                     SboFlags=SimpleNamespace(enable_dispatch_shared_one_stream_overlap=lambda: False,
                                              enable_combine_shared_two_stream_overlap=lambda: False),
                     envs=SimpleNamespace(SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO=SimpleNamespace(get=lambda: False)),
                     _use_aiter=False)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["forward_deepep"], hashlib.sha256(source).hexdigest()


class Experts:
    should_fuse_routed_scaling_factor_in_topk = True

    def __call__(self, *, hidden_states, topk_output):
        return torch.zeros_like(hidden_states)


def run(path):
    fn, source_sha = forward_from(path)
    side = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    layer = SimpleNamespace(_fuse_shared_experts_inside_sbo=False, is_nextn=True,
                            num_fused_shared_experts=0, alt_stream=side,
                            gate=lambda *args, **kwargs: None,
                            topk=lambda *args, **kwargs: None,
                            _forward_shared_experts=lambda x: x + 3,
                            experts=Experts(), routed_scaling_factor=1.0)
    hidden = torch.ones((1024, 64), device="cuda")
    out = fn(layer, hidden, SimpleNamespace(num_token_non_padded=None))
    torch.cuda.synchronize()
    pointer = out.data_ptr()
    # Keep main-stream reads pending while the CPU drops the returned tensor.
    torch.cuda._sleep(500_000_000)
    observed = out.clone()
    done = torch.cuda.Event();done.record(main)
    del out
    blocks = [b for segment in torch.cuda.memory_snapshot() for b in segment["blocks"]
              if b["address"] <= pointer < b["address"] + b["size"]]
    assert len(blocks) == 1
    pending = not done.query()
    state = blocks[0]["state"]
    done.synchronize()
    assert pending, "consumer finished before allocator observation; inconclusive"
    assert torch.equal(observed, torch.full_like(observed, 4))
    del observed, hidden
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {"source_sha256": source_sha, "allocation_state_while_consumer_pending": state,
            "consumer_pending_at_observation": pending, "output_exact": True}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path);p.add_argument("fixed", type=Path)
    args = p.parse_args()
    result = {"torch_version": torch.__version__, "baseline": run(args.baseline), "fixed": run(args.fixed)}
    print(json.dumps(result, indent=2), flush=True)
    assert result["baseline"]["allocation_state_while_consumer_pending"] == "inactive"
    # Snapshot exporters use both names for blocks whose recorded consumer
    # events are still pending. Preserve the actual spelling in the receipt.
    assert result["fixed"]["allocation_state_while_consumer_pending"] in {
        "active_awaiting_free", "active_pending_free"
    }
