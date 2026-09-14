"""Same-input screen: current TileLang, F28, and community paged Indexer.

Synthetic packed FP8 pages, FP32 reference, bounded representative shapes.
This is an operator screen, not service or full-model quality evidence.
"""
import argparse
import json
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_fp8_paged_mqa_logits
from sglang.srt.layers.attention.dsv4.community_paged_mqa import (
    fp8_paged_mqa_logits_triton_sm89,
)
from sglang.srt.layers.attention.dsv4.f28_mqa import sglang_paged_mqa_logits


SHAPES = [(m, s) for m in (6, 24, 96) for s in (512, 8192)] + [(512, 8192), (4096, 1024)]


def fixture(m, s):
    generator = torch.Generator(device="cuda").manual_seed(43100 + m + s)
    pages = max(32, s // 64 * 2)
    q = (torch.randn(m, 1, 64, 128, generator=generator, device="cuda") * .25).to(torch.float8_e4m3fn)
    values = (torch.randn(pages, 64, 128, generator=generator, device="cuda") * .25).to(torch.float8_e4m3fn)
    scales = torch.rand(pages, 64, generator=generator, device="cuda") + .5
    packed = torch.empty(pages, 64 * 132, device="cuda", dtype=torch.uint8)
    packed[:, :64 * 128].copy_(values.view(torch.uint8).reshape(pages, -1))
    packed[:, 64 * 128:].copy_(scales.view(torch.uint8).reshape(pages, -1))
    packed = packed.view(pages, 64, 1, 132)
    weights = torch.randn(m, 64, generator=generator, device="cuda") * .1
    table = torch.randint(pages, (m, s // 64), generator=generator, device="cuda", dtype=torch.int32)
    # Include partial pages and empty rows. Invalid pages are never read by valid lengths.
    lengths = torch.full((m,), s, device="cuda", dtype=torch.int32)
    boundary = [0, 1, 63, 64, 65]
    lengths[:min(m, len(boundary))] = torch.tensor(boundary[:m], device="cuda", dtype=torch.int32)
    valid_pages = torch.arange(s // 64, device="cuda")[None, :] * 64 < lengths[:, None]
    table.masked_fill_(~valid_pages, -1)
    return q, packed, weights, lengths, table, values, scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (8, 9)
    torch.backends.cuda.matmul.allow_tf32 = False
    functions = {"tilelang": tilelang_fp8_paged_mqa_logits,
                 "f28": sglang_paged_mqa_logits,
                 "community": fp8_paged_mqa_logits_triton_sm89}
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "triton": triton.__version__, "atol": .001, "rtol": .001,
              "scope": "synthetic operator screen; no service adoption", "cases": []}
    for m, s in SHAPES:
        q, packed, weights, lengths, table, values, scales = fixture(m, s)
        valid = torch.arange(s, device="cuda")[None, :] < lengths[:, None]
        case = {"query_rows": m, "max_compressed_tokens": s, "backends": {}}
        samples = sorted(set(list(range(min(m, 6))) + [m - 1]))
        references = {}
        for row in samples:
            n = int(lengths[row])
            ids = table[row, :triton.cdiv(n, 64)].long()
            k = values[ids].float().reshape(-1, 128)[:n]
            scale = scales[ids].reshape(-1)[:n]
            refs = (torch.relu(k @ q[row, 0].float().T) * weights[row]).sum(-1) * scale
            references[row] = refs
        outputs = {}
        for name, fn in functions.items():
            record = {}
            case["backends"][name] = record
            try:
                call = lambda: fn(q, packed, weights, lengths, table, None, s, clean_logits=False)
                output = call()
                torch.cuda.synchronize()
                record["finite_valid"] = bool(torch.isfinite(output[valid]).all())
                checks = []
                for row, ref in references.items():
                    actual = output[row, :ref.numel()]
                    checks.append({"row": row, "passed": bool(torch.allclose(actual, ref, atol=.001, rtol=.001)),
                                   "max_abs": float((actual - ref).abs().max()) if ref.numel() else 0.0})
                record["reference"] = checks
                record["correct"] = record["finite_valid"] and all(c["passed"] for c in checks)
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        call()
                torch.cuda.current_stream().wait_stream(stream)
                with torch.cuda.graph(graph, stream=stream):
                    captured = call()
                graph.replay()
                torch.cuda.synchronize()
                first = captured.clone()
                graph.replay()
                torch.cuda.synchronize()
                record["graph_repeat_exact_valid"] = bool(torch.equal(first[valid], captured[valid]))
                outputs[name] = output
                if record["correct"] and record["graph_repeat_exact_valid"]:
                    record["graph_ms"] = float(triton.testing.do_bench(graph.replay, warmup=20, rep=100))
                    record["eager_ms"] = float(triton.testing.do_bench(call, warmup=20, rep=100))
            except Exception as exc:
                record["error"] = type(exc).__name__ + ": " + str(exc)[-1600:]
                record["correct"] = False
            print(json.dumps({"shape": [m, s], "backend": name, **record}), flush=True)
        if "tilelang" in outputs and "community" in outputs:
            overlaps = []
            for row in samples:
                n = int(lengths[row])
                k = min(512, n)
                if k:
                    left = torch.topk(outputs["tilelang"][row, :n], k).indices
                    right = torch.topk(outputs["community"][row, :n], k).indices
                    overlaps.append(float(torch.isin(left, right).float().mean()))
            case["topk512_sample_overlap"] = overlaps
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
