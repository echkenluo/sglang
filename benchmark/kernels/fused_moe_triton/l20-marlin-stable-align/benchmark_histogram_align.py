"""Exact packing gates and matched stable/histogram CUDA Graph component timing.

Run on a CUDA host with the candidate SGLang checkout imported, for example:
  python benchmark_histogram_align.py --output /tmp/histogram-align.json

Only the active padded prefix is specified by this API. Unused capacity is
intentionally excluded from equality checks. These timings cover packing only;
they establish no full-MoE numerical, model-quality, or service-performance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import statistics
import subprocess
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

align_module = importlib.import_module("sglang.kernels.ops.moe.moe_align_stable")
align = align_module.moe_align_block_size_stable

TOPK = 6
GRAPH_CALLS = 16
ABA_BLOCKS = 3
SAMPLES_PER_ARM = 30
MAX_REFERENCE_DRIFT = 0.05
SEED = 20260914


@dataclass(frozen=True)
class Case:
    experts: int
    tokens: int
    block: int
    dtype: str = "int32"
    distribution: str = "random"

    @property
    def label(self):
        return (
            f"E{self.experts}/M{self.tokens}/K{TOPK}/B{self.block}/"
            f"{self.dtype}/{self.distribution}"
        )


# Deliberate coverage rather than a Cartesian product. M42/43 straddle the
# n=256 chunk boundary (252/258 flattened IDs), and E17 tests padded bins.
CORRECTNESS_CASES = [
    Case(8, 0, 8),
    Case(17, 0, 48, "int64"),
    Case(256, 0, 16),
    Case(8, 1, 8, "int64"),
    Case(256, 5, 8),
    Case(256, 6, 8, "int64"),
    Case(256, 20, 16),
    Case(256, 24, 16, "int64"),
    Case(17, 32, 32),
    Case(256, 42, 16),
    Case(256, 43, 16, "int64"),
    Case(17, 42, 48, "int64"),
    Case(17, 43, 48),
    Case(256, 80, 16),
    Case(256, 96, 16, "int64"),
    Case(256, 128, 32),
    Case(8, 513, 64, "int64"),
    Case(256, 513, 48),
    Case(256, 4096, 64, "int64"),
    Case(17, 5, 8, distribution="concentrated"),
    Case(256, 6, 8, "int64", "concentrated"),
    Case(256, 43, 48, distribution="concentrated"),
    Case(256, 513, 16, "int64", "concentrated"),
    Case(256, 4096, 64, distribution="concentrated"),
]

GRAPH_CASES = [
    Case(256, 5, 8),
    Case(256, 6, 8, "int64"),
    Case(256, 42, 16),
    Case(256, 43, 16, "int64"),
    Case(17, 43, 48),
    Case(8, 513, 64, "int64"),
    Case(256, 4096, 16),
    Case(256, 4096, 64),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_oracle(ids, experts, block):
    """Independent Python grouping; copied from test_moe_align_stable.oracle."""
    flat = ids.reshape(-1).cpu().tolist()
    groups = [[] for _ in range(experts)]
    for slot, expert in enumerate(flat):
        assert 0 <= expert < experts, (slot, expert, experts)
        groups[expert].append(slot)
    packed, expert_blocks = [], []
    for expert, slots in enumerate(groups):
        padding = (-len(slots)) % block
        packed.extend(slots + [len(flat)] * padding)
        expert_blocks.extend([expert] * ((len(slots) + padding) // block))
    return (
        torch.tensor(packed, dtype=torch.int32),
        torch.tensor(expert_blocks, dtype=torch.int32),
    )


def routes(case, seed):
    dtype = getattr(torch, case.dtype)
    if case.distribution == "concentrated":
        return torch.full((case.tokens, TOPK), case.experts - 1, dtype=dtype)
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        case.experts, (case.tokens, TOPK), generator=generator, dtype=dtype
    )


def marlin_block(tokens, experts):
    """Match fused_marlin_moe.py's M block-size selection."""
    for block in (8, 16, 32, 48, 64):
        if tokens * TOPK / experts / block < 0.9:
            break
    return block


def check_result(result, expected, label):
    packed, expert_blocks = expected
    actual, actual_experts, count = result
    assert count.dtype == torch.int32 and count.numel() == 1, label
    assert int(count.item()) == packed.numel(), f"{label}: padded count"
    assert torch.equal(actual[: packed.numel()].cpu(), packed), f"{label}: sorted IDs"
    assert torch.equal(
        actual_experts[: expert_blocks.numel()].cpu(), expert_blocks
    ), f"{label}: expert IDs"


def check_pair(reference, candidate, expected, label):
    check_result(reference, expected, f"{label}/A")
    check_result(candidate, expected, f"{label}/B")
    count = expected[0].numel()
    blocks = expected[1].numel()
    assert torch.equal(reference[2], candidate[2]), f"{label}: A/B count"
    assert torch.equal(reference[0][:count], candidate[0][:count]), label
    assert torch.equal(reference[1][:blocks], candidate[1][:blocks]), label


def capture(ids, case, histogram, calls=1):
    # Warm allocations and compilation on a side stream before capture. Retain
    # every captured output so all calls are checked, including scratch history.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            align(ids, case.block, case.experts, use_histogram=histogram)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [
            align(ids, case.block, case.experts, use_histogram=histogram)
            for _ in range(calls)
        ]
    torch.cuda.synchronize()
    return graph, outputs


def save(output, payload):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def record_check(output, payload, case, kind, operation):
    row = {"case": asdict(case), "kind": kind, "status": "running"}
    payload["correctness"].append(row)
    save(output, payload)
    try:
        operation()
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(output, payload)
        print(f"FAIL {kind} {case.label}: {error}", flush=True)
        raise
    row["status"] = "passed"
    save(output, payload)
    print(f"PASS {kind} {case.label}", flush=True)


def eager_check(case, seed):
    host = routes(case, seed)
    ids = host.cuda()
    expected = cpu_oracle(host, case.experts, case.block)
    for repeat in range(3):
        reference = align(ids, case.block, case.experts, use_histogram=False)
        candidate = align(ids, case.block, case.experts, use_histogram=True)
        check_pair(reference, candidate, expected, f"{case.label}/repeat{repeat}")


def graph_check(case, seed):
    original = routes(case, seed)
    ids = original.cuda()
    graph_a, output_a = capture(ids, case, False)
    graph_b, output_b = capture(ids, case, True)
    history = [
        ("original", original),
        ("shifted", (original + 3) % case.experts),
        ("concentrated", torch.full_like(original, case.experts - 1)),
        ("restore", original),
    ]
    for state, host in history:
        ids.copy_(host)
        expected = cpu_oracle(host, case.experts, case.block)
        for repeat in range(3):
            graph_a.replay()
            graph_b.replay()
            torch.cuda.synchronize()
            # The current eager reference sees exactly the same shape/routes.
            reference = align(ids, case.block, case.experts, use_histogram=False)
            label = f"{case.label}/{state}/repeat{repeat}"
            check_pair(reference, output_a[0], expected, label + "/captured-A")
            check_pair(reference, output_b[0], expected, label + "/captured-B")


def event_samples(graph):
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(SAMPLES_PER_ARM):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    torch.cuda.synchronize()
    return samples


def ratio_summary(a_graph_us, b_graph_us):
    if not a_graph_us or not b_graph_us:
        return None
    a = statistics.median(a_graph_us) / GRAPH_CALLS
    b = statistics.median(b_graph_us) / GRAPH_CALLS
    return {
        "A_median_us_per_call": a,
        "B_median_us_per_call": b,
        "A_over_B_speedup": a / b,
        "B_over_A_cost_ratio": b / a,
        "cost_reduction_fraction": 1.0 - b / a,
    }


def benchmark_case(case, seed, output, payload, block_policy):
    host = routes(case, seed)
    ids = host.cuda()
    expected = cpu_oracle(host, case.experts, case.block)
    graph_a, output_a = capture(ids, case, False, GRAPH_CALLS)
    graph_b, output_b = capture(ids, case, True, GRAPH_CALLS)
    graph_a.replay()
    graph_b.replay()
    torch.cuda.synchronize()
    reference = align(ids, case.block, case.experts, use_histogram=False)
    for index, (a, b) in enumerate(zip(output_a, output_b)):
        check_pair(reference, a, expected, f"{case.label}/timing-A{index}")
        check_pair(reference, b, expected, f"{case.label}/timing-B{index}")
    row = {
        "case": asdict(case),
        "block_policy": block_policy,
        "status": "running",
        "aba_blocks": [],
    }
    payload["timing"].append(row)
    save(output, payload)
    for block in range(ABA_BLOCKS):
        samples_a1 = event_samples(graph_a)
        samples_b = event_samples(graph_b)
        samples_a2 = event_samples(graph_a)
        median_a1 = statistics.median(samples_a1)
        median_a2 = statistics.median(samples_a2)
        drift = abs(median_a2 / median_a1 - 1.0)
        valid = drift <= MAX_REFERENCE_DRIFT
        block_row = {
            "block": block,
            "valid": valid,
            "invalid_reason": None if valid else "reference_drift_exceeds_5_percent",
            "reference_drift_fraction": drift,
            "median_us_per_call": {
                "A1": median_a1 / GRAPH_CALLS,
                "B": statistics.median(samples_b) / GRAPH_CALLS,
                "A2": median_a2 / GRAPH_CALLS,
            },
            "samples_graph_us": {"A1": samples_a1, "B": samples_b, "A2": samples_a2},
            "ratio": ratio_summary(samples_a1 + samples_a2, samples_b),
        }
        row["aba_blocks"].append(block_row)
        save(output, payload)
        medians = block_row["median_us_per_call"]
        print(
            f"{'VALID' if valid else 'INVALID'} timing {case.label} ABA{block} "
            f"A1/B/A2={medians['A1']:.4f}/{medians['B']:.4f}/"
            f"{medians['A2']:.4f} us/call drift={drift:.2%} "
            f"A/B={block_row['ratio']['A_over_B_speedup']:.4f}",
            flush=True,
        )
    # Include every raw row in pooled_all; the separate valid-only pool makes
    # any exclusion explicit. Never promote invalid timing into a valid claim.
    for pool_name, selected in (
        ("pooled_all", row["aba_blocks"]),
        ("pooled_valid", [item for item in row["aba_blocks"] if item["valid"]]),
    ):
        pooled_a, pooled_b = [], []
        for item in selected:
            samples = item["samples_graph_us"]
            pooled_a.extend(samples["A1"] + samples["A2"])
            pooled_b.extend(samples["B"])
        row[pool_name] = ratio_summary(pooled_a, pooled_b)
    row["valid_blocks"] = sum(item["valid"] for item in row["aba_blocks"])
    row["valid"] = row["valid_blocks"] == ABA_BLOCKS
    # Recheck all captured outputs after the measurement history.
    for index, (a, b) in enumerate(zip(output_a, output_b)):
        check_pair(a, b, expected, f"{case.label}/after-timing{index}")
    row["status"] = "completed"
    save(output, payload)
    print(
        f"DONE timing {case.label} valid_blocks={row['valid_blocks']}/{ABA_BLOCKS} "
        f"pooled_A/B={row['pooled_all']['A_over_B_speedup']:.4f}",
        flush=True,
    )


def gpu_snapshot(device):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(device),
                "--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return {"returncode": result.returncode, "stdout": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument(
        "--include-block16",
        action="store_true",
        help="Also time fixed block16 where runtime selects another block",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU parsing is not GPU validation")
    torch.cuda.set_device(args.device)
    kernel_path = Path(align_module.__file__).resolve()
    kernel_hash = sha256(kernel_path)
    properties = torch.cuda.get_device_properties(args.device)
    output = args.output.resolve()
    payload = {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "exact stable expert packing and isolated CUDA Graph packing cost",
        "correctness_gate_passed": False,
        "identity": {
            "kernel_file": str(kernel_path),
            "kernel_sha256": kernel_hash,
            "script_sha256": sha256(__file__),
            "torch_version": torch.__version__,
            "torch_file": torch.__file__,
            "triton_version": triton.__version__,
            "triton_file": triton.__file__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "device": args.device,
            "gpu_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "gpu_snapshot_start": gpu_snapshot(args.device),
        },
        "protocol": {
            "A": "moe_align_block_size_stable(use_histogram=False)",
            "B": "moe_align_block_size_stable(use_histogram=True)",
            "topk": TOPK,
            "seed": SEED,
            "calls_per_graph": GRAPH_CALLS,
            "aba_blocks": ABA_BLOCKS,
            "samples_per_arm_per_block": SAMPLES_PER_ARM,
            "event_sample": "one Graph replay; synchronize after each event pair",
            "drift": "abs(median(A2)/median(A1)-1)",
            "max_reference_drift_fraction": MAX_REFERENCE_DRIFT,
            "pooled_reference": "median of all A1 and A2 event samples",
            "unused_tail_compared": False,
            "numeric_tolerance": "exact integer equality, active prefixes only",
            "runtime_block_rule": "first block in 8/16/32/48/64 with M*K/E/block < 0.9; else 64",
            "include_block16": args.include_block16,
        },
        "correctness": [],
        "timing": [],
    }
    save(output, payload)
    try:
        timing_cases = [
            (
                Case(256, tokens, marlin_block(tokens, 256), distribution=distribution),
                "runtime_marlin_selection",
            )
            for distribution in ("random", "concentrated")
            for tokens in (5, 6, 20, 24, 80, 96, 128, 513, 4096)
        ]
        if args.include_block16:
            timing_cases.extend(
                (
                    Case(256, tokens, 16, distribution=distribution),
                    "optional_fixed_block16",
                )
                for distribution in ("random", "concentrated")
                for tokens in (5, 6, 20, 24, 80, 96, 128, 4096)
            )
        # Also gate every exact shape/distribution used by timing before the
        # first timing event. Deduplicate cases rather than expand all axes.
        correctness_cases = list(
            dict.fromkeys(CORRECTNESS_CASES + [case for case, _ in timing_cases])
        )
        for index, case in enumerate(correctness_cases):
            record_check(
                output,
                payload,
                case,
                "eager",
                lambda c=case, i=index: eager_check(c, SEED + i),
            )
        for index, case in enumerate(GRAPH_CASES):
            record_check(
                output,
                payload,
                case,
                "graph-changing-routes-and-restore",
                lambda c=case, i=index: graph_check(c, SEED + 100 + i),
            )
        payload["correctness_gate_passed"] = True
        save(output, payload)
        print("PASS all exact correctness gates; timing may start", flush=True)
        if not args.correctness_only:
            for index, (case, block_policy) in enumerate(timing_cases):
                benchmark_case(case, SEED + 1000 + index, output, payload, block_policy)
        assert (
            sha256(kernel_path) == kernel_hash
        ), "imported kernel source changed during run"
        payload["status"] = "completed"
    except Exception as error:
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc()
        raise
    finally:
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        payload["identity"]["gpu_snapshot_end"] = gpu_snapshot(args.device)
        save(output, payload)
    print(f"Saved {output}; component packing evidence only", flush=True)


if __name__ == "__main__":
    main()
