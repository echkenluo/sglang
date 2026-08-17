"""Bucket GPU kernel time in a model-level prefill trace (MoK or DeepEP).

Reads one SGLang torch-profiler chrome trace (optionally .gz) and prints a
bucket summary plus the top kernels by total time, so the MoK-vs-DeepEP
model gap can be attributed to MoE kernels, attention, communication, or
launch gaps instead of guessed at from TTFT deltas.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("moe_mok_comm", (
        "dispatch_kernel", "combine_kernel", "barrier_all",
        "all_gather_top_experts", "routed_epilogue", "scheduler",
    )),
    ("moe_mok_gemm", ("fp8_block_test", "fp8_block_routed")),
    ("moe_deepep", ("deep_ep", "internode", "intranode", "notify", "cached_notify")),
    ("moe_deepgemm", ("deep_gemm", "fp8_gemm", "m_grouped", "gemm_swapab")),
    ("moe_shared_or_dense", ("cutlass", "nvjet", "cublas")),
    ("activation_quant", ("silu", "group_quant", "quant_fp8", "act_quant")),
    ("attention", ("fmha", "flash", "mla", "dsv4", "attention", "paged", "bmm")),
    ("nccl", ("nccl",)),
    ("norm_elementwise", (
        "rmsnorm", "layernorm", "elementwise", "vectorized", "reduce_kernel",
        "index", "cat", "fill", "copy_", "unrolled",
    )),
    ("rope_embed", ("rotary", "rope", "embedding")),
)


def bucket_of(name: str, category: str) -> str:
    if category == "gpu_memcpy":
        return "memcpy"
    if category == "gpu_memset":
        return "memset"
    lowered = name.lower()
    for bucket, needles in BUCKETS:
        if any(needle in lowered for needle in needles):
            return bucket
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt") as source:
        payload = json.load(source)
    events = payload.get("traceEvents", payload)

    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    by_name: dict[str, float] = defaultdict(float)
    span_start = None
    span_end = None
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        if category not in ("kernel", "gpu_memcpy", "gpu_memset", "gpu_op"):
            continue
        name = event.get("name", "?")
        duration = float(event.get("dur", 0.0))
        timestamp = float(event.get("ts", 0.0))
        span_start = timestamp if span_start is None else min(span_start, timestamp)
        span_end = (
            timestamp + duration
            if span_end is None
            else max(span_end, timestamp + duration)
        )
        bucket = bucket_of(name, category)
        totals[bucket] += duration
        counts[bucket] += 1
        by_name[name] += duration

    span = (span_end - span_start) if span_start is not None else 0.0
    kernel_sum = sum(totals.values())
    print(f"span_ms={span / 1000:.3f}  kernel_sum_ms={kernel_sum / 1000:.3f}  "
          f"gap_ms={(span - kernel_sum) / 1000:.3f}")
    for bucket in sorted(totals, key=lambda b: -totals[b]):
        print(f"  {bucket:<20} {totals[bucket] / 1000:>10.3f} ms  x{counts[bucket]}")
    print("-- top kernels --")
    for name, duration in sorted(by_name.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"  {duration / 1000:>10.3f} ms  {name[:120]}")


if __name__ == "__main__":
    main()
