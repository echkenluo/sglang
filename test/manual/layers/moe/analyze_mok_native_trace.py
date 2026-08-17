"""Aggregate per-kernel phase timings from MoK native eager chrome traces.

Reads the ``eager-rank{N}.trace.json`` files exported by
``profile_mok_fp8_native_distributed.py`` (MOK_PROFILE_EAGER_ITERS mode) and
prints a per-rank, per-phase account of one native MoK FP8 layer:
quant / route AllGather / barrier / schedule / input copy / dispatch /
grouped GEMM / activation / combine / epilogue / memset / NCCL / other.

Kernel names that match no rule are listed verbatim so the rule table can be
extended instead of silently mis-binning time.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PHASE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("quant", ("per_token_group_quant", "group_quant")),
    ("route_allgather", ("all_gather_top_experts",)),
    ("barrier", ("barrier_all",)),
    ("schedule", ("scheduler",)),
    ("dispatch", ("dispatch_kernel", "fp8_block_routeddispatch")),
    ("combine", ("combine_kernel",)),
    ("epilogue", ("routed_epilogue", "fwd_epilogue")),
    ("gemm", ("fp8_block_test", "grouped", "contiguous_kernel")),
    ("activation", ("silu",)),
    ("nccl", ("nccl",)),
)


def classify(name: str, category: str) -> str | None:
    if category == "gpu_memcpy":
        return "memcpy"
    if category == "gpu_memset":
        return "memset"
    if category not in ("kernel", "gpu_op"):
        return None
    lowered = name.lower()
    for phase, needles in PHASE_RULES:
        if any(needle in lowered for needle in needles):
            return phase
    return "other"


def load_events(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    events = payload.get("traceEvents", payload)
    rows = []
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        if category not in ("kernel", "gpu_memcpy", "gpu_memset", "gpu_op"):
            continue
        rows.append(event)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--iters", type=int, required=True,
                        help="eager iterations captured inside the trace")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report: dict[str, dict] = {}
    for rank in range(4):
        path = args.trace_dir / f"eager-rank{rank}.trace.json"
        if not path.exists():
            raise SystemExit(f"missing trace: {path}")
        events = load_events(path)
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        unknown: dict[str, float] = defaultdict(float)
        barrier_durs: list[float] = []
        span_start = None
        span_end = None
        for event in events:
            phase = classify(event.get("name", ""), event.get("cat", ""))
            if phase is None:
                continue
            duration = float(event.get("dur", 0.0))
            timestamp = float(event.get("ts", 0.0))
            span_start = timestamp if span_start is None else min(span_start, timestamp)
            span_end = (
                timestamp + duration
                if span_end is None
                else max(span_end, timestamp + duration)
            )
            totals[phase] += duration
            counts[phase] += 1
            if phase == "barrier":
                barrier_durs.append(duration)
            if phase == "other":
                unknown[event.get("name", "?")] += duration

        span = (span_end - span_start) if span_start is not None else 0.0
        kernel_sum = sum(totals.values())
        report[f"rank{rank}"] = {
            "per_iter_us": {
                phase: round(totals[phase] / args.iters, 2)
                for phase in sorted(totals, key=lambda p: -totals[p])
            },
            "per_iter_counts": {
                phase: counts[phase] / args.iters for phase in counts
            },
            "span_per_iter_us": round(span / args.iters, 2),
            "gap_per_iter_us": round((span - kernel_sum) / args.iters, 2),
            "barrier_durs_us": [round(d, 1) for d in barrier_durs],
            "unclassified": {
                name: round(duration / args.iters, 2)
                for name, duration in sorted(
                    unknown.items(), key=lambda item: -item[1]
                )[:12]
            },
        }

    for rank_name, entry in report.items():
        print(f"== {rank_name}  span/iter={entry['span_per_iter_us']}us  "
              f"gap/iter={entry['gap_per_iter_us']}us ==")
        for phase, per_iter in entry["per_iter_us"].items():
            count = entry["per_iter_counts"].get(phase, 0)
            print(f"  {phase:<16} {per_iter:>10.2f} us/iter  x{count:.1f}")
        if entry["unclassified"]:
            print("  -- unclassified kernels (add rules): --")
            for name, per_iter in entry["unclassified"].items():
                print(f"    {per_iter:>10.2f} us/iter  {name[:110]}")

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"WROTE {args.json_out}")


if __name__ == "__main__":
    main()
