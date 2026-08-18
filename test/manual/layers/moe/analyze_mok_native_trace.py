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
import statistics
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
    parser.add_argument(
        "--drop-first-iters",
        type=int,
        default=0,
        help=(
            "discard this many leading iterations after segmenting on the "
            "one-per-layer input quant kernel; use this to remove profiler "
            "startup skew from cross-rank barriers"
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report: dict[str, dict] = {}
    for rank in range(4):
        path = args.trace_dir / f"eager-rank{rank}.trace.json"
        if not path.exists():
            raise SystemExit(f"missing trace: {path}")
        events = load_events(path)
        classified = [
            (event, classify(event.get("name", ""), event.get("cat", "")))
            for event in events
        ]
        classified = [(event, phase) for event, phase in classified if phase]
        classified.sort(key=lambda item: float(item[0].get("ts", 0.0)))
        quant_starts = [
            float(event["ts"]) for event, phase in classified if phase == "quant"
        ]
        if len(quant_starts) != args.iters:
            raise SystemExit(
                f"{path}: expected {args.iters} quant iteration markers, "
                f"found {len(quant_starts)}"
            )
        if not 0 <= args.drop_first_iters < args.iters:
            raise SystemExit("drop-first-iters must be in [0,iters)")

        final_end = max(
            float(event["ts"]) + float(event.get("dur", 0.0))
            for event, _phase in classified
        )
        iteration_rows = []
        unknown: dict[str, float] = defaultdict(float)
        barrier_durs: list[float] = []
        for index, start in enumerate(quant_starts):
            end = quant_starts[index + 1] if index + 1 < args.iters else final_end
            selected = [
                (event, phase)
                for event, phase in classified
                if start <= float(event["ts"]) < end
            ]
            totals: dict[str, float] = defaultdict(float)
            counts: dict[str, int] = defaultdict(int)
            for event, phase in selected:
                duration = float(event.get("dur", 0.0))
                totals[phase] += duration
                counts[phase] += 1
                if index >= args.drop_first_iters and phase == "barrier":
                    barrier_durs.append(duration)
                if index >= args.drop_first_iters and phase == "other":
                    unknown[event.get("name", "?")] += duration
            selected_start = min(float(event["ts"]) for event, _phase in selected)
            selected_end = max(
                float(event["ts"]) + float(event.get("dur", 0.0))
                for event, _phase in selected
            )
            kernel_sum = sum(totals.values())
            iteration_rows.append(
                {
                    "index": index,
                    "totals": dict(totals),
                    "counts": dict(counts),
                    "span_us": selected_end - selected_start,
                    "gap_us": selected_end - selected_start - kernel_sum,
                }
            )

        kept = iteration_rows[args.drop_first_iters :]
        phases = sorted(
            {phase for row in kept for phase in row["totals"]},
            key=lambda phase: -statistics.median(
                row["totals"].get(phase, 0.0) for row in kept
            ),
        )
        phase_medians = {
            phase: statistics.median(
                row["totals"].get(phase, 0.0) for row in kept
            )
            for phase in phases
        }
        count_medians = {
            phase: statistics.median(
                row["counts"].get(phase, 0) for row in kept
            )
            for phase in phases
        }
        report[f"rank{rank}"] = {
            "per_iter_us": {
                phase: round(phase_medians[phase], 2) for phase in phases
            },
            "per_iter_counts": count_medians,
            "span_per_iter_us": round(
                statistics.median(row["span_us"] for row in kept), 2
            ),
            "gap_per_iter_us": round(
                statistics.median(row["gap_us"] for row in kept), 2
            ),
            "barrier_durs_us": [round(d, 1) for d in barrier_durs],
            "drop_first_iters": args.drop_first_iters,
            "kept_iterations": len(kept),
            "iteration_rows": [
                {
                    "index": row["index"],
                    "span_us": round(row["span_us"], 2),
                    "gap_us": round(row["gap_us"], 2),
                    "phase_us": {
                        phase: round(value, 2)
                        for phase, value in sorted(row["totals"].items())
                    },
                }
                for row in kept
            ],
            "unclassified": {
                name: round(duration / len(kept), 2)
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
