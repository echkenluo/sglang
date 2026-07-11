"""Sampling step-segment wall-clock profiler (env ``SGLANG_STEP_PROF``).

Attribution-grade CPU wall timing for per-step hot paths (ngram spec step,
plain decode step, scheduler loop). Disabled by default: call sites hold
``None`` and add zero work. Enable with ``SGLANG_STEP_PROF=1`` (flush every
32 steps) or ``SGLANG_STEP_PROF=<N>`` for a custom flush interval.

All durations are host ``perf_counter`` deltas: a segment that wraps GPU
work measures launch (+ any implicit sync) cost on the caller thread, NOT
kernel time. Pair with the torch profiler (/start_profile) for device truth.

Only rank 0 creates an instance (``maybe_create``), so TP>1 logs once.
"""

from __future__ import annotations

import logging
import os
from statistics import median
from time import perf_counter  # noqa: F401  (re-exported for call sites)

logger = logging.getLogger(__name__)


class StepProf:
    """Per-step segment accumulator with periodic p50/max/share flushing.

    Usage at a call site (``prof`` may be None — always guard)::

        t = perf_counter() if prof else 0.0
        ...work...
        if prof:
            prof.add("seg_name", perf_counter() - t)
        ...
        if prof:
            prof.end_step(perf_counter() - t_step_start)

    ``end_step`` derives an ``other`` segment (step total minus attributed
    time) and flushes an aggregate log line every ``interval`` steps.
    """

    @staticmethod
    def maybe_create(tag: str, rank: int = 0) -> "StepProf | None":
        raw = os.environ.get("SGLANG_STEP_PROF", "")
        if not raw or raw.lower() in ("0", "false") or rank != 0:
            return None
        try:
            interval = int(raw)
        except ValueError:
            interval = 1
        if interval <= 1:
            interval = 32
        return StepProf(tag, interval)

    def __init__(self, tag: str, interval: int = 32):
        self.tag = tag
        self.interval = interval
        self._segs: dict[str, list[float]] = {}
        self._totals: list[float] = []
        self._cur = 0.0

    def add(self, name: str, dt: float) -> None:
        self._segs.setdefault(name, []).append(dt)
        self._cur += dt

    def end_step(self, total: float) -> None:
        other = total - self._cur
        if other > 0:
            self._segs.setdefault("other", []).append(other)
        self._cur = 0.0
        self._totals.append(total)
        if len(self._totals) >= self.interval:
            self._flush()

    def _flush(self) -> None:
        tot = sum(self._totals)
        if tot <= 0:
            self._segs.clear()
            self._totals.clear()
            return
        parts = []
        for name, vals in sorted(self._segs.items(), key=lambda kv: -sum(kv[1])):
            parts.append(
                f"{name} p50={median(vals) * 1e3:.3f} max={max(vals) * 1e3:.3f} "
                f"{100.0 * sum(vals) / tot:.1f}%"
            )
        logger.info(
            "[step-prof:%s] steps=%d step_p50=%.3fms step_max=%.3fms | %s",
            self.tag,
            len(self._totals),
            median(self._totals) * 1e3,
            max(self._totals) * 1e3,
            " | ".join(parts),
        )
        self._segs.clear()
        self._totals.clear()
