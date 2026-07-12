"""One-shot in-process boot-mode probe (env ``SGLANG_BOOT_MODE_PROBE=1``).

Temporary diagnostic for the per-boot CUDA API service-latency lottery:
right after all CUDA graphs are captured, time four call classes inside the
serving process and log a single classification line:

  [boot-mode-probe] enqueue=..us pageable=..us event=..us graph_replay=..us
      verdict=FAST|MILD|DEEP launch_coin=FAST|SLOW

Call classes (matrix-derived, 2026-07-12 nested-matrix verdict):
  - enqueue   : async kernel-launch class — untaxed in MILD boots, inflated
                only in DEEP boots (control).
  - pageable  : ``torch.tensor(list).to(cuda, non_blocking=True)`` — the same
                op shape as ForwardBatch.init_new's taxed calls (allocator +
                pageable staging).
  - event     : Event record+query — driver round-trip class.
  - graph_replay: synthetic ~1024-node CUDA graph replay — independent
                readout for the binary launch coin.

The probe MUST run in the server process: the lottery is drawn per process,
so an external prober samples its own ticket, not the server's.

Record-and-calibrate only — thresholds below are initial values; no gating
behavior is attached to the verdict.
"""

from __future__ import annotations

import logging
import os
from time import perf_counter

logger = logging.getLogger(__name__)

# Initial thresholds (us/op) pending calibration against step_p50 ground
# truth from the first 10-boot batch. Do NOT gate on these yet.
_PAGEABLE_MILD_US = 40.0
_EVENT_MILD_US = 25.0
_ENQUEUE_DEEP_US = 3.0
_REPLAY_BASE_US = 60.0
_REPLAY_SLOW_FACTOR = 1.5


def maybe_run_boot_mode_probe(rank: int) -> None:
    """Run the probe once on rank 0 when SGLANG_BOOT_MODE_PROBE is set.

    Diagnostic only: any failure is logged and swallowed so startup is never
    affected.
    """
    if rank != 0:
        return
    raw = os.environ.get("SGLANG_BOOT_MODE_PROBE", "")
    if not raw or raw.lower() in ("0", "false"):
        return
    try:
        _run_probe()
    except Exception as e:
        logger.info("[boot-mode-probe] failed: %r", e)


def _run_probe() -> None:
    import torch

    def timed(fn, n, warm=20):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (perf_counter() - t0) / n * 1e6  # us/op

    x = torch.ones(1024, device="cuda")
    enqueue_us = timed(lambda: x.add_(1), 1000)

    src = [0] * 64
    pageable_us = timed(
        lambda: torch.tensor(src, dtype=torch.int32).to("cuda", non_blocking=True),
        300,
    )

    ev = torch.cuda.Event()
    event_us = timed(lambda: (ev.record(), ev.query()), 500)

    # Synthetic ~1024-node graph for the launch coin. Captured on the side
    # stream torch.cuda.graph manages; freed with this frame.
    g = torch.cuda.CUDAGraph()
    y = torch.zeros(256, device="cuda")
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(1024):
            y.add_(1)
    replay_us = timed(g.replay, 50)

    verdict = (
        "DEEP"
        if enqueue_us > _ENQUEUE_DEEP_US
        else (
            "MILD"
            if pageable_us > _PAGEABLE_MILD_US or event_us > _EVENT_MILD_US
            else "FAST"
        )
    )
    coin = "SLOW" if replay_us > _REPLAY_SLOW_FACTOR * _REPLAY_BASE_US else "FAST"
    logger.info(
        "[boot-mode-probe] enqueue=%.1fus pageable=%.1fus event=%.1fus "
        "graph_replay=%.1fus verdict=%s launch_coin=%s",
        enqueue_us,
        pageable_us,
        event_us,
        replay_us,
        verdict,
        coin,
    )
