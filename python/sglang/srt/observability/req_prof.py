"""Per-request TTFT segment profiler (env ``SGLANG_REQUEST_PROF=1``).

Two one-line bills per request, joined offline by rid:

  [req-prof]        (TokenizerManager process, printed at the first-token
                     write point) — same-clock segments:
      http_parse    ASGI arrival -> handler entry (starlette body read +
                    pydantic validation; needs the env-gated middleware)
      template      apply_chat_template render time (serving_chat marks)
      encode        tokenizer.encode time (serving_chat marks)
      convert_other handler entry -> adapted request ready, minus the two
                    marks (validation, sampling params, python glue)
      tm_ingest     adapted request ready -> TokenizerManager tokenize done
                    (async handoff + TM-side passthrough)
      dispatch      tokenize done -> ZMQ send to scheduler
      wait_first    ZMQ send -> first token back in TokenizerManager
                    (merged cross-process envelope: IPC + scheduler admit +
                    prefill + first decode + detokenize + IPC back)
      total         arrival (or handler entry) -> first token

  [req-prof-sched]  (Scheduler process, printed at prefill-finish) — its own
                    clock, sub-splits wait_first:
      recv_to_wq    Req created -> wait queue entry
      wq_to_fwd     wait queue -> first forward entry (admission)
      prefill_fwd   forward entry -> prefill finished (incl. chunked)

Zero default overhead: every call site is gated on req_prof_enabled();
serving-layer marks ride a ContextVar so nothing is threaded through
signatures. All failures are swallowed into a log line — never a request
error. Streaming chat-completions is the covered path (codex's); other
template branches simply lack the template/encode split.
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger(__name__)

_ENABLED: Optional[bool] = None
_MARKS: ContextVar[Optional[dict]] = ContextVar("req_prof_marks", default=None)


def req_prof_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        raw = os.environ.get("SGLANG_REQUEST_PROF", "")
        _ENABLED = bool(raw) and raw.lower() not in ("0", "false")
    return _ENABLED


def start_marks() -> None:
    _MARKS.set({})


def add_mark(name: str, dt: float) -> None:
    d = _MARKS.get()
    if d is not None:
        d[name] = d.get(name, 0.0) + dt


def pop_marks() -> dict:
    d = _MARKS.get()
    _MARKS.set(None)
    return d or {}


def emit_tm_line(rid, req_prof: dict, ts) -> None:
    """ts: APIServerReqTimeStats (same process/clock as req_prof floats)."""
    try:
        t_asgi = req_prof.get("t_asgi")
        t_handler = req_prof["t_handler"]
        t_conv = req_prof["t_convert_done"]
        template_ms = req_prof.get("template", 0.0) * 1e3
        encode_ms = req_prof.get("encode", 0.0) * 1e3
        base = t_asgi if t_asgi else t_handler
        http_parse = (t_handler - t_asgi) * 1e3 if t_asgi else -1.0
        convert_other = (t_conv - t_handler) * 1e3 - template_ms - encode_ms
        tm_ingest = (ts.tokenize_finish_time - t_conv) * 1e3
        dispatch = (ts.api_server_dispatch_time - ts.tokenize_finish_time) * 1e3
        wait_first = (ts.first_token_time - ts.api_server_dispatch_time) * 1e3
        total = (ts.first_token_time - base) * 1e3
        logger.info(
            "[req-prof] rid=%s total=%.1f http_parse=%.1f template=%.1f "
            "encode=%.1f convert_other=%.1f tm_ingest=%.1f dispatch=%.1f "
            "wait_first=%.1f",
            rid,
            total,
            http_parse,
            template_ms,
            encode_ms,
            convert_other,
            tm_ingest,
            dispatch,
            wait_first,
        )
    except Exception as e:
        logger.info("[req-prof] emit failed rid=%s: %r", rid, e)


def emit_sched_line(req) -> None:
    """req: scheduler Req with SchedulerReqTimeStats (scheduler clock)."""
    try:
        ts = req.time_stats
        recv = ts.scheduler_recv_time
        wq = ts.wait_queue_entry_time
        fe = ts.forward_entry_time
        pf = ts.prefill_finished_time
        if not (recv and wq and fe and pf):
            return
        logger.info(
            "[req-prof-sched] rid=%s recv_to_wq=%.1f wq_to_fwd=%.1f "
            "prefill_fwd=%.1f sched_total=%.1f",
            req.rid,
            (wq - recv) * 1e3,
            (fe - wq) * 1e3,
            (pf - fe) * 1e3,
            (pf - recv) * 1e3,
        )
    except Exception as e:
        logger.info("[req-prof-sched] emit failed: %r", e)
