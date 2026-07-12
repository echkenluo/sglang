"""Per-request TTFT segment bill (env ``SGLANG_REQUEST_PROF=1``) — v2.

v2 zero-intervention principle (v1 post-mortem: a BaseHTTPMiddleware arrival
stamp deadlocked the kick-started StreamingResponse path — requests never
reached the scheduler and shutdown raised "aclose(): asynchronous generator
is already running"). v2 therefore:

  * adds NO middleware and touches NO async generator / control flow;
  * only (a) plain ``perf_counter()`` assignments in the serving layer
    (template/encode marks via a request-task-local ContextVar), and (b) ONE
    log-line emission at the existing request-termination wrap-up block in
    ``TokenizerManager._handle_batch_output`` (``state.finished``), once per
    request, outside the token hot path.

One line per request::

  [req-prof] rid=.. total=.. template=.. encode=.. convert_other=..
      tm_ingest=.. dispatch=.. wait_first=.. | sq_ipc_in=.. sq_recv2wq=..
      sq_wq2fwd=.. sq_prefill=.. sq_out=..

Same-clock segments (TokenizerManager process, perf_counter):
  template       apply_chat_template render (serving_chat marks, cumulative)
  encode         tokenizer.encode (serving_chat marks)
  convert_other  handler entry -> adapted request ready, minus the two marks
  tm_ingest      adapted ready -> TM tokenize done (async handoff+passthrough)
  dispatch       tokenize done -> ZMQ send
  wait_first     ZMQ send -> first token back in TM (cross-process envelope)
  total          handler entry -> first token   (http_parse is NOT measured;
                 approximate from the uvicorn access log if needed)

Scheduler sub-splits of wait_first (from the SchedulerReqTimeStats that
output_streamer attaches to every batch output; its ``__getstate__``
whitelists exactly wait_queue/forward_entry/prefill_finished for the wire
and only under ``--enable-metrics`` — scheduler_recv_time never travels, so
the recv->wait-queue us-scale hop is merged into sq_ipc_admit. Timestamps
are converted into this process's monotonic frame by
``ReqTimeStatsBase.__setstate__`` and subtract directly; small negatives
indicate cross-process clock-conversion skew and are printed as-is):
  sq_ipc_admit   ZMQ send -> wait queue entry (IPC in + recv handling)
  sq_wq2fwd      wait queue -> first forward entry (admission)
  sq_prefill     forward entry -> prefill finished (incl. chunked)
  sq_out         prefill finished -> first token back in TM (detok + IPC back)
Lanes without --enable-metrics get an explicit ``sq=unavailable`` marker
instead of silence.

Failure mode of every touch point is "one missing log line", never a request
error: emission wraps everything in try/except; marks are inert assignments.
Covered path: single-request streaming chat completions (codex's).
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


def emit_tm_line(rid, req_prof: dict, ts, sched_ts=None) -> None:
    """ts: APIServerReqTimeStats; sched_ts: SchedulerReqTimeStats or None."""
    try:
        t_handler = req_prof["t_handler"]
        t_conv = req_prof["t_convert_done"]
        template_ms = req_prof.get("template", 0.0) * 1e3
        encode_ms = req_prof.get("encode", 0.0) * 1e3
        convert_other = (t_conv - t_handler) * 1e3 - template_ms - encode_ms
        tm_ingest = (ts.tokenize_finish_time - t_conv) * 1e3
        dispatch = (ts.api_server_dispatch_time - ts.tokenize_finish_time) * 1e3
        wait_first = (ts.first_token_time - ts.api_server_dispatch_time) * 1e3
        total = (ts.first_token_time - t_handler) * 1e3
        line = (
            f"[req-prof] rid={rid} total={total:.1f} template={template_ms:.1f} "
            f"encode={encode_ms:.1f} convert_other={convert_other:.1f} "
            f"tm_ingest={tm_ingest:.1f} dispatch={dispatch:.1f} "
            f"wait_first={wait_first:.1f}"
        )
        if sched_ts is not None:
            try:
                # SchedulerReqTimeStats.__getstate__ whitelists exactly three
                # timestamps for the wire (wait_queue / forward_entry /
                # prefill_finished; scheduler_recv_time is NOT serialized and
                # unpickles to the 0.0 class default), and only under
                # --enable-metrics. Split wait_first with what travels;
                # recv->wait-queue is us-scale and merges into sq_ipc_admit.
                wq = getattr(sched_ts, "wait_queue_entry_time", 0.0)
                fe = getattr(sched_ts, "forward_entry_time", 0.0)
                pf = getattr(sched_ts, "prefill_finished_time", 0.0)
                if wq and fe and pf:
                    line += (
                        f" | sq_ipc_admit={(wq - ts.api_server_dispatch_time) * 1e3:.1f}"
                        f" sq_wq2fwd={(fe - wq) * 1e3:.1f}"
                        f" sq_prefill={(pf - fe) * 1e3:.1f}"
                        f" sq_out={(ts.first_token_time - pf) * 1e3:.1f}"
                    )
                else:
                    # Diagnosable, not silent (silence cost one roundtrip).
                    line += " | sq=unavailable(needs --enable-metrics)"
            except Exception:
                pass  # sched splits are best-effort garnish
        logger.info(line)
    except Exception as e:
        logger.info("[req-prof] emit failed rid=%s: %r", rid, e)
