"""Opt-in eager MoE boundary timing, drained only by an idle cache flush.

No per-layer synchronization or file I/O. Nested/alternate-stream intervals
overlap and must not be added. Measure instrumented/uninstrumented service
controls separately: CUDA events can perturb short kernels and host dispatch.
"""

import contextvars
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

DIRECTORY = os.environ.get("SGLANG_MOK_BOUNDARY_TIMING_DIR", "")
MIN_TOKENS = int(os.environ.get("SGLANG_MOK_BOUNDARY_TIMING_MIN_TOKENS", "256"))
MAX_RECORDS = int(os.environ.get("SGLANG_MOK_BOUNDARY_TIMING_MAX_RECORDS", "4096"))
_parent = contextvars.ContextVar("mok_boundary_parent", default=None)
_recorder = None
_flush_index = 0


class Recorder:
    def __init__(self, cuda, limit):
        if limit <= 0:
            raise ValueError("boundary timing record limit must be positive")
        self.cuda = cuda
        self.limit = limit
        self.pending = []
        self.pool = {}
        self.sequence = 0
        self.skipped = {}

    def skip(self, reason):
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def begin(self, label, owner, hidden):
        if self.cuda.is_current_stream_capturing():
            self.skip("cuda_graph_capture")
            return None
        if len(self.pending) >= self.limit:
            self.skip("record_capacity")
            return None
        stream = self.cuda.current_stream(hidden.device)
        device = hidden.device.index
        pool = self.pool.setdefault(device, [])
        events = pool.pop() if pool else (
            self.cuda.Event(enable_timing=True), self.cuda.Event(enable_timing=True)
        )
        row = {"sequence": self.sequence, "parent": _parent.get(), "label": label,
               "layer_id": getattr(owner, "layer_id", None), "rows": hidden.shape[0],
               "device": device, "stream": stream.cuda_stream,
               "host_start_ns": time.perf_counter_ns(), "outcome": "incomplete"}
        self.sequence += 1
        events[0].record(stream)
        item = (row, events, stream)
        self.pending.append(item)
        return item

    def finish(self, item, outcome):
        row, events, stream = item
        events[1].record(stream)
        row.update(host_end_ns=time.perf_counter_ns(), outcome=outcome)

    def drain(self):
        if any(row["outcome"] == "incomplete" for row, _, _ in self.pending):
            raise RuntimeError("cannot drain boundary timing inside an active scope")
        # Caller must already have established scheduler idleness. This is the
        # only synchronization in the recorder, outside measured requests.
        if self.pending:
            self.cuda.synchronize()
        rows = []
        for row, events, _ in self.pending:
            rows.append({**row, "cuda_elapsed_ms": events[0].elapsed_time(events[1]),
                         "host_enqueue_span_ns": row["host_end_ns"] - row["host_start_ns"]})
            self.pool[row["device"]].append(events)
        result = {"rows": rows, "skipped": dict(self.skipped),
                  "complete_eager_recording": not self.skipped,
                  "intervals_are_additive": False,
                  "graph_replay_measured": False}
        self.pending.clear()
        self.skipped.clear()
        return result


def boundary(label):
    """Decorate (owner, hidden_states, ...) without wrapping when disabled."""
    def decorate(function):
        if not DIRECTORY:
            return function
        owner_name = next(iter(inspect.signature(function).parameters))

        @functools.wraps(function)
        def measured(*args, **kwargs):
            global _recorder
            owner = args[0] if args else kwargs[owner_name]
            hidden_states = args[1] if len(args) > 1 else kwargs["hidden_states"]
            if hidden_states.shape[0] < MIN_TOKENS:
                return function(*args, **kwargs)
            import torch

            if _recorder is None:
                _recorder = Recorder(torch.cuda, MAX_RECORDS)
            item = _recorder.begin(label, owner, hidden_states)
            if item is None:
                return function(*args, **kwargs)
            token = _parent.set(item[0]["sequence"])
            outcome = "exception"
            try:
                result = function(*args, **kwargs)
                outcome = "none" if result is None else "returned_output"
                return result
            finally:
                _recorder.finish(item, outcome)
                _parent.reset(token)

        return measured
    return decorate


def flush_if_enabled():
    """Called on every rank after scheduler idleness, before clearing caches."""
    global _flush_index
    if not DIRECTORY:
        return
    import torch
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    payload = _recorder.drain() if _recorder is not None else {
        "rows": [], "skipped": {}, "complete_eager_recording": True,
        "intervals_are_additive": False, "graph_replay_measured": False,
    }
    payload.update(schema="mok-boundary-timing-v1", rank=rank, pid=os.getpid(),
                   flush_index=_flush_index, min_tokens=MIN_TOKENS, max_records=MAX_RECORDS,
                   device_uuid=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
                   recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   boundary="CUDA events around eager Python call on its current stream; host enqueue span reported separately",
                   model_quality_or_service_performance_go=False)
    directory = Path(DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank{rank}-pid{os.getpid()}-flush{_flush_index:04d}.json"
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    _flush_index += 1
