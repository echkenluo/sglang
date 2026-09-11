"""Opt-in crash journal for large eager attention/MoE calls.

Events are queried without synchronizing. Journal entries describe host
submission and known completed stream checkpoints, not the faulting kernel.
Event recording, queries and file writes can still perturb a race. Disabled
by default; intended for reproductions, never performance measurements.
"""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

DIRECTORY = os.environ.get("SGLANG_MOK_FAULT_PROGRESS_DIR", "")
MAX_EVENTS = 1024
_recorder = None


class Recorder:
    def __init__(self, cuda, path, rank, device):
        self.cuda = cuda
        self.rank = rank
        self.device = device
        self.sequence = 0
        self.pending = []
        self.failed = False
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.write("header", schema="mok-fault-progress-v1", rank=rank,
                   device=device, pid=os.getpid(),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   event_queries_synchronize=False,
                   checkpoints_do_not_identify_faulting_kernel=True)

    def write(self, kind, **values):
        data = (json.dumps({"kind": kind, "host_monotonic_ns": time.monotonic_ns(),
                            **values}, allow_nan=False) + "\n").encode()
        while data:
            count = os.write(self.fd, data)
            if count <= 0:
                raise OSError("fault journal write made no progress")
            data = data[count:]

    def poll(self, attempted):
        if self.failed:
            raise RuntimeError("fault journal context already failed")
        remaining = []
        for row, event in self.pending:
            try:
                complete = event.query()
            except BaseException as exc:
                self.failed = True
                self.write("query_error", checkpoint=row, attempted_scope=attempted,
                           exception_type=type(exc).__name__, error=str(exc))
                raise
            if complete:
                self.write("gpu_event_complete", **row)
            else:
                remaining.append((row, event))
        self.pending = remaining

    def mark(self, phase, scope, stream):
        if self.failed:
            raise RuntimeError("fault journal context already failed")
        if self.sequence >= MAX_EVENTS:
            raise RuntimeError("fault journal event bound exceeded")
        row = {**scope, "event_sequence": self.sequence, "phase": phase,
               "stream": stream.cuda_stream}
        self.sequence += 1
        event = self.cuda.Event(enable_timing=False)
        try:
            event.record(stream)
        except BaseException as exc:
            self.failed = True
            self.write("record_error", checkpoint=row,
                       exception_type=type(exc).__name__, error=str(exc))
            raise
        self.pending.append((row, event))
        self.write("gpu_event_submitted", **row)


def boundary(label, *, minimum_rows, tensor_argument="hidden_states"):
    def decorate(function):
        if not DIRECTORY:
            return function
        signature = inspect.signature(function)

        @functools.wraps(function)
        def recorded(*args, **kwargs):
            global _recorder
            bound = signature.bind(*args, **kwargs).arguments
            owner = bound[next(iter(signature.parameters))]
            hidden = bound[tensor_argument]
            if hidden.shape[0] < minimum_rows:
                return function(*args, **kwargs)
            import torch
            import torch.distributed as dist

            if torch.cuda.is_current_stream_capturing():
                return function(*args, **kwargs)
            device = hidden.device.index
            if _recorder is None:
                rank = dist.get_rank() if dist.is_initialized() else 0
                directory = Path(DIRECTORY)
                directory.mkdir(parents=True, exist_ok=True)
                _recorder = Recorder(torch.cuda, directory / f"rank{rank}-pid{os.getpid()}.jsonl", rank, device)
            if _recorder.device != device:
                raise RuntimeError("fault journal cannot mix devices in one rank")
            scope = {"label": label, "layer_id": getattr(owner, "layer_id", None),
                     "rows": int(hidden.shape[0]), "hidden_shape": list(hidden.shape),
                     "hidden_stride": list(hidden.stride()), "hidden_dtype": str(hidden.dtype),
                     "hidden_data_ptr": hidden.data_ptr(),
                     "hidden_storage_offset": hidden.storage_offset(),
                     "hidden_storage_bytes": hidden.untyped_storage().nbytes()}
            _recorder.write("host_enter", **scope)
            _recorder.poll(scope)
            stream = torch.cuda.current_stream(hidden.device)
            _recorder.mark("before", scope, stream)
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                # Do not issue more CUDA calls after an exception; the NCCL
                # watchdog may abort soon, and the first error must survive.
                _recorder.failed = True
                _recorder.write("host_exception", **scope,
                                exception_type=type(exc).__name__, error=str(exc))
                raise
            _recorder.mark("after", scope, stream)
            _recorder.write("host_return", **scope)
            _recorder.poll(scope)
            return result

        return recorded
    return decorate


def poll_if_enabled():
    """Called after the existing idle audit synchronization, never in a layer."""
    if DIRECTORY and _recorder is not None:
        _recorder.poll({"label": "idle_flush"})


def record_warmup_input(size, input_ids):
    """Save the CPU-generated reproduction inputs before their GPU request."""
    if not DIRECTORY:
        return
    directory = Path(DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(input_ids, separators=(",", ":")).encode()
    payload = {"size": size, "input_ids": input_ids,
               "input_sha256": hashlib.sha256(encoded).hexdigest(),
               "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
               "pid": os.getpid()}
    with (directory / f"warmup-input-{size}-pid{os.getpid()}.json").open("x") as stream:
        json.dump(payload, stream, allow_nan=False)
        stream.write("\n")
