"""Opt-in eager DSV4 MoK coverage counters, exported only at scheduler idle.

No CUDA calls, tensor retention or I/O on the request path. Python returns are
launch/return evidence only; an idle export synchronizes the device. Consumers
must additionally verify graph/overlap/CP are disabled and client outputs exist.
This is quality coverage instrumentation, not a performance measurement tool.
"""

import contextvars
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import uuid

DIRECTORY = os.environ.get("SGLANG_MOK_PATH_AUDIT_DIR", "")
_model = contextvars.ContextVar("mok_audit_model", default=None)
_outer = contextvars.ContextVar("mok_audit_outer", default=None)
_recorder = None
_flush_index = 0
_session_id = uuid.uuid4().hex

COUNTERS = ("entered", "returned", "none", "exception", "tokens",
            "native_entered", "native_returned", "native_none", "native_exception",
            "core_entered", "core_returned", "core_none", "core_exception")


def runtime_policy():
    from sglang.srt.environ import envs

    return {
        "native": envs.SGLANG_OPT_USE_MOK_FP8_NATIVE.get(),
        "warprole": envs.SGLANG_OPT_MOK_WARPROLE.get(),
        "variant": envs.SGLANG_OPT_MOK_WARPROLE_VARIANT.get(),
        "min_tokens": envs.SGLANG_OPT_MOK_MIN_TOKENS.get(),
        "max_tokens": envs.SGLANG_OPT_MOK_MAX_TOKENS.get(),
        "max_sequence_tokens": envs.SGLANG_OPT_MOK_MAX_SEQUENCE_TOKENS.get(),
        "strict": envs.SGLANG_OPT_MOK_FP8_NATIVE_STRICT.get(),
        "prefill_graph": envs.SGLANG_OPT_MOK_FP8_NATIVE_PREFILL_GRAPH.get(),
        "workspace_cap": envs.SGLANG_OPT_MOK_WORKSPACE_CACHE_CAP.get(),
    }


def batch_policy(rows, policy):
    from sglang.srt.layers.dp_attention import (
        get_is_extend_in_batch,
        get_max_sequence_length,
    )

    extend = get_is_extend_in_batch()
    mode = "extend" if extend else "decode"
    if policy["min_tokens"] > 0 and not extend:
        reason = "decode"
    elif policy["min_tokens"] > 0 and rows < policy["min_tokens"]:
        reason = "short_prefill"
    elif policy["max_tokens"] > 0 and rows > policy["max_tokens"]:
        reason = "max_tokens"
    elif (policy["max_sequence_tokens"] > 0
          and get_max_sequence_length() > policy["max_sequence_tokens"]):
        reason = "max_sequence_tokens"
    else:
        reason = "eligible"
    return mode, reason


def runtime_layout():
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    return {"attn_tp_size": parallel.attn_tp_size,
            "attn_tp_rank": parallel.attn_tp_rank,
            "attn_dp_size": parallel.attn_dp_size,
            "attn_cp_size": parallel.attn_cp_size,
            "moe_ep_size": parallel.moe_ep_size,
            "backend": get_moe_a2a_backend().value,
            "tp_attention_scatter": envs.SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER.get()}


def local_moe_tokens(tokens, layout):
    """Mirror tensor_split's row partition, including a possible remainder.

    DSV4 _run_moe_ffn_dp_sync scatters replicated attention-TP tokens before
    the EP MoE call. Model input_ids may already include MLP-sync padding.
    CP/DP-attention paths are outside this recorder's quality contract.
    """
    if layout["attn_cp_size"] != 1 or layout["attn_dp_size"] != 1:
        raise ValueError("MoK path audit requires CP1 and attention DP1")
    if layout["tp_attention_scatter"] and layout["attn_tp_size"] > 1 and layout["backend"] != "none":
        quotient, remainder = divmod(tokens, layout["attn_tp_size"])
        return quotient + int(layout["attn_tp_rank"] < remainder)
    return tokens


class Recorder:
    def __init__(self):
        self.rows = {}
        self.models = dict.fromkeys(("entered", "returned", "none", "exception"), 0)
        self.errors = {}
        self.active = 0
        self.policy = None
        self.layout = None
        self.batches = {}
        self.order = hashlib.sha256()

    def error(self, name):
        self.errors[name] = self.errors.get(name, 0) + 1

    def check_policy(self, policy):
        if self.policy is None:
            self.policy = dict(policy)
        elif self.policy != policy:
            self.error("policy_changed")

    def check_layout(self, layout):
        if self.layout is None:
            self.layout = dict(layout)
        elif self.layout != layout:
            self.error("layout_changed")

    def begin_outer(self, owner, hidden, policy, mode, eligibility):
        self.check_policy(policy)
        layer_id = getattr(owner, "layer_id", None)
        model = _model.get()
        rows = int(hidden.shape[0])
        if model is None:
            self.error("outer_without_model")
        else:
            model["layers"].append(layer_id)
            if rows != model["local_tokens"]:
                self.error("model_moe_token_mismatch")
            if (mode, eligibility) != (model["mode"], model["eligibility"]):
                self.error("model_moe_policy_mismatch")
        if _outer.get() is not None:
            self.error("nested_outer")
        if not isinstance(layer_id, int):
            self.error("missing_layer_id")
        key = (layer_id, int(hidden.device.index), mode,
               "ge256" if rows >= 256 else "lt256", eligibility)
        if key not in self.rows:
            self.rows[key] = {"layer_id": key[0], "device": key[1], "mode": key[2],
                              "bucket": key[3], "eligibility": key[4],
                              **dict.fromkeys(COUNTERS, 0)}
        row = self.rows[key]
        row["entered"] += 1
        row["tokens"] += rows
        return {"row": row, "layer_id": layer_id, "native_active": 0,
                "before": dict(row), "native": policy["native"],
                "eligible": eligibility == "eligible"}

    def snapshot(self):
        if self.active:
            raise RuntimeError("path audit snapshot inside active execution")
        return {"rows": [dict(v) for _, v in sorted(self.rows.items(), key=lambda x: str(x[0]))],
                "models": dict(self.models), "errors": dict(self.errors),
                "policy": self.policy, "layout": self.layout,
                "model_batches": [dict(v) for _, v in sorted(self.batches.items())],
                "model_order_sha256": self.order.hexdigest()}


def _get_recorder():
    global _recorder
    if _recorder is None:
        _recorder = Recorder()
    return _recorder


def scope(label):
    """Supported hooks: DSV4 model, routed MoE, native adapter, native core."""
    if label not in ("model", "outer", "native", "core"):
        raise ValueError("unknown MoK audit scope")

    def decorate(function):
        if not DIRECTORY:
            return function
        signature = inspect.signature(function)
        owner_name = next(iter(signature.parameters))

        @functools.wraps(function)
        def audited(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            owner = bound[owner_name]
            recorder = _get_recorder()
            frame = None
            context_token = None
            counters = None
            prefix = ""
            if label == "model":
                if _model.get() is not None:
                    recorder.error("nested_model")
                layout, policy = runtime_layout(), runtime_policy()
                recorder.check_layout(layout)
                recorder.check_policy(policy)
                tokens = int(bound["input_ids"].shape[0])
                local_tokens = local_moe_tokens(tokens, layout)
                mode, eligibility = batch_policy(local_tokens, policy)
                frame = {"layers": [], "tokens": tokens, "local_tokens": local_tokens,
                         "mode": mode, "eligibility": eligibility,
                         "expected": list(range(owner.start_layer, owner.end_layer))}
                key = (mode, tokens, local_tokens, eligibility)
                row = recorder.batches.setdefault(key, {"mode": mode,
                    "model_tokens": tokens, "local_tokens": local_tokens,
                    "eligibility": eligibility, "calls": 0})
                row["calls"] += 1
                context_token = _model.set(frame)
                counters = recorder.models
            elif label == "outer":
                policy = runtime_policy()
                hidden = bound["hidden_states"]
                mode, eligibility = batch_policy(int(hidden.shape[0]), policy)
                frame = recorder.begin_outer(owner, hidden, policy, mode, eligibility)
                context_token = _outer.set(frame)
                # begin_outer already records entered and token counts.
                counters = frame["row"]
            else:
                recorder.check_policy(runtime_policy())
                frame = _outer.get()
                if frame is None or frame["layer_id"] != getattr(owner, "layer_id", None):
                    recorder.error(label + "_without_matching_outer")
                else:
                    counters = frame["row"]
                    prefix = label + "_"
                    if label == "native":
                        frame["native_active"] += 1
                    elif frame["native_active"] != 1:
                        recorder.error("core_without_single_native")
            if counters is not None and label != "outer":
                counters[prefix + "entered"] += 1
            recorder.active += 1
            outcome = "exception"
            try:
                result = function(*args, **kwargs)
                outcome = "none" if result is None else "returned"
                return result
            finally:
                if counters is not None:
                    counters[prefix + outcome] += 1
                recorder.active -= 1
                if label == "model":
                    if frame["layers"] != frame["expected"]:
                        recorder.error("model_layer_coverage")
                    recorder.order.update(json.dumps(
                        [frame["tokens"], frame["mode"], frame["layers"], outcome],
                        separators=(",", ":")).encode() + b"\n")
                    _model.reset(context_token)
                elif label == "outer":
                    delta = {k: counters[k] - frame["before"][k] for k in COUNTERS}
                    expected_native = int(frame["native"])
                    expected_core = int(frame["native"] and frame["eligible"])
                    if outcome == "returned" and (
                        delta["native_entered"] != expected_native
                        or delta["native_returned"] != expected_core
                        or delta["native_none"] != expected_native - expected_core
                        or delta["native_exception"] != 0
                        or delta["core_entered"] != expected_core
                        or delta["core_returned"] != expected_core
                        or delta["core_none"] != 0
                        or delta["core_exception"] != 0
                    ):
                        recorder.error("outer_path_completion")
                    _outer.reset(context_token)
                elif label == "native" and counters is not None:
                    frame["native_active"] -= 1

        return audited
    return decorate


def flush_if_enabled():
    """Scheduler has established idleness; fail export on device sync failure."""
    global _flush_index
    if not DIRECTORY:
        return
    import torch
    import torch.distributed as dist

    recorder = _get_recorder()
    recorder.check_policy(runtime_policy())
    recorder.check_layout(runtime_layout())
    payload = recorder.snapshot()
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    from sglang.srt.utils.mok_fault_progress import poll_if_enabled

    poll_if_enabled()
    rank = dist.get_rank() if dist.is_initialized() else 0
    payload.update(schema="mok-path-audit-v2", rank=rank, pid=os.getpid(),
                   session_id=_session_id, flush_index=_flush_index,
                   device_uuid=str(torch.cuda.get_device_properties(device).uuid),
                   device=device, monotonic_ns=time.monotonic_ns(),
                   device_synchronized=True, graph_replay_measured=False,
                   recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   quality_or_performance_go=False)
    directory = Path(DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank{rank}-pid{os.getpid()}-flush{_flush_index:04d}.json"
    # Publish an entire immutable snapshot atomically. A failed writer leaves
    # a temporary file, which the consumer must reject rather than ignore.
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o444)
    os.link(temporary, path)
    temporary.unlink()
    _flush_index += 1
