"""Correctness-first full-native MoK FP8 routed-expert path for SM90."""

from __future__ import annotations

import atexit
import logging
import math
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import NoReturn, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper

logger = logging.getLogger(__name__)


_TRAP_WATCHDOG_LOCK = threading.Lock()
_TRAP_WATCHDOG_ENTRIES: list = []
_TRAP_WATCHDOG_STOP = threading.Event()
_TRAP_WATCHDOG_THREAD: Optional[threading.Thread] = None
_TRAP_WATCHDOG_STARTED = False


def _trap_watchdog_loop() -> None:
    """Async production endpoint of the MoK trap contract: kernel launches
    and graph replays are asynchronous, so a device trap may never surface
    as a RuntimeError inside the local try/except boundaries.  This daemon
    thread polls the host-mapped pinned records with pure CPU reads (never
    a CUDA call) and performs the log-and-exit contract the moment any
    record turns non-zero."""
    while not _TRAP_WATCHDOG_STOP.is_set():
        for workspace, mok_functional in list(_TRAP_WATCHDOG_ENTRIES):
            record = mok_functional.format_trap_record(workspace)
            if record is not None:
                logger.error(record)
                logging.shutdown()
                os._exit(70)
        _TRAP_WATCHDOG_STOP.wait(0.05)


def _register_trap_watchdog(workspace, mok_functional) -> None:
    global _TRAP_WATCHDOG_STARTED, _TRAP_WATCHDOG_THREAD
    with _TRAP_WATCHDOG_LOCK:
        if not any(w is workspace for w, _ in _TRAP_WATCHDOG_ENTRIES):
            _TRAP_WATCHDOG_ENTRIES.append((workspace, mok_functional))
        if _TRAP_WATCHDOG_THREAD is None or not _TRAP_WATCHDOG_THREAD.is_alive():
            _TRAP_WATCHDOG_STOP.clear()
            _TRAP_WATCHDOG_THREAD = threading.Thread(
                target=_trap_watchdog_loop,
                name="mok-trap-watchdog",
                daemon=True,
            )
            _TRAP_WATCHDOG_THREAD.start()
            _TRAP_WATCHDOG_STARTED = True


def shutdown_trap_watchdog() -> None:
    """Stop the host-only trap poller before distributed/CUDA teardown.

    The poller deliberately remains a daemon for fatal production shutdown,
    but graceful exits must not leave it running while PyTorch unloads.  This
    operation is CPU-only, idempotent, and leaves the watchdog restartable for
    a later workspace registration in the same process.
    """
    global _TRAP_WATCHDOG_STARTED, _TRAP_WATCHDOG_THREAD
    with _TRAP_WATCHDOG_LOCK:
        _TRAP_WATCHDOG_STOP.set()
        thread = _TRAP_WATCHDOG_THREAD
        if thread is not None:
            thread.join()
        _TRAP_WATCHDOG_ENTRIES.clear()
        _TRAP_WATCHDOG_THREAD = None
        _TRAP_WATCHDOG_STARTED = False
        _TRAP_WATCHDOG_STOP.clear()


atexit.register(shutdown_trap_watchdog)


def _die_if_trapped(workspace, mok_functional) -> None:
    """CPU-only fatal boundary of the MoK trap contract: after a CUDA error,
    read the host-mapped pinned record WITHOUT issuing any CUDA call; a
    non-zero record means a device trap poisoned the context -- log the
    structured MOK_TRAP line and exit so the supervisor restarts us.  Callers
    invoke this from every exception path (eager, graph warmup, capture,
    replay) before doing anything else."""
    record = mok_functional.format_trap_record(workspace)
    if record is not None:
        logger.error(record)
        logging.shutdown()
        os._exit(70)


def _fatal_terminal_transaction_failure(
    workspace,
    mok_functional,
    phase: str,
    error: BaseException,
) -> NoReturn:
    """CPU-only endpoint for every exception after lease acquire is attempted.

    The acquire kernel may already have set ``in_use`` and an arbitrary CUDA
    exception may have poisoned the context.  Do not issue a release or any
    other CUDA operation; format the host-mapped receipt and terminate so the
    supervisor recreates the process/context.
    """
    try:
        receipt = mok_functional.format_terminal_transaction_failure(
            workspace, phase, error
        )
    except BaseException:
        receipt = (
            "MOK_TERMINAL_TRANSACTION_FATAL"
            f"|phase={phase}|error_type={type(error).__name__}"
        )
    try:
        logger.error(receipt)
        logging.shutdown()
    finally:
        os._exit(70)


_REPORTED_FALLBACKS: set[str] = set()
_REPORTED_ACTIVE = False
_REPORTED_TERMINAL_LAYER_IDS: set[object] = set()
_REPORTED_TERMINAL_QUANT_PREWARM: set[tuple] = set()
_ROUTE_EXPERT_PADDING = 64
_TERMINAL_DEEPEP_OUTER_ACTIVE: ContextVar[bool] = ContextVar(
    "mok_terminal_deepep_outer_active", default=False
)
_TERMINAL_DECODE_GRAPH_ACTIVE: ContextVar[bool] = ContextVar(
    "mok_terminal_decode_graph_active", default=False
)


def _terminal_forward_mode_is_decode(forward_mode) -> bool:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return forward_mode == ForwardMode.DECODE


def _terminal_concurrency_mode_error() -> Optional[str]:
    """Return a configured concurrent-mode error, if runtime config exists."""
    from sglang.srt.runtime_context import get_server_args

    try:
        server_args = get_server_args()
    except ValueError:
        # Direct unit/kernel callers have no process-wide ServerArgs.  Catch
        # only that explicit condition: a present but incomplete config must
        # raise instead of silently defaulting a safety field to false.
        return None
    if server_args.enable_pdmux:
        return "terminal MoK rejects PDMux concurrent graph streams"
    if server_args.enable_two_batch_overlap:
        return "terminal MoK rejects two-batch overlap (TBO)"
    if server_args.enable_single_batch_overlap:
        return "terminal MoK rejects single-batch overlap (SBO)"
    return None


def _is_terminal_full_decode_graph(forward_mode) -> bool:
    """Whether this DeepEP outer is SGLang's supported graph context."""
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        is_in_tc_piecewise_cuda_graph,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_full_decode_cuda_graph_mode,
        get_is_capture_mode,
    )

    return (
        _terminal_forward_mode_is_decode(forward_mode)
        and get_is_capture_mode()
        and get_is_full_decode_cuda_graph_mode()
        and not is_in_tc_piecewise_cuda_graph()
        and not is_in_breakable_cuda_graph()
        and _terminal_concurrency_mode_error() is None
    )


@contextmanager
def terminal_deepep_outer_context(forward_mode=None):
    """Mark the exact DeepSeekV2 DeepEP outer semantics for terminal MoK.

    The terminal kernel returns the routed-expert contribution.  The enclosing
    DeepseekV2MoE.forward_deepep path owns routed_scaling_factor and the shared
    expert add.  Keeping this explicit prevents another FusedMoE caller from
    accidentally treating the routed output as a fully finalized model output.
    """
    outer_token = _TERMINAL_DEEPEP_OUTER_ACTIVE.set(True)
    graph_token = _TERMINAL_DECODE_GRAPH_ACTIVE.set(
        _is_terminal_full_decode_graph(forward_mode)
    )
    try:
        yield
    finally:
        _TERMINAL_DECODE_GRAPH_ACTIVE.reset(graph_token)
        _TERMINAL_DEEPEP_OUTER_ACTIVE.reset(outer_token)


def _terminal_deepep_backend_is_active() -> bool:
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    return get_moe_a2a_backend().is_deepep()


def _terminal_graph_context_error(forward_mode=None) -> Optional[str]:
    """Allow only SGLang's whole-model Full graph for plain Decode.

    The top-level DeepSeek MoE call supplies ``forward_mode``.  The nested
    terminal adapter deliberately supplies none and is allowed only when the
    validated DeepEP outer context propagated the decode-graph capability.
    """
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        is_in_tc_piecewise_cuda_graph,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_full_decode_cuda_graph_mode,
        get_is_capture_mode,
    )

    if is_in_tc_piecewise_cuda_graph():
        return "terminal MoK rejects tc_piecewise CUDA graph"
    if is_in_breakable_cuda_graph():
        return "terminal MoK rejects breakable CUDA graph"
    if get_is_capture_mode():
        if not get_is_full_decode_cuda_graph_mode():
            return (
                "terminal MoK rejects graph capture outside the production "
                "DecodeCudaGraphRunner Full backend"
            )
        if concurrency_error := _terminal_concurrency_mode_error():
            return concurrency_error
        if _TERMINAL_DECODE_GRAPH_ACTIVE.get():
            return None
        if forward_mode is None:
            return (
                "terminal MoK whole-model graph capture requires the validated "
                "DeepEP ForwardMode.DECODE outer context"
            )
        if not _terminal_forward_mode_is_decode(forward_mode):
            return (
                "terminal MoK whole-model graph capture supports only "
                "ForwardMode.DECODE"
            )
        return None
    if torch.cuda.is_current_stream_capturing():
        return "terminal MoK rejects external active CUDA graph capture"
    return None


def native_shape_contract_error(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    num_local_experts: int,
    num_global_experts: int,
    ep_size: int,
) -> Optional[str]:
    """Return the first unsupported static full-native tensor contract."""
    if hidden_states.ndim != 2:
        return "hidden_states must have shape [T,H]"
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens <= 0:
        return "at least one token is required"
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        return "hidden_states must be contiguous bfloat16"
    if topk_ids.ndim != 2 or topk_ids.shape[0] != num_tokens:
        return "topk_ids must have shape [T,topk]"
    if topk_ids.dtype not in (torch.int32, torch.int64):
        return "topk_ids must use int32 or int64"
    if not topk_ids.is_contiguous():
        return "topk_ids must be contiguous"
    if (
        topk_weights.ndim != 2
        or tuple(topk_weights.shape) != tuple(topk_ids.shape)
        or topk_weights.dtype != torch.float32
        or not topk_weights.is_contiguous()
    ):
        return "topk_weights must be contiguous float32 with the topk_ids shape"
    topk = topk_ids.shape[1]
    if not 0 < topk <= 255:
        return "topk must be in [1,255]"
    if hidden_size <= 0 or hidden_size % 256 != 0:
        return "hidden size must be positive and divisible by 256"
    if w13_weight.ndim != 3 or w2_weight.ndim != 3:
        return "expert weights must be rank-3"
    if num_local_experts <= 0 or num_global_experts <= 0 or ep_size <= 0:
        return "expert and EP counts must be positive"
    if ep_size not in (4, 8, 16, 32, 64):
        return "MoK EP size must be one of 4, 8, 16, 32, 64"
    if num_global_experts != num_local_experts * ep_size:
        return "global experts must equal local experts times EP size"
    if (
        w13_weight.shape[0] != num_local_experts
        or w2_weight.shape[0] != num_local_experts
    ):
        return "weight expert dimension must equal num_local_experts"
    gate_up_size = w13_weight.shape[1]
    intermediate_size = w2_weight.shape[2]
    if gate_up_size != 2 * intermediate_size:
        return "w13 output must be twice the w2 reduction dimension"
    if w13_weight.shape[2] != hidden_size or w2_weight.shape[1] != hidden_size:
        return "expert weight hidden dimensions do not match hidden_states"
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        return (
            "intermediate size must be positive and divisible by 256 for "
            "the production contiguous clamp-SwiGLU kernel"
        )
    if any(tensor.dtype != torch.float8_e4m3fn for tensor in (w13_weight, w2_weight)):
        return "expert weights must use float8_e4m3fn"
    if any(tensor.dtype != torch.float32 for tensor in (w13_scale, w2_scale)):
        return "expert block scales must use float32"
    expected_scale_shapes = (
        (
            num_local_experts,
            gate_up_size // 128,
            hidden_size // 128,
        ),
        (
            num_local_experts,
            hidden_size // 128,
            intermediate_size // 128,
        ),
    )
    if (tuple(w13_scale.shape), tuple(w2_scale.shape)) != expected_scale_shapes:
        return (
            "invalid expert block-scale shapes: "
            f"expected={expected_scale_shapes}, actual="
            f"{(tuple(w13_scale.shape), tuple(w2_scale.shape))}"
        )
    tensors = (
        hidden_states,
        topk_ids,
        topk_weights,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
    )
    if not all(tensor.is_contiguous() for tensor in tensors):
        return "all full-native tensors must be contiguous"
    return None


def native_runtime_contract_error(layer, hidden_states, topk_output) -> Optional[str]:
    """Validate the production-only conditions not encoded by tensor shapes."""
    from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled

    if not hasattr(topk_output, "topk_ids") or not hasattr(topk_output, "topk_weights"):
        return "standard top-k tensors are required"
    if layer.quant_method.__class__.__name__ != "Fp8MoEMethod":
        return "Fp8MoEMethod is required"
    quant_config = layer.quant_method.quant_config
    if tuple(quant_config.weight_block_size or ()) != (128, 128):
        return "FP8 [128,128] block quantization is required"
    if getattr(layer.quant_method, "is_fp4_expert", False):
        return "FP4 experts are unsupported"
    config = layer.moe_runner_config
    if config.activation != "silu" or not config.is_gated:
        return "gated SiLU experts are required"
    if config.swiglu_limit != 10:
        return "DeepSeek-V4 swiglu_limit=10 is required"
    if config.apply_router_weight_on_input:
        return "router weights applied on input are unsupported"
    if config.no_combine:
        return "no_combine layers are unsupported"
    if getattr(layer.quant_method, "with_bias", False):
        return "expert bias is unsupported"
    if getattr(layer, "moe_tp_size", 1) != 1:
        return "MoE tensor parallelism is unsupported"
    if is_tbo_enabled() or is_sbo_enabled():
        return "batch-overlap modes are unsupported"
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        return "UE8M0 block scales are unsupported"
    scales = (
        getattr(layer, "w13_weight_scale_inv", None),
        getattr(layer, "w2_weight_scale_inv", None),
    )
    if any(scale is None for scale in scales):
        return "FP8 inverse block scales are required"
    error = native_shape_contract_error(
        hidden_states,
        topk_output.topk_ids,
        topk_output.topk_weights,
        layer.w13_weight,
        scales[0],
        layer.w2_weight,
        scales[1],
        num_local_experts=layer.num_local_experts,
        num_global_experts=layer.num_experts,
        ep_size=layer.moe_ep_size,
    )
    if error is not None:
        return error
    tensors = (
        hidden_states,
        topk_output.topk_ids,
        topk_output.topk_weights,
        layer.w13_weight,
        scales[0],
        layer.w2_weight,
        scales[1],
    )
    if not all(tensor.is_cuda for tensor in tensors):
        return "all full-native tensors must be CUDA tensors"
    if not all(tensor.device == hidden_states.device for tensor in tensors):
        return "all full-native tensors must share one CUDA device"
    if torch.cuda.get_device_capability(hidden_states.device) != (9, 0):
        return "full-native MoK currently requires SM90"
    return None


def native_terminal_contract_error(
    layer, hidden_states, topk_output
) -> Optional[str]:
    """Validate the exact eager terminal DeepSeek-V4 specialization."""
    # The production target deliberately retains the DeepSeekV2 DeepEP outer
    # path: terminal MoK replaces only its routed-expert implementation, while
    # that outer applies routed_scaling_factor and adds the separate shared
    # expert.  No other FusedMoE caller has the same finalization semantics.
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE

    if not isinstance(layer, DeepEPMoE):
        return "terminal MoK requires the DeepEPMoE production layer"
    if not _terminal_deepep_backend_is_active():
        return "terminal MoK requires --moe-a2a-backend deepep"
    if not _TERMINAL_DEEPEP_OUTER_ACTIVE.get():
        return "terminal MoK requires DeepseekV2MoE.forward_deepep outer semantics"
    if getattr(layer, "reduce_results", False):
        return "terminal MoK routed output must not be all-reduced inside FusedMoE"
    if getattr(layer, "num_fused_shared_experts", 0) != 0 or getattr(
        layer, "_has_fused_shared", False
    ):
        return (
            "terminal MoK requires shared-expert fusion disabled; use "
            "--disable-shared-experts-fusion"
        )
    error = native_runtime_contract_error(layer, hidden_states, topk_output)
    if error is not None:
        return error
    if layer.moe_ep_size != 4:
        return "terminal MoK requires EP4"
    if layer.num_experts != 256 or layer.num_local_experts != 64:
        return "terminal MoK requires E_global=256 and E_local=64"
    if hidden_states.shape[1] != 4096:
        return "terminal MoK requires hidden_size=4096"
    if topk_output.topk_ids.shape[1] != 6:
        return "terminal MoK requires topk=6"
    if tuple(layer.w13_weight.shape) != (64, 4096, 4096):
        return "terminal MoK requires canonical w13 [64,4096,4096]"
    if tuple(layer.w2_weight.shape) != (64, 4096, 2048):
        return "terminal MoK requires canonical w2 [64,4096,2048]"
    if tuple(layer.w13_weight_scale_inv.shape) != (64, 32, 32):
        return "terminal MoK requires canonical w13 block scales [64,32,32]"
    if tuple(layer.w2_weight_scale_inv.shape) != (64, 32, 16):
        return "terminal MoK requires canonical w2 block scales [64,32,16]"
    if getattr(layer.quant_method, "load_up_proj_weight_first", False):
        return "terminal MoK requires canonical gate-then-up w13 packing"
    if any(
        bool(getattr(weight, "is_shuffled", False))
        for weight in (layer.w13_weight, layer.w2_weight)
    ):
        return "terminal MoK does not accept shuffled expert weights"
    if any(
        bool(getattr(scale, "format_ue8m0", False))
        for scale in (
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
        )
    ):
        return "terminal MoK requires canonical FP32 block scales, not UE8M0"
    return None


def _terminal_mode_config_error() -> Optional[str]:
    """Reject modes that cannot preserve the terminal workspace contract."""
    if envs.SGLANG_OPT_MOK_FP8_NATIVE_PREFILL_GRAPH.get():
        return "terminal MoK eager adapter does not support prefill graphs yet"
    if envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_DISPATCH_GEMM.get():
        return "terminal MoK is incompatible with the intermediate K1 flag"
    if envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE.get():
        return "terminal MoK is incompatible with the intermediate K2 flag"
    return _terminal_concurrency_mode_error()


def _consensus_supported(local_supported: bool, device: torch.device, group) -> bool:
    flag = torch.tensor([int(local_supported)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def _accept_runtime_contract(
    reason: Optional[str],
    device: torch.device,
    group,
    *,
    strict: bool,
) -> bool:
    """Accept a native contract, optionally without a per-layer collective."""
    if strict:
        if reason is not None:
            raise RuntimeError(f"strict full-native MoK contract rejected: {reason}")
        return True
    return _consensus_supported(reason is None, device, group)


def _capacity_factor_from_global_counts(
    global_counts: torch.Tensor,
    *,
    base_rows: int,
    num_local_experts: int,
    ep_size: int,
    expert_padding: int = 256,
) -> int:
    """Return the smallest workspace factor that holds padded experts."""
    if global_counts.numel() != num_local_experts * ep_size:
        raise ValueError("global route counts do not match the expert topology")
    if base_rows <= 0:
        raise ValueError("base_rows must be positive")
    if expert_padding <= 0:
        raise ValueError("expert_padding must be positive")
    padded_counts = (
        torch.div(
            global_counts + expert_padding - 1,
            expert_padding,
            rounding_mode="floor",
        )
        * expert_padding
    )
    max_required_rows = int(
        padded_counts.view(ep_size, num_local_experts).sum(dim=1).max().item()
    )
    return max(2, (max_required_rows + base_rows - 1) // base_rows)


def _conservative_route_capacity_factor(
    *,
    base_rows: int,
    num_local_experts: int,
    ep_size: int,
    expert_padding: int,
) -> int:
    """Return a collective-free upper bound for one destination rank.

    Every EP rank contributes at most ``base_rows`` valid routes to one
    destination. Padding can add at most ``expert_padding - 1`` rows to each
    local expert. The bound therefore covers every valid route distribution,
    including the fully concentrated case, without inspecting GPU route data
    or synchronizing it back to the host on every layer.
    """
    if base_rows <= 0:
        raise ValueError("base_rows must be positive")
    if num_local_experts <= 0 or ep_size <= 0:
        raise ValueError("expert and EP counts must be positive")
    if expert_padding <= 0:
        raise ValueError("expert_padding must be positive")
    max_destination_rows = ep_size * base_rows
    max_padding_rows = num_local_experts * (expert_padding - 1)
    minimum_factor = max(
        2,
        (max_destination_rows + max_padding_rows + base_rows - 1) // base_rows,
    )
    # The scheduler owns M256-aligned metadata buffers.  Small Decode shapes
    # no longer make base_rows itself M256, so align the integer multiplier
    # instead while preserving the same worst-case row bound.
    factor_alignment = 256 // math.gcd(base_rows, 256)
    return (
        (minimum_factor + factor_alignment - 1) // factor_alignment
    ) * factor_alignment


def _route_padding_config(num_tokens: int, topk: int) -> tuple[int, int]:
    """Return the padded token count and route-gather chunk size."""
    if num_tokens <= 0 or topk <= 0:
        raise ValueError("token and top-k counts must be positive")
    if num_tokens <= 4:
        route_token_alignment = math.lcm(2, 4 // math.gcd(topk, 4))
        padded_tokens = (
            (num_tokens + route_token_alignment - 1) // route_token_alignment
        ) * route_token_alignment
        return padded_tokens, 16
    return max(256, ((num_tokens + 255) // 256) * 256), 1024


def _required_route_capacity_factor(
    topk_ids: torch.Tensor,
    *,
    num_global_experts: int,
    num_local_experts: int,
    ep_size: int,
    expert_padding: int,
    group,
) -> Optional[int]:
    """Collectively size storage for the scheduler's expert padding.

    Counting before workspace allocation is necessary for production shapes:
    lightly loaded local experts each require one aligned segment even when
    the raw routed-token count is much smaller. A fixed multiple of ``T *
    topk`` can therefore reject every small-batch DeepSeek-V4 layer.
    """
    valid = (topk_ids >= 0) & (topk_ids < num_global_experts)
    invalid = ((topk_ids < -1) | (topk_ids >= num_global_experts)).any()
    base_rows = topk_ids.numel()
    # One collective carries route counts, invalid-ID state, and enough shape
    # statistics to reject unequal T*topk before symmetric allocation.
    summary = torch.zeros(
        num_global_experts + 3,
        dtype=torch.int64,
        device=topk_ids.device,
    )
    summary[:num_global_experts] = torch.bincount(
        topk_ids[valid].to(torch.int64), minlength=num_global_experts
    )
    summary[num_global_experts] = invalid.to(torch.int64)
    summary[num_global_experts + 1] = base_rows
    summary[num_global_experts + 2] = base_rows * base_rows
    dist.all_reduce(summary, op=dist.ReduceOp.SUM, group=group)

    invalid_count = int(summary[num_global_experts].item())
    base_sum = int(summary[num_global_experts + 1].item())
    base_square_sum = int(summary[num_global_experts + 2].item())
    if invalid_count or ep_size * base_square_sum != base_sum * base_sum:
        return None
    common_base_rows = base_sum // ep_size
    return _capacity_factor_from_global_counts(
        summary[:num_global_experts],
        base_rows=common_base_rows,
        num_local_experts=num_local_experts,
        ep_size=ep_size,
        expert_padding=expert_padding,
    )


def _report_fallback(reason: str) -> None:
    if reason not in _REPORTED_FALLBACKS:
        _REPORTED_FALLBACKS.add(reason)
        logger.info("MoK full-native fallback: %s", reason)


class _PrefillGraphEntry:
    __slots__ = ("graph", "in_hidden", "in_ids", "in_weights", "out")

    def __init__(self, graph, in_hidden, in_ids, in_weights, out):
        self.graph = graph
        self.in_hidden = in_hidden
        self.in_ids = in_ids
        self.in_weights = in_weights
        self.out = out


# One graph per (layer object, padded token bucket).  All graphs share one
# memory pool so intermediate activations are reused across layers instead of
# being held 43 times; layers execute serially on one stream, which is the
# safety contract for both the shared pool and the shared workspace tensors.
_PREFILL_GRAPHS: dict[tuple[int, int], _PrefillGraphEntry] = {}
_PREFILL_GRAPH_POOL = None
_PREFILL_GRAPH_DISABLED = False


def _run_native_core(
    layer,
    workspace,
    mok_functional,
    config,
    padded_hidden: torch.Tensor,
    padded_topk_ids: torch.Tensor,
    padded_topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Graph-capturable native MoE sequence.

    Strict-contract only: no host reads, no collectives, no data-dependent
    shapes.  Every launch depends solely on tensor contents plus the fixed
    workspace geometry, which is what CUDA Graph capture requires.
    """
    from sglang.jit_kernel.dsv4 import silu_and_mul_contig_post_quant_dynamic
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    hidden_size = padded_hidden.shape[1]
    input_fp8, input_scale = sglang_per_token_group_quant_fp8(
        padded_hidden,
        128,
        column_major_scales=False,
        scale_tma_aligned=False,
        scale_ue8m0=False,
    )
    # Workspace lease: the orchestrator owns it across the whole pipeline
    # (build_schedule is the first workspace write).  Every branch below
    # runs in leased mode; release happens exactly once -- by the fused
    # epilogue's last CTA (K2 branch) or by the trailing release at the end
    # of the combine path.  Concurrent reuse of the workspace fails closed
    # in the acquire kernel (REENTRANT trap).
    mok_functional.acquire_workspace_lease(workspace)
    schedule = mok_functional.build_schedule(
        workspace,
        config,
        padded_topk_ids,
        num_local_experts=layer.num_local_experts,
        expert_padding=_ROUTE_EXPERT_PADDING,
    )
    gate_up_size = layer.w13_weight.shape[1]
    intermediate_size = layer.w2_weight.shape[2]
    if envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_DISPATCH_GEMM.get():
        # First fusion cut: the input barrier, pull dispatch, and gate/up
        # GEMM run as one persistent kernel with per-M64-tile handoff.
        capacity_rows = workspace.schedule_capacity
        gate_up = torch.empty(
            (capacity_rows, gate_up_size),
            dtype=torch.bfloat16,
            device=padded_hidden.device,
        )
        routed_x, routed_x_scale, m_indices = (
            mok_functional.dispatch_gemm_fused_fp8_block(
                workspace,
                schedule,
                input_fp8,
                input_scale,
                layer.w13_weight,
                layer.w13_weight_scale_inv,
                gate_up,
                copy_clusters=(
                    envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_COPY_CLUSTERS.get()
                ),
            )
        )
    else:
        routed_x, routed_x_scale, m_indices = mok_functional.dispatch_fp8_block(
            workspace,
            schedule,
            input_fp8,
            input_scale,
            trim_to_active_rows=False,
            prepare_combine=True,
        )
        capacity_rows = routed_x.shape[0]
        gate_up = torch.empty(
            (capacity_rows, gate_up_size),
            dtype=torch.bfloat16,
            device=padded_hidden.device,
        )
        mok_functional.grouped_gemm_fp8_block_dynamic_out(
            routed_x,
            layer.w13_weight,
            routed_x_scale,
            layer.w13_weight_scale_inv,
            m_indices,
            schedule.num_tokens,
            gate_up,
        )

    down_input = torch.empty(
        (capacity_rows, intermediate_size),
        dtype=torch.float8_e4m3fn,
        device=padded_hidden.device,
    )
    down_input_scale = torch.empty(
        (capacity_rows, intermediate_size // 128),
        dtype=torch.float32,
        device=padded_hidden.device,
    )
    silu_and_mul_contig_post_quant_dynamic(
        input=gate_up,
        output=down_input,
        output_scale=down_input_scale,
        active_tokens=schedule.num_tokens,
        quant_group_size=128,
        scale_ue8m0=False,
        transposed=False,
        swiglu_limit=layer.moe_runner_config.swiglu_limit,
        swizzle=False,
    )

    routed_y = torch.empty(
        (capacity_rows, hidden_size),
        dtype=torch.bfloat16,
        device=padded_hidden.device,
    )
    if envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE.get():
        # Symmetric second cut: the GEMM CTA finishing each M64 block last
        # pushes it to the peers, and the fused arrive replaces the separate
        # barrier kernel.  No resident communication CTAs.
        # Caller-owned output: the epilogue writes here BEFORE its last CTA
        # releases the lease, so a subsequent acquirer overwriting workspace
        # state cannot corrupt this result (safe in-graph release).
        core_out = torch.empty_like(workspace.output)
        return mok_functional.gemm_combine_fused_fp8_block(
            workspace,
            schedule,
            down_input,
            down_input_scale,
            layer.w2_weight,
            layer.w2_weight_scale_inv,
            routed_y,
            padded_topk_weights,
            release_lease=True,
            output=core_out,
        )
    mok_functional.grouped_gemm_fp8_block_dynamic_out(
        down_input,
        layer.w2_weight,
        down_input_scale,
        layer.w2_weight_scale_inv,
        m_indices,
        schedule.num_tokens,
        routed_y,
    )
    # Combine path: the result lives in workspace.output, so the lease is
    # NOT released here -- the outer boundary releases it after the caller's
    # materializing copy (see _finish_native_output).
    return mok_functional.combine_reduce_fp8_block_routes(
        workspace,
        schedule,
        routed_y,
        padded_topk_weights,
        combine_precleared=True,
    )


def _pad_eager_inputs(
    hidden_states: torch.Tensor,
    topk_output,
    *,
    padded_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize one fixed eager bucket without changing route ordering."""
    num_tokens, hidden_size = hidden_states.shape
    topk = topk_output.topk_ids.shape[1]
    if padded_tokens == num_tokens:
        return (
            hidden_states,
            topk_output.topk_ids.to(torch.int32),
            topk_output.topk_weights,
        )

    padded_hidden = hidden_states.new_zeros((padded_tokens, hidden_size))
    padded_hidden[:num_tokens].copy_(hidden_states)
    padded_topk_ids = torch.full(
        (padded_tokens, topk),
        -1,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    padded_topk_ids[:num_tokens].copy_(topk_output.topk_ids.to(torch.int32))
    padded_topk_weights = torch.zeros(
        (padded_tokens, topk),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    padded_topk_weights[:num_tokens].copy_(topk_output.topk_weights)
    return padded_hidden, padded_topk_ids, padded_topk_weights


def _run_terminal_eager(
    layer,
    workspace,
    mok_functional,
    config,
    padded_hidden: torch.Tensor,
    padded_topk_ids: torch.Tensor,
    padded_topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Run the strict caller-owned-output terminal megakernel in eager mode."""
    from sglang.srt.layers.quantization.fp8_kernel import (
        launch_sglang_per_token_group_quant_fp8_out_prevalidated,
        prewarm_sglang_per_token_group_quant_fp8_out,
    )

    output = torch.empty(
        (padded_hidden.shape[0], 4096),
        dtype=torch.bfloat16,
        device=padded_hidden.device,
    )
    prewarm_receipt = prewarm_sglang_per_token_group_quant_fp8_out(
        padded_hidden,
        workspace.x_buffer,
        workspace.x_scale_buffer,
        128,
    )
    if prewarm_receipt not in _REPORTED_TERMINAL_QUANT_PREWARM:
        _REPORTED_TERMINAL_QUANT_PREWARM.add(prewarm_receipt)
        logger.info(
            "MOK_TERMINAL_QUANT_PREWARM|backend=%s|group=%d|pdl=%s",
            prewarm_receipt[0],
            prewarm_receipt[5],
            str(prewarm_receipt[-1]).lower(),
        )

    phase = "acquire"
    try:
        mok_functional.acquire_megakernel_fp8_block_from_topk_lease(
            workspace,
            config,
            layer.w13_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight,
            layer.w2_weight_scale_inv,
            padded_topk_weights,
            padded_topk_ids,
            output,
            swiglu_limit=layer.moe_runner_config.swiglu_limit,
        )
        phase = "quant"
        launch_sglang_per_token_group_quant_fp8_out_prevalidated(
            padded_hidden,
            workspace.x_buffer,
            workspace.x_scale_buffer,
            128,
            prewarm_receipt,
        )
        phase = "terminal"
        return mok_functional.megakernel_fp8_block_from_topk_preloaded_leased(
            workspace,
            config,
            layer.w13_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight,
            layer.w2_weight_scale_inv,
            padded_topk_weights,
            padded_topk_ids,
            output,
            swiglu_limit=layer.moe_runner_config.swiglu_limit,
        )
    except BaseException as error:
        _fatal_terminal_transaction_failure(
            workspace, mok_functional, phase, error
        )


def _capture_prefill_graph(
    layer,
    workspace,
    mok_functional,
    config,
    *,
    padded_tokens: int,
    hidden_size: int,
    topk: int,
    device: torch.device,
) -> Optional[_PrefillGraphEntry]:
    """Lazily capture one per-(layer, bucket) graph; None disables graphs."""
    global _PREFILL_GRAPH_POOL, _PREFILL_GRAPH_DISABLED
    in_hidden = torch.zeros(
        (padded_tokens, hidden_size), dtype=torch.bfloat16, device=device
    )
    in_ids = torch.full(
        (padded_tokens, topk), -1, dtype=torch.int32, device=device
    )
    in_weights = torch.zeros(
        (padded_tokens, topk), dtype=torch.float32, device=device
    )
    # Flush lazy JIT/init on the exact capture shape.  This runs for real and
    # is rank-synchronous: every EP rank reaches capture for the same
    # (layer, bucket) key in the same order under the strict contract.
    try:
        _run_native_core(
            layer, workspace, mok_functional, config,
            in_hidden, in_ids, in_weights,
        )
        torch.cuda.synchronize(device)
        if not envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE.get():
            # Combine-path warmup held the lease past the core; release it
            # before capture (nothing reads workspace.output here).
            mok_functional.release_workspace_lease(workspace)
    except RuntimeError:
        _die_if_trapped(workspace, mok_functional)
        raise
    if _PREFILL_GRAPH_POOL is None:
        _PREFILL_GRAPH_POOL = torch.cuda.graph_pool_handle()
    graph = torch.cuda.CUDAGraph()
    captured = True
    try:
        with torch.cuda.graph(graph, pool=_PREFILL_GRAPH_POOL):
            out = _run_native_core(
                layer,
                workspace,
                mok_functional,
                config,
                in_hidden,
                in_ids,
                in_weights,
            )
    except Exception:
        # A trap during capture poisons the context: check the pinned record
        # BEFORE any further CUDA work (the consensus path below allocates
        # tensors and all-reduces, which the trap contract forbids).
        _die_if_trapped(workspace, mok_functional)
        captured = False
        logger.exception("MoK prefill graph capture failed on this rank")
    # Rank consensus: replaying on some ranks while others run eager would
    # unbalance the barrier arrive counts and deadlock the flag spins, so a
    # single capture failure disables graphs everywhere.  Capture records
    # without executing, so no arrives happened yet and both outcomes leave
    # every rank with a symmetric arrive history.
    group = get_tp_group().device_group
    flag = torch.tensor(
        [int(captured)], dtype=torch.int32, device=device
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    if int(flag.item()) == 0:
        _PREFILL_GRAPH_DISABLED = True
        if captured:
            graph.reset()
        logger.warning(
            "MoK prefill graphs disabled process-wide: capture failed on at "
            "least one EP rank"
        )
        return None
    return _PrefillGraphEntry(graph, in_hidden, in_ids, in_weights, out)


@torch.no_grad()
def maybe_run_mok_fp8_native(layer, hidden_states, topk_output):
    """Return native output, or ``None`` before any MoK collective on fallback."""
    group = get_tp_group().device_group
    terminal_mode = envs.SGLANG_OPT_MOK_FP8_NATIVE_TERMINAL.get()
    reason = (
        native_terminal_contract_error(layer, hidden_states, topk_output)
        if terminal_mode
        else native_runtime_contract_error(layer, hidden_states, topk_output)
    )
    if terminal_mode and reason is None:
        reason = _terminal_mode_config_error()
    if terminal_mode and reason is None:
        reason = _terminal_graph_context_error()
    if reason is None and dist.get_world_size(group) != layer.moe_ep_size:
        reason = "the MoK process group must match moe_ep_size"
    strict_contract = terminal_mode or envs.SGLANG_OPT_MOK_FP8_NATIVE_STRICT.get()
    if not _accept_runtime_contract(
        reason,
        hidden_states.device,
        group,
        strict=strict_contract,
    ):
        _report_fallback(reason or "another EP rank rejected the native contract")
        return None

    try:
        from mok import functional as mok_functional
    except ImportError as exc:
        raise RuntimeError(
            "MoK native or terminal mode requires the MoK extension"
        ) from exc
    if terminal_mode:
        required_apis = (
            "MoKConfig",
            "get_fp8_terminal_workspace",
            "acquire_megakernel_fp8_block_from_topk_lease",
            "megakernel_fp8_block_from_topk_preloaded_leased",
            "format_trap_record",
            "format_terminal_transaction_failure",
        )
    else:
        required_apis = (
            "get_fp8_route_workspace",
            "build_schedule",
            "dispatch_fp8_block",
            "grouped_gemm_fp8_block_dynamic_out",
            "combine_fp8_block",
            "combine_reduce_fp8_block_routes",
            "reduce_fp8_block_routes",
            "acquire_workspace_lease",
            "release_workspace_lease",
            "format_trap_record",
        )
        if envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_DISPATCH_GEMM.get():
            required_apis = required_apis + ("dispatch_gemm_fused_fp8_block",)
        if envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE.get():
            required_apis = required_apis + ("gemm_combine_fused_fp8_block",)
    missing = [name for name in required_apis if not hasattr(mok_functional, name)]
    if missing:
        raise RuntimeError(f"loaded MoK package lacks native APIs: {missing}")

    num_tokens, hidden_size = hidden_states.shape
    topk = topk_output.topk_ids.shape[1]
    # Decode only needs enough padding for the even-token reducer and an M16
    # route-buffer chunk.  Retain M256 token padding for larger batches until
    # their route-chunk/capacity tradeoff is measured independently.
    padded_tokens, route_chunk_bytes = _route_padding_config(num_tokens, topk)

    # SGLang's TP/EP model contract gives every rank the same padded shape,
    # and the MoK workspace validates that invariant when a shape is first
    # created. Use a worst-case route-distribution bound here instead of an
    # extra route-count AllReduce plus GPU-to-host synchronization per layer.
    capacity_factor = _conservative_route_capacity_factor(
        base_rows=padded_tokens * topk,
        num_local_experts=layer.num_local_experts,
        ep_size=layer.moe_ep_size,
        expert_padding=_ROUTE_EXPERT_PADDING,
    )

    # The conservative multiplier accounts for the scheduler's per-expert
    # alignment under the worst valid route distribution.  Decode uses the
    # smallest legal chunk; 1024 bytes keeps larger route gathers compact and
    # divides every T*topk*sizeof(int32) buffer padded to M256.
    fwd_num_comm_sms = envs.SGLANG_OPT_MOK_FP8_NATIVE_FWD_NUM_COMM_SMS.get()
    if type(fwd_num_comm_sms) is not int or fwd_num_comm_sms <= 0:
        raise RuntimeError("MoK fwd_num_comm_sms must be a positive integer")
    if terminal_mode and fwd_num_comm_sms % 2 != 0:
        raise RuntimeError(
            "terminal MoK fwd_num_comm_sms must be even for cluster2 roles"
        )
    config = mok_functional.MoKConfig(
        schedule_capacity_multiplier=capacity_factor / layer.moe_ep_size,
        all_gather_top_experts_chunk_bytes=route_chunk_bytes,
        fwd_num_comm_sms=fwd_num_comm_sms,
    )
    if terminal_mode:
        schedule_capacity = padded_tokens * topk * capacity_factor
        if schedule_capacity <= 0 or schedule_capacity % 256 != 0:
            raise RuntimeError(
                "terminal MoK schedule capacity must be positive and M256"
            )
        workspace = mok_functional.get_fp8_terminal_workspace(
            group,
            device=hidden_states.device,
            num_local_tokens=padded_tokens,
            schedule_capacity=schedule_capacity,
            num_local_experts=layer.num_local_experts,
            comm_clusters=fwd_num_comm_sms // 2,
        )
    else:
        workspace = mok_functional.get_fp8_route_workspace(
            config,
            group,
            device=hidden_states.device,
            num_local_tokens=padded_tokens,
            hidden_size=hidden_size,
            topk=topk,
            num_local_experts=layer.num_local_experts,
        )
    _register_trap_watchdog(workspace, mok_functional)

    _report_active(
        layer,
        num_tokens,
        padded_tokens,
        topk,
        workspace,
        strict_contract,
        terminal_mode=terminal_mode,
    )

    if terminal_mode:
        padded_hidden, padded_topk_ids, padded_topk_weights = _pad_eager_inputs(
            hidden_states,
            topk_output,
            padded_tokens=padded_tokens,
        )
        try:
            output = _run_terminal_eager(
                layer,
                workspace,
                mok_functional,
                config,
                padded_hidden,
                padded_topk_ids,
                padded_topk_weights,
            )
            # The terminal kernel wrote caller-owned storage before releasing
            # its workspace lease.  A leading-row view remains caller-owned;
            # no clone or Python release is needed.
            return output[:num_tokens]
        except RuntimeError:
            _die_if_trapped(workspace, mok_functional)
            raise

    use_graph = (
        envs.SGLANG_OPT_MOK_FP8_NATIVE_PREFILL_GRAPH.get()
        and strict_contract
        and not _PREFILL_GRAPH_DISABLED
        and padded_tokens >= 256
        and not torch.cuda.is_current_stream_capturing()
    )
    if use_graph:
        key = (id(layer), padded_tokens)
        entry = _PREFILL_GRAPHS.get(key)
        if entry is None:
            entry = _capture_prefill_graph(
                layer,
                workspace,
                mok_functional,
                config,
                padded_tokens=padded_tokens,
                hidden_size=hidden_size,
                topk=topk,
                device=hidden_states.device,
            )
            if entry is not None:
                _PREFILL_GRAPHS[key] = entry
        if entry is not None:
            entry.in_hidden[:num_tokens].copy_(hidden_states)
            entry.in_ids[:num_tokens].copy_(topk_output.topk_ids)
            entry.in_weights[:num_tokens].copy_(topk_output.topk_weights)
            if padded_tokens > num_tokens:
                entry.in_hidden[num_tokens:].zero_()
                entry.in_ids[num_tokens:].fill_(-1)
                entry.in_weights[num_tokens:].zero_()
            try:
                entry.graph.replay()
                if envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE.get():
                    # entry.out is the caller-owned K2 output tensor.
                    return entry.out[:num_tokens].contiguous()
                # Combine-path graphs record the acquire but no release,
                # and entry.out aliases workspace.output: force a
                # new-storage copy (contiguous() would alias), then
                # release outside the graph.
                result = entry.out[:num_tokens].clone(
                    memory_format=torch.contiguous_format
                )
                mok_functional.release_workspace_lease(workspace)
                return result
            except RuntimeError:
                _die_if_trapped(workspace, mok_functional)
                raise

    padded_hidden, padded_topk_ids, padded_topk_weights = _pad_eager_inputs(
        hidden_states,
        topk_output,
        padded_tokens=padded_tokens,
    )

    try:
        output = _run_native_core(
            layer,
            workspace,
            mok_functional,
            config,
            padded_hidden,
            padded_topk_ids,
            padded_topk_weights,
        )
        if envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE.get():
            # K2 path already wrote a caller-owned tensor; slicing it is safe.
            return output[:num_tokens]
        # Combine path: the result lives in workspace.output and
        # .contiguous() on a contiguous slice is a NO-OP alias -- force a
        # new-storage copy, then release (stream-ordered after the copy).
        result = output[:num_tokens].clone(
            memory_format=torch.contiguous_format
        )
        mok_functional.release_workspace_lease(workspace)
        return result
    except RuntimeError:
        _die_if_trapped(workspace, mok_functional)
        raise


def _report_active(
    layer,
    num_tokens,
    padded_tokens,
    topk,
    workspace,
    strict_contract,
    *,
    terminal_mode: bool = False,
) -> None:
    global _REPORTED_ACTIVE
    if terminal_mode:
        layer_id = layer.layer_id
        if layer_id in _REPORTED_TERMINAL_LAYER_IDS:
            return
        _REPORTED_TERMINAL_LAYER_IDS.add(layer_id)
        terminal_graph = _TERMINAL_DECODE_GRAPH_ACTIVE.get()
        logger.info(
            "MoK full-native FP8 active: layer_id=%s class=%s "
            "deprecate_flag=%s T=%d padded=%d topk=%d E_local=%d "
            "capacity=%d device_active_rows=true expert_padding=%d "
            "fwd_num_comm_sms=%d comm_clusters=%d compute_clusters=%d "
            "strict_contract=%s terminal=true terminal_graph=%s eager_only=%s "
            "prefill_graph=False fused_k1=False fused_k2=False",
            layer_id,
            type(layer).__name__,
            str(bool(getattr(layer, "deprecate_flag", False))).lower(),
            num_tokens,
            padded_tokens,
            topk,
            layer.num_local_experts,
            workspace.schedule_capacity,
            _ROUTE_EXPERT_PADDING,
            workspace.comm_clusters * 2,
            workspace.comm_clusters,
            workspace.compute_clusters,
            strict_contract,
            str(terminal_graph).lower(),
            str(not terminal_graph).lower(),
        )
        return

    if not _REPORTED_ACTIVE:
        _REPORTED_ACTIVE = True
        logger.info(
            "MoK full-native FP8 active: layer=%s T=%d padded_T=%d "
            "topk=%d E_local=%d capacity=%d device_active_rows=true "
            "expert_padding=%d strict_contract=%s terminal=%s "
            "eager_only=%s prefill_graph=%s "
            "fused_k1=%s fused_k2=%s",
            layer.layer_id,
            num_tokens,
            padded_tokens,
            topk,
            layer.num_local_experts,
            workspace.schedule_capacity,
            _ROUTE_EXPERT_PADDING,
            strict_contract,
            terminal_mode,
            terminal_mode,
            False
            if terminal_mode
            else envs.SGLANG_OPT_MOK_FP8_NATIVE_PREFILL_GRAPH.get(),
            False
            if terminal_mode
            else envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_DISPATCH_GEMM.get(),
            False
            if terminal_mode
            else envs.SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE.get(),
        )
