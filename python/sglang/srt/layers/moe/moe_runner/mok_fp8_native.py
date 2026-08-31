"""Correctness-first full-native MoK FP8 routed-expert path for SM90."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import (
    get_is_extend_in_batch,
    get_max_sequence_length,
)

logger = logging.getLogger(__name__)


_HIT_COUNTS: dict = {}


def _note_path_hit(path: str, *, mode: str, num_tokens: Optional[int]):
    """Env-gated canary counter for MoE path-coexistence evidence.

    _report_active and the warp first-hit log fire once per process, so
    their absence proves nothing about later batches; behind
    SGLANG_MOE_PATH_HIT_COUNTERS every **Python-path invocation** is
    counted by (path, mode, token-bucket) and a line is emitted on the
    first hit and every 500th. CUDA-graph replay does not re-enter Python,
    so under graph the decode counts reflect capture-time and eager-path
    invocations only. Returns the emitted line, or None."""
    if not envs.SGLANG_MOE_PATH_HIT_COUNTERS.get():
        return None
    if num_tokens is None:
        bucket = "na"
    else:
        bucket = "ge256" if num_tokens >= 256 else "lt256"
    key = (path, mode, bucket)
    count = _HIT_COUNTS.get(key, 0) + 1
    _HIT_COUNTS[key] = count
    if count == 1 or count % 500 == 0:
        line = f"MOE_PATH_HIT path={path} mode={mode} bucket={bucket} count={count}"
        logger.info(line)
        return line
    return None


_TRAP_WATCHDOG_LOCK = threading.Lock()
_TRAP_WATCHDOG_ENTRIES: list = []
_TRAP_WATCHDOG_STARTED = False


def _trap_watchdog_loop() -> None:
    """Async production endpoint of the MoK trap contract: kernel launches
    and graph replays are asynchronous, so a device trap may never surface
    as a RuntimeError inside the local try/except boundaries.  This daemon
    thread polls the host-mapped pinned records with pure CPU reads (never
    a CUDA call) and performs the log-and-exit contract the moment any
    record turns non-zero."""
    while True:
        for workspace, mok_functional in list(_TRAP_WATCHDOG_ENTRIES):
            record = mok_functional.format_trap_record(workspace)
            if record is not None:
                logger.error(record)
                logging.shutdown()
                os._exit(70)
        time.sleep(0.05)


def _register_trap_watchdog(workspace, mok_functional) -> None:
    global _TRAP_WATCHDOG_STARTED
    with _TRAP_WATCHDOG_LOCK:
        if not any(w is workspace for w, _ in _TRAP_WATCHDOG_ENTRIES):
            _TRAP_WATCHDOG_ENTRIES.append((workspace, mok_functional))
        if not _TRAP_WATCHDOG_STARTED:
            threading.Thread(
                target=_trap_watchdog_loop,
                name="mok-trap-watchdog",
                daemon=True,
            ).start()
            _TRAP_WATCHDOG_STARTED = True


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
_REPORTED_FALLBACKS: set[str] = set()
_REPORTED_ACTIVE = False
_ROUTE_EXPERT_PADDING = 64
_WORKSPACE_GEOMETRY_LOCK = threading.Lock()
_WORKSPACE_GEOMETRIES: set[tuple] = set()


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
    if getattr(layer, "_dwdp_bound", False):
        return "DWDP prefetch integration is unsupported"
    if getattr(layer, "use_triton_kernels", False):
        return "transposed Triton expert weights are unsupported"
    # gate_up_interleaved defaults True on FusedMoE but does not describe the
    # w13 byte layout on this stack: _load_w13 places [gate; up] halves
    # unconditionally for split w1/w3 checkpoint shards (DeepSeek-V4's case),
    # and the deepgemm silu stage consumes halves. Interleaved bytes can only
    # arrive via weight_loader_fused checkpoints, which V4 does not use, so a
    # blanket flag check here mis-rejects the whole stack (seen live on
    # 2026-08-25: strict raise on the first prefill batch).
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


def _route_padding_config(
    num_tokens: int, topk: int, *, prefill_pow2_bucket: bool = False
) -> tuple[int, int]:
    """Return the padded token count and route-gather chunk size."""
    if num_tokens <= 0 or topk <= 0:
        raise ValueError("token and top-k counts must be positive")
    if num_tokens <= 4:
        route_token_alignment = math.lcm(2, 4 // math.gcd(topk, 4))
        padded_tokens = (
            (num_tokens + route_token_alignment - 1) // route_token_alignment
        ) * route_token_alignment
        return padded_tokens, 16
    padded_tokens = max(256, ((num_tokens + 255) // 256) * 256)
    if prefill_pow2_bucket:
        padded_tokens = 1 << (padded_tokens - 1).bit_length()
    return padded_tokens, 1024


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


def _admit_workspace_geometry(
    *,
    group,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
    num_local_experts: int,
    schedule_capacity_factor: int,
) -> bool:
    """Admit a bounded set of extension workspace geometries.

    The extension caches one workspace per geometry and does not evict.  The
    previous integration cleared every cached workspace synchronously from
    the forward hot path once the count reached the cap.  Besides clearing on
    cache hits, that policy inserted collective barriers, device-wide
    synchronization, allocator release, and workspace rebuilds into timed
    requests.

    Keep already admitted geometries hot.  Once the per-process cap is full,
    reject only unseen geometries so the caller can fall back to stock DeepEP
    before any MoK collective or allocation.  Every EP rank observes the same
    model batch shape, so admission remains rank-symmetric.  A non-positive
    cap explicitly restores the extension's unbounded behavior.
    """
    cap = envs.SGLANG_OPT_MOK_WORKSPACE_CACHE_CAP.get()
    if cap <= 0:
        return True

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    key = (
        group.group_name,
        device_index,
        num_local_tokens,
        hidden_size,
        topk,
        num_local_experts,
        schedule_capacity_factor,
    )
    with _WORKSPACE_GEOMETRY_LOCK:
        if key in _WORKSPACE_GEOMETRIES:
            return True
        if len(_WORKSPACE_GEOMETRIES) >= cap:
            return False
        _WORKSPACE_GEOMETRIES.add(key)
        logger.info(
            "MoK workspace geometry admitted: count=%d cap=%d tokens=%d "
            "capacity_factor=%d",
            len(_WORKSPACE_GEOMETRIES),
            cap,
            num_local_tokens,
            schedule_capacity_factor,
        )
        return True


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
    from sglang.kernels.ops.attention.dsv4 import (
        silu_and_mul_contig_post_quant_dynamic,
    )
    # v0.5.17 moved this kernel from srt.layers.quantization.fp8_kernel to
    # the sglang.kernels tree (same name and keyword surface).
    from sglang.kernels.ops.quantization.fp8_kernel import (
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
    # The split orchestrator owns the workspace across dispatch, both GEMMs,
    # and combine.  Release happens only after the caller-owned output copy.
    # Concurrent reuse fails closed in the acquire kernel (REENTRANT trap).
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
        # Warmup held the lease past the core; release it before capture
        # because nothing consumes workspace.output on this path.
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
def maybe_run_mok_fp8_native(
    layer, hidden_states, topk_output, *, pre_quant_input=None
):
    """Return native output, or ``None`` before any MoK collective on fallback."""
    # Policy gate, not a contract error: when armed (min_tokens > 0) the
    # native path only takes extend batches of at least min_tokens tokens.
    # get_is_extend_in_batch() is the same predicate DeepEP auto uses to
    # pick normal vs low-latency dispatch, and graph runners force it to
    # False around capture, so gated-off batches (all decode, short extend)
    # fall through to the stock DeepEP path -- where the warp masked kernel
    # may take decode -- before any collective or strict-contract check.
    min_tokens = envs.SGLANG_OPT_MOK_MIN_TOKENS.get()
    if min_tokens > 0 and (
        not get_is_extend_in_batch() or hidden_states.shape[0] < min_tokens
    ):
        return None
    max_tokens = envs.SGLANG_OPT_MOK_MAX_TOKENS.get()
    if max_tokens > 0 and hidden_states.shape[0] > max_tokens:
        _report_fallback(
            f"batch tokens exceed SGLANG_OPT_MOK_MAX_TOKENS={max_tokens}"
        )
        return None
    max_sequence_tokens = envs.SGLANG_OPT_MOK_MAX_SEQUENCE_TOKENS.get()
    if (
        max_sequence_tokens > 0
        and get_max_sequence_length() > max_sequence_tokens
    ):
        _report_fallback(
            "request length exceeds "
            f"SGLANG_OPT_MOK_MAX_SEQUENCE_TOKENS={max_sequence_tokens}"
        )
        return None
    group = get_tp_group().device_group
    reason = (
        "pre-quantized MoE input is unsupported"
        if pre_quant_input is not None
        else native_runtime_contract_error(layer, hidden_states, topk_output)
    )
    if reason is None and dist.get_world_size(group) != layer.moe_ep_size:
        reason = "the MoK process group must match moe_ep_size"
    strict_contract = envs.SGLANG_OPT_MOK_FP8_NATIVE_STRICT.get()
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
            "SGLANG_OPT_USE_MOK_FP8_NATIVE requires the MoK extension"
        ) from exc
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
    missing = [name for name in required_apis if not hasattr(mok_functional, name)]
    if missing:
        raise RuntimeError(f"loaded MoK package lacks native APIs: {missing}")

    num_tokens, hidden_size = hidden_states.shape
    topk = topk_output.topk_ids.shape[1]
    # Decode only needs enough padding for the even-token reducer and an M16
    # route-buffer chunk.  Retain M256 token padding for larger batches until
    # their route-chunk/capacity tradeoff is measured independently.
    padded_tokens, route_chunk_bytes = _route_padding_config(
        num_tokens,
        topk,
        prefill_pow2_bucket=(
            envs.SGLANG_OPT_MOK_PREFILL_POW2_BUCKET.get()
            and get_is_extend_in_batch()
        ),
    )

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
    config = mok_functional.MoKConfig(
        schedule_capacity_multiplier=capacity_factor / layer.moe_ep_size,
        all_gather_top_experts_chunk_bytes=route_chunk_bytes,
    )
    if not _admit_workspace_geometry(
        group=group,
        device=hidden_states.device,
        num_local_tokens=padded_tokens,
        hidden_size=hidden_size,
        topk=topk,
        num_local_experts=layer.num_local_experts,
        schedule_capacity_factor=capacity_factor,
    ):
        _report_fallback(
            "workspace geometry cap reached: "
            f"tokens={padded_tokens} capacity_factor={capacity_factor}; "
            "unseen geometry uses stock DeepEP"
        )
        return None
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

    _report_active(layer, num_tokens, padded_tokens, topk, workspace, strict_contract)
    _note_path_hit(
        "mok",
        mode="extend" if get_is_extend_in_batch() else "decode",
        num_tokens=num_tokens,
    )

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
                # Graphs record the acquire but not the release, and entry.out
                # aliases workspace.output. Force a new-storage copy before
                # releasing outside the graph.
                result = entry.out[:num_tokens].clone(
                    memory_format=torch.contiguous_format
                )
                mok_functional.release_workspace_lease(workspace)
                return result
            except RuntimeError:
                _die_if_trapped(workspace, mok_functional)
                raise

    if padded_tokens == num_tokens:
        padded_hidden = hidden_states
        padded_topk_ids = topk_output.topk_ids.to(torch.int32)
        padded_topk_weights = topk_output.topk_weights
    else:
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
        # The result lives in workspace.output and .contiguous() on a
        # contiguous slice is a no-op alias. Force a new-storage copy, then
        # release (stream-ordered after the copy).
        result = output[:num_tokens].clone(
            memory_format=torch.contiguous_format
        )
        mok_functional.release_workspace_lease(workspace)
        return result
    except RuntimeError:
        _die_if_trapped(workspace, mok_functional)
        raise


def _report_active(
    layer, num_tokens, padded_tokens, topk, workspace, strict_contract
) -> None:
    global _REPORTED_ACTIVE
    if not _REPORTED_ACTIVE:
        _REPORTED_ACTIVE = True
        logger.info(
            "MoK full-native FP8 active: layer=%s T=%d padded_T=%d "
            "topk=%d E_local=%d capacity=%d device_active_rows=true "
            "expert_padding=%d strict_contract=%s prefill_graph=%s",
            layer.layer_id,
            num_tokens,
            padded_tokens,
            topk,
            layer.num_local_experts,
            workspace.schedule_capacity,
            _ROUTE_EXPERT_PADDING,
            strict_contract,
            envs.SGLANG_OPT_MOK_FP8_NATIVE_PREFILL_GRAPH.get(),
        )
