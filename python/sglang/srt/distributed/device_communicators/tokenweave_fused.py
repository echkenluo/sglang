# TokenWeave-style fused multimem AllReduce+RMSNorm backend (ladder4 stage A).
#
# Wraps the vendored fused_rs_ln_ag_cta CUDA op (multimem ReduceScatter +
# residual add + RMSNorm + AllGather in ONE launch) behind the same
# optional-communicator shape as TorchSymmMemCommunicator: capability probing
# at init, a cheap should_engage() guard per call, graceful disable on any
# missing prerequisite, sticky disable on the first runtime failure.
#
# Everything is env-gated by SGLANG_ENABLE_TOKENWEAVE_FUSION=1 (default off):
# with the flag unset __init__ returns a disabled stub that imports and
# allocates NOTHING, so unset-flag behavior is bit-identical to before.
#
# The kernel sources are NOT vendored into sglang: the tokenweave-line repo's
# launcher.py (which cpp_extension.load()s vendor/csrc with
# is_python_module=False into the TORCH_EXTENSIONS_DIR warm cache) is loaded
# from SGLANG_TOKENWEAVE_KERNEL_DIR at init, only when the flag is on.
#
# WORKSPACE LIFETIME (torch 2.9.1 cuMulticastUnbind destructor trap): freeing
# a rendezvous'd (multicast-bound) symm-mem tensor mid-run runs
# cuMulticastUnbind inside ~AllocationRef (CUDASymmetricMemory.cu:71), which
# fails with "invalid argument" and std::terminate's every rank -- the
# 2026-07-03 first-H20-run crash. Therefore ONE symm workspace is allocated
# per process (lazily, on the first eligible call, because hidden_size is
# unknown at engine init), kept alive in the module-global _KEEPALIVE list,
# and never freed; process exit reclaims it.
#
# STAGE B: CUDA-GRAPH COMPATIBILITY (V2 probe green on the H20: multimem
# kernel + CAS handshake + all-gather all record/replay correctly, and the
# graph swallows ~53% of the eager call). What this file guarantees:
#   1. PRE-CAPTURE INIT: workspace rendezvous / extension build / hidden lock
#      happen on the FIRST ENGAGED CALL, and sglang's graph runner runs two
#      eager, TP-barrier-synchronized warmup forwards per batch size BEFORE
#      recording (cuda_graph_runner.capture_one_batch_size: the
#      `for _ in range(2): synchronize(); tp_group.barrier(); run_once()`
#      loop precedes _capture_graph). The first warmup therefore performs the
#      lazy init eagerly and in rank lockstep. Belt: _ensure_workspace
#      declines (returns False -> None -> unfused fallback; rank-uniform
#      because capture is lockstep) if it is ever reached with stream capture
#      in progress -- rendezvous/allocation must never be recorded.
#   2. FROZEN ENGAGEMENT: every should_engage guard reads static per-graph
#      facts (tensor shapes/dtypes are fixed by the captured batch size --
#      tokens = bs * num_tokens_per_bs is a capture-time constant; env values
#      are latched at init; the disabled bit is frozen after init). The
#      verdict at capture time equals the verdict at every replay; no guard
#      reads tensor DATA.
#   3. ALLOC-FREE, SIDE-EFFECT-FREE HOT PATH: between should_engage and
#      return, the fused call performs only views/slices, in-place
#      copy_/zero_ on preallocated buffers, the kernel launch, and a
#      preallocated-buffer all-gather. No torch.empty/clone, no python state
#      mutation (the one-time engaged log lives in workspace init, which
#      never runs under capture).
#   4. RANK-UNIFORM INIT VERDICT: workspace init all-reduces its success
#      across ranks (phase 1 local allocation, phase 2 collective rendezvous
#      + capability checks), so an init failure can never leave the sticky
#      disable half-set across ranks. Post-init per-call failures are
#      rank-uniform by construction (all guards are shape-static).
#   5. COLLECTIVE BACKEND FOR THE RESIDUAL AG: per sglang's graph-mode table
#      (parallel_state.GroupCoordinator.graph_capture: torch.distributed is
#      eager-only, pynccl is the in-graph path), the all-gather uses pynccl
#      when it is enabled -- graph_capture enables it around BOTH the eager
#      warmups and the recording, so the warmups exercise exactly the branch
#      the graph records -- and torch.distributed otherwise. Same idiom as
#      GroupCoordinator._all_reduce_in_place.

import importlib.util
import logging
import os
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_cuda

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    torch_symm_mem_available = is_cuda()
except ImportError:
    torch_symm_mem_available = False


logger = logging.getLogger(__name__)

# Where the tokenweave-line kernel prototype lives (launcher.py + vendor/csrc).
# Override with SGLANG_TOKENWEAVE_KERNEL_DIR on boxes with different mounts.
_DEFAULT_KERNEL_DIR = (
    "/home/luocc4/workspace/lenovo-research/tokenweave-line/kernels/fused_ar_rmsnorm"
)

# Op-name resolution, in priority order: the vendor slice's own bindings
# namespace first, then upstream TokenWeave's. Mirrors
# test_numerics.py FUSED_NAMESPACES / FUSED_OP_NAME exactly.
_FUSED_NAMESPACES = ["_fused_ar_rmsnorm", "_C"]
_FUSED_OP_NAME = "fused_rs_ln_ag_cta"

# NVSwitch multimem world sizes stage A supports (ladder4 design 02 section
# 2.2 guard 2). sm90 multimem also allows 6, but the shard math and the
# validated timing/numeric legs cover 4 and 8 only.
_SUPPORTED_WORLD_SIZES = (4, 8)

_TRUTHY = ("1", "true", "yes", "on")

# Module-global keep-alive for (symm buffer, rendezvous handle) pairs: the
# multicast-bound allocation must survive even if the communicator (or its
# GroupCoordinator) is dropped -- see WORKSPACE LIFETIME in the header.
_KEEPALIVE: List[Any] = []

_FUSION_FLAG_CACHE: Optional[bool] = None


def _tokenweave_fusion_enabled() -> bool:
    """Cheap, cached read of SGLANG_ENABLE_TOKENWEAVE_FUSION.

    The single flag helper the upstream call-site gates import
    (layers/communicator.py and layers/layernorm.py) so the generic fused
    dispatcher becomes reachable on CUDA when -- and only when -- the flag is
    on. Deliberately module-level and torch-op free: importing this module
    never touches the kernel dir (the launcher/extension load happens inside
    TokenWeaveFusedCommunicator.__init__ only).

    The value is latched on first call. That matches the backend itself: the
    communicator is constructed (or not) from the same env var at engine
    init, so flipping the env after process start cannot enable or disable
    the backend anyway -- the only supported state is the process-start one.
    """
    global _FUSION_FLAG_CACHE
    if _FUSION_FLAG_CACHE is None:
        _FUSION_FLAG_CACHE = (
            os.environ.get("SGLANG_ENABLE_TOKENWEAVE_FUSION", "0") == "1"
        )
    return _FUSION_FLAG_CACHE


class TokenWeaveFusedCommunicator:
    """Fused multimem AllReduce+RMSNorm over one persistent symm workspace.

    Engagement contract (stage A, eager decode):
      - inputs: input_ is the TP-partial hidden [tokens, hidden] (bf16, CUDA);
        residual and weight are TP-REPLICATED (identical bytes on every rank)
        -- the kernel reduces only the hidden and consumes residual/weight as
        passed. Stage A trusts the non-scattered call site for replication and
        checks shapes only: a real per-call cross-rank equality check would
        cost a collective per layer.
      - returns (norm_out, residual_out) with the SAME in-place semantics as
        the native allreduce + fused_add_rmsnorm path the caller otherwise
        runs (layernorm.py RMSNorm.forward_cuda): the caller's input_ and
        residual tensors are mutated in place and returned, so the call site
        cannot tell the backends apart. Returns None (after sticky-disabling)
        on the first runtime failure so the caller falls back.

    Stage B: the engaged call path is CUDA-graph capturable end to end --
    see "STAGE B: CUDA-GRAPH COMPATIBILITY" in the module header for the
    five guarantees (pre-capture init, frozen engagement, alloc-free hot
    path, rank-uniform init verdict, graph-safe collective backend).
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        device_group: Optional[ProcessGroup] = None,
        pynccl_comm: Optional[Any] = None,
    ):
        """
        Args:
            group: Process group used for symm-mem rendezvous naming and
                rank/world-size queries (the GroupCoordinator's cpu_group,
                same as TorchSymmMemCommunicator).
            device: Target CUDA device.
            device_group: NCCL process group of the same ranks, used for the
                stage-A residual shard all-gather (the kernel writes the
                pre-norm sum only into this rank's token shard; non-token-
                split callers need the full replicated residual stream).
            pynccl_comm: The GroupCoordinator's PyNcclCommunicator, used for
                the residual all-gather whenever it is enabled -- i.e. inside
                graph_capture (warmups and recording), where raw
                torch.distributed collectives are off-limits per sglang's
                graph-mode table (stage B guarantee 5 in the module header).
        """
        self.disabled = True
        # Round-2 native-mode attrs must exist even on the inert stub: the
        # dispatcher and the RowParallelLinear producer hook getattr them
        # before any engagement check.
        self.native_enabled = False
        self.stagea_enabled = True
        self._native_armed: Optional[Tuple[int, int]] = None  # (tokens, half)
        self._native_half = 0
        self._native_arms = 0
        self._native_claims = 0

        if os.environ.get("SGLANG_ENABLE_TOKENWEAVE_FUSION", "0") != "1":
            # Default off: stay an inert stub. No kernel-dir import, no
            # allocation, no logging.
            return

        if not torch_symm_mem_available:
            logger.warning(
                "[TOKENWEAVE] disabled: torch symmetric memory not available"
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        if device.type != "cuda":
            logger.warning("[TOKENWEAVE] disabled: non-CUDA device %s", device)
            return

        self.dtype = torch.bfloat16
        self.device = device
        self.group = group
        self.device_group = device_group
        self.pynccl_comm = pynccl_comm
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)

        if torch.cuda.get_device_capability(device)[0] < 9:
            logger.warning(
                "[TOKENWEAVE] disabled: device capability < 9 (multimem "
                "ld_reduce/st requires sm90+)"
            )
            return
        if self.world_size not in _SUPPORTED_WORLD_SIZES:
            logger.warning(
                "[TOKENWEAVE] disabled: world size %d not in %s",
                self.world_size,
                _SUPPORTED_WORLD_SIZES,
            )
            return
        if self.device_group is None:
            logger.warning(
                "[TOKENWEAVE] disabled: no device group for the residual "
                "shard all-gather"
            )
            return

        kernel_dir = os.environ.get(
            "SGLANG_TOKENWEAVE_KERNEL_DIR", _DEFAULT_KERNEL_DIR
        )
        try:
            self._op = self._build_and_resolve_op(kernel_dir)
        except Exception as e:
            logger.warning(
                "[TOKENWEAVE] disabled: kernel build/registration failed "
                "(kernel_dir=%s): %s",
                kernel_dir,
                e,
                exc_info=True,
            )
            return

        self.max_tokens = int(os.environ.get("SGLANG_TOKENWEAVE_MAX_TOKENS", "4096"))
        # The kernel's reduce-scatter shard math needs world_size | rows, so
        # per-call token counts are padded up and the workspace is sized for
        # the padded maximum.
        self.max_tokens_pad = (
            -(-self.max_tokens // self.world_size) * self.world_size
        )
        max_ctas_env = os.environ.get("SGLANG_TOKENWEAVE_MAX_CTAS", "").strip()
        self._max_ctas_override = int(max_ctas_env) if max_ctas_env else None

        # Round-2 native mode (plan 2026-07-31-h20-comm-reaudit-native-fusion):
        # the producer RowParallelLinear writes its partial sums STRAIGHT into
        # this communicator's region (provide_gemm_out) and the fused site
        # consumes them in place -- no copy-in and no output copy-out. A
        # ping-pong pair of regions keeps a returned output view alive while
        # the NEXT site's gemm writes the other half (a view is consumed by
        # the immediately following matmul/norm, so two halves suffice).
        # Residual handling stays stage-A (shard all-gather) until N2
        # composes with the scattered flow. Op-level numerics + timing for
        # this exact path: tokenweave-line/kernels/native_fused (gate
        # NATIVEFUSED-NUM-OK 07-31; wins vs strongest incumbent at M>=4096).
        self.native_enabled = (
            os.environ.get("SGLANG_TOKENWEAVE_NATIVE", "0") == "1"
        )
        # Data-driven engagement floor (roundtwo A/B vs strongest incumbent,
        # K=1024 N=5120 TP8: -29us at M=1024, ~par at 2560, +89us at 4096,
        # +234us at 8192): below this token count the native path LOSES to
        # the incumbent, so it declines and the call falls back. Note the
        # inherited SGLANG_TOKENWEAVE_MAX_TOKENS is the workspace-sizing
        # UPPER cap (raise it to cover prefill chunks, e.g. 8192).
        self._native_min_tokens = int(
            os.environ.get("SGLANG_TOKENWEAVE_NATIVE_MIN_TOKENS", "2560")
        )
        # Single-variable A/B isolation: SGLANG_TOKENWEAVE_STAGEA=0 turns the
        # stage-A (decode/small-batch copy-in) engagement OFF while native
        # stays on -- the only behavioral delta vs baseline is then the
        # native path on prefill chunks >= the floor. Stage-A's ULP-level
        # reduction differences flip greedy argmax ties on short prompts
        # (07-31 gate: 2/3 probes differed with coherent continuations),
        # which is why the July quality gate is teacher-forced logprob, not
        # token identity.
        self.stagea_enabled = (
            os.environ.get("SGLANG_TOKENWEAVE_STAGEA", "1") != "0"
        )

        # hidden_size is unknown at engine init; the workspace is allocated
        # once, on the first eligible call (_ensure_workspace), never freed.
        # With CUDA graphs, that first call is one of the runner's eager
        # pre-capture warmup forwards (stage B guarantee 1).
        self.hidden: Optional[int] = None
        self._region_buf: Optional[torch.Tensor] = None  # symm-mem, flat
        self._handle: Optional[Any] = None
        self._mc_ptr: int = 0
        self._signal_pads: int = 0
        self._res_shard: Optional[torch.Tensor] = None
        self._res_stage: Optional[torch.Tensor] = None

        self.disabled = False
        logger.info(
            "[TOKENWEAVE] enabled: world_size=%d rank=%d max_tokens=%d(pad %d) "
            "max_ctas=%s kernel_dir=%s (symm workspace deferred to first "
            "eligible call)",
            self.world_size,
            self.rank,
            self.max_tokens,
            self.max_tokens_pad,
            self._max_ctas_override if self._max_ctas_override else "auto(8/16)",
            kernel_dir,
        )

    @staticmethod
    def _build_and_resolve_op(kernel_dir: str):
        """Load launcher.py from the kernel dir and build/resolve the op.

        launcher.build_extension() runs cpp_extension.load() with
        is_python_module=False against a TORCH_EXTENSIONS_DIR warm cache and
        resolves vendor/csrc relative to launcher.py's own location, so
        nothing is copied into the sglang tree.
        """
        launcher_path = os.path.join(kernel_dir, "launcher.py")
        if not os.path.isfile(launcher_path):
            raise FileNotFoundError(
                f"launcher.py not found under SGLANG_TOKENWEAVE_KERNEL_DIR="
                f"{kernel_dir}"
            )
        spec = importlib.util.spec_from_file_location(
            "_sglang_tokenweave_launcher", launcher_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load launcher module from {launcher_path}")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        verbose = (
            os.environ.get("LAUNCHER_VERBOSE", "").strip().lower() in _TRUTHY
        )
        launcher.build_extension(verbose=verbose)
        launcher.assert_op_registered()
        for ns_name in _FUSED_NAMESPACES:
            ns = getattr(torch.ops, ns_name, None)
            if ns is not None and getattr(ns, _FUSED_OP_NAME, None) is not None:
                return getattr(ns, _FUSED_OP_NAME)
        raise RuntimeError(
            f"fused op {_FUSED_OP_NAME} not registered after build "
            f"(tried namespaces {_FUSED_NAMESPACES})"
        )

    def should_engage(
        self,
        input_: torch.Tensor,
        residual: Optional[torch.Tensor],
        weight: torch.Tensor,
    ) -> bool:
        """Per-call eligibility. Guard order: cheapest first. Every check is
        rank-uniform (same shapes/dtypes/env on all TP ranks) so all ranks
        take the same branch -- required for collective safety."""
        # 1. Flag / capability / sticky-runtime disable, all folded into one
        #    bit set at init or on first failure.
        if self.disabled:
            return False
        # 1b. Stage-A engagement can be disabled independently of native
        #     (SGLANG_TOKENWEAVE_STAGEA=0) for single-variable serving A/Bs.
        if not self.stagea_enabled:
            return False
        # 2. The fused op needs the residual stream (it is the AR+add+norm).
        if residual is None:
            return False
        # 3. [tokens, hidden] only.
        if input_.dim() != 2:
            return False
        num_tokens, hidden = input_.shape
        # 4. Stage-A decode/small-batch gate: forward mode is not visible at
        #    the fused_allreduce_rmsnorm dispatcher (tensor args only), so
        #    tokens <= max_tokens stands in for "decode only" -- decode and
        #    small extend batches pass, large prefill falls back (ladder4
        #    design 02 section 2.2 guard 4 accepts this for stage A).
        if num_tokens <= 0 or num_tokens > self.max_tokens:
            return False
        # 5. Kernel is bf16 width-8 vectorized only (16B-aligned rows).
        if hidden % 8 != 0:
            return False
        # 6. Workspace is sized for one hidden size, locked on first call.
        if self.hidden is not None and hidden != self.hidden:
            return False
        # 7. bf16 everywhere (buffer, kernel dispatch, multimem 16B ops).
        if (
            input_.dtype != self.dtype
            or residual.dtype != self.dtype
            or weight.dtype != self.dtype
        ):
            return False
        if not input_.is_cuda:
            return False
        # 8. Cheap stand-in for the TP-replicated residual contract: stage A
        #    trusts the non-scattered call site to pass the same residual on
        #    every rank (a per-call cross-rank equality check would cost a
        #    collective); shape agreement is all we assert here.
        if residual.shape != input_.shape:
            return False
        if weight.dim() != 1 or weight.shape[0] != hidden:
            return False
        return True

    def fused_ar_rmsnorm(
        self,
        input_: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Run the fused op; on any failure sticky-disable and return None so
        the caller falls back to the unfused path (same fail-safe shape as
        fp8.py's _FLUX_FP8_DISABLED)."""
        try:
            if not self._ensure_workspace(input_.shape[-1]):
                # Not ready (mid-capture first call, or uniformly disabled at
                # init) -- rank-uniform None, caller runs the unfused path.
                return None
            return self._fused_ar_rmsnorm_impl(input_, residual, weight, eps)
        except Exception as exc:  # noqa: BLE001 -- sticky fail-safe
            # NOTE: a mid-forward failure is only collective-safe if it
            # raises on ALL ranks (deterministic build/shape/topology errors
            # do; the caller's tensors are mutated only at the very end, so
            # the fallback recomputes from unmodified inputs).
            self.disabled = True
            logger.warning(
                "[TOKENWEAVE] disabled after runtime failure (falling back "
                "to the unfused all-reduce + rmsnorm path): %s",
                exc,
                exc_info=True,
            )
            return None

    def _fused_ar_rmsnorm_impl(
        self,
        input_: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # CAPTURE AUDIT (stage B guarantee 3): everything below is legal
        # inside CUDA-graph recording -- python-side arithmetic/views only,
        # in-place device ops on PREALLOCATED buffers (workspace region,
        # _res_shard, _res_stage -- all created in _ensure_workspace, which
        # cannot run here: the caller already guaranteed readiness), one
        # kernel launch with baked scalar args, one preallocated-buffer
        # all-gather. No torch.empty/clone, no host sync/barrier, no logging,
        # no attribute writes. Every branch (padding, CTA policy) depends
        # only on shapes, which are capture-time constants per graph.
        num_tokens, hidden = input_.shape

        world_size = self.world_size
        # Pad the row count up to a multiple of world_size for the kernel's
        # reduce-scatter shard math. Pad rows are zeroed on every rank's
        # replica, so their reduced sum, residual add and normed output are
        # exact zeros -- and they are never copied back out.
        m_pad = -(-num_tokens // world_size) * world_size
        blpr = m_pad // world_size
        start = self.rank * blpr
        end = start + blpr

        region = self._region_buf[: m_pad * hidden].view(m_pad, hidden)
        region[:num_tokens].copy_(input_)
        if m_pad > num_tokens:
            region[num_tokens:].zero_()

        # The kernel reads AND writes only this rank's row shard of its
        # residual arg (contract output 2: the pre-norm sum lands there in
        # place), so only the shard rows are staged. Shards owning pad rows
        # (rows >= num_tokens) get zeros there.
        res_shard = self._res_shard[:blpr]
        copy_rows = min(max(num_tokens - start, 0), blpr)
        if copy_rows > 0:
            res_shard[:copy_rows].copy_(residual[start : start + copy_rows])
        if copy_rows < blpr:
            res_shard[copy_rows:].zero_()

        if self._max_ctas_override is not None:
            max_ctas = self._max_ctas_override
        else:
            # Per-size SM budget: small payloads saturate with 8 CTAs; give
            # larger ones 16 (upstream default).
            max_ctas = 8 if m_pad * hidden * 2 <= 2 * 1024 * 1024 else 16

        # Exactly the validated call convention of test_numerics.run_fused_sut
        # / bench_fused_timing.make_fused_fn: shard views into the region and
        # the residual staging, multicast base + per-rank byte offset, the
        # handle's device-side signal pads. No host barrier/sync around the
        # launch: the kernel's own signal-pad handshake enforces cross-rank
        # entry/exit sync (the timing leg runs it back-to-back the same way).
        self._op(
            region[start:end],
            res_shard,
            weight,
            self._mc_ptr + start * hidden * region.element_size(),
            self._signal_pads,
            self.rank,
            world_size,
            max_ctas,
            eps,
        )

        # The kernel leaves (a) the normed output for ALL rows in the
        # multicast region on every rank, but (b) the pre-norm sum only for
        # THIS RANK's shard rows. Stage A callers are not token-split: every
        # rank needs the full replicated residual stream, so gather the
        # shards (rank order == row order by construction). This all-gather
        # is a stage-A cost the token-split stage C removes.
        #
        # Backend per sglang's graph-mode table (stage B guarantee 5, same
        # idiom as GroupCoordinator._all_reduce_in_place): pynccl whenever it
        # is enabled -- graph_capture flips it on around both the eager
        # warmups and the recording, so the captured branch is the warmed
        # branch -- raw torch.distributed otherwise (plain eager serving,
        # gate-A1 probe). Both buffers preallocated either way.
        res_stage = self._res_stage[:m_pad]
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.all_gather(res_stage, res_shard)
        else:
            dist.all_gather_into_tensor(
                res_stage, res_shard, group=self.device_group
            )

        # Copy out of the persistent workspace (it is reused by the very next
        # fused call while these outputs are still live) with the SAME
        # in-place semantics as the native path -- fused_add_rmsnorm mutates
        # the caller's x and residual and returns them -- so the call site
        # cannot tell the backends apart. No per-call allocation.
        residual.copy_(res_stage[:num_tokens])
        input_.copy_(region[:num_tokens])
        return input_, residual

    # ------------------------------------------------------------------
    # Round-2 native path (plan 2026-07-31): producer writes the region,
    # the fused site consumes it in place. See the __init__ note.
    # ------------------------------------------------------------------
    def _half_view(self, half: int, m_pad: int, hidden: int) -> torch.Tensor:
        base = half * self.max_tokens_pad * hidden
        return self._region_buf[base : base + m_pad * hidden].view(m_pad, hidden)

    def provide_gemm_out(self, num_tokens: int, hidden: int) -> Optional[torch.Tensor]:
        """Hand the producer gemm a region view to write partial sums into.

        Returns a [num_tokens, hidden] bf16 view at the freshly flipped
        ping-pong half, or None when any guard misses (caller keeps its
        normal allocation path). CAPTURE AUDIT: slicing + in-place zeroing of
        preallocated memory + host-side attribute flips only;
        _ensure_workspace declines inside capture unless already initialized
        by the eager warmups (stage B guarantee 1).
        """
        if (
            self.disabled
            or not self.native_enabled
            or num_tokens < self._native_min_tokens  # loss region: fall back
            or num_tokens > self.max_tokens
            or hidden % 8 != 0
            or (self.hidden is not None and hidden != self.hidden)
        ):
            return None
        if not self._ensure_workspace(hidden):
            return None
        half = self._native_half ^ 1
        self._native_half = half
        m_pad = -(-num_tokens // self.world_size) * self.world_size
        region = self._half_view(half, m_pad, hidden)
        if m_pad > num_tokens:
            # Pad rows must be zero on every rank's replica so their reduced
            # sums are exact zeros (same contract as the stage-A impl); the
            # producer gemm only writes the real rows.
            region[num_tokens:].zero_()
        self._native_armed = (num_tokens, half)
        self._native_arms += 1
        if self._native_arms == 1 and not torch.cuda.is_current_stream_capturing():
            logger.info(
                "[TOKENWEAVE-NATIVE] first direct-write arm "
                "(tokens=%d hidden=%d half=%d)",
                num_tokens,
                hidden,
                half,
            )
        return region[:num_tokens]

    def native_fused_ar_rmsnorm(
        self,
        input_: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Fused-site entry for the native path.

        Engages ONLY when input_ IS the armed region view (data_ptr identity
        -- the producer hook wrote the partials there). Always disarms on
        entry: an armed-but-unclaimed view (some other consumer took the
        partials) must never leak to a later site. Returns None on any miss
        so the caller falls through to the stage-A / composed paths with
        input_ unmodified."""
        armed = self._native_armed
        self._native_armed = None
        if (
            armed is None
            or self.disabled
            or not self.native_enabled
            or residual is None
        ):
            return None
        num_tokens, half = armed
        if (
            input_.dim() != 2
            or input_.shape[0] != num_tokens
            or self.hidden is None
            or input_.shape[1] != self.hidden
            or input_.dtype != self.dtype
            or residual.shape != input_.shape
            or residual.dtype != self.dtype
            or weight.dim() != 1
            or weight.shape[0] != self.hidden
            or weight.dtype != self.dtype
        ):
            return None
        m_pad = -(-num_tokens // self.world_size) * self.world_size
        region = self._half_view(half, m_pad, self.hidden)
        if input_.data_ptr() != region.data_ptr():
            return None
        try:
            out = self._native_impl(
                region, m_pad, num_tokens, half, residual, weight, eps
            )
            self._native_claims += 1
            if (
                self._native_claims == 1
                and not torch.cuda.is_current_stream_capturing()
            ):
                logger.info(
                    "[TOKENWEAVE-NATIVE] engaged: first in-place fused "
                    "AR+RMSNorm claim (tokens=%d half=%d)",
                    num_tokens,
                    half,
                )
            return out
        except Exception as exc:  # noqa: BLE001 -- sticky fail-safe
            self.disabled = True
            logger.warning(
                "[TOKENWEAVE-NATIVE] disabled after runtime failure "
                "(falling back to unfused paths): %s",
                exc,
                exc_info=True,
            )
            return None

    def _native_impl(
        self,
        region: torch.Tensor,
        m_pad: int,
        num_tokens: int,
        half: int,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # CAPTURE AUDIT: identical envelope to _fused_ar_rmsnorm_impl (views,
        # in-place ops on preallocated buffers, one kernel launch, one
        # preallocated all-gather) MINUS the region copy-in (the producer
        # already wrote the partials) and MINUS the output copy-out (the
        # region view itself is returned; the ping-pong half protects it
        # until the site after next).
        hidden = self.hidden
        world_size = self.world_size
        blpr = m_pad // world_size
        start = self.rank * blpr
        end = start + blpr

        res_shard = self._res_shard[:blpr]
        copy_rows = min(max(num_tokens - start, 0), blpr)
        if copy_rows > 0:
            res_shard[:copy_rows].copy_(residual[start : start + copy_rows])
        if copy_rows < blpr:
            res_shard[copy_rows:].zero_()

        if self._max_ctas_override is not None:
            max_ctas = self._max_ctas_override
        else:
            max_ctas = 8 if m_pad * hidden * 2 <= 2 * 1024 * 1024 else 16

        # The multicast offset must address THIS half of the workspace: the
        # region base sits half * max_tokens_pad * hidden elements into the
        # rendezvous'd buffer (the round-2 op-level bench validated exactly
        # this shard/offset math via token_split.chunk_shard).
        half_base = half * self.max_tokens_pad * hidden
        self._op(
            region[start:end],
            res_shard,
            weight,
            self._mc_ptr + (half_base + start * hidden) * region.element_size(),
            self._signal_pads,
            self.rank,
            world_size,
            max_ctas,
            eps,
        )

        res_stage = self._res_stage[:m_pad]
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.all_gather(res_stage, res_shard)
        else:
            dist.all_gather_into_tensor(
                res_stage, res_shard, group=self.device_group
            )

        residual.copy_(res_stage[:num_tokens])
        return region[:num_tokens], residual

    def _agree_all_ranks(self, local_ok: bool) -> bool:
        """Rank-uniform verdict on an init step (stage B guarantee 4). Eager
        only -- never reached from inside a capture (_ensure_workspace's
        capture guard precedes every use). Raw torch.distributed is fine
        here: sglang's graph-mode restriction applies to RECORDED
        collectives, not eager ones."""
        flag = torch.tensor(
            [0 if local_ok else 1], dtype=torch.int32, device=self.device
        )
        dist.all_reduce(flag, group=self.device_group)
        return int(flag.item()) == 0

    def _ensure_workspace(self, hidden: int) -> bool:
        """Allocate + rendezvous the symm workspace once, on the first
        eligible call; True when ready. Rank-uniform by construction: every
        verdict (including failures) is agreed via all-reduce, so either all
        ranks commit or all ranks disable (stage B guarantee 4). Never freed
        -- see WORKSPACE LIFETIME in the module header. With CUDA graphs the
        first eligible call is one of the runner's eager pre-capture warmups
        (stage B guarantee 1); the capture guard below is the belt for any
        path that skipped them."""
        if self._region_buf is not None:
            return True
        if torch.cuda.is_current_stream_capturing():
            # Rendezvous/allocation must never be recorded into a graph.
            # Capture is TP-lockstep, so this decline is rank-uniform and the
            # graph simply records the unfused fallback. Not sticky: an eager
            # call afterwards may still initialize.
            return False

        # Phase 1: local allocations (symm buffer + AG staging). Failures
        # here (e.g. OOM) can be rank-local, so agree before the collective
        # rendezvous -- a one-sided rendezvous entry would deadlock.
        buf = None
        err = ""
        try:
            # Native mode ping-pongs two regions (see __init__ note); the
            # stage-A path keeps using half 0 untouched semantics.
            num_regions = 2 if self.native_enabled else 1
            buf = torch_symm_mem.empty(
                num_regions * self.max_tokens_pad * hidden,
                dtype=self.dtype,
                device=self.device,
            )
            # Keep-alive immediately: even a not-yet-rendezvous'd symm tensor
            # must never reach GC on a failure path (belt against the
            # destructor trap; leaks only on the disable path, by design).
            _KEEPALIVE.append(buf)
            res_shard = torch.empty(
                self.max_tokens_pad // self.world_size,
                hidden,
                dtype=self.dtype,
                device=self.device,
            )
            res_stage = torch.empty(
                self.max_tokens_pad, hidden, dtype=self.dtype, device=self.device
            )
        except Exception as e:  # noqa: BLE001 -- folded into the verdict
            err = f"workspace allocation failed: {e}"
        if not self._agree_all_ranks(not err):
            self.disabled = True
            logger.warning(
                "[TOKENWEAVE] disabled uniformly at workspace init: %s",
                err or "allocation failed on another rank",
            )
            return False

        # Phase 2: collective rendezvous (all ranks enter together -- phase 1
        # agreement guarantees that) + local capability checks, agreed again.
        err = ""
        try:
            handle = torch_symm_mem.rendezvous(buf, self.group.group_name)
            # GC trap: keep the multicast-bound allocation alive for the
            # process lifetime even if this communicator is dropped.
            _KEEPALIVE.append(handle)
            mc_ptr = getattr(handle, "multicast_ptr", 0)
            if not mc_ptr:
                raise RuntimeError(
                    "symm-mem handle has multicast_ptr==0: NVLS multicast is "
                    "not available on this topology/torch build; the fused "
                    "kernel hard-requires it"
                )
            signal_pads = getattr(handle, "signal_pad_ptrs_dev", None)
            if signal_pads is None:
                raise RuntimeError(
                    "symm-mem handle lacks signal_pad_ptrs_dev on this torch "
                    "build"
                )
            pad_size = getattr(handle, "signal_pad_size", None)
            max_ctas_bound = self._max_ctas_override or 16
            if (
                pad_size is not None
                and max_ctas_bound * self.world_size * 4 > pad_size
            ):
                raise RuntimeError(
                    f"max_ctas {max_ctas_bound} x world_size {self.world_size} "
                    f"needs {max_ctas_bound * self.world_size * 4}B of signal "
                    f"pad but the handle reports {pad_size}B; lower "
                    f"SGLANG_TOKENWEAVE_MAX_CTAS"
                )
        except Exception as e:  # noqa: BLE001 -- folded into the verdict
            err = f"rendezvous/capability checks failed: {e}"
        if not self._agree_all_ranks(not err):
            self.disabled = True
            logger.warning(
                "[TOKENWEAVE] disabled uniformly at workspace init: %s",
                err or "rendezvous/capability checks failed on another rank",
            )
            return False

        self._res_shard = res_shard
        self._res_stage = res_stage
        self._handle = handle
        self._mc_ptr = mc_ptr
        self._signal_pads = signal_pads
        self._region_buf = buf
        self.hidden = hidden
        # One-time "engaged" evidence, deliberately OUTSIDE the hot path so
        # nothing flips python state inside a captured region (stage B
        # guarantee 3). This point is never under capture (guard above).
        logger.info(
            "[TOKENWEAVE] engaged: workspace ready (hidden=%d max_tokens=%d"
            "(pad %d) world_size=%d rank=%d max_ctas=%s); fused path live "
            "from here on",
            hidden,
            self.max_tokens,
            self.max_tokens_pad,
            self.world_size,
            self.rank,
            self._max_ctas_override if self._max_ctas_override else "auto(8/16)",
        )
        return True

    def close(self) -> None:
        # Intentionally a no-op: the symm workspace is multicast-bound and
        # must never be freed while the process lives (torch 2.9.1
        # cuMulticastUnbind destructor trap); _KEEPALIVE holds the refs and
        # process exit reclaims the memory.
        pass


def native_gemm_out(num_tokens: int, hidden: int) -> Optional["torch.Tensor"]:
    """Producer-side entry for the round-2 native direct write.

    Called from RowParallelLinear.forward on no-reduce calls at marked-shape
    sites; returns the fused-site region view to matmul into, or None (caller
    keeps its normal path). The parallel_state import is lazy to avoid a
    module cycle (parallel_state imports this module at its top), and every
    real guard lives in provide_gemm_out."""
    from sglang.srt.distributed.parallel_state import get_tp_group

    comm = getattr(get_tp_group(), "tokenweave_comm", None)
    if comm is None or not getattr(comm, "native_enabled", False):
        return None
    return comm.provide_gemm_out(num_tokens, hidden)
