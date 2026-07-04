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
# and never freed; process exit reclaims it. The fixed-address, no-per-call-
# allocation design also keeps the door open for CUDA-graph capture (stage B).

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
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        device_group: Optional[ProcessGroup] = None,
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
        """
        self.disabled = True

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

        # hidden_size is unknown at engine init; the workspace is allocated
        # once, on the first eligible call (_ensure_workspace), never freed.
        self.hidden: Optional[int] = None
        self._region_buf: Optional[torch.Tensor] = None  # symm-mem, flat
        self._handle: Optional[Any] = None
        self._mc_ptr: int = 0
        self._signal_pads: int = 0
        self._res_shard: Optional[torch.Tensor] = None
        self._res_stage: Optional[torch.Tensor] = None
        self._engaged_logged = False

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
        num_tokens, hidden = input_.shape
        self._ensure_workspace(hidden)

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
        res_stage = self._res_stage[:m_pad]
        dist.all_gather_into_tensor(res_stage, res_shard, group=self.device_group)

        # Copy out of the persistent workspace (it is reused by the very next
        # fused call while these outputs are still live) with the SAME
        # in-place semantics as the native path -- fused_add_rmsnorm mutates
        # the caller's x and residual and returns them -- so the call site
        # cannot tell the backends apart. No per-call allocation.
        residual.copy_(res_stage[:num_tokens])
        input_.copy_(region[:num_tokens])

        if not self._engaged_logged:
            self._engaged_logged = True
            logger.info(
                "[TOKENWEAVE] engaged: tokens=%d hidden=%d world_size=%d "
                "max_ctas=%d",
                num_tokens,
                hidden,
                world_size,
                max_ctas,
            )
        return input_, residual

    def _ensure_workspace(self, hidden: int) -> None:
        """Allocate + rendezvous the symm workspace once, on the first
        eligible call (collective: every rank reaches here together because
        should_engage is rank-uniform). Never freed -- see WORKSPACE LIFETIME
        in the module header."""
        if self._region_buf is not None:
            return

        buf = torch_symm_mem.empty(
            self.max_tokens_pad * hidden, dtype=self.dtype, device=self.device
        )
        handle = torch_symm_mem.rendezvous(buf, self.group.group_name)
        # GC trap: keep the multicast-bound allocation alive for the process
        # lifetime even if this communicator is dropped.
        _KEEPALIVE.append((buf, handle))

        mc_ptr = getattr(handle, "multicast_ptr", 0)
        if not mc_ptr:
            raise RuntimeError(
                "symm-mem handle has multicast_ptr==0: NVLS multicast is not "
                "available on this topology/torch build; the fused kernel "
                "hard-requires it"
            )
        signal_pads = getattr(handle, "signal_pad_ptrs_dev", None)
        if signal_pads is None:
            raise RuntimeError(
                "symm-mem handle lacks signal_pad_ptrs_dev on this torch build"
            )
        pad_size = getattr(handle, "signal_pad_size", None)
        max_ctas_bound = self._max_ctas_override or 16
        if pad_size is not None and max_ctas_bound * self.world_size * 4 > pad_size:
            raise RuntimeError(
                f"max_ctas {max_ctas_bound} x world_size {self.world_size} "
                f"needs {max_ctas_bound * self.world_size * 4}B of signal pad "
                f"but the handle reports {pad_size}B; lower "
                f"SGLANG_TOKENWEAVE_MAX_CTAS"
            )

        self._res_shard = torch.empty(
            self.max_tokens_pad // self.world_size,
            hidden,
            dtype=self.dtype,
            device=self.device,
        )
        self._res_stage = torch.empty(
            self.max_tokens_pad, hidden, dtype=self.dtype, device=self.device
        )
        self._handle = handle
        self._mc_ptr = mc_ptr
        self._signal_pads = signal_pads
        self._region_buf = buf
        self.hidden = hidden

    def close(self) -> None:
        # Intentionally a no-op: the symm workspace is multicast-bound and
        # must never be freed while the process lives (torch 2.9.1
        # cuMulticastUnbind destructor trap); _KEEPALIVE holds the refs and
        # process exit reclaims the memory.
        pass
