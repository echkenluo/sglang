"""Keep eager FlashMLA output allocations separate from model temporaries."""

from contextlib import nullcontext
from weakref import WeakSet

import torch

_OUTPUT_POOLS = WeakSet()


def release_sparse_prefill_output_cache(device: torch.device) -> bool:
    """Yield idle output blocks before cold, direct CUDA allocations.

    Symmetric memory bypasses the caching allocator's OOM recovery. Retiring
    the private pool lets empty_cache return its unused blocks to CUDA; live
    tensors retain their storage. Existing-workspace calls do not use this.
    """
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    released = False
    for owner in list(_OUTPUT_POOLS):
        if owner._pool is not None and owner.device.index == index:
            owner._pool = None
            released = True
    if released:
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
    return released


class SparsePrefillOutputPool:
    """Retain reusable blocks without retaining or aliasing returned tensors.

    FlashMLA allocates its output and two statistics tensors internally. Its
    returned output must retain ordinary allocator-managed lifetime, including
    when callers keep multiple results alive. A private caching pool preserves
    that contract while preventing unrelated MoE allocations from fragmenting
    the large blocks warmed for sparse attention.
    """

    def __init__(self, device: torch.device, enabled: bool = False):
        self.device = torch.device(device)
        self.enabled = enabled
        self._pool = None
        if enabled:
            _OUTPUT_POOLS.add(self)

    def context(self):
        if not self.enabled:
            return nullcontext()
        # Capture owns its own pool. This option only changes eager prefill.
        if torch.cuda.is_current_stream_capturing():
            return nullcontext()
        if self._pool is None:
            # Backends use torch.device("cuda"); use_mem_pool requires an
            # explicit ordinal. Bind lazily to this worker's selected device.
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            with torch.cuda.device(self.device):
                self._pool = torch.cuda.MemPool()
        return torch.cuda.use_mem_pool(self._pool, device=self.device)
