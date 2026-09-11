"""Keep eager FlashMLA output allocations separate from model temporaries."""

from contextlib import nullcontext

import torch


class SparsePrefillOutputPool:
    """Retain reusable blocks without retaining or aliasing returned tensors.

    FlashMLA allocates its output and two statistics tensors internally. Its
    returned output must retain ordinary allocator-managed lifetime, including
    when callers keep multiple results alive. A private caching pool preserves
    that contract while preventing unrelated MoE allocations from fragmenting
    the large blocks warmed for sparse attention.
    """

    def __init__(self, device: torch.device, enabled: bool = False):
        self.device = device
        self.enabled = enabled
        self._pool = None

    def context(self):
        if not self.enabled:
            return nullcontext()
        # Capture owns its own pool. This option only changes eager prefill.
        if torch.cuda.is_current_stream_capturing():
            return nullcontext()
        if self._pool is None:
            with torch.cuda.device(self.device):
                self._pool = torch.cuda.MemPool()
        return torch.cuda.use_mem_pool(self._pool, device=self.device)
