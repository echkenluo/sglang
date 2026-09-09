# SPDX-License-Identifier: Apache-2.0
"""DSV4 adapter for the pinned FlashInfer SM89 sparse MLA fork.

Validated dependency target (GPU validation pending at initial implementation):
flashinfer 0.6.18+glm53.dsv4.vision1.sm89sm120.cu130.pt213,
wheel SHA256 72b774b474b7f961c2993b9cdd2337d2bec7c70560d508fee654f3423757863d.

The cache view describes a page, not independent 584-byte token records:
KV payloads precede the page scale footer. Preserve stride(0), including
SGLang's physical page padding. Never make the cache contiguous here.
"""

import torch


class Dsv4FlashInferSm89:
    """Backend-owned runner; no module-global workspace shared across GPUs."""

    def __init__(self):
        from flashinfer.mla._sparse_mla_sm120 import (
            _DECODE_DSV4_PAGE_BLOCK_SIZES,
            _SparseMLAPagedAttentionRunner,
        )

        if not {64, 256}.issubset(_DECODE_DSV4_PAGE_BLOCK_SIZES):
            raise RuntimeError("DSV4 SM89 adapter requires the F28 FlashInfer fork")
        # The runner rejects architectures other than 89/120/121. Our model
        # dispatch is SM89-only; other SGLang backends remain unchanged.
        self.runner = _SparseMLAPagedAttentionRunner(
            d_v=512, device=torch.device("cuda", torch.cuda.current_device())
        )

    def __call__(
        self,
        *,
        q,
        k_cache,
        head_dim_v,
        softmax_scale,
        indices,
        topk_length,
        attn_sink,
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        extra_topk_length=None,
        **unused,
    ):
        if q.ndim != 4 or q.shape[1] != 1 or q.shape[-1] != 512:
            raise ValueError(f"Expected DSV4 query [T,1,H,512], got {q.shape}")
        if q.dtype != torch.bfloat16 or head_dim_v != 512:
            raise ValueError("DSV4 SM89 sparse MLA requires BF16 query and d_v=512")
        for cache in (k_cache, extra_k_cache):
            if cache is not None and (
                cache.ndim != 4
                or cache.shape[2:] != (1, 584)
                or cache.dtype != torch.uint8
            ):
                raise ValueError(
                    "Expected packed DSV4 NHD byte cache [pages,page_size,1,584]"
                )
        q3 = q.squeeze(1).contiguous()
        out = torch.empty_like(q3)
        lse = torch.empty(q3.shape[:2], dtype=torch.float32, device=q.device)
        if q3.shape[0] == 0:
            return out.unsqueeze(1), lse.unsqueeze(-1)
        self.runner.run(
            q3,
            k_cache,
            indices.contiguous(),
            out,
            sm_scale=softmax_scale,
            topk_length=None
            if topk_length is None
            else topk_length.reshape(-1).contiguous(),
            attn_sink=None if attn_sink is None else attn_sink.float().contiguous(),
            extra_kv_cache=extra_k_cache,
            extra_indices=None
            if extra_indices_in_kvcache is None
            else extra_indices_in_kvcache.contiguous(),
            extra_topk_length=None
            if extra_topk_length is None
            else extra_topk_length.reshape(-1).contiguous(),
            out_lse=lse,
        )
        return out.unsqueeze(1), lse.unsqueeze(-1)
