"""Stable expert packing for the experimental SM89 MXFP4 Marlin path.

Within each expert, flattened token/slot IDs are ascending. Integer prefix
sums determine destinations; no scheduling-dependent atomic allocation is used.
Only the prefix described by num_tokens_post_padded is initialized/valid.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts(
    IDS,
    COUNTS,
    N: tl.constexpr,
    CHUNKS: tl.constexpr,
    TILE: tl.constexpr,
    EXPERTS: tl.constexpr,
    BINS: tl.constexpr,
):
    chunk = tl.program_id(0)
    rows = chunk * TILE + tl.arange(0, TILE)
    ids = tl.load(IDS + rows, rows < N, other=0).to(tl.int32)
    counts = tl.histogram(ids, BINS, mask=rows < N)
    experts = tl.arange(0, BINS)
    tl.store(COUNTS + experts * CHUNKS + chunk, counts, experts < EXPERTS)


@triton.jit
def _single_chunk_histogram_prefix(
    IDS,
    TOTALS,
    STARTS,
    POST,
    N: tl.constexpr,
    TILE: tl.constexpr,
    EXPERTS: tl.constexpr,
    BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.arange(0, TILE)
    ids = tl.load(IDS + rows, rows < N, other=0).to(tl.int32)
    counts = tl.histogram(ids, BINS, mask=rows < N)
    experts = tl.arange(0, BINS)
    counts = tl.where(experts < EXPERTS, counts, 0)
    padded = tl.cdiv(counts, BLOCK) * BLOCK
    prefix = tl.cumsum(padded, 0)
    tl.store(TOTALS + experts, counts, experts < EXPERTS)
    tl.store(STARTS + experts, prefix - padded, experts < EXPERTS)
    tl.store(POST, tl.sum(padded, 0))


@triton.jit
def _counts(IDS, COUNTS, N: tl.constexpr, CHUNKS: tl.constexpr, TILE: tl.constexpr):
    expert = tl.program_id(0)
    chunk = tl.program_id(1)
    rows = chunk * TILE + tl.arange(0, TILE)
    ids = tl.load(IDS + rows, rows < N, other=-1)
    count = tl.sum(((rows < N) & (ids == expert)).to(tl.int32), 0)
    tl.store(COUNTS + expert * CHUNKS + chunk, count)


@triton.jit
def _chunk_prefix(COUNTS, PREFIX, TOTALS, CHUNKS: tl.constexpr, WIDTH: tl.constexpr):
    expert = tl.program_id(0)
    chunks = tl.arange(0, WIDTH)
    counts = tl.load(COUNTS + expert * CHUNKS + chunks, chunks < CHUNKS, other=0)
    prefix = tl.cumsum(counts, 0)
    tl.store(PREFIX + expert * CHUNKS + chunks, prefix - counts, chunks < CHUNKS)
    tl.store(TOTALS + expert, tl.sum(counts, 0))


@triton.jit
def _expert_prefix(
    TOTALS,
    STARTS,
    POST,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
):
    experts = tl.arange(0, WIDTH)
    counts = tl.load(TOTALS + experts, experts < EXPERTS, other=0)
    padded = tl.cdiv(counts, BLOCK) * BLOCK
    prefix = tl.cumsum(padded, 0)
    tl.store(STARTS + experts, prefix - padded, experts < EXPERTS)
    tl.store(POST, tl.sum(padded, 0))


@triton.jit
def _scatter(
    IDS,
    PREFIX,
    TOTALS,
    STARTS,
    SORTED,
    EXPERT_IDS,
    N: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    PAD_WIDTH: tl.constexpr,
    META_WIDTH: tl.constexpr,
):
    expert = tl.program_id(0)
    chunk = tl.program_id(1)
    rows = chunk * TILE + tl.arange(0, TILE)
    ids = tl.load(IDS + rows, rows < N, other=-1)
    belongs = (rows < N) & (ids == expert)
    local_rank = tl.cumsum(belongs.to(tl.int32), 0) - 1
    start = tl.load(STARTS + expert)
    if CHUNKS == 1:
        preceding = 0
    else:
        preceding = tl.load(PREFIX + expert * CHUNKS + chunk)
    tl.store(SORTED + start + preceding + local_rank, rows, belongs)
    if chunk == 0:
        count = tl.load(TOTALS + expert)
        blocks = tl.cdiv(count, BLOCK)
        pads = tl.arange(0, PAD_WIDTH)
        tl.store(SORTED + start + count + pads, N, count + pads < blocks * BLOCK)
        meta = tl.arange(0, META_WIDTH)
        tl.store(EXPERT_IDS + start // BLOCK + meta, expert, meta < blocks)


def moe_align_block_size_stable(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    use_histogram: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack valid [0, num_experts) IDs, ordered by expert then token/slot.

    No expert map or dropped/invalid IDs are supported. Output padding uses
    topk_ids.numel(), matching Marlin. Unused capacity beyond the returned
    padded count must not be read. Shape-only host logic supports CUDA Graph
    replay with changing routes; every active scratch element is overwritten.
    The opt-in histogram path preserves integer counts and stable ordering,
    combining the counting and expert-prefix launches for a single chunk.
    """
    if not topk_ids.is_cuda or not topk_ids.is_contiguous():
        raise ValueError("stable alignment requires contiguous CUDA topk_ids")
    if topk_ids.ndim != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be a two-dimensional int32/int64 tensor")
    if block_size not in (8, 16, 32, 48, 64) or num_experts < 1:
        raise ValueError("unsupported Marlin block size or expert count")
    n = topk_ids.numel()
    kwargs = {"device": topk_ids.device, "dtype": torch.int32}
    capacity = min(n * block_size, n + (num_experts + 1) * (block_size - 1))
    sorted_ids = torch.empty((capacity,), **kwargs)
    expert_ids = torch.empty((triton.cdiv(capacity, block_size),), **kwargs)
    if n == 0:
        return sorted_ids, expert_ids, torch.zeros((1,), **kwargs)
    post = torch.empty((1,), **kwargs)
    tile = min(256, triton.next_power_of_2(n))
    chunks = triton.cdiv(n, tile)
    counts = torch.empty((num_experts, chunks), **kwargs)
    starts = torch.empty((num_experts,), **kwargs)
    if chunks == 1:
        totals = counts.view(-1)
        prefix = counts  # Not read in the single-chunk scatter specialization.
        if use_histogram:
            _single_chunk_histogram_prefix[(1,)](
                topk_ids,
                totals,
                starts,
                post,
                n,
                tile,
                num_experts,
                triton.next_power_of_2(num_experts),
                block_size,
            )
        else:
            _counts[(num_experts, chunks)](topk_ids, counts, n, chunks, tile)
    else:
        totals = torch.empty((num_experts,), **kwargs)
        prefix = torch.empty_like(counts)
        if use_histogram:
            _histogram_counts[(chunks,)](
                topk_ids,
                counts,
                n,
                chunks,
                tile,
                num_experts,
                triton.next_power_of_2(num_experts),
            )
        else:
            _counts[(num_experts, chunks)](topk_ids, counts, n, chunks, tile)
        _chunk_prefix[(num_experts,)](
            counts, prefix, totals, chunks, triton.next_power_of_2(chunks)
        )
    if not (use_histogram and chunks == 1):
        _expert_prefix[(1,)](
            totals,
            starts,
            post,
            num_experts,
            block_size,
            triton.next_power_of_2(num_experts),
        )
    _scatter[(num_experts, chunks)](
        topk_ids,
        prefix,
        totals,
        starts,
        sorted_ids,
        expert_ids,
        n,
        chunks,
        block_size,
        tile,
        triton.next_power_of_2(block_size),
        triton.next_power_of_2(triton.cdiv(n, block_size)),
    )
    return sorted_ids, expert_ids, post
