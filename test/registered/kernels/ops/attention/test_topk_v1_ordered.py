"""The ordered build of the DeepSeek-V4 JIT top-k transform v1 must be reproducible.

The stock v1 kernel hands out its output slots with an atomic counter, so a row with more than k
entries gets the same selected set back in an order that depends on GPU thread timing. The sparse
attention that consumes these indices then accumulates in a different order, and a context above
4 * index_topk tokens gives different logits from run to run at temperature 0 (measured on 8xL20:
prefills up to 1609 tokens repeat bit-identically, 2672 tokens differ at 2650 of 2672 positions).
``SGLANG_DSV4_TOPK_ORDERED`` selects a build that places the selected entries by ascending position.

What matters and is checked here: the ordered build selects the same set as torch.topk, its output is
ascending, the page transform matches the raw positions, rows that fit into k are untouched, and
repeated calls return identical tensors. The last check is the one that fails on the stock build.

Order is not the only thing the stock kernel leaves to timing. When several scores tie exactly at the
threshold, its last radix round lets the threads race for the remaining slots, so the selected set
changes too. Real indexer scores are sums of FP8 products and do tie: with the first ordered build an
8xL20 service still gave 4 different results in 12 repeats of a 2673-token prefill. The ordered build
therefore takes the lowest tied positions, which is checked on scores with few distinct values.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import _jit_topk_v1_module
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE_SIZE = 64  # c4 page size = 256 // 4
PAGE_BITS = PAGE_SIZE.bit_length() - 1
PAGE_MASK = PAGE_SIZE - 1
K = 512


def _inputs(batch: int, max_seq: int, seed: int):
    torch.manual_seed(seed)
    device = "cuda"
    width = (max_seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)
    # a mix of rows that fit into k (copied in order by the kernel) and rows that need a selection
    seq_lens = torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32, device=device)
    seq_lens[0], seq_lens[1], seq_lens[2] = K, K + 1, max_seq
    num_pages = (width + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = torch.stack(
        [torch.randperm(num_pages, device=device) for _ in range(batch)]
    ).to(torch.int32)
    return scores, seq_lens, page_table


def _run(ordered: bool, scores, seq_lens, page_table):
    batch = scores.shape[0]
    out = torch.full((batch, K), -1, dtype=torch.int32, device=scores.device)
    raw = torch.full((batch, K), -1, dtype=torch.int32, device=scores.device)
    module = _jit_topk_v1_module(K, ordered)
    module.topk_transform(scores, seq_lens, page_table, out, PAGE_SIZE, raw)
    torch.cuda.synchronize()
    return out, raw


@pytest.mark.parametrize("batch,max_seq", [(4, 700), (64, 2048), (256, 6000), (8, 40000)])
@torch.inference_mode()
def test_ordered_selection_is_the_topk_set_in_ascending_order(batch: int, max_seq: int):
    scores, seq_lens, page_table = _inputs(batch, max_seq, seed=batch * 7919 + max_seq)
    out, raw = _run(True, scores, seq_lens, page_table)
    out_cpu, raw_cpu, lens = out.cpu(), raw.cpu(), seq_lens.cpu().tolist()
    table_cpu, scores_cpu = page_table.cpu(), scores.cpu()
    for row, length in enumerate(lens):
        valid = min(K, length)
        picked = raw_cpu[row, :valid].tolist()
        assert raw_cpu[row, valid:].eq(-1).all() and out_cpu[row, valid:].eq(-1).all()
        assert picked == sorted(picked), f"row {row} (length {length}) is not ascending"
        if length <= K:
            assert picked == list(range(length))
        else:
            ref = set(torch.topk(scores_cpu[row, :length], K, sorted=False).indices.tolist())
            ours = set(picked)
            assert len(ours) == K
            # equal scores at the boundary may be swapped; anything else is a wrong selection
            gained = sorted(scores_cpu[row, list(ours - ref)].tolist())
            lost = sorted(scores_cpu[row, list(ref - ours)].tolist())
            assert gained == lost, f"row {row}: selection differs from torch.topk"
        pages = torch.tensor(picked, dtype=torch.int64)
        expect = (table_cpu[row, pages >> PAGE_BITS].long() << PAGE_BITS) | (pages & PAGE_MASK)
        assert out_cpu[row, :valid].long().equal(expect), f"row {row}: page transform mismatch"


@torch.inference_mode()
def test_ordered_build_repeats_bit_identically():
    scores, seq_lens, page_table = _inputs(256, 6000, seed=20260922)
    first_out, first_raw = _run(True, scores, seq_lens, page_table)
    for _ in range(50):
        out, raw = _run(True, scores, seq_lens, page_table)
        assert out.equal(first_out) and raw.equal(first_raw)


@torch.inference_mode()
def test_rows_that_fit_into_k_match_the_stock_build():
    scores, seq_lens, page_table = _inputs(64, 2048, seed=1)
    ordered_out, ordered_raw = _run(True, scores, seq_lens, page_table)
    stock_out, stock_raw = _run(False, scores, seq_lens, page_table)
    fits = seq_lens <= K
    assert fits.any()
    assert ordered_out[fits].equal(stock_out[fits]) and ordered_raw[fits].equal(stock_raw[fits])
    # the rows that need a selection hold the same set in both builds
    assert ordered_raw.sort(dim=1).values.equal(stock_raw.sort(dim=1).values)


def _tied_inputs(batch: int, max_seq: int, levels: int, seed: int):
    scores, seq_lens, page_table = _inputs(batch, max_seq, seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # few distinct values: nearly every row has more scores equal to the k-th one than free slots
    scores = torch.randint(0, levels, scores.shape, generator=gen, device="cuda").float() / 8
    return scores, seq_lens, page_table


@pytest.mark.parametrize("batch,max_seq,levels", [(64, 2048, 7), (256, 6000, 50), (8, 40000, 300)])
@torch.inference_mode()
def test_exact_ties_at_the_threshold_take_the_lowest_positions(batch: int, max_seq: int, levels: int):
    scores, seq_lens, page_table = _tied_inputs(batch, max_seq, levels, seed=batch + levels)
    _, raw = _run(True, scores, seq_lens, page_table)
    raw_cpu, scores_cpu, lens = raw.cpu(), scores.cpu(), seq_lens.cpu().tolist()
    ambiguous = 0
    for row, length in enumerate(lens):
        if length <= K:
            continue
        # a stable descending sort keeps equal scores in ascending position: the first k entries are
        # the only selection that does not depend on which GPU thread came first
        order = torch.sort(scores_cpu[row, :length], descending=True, stable=True).indices[:K]
        kth = scores_cpu[row, order[-1]]
        ambiguous += int((scores_cpu[row, :length] == kth).sum() > (scores_cpu[row, order] == kth).sum())
        assert raw_cpu[row].tolist() == sorted(order.tolist()), f"row {row} (length {length})"
    assert ambiguous > 0, "the inputs do not exercise a tie at the threshold"


@torch.inference_mode()
def test_tied_scores_repeat_bit_identically():
    scores, seq_lens, page_table = _tied_inputs(256, 6000, 50, seed=11)
    first_out, first_raw = _run(True, scores, seq_lens, page_table)
    for _ in range(50):
        out, raw = _run(True, scores, seq_lens, page_table)
        assert out.equal(first_out) and raw.equal(first_raw)


@torch.inference_mode()
def test_scores_crowded_into_one_coarse_bin():
    """40000 scores within 10 percent of each other share one bin of the first, 8-bit pass.

    The kernel keeps at most 8192 candidates of the threshold bin in shared memory and drops the rest
    in arrival order. A C4 context above 32768 tokens can exceed that when the scores are close
    together; the selection is then neither exact nor reproducible."""
    torch.manual_seed(5)
    batch, length = 4, 40000
    scores = 1.0 + 0.1 * torch.rand(batch, length, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    num_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").repeat(batch, 1)
    _, raw = _run(True, scores, seq_lens, page_table)
    ref = torch.topk(scores, K, dim=1, sorted=False).indices.sort(dim=1).values.to(torch.int32)
    assert raw.equal(ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
