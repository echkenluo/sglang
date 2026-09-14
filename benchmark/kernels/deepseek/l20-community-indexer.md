# L20 community paged Indexer candidate

Source: xltzsoft/deepseek-v4-sm89 at ecd6f7f25c8fad4b2c1904a622270004b300532c,
`python/sglang/srt/layers/attention/dsv4/triton_paged_mqa_logits.py`.

The opt-in `SGLANG_DSV4_SM89_COMMUNITY_INDEXER=1` selects the community
packed-page Triton logits before the existing TileLang fallback. It is disabled
by default and does not alter FP4 expert weights or replace nonpaged plans.
The existing F28 Indexer flag takes precedence; do not combine candidate flags.
The adaptation widens physical page arithmetic to int64 before multiplication.

The current TileLang wrapper already uses a zero-copy packed view. A gain cannot
be credited to removing a whole-pool copy that is absent from this baseline.

`benchmark_sm89_indexer_candidates.py --output result.json` compares TileLang,
F28 and community on the same packed FP8 fixtures. Query rows 6/24/96 cover
representative static DSpark verify sizes; 512/4096 exercise larger rows. These
are synthetic shapes, not receipts of exact runtime frequency. References use
FP32 without TF32; the predeclared check is atol=rtol=0.001 on seven sampled
rows, plus finite valid outputs for all rows and exact Graph repeat checks.
An extra boundary-only fixture includes empty/partial pages. All eight timed
fixtures use full lengths for every query, avoiding sparse-boundary geometry in
performance results. FP8 values are converted before reference gathers. Top-k
overlap is recorded separately.
Eager and Graph times include each public wrapper. No service speedup or full
quality claim follows from this component screen.

Next gate: L20 component correctness and timing, then real-path service checks
only if the candidate is competitive. R43 cross-engine measurements use the
unchanged cdae source and are independent of this candidate.
