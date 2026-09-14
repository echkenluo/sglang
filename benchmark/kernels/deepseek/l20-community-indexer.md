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

## R44 result (2026-09-14)

Tested source: 7b101a8c147ab81d157f21e7778e2ce5d50f4385. All three backends
passed 9/9 sampled-reference/finite-output checks and 9/9 Graph-repeat checks.
The eight timed fixtures have full lengths; the extra boundary fixture is untimed.
Graph wrapper microseconds in the same SGLang image:

| M / K | TileLang | F28 port | Community | Community extra time |
|---|---:|---:|---:|---:|
| 6 / 512 | 7.99 | 21.29 | 9.54 | +19.28% |
| 6 / 8192 | 17.38 | 48.91 | 20.72 | +19.21% |
| 24 / 512 | 10.27 | 27.45 | 12.44 | +21.17% |
| 24 / 8192 | 35.92 | 133.79 | 41.96 | +16.81% |
| 96 / 512 | 16.58 | 48.26 | 18.48 | +11.43% |
| 96 / 8192 | 127.33 | 487.54 | 127.85 | +0.40% |
| 512 / 8192 | 456.23 | 2523.60 | 636.82 | +39.58% |
| 4096 / 1024 | 386.85 | 2525.83 | 654.95 | +69.30% |

Keep TileLang. The community candidate is slower in all eight timed shapes
(0.4%-69.3% extra Graph time); eager timings agree in direction. Do not enable
this flag by default or expand service benchmarking for this implementation.
The implementation and negative result remain available for later tuning.

One sampled M24/K8192 row has 511/512 top-k overlap; the other recorded rows
match. Tolerance checks do not establish exact selection or model equivalence.
F28 here is the local port in the same image, not the standalone F28 service.

Raw archive SHA256:
`f7e0704d0b44aa58e13c1dfb0e3e377fd42cff4acc043bda501bceb397f16e79`.
The AILearning `l20-focused-sm89-comparison-20260914.md` report contains raw
records, independent audit, and the separate R43 full-service comparison.
Both windows restored GPU18's original service; no production image was built.
