# L20 dense decode CUDA Graph buckets

Branch: `codex/dsv4-sm89-dense-decode-v0517`.

R41 screens a configuration optimization in the R40 native MXFP4 Marlin,
stable packing, communication and static DSpark stack. It uses existing
SGLang graph capture support; no inference kernel or default is changed.
Measured engine source remains `cdae64f12610587f8e20d68a6fd11a2ff97235ef`.

With the baseline buckets, a live batch of three requests replays the graph
for four; five replays eight, and nine replays sixteen. The candidate adds
all missing buckets from one to sixteen. This removes graph padding in
the tested non-DP configuration, at the cost of more startup capture work
and potentially more graph memory. Reduced padding is not itself proof
of improved serving throughput.

The YAML files contain only the changed setting. Merge the field into the
complete deployment config, or replace the existing CLI argument with:

```text
--cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
```

Do not leave a baseline CLI value overriding the YAML field. All other
R40 settings, including five drafted/six verified tokens, stay identical.

R41 uses baseline A1 / dense B / baseline A2 on GPU18. Each arm measures
32-input/128-output at concurrency 1, 4 and 16, and 1024-input/1024-output
at concurrency 4 and 16, with three warmup and five formal batches each.
The existing four-metric 5% anchor-drift gate and known-answer admission
apply. Post-measurement profiler requests are diagnostics only.

Status: candidate under measurement; no performance adoption yet.

The driver, accounting and evidence live in AILearning under
`research/inference-research/optimization/sglang-e2e-optimization-20260611-13/`:

- `05-code-branches/run-l20-dense-decode-graphs.py`
- `05-code-branches/analyze-l20-dense-decode-graphs.py`
- `05-code-branches/analyze-l20-decode-traces.py`
- `02-experiment-reports/l20-sglang-community-sm89-integration-20260911.md`

Preserve the older FP4 numeric failures and the limited R39/R40 admission
boundary. Graph bucket tuning does not establish full-model quality or
cross-batch bitwise invariance.
