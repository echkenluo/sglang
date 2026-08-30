# Humming MoE shape tuning

This directory contains the production-shape capture and tuning tools for the
SGLang Humming MoE runner.

## Capture routing shapes

Build an exact-length, reproducible natural-token input from ShareGPT with the
target tokenizer:

```bash
python benchmark/kernels/humming/build_capture_input.py \
  --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenizer /path/to/model \
  --encoding-spec dsv4 \
  --prompt-tokens 32768 \
  --seed 20260828 \
  --out /path/to/fixed-input-ids.json
```

The builder uses an explicitly selected production chat encoder (`dsv4` uses
SGLang's DeepSeek-V4 encoder; `hf` uses the tokenizer template) and records the
dataset hash, selected conversation hashes, exact token count and final
truncation receipt.

Start the server with `--enable-return-routed-experts`, then run:

```bash
python benchmark/kernels/humming/capture_routing.py \
  --base-url http://127.0.0.1:30000 \
  --model-config /path/to/model/config.json \
  --input-ids-json /path/to/fixed-input-ids.json \
  --chunk-size 32768 \
  --out-dir /path/to/new/capture-directory
```

The output preserves the raw little-endian int32 routing tensor and emits a manifest point for
every `(prefill chunk, MoE layer)`.  It deliberately does not pool all layers
into one marginal histogram.  A formal capture requires explicit input IDs;
synthetic IDs are allowed only for plumbing checks.

Run the CPU-only contract tests with:

```bash
python -m unittest \
  benchmark/kernels/humming/test_build_capture_input.py \
  benchmark/kernels/humming/test_capture_routing.py
```

## Kernel screen

Run the H20 kernel screen after a valid capture:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/kernels/humming/tune_humming_moe.py \
  --model-config /path/to/model/config.json \
  --capture-manifest /path/to/capture/manifest.json \
  --tp-size 4 \
  --candidate-count 0 \
  --route-samples 5 \
  --out /path/to/tuning-result.json
```

For W13, candidate count zero tests the complete deterministic set returned by
the official Humming 0.1.12 test sampler (target 100, possibly larger when its
pairwise-coverage seed requires it) plus the exact-shape heuristic; a positive
value is only a smoke-test cap.  The tool fails closed if
the installed Humming version is not exactly 0.1.12, so candidate enumeration
and kernel construction cannot silently cross releases.  Do not treat the
piecewise production dispatch table as a cross-shape candidate pool: entries
outside their M interval are not guaranteed to be safe.  The benchmark-local
sampler compatibility copy remains only as historical source comparison and is
not used by formal W13 runs.  The screen precompiles candidates,
requires every output element to pass Humming's MoE numerical gate
(`rtol=0.01`, `atol=0.2`), records scale-aware aggregate errors, times candidate
order in randomized rounds over multiple real layer routes, and invalidates the
point when the heuristic A1/A2 drift is at least 2%.  The numerical reference is
the current heuristic kernel, so this gate establishes candidate equivalence,
not agreement with an independent high-precision implementation.  A kernel
winner is not a deployment result; it must still pass the full MoE-stage and
matched service A/B/A gates.

For W2, the tool does not use the cross-version sampler.  It compares the
piecewise production-table config with the heuristic recomputed at the real
routed M, then sweeps only `num_sms` around that shape-specific persistent-grid
value while holding kernel geometry fixed.  The sweep includes coarse bounds
and a 1/16-step refinement around half of the direct-shape grid.  This isolates
the production-table tail issue: Humming's table is generated only through
roughly 65K routed rows, whereas a 32K-token, top-k 6 prefill chunk has routed M
196608.

For W13, the formal path uses Humming 0.1.12's own resource-filtered,
pairwise-covering sampler against the 0.1.12 kernel constructor.  It therefore
explores legal block/warp, transfer and scheduling combinations rather than
only the old eight-config schedule surface.  Every sampled candidate remains
subject to precompile, full-output correctness and fail-closed route gates; a
sampler-issued config is not assumed correct merely because it compiled.

## Cold-prefill service leg

Restart the server for every tuning variant, disable radix caching, and run a
fixed-input service leg with:

```bash
python benchmark/kernels/humming/bench_prefill_service.py \
  --base-url http://127.0.0.1:31260 \
  --input-ids-json /path/to/input-32768.json /path/to/input-65536.json \
  --variant default-a1 \
  --warmups 2 \
  --repeats 8 \
  --out /path/to/default-a1.json
```

The tool also flushes the server radix cache before every request, uses one
greedy output token, randomizes the two shape orders within each repeat, and
records response hashes.  It measures one isolated server leg only.  A valid
deployment comparison still requires full server restarts and an A/B/A-style
leg order with the same launch arguments, input artifacts, warmup count, repeat
count, and seed.

After all legs finish, validate the controls and derive a fail-closed decision:

```bash
python benchmark/kernels/humming/compare_prefill_service.py \
  --a1 /path/to/a1.json \
  --candidate /path/to/4096.json \
  --candidate /path/to/5120.json \
  --a2 /path/to/a2.json \
  --drift-threshold-pct 2 \
  --out /path/to/comparison.json
```

The comparison requires identical contracts, input hashes and response hashes.
If either baseline median drifts by at least the threshold, the group is
`INVALID_DRIFT`; candidate deltas remain diagnostic and their deployment
decisions are `UNANSWERABLE_INVALID_GROUP`.
