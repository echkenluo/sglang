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

### Production W13 correctness filter

The production-tuner workflow treats compile or numerical failures as recorded
candidate rejections instead of invalidating the entire sampler universe.  It
is opt-in: the historical default remains fail-closed.  Freeze five
representative routes, use positions 0/2/4 as train routes and positions 1/3
as heldout routes, then distribute only the train correctness screen:

```bash
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" python benchmark/kernels/humming/tune_humming_moe.py \
    --model-config /path/to/model/config.json \
    --capture-manifest /path/to/capture/manifest.json \
    --sublayer w13 --expected-humming-version 0.1.13 \
    --route-samples 5 --route-split train \
    --candidate-shard-count 4 --candidate-shard-index "$shard" \
    --candidate-rejection-policy filter --correctness-only \
    --out "/path/to/screen-$shard.json" &
done
wait
```

Every shard repeats the heuristic reference; non-heuristic candidates are
assigned exactly once by deterministic sampler order modulo shard count.  The
merger rejects missing, overlapping or contract-mismatched shards and emits an
ordered survivor-ID file:

```bash
python benchmark/kernels/humming/merge_humming_w13_screens.py \
  --screen /path/to/screen-0.json --screen /path/to/screen-1.json \
  --screen /path/to/screen-2.json --screen /path/to/screen-3.json \
  --out /path/to/survivors.json
```

If a one-pass shard screen exposes a candidate that passes once but fails the
same contract in a later process, use the stability protocol instead of
deleting that candidate after the fact.  Every GPU then screens the complete
universe with a distinct preregistered input seed and multiple executions per
route:

```bash
CUDA_VISIBLE_DEVICES="$gpu" python benchmark/kernels/humming/tune_humming_moe.py \
  ... \
  --candidate-shard-count 4 --candidate-shard-index "$gpu" \
  --replicate-candidate-universe --correctness-repeats 3 \
  --candidate-rejection-policy filter --correctness-only \
  --seed "$replica_seed" --out "/path/to/replica-$gpu.json"

python benchmark/kernels/humming/merge_humming_w13_screens.py \
  --coverage-mode replicated \
  --screen /path/to/replica-0.json --screen /path/to/replica-1.json \
  --screen /path/to/replica-2.json --screen /path/to/replica-3.json \
  --out /path/to/stable-survivors.json
```

The replicated merger requires unique seeds, identical candidate universes and
full coverage in every replica.  Its survivor list is the intersection across
all GPUs; every rejection observation remains in the receipt.  Reuse a
completed replica's compile cache only after all replicas exit, and still rerun
the declared correctness repeats before timing.

If replicated screening shows that instability is confined to stream-K
challengers, `--challenger-stream-k-policy exclude` retains the deployed
exact-shape heuristic as the baseline while removing other stream-K configs
before correctness and timing. The result records the pre/post class counts
and every excluded config ID; this is a class-level safety policy, not an
individual failed-ID denylist.

Time all survivors on one GPU with `--candidate-ids-file` and the train split;
different GPUs must not time disjoint candidates because device-to-device
variation would bias selection.  Only a preregistered train winner is then run
against the heldout split.  Candidate rejection, train selection and heldout
validation are separate result states; none is a service-level deployment
result.

To isolate a whole-Humming runtime upgrade without changing the selected W13
schedule, use `--w13-candidate-source heuristic-only` together with an explicit
`--expected-humming-version`.  This mode tests exactly one candidate: the
installed runtime's production-table heuristic for the captured M.  It is for
matched runtime A/B/A screening, not shape search; compare the reported
`heuristic_median_us` across runs only after asserting the heuristic config IDs
match and the A1/A2 drift gate passes in every run.  A runtime screen still
needs a separate cross-version correctness and service gate before deployment.

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
