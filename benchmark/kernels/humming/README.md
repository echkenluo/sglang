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

Candidate count zero tests the complete deduplicated production ladder; a
positive value is only a smoke-test cap.  The screen precompiles candidates,
requires every output element to pass Humming's MoE numerical gate
(`rtol=0.01`, `atol=0.2`), records scale-aware aggregate errors, times candidate
order in randomized rounds over multiple real layer routes, and invalidates the
point when the heuristic A1/A2 drift is at least 2%.  The numerical reference is
the current heuristic kernel, so this gate establishes candidate equivalence,
not agreement with an independent high-precision implementation.  A kernel
winner is not a deployment result; it must still pass the full MoE-stage and
matched service A/B/A gates.
