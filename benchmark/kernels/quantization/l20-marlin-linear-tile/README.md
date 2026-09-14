# L20 exact-shape FP8 Marlin Linear tile candidate

Branch `codex/dsv4-sm89-marlin-linear-tile-v0517`, based on `cdae64f126`.
The candidate is off by default. Enable with
`SGLANG_DSV4_SM89_MARLIN_LINEAR_SMALL_TILE=1` or call
`set_marlin_linear_small_tile_enabled(True)` from
`sglang.srt.layers.quantization.marlin_utils_fp8`.

Eligibility is exactly `(M, N, K) = (6, 512, 4096)`, SM89, BF16 input,
FP8 E4M3fn weights, group size 128, full K, no activation ordering or zero
points, non-atomic reduction, and `use_fp32_reduce=True`. All other calls
keep the original routing even when the flag is enabled.

For this shape only, the host selector promotes the existing configuration
`(thread_k, thread_n, num_threads) = (64, 128, 128)` ahead of the original
`(128, 128, 256)`. It retains the existing validity checks and original
configuration as fallback. No GEMM arithmetic, weight preprocessing,
quantization, FP32 reduction, SM count, workspace contract, blocks per SM,
or dynamic shared-memory allocation changes.

`gptq_marlin_gemm` receives a default-false `use_sm89_small_tile` runtime
argument. A and B use the same dtype-cached JIT module and existing kernel
instantiations; the flag does not enter the compile flags or module cache key.
Warm both tiles before timing. Change the flag outside Graph capture and
capture a separate Graph for each arm: replay keeps the tile selected at
capture time. Do not switch the process flag concurrently with other calls.

`get_marlin_linear_tile_config(input, weight_scale, use_fp32_reduce=True)`
calls the same module's C++ host selector and returns `thread_k`, `thread_n`,
`num_threads`, and `blocks_per_sm` for the first M split, including the ordinary
split/fallback rule. It does not report every launch when M needs multiple
splits. This query is diagnostic only and must run outside capture and timed
regions. It uses the prepared Marlin scales and the ordinary zero-point-free
FP8 Linear contract;
it does not launch GEMM. Record the receipt in each arm and verify neighboring
shapes retain identical configurations.

The existing `test/manual/test_dsv4_sm89_fp8_marlin.py` defines the numeric
and Graph-update contract. Preserve its BF16-rounded scaled-weight reference,
relative L2 strictly below 0.002, and `atol=rtol=0.02`. Record A/B output
differences separately. Require exact repeated output within each arm and
Graph updates/restoration under already captured graphs. A component pass or
speedup does not establish model
quality, service throughput, or admission of native FP4 MoE.

Local validation is limited to Python parsing, a CPU-only host-selector guard
fixture, and diff checks. CUDA compilation, real GPU correctness, bounded
component timing, and service benefit remain pending. The changed typed JIT
signature and added diagnostic entrypoint require compiling this source once
before any timing; they do not require rebuilding the image's AOT kernel wheel.
