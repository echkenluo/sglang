# MoK full-MoE boundary diagnostic

Set `SGLANG_MOK_BOUNDARY_TIMING_DIR` before starting an **eager** diagnostic
server to collect CUDA event intervals on every rank. The default is disabled;
decorated functions are returned unchanged when the variable is absent.
`SGLANG_MOK_BOUNDARY_TIMING_MIN_TOKENS` defaults to 256 and
`SGLANG_MOK_BOUNDARY_TIMING_MAX_RECORDS` to 4096 per flush and process.

Recorded scopes:

| Label | Actual call boundary |
|---|---|
| `full_moe` | V4 `_run_moe_ffn_dp_sync`, including surrounding gather/scatter, shared-local addition and output return |
| `routed_moe` | `FusedMoE.forward_impl`, the common DeepEP/split/warp-role boundary, including final output materialization and any final all-reduce |
| `shared_expert` | V2 `_forward_shared_experts`, recorded on the stream actually executing it |
| `mok_native_call` | `maybe_run_mok_fp8_native`, including admission, prepare, schedule, copies and lease lifetime; `outcome=none` identifies fallback |

Complete requests, then call the ordinary `/flush_cache` endpoint. Each idle
scheduler rank synchronizes and writes a unique `rank*-pid*-flush*.json` before
clearing caches. No per-layer synchronization, elapsed-time query or file write
occurs on the measured request path. A drain during an active scope is rejected.
Events are pooled after drain: run warmup requests and flush them separately
before collecting measurement requests, so recurring event allocation is avoided.

`sequence` and `parent` link nested scopes within one process. `stream` identifies
which CUDA stream recorded each pair. `cuda_elapsed_ms` includes device idle time
caused by host dispatch gaps inside the call; it is not a sum of kernel busy
times. `host_enqueue_span_ns` is separate host time and must not be added to the
CUDA interval. Nested/shared intervals can overlap and must not be summed.
No cross-GPU clock alignment is assumed; aggregate comparable intervals per rank
first, then apply the benchmark's preregistered rank reduction.

Graph capture is skipped and counted; replay is not measured. Shorter-than-minimum
inputs are deliberately outside this diagnostic. Record-capacity omissions are
explicit and invalidate complete coverage. An empty or successfully written file
does not establish expected layer/call coverage. Verify all rank IDs, expected
43 layers, token shapes, outcomes, parent relationships and process provenance.
TBO's decomposed path and implementations bypassing these methods are outside
the scope; they require separate instrumentation if enabled. Shared-expert
fusion can also eliminate the standalone shared scope, so missing shared spans
must not automatically be interpreted as zero shared-expert cost.

Freeze the same instrumented SGLang source for all comparison arms. Use separate
uninstrumented/instrumented service controls to measure event overhead; neither
these spans nor the CPU tests establish model quality or an end-to-end gain.
The running interleave correctness test still uses its original frozen SGLang
source and is not changed by this diagnostic implementation.

CPU validation:

```text
python3 test/manual/layers/moe/test_mok_boundary_timing.py
```

Six tests cover disabled identity, deferred synchronization, nested scopes,
event reuse, fallback/exception preservation, capture/capacity omissions,
active-scope drain rejection and keyword/small-input behavior. Actual-model
coverage and observer overhead validation remain pending.

The separate `test_mok_boundary_timing_cuda.py` requires an owned CUDA device.
It exercises nested scopes on two joined streams, compares the outer interval
with enclosing reference CUDA events, checks the real JSON export, and verifies
that graph capture is explicitly omitted rather than counted as replay timing.
Set `MOK_BOUNDARY_CUDA_TEST_OUTPUT` to a fresh persistent directory to retain the
warmup, checked-call and reference records. Both CUDA tests passed on GPU9 H20
GPU0 with PyTorch 2.11.0+cu130 on 2026-09-09 at 15:58 UTC (source 512bda2,
recorder SHA256 9bc2e479c9ce252eab9969df6fa9e4aa204745562d8a7163e254413b92d322ff).
The run retained four event exports and checked their GPU UUID and recorder
source hash; no tests were skipped. These tests do not establish actual model
coverage or acceptable instrumentation overhead. The original result bundle is
`boundary-cuda-probe-v1` under the AILearning H20 MoK experiment assets.
