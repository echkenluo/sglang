# L20 native MXFP4 stable expert packing

Branch: `codex/dsv4-sm89-marlin-stable-align-v0517`, based on `b26362f636`.

The paired R32 synthetic experiment found that E8/M513 native token order and
FP4 output changed on every repeat in both SGLang and F28. Freezing either of
two valid layouts removed repeat differences. This candidate recomputes stable
expert/token-slot order on GPU for changing inputs; it does not cache routes.

Enable explicitly with `SGLANG_DSV4_SM89_MARLIN_STABLE_ALIGN=1`. Default is off.
The integration is limited to native MXFP4, SM89, no expert map, and all global
experts present locally. It leaves quantization, activation and GEMM reductions
unchanged. Three launches handle one routing chunk; four handle larger inputs.

The packing API requires valid expert IDs, initializes only the active padded
prefix, and uses token-count sentinels for padding. Chunk/expert prefix sums use
integer arithmetic and scratch is recomputed on every invocation. This seeks
repeatability for identical shapes/inputs, not cross-batch bitwise invariance.

R33 GPU validation completed on GPU18 L20 with source `85774114b6`:

- Three test methods / 29 subtests passed, including E256 changing routes and
  restoration under CUDA Graph, all supported blocks, and int32/int64 IDs.
- The unchanged E8 FP4 fixture passed repeat/history/Graph checks at
  M1/7/16/128/513. Python dispatch/capture records confirm the stable path.
- The original numeric gate still fails: M128 has 4 FP32 / 2 FP64 outliers;
  M513 has 7 / 3. The other three shapes pass. Repeatability is not admission.
- Isolated packing ABA (three rounds, 30 samples/arm, 16 calls/Graph) was valid
  in all six cases. E256 M5/M6 measured 3.584/3.776 us versus native 3.648/3.648;
  M128/513/4096 measured 5.952/9.664/40.064 versus 3.840/4.800/10.176 us.
  E8 M513 measured 4.480 versus native 4.672 us. These are packing costs,
  not complete MoE or service throughput. Long-prefill cost remains material.

Keep the flag off. The R33 E256 checks cover routing, not the full checkpoint's
MoE computation. No full-model quality, batch invariance, service gain or
production readiness is established. Preserve original numeric failures and
all community optimization candidates. GPU18 original service was restored
with health and fixed-token checks passing.


R35/R36 retained the original numeric failures and corrected a diagnostic
FP64-to-BF16 double-rounding reference. Exact arithmetic traced all six remaining
final-output outliers: five propagate from three GEMM1 rounding differences and
one originates in GEMM2. Two intermediate differences agree with correctly
rounded FP32 followed by BF16; two retain accumulation error. The stable-packing
implementation does not change GEMM arithmetic.

R37 completed a checkpoint diagnostic on GPU18, with the same tested source
`85774114b6` in all arms. Stable/native/stable changed only the alignment flag;
all arms used native FP4, static DSpark, communication, community long prefill,
the corrected FlashInfer cache, and five short-prefill Graph buckets.

- Stable A1: all 12 triplicate groups passed within-group, cold/cached and
  before/after-mixed token equality.
- Native B: 11 of 12 triplicate groups diverged; 4 of 6 before/after comparisons
  and all 6 cold/cached comparisons differed.
- Stable A2: 11 of 12 triplicate groups passed. The third cached long-prefix
  request after mixed reuse differed at zero-based token 106. Its selected-token
  logprobs already differed at token 0; the cause remains unproven.
- Stable A1/A2 matched 35 of 36 corresponding requests. All arms confirmed
  4608 cached tokens in repeated long-prefix requests, native E2M1 MoE, M5/M6,
  and the intended alignment path on all eight target/draft Graph ranks.

This is substantial repeatability improvement, not a complete fix. The model
repeatability gate and original numerical gate remain failed. Hooks collected
metadata, so this run provides no throughput result. Keep default off pending
the long-prefill investigation, quality checks and a separate performance
comparison. The original service was restored with identical fixed outputs.


R38 reanalysis found reported probabilities differed in all eight stable-arm
long-input triplicate groups from R37, including cold requests. All sixteen
short-input groups had identical reported probabilities. The sole warm-request
token divergence does not establish cache reuse as the cause.

R38 then repeated real checkpoint GEMMs during three cold 4817-token, one-output
requests, using the same implementation and stable flag. On each of eight ranks,
all 43 target layers, both GEMMs, and both prefill chunks were covered: 516 calls
per rank, each repeated five times, 20640 comparisons overall. Every comparison
was bitwise identical. GEMM1 used M4096/M728; GEMM2 used M24576/M4368. These are
real E256 model weights, not the old E8 numeric fixture. Draft GEMMs were not
covered by these one-output requests.

Three ordinary and three instrumented requests all returned the same first
token, but reported probabilities varied in both groups. Repetition adds GPU
work and synchronization, so this is a bounded same-input GEMM diagnostic, not
proof of global determinism or timing equivalence. No failure tensor was created.
The original service was restored with health and fixed-output checks passing.
No new engine implementation or performance result was produced. Preserve all
community candidates; native FP4 remains unadmitted and default off. Next locate
the first changed intermediate value in cold long prefill.
