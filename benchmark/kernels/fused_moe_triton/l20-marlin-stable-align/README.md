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

Keep the flag off. The E256 checks cover routing, not the full checkpoint's
MoE computation. No full-model quality, batch invariance, service gain or
production readiness is established. Preserve original numeric failures and
all community optimization candidates. GPU18 original service was restored
with health and fixed-token checks passing.
