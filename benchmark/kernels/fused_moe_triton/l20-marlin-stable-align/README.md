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

GPU validation is pending: independent layout oracle and repeat checks, E256
M5/M6/prefill route changes under CUDA Graph, the old E8 FP4 numeric/repeat probe,
and isolated packing cost. No numeric admission, model quality, service
performance gain or deployment approval is claimed. Preserve the original FP4
numeric failures and all community optimization candidates.
