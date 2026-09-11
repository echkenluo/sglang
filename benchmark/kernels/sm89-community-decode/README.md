# DSV4 SM89 community decode candidate

This opt-in branch adapts the tiled Triton sparse decode route from
`xltzsoft/deepseek-v4-sm89@ecd6f7f25c8fad4b2c1904a622270004b300532c`.
The fixed SM89 launch selection follows `e20d29454d`: BLOCK_T=32,
8 warps, 2 stages. The community measured RTX4090; L20 benefit is unverified.

Set `SGLANG_DSV4_SM89_SPARSE_DECODE=1` to use this implementation for
continuation modes (decode/idle, target verify and draft extend v2).
The default is false; non-SM89 devices reject this opt-in at startup.
Ordinary prefill retains the existing community-prefill/FlashInfer selection.
No change to the global `SGLANG_SM120_FLASHMLA_BACKEND` setting is needed.
Other architectures retain the existing three Triton launch candidates.

The route passes the existing SWA and optional C4/C128 cache views and valid
lengths directly to the community wrapper. It preserves native query heads,
merges the two attention outputs and applies the attention sink using the
existing implementation. The `invoked` log records Python entry, which may
occur during Graph capture; it alone is not proof of later Graph replay.

CPU checks:

```sh
python3 test/registered/unit/layers/test_dsv4_sm89_community_decode.py
```

The separate GPU check must run in the pinned SGLang environment on an idle
SM89 GPU, after any ongoing service performance window has ended:

```sh
python3 test/manual/test_dsv4_sm89_community_decode.py
```

It uses known footer-packed data and a dense FP32 reference for H8 at query
widths 1/5/6/10/12/40/48/80/96, SWA page256, C4 page64 and C128 page2,
strided indices, invalid entries, valid-length masks, both uint8/FP8 cache
views, attention sink, and repeated Graph replay with changed queries.
The output tolerance is atol=0.05, rtol=0.05, inherited from the existing
FlashInfer fixture. These are component tests, not a model quality claim.

## Active-prefix loop candidate

The original `160f6e88aa` integration passed the synthetic GPU/Graph checks.
A same-input study on 32 real L20 attention calls found the community output
closer to a dense FP32 reference than FlashInfer in every sample. Full-model
continuation nevertheless differed, so model quality and service adoption
remain unestablished. In four single-GPU hot-cache Graph cases, the unchanged
community implementation took about 3-26 times the FlashInfer latency.

This branch bounds each tile loop by the clamped per-query valid length,
instead of scanning the whole padded index capacity. The existing masks,
attention arithmetic, partial-output dtype, merge, sink and SM89 launch
configuration stay unchanged. A new GPU regression compares the same active
keys with small and large padded capacities, including zero lengths and tile
boundaries, and requires identical output and LSE.

The active-prefix change has not yet been GPU-validated or timed. It remains
an independent opt-in candidate; the currently running remote comparison
uses the original frozen kernel. CPU dispatch checks and static validation
are separate from these pending GPU checks.
