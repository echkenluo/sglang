# DSV4 L20 M5/M6 MoE candidate

Opt-in configuration for the measured DeepSeek-V4-Flash TP8 per-rank FP8 MoE shapes on L20, Triton 3.6.0. This remains a real-model validation candidate.

Enable only for this experimental stack by setting `SGLANG_MOE_CONFIG_DIR` to this directory. The ordinary built-in configuration directory is unchanged.

- Input: BF16, hidden 4096; w13 `[256,512,4096]`, w2 `[256,4096,256]`; topk 6, block quantization 128x128, clamp 10, routed scale 1.5.
- M=5/6 use BLOCK_SIZE_M=16; BLOCK_SIZE_N/K=128, GROUP_SIZE_M=32, num_warps=4, num_stages=3.
- All other integer M values select the original BLOCK_SIZE_M=64 configuration. Neighbor keys 4 and 7 prevent the loader's nearest-M lookup from spreading the candidate beyond M5/M6; keys 0 and 4096 keep both tails at the default.
- Do not reuse this directory for other model shapes, quantization formats, deterministic mode or Triton versions without a separate check. The directory replaces the configuration search root, so unrelated shapes may fall back to defaults.

Evidence: GPU18 R19 linked actual draft M5 and target-verify M6 Graph replays on eight ranks. R21 synthetic screening compared four candidates across eight shape/routing/input-scale cases each. This candidate passed all eight numeric and drift checks; synthetic outputs matched the default exactly. Kernel speedup ranged from 1.030x to 1.517x, depending on routing. These component results do not establish real-weight correctness, model quality or service throughput.

The complete experiment and immutable raw artifacts are indexed in the AILearning research repository, `l20-sglang-community-sm89-integration-20260911.md`.
