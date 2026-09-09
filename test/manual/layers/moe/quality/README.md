# Recovered MoK quality clients

These six client/test files are copied byte-for-byte from
`3d5ca220a5696535a539025d619b497a0aa2472d`. `recovery-source.json` records
their hashes. This restores the tools needed for the H20 complete-port work;
it does not restore or approve the old execution environment.

- `logprob_client.py`: freeze 128 ShareGPT prompts and 512 split-generated
  target tokens per prompt, then score the identical target IDs in batches of
  16 with response-order/RID and token-position validation.
- `gsm8k_client.py`: frozen-answer task evaluation with dataset checks.
- `longgen_client.py`: free-running trajectory diagnostics. Output divergence
  is observational and is not a formal quality verdict.

The historical `phase2-v4` protocol uses `fused` as the candidate role name.
That field does not prove warp-role activation. Any future use with the H20
megakernel must bind the actual candidate source, extension, launch settings,
and path-completion evidence in a new run manifest.

The old `expected_assets.json`, launch scripts, power assets and evaluator
have not been copied: their frozen commits/image and prerequisite receipts
do not describe the current runtime. Formal v4 remains non-executable until
those dependencies and current APIs have been validated and frozen. These
client tests use simulated HTTP replies; they do not establish server behavior,
model quality, or performance.

CPU validation on 2026-09-09: 17 tests passed with the command below.

```text
python -m pytest -q test/manual/layers/moe/quality/tests/test_logprob_client_v4.py test/manual/layers/moe/quality/tests/test_gsm8k_client_v4.py test/manual/layers/moe/quality/tests/test_longgen_client_v4.py
```
