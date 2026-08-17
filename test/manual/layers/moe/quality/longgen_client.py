"""Long-generation and concurrency determinism (Phase 2 leg 4).

Serial: the first 16 targets prompts, greedy 512 tokens each, record
output token ids.  Concurrent: the same 16 as one wave, repeated three
times; the gate compares the three waves pairwise for exact token
sequences (the serial pass is recorded for cross-config comparison, not
compared against the waves -- batch composition legitimately changes
numerics).
"""

import argparse
import concurrent.futures as futures
import hashlib
import json
import urllib.request


def gen(port, text):
    # text may be a single string (serial) or a list of 16 (one explicit
    # batch payload -- frozen batch composition for the determinism gate).
    body = json.dumps({
        "text": text,
        "sampling_params": {"temperature": 0, "max_new_tokens": 512,
                            "ignore_eos": True},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        payload = json.load(resp)
    if isinstance(payload, list):
        return [p["output_ids"] for p in payload]
    return payload["output_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--out-dir", default="/mok/claude-mok/quality")
    args = ap.parse_args()
    targets = json.load(open(f"{args.out_dir}/logprob-targets.json"))[:16]
    prompts = [t["text"] for t in targets]

    serial = [gen(args.port, p) for p in prompts]
    waves = [gen(args.port, prompts) for _ in range(3)]
    wave_exact = all(waves[0] == waves[k] for k in (1, 2))
    out = f"{args.out_dir}/longgen-{args.tag}.json"
    json.dump({"serial": serial, "waves_exact": wave_exact,
               "wave0": waves[0]}, open(out, "w"))
    print(f"LONGGEN|tag={args.tag}|waves_exact={wave_exact}"
          f"|sha={hashlib.sha256(open(out,rb).read()).hexdigest()[:16]}",
          flush=True)


if __name__ == "__main__":
    main()
