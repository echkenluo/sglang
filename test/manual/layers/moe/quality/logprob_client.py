"""Teacher-logprob harness (Phase 2 legs 2b/3b).

stage=target (split server): deterministically select 128 ShareGPT first
human turns whose server-side prompt length lands in [256, 2048] tokens,
record prompt token ids (via return_logprob over the prompt) and a greedy
64-token target.  stage=score (any config): teacher-forced scoring of
prompt+target with logprob_start_len at the prompt boundary; emits
per-token logprobs for the 64 target positions.
"""

import argparse
import hashlib
import json
import urllib.request


def post(port, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def sharegpt_prompts(path):
    data = json.load(open(path))
    for conv in data:
        turns = conv.get("conversations", [])
        if turns and turns[0].get("from") == "human":
            text = turns[0].get("value", "").strip()
            if text:
                yield text


def stage_target(args):
    selected = []
    for text in sharegpt_prompts(args.sharegpt):
        r = post(args.port, {
            "text": text,
            "sampling_params": {"temperature": 0, "max_new_tokens": 64,
                                "ignore_eos": True},
            "return_logprob": True,
            "logprob_start_len": 0,
        })
        meta = r["meta_info"]
        n_prompt = meta.get("prompt_tokens", -1)
        if not 256 <= n_prompt <= 2048:
            continue
        prompt_ids = [t[1] for t in meta["input_token_logprobs"]]
        target_ids = r["output_ids"][:64]
        if len(target_ids) < 64:
            continue  # frozen rule: full-length targets only
        selected.append({
            "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
            "text": text,
            "prompt_ids": prompt_ids,
            "target_ids": target_ids,
        })
        if len(selected) == 128:
            break
    assert len(selected) == 128, (
        f"target selection exhausted ShareGPT with only {len(selected)}")
    out = f"{args.out_dir}/logprob-targets.json"
    json.dump(selected, open(out, "w"))
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()[:16]
    print(f"LOGPROB_TARGETS|n={len(selected)}|sha={sha}", flush=True)


def stage_score(args):
    targets = json.load(open(args.targets))
    assert len(targets) == 128, f"expected 128 targets, got {len(targets)}"
    rows = []
    for t in targets:
        ids = t["prompt_ids"] + t["target_ids"]
        r = post(args.port, {
            "input_ids": ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 0},
            "return_logprob": True,
            "logprob_start_len": len(t["prompt_ids"]) - 1,
            "top_logprobs_num": 1,
        })
        meta = r["meta_info"]
        tail = meta["input_token_logprobs"][-len(t["target_ids"]):]
        ids_seen = [x[1] for x in tail]
        assert ids_seen == t["target_ids"], (
            f"token-ID misalignment for {t['text_sha']}")
        lps = [float(x[0]) for x in tail]
        assert all(l == l and abs(l) != float("inf") for l in lps), (
            f"non-finite logprob for {t['text_sha']}")
        ttop = meta.get("input_top_logprobs")
        assert ttop, f"missing input_top_logprobs for {t['text_sha']}"
        ttail = ttop[-len(t["target_ids"]):]
        assert len(ttail) == 64, f"top-logprob slice short for {t['text_sha']}"
        top1 = []
        for pos, x in enumerate(ttail):
            assert x and len(x) >= 1 and isinstance(x[0][1], int), (
                f"malformed top-1 at {t['text_sha']}:{pos}")
            top1.append(x[0][1])
        rows.append({
            "text_sha": t["text_sha"],
            "logprobs": lps,
            "token_ids": ids_seen,
            "top1_ids": top1,
        })
    total = sum(len(r["logprobs"]) for r in rows)
    assert total == 128 * 64, f"expected 8192 positions, got {total}"
    out = f"{args.out_dir}/logprob-score-{args.tag}.json"
    json.dump(rows, open(out, "w"))
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()[:16]
    print(f"LOGPROB_SCORE|tag={args.tag}|n={len(rows)}|sha={sha}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--stage", choices=["target", "score"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sharegpt")
    ap.add_argument("--targets")
    ap.add_argument("--out-dir", default="/mok/claude-mok/quality")
    args = ap.parse_args()
    if args.stage == "target":
        stage_target(args)
    else:
        stage_score(args)
