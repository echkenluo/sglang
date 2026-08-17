"""GSM8K gate client (Phase 2 harness).

Frozen protocol: temperature=0, max_new_tokens=512, wave=16, fixed 5-shot
prefix from the first five rows, rows 6..1319 scored (1314 questions),
answers extracted as the normalized number after "####".  A stop string
cuts generation at the next question boundary; it applies identically to
every config.  Outputs a per-question CSV and a summary JSON with SHAs.
"""

import argparse
import concurrent.futures as futures
import hashlib
import json
import re
import urllib.request

ANSWER_RE = re.compile(r"####\s*([\-0-9,.$]+)")


def norm(ans: str) -> str:
    return ans.replace(",", "").replace("$", "").rstrip(".").strip()


def build_prompts(path):
    rows = [json.loads(l) for l in open(path)]
    shots = rows[:5]
    prefix = ""
    for r in shots:
        prefix += (
            "Question: " + r["question"] + "\nAnswer: " + r["answer"] + "\n\n"
        )
    items = []
    for i, r in enumerate(rows[5:]):
        gold = norm(ANSWER_RE.search(r["answer"]).group(1))
        items.append(
            (i, prefix + "Question: " + r["question"] + "\nAnswer:", gold)
        )
    return items


def ask(port, prompt):
    body = json.dumps({
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 512,
            "stop": ["Question:"],
        },
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wave", type=int, default=16)
    ap.add_argument("--out-dir", default="/mok/claude-mok/quality")
    args = ap.parse_args()

    items = build_prompts(args.data)
    if args.limit:
        items = items[: args.limit]
    else:
        assert len(items) == 1314, f"expected 1314 questions, got {len(items)}"
    results = {}
    with futures.ThreadPoolExecutor(max_workers=args.wave) as pool:
        futs = {pool.submit(ask, args.port, p): (i, gold)
                for i, p, gold in items}
        for fut in futures.as_completed(futs):
            i, gold = futs[fut]
            text = fut.result()
            m = ANSWER_RE.search(text)
            pred = norm(m.group(1)) if m else "NO_ANSWER"
            results[i] = (pred, gold, int(pred == gold),
                          hashlib.sha256(text.encode()).hexdigest()[:16])
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = f"{args.out_dir}/gsm8k-{args.tag}.csv"
    with open(csv_path, "w") as f:
        f.write("idx,pred,gold,correct,text_sha\n")
        for i in sorted(results):
            p, g, c, h = results[i]
            f.write(f"{i},{p},{g},{c},{h}\n")
    n = len(results)
    correct = sum(v[2] for v in results.values())
    summary = {
        "tag": args.tag, "n": n, "correct": correct,
        "accuracy": correct / n,
        "data_sha": hashlib.sha256(open(args.data, "rb").read()).hexdigest()[:16],
        "csv_sha": hashlib.sha256(open(csv_path, "rb").read()).hexdigest()[:16],
    }
    with open(f"{args.out_dir}/gsm8k-{args.tag}.json", "w") as f:
        json.dump(summary, f)
    print(f"GSM8K_SUMMARY|{json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
