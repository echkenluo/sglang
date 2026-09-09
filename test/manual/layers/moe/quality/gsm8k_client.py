"""GSM8K gate client (Phase 2 harness).

Frozen protocol: temperature=0, max_new_tokens=512, wave=16, fixed 5-shot
prefix from the first five rows, rows 6..1319 scored (1314 questions),
answers extracted as the normalized number after "####".  A stop string
cuts generation at the next question boundary; it applies identically to
every config.  Outputs a per-question CSV and a summary JSON with SHAs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.request

ANSWER_RE = re.compile(r"####\s*([\-0-9,.$]+)")
REQUEST_BATCHING_SCHEMA = "phase2-v4-request-batching-v1"
BATCH_SIZE = 16


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value, label):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a full SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a full SHA256") from error
    return value.lower()


def atomic_freeze_bytes(path, payload):
    """Publish an immutable artifact without a check-then-overwrite race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
        temporary_path.unlink()
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


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


def _chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _batching_contract():
    return {
        "schema": REQUEST_BATCHING_SCHEMA,
        "generate_endpoint": "/generate",
        "generate_batch_size": BATCH_SIZE,
        "response_identity": "request-order-and-rid-echo",
    }


def ask_batch(port, items, tag):
    rids = [f"p2v4-gsm8k-{tag}-{index:04d}" for index, _prompt, _gold in items]
    body = json.dumps({
        "text": [prompt for _index, prompt, _gold in items],
        "rid": rids,
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
        response = json.load(resp)
    if not isinstance(response, list) or len(response) != len(items):
        observed = len(response) if isinstance(response, list) else type(response).__name__
        raise ValueError(
            f"GSM8K batch response must contain exactly {len(items)} items, "
            f"got {observed}"
        )
    outputs = []
    for offset, (item, expected_rid) in enumerate(zip(response, rids)):
        if not isinstance(item, dict):
            raise ValueError(f"GSM8K response item {offset} must be an object")
        meta = item.get("meta_info")
        if not isinstance(meta, dict) or meta.get("id") != expected_rid:
            observed_rid = meta.get("id") if isinstance(meta, dict) else None
            raise ValueError(
                f"GSM8K response order/RID mismatch at {offset}: "
                f"expected {expected_rid}, got {observed_rid}"
            )
        text = item.get("text")
        if not isinstance(text, str):
            raise ValueError(f"GSM8K response text {offset} must be a string")
        outputs.append(text)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset-sha256", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wave", type=int, default=16)
    ap.add_argument("--out-dir", default="/mok/claude-mok/quality")
    args = ap.parse_args()

    if args.wave != BATCH_SIZE:
        raise ValueError(
            f"Phase 2 v4 freezes GSM8K batch size at {BATCH_SIZE}; got {args.wave}"
        )

    dataset_sha256 = sha256_file(args.data)
    expected_dataset_sha256 = _require_sha256(
        args.dataset_sha256, "dataset_sha256"
    )
    if dataset_sha256 != expected_dataset_sha256:
        raise ValueError(
            "GSM8K dataset SHA mismatch: "
            f"expected {expected_dataset_sha256}, got {dataset_sha256}"
        )

    items = build_prompts(args.data)
    if args.limit:
        items = items[: args.limit]
    else:
        assert len(items) == 1314, f"expected 1314 questions, got {len(items)}"
    results = {}
    for item_batch in _chunks(items, BATCH_SIZE):
        texts = ask_batch(args.port, item_batch, args.tag)
        if len(texts) != len(item_batch):
            raise ValueError("GSM8K parsed batch response count mismatch")
        for (i, _prompt, gold), text in zip(item_batch, texts):
            m = ANSWER_RE.search(text)
            pred = norm(m.group(1)) if m else "NO_ANSWER"
            results[i] = (
                pred,
                gold,
                int(pred == gold),
                hashlib.sha256(text.encode()).hexdigest(),
            )
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = f"{args.out_dir}/gsm8k-{args.tag}.csv"
    csv_lines = ["idx,pred,gold,correct,text_sha256\n"]
    for i in sorted(results):
        p, g, c, h = results[i]
        csv_lines.append(f"{i},{p},{g},{c},{h}\n")
    atomic_freeze_bytes(csv_path, "".join(csv_lines).encode())
    n = len(results)
    correct = sum(v[2] for v in results.values())
    summary = {
        "schema": "phase2-v4-gsm8k-v1",
        "tag": args.tag,
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "dataset_sha256": dataset_sha256,
        "csv_sha256": sha256_file(csv_path),
        "request_batching": _batching_contract(),
    }
    summary_path = f"{args.out_dir}/gsm8k-{args.tag}.json"
    atomic_freeze_bytes(
        summary_path,
        (json.dumps(summary, separators=(",", ":")) + "\n").encode(),
    )
    print(f"GSM8K_SUMMARY|{json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
