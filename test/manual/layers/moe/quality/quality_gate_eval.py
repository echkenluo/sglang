"""Phase 2 quality gate evaluator (freeze v2).

Reads the collected artifacts and decides every gate mechanically:
leg 1 A/A validity (GSM8K answers, logprob parsed-float exactness,
longgen serial exactness for D1/D2 and S1/S2), leg 2 fused-vs-split
zero-delta gates (including the teacher-forced argmax chain), leg 3
deployment thresholds against the worse DeepEP leg, leg 4 longgen and
batch determinism.  Emits a gate-by-gate report JSON with an asset SHA
manifest and exits non-zero if any gate fails.
"""

import csv
import hashlib
import json
import math
import os
import sys

QDIR = sys.argv[1] if len(sys.argv) > 1 else "/mok/claude-mok/quality"
GATES = []


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def gate(name, ok, detail=""):
    GATES.append({"gate": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print(f"GATE|{name}|{'PASS' if ok else 'FAIL'}|{detail}", flush=True)


def load_gsm8k(tag):
    rows = {}
    with open(f"{QDIR}/gsm8k-{tag}.csv") as f:
        for row in csv.DictReader(f):
            rows[int(row["idx"])] = (row["pred"], int(row["correct"]))
    summary = json.load(open(f"{QDIR}/gsm8k-{tag}.json"))
    return rows, summary


def gsm8k_exact(tag_a, tag_b):
    a, _ = load_gsm8k(tag_a)
    b, _ = load_gsm8k(tag_b)
    if set(a) != set(b):
        return False, "question sets differ"
    diff = [i for i in a if a[i][0] != b[i][0]]
    return not diff, f"pred_mismatches={len(diff)} first={diff[:5]}"


def load_score(tag):
    return json.load(open(f"{QDIR}/logprob-score-{tag}.json"))


def score_exact(tag_a, tag_b):
    a, b = load_score(tag_a), load_score(tag_b)
    if len(a) != len(b):
        return False, "row count differs"
    bad = 0
    for ra, rb in zip(a, b):
        if ra["logprobs"] != rb["logprobs"]:
            bad += 1
    return bad == 0, f"rows_with_delta={bad}/128"


def score_delta_stats(tag_a, tag_b):
    a, b = load_score(tag_a), load_score(tag_b)
    deltas, flips = [], 0
    for ra, rb in zip(a, b):
        for x, y in zip(ra["logprobs"], rb["logprobs"]):
            deltas.append(abs(x - y))
        for x, y in zip(ra["top1_ids"], rb["top1_ids"]):
            if x != y:
                flips += 1
    deltas.sort()
    n = len(deltas)
    assert n == 8192, f"expected 8192 deltas, got {n}"
    return {
        "mean": sum(deltas) / n,
        "p95": deltas[math.ceil(0.95 * n) - 1],
        "max": deltas[-1],
        "flip_rate": flips / n,
    }


def main():
    manifest = {}
    for name in sorted(os.listdir(QDIR)):
        if name.endswith((".json", ".csv")):
            manifest[name] = sha(f"{QDIR}/{name}")

    # --- Leg 1: A/A validity ---
    for a, b, label in (("d1", "d2", "deepep"), ("s1", "s2", "split")):
        ok, detail = gsm8k_exact(a, b)
        gate(f"1a-gsm8k-aa-{label}", ok, detail)
        ok, detail = score_exact(a, b)
        gate(f"1b-logprob-aa-{label}", ok, detail)
        la = json.load(open(f"{QDIR}/longgen-{a}.json"))
        lb = json.load(open(f"{QDIR}/longgen-{b}.json"))
        gate(f"1c-longgen-aa-{label}", la["serial"] == lb["serial"])

    # --- Leg 2: fused vs split (zero-delta) ---
    ok, detail = gsm8k_exact("f1", "s1")
    gate("2a-gsm8k-fused-vs-split", ok, detail)
    ok, detail = score_exact("f1", "s1")
    gate("2b-logprob-fused-vs-split-exact", ok, detail)
    targets = json.load(open(f"{QDIR}/logprob-targets.json"))
    s1 = load_score("s1")
    f1 = load_score("f1")
    split_matches_target = all(
        r["top1_ids"] == t["target_ids"] for r, t in zip(s1, targets)
    )
    fused_matches_split = all(
        rf["top1_ids"] == rs["top1_ids"] for rf, rs in zip(f1, s1)
    )
    gate("2b-argmax-chain",
         split_matches_target and fused_matches_split,
         f"split_vs_target={split_matches_target} "
         f"fused_vs_split={fused_matches_split}")

    # --- Leg 3: fused vs deepep (deployment gate) ---
    worst = None
    for dtag in ("d1", "d2"):
        st = score_delta_stats("f1", dtag)
        if worst is None or st["mean"] > worst["mean"]:
            worst = st
    gate("3b-logprob-mean", worst["mean"] <= 0.02, f"{worst['mean']:.6f}")
    gate("3b-logprob-p95", worst["p95"] <= 0.10, f"{worst['p95']:.6f}")
    gate("3b-logprob-max", worst["max"] <= 0.20, f"{worst['max']:.6f}")
    print(f"INFO|top1_flip_rate={worst['flip_rate']:.6f}", flush=True)
    _, sf = load_gsm8k("f1")
    _, sd1 = load_gsm8k("d1")
    _, sd2 = load_gsm8k("d2")
    ref = (sd1["accuracy"] + sd2["accuracy"]) / 2
    gate("3a-gsm8k-accuracy",
         sf["accuracy"] >= ref - 0.01,
         f"fused={sf['accuracy']:.4f} deepep_mean={ref:.4f}")

    # --- Leg 4: long generation + determinism ---
    lf = json.load(open(f"{QDIR}/longgen-f1.json"))
    ls = json.load(open(f"{QDIR}/longgen-s1.json"))
    ld = json.load(open(f"{QDIR}/longgen-d1.json"))
    gate("4a-longgen-fused-vs-split", lf["serial"] == ls["serial"])
    div = []
    for i, (fa, da) in enumerate(zip(lf["serial"], ld["serial"])):
        first = next((k for k, (x, y) in enumerate(zip(fa, da)) if x != y), None)
        div.append(first)
    print(f"INFO|fused_vs_deepep_first_divergence={div}", flush=True)
    gate("4b-batch-determinism", lf["waves_exact"])

    report = {"gates": GATES, "manifest": manifest,
              "all_pass": all(g["pass"] for g in GATES)}
    out = f"{QDIR}/quality-gate-verdict.json"
    json.dump(report, open(out, "w"), indent=1)
    print(f"QUALITY_GATE_VERDICT|all_pass={report['all_pass']}|report_sha={sha(out)}",
          flush=True)
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
