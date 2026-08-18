"""Phase 2 v3 quality evaluator: empirical drift and non-inferiority.

DeepEP and split each have three independent repeats. Their pairwise
distances define the natural A/A band. Fused must remain inside that band
relative to split and must not worsen split-vs-DeepEP by more than the same
band. Direct MoK/DeepGEMM numeric audits remain a separate hard gate.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import statistics
import sys

QDIR = sys.argv[1] if len(sys.argv) > 1 else "/mok/claude-mok/quality"
EXPECT_SGLANG_HEAD = sys.argv[2] if len(sys.argv) > 2 else ""
CHECKPOINT = sys.argv[3] if len(sys.argv) > 3 else ""
GATES: list[dict[str, object]] = []

DEEP_TAGS = ("d1", "d2", "d3")
SPLIT_TAGS = ("s1", "s2", "s3")
SCORE_METRICS = ("mean", "p95", "max", "flip_rate")


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def gate(name: str, ok: bool, detail: object = "") -> None:
    GATES.append({"gate": name, "pass": bool(ok), "detail": str(detail)[:800]})
    print(f"GATE|{name}|{'PASS' if ok else 'FAIL'}|{detail}", flush=True)


def load_gsm8k(tag: str) -> tuple[dict[int, tuple[str, int]], dict]:
    rows = {}
    with open(f"{QDIR}/gsm8k-{tag}.csv") as input_file:
        for row in csv.DictReader(input_file):
            rows[int(row["idx"])] = (row["pred"], int(row["correct"]))
    summary = json.load(open(f"{QDIR}/gsm8k-{tag}.json"))
    assert summary["n"] == 1314 and len(rows) == 1314
    return rows, summary


def gsm8k_distance(tag_a: str, tag_b: str) -> dict[str, float]:
    a, _ = load_gsm8k(tag_a)
    b, _ = load_gsm8k(tag_b)
    assert set(a) == set(b)
    mismatches = sum(a[index][0] != b[index][0] for index in a)
    return {"mismatch_rate": mismatches / 1314, "mismatches": mismatches}


def load_score(tag: str) -> list[dict]:
    rows = json.load(open(f"{QDIR}/logprob-score-{tag}.json"))
    assert len(rows) == 128
    for row in rows:
        assert len(row["logprobs"]) == 64
        assert len(row["token_ids"]) == 64
        assert len(row["top1_ids"]) == 64
        assert all(math.isfinite(value) for value in row["logprobs"])
    return rows


def score_delta_stats(tag_a: str, tag_b: str) -> dict[str, float]:
    a, b = load_score(tag_a), load_score(tag_b)
    deltas: list[float] = []
    flips = 0
    for row_a, row_b in zip(a, b):
        assert row_a["token_ids"] == row_b["token_ids"]
        deltas.extend(
            abs(value_a - value_b)
            for value_a, value_b in zip(row_a["logprobs"], row_b["logprobs"])
        )
        flips += sum(
            value_a != value_b
            for value_a, value_b in zip(row_a["top1_ids"], row_b["top1_ids"])
        )
    deltas.sort()
    assert len(deltas) == 8192
    return {
        "mean": sum(deltas) / len(deltas),
        "p95": deltas[math.ceil(0.95 * len(deltas)) - 1],
        "max": deltas[-1],
        "flip_rate": flips / len(deltas),
    }


def load_longgen(tag: str) -> dict:
    payload = json.load(open(f"{QDIR}/longgen-{tag}.json"))
    assert len(payload["serial"]) == 16
    assert len(payload["waves"]) == 3
    for collection in [payload["serial"], *payload["waves"]]:
        assert len(collection) == 16
        assert all(len(tokens) == 512 for tokens in collection)
    return payload


def sequence_distance(a: list[list[int]], b: list[list[int]]) -> dict[str, float]:
    assert len(a) == len(b) == 16
    mismatched_tokens = 0
    divergent_prompts = 0
    for tokens_a, tokens_b in zip(a, b):
        assert len(tokens_a) == len(tokens_b) == 512
        row_mismatches = sum(x != y for x, y in zip(tokens_a, tokens_b))
        mismatched_tokens += row_mismatches
        divergent_prompts += row_mismatches > 0
    return {
        "token_mismatch_rate": mismatched_tokens / (16 * 512),
        "prompt_divergence_rate": divergent_prompts / 16,
    }


def longgen_distance(tag_a: str, tag_b: str) -> dict[str, float]:
    return sequence_distance(
        load_longgen(tag_a)["serial"], load_longgen(tag_b)["serial"]
    )


def wave_distance(tag: str) -> dict[str, float]:
    waves = load_longgen(tag)["waves"]
    pair_stats = [
        sequence_distance(waves[left], waves[right])
        for left, right in itertools.combinations(range(3), 2)
    ]
    return {
        metric: max(stats[metric] for stats in pair_stats)
        for metric in ("token_mismatch_rate", "prompt_divergence_rate")
    }


def pairwise(tags: tuple[str, ...], distance) -> list[dict[str, float]]:
    return [distance(a, b) for a, b in itertools.combinations(tags, 2)]


def max_band(stats: list[dict[str, float]]) -> dict[str, float]:
    return {key: max(item[key] for item in stats) for key in stats[0]}


def merged_band(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: max(left[key], right[key]) for key in left}


def median_distance(tag: str, refs: tuple[str, ...], distance) -> dict[str, float]:
    stats = [distance(tag, ref) for ref in refs]
    return {key: statistics.median(item[key] for item in stats) for key in stats[0]}


def check_aa_family(label: str, tags: tuple[str, ...]) -> dict[str, dict[str, float]]:
    gsm_stats = pairwise(tags, gsm8k_distance)
    score_stats = pairwise(tags, score_delta_stats)
    long_stats = pairwise(tags, longgen_distance)
    wave_stats = [wave_distance(tag) for tag in tags]
    gsm_band = max_band(gsm_stats)
    score_band = max_band(score_stats)
    long_band = max_band(long_stats)
    wave_band = max_band(wave_stats)

    accuracies = [load_gsm8k(tag)[1]["accuracy"] for tag in tags]
    gate(
        f"1a-{label}-accuracy-span",
        max(accuracies) - min(accuracies) <= 0.01,
        f"values={accuracies}",
    )
    # Broad admission caps prevent an unbounded environment from making every
    # later non-inferiority comparison vacuous. They are not quality claims.
    gate(
        f"1b-{label}-score-admission",
        score_band["mean"] <= 0.08
        and score_band["p95"] <= 0.30
        and score_band["flip_rate"] <= 0.02,
        score_band,
    )
    gate(
        f"1c-{label}-longgen-admission",
        all(
            math.isfinite(value) and 0 <= value <= 1
            for value in [*long_band.values(), *wave_band.values()]
        ),
        {"serial": long_band, "waves": wave_band},
    )
    return {
        "gsm": gsm_band,
        "score": score_band,
        "longgen": long_band,
        "waves": wave_band,
    }


def emit_checkpoint(name: str, bands: dict) -> int:
    report = {
        "checkpoint": name,
        "gates": GATES,
        "bands": bands,
        "all_pass": all(item["pass"] for item in GATES),
    }
    output = f"{QDIR}/quality-gate-checkpoint-{name}.json"
    json.dump(report, open(output, "w"), indent=1)
    print(
        f"QUALITY_GATE_CHECKPOINT|name={name}|all_pass={report['all_pass']}"
        f"|report_sha={sha(output)}",
        flush=True,
    )
    return 0 if report["all_pass"] else 1


def check_receipts() -> None:
    expected = {("split", "t-s", "logprob-target")}
    for mode, tags in (
        ("deepep", DEEP_TAGS),
        ("split", SPLIT_TAGS),
        ("fused", ("f1",)),
    ):
        for tag in tags:
            for stage in ("gsm8k", "logprob-score", "longgen"):
                expected.add((mode, tag, stage))
    seen = set()
    bad = []
    for name in sorted(os.listdir(QDIR)):
        if not name.startswith("path-receipt-"):
            continue
        receipt = json.load(open(f"{QDIR}/{name}"))
        key = (receipt.get("mode"), receipt.get("tag"), receipt.get("stage"))
        want_name = (
            f"path-receipt-{receipt.get('tag')}-{receipt.get('stage')}.json"
        )
        if (
            name != want_name
            or receipt.get("rc") != 0
            or receipt.get("path_ok") != 1
            or key not in expected
        ):
            bad.append(name)
        seen.add(key)
    gate(
        "path-receipts-22",
        not bad and seen == expected,
        f"seen={len(seen)}/22 bad={bad[:3]} missing={sorted(expected-seen)[:3]}",
    )


def check_preflight(manifest: dict[str, str]) -> None:
    path = f"{QDIR}/preflight-manifest.json"
    if not os.path.exists(path):
        gate("preflight-manifest", False, "missing")
        return
    payload = json.load(open(path))
    expected = json.load(
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected_assets.json"))
    )
    ok = (
        payload.get("verified") is True
        and payload.get("mok_head") == expected["mok_head"]
        and payload.get("so_content_md5s") == expected["so_content_md5s"]
        and payload.get("image_id") == expected["image_id"]
        and payload.get("sharegpt_sha256") == expected["sharegpt_sha256"]
        and payload.get("gsm8k_sha256") == expected["gsm8k_sha256"]
        and payload.get("numeric_audit", {}).get("verified") is True
        and len(EXPECT_SGLANG_HEAD) == 40
        and payload.get("sglang_head") == EXPECT_SGLANG_HEAD
    )
    gate(
        "preflight-manifest",
        ok,
        f"verified={payload.get('verified')} failures={payload.get('failures')}",
    )
    numeric = payload.get("numeric_audit", {})
    gate(
        "kernel-numeric-audit",
        numeric.get("min_exact", 0) >= 0.999
        and numeric.get("max_relative_l2", math.inf) <= 1e-3
        and numeric.get("live_records") == 344
        and numeric.get("live_layers") == 43,
        numeric,
    )
    manifest["preflight-manifest.json"] = sha(path)


def main() -> int:
    manifest = {
        name: sha(f"{QDIR}/{name}")
        for name in sorted(os.listdir(QDIR))
        if name.endswith((".json", ".csv"))
    }

    if CHECKPOINT:
        if CHECKPOINT == "deepep-aa":
            bands = check_aa_family("deepep", DEEP_TAGS)
        elif CHECKPOINT == "split-aa":
            bands = check_aa_family("split", SPLIT_TAGS)
        else:
            raise ValueError(f"unknown checkpoint: {CHECKPOINT}")
        return emit_checkpoint(CHECKPOINT, bands)

    deep_bands = check_aa_family("deepep", DEEP_TAGS)
    split_bands = check_aa_family("split", SPLIT_TAGS)
    bands = {
        family: merged_band(deep_bands[family], split_bands[family])
        for family in deep_bands
    }

    # Leg 2: fusion must look no worse than natural A/A drift relative to split.
    fused_split_score = median_distance("f1", SPLIT_TAGS, score_delta_stats)
    for metric in ("mean", "p95", "flip_rate"):
        gate(
            f"2b-fused-split-{metric}",
            fused_split_score[metric] <= bands["score"][metric],
            f"observed={fused_split_score[metric]:.8g} "
            f"band={bands['score'][metric]:.8g}",
        )
    print(
        f"INFO|fused_split_max={fused_split_score['max']:.8g}|"
        f"aa_max_band={bands['score']['max']:.8g}",
        flush=True,
    )
    fused_split_gsm = median_distance("f1", SPLIT_TAGS, gsm8k_distance)
    split_accuracy = statistics.mean(
        load_gsm8k(tag)[1]["accuracy"] for tag in SPLIT_TAGS
    )
    fused_accuracy = load_gsm8k("f1")[1]["accuracy"]
    gate(
        "2a-fused-split-gsm8k",
        fused_accuracy >= split_accuracy - 0.01
        and fused_split_gsm["mismatch_rate"] <= bands["gsm"]["mismatch_rate"],
        f"fused={fused_accuracy:.6f} split_mean={split_accuracy:.6f} "
        f"mismatch={fused_split_gsm['mismatch_rate']:.6f} "
        f"band={bands['gsm']['mismatch_rate']:.6f}",
    )
    fused_split_long = median_distance("f1", SPLIT_TAGS, longgen_distance)
    gate(
        "2c-fused-split-longgen",
        fused_split_long["token_mismatch_rate"]
        <= bands["longgen"]["token_mismatch_rate"],
        f"observed={fused_split_long} band={bands['longgen']}",
    )
    fused_waves = wave_distance("f1")
    gate(
        "2d-fused-wave-determinism",
        fused_waves["token_mismatch_rate"]
        <= bands["waves"]["token_mismatch_rate"],
        f"observed={fused_waves} band={bands['waves']}",
    )

    # Leg 3: fusion may not add more than A/A drift to the base MoK-vs-DeepEP gap.
    deployment_excess = {metric: 0.0 for metric in SCORE_METRICS}
    deployment_observed = []
    for deep_tag in DEEP_TAGS:
        fused_deep = score_delta_stats("f1", deep_tag)
        split_deep = [score_delta_stats(split, deep_tag) for split in SPLIT_TAGS]
        baseline = {
            metric: statistics.median(item[metric] for item in split_deep)
            for metric in SCORE_METRICS
        }
        deployment_observed.append(
            {"deep": deep_tag, "fused": fused_deep, "split_median": baseline}
        )
        for metric in SCORE_METRICS:
            deployment_excess[metric] = max(
                deployment_excess[metric], fused_deep[metric] - baseline[metric]
            )
    for metric in SCORE_METRICS:
        gate(
            f"3b-deployment-excess-{metric}",
            deployment_excess[metric] <= bands["score"][metric],
            f"excess={deployment_excess[metric]:.8g} "
            f"band={bands['score'][metric]:.8g}",
        )
    print(f"INFO|deployment_score={deployment_observed}", flush=True)

    deep_accuracy = statistics.mean(
        load_gsm8k(tag)[1]["accuracy"] for tag in DEEP_TAGS
    )
    gate(
        "3a-gsm8k-noninferiority",
        fused_accuracy >= deep_accuracy - 0.01
        and fused_accuracy >= split_accuracy - 0.01,
        f"fused={fused_accuracy:.6f} deep={deep_accuracy:.6f} "
        f"split={split_accuracy:.6f}",
    )

    deployment_long_excess = 0.0
    for deep_tag in DEEP_TAGS:
        fused_deep = longgen_distance("f1", deep_tag)["token_mismatch_rate"]
        split_baseline = statistics.median(
            longgen_distance(split, deep_tag)["token_mismatch_rate"]
            for split in SPLIT_TAGS
        )
        deployment_long_excess = max(
            deployment_long_excess, fused_deep - split_baseline
        )
    gate(
        "3c-longgen-deployment-excess",
        deployment_long_excess <= bands["longgen"]["token_mismatch_rate"],
        f"excess={deployment_long_excess:.8g} "
        f"band={bands['longgen']['token_mismatch_rate']:.8g}",
    )

    check_receipts()
    check_preflight(manifest)

    report = {
        "protocol": "phase2-v3-natural-drift-noninferiority",
        "gates": GATES,
        "aa_bands": bands,
        "manifest": manifest,
        "all_pass": all(item["pass"] for item in GATES),
    }
    output = f"{QDIR}/quality-gate-verdict.json"
    json.dump(report, open(output, "w"), indent=1)
    print(
        f"QUALITY_GATE_VERDICT|all_pass={report['all_pass']}|"
        f"report_sha={sha(output)}",
        flush=True,
    )
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
