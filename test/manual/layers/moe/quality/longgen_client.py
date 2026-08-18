"""Phase 2 v4 free-run information client.

The frozen first 16 prompt-token sequences are generated once serially and as
one explicit 16-prompt batch three times.  Divergence is recorded, never gated:
any complete 512-token outputs produce a successful client result.
"""

import argparse
import json
from pathlib import Path
import urllib.request

import logprob_client as teacher_client


PROMPT_COUNT = 16
OUTPUT_LENGTH = 512


def gen(port, input_ids):
    body = json.dumps(
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": OUTPUT_LENGTH,
                "ignore_eos": True,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        payload = json.load(resp)
    if isinstance(payload, list):
        return [item["output_ids"] for item in payload]
    return payload["output_ids"]


def _validate_run(run, label):
    if not isinstance(run, list) or len(run) != PROMPT_COUNT:
        raise ValueError(f"{label} must contain exactly {PROMPT_COUNT} outputs")
    for prompt_index, output_ids in enumerate(run):
        teacher_client._require_token_ids(
            output_ids,
            OUTPUT_LENGTH,
            f"{label} prompt {prompt_index}",
        )


def compare_runs(left, right, prompt_refs):
    """Describe token differences without assigning any quality verdict."""
    _validate_run(left, "left comparison run")
    _validate_run(right, "right comparison run")
    if len(prompt_refs) != PROMPT_COUNT:
        raise ValueError(f"prompt_refs must contain {PROMPT_COUNT} prompts")

    prompt_divergence_count = 0
    token_divergence_count = 0
    per_prompt_first_divergence = []
    first_divergence = None
    for prompt_index, (left_ids, right_ids) in enumerate(zip(left, right)):
        positions = [
            position
            for position, (left_id, right_id) in enumerate(
                zip(left_ids, right_ids)
            )
            if left_id != right_id
        ]
        token_divergence_count += len(positions)
        first_position = positions[0] if positions else None
        per_prompt_first_divergence.append(first_position)
        if first_position is None:
            continue
        prompt_divergence_count += 1
        if first_divergence is None:
            first_divergence = {
                "prompt_index": prompt_index,
                "prompt_id": prompt_refs[prompt_index]["prompt_id"],
                "token_index": first_position,
                "left_token_id": left_ids[first_position],
                "right_token_id": right_ids[first_position],
            }
    return {
        "prompt_divergence_count": prompt_divergence_count,
        "token_divergence_count": token_divergence_count,
        "token_divergence_rate": token_divergence_count
        / (PROMPT_COUNT * OUTPUT_LENGTH),
        "per_prompt_first_divergence": per_prompt_first_divergence,
        "first_divergence": first_divergence,
    }


def validate_free_run(document):
    if document.get("protocol") != teacher_client.PROTOCOL:
        raise ValueError("free-run protocol mismatch")
    if document.get("kind") != "free-run-info":
        raise ValueError("free-run kind must be free-run-info")
    if document.get("mode") not in {"deepep", "split", "fused"}:
        raise ValueError("free-run mode must be deepep, split, or fused")
    if not isinstance(document.get("tag"), str) or not document["tag"]:
        raise ValueError("free-run tag must be nonempty")
    for field in ("targets_sha256", "path_config_receipt_sha256"):
        if not teacher_client._is_sha256(document.get(field)):
            raise ValueError(f"free-run {field} must be a full SHA256")

    prompts = document.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != PROMPT_COUNT:
        raise ValueError(f"free-run must contain {PROMPT_COUNT} prompt refs")
    for prompt_index, prompt in enumerate(prompts):
        if not teacher_client._is_token_id(prompt.get("prompt_id")):
            raise ValueError(f"invalid prompt_id at free-run row {prompt_index}")
        if not teacher_client._is_sha256(prompt.get("prompt_sha256")):
            raise ValueError(f"invalid prompt SHA at free-run row {prompt_index}")
        prompt_ids = prompt.get("prompt_ids")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError(f"missing prompt IDs at free-run row {prompt_index}")
        teacher_client._require_token_ids(
            prompt_ids, len(prompt_ids), f"free-run row {prompt_index} prompt_ids"
        )
        if teacher_client.ids_sha256(prompt_ids) != prompt["prompt_sha256"]:
            raise ValueError(f"prompt SHA mismatch at free-run row {prompt_index}")

    serial = document.get("serial")
    waves = document.get("waves")
    _validate_run(serial, "serial")
    if not isinstance(waves, list) or len(waves) != 3:
        raise ValueError("free-run must contain exactly three waves")
    for wave_index, wave in enumerate(waves):
        _validate_run(wave, f"wave {wave_index}")

    expected_pairs = {
        "serial_vs_wave1",
        "serial_vs_wave2",
        "serial_vs_wave3",
        "wave1_vs_wave2",
        "wave1_vs_wave3",
        "wave2_vs_wave3",
    }
    first_divergence = document.get("first_divergence")
    info = document.get("info")
    if not isinstance(first_divergence, dict) or set(first_divergence) != expected_pairs:
        raise ValueError("free-run first_divergence pair set mismatch")
    if not isinstance(info, dict) or set(info) != expected_pairs:
        raise ValueError("free-run info pair set mismatch")
    expected_comparisons = {
        "serial_vs_wave1": compare_runs(serial, waves[0], prompts),
        "serial_vs_wave2": compare_runs(serial, waves[1], prompts),
        "serial_vs_wave3": compare_runs(serial, waves[2], prompts),
        "wave1_vs_wave2": compare_runs(waves[0], waves[1], prompts),
        "wave1_vs_wave3": compare_runs(waves[0], waves[2], prompts),
        "wave2_vs_wave3": compare_runs(waves[1], waves[2], prompts),
    }
    for pair in expected_pairs:
        comparison = info[pair]
        if comparison != expected_comparisons[pair]:
            raise ValueError(f"free-run divergence info mismatch for {pair}")
        if comparison.get("first_divergence") != first_divergence[pair]:
            raise ValueError(f"free-run first-divergence mismatch for {pair}")
        prompt_count = comparison.get("prompt_divergence_count")
        token_count = comparison.get("token_divergence_count")
        if not isinstance(prompt_count, int) or not 0 <= prompt_count <= PROMPT_COUNT:
            raise ValueError(f"invalid prompt divergence count for {pair}")
        if (
            not isinstance(token_count, int)
            or not 0 <= token_count <= PROMPT_COUNT * OUTPUT_LENGTH
        ):
            raise ValueError(f"invalid token divergence count for {pair}")
    return document


def build_free_run(args):
    with open(args.targets, encoding="utf-8") as stream:
        targets = json.load(stream)
    teacher_client.validate_targets(targets, args.context_length)
    target_rows = targets["rows"][:PROMPT_COUNT]
    prompts = [row["prompt_ids"] for row in target_rows]
    prompt_refs = [
        {
            "prompt_id": row["prompt_id"],
            "prompt_sha256": row["prompt_sha256"],
            "prompt_ids": row["prompt_ids"],
        }
        for row in target_rows
    ]
    if len(prompt_refs) != PROMPT_COUNT:
        raise ValueError(f"expected {PROMPT_COUNT} prompts, got {len(prompt_refs)}")

    serial = [gen(args.port, prompt) for prompt in prompts]
    _validate_run(serial, "serial")
    waves = [gen(args.port, prompts) for _ in range(3)]
    for wave_index, wave in enumerate(waves):
        _validate_run(wave, f"wave {wave_index}")

    comparisons = {
        "serial_vs_wave1": compare_runs(serial, waves[0], prompt_refs),
        "serial_vs_wave2": compare_runs(serial, waves[1], prompt_refs),
        "serial_vs_wave3": compare_runs(serial, waves[2], prompt_refs),
        "wave1_vs_wave2": compare_runs(waves[0], waves[1], prompt_refs),
        "wave1_vs_wave3": compare_runs(waves[0], waves[2], prompt_refs),
        "wave2_vs_wave3": compare_runs(waves[1], waves[2], prompt_refs),
    }
    document = {
        "protocol": teacher_client.PROTOCOL,
        "kind": "free-run-info",
        "mode": args.mode,
        "tag": args.tag,
        "targets_sha256": teacher_client.sha256_file(args.targets),
        "path_config_receipt_sha256": teacher_client._receipt_sha(
            args.path_config_receipt
        ),
        "prompts": prompt_refs,
        "serial": serial,
        "waves": waves,
        "first_divergence": {
            name: comparison["first_divergence"]
            for name, comparison in comparisons.items()
        },
        "info": comparisons,
    }
    validate_free_run(document)
    return document


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=["deepep", "split", "fused"], required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--path-config-receipt", required=True)
    parser.add_argument(
        "--context-length",
        type=int,
        default=teacher_client.DEFAULT_CONTEXT_LENGTH,
    )
    parser.add_argument("--out-dir", default="/mok/claude-mok/quality")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    document = build_free_run(args)
    out = Path(args.out_dir) / f"free-run-info-{args.tag}.json"
    teacher_client.atomic_json_freeze(out, document)
    sha = teacher_client.sha256_file(out)
    prompt_divergences = sum(
        item["prompt_divergence_count"] for item in document["info"].values()
    )
    token_divergences = sum(
        item["token_divergence_count"] for item in document["info"].values()
    )
    print(
        f"FREE_RUN_INFO|mode={args.mode}|tag={args.tag}"
        f"|prompt_divergences={prompt_divergences}"
        f"|token_divergences={token_divergences}|sha256={sha}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
