"""Phase 2 v4 frozen-target and Teacher512 client.

``target`` builds the immutable split-generated trajectory corpus.  ``score``
teacher-forces exactly those token IDs on a D/S/F server and records the 512
target positions.  The client validates every artifact before publishing it;
the target corpus is written through a same-directory temporary file, fsync'd,
and atomically renamed to ``targets-freeze.json``.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import urllib.request


PROTOCOL = "phase2-v4"
SEED = 196944571
TARGET_COUNT = 128
TARGET_LENGTH = 512
PROMPT_MIN_TOKENS = 256
PROMPT_MAX_TOKENS = 2048
DEFAULT_CONTEXT_LENGTH = 32768
TARGETS_FILENAME = "targets-freeze.json"
REQUEST_BATCHING_SCHEMA = "phase2-v4-request-batching-v1"
TOKENIZE_BATCH_SIZE = 128
TARGET_BATCH_SIZE = 16
TEACHER_BATCH_SIZE = 16


def post(port, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.load(resp)


def post_tokenize(port, texts):
    body = {
        "prompt": texts,
        "add_special_tokens": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/tokenize",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.load(resp)


def _chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _batching_contract(kind):
    if kind == "target":
        return {
            "schema": REQUEST_BATCHING_SCHEMA,
            "tokenize_endpoint": "/v1/tokenize",
            "tokenize_batch_size": TOKENIZE_BATCH_SIZE,
            "tokenize_response_identity": "server-request-order-contract",
            "generate_endpoint": "/generate",
            "generate_batch_size": TARGET_BATCH_SIZE,
            "generate_response_identity": "request-order-and-rid-echo",
        }
    if kind == "teacher512":
        return {
            "schema": REQUEST_BATCHING_SCHEMA,
            "generate_endpoint": "/generate",
            "generate_batch_size": TEACHER_BATCH_SIZE,
            "response_identity": "request-order-and-rid-echo",
        }
    raise ValueError(f"unknown batching contract {kind}")


def _validate_batch_response(response, expected_rids, label):
    if not isinstance(response, list) or len(response) != len(expected_rids):
        observed = len(response) if isinstance(response, list) else type(response).__name__
        raise ValueError(
            f"{label} response must contain exactly {len(expected_rids)} items, "
            f"got {observed}"
        )
    for index, (item, expected_rid) in enumerate(zip(response, expected_rids)):
        if not isinstance(item, dict):
            raise ValueError(f"{label} response item {index} must be an object")
        meta = item.get("meta_info")
        if not isinstance(meta, dict) or meta.get("id") != expected_rid:
            observed_rid = meta.get("id") if isinstance(meta, dict) else None
            raise ValueError(
                f"{label} response order/RID mismatch at {index}: "
                f"expected {expected_rid}, got {observed_rid}"
            )
    return response


def _validate_tokenize_response(response, expected_count):
    if not isinstance(response, dict):
        raise ValueError("tokenize response must be an object")
    tokens = response.get("tokens")
    counts = response.get("count")
    max_model_len = response.get("max_model_len")
    if (
        not isinstance(tokens, list)
        or not isinstance(counts, list)
        or len(tokens) != expected_count
        or len(counts) != expected_count
    ):
        raise ValueError(
            f"tokenize response must contain exactly {expected_count} ordered rows"
        )
    if (
        not isinstance(max_model_len, int)
        or isinstance(max_model_len, bool)
    ):
        raise ValueError("tokenize response max_model_len must be an integer")
    validated = []
    for index, (token_ids, count) in enumerate(zip(tokens, counts)):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"tokenize count {index} is invalid")
        _require_token_ids(token_ids, count, f"tokenize row {index}")
        validated.append(token_ids)
    return validated


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids):
    encoded = json.dumps(ids, separators=(",", ":"), ensure_ascii=True).encode()
    return sha256_bytes(encoded)


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_token_id(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _require_token_ids(values, expected_length, label):
    if not isinstance(values, list) or len(values) != expected_length:
        raise ValueError(
            f"{label} must contain exactly {expected_length} token IDs"
        )
    if not all(_is_token_id(value) and value >= 0 for value in values):
        raise ValueError(f"{label} must contain nonnegative integer token IDs")


def sharegpt_prompts(path):
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    for source_index, conv in enumerate(data):
        turns = conv.get("conversations", [])
        if turns and turns[0].get("from") == "human":
            text = turns[0].get("value", "").strip()
            if text:
                yield source_index, text


def validate_targets(document, context_length=DEFAULT_CONTEXT_LENGTH):
    if not isinstance(document, dict):
        raise ValueError("targets artifact must be a JSON object")
    if document.get("protocol") != PROTOCOL:
        raise ValueError(f"targets protocol must be {PROTOCOL}")
    if document.get("seed") != SEED:
        raise ValueError(f"targets seed must be {SEED}")
    if document.get("request_batching") != _batching_contract("target"):
        raise ValueError("targets request batching contract mismatch")
    for field in (
        "dataset_sha256",
        "tokenizer_sha256",
        "generator_sha256",
    ):
        if not _is_sha256(document.get(field)):
            raise ValueError(f"targets {field} must be a full SHA256")

    rows = document.get("rows")
    if not isinstance(rows, list) or len(rows) != TARGET_COUNT:
        raise ValueError(f"targets must contain exactly {TARGET_COUNT} rows")
    if not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("context_length must be a positive integer")

    prompt_ids_seen = set()
    prompt_hashes_seen = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"target row {row_index} must be an object")
        prompt_id = row.get("prompt_id")
        if not _is_token_id(prompt_id) or prompt_id < 0:
            raise ValueError(f"target row {row_index} has invalid prompt_id")
        if prompt_id in prompt_ids_seen:
            raise ValueError(f"duplicate prompt_id {prompt_id}")
        prompt_ids_seen.add(prompt_id)

        text_sha = row.get("text_sha")
        prompt_sha = row.get("prompt_sha256")
        target_sha = row.get("target_sha256")
        if not _is_sha256(text_sha):
            raise ValueError(f"target row {row_index} has invalid text_sha")
        if not _is_sha256(prompt_sha) or not _is_sha256(target_sha):
            raise ValueError(f"target row {row_index} has invalid ID SHA")
        if prompt_sha in prompt_hashes_seen:
            raise ValueError(f"duplicate prompt IDs at row {row_index}")
        prompt_hashes_seen.add(prompt_sha)

        prompt_len = row.get("prompt_len")
        if (
            not isinstance(prompt_len, int)
            or isinstance(prompt_len, bool)
            or not PROMPT_MIN_TOKENS <= prompt_len <= PROMPT_MAX_TOKENS
        ):
            raise ValueError(f"target row {row_index} has invalid prompt_len")
        prompt_ids = row.get("prompt_ids")
        target_ids = row.get("target_ids")
        _require_token_ids(prompt_ids, prompt_len, f"row {row_index} prompt_ids")
        _require_token_ids(
            target_ids, TARGET_LENGTH, f"row {row_index} target_ids"
        )
        if ids_sha256(prompt_ids) != prompt_sha:
            raise ValueError(f"target row {row_index} prompt SHA mismatch")
        if ids_sha256(target_ids) != target_sha:
            raise ValueError(f"target row {row_index} target SHA mismatch")
        if prompt_len + TARGET_LENGTH > context_length:
            raise ValueError(f"target row {row_index} exceeds context length")
    return document


def atomic_json_freeze(path, document):
    """Durably publish one immutable JSON artifact without clobber races."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(
                document,
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        # link(2) is an atomic no-clobber publication: if another writer has
        # already frozen this path, the kernel returns EEXIST and the winner's
        # inode/content cannot be replaced.  A preflight exists() check plus
        # os.replace() would leave a TOCTOU overwrite window.
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace frozen artifact: {path}"
            ) from error
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


def _required_sha(value, field):
    if not _is_sha256(value):
        raise ValueError(f"{field} must be supplied as a full SHA256")
    return value.lower()


def build_targets(args):
    dataset_sha = sha256_file(args.sharegpt)
    if args.dataset_sha256 is not None:
        expected = _required_sha(args.dataset_sha256, "dataset_sha256")
        if dataset_sha != expected:
            raise ValueError(
                f"ShareGPT SHA mismatch: expected {expected}, got {dataset_sha}"
            )
    tokenizer_sha = _required_sha(args.tokenizer_sha256, "tokenizer_sha256")
    generator_sha = sha256_file(__file__)
    if args.generator_sha256 is not None:
        expected = _required_sha(args.generator_sha256, "generator_sha256")
        if generator_sha != expected:
            raise ValueError(
                f"generator SHA mismatch: expected {expected}, got {generator_sha}"
            )

    candidates = []
    selected_text_hashes = set()
    for prompt_id, text in sharegpt_prompts(args.sharegpt):
        text_sha = sha256_bytes(text.encode("utf-8"))
        if text_sha in selected_text_hashes:
            continue
        selected_text_hashes.add(text_sha)
        candidates.append((prompt_id, text, text_sha))

    selected_prompts = []
    selected_prompt_hashes = set()
    for candidate_batch in _chunks(candidates, TOKENIZE_BATCH_SIZE):
        tokenized = _validate_tokenize_response(
            post_tokenize(args.port, [item[1] for item in candidate_batch]),
            len(candidate_batch),
        )
        for (prompt_id, _text, text_sha), prompt_ids in zip(
            candidate_batch, tokenized
        ):
            prompt_len = len(prompt_ids)
            if not PROMPT_MIN_TOKENS <= prompt_len <= PROMPT_MAX_TOKENS:
                continue
            prompt_sha = ids_sha256(prompt_ids)
            if prompt_sha in selected_prompt_hashes:
                continue
            selected_prompt_hashes.add(prompt_sha)
            selected_prompts.append(
                {
                    "prompt_id": prompt_id,
                    "text_sha": text_sha,
                    "prompt_ids": prompt_ids,
                    "prompt_len": prompt_len,
                    "prompt_sha256": prompt_sha,
                }
            )
            if len(selected_prompts) == TARGET_COUNT:
                break
        if len(selected_prompts) == TARGET_COUNT:
            break
    if len(selected_prompts) != TARGET_COUNT:
        raise ValueError(
            f"only {len(selected_prompts)} unique in-range prompts after tokenization"
        )

    selected = []
    for batch_index, prompt_batch in enumerate(
        _chunks(selected_prompts, TARGET_BATCH_SIZE)
    ):
        rids = [
            f"p2v4-target-{batch_index:02d}-{offset:02d}-{row['prompt_sha256'][:16]}"
            for offset, row in enumerate(prompt_batch)
        ]
        response = _validate_batch_response(
            post(
                args.port,
                {
                    "input_ids": [row["prompt_ids"] for row in prompt_batch],
                    "rid": rids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": TARGET_LENGTH,
                        "ignore_eos": True,
                    },
                },
            ),
            rids,
            "target-build",
        )
        for row, item in zip(prompt_batch, response):
            meta = item["meta_info"]
            if meta.get("prompt_tokens") != row["prompt_len"]:
                raise ValueError(
                    f"target-build prompt length mismatch for {row['prompt_id']}"
                )
            target_ids = item.get("output_ids")
            _require_token_ids(
                target_ids,
                TARGET_LENGTH,
                f"target for prompt {row['prompt_id']}",
            )
            selected.append(
                {
                    **row,
                    "target_ids": target_ids,
                    "target_sha256": ids_sha256(target_ids),
                }
            )

    document = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "dataset_sha256": dataset_sha,
        "tokenizer_sha256": tokenizer_sha,
        "generator_sha256": generator_sha,
        "request_batching": _batching_contract("target"),
        "rows": selected,
    }
    validate_targets(document, args.context_length)
    return document


def stage_target(args):
    if getattr(args, "mode", "split") != "split":
        raise ValueError("target-build must run against split mode")
    document = build_targets(args)
    out = Path(args.out_dir) / TARGETS_FILENAME
    atomic_json_freeze(out, document)
    sha = sha256_file(out)
    print(f"LOGPROB_TARGETS_V4|n={TARGET_COUNT}|sha256={sha}", flush=True)
    return out


def _receipt_sha(path):
    if not path:
        raise ValueError("path_config_receipt is required for formal scoring")
    return sha256_file(path)


def _parse_score_response(row, response):
    target_ids = row["target_ids"]
    meta = response["meta_info"]
    expected_prompt_tokens = len(row["prompt_ids"]) + TARGET_LENGTH
    if meta.get("prompt_tokens") != expected_prompt_tokens:
        raise ValueError(
            f"prompt {row['prompt_id']} scored prompt length mismatch"
        )
    input_logprobs = meta.get("input_token_logprobs")
    if not isinstance(input_logprobs, list):
        raise ValueError(
            f"prompt {row['prompt_id']} is missing input_token_logprobs"
        )
    token_tail = input_logprobs[-TARGET_LENGTH:]
    if len(token_tail) != TARGET_LENGTH:
        raise ValueError(f"prompt {row['prompt_id']} has a short logprob tail")
    token_ids = []
    logprobs = []
    for position, item in enumerate(token_tail):
        if (
            not isinstance(item, (list, tuple))
            or len(item) < 2
            or not isinstance(item[0], (int, float))
            or isinstance(item[0], bool)
            or not math.isfinite(float(item[0]))
            or not _is_token_id(item[1])
            or item[1] < 0
        ):
            raise ValueError(
                f"malformed token logprob for prompt {row['prompt_id']}:{position}"
            )
        logprobs.append(float(item[0]))
        token_ids.append(item[1])
    _require_token_ids(
        token_ids, TARGET_LENGTH, f"prompt {row['prompt_id']} scored token_ids"
    )
    if token_ids != target_ids:
        raise ValueError(f"token-ID misalignment for prompt {row['prompt_id']}")

    top_logprobs = meta.get("input_top_logprobs")
    if not isinstance(top_logprobs, list):
        raise ValueError(f"missing input_top_logprobs for prompt {row['prompt_id']}")
    top_tail = top_logprobs[-TARGET_LENGTH:]
    if len(top_tail) != TARGET_LENGTH:
        raise ValueError(f"prompt {row['prompt_id']} has a short top-1 tail")
    top1_ids = []
    for position, choices in enumerate(top_tail):
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], (list, tuple))
            or len(choices[0]) < 2
            or not _is_token_id(choices[0][1])
            or choices[0][1] < 0
        ):
            raise ValueError(
                f"malformed top-1 at prompt {row['prompt_id']}:{position}"
            )
        top1_ids.append(choices[0][1])
    return {
        "prompt_id": row["prompt_id"],
        "prompt_sha256": row["prompt_sha256"],
        "target_sha256": row["target_sha256"],
        "token_ids": token_ids,
        "logprobs": logprobs,
        "top1_ids": top1_ids,
    }


def score_rows(port, rows):
    scored = []
    for batch_index, row_batch in enumerate(_chunks(rows, TEACHER_BATCH_SIZE)):
        rids = [
            f"p2v4-teacher-{batch_index:02d}-{offset:02d}-{row['prompt_sha256'][:16]}"
            for offset, row in enumerate(row_batch)
        ]
        response = _validate_batch_response(
            post(
                port,
                {
                    "input_ids": [
                        row["prompt_ids"] + row["target_ids"] for row in row_batch
                    ],
                    "rid": rids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 0},
                    "return_logprob": True,
                    "logprob_start_len": [
                        len(row["prompt_ids"]) - 1 for row in row_batch
                    ],
                    "top_logprobs_num": 1,
                },
            ),
            rids,
            "teacher512",
        )
        scored.extend(
            _parse_score_response(row, item)
            for row, item in zip(row_batch, response)
        )
    if len(scored) != len(rows):
        raise ValueError("teacher512 response count mismatch")
    return scored


def validate_teacher(document, targets):
    if document.get("protocol") != PROTOCOL:
        raise ValueError(f"teacher protocol must be {PROTOCOL}")
    if document.get("mode") not in {"deepep", "split", "fused"}:
        raise ValueError("teacher mode must be deepep, split, or fused")
    if not isinstance(document.get("tag"), str) or not document["tag"]:
        raise ValueError("teacher tag must be nonempty")
    if document.get("request_batching") != _batching_contract("teacher512"):
        raise ValueError("teacher request batching contract mismatch")
    for field in ("targets_sha256", "path_config_receipt_sha256"):
        if not _is_sha256(document.get(field)):
            raise ValueError(f"teacher {field} must be a full SHA256")
    rows = document.get("rows")
    target_rows = targets["rows"]
    if not isinstance(rows, list) or len(rows) != TARGET_COUNT:
        raise ValueError(f"teacher must contain exactly {TARGET_COUNT} rows")

    for index, (row, target) in enumerate(zip(rows, target_rows)):
        if row.get("prompt_id") != target["prompt_id"]:
            raise ValueError(f"teacher prompt_id mismatch at row {index}")
        if row.get("prompt_sha256") != target["prompt_sha256"]:
            raise ValueError(f"teacher prompt SHA mismatch at row {index}")
        if row.get("target_sha256") != target["target_sha256"]:
            raise ValueError(f"teacher target SHA mismatch at row {index}")
        _require_token_ids(
            row.get("token_ids"), TARGET_LENGTH, f"teacher row {index} token_ids"
        )
        if row["token_ids"] != target["target_ids"]:
            raise ValueError(f"teacher target IDs mismatch at row {index}")
        logprobs = row.get("logprobs")
        if (
            not isinstance(logprobs, list)
            or len(logprobs) != TARGET_LENGTH
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in logprobs
            )
        ):
            raise ValueError(f"teacher row {index} has invalid logprobs")
        _require_token_ids(
            row.get("top1_ids"), TARGET_LENGTH, f"teacher row {index} top1_ids"
        )
    position_count = sum(len(row["logprobs"]) for row in rows)
    if position_count != TARGET_COUNT * TARGET_LENGTH:
        raise ValueError(
            f"expected {TARGET_COUNT * TARGET_LENGTH} positions, "
            f"got {position_count}"
        )
    return document


def stage_score(args):
    with open(args.targets, encoding="utf-8") as stream:
        targets = json.load(stream)
    validate_targets(targets, args.context_length)
    targets_sha = sha256_file(args.targets)
    receipt_sha = _receipt_sha(args.path_config_receipt)
    rows = score_rows(args.port, targets["rows"])
    document = {
        "protocol": PROTOCOL,
        "mode": args.mode,
        "tag": args.tag,
        "targets_sha256": targets_sha,
        "path_config_receipt_sha256": receipt_sha,
        "request_batching": _batching_contract("teacher512"),
        "rows": rows,
    }
    validate_teacher(document, targets)
    out = Path(args.out_dir) / f"teacher512-{args.tag}.json"
    atomic_json_freeze(out, document)
    sha = sha256_file(out)
    print(
        f"TEACHER512|mode={args.mode}|tag={args.tag}|n={TARGET_COUNT}"
        f"|positions={TARGET_COUNT * TARGET_LENGTH}|sha256={sha}",
        flush=True,
    )
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--stage", choices=["target", "score"], required=True)
    parser.add_argument("--mode", choices=["deepep", "split", "fused"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sharegpt")
    parser.add_argument("--targets")
    parser.add_argument("--path-config-receipt")
    parser.add_argument("--dataset-sha256")
    parser.add_argument("--tokenizer-sha256")
    parser.add_argument("--generator-sha256")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--out-dir", default="/mok/claude-mok/quality")
    args = parser.parse_args(argv)
    if args.stage == "target":
        if args.mode not in (None, "split"):
            parser.error("target stage requires --mode split")
        args.mode = "split"
        if not args.sharegpt:
            parser.error("target stage requires --sharegpt")
        if not args.tokenizer_sha256:
            parser.error("target stage requires --tokenizer-sha256")
    else:
        if not args.mode:
            parser.error("score stage requires --mode")
        if not args.targets:
            parser.error("score stage requires --targets")
        if not args.path_config_receipt:
            parser.error("score stage requires --path-config-receipt")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "target":
        stage_target(args)
    else:
        stage_score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
