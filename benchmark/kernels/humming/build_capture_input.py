#!/usr/bin/env python3
"""Build a reproducible natural-token prompt for Humming route capture.

The output is an exact-length input-id stream assembled from deterministically
shuffled ShareGPT conversations using the target model's chat template.  It
also records enough source and selection metadata to reproduce and audit the
capture.  Only the final conversation may be truncated to hit the requested
token count exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Callable


FORMAT_VERSION = 1
ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize_conversation(record: dict) -> list[dict[str, str]]:
    raw_turns = record.get("conversations", record.get("conversation", []))
    if not isinstance(raw_turns, list):
        return []
    messages: list[dict[str, str]] = []
    for turn in raw_turns:
        if not isinstance(turn, dict):
            continue
        role = ROLE_MAP.get(str(turn.get("from", turn.get("role", ""))).lower())
        content = turn.get("value", turn.get("content"))
        if role is None or not isinstance(content, str) or not content.strip():
            continue
        messages.append({"role": role, "content": content})
    return messages


def build_token_stream(
    records: list[dict],
    encode_conversation: Callable[[list[dict[str, str]]], list[int]],
    target_tokens: int,
    seed: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    if target_tokens <= 0:
        raise ValueError("target token count must be positive")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)

    input_ids: list[int] = []
    selections: list[dict[str, Any]] = []
    for index in indices:
        messages = normalize_conversation(records[index])
        if len(messages) < 2:
            continue
        encoded = encode_conversation(messages)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if not isinstance(encoded, list) or not encoded:
            raise ValueError(f"chat template returned no input ids for record {index}")
        if not all(
            isinstance(token, int) and not isinstance(token, bool) for token in encoded
        ):
            raise ValueError(
                f"chat template returned non-integer ids for record {index}"
            )

        remaining = target_tokens - len(input_ids)
        used = encoded[:remaining]
        input_ids.extend(used)
        selections.append(
            {
                "record_index": index,
                "conversation_sha256": canonical_json_sha256(messages),
                "conversation_tokens": len(encoded),
                "used_tokens": len(used),
                "truncated": len(used) != len(encoded),
            }
        )
        if len(input_ids) == target_tokens:
            break

    if len(input_ids) != target_tokens:
        raise ValueError(
            f"dataset yielded only {len(input_ids)} tokens, requested {target_tokens}"
        )
    return input_ids, selections


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--encoding-spec", choices=("dsv4", "hf"), required=True)
    parser.add_argument("--prompt-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    # Keep transformers optional for CPU-only unit tests of the pure helpers.
    from transformers import AutoTokenizer

    records = json.loads(args.dataset.read_text())
    if not isinstance(records, list):
        raise ValueError("ShareGPT dataset root must be a list")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    if args.encoding_spec == "dsv4":
        from sglang.srt.entrypoints.openai.chat_encoding import encode_simple_chat

        def encode_conversation(messages):
            return encode_simple_chat(
                tokenizer=tokenizer,
                spec="dsv4",
                messages=messages,
                thinking_mode="chat",
            )

    else:
        if tokenizer.chat_template is None:
            raise ValueError("--encoding-spec hf requires tokenizer.chat_template")

        def encode_conversation(messages):
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=False,
            )

    input_ids, selections = build_token_stream(
        records, encode_conversation, args.prompt_tokens, args.seed
    )
    output = {
        "format_version": FORMAT_VERSION,
        "state": "BUILT",
        "input_ids": input_ids,
        "metadata": {
            "dataset_path": str(args.dataset),
            "dataset_size_bytes": args.dataset.stat().st_size,
            "dataset_sha256": sha256_file(args.dataset),
            "tokenizer_path": args.tokenizer,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_vocab_size": len(tokenizer),
            "encoding_spec": args.encoding_spec,
            "prompt_tokens": len(input_ids),
            "seed": args.seed,
            "input_ids_sha256": canonical_json_sha256(input_ids),
            "selected_conversations": selections,
        },
    }
    atomic_write_json(args.out, output)
    print(
        json.dumps(
            {
                "state": output["state"],
                "prompt_tokens": len(input_ids),
                "selected_conversations": len(selections),
                "input_ids_sha256": output["metadata"]["input_ids_sha256"],
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
