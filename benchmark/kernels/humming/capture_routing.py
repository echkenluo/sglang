#!/usr/bin/env python3
"""Capture serving-faithful MoE routing shapes for Humming tuning.

The server must be started with ``--enable-return-routed-experts``.  SGLang
returns the captured int32 tensor as base64 with logical shape
``[tokens, num_hidden_layers, top_k]``.  This tool validates that contract,
keeps the raw per-layer routing instead of pooling it into one histogram, and
emits one tuning point per real chunk/layer.

Formal captures require an explicit input-id JSON file.  Synthetic ids are
available only behind ``--allow-synthetic-input`` for plumbing checks.
"""

from __future__ import annotations

import argparse
import array
import base64
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_sha256(data: Any) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def load_model_shape(path: Path) -> dict[str, int]:
    raw = path.read_bytes()
    config = json.loads(raw)
    text_config = config.get("text_config", config)
    required = ("num_hidden_layers", "num_experts_per_tok", "n_routed_experts")
    missing = [name for name in required if name not in text_config]
    if missing:
        raise ValueError(f"model config is missing: {', '.join(missing)}")

    # DeepSeek-V4-Flash checkpoints may serialize the optional field as null.
    # SGLang's V4 config defaults that value to zero, so mirror the production
    # interpretation instead of calling int(None).
    first_moe_layer_raw = text_config.get("first_k_dense_replace")
    first_moe_layer = (
        0 if first_moe_layer_raw is None else int(first_moe_layer_raw)
    )
    num_layers = int(text_config["num_hidden_layers"])
    if not 0 <= first_moe_layer < num_layers:
        raise ValueError(
            f"invalid first MoE layer {first_moe_layer} for {num_layers} layers"
        )
    return {
        "num_layers": num_layers,
        "top_k": int(text_config["num_experts_per_tok"]),
        "num_experts": int(text_config["n_routed_experts"]),
        "first_moe_layer": first_moe_layer,
        "model_config_sha256": _sha256_bytes(raw),
    }


def load_input_ids(
    path: Path | None, prompt_tokens: int | None, allow_synthetic: bool
) -> tuple[list[int], str]:
    if path is not None:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("input_ids")
        if not isinstance(payload, list) or not payload:
            raise ValueError("input-id JSON must be a non-empty list or {input_ids: [...]}")
        if not all(isinstance(token, int) and not isinstance(token, bool) for token in payload):
            raise ValueError("every input id must be an integer")
        if prompt_tokens is not None and len(payload) != prompt_tokens:
            raise ValueError(
                f"input-id count {len(payload)} does not match --prompt-tokens "
                f"{prompt_tokens}"
            )
        return payload, "explicit"

    if not allow_synthetic:
        raise ValueError(
            "formal capture requires --input-ids-json; use "
            "--allow-synthetic-input only for a plumbing check"
        )
    if prompt_tokens is None or prompt_tokens <= 0:
        raise ValueError("synthetic input requires positive --prompt-tokens")
    return [1000 + index % 4000 for index in range(prompt_tokens)], "synthetic"


def decode_routed_experts(
    encoded: str,
    *,
    num_tokens: int,
    num_layers: int,
    top_k: int,
) -> tuple[array.array, str]:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("response has no base64 routed_experts payload")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError("routed_experts is not valid base64") from exc

    expected_values = num_tokens * num_layers * top_k
    expected_bytes = expected_values * 4
    if len(raw) != expected_bytes:
        raise ValueError(
            f"routed_experts byte length {len(raw)} != expected {expected_bytes} "
            f"for shape ({num_tokens}, {num_layers}, {top_k})"
        )
    values = array.array("i")
    if values.itemsize != 4:
        raise RuntimeError(f"native signed-int item size is {values.itemsize}, expected 4")
    values.frombytes(raw)
    if values.itemsize == 4 and os.sys.byteorder != "little":
        values.byteswap()
    return values, _sha256_bytes(raw)


def chunk_ranges(num_tokens: int, chunk_size: int) -> list[tuple[int, int]]:
    if num_tokens <= 0 or chunk_size <= 0:
        raise ValueError("num_tokens and chunk_size must be positive")
    return [
        (start, min(start + chunk_size, num_tokens))
        for start in range(0, num_tokens, chunk_size)
    ]


def summarize_capture(
    routed: array.array,
    *,
    num_tokens: int,
    num_layers: int,
    top_k: int,
    first_moe_layer: int,
    num_experts: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    expected_values = num_tokens * num_layers * top_k
    if len(routed) != expected_values:
        raise ValueError(
            f"routed value count {len(routed)} != expected {expected_values}"
        )
    if not 0 <= first_moe_layer < num_layers:
        raise ValueError("first_moe_layer is outside the routed tensor")

    points: list[dict[str, Any]] = []
    for chunk_index, (start, end) in enumerate(chunk_ranges(num_tokens, chunk_size)):
        token_count = end - start
        valid_shape_m = token_count * top_k
        for layer_id in range(first_moe_layer, num_layers):
            counts = [0] * num_experts
            for token_index in range(start, end):
                base = (token_index * num_layers + layer_id) * top_k
                for expert_id in routed[base : base + top_k]:
                    if not 0 <= expert_id < num_experts:
                        raise ValueError(
                            f"MoE routed expert {expert_id} is outside "
                            f"[0, {num_experts})"
                        )
                    counts[expert_id] += 1
            nonzero = [count for count in counts if count > 0]
            points.append(
                {
                    "chunk_index": chunk_index,
                    "token_start": start,
                    "token_end": end,
                    "token_count": token_count,
                    "layer_id": layer_id,
                    "top_k": top_k,
                    "valid_shape_m": valid_shape_m,
                    "active_experts": len(nonzero),
                    "min_active_expert_rows": min(nonzero),
                    "max_expert_rows": max(nonzero),
                    "expert_counts": counts,
                }
            )
    return points


def request_capture(base_url: str, input_ids: list[int], timeout: int) -> dict:
    body = json.dumps(
        {
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_routed_experts": True,
            "routed_experts_start_len": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    result["_client_elapsed_seconds"] = time.time() - started
    return result


def atomic_write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--input-ids-json", type=Path)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--allow-synthetic-input", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_shape = load_model_shape(args.model_config)
    input_ids, input_source = load_input_ids(
        args.input_ids_json, args.prompt_tokens, args.allow_synthetic_input
    )
    response = request_capture(args.base_url, input_ids, args.timeout)
    encoded = (response.get("meta_info") or {}).get("routed_experts")
    routed, payload_sha256 = decode_routed_experts(
        encoded,
        num_tokens=len(input_ids),
        num_layers=model_shape["num_layers"],
        top_k=model_shape["top_k"],
    )
    points = summarize_capture(
        routed,
        num_tokens=len(input_ids),
        num_layers=model_shape["num_layers"],
        top_k=model_shape["top_k"],
        first_moe_layer=model_shape["first_moe_layer"],
        num_experts=model_shape["num_experts"],
        chunk_size=args.chunk_size,
    )

    args.out_dir.mkdir(parents=True, exist_ok=False)
    raw_path = args.out_dir / "routed-experts.i32"
    raw_path.write_bytes(routed.tobytes())
    manifest = {
        "format_version": FORMAT_VERSION,
        "state": "CAPTURED",
        "input_source": input_source,
        "input_ids_sha256": _canonical_json_sha256(input_ids),
        "prompt_tokens": len(input_ids),
        "chunk_size": args.chunk_size,
        "num_chunks": len(chunk_ranges(len(input_ids), args.chunk_size)),
        "routed_payload_sha256": payload_sha256,
        "raw_file": raw_path.name,
        "raw_dtype": "little-endian-int32",
        "raw_shape": [
            len(input_ids),
            model_shape["num_layers"],
            model_shape["top_k"],
        ],
        "client_elapsed_seconds": response["_client_elapsed_seconds"],
        "model": model_shape,
        "points": points,
    }
    atomic_write_json(args.out_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "state": manifest["state"],
                "prompt_tokens": manifest["prompt_tokens"],
                "num_chunks": manifest["num_chunks"],
                "num_points": len(points),
                "raw_shape": manifest["raw_shape"],
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
