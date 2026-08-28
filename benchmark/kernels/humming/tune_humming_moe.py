#!/usr/bin/env python3
"""Tune Humming indexed MoE GEMMs against captured production routing.

This is a kernel-screening tool.  It uses the exact SGLang/Humming metadata
construction path, Humming's legal test-config generator, real per-layer
``topk_ids``, correctness gates, pre-JIT, randomized measurement rounds and an
A1/A2 drift guard.  A later service gate must validate any selected config.
"""

from __future__ import annotations

import argparse
import array
import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

from humming.config import ComputeConfig, GemmType
from humming.kernel.humming import HummingKernel
from humming.layer import HummingMethod
from humming.schema import BaseInputSchema, BaseWeightSchema, HummingInputSchema
from humming.testing.tuning import sample_test_tuning_configs

from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size


BIG_M = 1 << 40
FORMAT_VERSION = 1


class StubMoeLayer(torch.nn.Module):
    def __init__(self, num_experts: int, param_dtype: torch.dtype):
        super().__init__()
        self.num_experts = num_experts
        self.param_dtype = param_dtype
        self.params_dtype = param_dtype
        self.with_bias = False


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def config_id(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()[:16]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def deduplicate_configs(configs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for config in configs:
        normalized = json.loads(canonical_json(config))
        key = canonical_json(normalized)
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def cap_sampled_configs(sampled: list[dict], candidate_count: int) -> list[dict]:
    """Reserve one candidate slot for the production heuristic."""
    if candidate_count <= 0:
        raise ValueError("candidate count must be positive")
    return sampled[: max(candidate_count - 1, 0)]


def choose_representative_points(points: list[dict], count: int) -> list[dict]:
    if count <= 0:
        raise ValueError("route sample count must be positive")
    if len(points) <= count:
        return sorted(points, key=lambda point: (point["chunk_index"], point["layer_id"]))
    ordered = sorted(
        points,
        key=lambda point: (
            point["max_expert_rows"],
            point["active_experts"],
            point["layer_id"],
        ),
    )
    indices = {
        round(index * (len(ordered) - 1) / (count - 1))
        for index in range(count)
    } if count > 1 else {len(ordered) // 2}
    return [ordered[index] for index in sorted(indices)]


def load_capture(manifest_path: Path, shape_m: int | None, route_samples: int):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("state") != "CAPTURED":
        raise ValueError("capture manifest is not in CAPTURED state")
    if manifest.get("raw_dtype") != "little-endian-int32":
        raise ValueError(f"unsupported raw dtype: {manifest.get('raw_dtype')}")
    raw_shape = manifest["raw_shape"]
    if len(raw_shape) != 3:
        raise ValueError(f"invalid raw shape: {raw_shape}")

    available_shape_ms = sorted({int(point["valid_shape_m"]) for point in manifest["points"]})
    if shape_m is None:
        if len(available_shape_ms) != 1:
            raise ValueError(
                f"capture has multiple shape_m values {available_shape_ms}; pass --shape-m"
            )
        shape_m = available_shape_ms[0]
    if shape_m not in available_shape_ms:
        raise ValueError(f"shape_m {shape_m} is not in capture {available_shape_ms}")

    points = [point for point in manifest["points"] if point["valid_shape_m"] == shape_m]
    selected = choose_representative_points(points, route_samples)
    raw_path = manifest_path.parent / manifest["raw_file"]
    raw = raw_path.read_bytes()
    expected_bytes = math.prod(raw_shape) * 4
    if len(raw) != expected_bytes:
        raise ValueError(f"raw routing bytes {len(raw)} != expected {expected_bytes}")
    if hashlib.sha256(raw).hexdigest() != manifest["routed_payload_sha256"]:
        raise ValueError("raw routing SHA256 does not match capture manifest")
    values = array.array("i")
    values.frombytes(raw)
    if values.itemsize != 4:
        raise RuntimeError("native signed-int is not 32-bit")
    if sys.byteorder != "little":
        values.byteswap()
    return manifest, values, selected, shape_m


def extract_topk_ids(
    routed: array.array, raw_shape: list[int], point: dict
) -> torch.Tensor:
    _, num_layers, top_k = raw_shape
    values: list[int] = []
    layer_id = int(point["layer_id"])
    for token_index in range(int(point["token_start"]), int(point["token_end"])):
        base = (token_index * num_layers + layer_id) * top_k
        values.extend(routed[base : base + top_k])
    return torch.tensor(values, dtype=torch.int32).reshape(-1, top_k)


def _initialize_tensor(name: str, tensor: torch.Tensor) -> None:
    if "scale" in name:
        tensor.fill_(1)
    elif "zero" in name:
        tensor.zero_()
    elif tensor.ndim >= 1:
        # Expert-dependent values make expert-indexing mistakes observable while
        # avoiding a full-size temporary random FP32 tensor.
        for expert in range(tensor.shape[0]):
            value = ((expert % 7) + 1) / 16 if tensor.dtype.is_floating_point else expert % 7
            tensor[expert].fill_(value)
    else:
        tensor.fill_(1)


def build_layer(model_config: dict, tp_size: int, device: str):
    text_config = model_config.get("text_config", model_config)
    hidden = int(text_config["hidden_size"])
    intermediate = int(text_config["moe_intermediate_size"])
    num_experts = int(text_config["n_routed_experts"])
    if intermediate % tp_size:
        raise ValueError(f"moe_intermediate_size {intermediate} is not divisible by TP {tp_size}")
    quant_config = dict(text_config["quantization_config"])
    param_dtype = torch.bfloat16
    weight_schema = BaseWeightSchema.from_config(quant_config)
    input_schema = BaseInputSchema.from_config(quant_config) or HummingInputSchema()
    layer = StubMoeLayer(num_experts, param_dtype).to(device)
    intermediate_per_rank = intermediate // tp_size
    layer.hidden_size = hidden
    layer.intermediate_size_per_partition = intermediate_per_rank
    layer.sublayer_configs = {
        "w13": {"shape_n": intermediate_per_rank * 2, "shape_k": hidden},
        "w2": {"shape_n": hidden, "shape_k": intermediate_per_rank},
    }

    for sublayer, config in layer.sublayer_configs.items():
        tensor_attrs = weight_schema.get_padded_tensors_attrs(
            shape_n=config["shape_n"],
            shape_k=config["shape_k"],
            num_experts=num_experts,
            param_dtype=param_dtype,
            has_bias=False,
        )
        for name, attrs in tensor_attrs.items():
            tensor = torch.empty(attrs["shape"], dtype=attrs["dtype"], device=device)
            _initialize_tensor(name, tensor)
            setattr(
                layer,
                f"{sublayer}_{name}",
                torch.nn.Parameter(tensor, requires_grad=False),
            )

    layer.register_buffer("locks", torch.zeros(1024, dtype=torch.int32, device=device))
    for sublayer, config in layer.sublayer_configs.items():
        tensors = {
            key.removeprefix(sublayer + "_"): value
            for key, value in layer.state_dict().items()
            if key.startswith(sublayer + "_")
        }
        shape_n_stacks = (
            [config["shape_n"] // 2, config["shape_n"] // 2]
            if sublayer == "w13"
            else [config["shape_n"]]
        )
        shape_k_stacks = [config["shape_k"]]
        converted_weight_schema, tensors = weight_schema.convert_humming(
            tensors=tensors,
            shape_n_stacks=shape_n_stacks,
            shape_k_stacks=shape_k_stacks,
            param_dtype=param_dtype,
            num_experts=num_experts,
        )
        converted_input_schema, _ = input_schema.convert_humming(
            tensors=tensors,
            shape_n_stacks=shape_n_stacks,
            shape_k_stacks=shape_k_stacks,
            param_dtype=param_dtype,
            num_experts=num_experts,
        )
        for name in [
            name for name, _ in layer.named_parameters() if name.startswith(sublayer + "_")
        ]:
            delattr(layer, name)
        for name, tensor in tensors.items():
            setattr(
                layer,
                f"{sublayer}_{name}",
                torch.nn.Parameter(tensor, requires_grad=False),
            )
        HummingMethod.prepare_layer_meta(
            layer=layer,
            shape_n=config["shape_n"],
            shape_k=config["shape_k"],
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            input_schema=converted_input_schema,
            weight_schema=converted_weight_schema,
            has_bias=False,
            num_experts=num_experts,
            torch_dtype=param_dtype,
            sublayer_name=sublayer,
        )
        HummingMethod.transform_humming_layer(layer, sublayer_name=sublayer)
    return layer


def exact_heuristic_config(layer, sublayer: str, shape_m: int) -> dict:
    ladder = HummingMethod.get_default_tuning_configs(
        layer=layer,
        use_f16_accum=False,
        gemm_type=GemmType.INDEXED,
        sublayer_name=sublayer,
    )
    for min_m, max_m, config in ladder:
        if shape_m > min_m and shape_m <= max_m:
            return json.loads(canonical_json(config))
    raise ValueError(f"heuristic has no {sublayer} config for shape_m={shape_m}")


def build_candidates(
    layer, sublayer: str, shape_m: int, candidate_count: int
) -> tuple[dict, list[dict]]:
    heuristic = exact_heuristic_config(layer, sublayer, shape_m)
    compute_config = ComputeConfig(use_f16_accum=False, gemm_type=GemmType.INDEXED)
    sampled = sample_test_tuning_configs(
        layer.humming_metas[sublayer], compute_config, sample_size=candidate_count
    )
    candidates = deduplicate_configs(
        [heuristic, *cap_sampled_configs(sampled, candidate_count)]
    )
    return heuristic, candidates


def precompile_candidate(layer, sublayer: str, config: dict) -> None:
    compute = ComputeConfig(use_f16_accum=False, gemm_type=GemmType.INDEXED)
    HummingKernel.prepare_kernels(
        layer.humming_metas[sublayer].to_str(),
        compute.to_str(),
        [[0, BIG_M, config]],
    )


def make_inputs(layer, sublayer: str, token_count: int, shape_m: int, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    if sublayer == "w13":
        rows = token_count
        width = layer.sublayer_configs["w13"]["shape_k"]
    else:
        rows = shape_m
        width = layer.sublayer_configs["w2"]["shape_k"]
    original = torch.randn(
        rows,
        width,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    return HummingMethod.may_quant_input(
        layer=layer, inputs=original, sublayer_name=sublayer
    )


def alignment_for_route(topk_ids: torch.Tensor, block_m: int, num_experts: int):
    return moe_align_block_size(
        topk_ids=topk_ids,
        block_size=block_m,
        num_experts=num_experts,
        ignore_invalid_expert=True,
    )


def run_forward(
    layer,
    sublayer: str,
    inputs: torch.Tensor,
    input_scale: torch.Tensor | None,
    output: torch.Tensor,
    alignment,
    top_k: int,
    shape_m: int,
    config: dict,
) -> None:
    sorted_ids, expert_ids, num_tokens_padded = alignment
    HummingMethod.forward_layer(
        layer=layer,
        inputs=inputs,
        input_scale=input_scale,
        outputs=output,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        top_k=top_k,
        valid_shape_m=shape_m,
        compute_config=json.dumps({"use_f16_accum": False, "gemm_type": "indexed"}),
        tuning_config=json.dumps([[0, BIG_M, config]]),
        sublayer_name=sublayer,
    )


def measure_forward(forward, inner_iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(inner_iters):
        forward()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / inner_iters


def correctness_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    absolute = (actual_float - expected_float).abs()
    relative = absolute / expected_float.abs().clamp_min(1e-5)
    return {
        "max_abs": absolute.max().item(),
        "max_rel": relative.max().item(),
    }


def tune_sublayer(
    *,
    layer,
    sublayer: str,
    shape_m: int,
    route_tensors: list[torch.Tensor],
    route_points: list[dict],
    candidate_count: int,
    rounds: int,
    inner_iters: int,
    warmup: int,
    rtol: float,
    atol: float,
    seed: int,
    w13_alignment_config: dict | None = None,
) -> dict:
    heuristic, candidates = build_candidates(layer, sublayer, shape_m, candidate_count)
    result: dict[str, Any] = {
        "sublayer": sublayer,
        "shape_m": shape_m,
        "heuristic_config": heuristic,
        "heuristic_id": config_id(heuristic),
        "candidate_count": len(candidates),
        "route_points": route_points,
        "rejected": [],
    }

    compiled = []
    for config in candidates:
        try:
            precompile_candidate(layer, sublayer, config)
            compiled.append(config)
        except Exception as exc:
            result["rejected"].append(
                {"config_id": config_id(config), "phase": "compile", "error": repr(exc)}
            )
    if config_id(heuristic) not in {config_id(config) for config in compiled}:
        result.update(state="INVALID_HEURISTIC_COMPILE")
        return result

    contexts = []
    alignment_w13 = w13_alignment_config or exact_heuristic_config(
        layer, "w13", shape_m
    )
    if sublayer == "w2":
        result["alignment_w13_config"] = alignment_w13
        result["alignment_w13_config_id"] = config_id(alignment_w13)
    for route_index, (topk_ids_cpu, point) in enumerate(
        zip(route_tensors, route_points, strict=True)
    ):
        topk_ids = topk_ids_cpu.cuda()
        token_count = int(point["token_count"])
        inputs, input_scale = make_inputs(
            layer, sublayer, token_count, shape_m, seed + route_index
        )
        output_rows = shape_m
        output_width = layer.sublayer_configs[sublayer]["shape_n"]
        output = torch.empty(
            output_rows, output_width, dtype=torch.bfloat16, device="cuda"
        )
        alignments = {}
        if sublayer == "w2":
            # Runtime builds indexed alignment once from the selected W13 block M
            # and reuses it for W2.  Kernel screening must reproduce that contract.
            alignments["runtime"] = alignment_for_route(
                topk_ids, alignment_w13["block_shape"][0], layer.num_experts
            )
        contexts.append(
            {
                "topk_ids": topk_ids,
                "inputs": inputs,
                "input_scale": input_scale,
                "output": output,
                "alignments": alignments,
                "top_k": topk_ids.shape[1] if sublayer == "w13" else 1,
            }
        )

    def get_forward(context, config):
        if sublayer == "w13":
            block_m = int(config["block_shape"][0])
            alignment = context["alignments"].get(block_m)
            if alignment is None:
                alignment = alignment_for_route(
                    context["topk_ids"], block_m, layer.num_experts
                )
                context["alignments"][block_m] = alignment
        else:
            alignment = context["alignments"]["runtime"]
        return lambda: run_forward(
            layer,
            sublayer,
            context["inputs"],
            context["input_scale"],
            context["output"],
            alignment,
            context["top_k"],
            shape_m,
            config,
        )

    references = []
    for context in contexts:
        forward = get_forward(context, heuristic)
        for _ in range(warmup):
            forward()
        torch.cuda.synchronize()
        context["output"].fill_(float("nan"))
        forward()
        torch.cuda.synchronize()
        references.append(context["output"].clone())

    valid = []
    for config in compiled:
        candidate_ok = True
        worst = {"max_abs": 0.0, "max_rel": 0.0}
        try:
            for context, reference in zip(contexts, references, strict=True):
                context["output"].fill_(float("nan"))
                get_forward(context, config)()
                torch.cuda.synchronize()
                if not torch.isfinite(context["output"]).all().item():
                    raise ValueError("candidate produced non-finite output")
                metrics = correctness_metrics(context["output"], reference)
                worst = {name: max(worst[name], value) for name, value in metrics.items()}
                torch.testing.assert_close(
                    context["output"], reference, rtol=rtol, atol=atol
                )
        except Exception as exc:
            candidate_ok = False
            result["rejected"].append(
                {
                    "config_id": config_id(config),
                    "phase": "correctness",
                    "error": repr(exc),
                    **worst,
                }
            )
        if candidate_ok:
            valid.append((config, worst))
    if config_id(heuristic) not in {config_id(config) for config, _ in valid}:
        result.update(state="INVALID_HEURISTIC_CORRECTNESS")
        return result

    for context in contexts:
        for _ in range(warmup):
            get_forward(context, heuristic)()
    torch.cuda.synchronize()

    def measure_config(config):
        return [
            measure_forward(get_forward(context, config), inner_iters)
            for context in contexts
        ]

    a1 = measure_config(heuristic)
    timings: dict[str, list[float]] = {config_id(config): [] for config, _ in valid}
    rng = random.Random(seed)
    for _ in range(rounds):
        order = list(valid)
        rng.shuffle(order)
        for config, _ in order:
            timings[config_id(config)].extend(measure_config(config))
    a2 = measure_config(heuristic)

    a1_median = statistics.median(a1)
    a2_median = statistics.median(a2)
    drift = abs(a2_median - a1_median) / a1_median
    table = []
    correctness_by_id = {config_id(config): metrics for config, metrics in valid}
    for config, _ in valid:
        values = timings[config_id(config)]
        table.append(
            {
                "config_id": config_id(config),
                "config": config,
                "median_us": statistics.median(values),
                "p95_us": percentile(values, 0.95),
                "samples": len(values),
                "correctness": correctness_by_id[config_id(config)],
            }
        )
    table.sort(key=lambda row: (row["median_us"], row["p95_us"]))
    heuristic_row = next(row for row in table if row["config_id"] == config_id(heuristic))
    best = table[0]
    result.update(
        {
            "state": "INVALID_DRIFT" if drift >= 0.02 else "MEASURED",
            "a1_route_us": a1,
            "a2_route_us": a2,
            "a1_median_us": a1_median,
            "a2_median_us": a2_median,
            "drift": drift,
            "valid_candidate_count": len(valid),
            "heuristic_median_us": heuristic_row["median_us"],
            "best_config_id": best["config_id"],
            "best_config": best["config"],
            "best_median_us": best["median_us"],
            "headroom": (heuristic_row["median_us"] - best["median_us"])
            / heuristic_row["median_us"],
            "table": table,
        }
    )
    return result


def atomic_write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--shape-m", type=int)
    parser.add_argument("--sublayer", choices=("w13", "w2", "both"), default="both")
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--route-samples", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--inner-iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Humming tuning requires a CUDA GPU")
    torch.cuda.set_device(0)
    capture, routed, points, shape_m = load_capture(
        args.capture_manifest, args.shape_m, args.route_samples
    )
    model_config_bytes = args.model_config.read_bytes()
    model_config = json.loads(model_config_bytes)
    layer = build_layer(model_config, args.tp_size, "cuda")
    route_tensors = [
        extract_topk_ids(routed, capture["raw_shape"], point) for point in points
    ]
    sublayers = ("w13", "w2") if args.sublayer == "both" else (args.sublayer,)
    result = {
        "format_version": FORMAT_VERSION,
        "state": "RUNNING",
        "capture_manifest": str(args.capture_manifest),
        "capture_sha256": hashlib.sha256(args.capture_manifest.read_bytes()).hexdigest(),
        "model_config_sha256": hashlib.sha256(model_config_bytes).hexdigest(),
        "tp_size": args.tp_size,
        "shape_m": shape_m,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "parameters": {
            name: getattr(args, name)
            for name in (
                "candidate_count",
                "route_samples",
                "rounds",
                "inner_iters",
                "warmup",
                "rtol",
                "atol",
                "seed",
            )
        },
        "sublayers": {},
    }
    selected_w13_config = None
    for sublayer in sublayers:
        result["sublayers"][sublayer] = tune_sublayer(
            layer=layer,
            sublayer=sublayer,
            shape_m=shape_m,
            route_tensors=route_tensors,
            route_points=points,
            candidate_count=args.candidate_count,
            rounds=args.rounds,
            inner_iters=args.inner_iters,
            warmup=args.warmup,
            rtol=args.rtol,
            atol=args.atol,
            seed=args.seed + (0 if sublayer == "w13" else 1000),
            w13_alignment_config=selected_w13_config,
        )
        if (
            sublayer == "w13"
            and result["sublayers"][sublayer]["state"] == "MEASURED"
        ):
            selected_w13_config = result["sublayers"][sublayer]["best_config"]
        atomic_write_json(args.out, result)
    states = [value["state"] for value in result["sublayers"].values()]
    result["state"] = "MEASURED" if all(state == "MEASURED" for state in states) else "INVALID"
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "state": result["state"],
                "sublayers": {
                    name: {
                        key: value.get(key)
                        for key in ("state", "candidate_count", "valid_candidate_count", "headroom")
                    }
                    for name, value in result["sublayers"].items()
                },
                "out": str(args.out),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["state"] == "MEASURED" else 1)


if __name__ == "__main__":
    main()
