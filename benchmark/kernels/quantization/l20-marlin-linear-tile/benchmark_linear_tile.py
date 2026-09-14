# SPDX-License-Identifier: Apache-2.0
"""Bounded SM89 FP8 Marlin config screen, with the unchanged linear numeric gate.

Only M6/N512/K4096 is timed by default. --include-m5-control adds one
non-target timing control. Graph timing excludes preparation/reference work;
it establishes no checkpoint-quality or service-performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

sys.dont_write_bytecode = True
linear = importlib.import_module("sglang.srt.layers.quantization.marlin_utils_fp8")
kernel = importlib.import_module("sglang.kernels.ops.quantization.gptq_marlin")
FLAG = "SGLANG_DSV4_SM89_MARLIN_LINEAR_SMALL_TILE"
SEED = 20260914
CALLS = 16
SAMPLES = 30
ABA_BLOCKS = 3
DRIFT_LIMIT = 0.05
RELATIVE_L2_LIMIT = 0.002
ATOL = 0.02
RTOL = 0.02
SHAPES = (
    (5, 512, 4096),
    (6, 512, 4096),
    (20, 512, 4096),
    (96, 512, 4096),
    (6, 4096, 12288),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare(n, k):
    """Same random FP8/block-scale/packing contract as the existing manual test."""
    layer = torch.nn.Module()
    layer.output_size_per_partition, layer.input_size_per_partition = n, k
    layer.orig_dtype = torch.bfloat16
    layer.weight_block_size = [128, 128]
    weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    scales = (
        (torch.rand(n // 128, k // 128, device="cuda") * 0.02 + 0.01).bfloat16().float()
    )
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
    unrounded = weight.float() * scales.repeat_interleave(128, 0).repeat_interleave(
        128, 1
    )
    dense = unrounded.bfloat16().float()
    linear.prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    identity = {
        "N": n,
        "K": k,
        "weight_block_size": [128, 128],
        "packed_weight_sha256": hashlib.sha256(
            layer.weight.cpu().numpy().tobytes()
        ).hexdigest(),
        "packed_weight_shape": list(layer.weight.shape),
        "scale_shape": list(layer.weight_scale.shape),
    }
    return layer, dense, unrounded, identity


def call(x, layer, n, k):
    return linear.apply_fp8_marlin_linear(
        x,
        layer.weight,
        layer.weight_scale,
        layer.workspace,
        n,
        k,
        None,
        use_fp32_reduce=True,
    )


def toggle(enabled):
    linear.set_marlin_linear_small_tile_enabled(enabled)


def config_receipt(x, layer, n, k, enabled):
    # This is the actual C++ host selector in the same loaded JIT module.
    config = linear.get_marlin_linear_tile_config(
        x, layer.weight_scale, use_fp32_reduce=True
    )
    return {
        "candidate_enabled": enabled,
        "M": x.shape[0],
        "N": n,
        "K": k,
        "selector": "same-module C++ actual host selection",
        "selector_scope": "first M split when M>64; complete operation when M<=64",
        "activation_dtype": str(x.dtype),
        "weight_scale_dtype": str(layer.weight_scale.dtype),
        "group_size": k // layer.weight_scale.shape[0],
        "use_fp32_reduce": True,
        "use_atomic_add": linear.should_use_atomic_add_reduce(
            x.shape[0], n, k, x.device, x.dtype
        ),
        "config": dict(config),
    }


def validate_configs(receipts, shape):
    a, b = receipts["A"]["config"], receipts["B"]["config"]
    fields = ("thread_k", "thread_n", "num_threads")
    if shape == (6, 512, 4096):
        if tuple(a[key] for key in fields) != (128, 128, 256):
            raise AssertionError(f"Target reference config differs from profile: {a}")
        if tuple(b[key] for key in fields) != (64, 128, 128):
            raise AssertionError(f"Target candidate config did not activate: {b}")
    elif a != b:
        raise AssertionError(f"Non-target host selector changed: {a} versus {b}")
    if any(config.get("blocks_per_sm", 0) <= 0 for config in (a, b)):
        raise AssertionError("Invalid selected blocks-per-SM receipt")


def reference(x, dense):
    return (x.float() @ dense.T).bfloat16().float()


def error_metrics(actual, expected):
    actual = actual.float()
    expected = expected.float()
    difference = actual - expected
    denominator = expected.norm().item()
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "relative_l2": (
            difference.norm().item() / denominator
            if denominator
            else (0.0 if difference.norm().item() == 0 else None)
        ),
        "max_abs": difference.abs().max().item(),
        "different_elements": int(torch.count_nonzero(difference).item()),
        "elements": actual.numel(),
    }


def numeric_check(out, ref, row):
    metrics = error_metrics(out, ref)
    row.update(metrics)
    if (
        not metrics["finite"]
        or metrics["relative_l2"] is None
        or metrics["relative_l2"] >= RELATIVE_L2_LIMIT
    ):
        raise AssertionError(
            f"Original FP8 linear finite/relative-L2 gate failed: {metrics}"
        )
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)
    row["numeric_gate_passed"] = True


def capture(x, layer, n, k, enabled, calls=1):
    toggle(enabled)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call(x, layer, n, k)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [call(x, layer, n, k) for _ in range(calls)]
    torch.cuda.synchronize()
    return graph, outputs


def shape_check(shape, prepared, output, payload):
    m, n, k = shape
    layer, dense, unrounded, identity = prepared
    row = {
        "M": m,
        "N": n,
        "K": k,
        "status": "running",
        "weights": identity,
        "checks": [],
        "config_receipts": {},
    }
    payload["correctness"].append(row)
    save(output, payload)
    original = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    x = original.clone()
    ref = reference(x, dense)
    eager = {}
    for arm, enabled in (("A", False), ("B", True)):
        toggle(enabled)
        row["config_receipts"][arm] = config_receipt(x, layer, n, k, enabled)
        values = []
        for repeat in range(3):
            value = call(x, layer, n, k)
            check = {"arm": arm, "mode": "eager", "repeat": repeat}
            row["checks"].append(check)
            numeric_check(value, ref, check)
            check["same_input_repeat_bitwise_equal"] = repeat == 0 or torch.equal(
                value, values[0]
            )
            save(output, payload)
            if not check["same_input_repeat_bitwise_equal"]:
                raise AssertionError(f"{arm} eager same-input repeat{repeat} changed")
            values.append(value)
        eager[arm] = values[0]
        row["checks"][-1]["repeat_bitwise_equal"] = all(
            torch.equal(values[0], v) for v in values[1:]
        )
        row["checks"][-1]["unrounded_weight_precision_delta"] = error_metrics(
            values[0], x.float() @ unrounded.T
        )
        save(output, payload)
    row["A_vs_B_eager"] = error_metrics(eager["B"], eager["A"])
    validate_configs(row["config_receipts"], shape)
    if shape != (6, 512, 4096) and not torch.equal(eager["A"], eager["B"]):
        raise AssertionError("Non-target output changed when the candidate was enabled")
    graph_a, outputs_a = capture(x, layer, n, k, False)
    graph_b, outputs_b = capture(x, layer, n, k, True)
    history = [("original", original)]
    history.extend((f"changed-{i}", torch.randn_like(original)) for i in range(3))
    history.append(("restore", original))
    row["graph_history"] = []
    graph_original = {}
    for state, changed in history:
        x.copy_(changed)
        ref = reference(x, dense)
        pair = {"state": state, "repeats": []}
        row["graph_history"].append(pair)
        state_first = {}
        for repeat in range(3):
            graph_a.replay()
            graph_b.replay()
            torch.cuda.synchronize()
            comparison = {"repeat": repeat, "A": {}, "B": {}}
            pair["repeats"].append(comparison)
            numeric_check(outputs_a[0], ref, comparison["A"])
            numeric_check(outputs_b[0], ref, comparison["B"])
            comparison["A_vs_B"] = error_metrics(outputs_b[0], outputs_a[0])
            for arm, value in (("A", outputs_a[0]), ("B", outputs_b[0])):
                if repeat == 0:
                    state_first[arm] = value.clone()
                    if state == "original":
                        graph_original[arm] = value.clone()
                comparison[arm]["same_input_repeat_bitwise_equal"] = torch.equal(
                    value, state_first[arm]
                )
            save(output, payload)
            if not all(
                comparison[arm]["same_input_repeat_bitwise_equal"] for arm in ("A", "B")
            ):
                raise AssertionError(f"Graph {state} same-input repeat{repeat} changed")
            if shape != (6, 512, 4096) and not torch.equal(outputs_a[0], outputs_b[0]):
                raise AssertionError("Non-target Graph output changed")
        if state == "restore":
            row["restore_vs_original_graph"] = {
                "A_bitwise_equal": torch.equal(outputs_a[0], graph_original["A"]),
                "B_bitwise_equal": torch.equal(outputs_b[0], graph_original["B"]),
            }
            row["restore_vs_original_eager"] = {
                "A_bitwise_equal": torch.equal(outputs_a[0], eager["A"]),
                "B_bitwise_equal": torch.equal(outputs_b[0], eager["B"]),
            }
            save(output, payload)
            if not all(row["restore_vs_original_graph"].values()):
                raise AssertionError(
                    "Graph restored input differs from first original Graph output"
                )
    row["status"] = "passed"
    save(output, payload)
    print(
        f"PASS correctness M{m}/N{n}/K{k} A/B exact={row['A_vs_B_eager']['different_elements'] == 0}",
        flush=True,
    )


def event_samples(graph):
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    samples = []
    for _ in range(SAMPLES):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    torch.cuda.synchronize()
    return samples


def ratio(a, b):
    if not a or not b:
        return None
    a, b = statistics.median(a) / CALLS, statistics.median(b) / CALLS
    return {
        "A_us_per_call": a,
        "B_us_per_call": b,
        "A_over_B_speedup": a / b,
        "B_over_A_cost_ratio": b / a,
        "cost_reduction_fraction": 1 - b / a,
    }


def timing(shape, prepared, output, payload):
    m, n, k = shape
    layer, dense, _, identity = prepared
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    ref = reference(x, dense)
    row = {
        "M": m,
        "N": n,
        "K": k,
        "target": m == 6,
        "status": "running",
        "weights": identity,
        "aba_blocks": [],
        "config_receipts": {},
        "numeric_before": {},
        "numeric_after": {},
    }
    payload["timing"].append(row)
    save(output, payload)
    graphs, outputs = {}, {}
    for arm, enabled in (("A", False), ("B", True)):
        toggle(enabled)
        row["config_receipts"][arm] = config_receipt(x, layer, n, k, enabled)
        graphs[arm], outputs[arm] = capture(x, layer, n, k, enabled, CALLS)
        graphs[arm].replay()
        torch.cuda.synchronize()
        row["numeric_before"][arm] = []
        for index, value in enumerate(outputs[arm]):
            check = {"captured_call": index}
            row["numeric_before"][arm].append(check)
            numeric_check(value, ref, check)
    validate_configs(row["config_receipts"], shape)
    row["A_vs_B_before"] = error_metrics(outputs["B"][0], outputs["A"][0])
    if shape != (6, 512, 4096) and row["A_vs_B_before"]["different_elements"]:
        raise AssertionError("Non-target timing output changed before measurement")
    for block in range(ABA_BLOCKS):
        a1 = event_samples(graphs["A"])
        b = event_samples(graphs["B"])
        a2 = event_samples(graphs["A"])
        drift = abs(statistics.median(a2) / statistics.median(a1) - 1)
        item = {
            "block": block,
            "valid": drift <= DRIFT_LIMIT,
            "reference_drift_fraction": drift,
            "invalid_reason": (
                None if drift <= DRIFT_LIMIT else "reference_drift_exceeds_5_percent"
            ),
            "samples_graph_us": {"A1": a1, "B": b, "A2": a2},
            "median_us_per_call": {
                "A1": statistics.median(a1) / CALLS,
                "B": statistics.median(b) / CALLS,
                "A2": statistics.median(a2) / CALLS,
            },
            "ratio": ratio(a1 + a2, b),
        }
        row["aba_blocks"].append(item)
        save(output, payload)
        print(
            f"{'VALID' if item['valid'] else 'INVALID'} timing M{m}/N{n}/K{k} ABA{block} drift={drift:.2%} A/B={item['ratio']['A_over_B_speedup']:.4f}",
            flush=True,
        )
    for pool, blocks in (
        ("pooled_all", row["aba_blocks"]),
        ("pooled_valid", [b for b in row["aba_blocks"] if b["valid"]]),
    ):
        a, b = [], []
        for block in blocks:
            values = block["samples_graph_us"]
            a.extend(values["A1"] + values["A2"])
            b.extend(values["B"])
        row[pool] = ratio(a, b)
    for arm in ("A", "B"):
        row["numeric_after"][arm] = []
        for index, value in enumerate(outputs[arm]):
            check = {"captured_call": index}
            row["numeric_after"][arm].append(check)
            numeric_check(value, ref, check)
    row["A_vs_B_after"] = error_metrics(outputs["B"][0], outputs["A"][0])
    if shape != (6, 512, 4096) and row["A_vs_B_after"]["different_elements"]:
        raise AssertionError("Non-target timing output changed after measurement")
    row["valid_blocks"] = sum(b["valid"] for b in row["aba_blocks"])
    row["valid"] = row["valid_blocks"] == ABA_BLOCKS
    row["status"] = "completed"
    save(output, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--include-m5-control", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/SM89 required; CPU parsing is not GPU validation")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("This candidate screen requires SM89")
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(args.device)
    source_root = Path(kernel.__file__).resolve().parents[3]
    source_files = [
        Path(kernel.__file__).resolve(),
        Path(linear.__file__).resolve(),
        source_root / "kernels/jit/csrc/gemm/marlin/gptq_marlin.cuh",
        source_root / "kernels/jit/csrc/gemm/marlin/marlin_template.h",
    ]
    hashes = {str(p): sha(p) for p in source_files}
    output = args.output.resolve()
    payload = {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic FP8 Marlin linear configuration component screen",
        "identity": {
            "source_sha256": hashes,
            "script_sha256": sha(__file__),
            "gpu": properties.name,
            "compute_capability": [8, 9],
            "multiprocessor_count": properties.multi_processor_count,
            "torch_version": torch.__version__,
            "torch_file": torch.__file__,
            "triton_version": triton.__version__,
            "triton_file": triton.__file__,
            "cuda_version": torch.version.cuda,
            "device": args.device,
        },
        "protocol": {
            "seed": SEED,
            "candidate_flag": FLAG,
            "candidate_default": False,
            "A": "candidate toggle false",
            "B": "candidate toggle true",
            "calls_per_graph": CALLS,
            "aba_blocks": ABA_BLOCKS,
            "samples_per_arm": SAMPLES,
            "max_reference_drift_fraction": DRIFT_LIMIT,
            "reference": "BF16-rounded FP8*block-scales weights; FP32 matmul followed by BF16",
            "numeric_gate": {
                "relative_l2_strictly_less_than": RELATIVE_L2_LIMIT,
                "atol": ATOL,
                "rtol": RTOL,
            },
            "unrounded_weight_reference": "precision diagnostic only; not admission gate",
            "allow_tf32": False,
            "include_m5_control": args.include_m5_control,
        },
        "correctness_gate_passed": False,
        "correctness": [],
        "timing": [],
    }
    save(output, payload)
    try:
        module = kernel._jit_gptq_marlin_module(torch.bfloat16)
        payload["identity"]["jit_module_process_id"] = id(module)
        # Reuse one prepared weight set for all same-N/K correctness and timing.
        prepared = prepare(512, 4096)
        for shape in SHAPES[:-1]:
            shape_check(shape, prepared, output, payload)
        large = prepare(4096, 12288)
        shape_check(SHAPES[-1], large, output, payload)
        del large
        payload["correctness_gate_passed"] = True
        save(output, payload)
        if not args.correctness_only:
            timing((6, 512, 4096), prepared, output, payload)
            if args.include_m5_control:
                timing((5, 512, 4096), prepared, output, payload)
        if kernel._jit_gptq_marlin_module(torch.bfloat16) is not module:
            raise AssertionError("Toggle switched the JIT module")
        if any(sha(p) != value for p, value in hashes.items()):
            raise AssertionError("Source files changed during the screen")
        payload["status"] = "completed"
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        for section in ("correctness", "timing"):
            if payload[section] and payload[section][-1]["status"] == "running":
                payload[section][-1].update(status="failed", error=payload["error"])
        raise
    finally:
        toggle(False)
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(output, payload)
    print(f"Saved {output}; linear component evidence only", flush=True)


if __name__ == "__main__":
    main()
