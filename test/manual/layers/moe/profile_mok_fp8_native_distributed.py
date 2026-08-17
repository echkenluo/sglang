"""Manual four-rank profiler for the production-shape native MoK FP8 path."""

import os
import statistics
from types import SimpleNamespace

os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
os.environ["SGLANG_MASKED_GEMM_FAST_ACT"] = "0"

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.moe_runner import mok_fp8_native
from test_mok_fp8_native_distributed import _fake_layer


def _filled_fp8(shape, value, device):
    return torch.empty(shape, dtype=torch.float8_e4m3fn, device=device).fill_(value)


def _rank_max_samples(samples, device):
    local = torch.tensor(samples, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return torch.stack(gathered).max(dim=0).values.cpu().tolist()


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4

    tokens = int(os.environ.get("MOK_PROFILE_TOKENS", "4"))
    topk = int(os.environ.get("MOK_PROFILE_TOPK", "6"))
    local_experts = int(os.environ.get("MOK_PROFILE_LOCAL_EXPERTS", "64"))
    hidden = int(os.environ.get("MOK_PROFILE_HIDDEN", "7168"))
    intermediate = int(os.environ.get("MOK_PROFILE_INTERMEDIATE", "2048"))
    iterations = int(os.environ.get("MOK_PROFILE_ITERS", "30"))

    w13 = _filled_fp8(
        (local_experts, 2 * intermediate, hidden), 0.015625, device
    )
    w2 = _filled_fp8((local_experts, hidden, intermediate), 0.015625, device)
    w13_scale = torch.full(
        (local_experts, 2 * intermediate // 128, hidden // 128),
        0.03125,
        dtype=torch.float32,
        device=device,
    )
    w2_scale = torch.full(
        (local_experts, hidden // 128, intermediate // 128),
        0.03125,
        dtype=torch.float32,
        device=device,
    )
    hidden_states = torch.full(
        (tokens, hidden), 0.125, dtype=torch.bfloat16, device=device
    )
    token = torch.arange(tokens, dtype=torch.int64, device=device)[:, None]
    route = torch.arange(topk, dtype=torch.int64, device=device)[None, :]
    route_ordinal = (rank * tokens + token) * topk + route
    topk_ids = (route_ordinal * 37 + 11) % (local_experts * world_size)
    topk_weights = torch.full(
        (tokens, topk), 1.0 / topk, dtype=torch.float32, device=device
    )
    topk_output = SimpleNamespace(
        topk_ids=topk_ids.contiguous(), topk_weights=topk_weights
    )
    layer = _fake_layer(
        rank=rank,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        local_experts=local_experts,
        topk=topk,
        w13=w13,
        w13_scale=w13_scale,
        w2=w2,
        w2_scale=w2_scale,
    )
    mok_fp8_native.get_tp_group = lambda: SimpleNamespace(
        device_group=dist.group.WORLD
    )

    eager = mok_fp8_native.maybe_run_mok_fp8_native(
        layer, hidden_states, topk_output
    )
    assert eager is not None and torch.isfinite(eager).all()
    torch.cuda.synchronize(device)
    dist.barrier()
    if os.environ.get("MOK_PROFILE_EAGER_ONLY") == "1":
        if rank == 0:
            print(
                "MOK_PROFILE_EAGER_OK|"
                f"T={tokens}|topk={topk}|E_local={local_experts}|H={hidden}|"
                f"I={intermediate}",
                flush=True,
            )
        dist.destroy_process_group()
        return
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mok_fp8_native.maybe_run_mok_fp8_native(
            layer, hidden_states, topk_output
        )
    dist.barrier()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize(device)

    trace_dir = os.environ.get("MOK_PROFILE_TORCH_TRACE_DIR")
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        dist.barrier()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            graph.replay()
            torch.cuda.synchronize(device)
        profiler.export_chrome_trace(
            os.path.join(trace_dir, f"rank{rank}.trace.json")
        )
        dist.barrier()
        if rank == 0:
            print(f"MOK_PROFILE_TRACE|dir={trace_dir}", flush=True)

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iterations)
    ]
    dist.barrier()
    use_cuda_profiler_api = os.environ.get("MOK_PROFILE_CUDA_API") == "1"
    if use_cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStart()
    for start, end in events:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize(device)
    dist.barrier()
    if use_cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStop()
    samples = _rank_max_samples([s.elapsed_time(e) for s, e in events], device)
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    assert torch.isfinite(captured).all()
    if rank == 0:
        print(
            "MOK_PROFILE_GRAPH|"
            f"T={tokens}|topk={topk}|E_local={local_experts}|H={hidden}|"
            f"I={intermediate}|iters={iterations}|p50_ms={p50:.6f}|"
            f"p95_ms={p95:.6f}|samples_ms="
            + ",".join(f"{sample:.6f}" for sample in samples),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
