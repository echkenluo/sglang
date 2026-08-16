"""Four-rank microbenchmark for MoK native route control-plane overhead."""

from __future__ import annotations

import os
import statistics
import time

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.moe_runner.mok_fp8_native import (
    _consensus_supported,
    _required_route_capacity_factor,
)


def _measure(name: str, fn, *, warmup: int = 10, repeat: int = 50) -> None:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1_000)

    result = torch.tensor(
        [statistics.median(samples), statistics.mean(samples)],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(
            f"MOK_CONTROL_PLANE|name={name}|"
            f"p50_us={result[0].item():.3f}|mean_us={result[1].item():.3f}",
            flush=True,
        )


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 4

    device = torch.device("cuda", local_rank)
    tokens = 512
    topk = 6
    num_global_experts = 256
    num_local_experts = 64
    expert_padding = 64
    token = torch.arange(tokens, dtype=torch.int64, device=device)
    topk_ids = torch.stack(
        tuple(
            (token * (2 * index + 1) + local_rank * 17 + index * 29)
            % num_global_experts
            for index in range(topk)
        ),
        dim=1,
    ).contiguous()

    def consensus() -> None:
        assert _consensus_supported(True, device, dist.group.WORLD)

    def capacity() -> None:
        factor = _required_route_capacity_factor(
            topk_ids,
            num_global_experts=num_global_experts,
            num_local_experts=num_local_experts,
            ep_size=dist.get_world_size(),
            expert_padding=expert_padding,
            group=dist.group.WORLD,
        )
        assert factor == 2

    def combined() -> None:
        consensus()
        capacity()

    _measure("support_consensus", consensus)
    _measure("capacity_summary", capacity)
    _measure("combined", combined)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
