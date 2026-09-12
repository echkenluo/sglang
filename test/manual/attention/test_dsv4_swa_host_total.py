"""GPU equivalence and scalar-read check for CPU-sized sparse SWA gathers."""
import argparse
import json
import statistics
import time
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import build_swa_token_ids


def check(seq, extend, window):
    device = 'cuda'
    n = len(seq)
    width = max(max(seq), 1)
    # Scramble physical cache IDs; a positional identity mapping would miss
    # using the wrong request row or skipping the full-to-SWA translation.
    generator = torch.Generator().manual_seed(913)
    req_to_token = torch.randperm(n * width, generator=generator).reshape(n, width).int()
    mapping = torch.randperm(n * width, generator=generator).long()
    rows = torch.arange(n - 1, -1, -1, dtype=torch.int32)
    expected = []
    for r, s, e in zip(rows.tolist(), seq, extend):
        start = max(0, s - e - window + 1)
        expected.extend(mapping[req_to_token[r, start:s].long()].tolist())
    kwargs = dict(seq_lens=torch.tensor(seq, dtype=torch.int32, device=device),
                  extend_seq_lens=torch.tensor(extend, dtype=torch.int32, device=device),
                  req_pool_indices=rows.to(device), req_to_token=req_to_token.to(device),
                  full_to_swa=mapping.to(device), swa_window=window)
    baseline = build_swa_token_ids(**kwargs)
    total = len(expected)
    candidate = build_swa_token_ids(**kwargs, total_swa=total)
    torch.cuda.synchronize()
    assert baseline[0].cpu().tolist() == expected
    assert all(torch.equal(a, b) for a, b in zip(baseline, candidate))
    # After JIT warmup the fast path must not ask PyTorch for any scalar value.
    with patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar read')):
        checked = build_swa_token_ids(**kwargs, total_swa=total)
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(baseline, checked))
    times = {'device_total': [], 'host_total': []}
    for repeat in range(8):
        for name in (('device_total', 'host_total') if repeat % 2 == 0 else ('host_total', 'device_total')):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                build_swa_token_ids(**kwargs, **({'total_swa': total} if name == 'host_total' else {}))
            torch.cuda.synchronize()
            times[name].append((time.perf_counter() - start) / 20)
    return {'seq': seq, 'extend': extend, 'window': window, 'total': total,
            'equal': True, 'host_path_no_item': True,
            'wall_us_median': {k: statistics.median(v) * 1e6 for k, v in times.items()},
            'wall_seconds_samples': times}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    cases = [([0], [0], 128), ([127, 128, 129], [1, 128, 2], 128),
             ([4096] * 3, [4096] * 3, 128), ([8192, 4096, 2048], [1024, 2048, 0], 128)]
    out = {'scope': 'one-GPU gather equivalence and component timing; not model E2E',
           'device': torch.cuda.get_device_name(), 'cases': [check(*c) for c in cases]}
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
