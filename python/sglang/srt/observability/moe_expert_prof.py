"""Per-forward MoE expert-diversity counter (env ``SGLANG_MOE_EXPERT_PROF``).

Decisive observation for the "16-token verify MoE x5.1" question: is the
amplification expert-load bandwidth bound (more distinct experts -> more
weight bytes streamed -> EVICT-style tree pruning has meat) or PCIe-allreduce
communication bound (no meat)? This counts, per MoE-layer routing, the number
of DISTINCT routed experts activated across all tokens in that forward.

Discriminator = ``n_tokens`` (topk_ids.shape[0]): under the NGRAM lane
(max_running_requests 4, draft 16, no dp-attention) decode steps route
n_tokens <= 4 (1 tok/req) and verify steps route n_tokens >= 16
(draft_token_num x bs) — non-overlapping, so bucketing by n_tokens cleanly
separates decode vs verify without any forward-mode plumbing. Prefill lands in
large-n_tokens buckets.

Hooked at the single must-pass point ``select_experts`` -> ``on_select_experts``
where ``recorder_topk_ids`` (the finalized [n_tokens, top_k] routed ids, all
256 experts visible — router is replicated, not EP-sharded) is in hand.

Cost: default OFF, every call site guarded — zero overhead. When ON, one
gated D2H per MoE-layer invocation (``seen.sum().item()`` over a 256-bool
mask); n_tokens / topk_sum are tensor metadata (no sync). This perturbs
timing, so run it SEPARATELY from the kernel-timing torch-profiler batch —
here we measure expert COUNT, not latency. rank0 only (routing is identical
across TP ranks).

NOTE: counts ROUTED experts only. Qwen3.5 also has 1 shared expert that is
always active for every token (+1, not in topk_ids) — matches the research
convention (MoE-Spec: 127-token tree -> 54/64 routed experts).
"""

from __future__ import annotations

import logging
import os
from statistics import median
from typing import Optional

logger = logging.getLogger(__name__)

_ENABLED: Optional[bool] = None
_INTERVAL: int = 512
_PROF: Optional["MoEExpertProf"] = None
_RANK_OK: Optional[bool] = None


def moe_expert_prof_enabled() -> bool:
    global _ENABLED, _INTERVAL
    if _ENABLED is None:
        raw = os.environ.get("SGLANG_MOE_EXPERT_PROF", "")
        if not raw or raw.lower() in ("0", "false"):
            _ENABLED = False
        else:
            _ENABLED = True
            try:
                n = int(raw)
                if n > 1:
                    _INTERVAL = n
            except ValueError:
                pass
    return _ENABLED


def _is_rank0() -> bool:
    global _RANK_OK
    if _RANK_OK is None:
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_rank

            _RANK_OK = get_tensor_model_parallel_rank() == 0
        except Exception:
            _RANK_OK = True  # single-process / pre-init: count anyway
    return _RANK_OK


class MoEExpertProf:
    """Aggregates distinct-routed-expert counts, bucketed by n_tokens, and
    flushes a per-bucket line every ``interval`` layer-invocations."""

    def __init__(self, interval: int):
        self.interval = interval
        self.buckets: dict[int, list[int]] = {}  # n_tokens -> [unique_count...]
        self.topk_of: dict[int, int] = {}  # n_tokens -> top_k (constant)
        self.num_experts = 0
        self.count = 0

    def record(
        self, n_tokens: int, unique: int, topk_sum: int, num_experts: int
    ) -> None:
        self.buckets.setdefault(n_tokens, []).append(unique)
        if n_tokens > 0:
            self.topk_of[n_tokens] = topk_sum // n_tokens
        self.num_experts = num_experts
        self.count += 1
        if self.count >= self.interval:
            self._flush()

    def _flush(self) -> None:
        for n_tokens in sorted(self.buckets):
            vals = self.buckets[n_tokens]
            if not vals:
                continue
            top_k = self.topk_of.get(n_tokens, 0)
            p50 = int(median(vals))
            logger.info(
                "[moe-expert-prof] n_tokens=%d samples=%d unique_experts "
                "p50=%d max=%d min=%d (of %d) top_k=%d topk_sum=%d "
                "reuse=%.2f frac=%.2f",
                n_tokens,
                len(vals),
                p50,
                max(vals),
                min(vals),
                self.num_experts,
                top_k,
                n_tokens * top_k,
                (n_tokens * top_k / p50) if p50 else 0.0,
                (p50 / self.num_experts) if self.num_experts else 0.0,
            )
        self.buckets.clear()
        self.count = 0


def _get_prof() -> Optional[MoEExpertProf]:
    global _PROF
    if not _is_rank0():
        return None
    if _PROF is None:
        _PROF = MoEExpertProf(_INTERVAL)
    return _PROF


def record_moe_experts(topk_ids, router_logits) -> None:
    """Count distinct routed experts in this MoE-layer forward. Gated + rank0;
    any failure degrades to a warning, never perturbs routing."""
    prof = _get_prof()
    if prof is None:
        return
    try:
        import torch

        if topk_ids is None or topk_ids.numel() == 0:
            return
        num_experts = int(router_logits.shape[-1])
        n_tokens = int(topk_ids.shape[0])
        topk_sum = int(topk_ids.numel())  # metadata, no sync
        flat = topk_ids.reshape(-1)
        seen = torch.zeros(num_experts, dtype=torch.bool, device=flat.device)
        # Guard OOB expert ids (padding sentinels) so scatter can't wrap.
        valid = (flat >= 0) & (flat < num_experts)
        seen[flat[valid]] = True
        unique = int(seen.sum().item())  # the single gated D2H
        prof.record(n_tokens, unique, topk_sum, num_experts)
    except Exception as e:
        logger.warning("[moe-expert-prof] record failed: %r", e)
