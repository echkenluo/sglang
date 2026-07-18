"""Opt-in, zero-effect token-position tracing for NGRAM verification.

The trace is carried through the existing per-token ``customized_info`` path.
It never participates in drafting, sampling, KV management, or finish logic.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

ENV_NAME = "SGLANG_NGRAM_ACCEPT_TRACE"
TRACE_KEY = "ngram_accept_trace_v1"
_INVALID_ATTR = "_ngram_accept_trace_v1_invalid"
_ENABLED = os.environ.get(ENV_NAME) == "1"


def align_ngram_accept_trace_prefix(req, *, enabled: Optional[bool] = None) -> bool:
    """Align the trace with tokens emitted outside NGRAM verification.

    The generation prefill path calls this after appending its first sampled
    token, before streaming. Existing output positions are represented by
    ``None``. A trace that already exists must be exactly aligned.
    """

    if enabled is None:
        enabled = _ENABLED
    if not enabled:
        return False
    if getattr(req, _INVALID_ATTR, None) is not None:
        return False

    try:
        output_len = len(req.output_ids)
        info = req.customized_info
        if info is None:
            info = {}
        elif not isinstance(info, dict):
            raise TypeError("req.customized_info is not a dict")

        if TRACE_KEY in info:
            prior = info[TRACE_KEY]
            if not isinstance(prior, list):
                raise TypeError(f"{TRACE_KEY} is not a list")
            if len(prior) != output_len:
                raise ValueError(
                    f"{TRACE_KEY} length {len(prior)} != output length {output_len}"
                )
            return True

        new_info = dict(info)
        new_info[TRACE_KEY] = [None] * output_len
        req.customized_info = new_info
        return True
    except Exception as exc:  # observability must never break generation
        setattr(req, _INVALID_ATTR, str(exc))
        logger.warning("Disabled NGRAM accept trace for rid=%s: %s", req.rid, exc)
        return False


def append_ngram_accept_trace(
    req,
    retained_tokens: Sequence[int],
    num_correct_drafts: int,
    *,
    enabled: Optional[bool] = None,
) -> bool:
    """Append one verified run to a request-aligned NGRAM acceptance trace.

    Values align one-for-one with ``req.output_ids`` after the caller appends
    ``retained_tokens``:

    * ``1``: a correct NGRAM draft accepted by target verification;
    * ``0``: a retained target-model bonus token;
    * ``None``: an earlier token emitted outside an NGRAM verify step.

    This function is deliberately fail-open for serving: a trace collision or
    alignment error marks the request's trace invalid and returns ``False``,
    but never changes output tokens or raises into generation.
    """

    if enabled is None:
        enabled = _ENABLED
    if not enabled:
        return False
    if getattr(req, _INVALID_ATTR, None) is not None:
        return False

    try:
        output_len = len(req.output_ids)
        info = req.customized_info
        if info is None:
            info = {}
        elif not isinstance(info, dict):
            raise TypeError("req.customized_info is not a dict")

        if TRACE_KEY in info:
            prior = info[TRACE_KEY]
            if not isinstance(prior, list):
                raise TypeError(f"{TRACE_KEY} is not a list")
            if len(prior) != output_len:
                raise ValueError(
                    f"{TRACE_KEY} length {len(prior)} != output length {output_len}"
                )
            trace = list(prior)
        else:
            trace = [None] * output_len

        retained_len = len(retained_tokens)
        correct = min(max(int(num_correct_drafts), 0), retained_len)
        trace.extend([1] * correct)
        trace.extend([0] * (retained_len - correct))

        # Copy-on-write keeps unrelated customized_info untouched if any
        # validation above fails.
        new_info = dict(info)
        new_info[TRACE_KEY] = trace
        req.customized_info = new_info
        return True
    except Exception as exc:  # observability must never break generation
        setattr(req, _INVALID_ATTR, str(exc))
        logger.warning("Disabled NGRAM accept trace for rid=%s: %s", req.rid, exc)
        return False


def trace_invalid_reason(req) -> Optional[str]:
    """Return the request-local trace failure reason, if any."""

    return getattr(req, _INVALID_ATTR, None)
