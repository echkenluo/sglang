# SPDX-License-Identifier: Apache-2.0
"""Configuration boundary for the experimental short-Graph/long-scatter split."""


def validate_short_prefill_graph_buckets(backend, capture_buckets, scatter_min_tokens):
    """Reject any capture bucket that could execute the scattered token layout.

    Prefill buckets count aggregate tokens, not requests. This only validates
    disjoint routing ranges; capture/replay and model correctness need GPU tests.
    """
    if backend != "breakable":
        raise ValueError("DSV4 short prefill Graph requires the breakable backend")
    if type(scatter_min_tokens) is not int or scatter_min_tokens <= 0:
        raise ValueError(
            "DSV4 scattered communication needs a positive token threshold"
        )
    if not isinstance(capture_buckets, (list, tuple)) or not capture_buckets:
        raise ValueError("DSV4 short prefill Graph requires explicit capture buckets")
    if any(type(size) is not int or size <= 0 for size in capture_buckets):
        raise ValueError("DSV4 prefill capture buckets must be positive integers")
    if max(capture_buckets) >= scatter_min_tokens:
        raise ValueError(
            "DSV4 prefill Graph buckets must all be smaller than the scattered "
            f"communication threshold ({scatter_min_tokens} tokens)"
        )
