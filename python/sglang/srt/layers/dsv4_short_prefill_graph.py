# SPDX-License-Identifier: Apache-2.0
"""Configuration boundary for the experimental short-Graph/long-scatter split."""


def validate_short_prefill_graph_buckets(
    backend,
    capture_buckets,
    scatter_min_tokens,
    allow_scattered_buckets=False,
    tp_size=1,
):
    """Reject any capture bucket that could execute the scattered token layout.

    Prefill buckets count aggregate tokens, not requests. This only validates
    disjoint routing ranges; capture/replay and model correctness need GPU tests.

    With ``allow_scattered_buckets`` a bucket at or above the threshold is
    captured in the scattered layout instead (the model picks the layout from
    the row count, and a capture batch has exactly the bucket's rows). Such a
    bucket must divide by the TP size: the Graph route skips the eager row
    padding, and a captured replay must not depend on padding rows that the
    live batch does not have.
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
    if not allow_scattered_buckets:
        if max(capture_buckets) >= scatter_min_tokens:
            raise ValueError(
                "DSV4 prefill Graph buckets must all be smaller than the scattered "
                f"communication threshold ({scatter_min_tokens} tokens)"
            )
        return
    if type(tp_size) is not int or tp_size <= 0:
        raise ValueError("DSV4 scattered prefill Graph needs a positive TP size")
    uneven = [
        size
        for size in capture_buckets
        if size >= scatter_min_tokens and size % tp_size != 0
    ]
    if uneven:
        raise ValueError(
            "DSV4 prefill Graph buckets captured in the scattered layout must "
            f"divide by the TP size ({tp_size}): {uneven}"
        )
