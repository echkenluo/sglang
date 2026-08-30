from __future__ import annotations

import numpy as np


def derive_tree_links(
    tree_mask: np.ndarray, batch_size: int, draft_token_num: int
) -> tuple[np.ndarray, np.ndarray]:
    """Derive next-child and next-sibling links from an ancestor mask.

    ``tree_mask[b, i, j]`` means node ``j`` is an ancestor of node ``i``.
    The result matches ``reconstruct_indices_from_tree_mask`` but stays on the
    host, where DSV4 chain-only NGRAM already owns the corpus result.
    """
    bs = int(batch_size)
    d = int(draft_token_num)
    mask = np.asarray(tree_mask, dtype=bool)
    if mask.size != bs * d * d:
        raise ValueError(
            f"tree mask size {mask.size} does not match bs * D * D = {bs * d * d}"
        )
    tree = mask.reshape(bs, d, d)
    node_order = np.arange(d)
    ancestors = tree & (node_order < node_order[:, None])
    parents = np.where(ancestors.any(-1), (ancestors * node_order).argmax(-1), -1)

    next_token = np.full((bs, d), -1, dtype=np.int64)
    next_sibling = np.full((bs, d), -1, dtype=np.int64)
    for batch_idx in range(bs):
        # Descending scan: all later siblings/children are already recorded.
        earliest_child_of: dict[int, int] = {}
        for node_idx in reversed(range(d)):
            next_token[batch_idx, node_idx] = earliest_child_of.get(node_idx, -1)
            parent = int(parents[batch_idx, node_idx])
            if parent >= 0:
                next_sibling[batch_idx, node_idx] = earliest_child_of.get(parent, -1)
                earliest_child_of[parent] = node_idx
    return next_token, next_sibling


def select_longest_ngram_chains(
    draft_tokens: np.ndarray,
    tree_mask: np.ndarray,
    valid_node_lens: np.ndarray,
    draft_token_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert each padded NGRAM tree to its earliest longest root-to-leaf chain.

    ``valid_node_lens`` is producer truth from the corpus. It is required because
    token 0 is legal and padded rows are structurally indistinguishable from real
    depth-1 token-0 children. The returned masks contain one linear chain; padded
    rows are isolated self-nodes so acceptance cannot walk into the padding tail.
    """
    valid_node_lens = np.asarray(valid_node_lens, dtype=np.int64).reshape(-1)
    bs = valid_node_lens.size
    d = int(draft_token_num)
    if d <= 0:
        raise ValueError(f"draft_token_num must be positive, got {d}")

    tokens = np.asarray(draft_tokens, dtype=np.int64)
    masks = np.asarray(tree_mask, dtype=np.int64)
    if tokens.size != bs * d:
        raise ValueError(
            f"draft token size {tokens.size} does not match bs * D = {bs * d}"
        )
    if masks.size != bs * d * d:
        raise ValueError(
            f"tree mask size {masks.size} does not match bs * D * D = {bs * d * d}"
        )
    if np.any(valid_node_lens < 1) or np.any(valid_node_lens > d):
        raise ValueError(
            f"valid node lengths must be within [1, {d}], got "
            f"{valid_node_lens.tolist()}"
        )

    tokens = tokens.reshape(bs, d)
    masks = masks.reshape(bs, d, d).astype(bool, copy=False)
    chain_tokens = np.zeros((bs, d), dtype=np.int64)
    chain_masks = np.zeros((bs, d, d), dtype=np.int64)
    chain_lens = np.empty(bs, dtype=np.int64)

    for batch_idx, valid_nodes_value in enumerate(valid_node_lens):
        valid_nodes = int(valid_nodes_value)
        valid_mask = masks[batch_idx, :valid_nodes, :valid_nodes]
        depths = valid_mask.sum(axis=1)
        leaf = int(np.argmax(depths))
        path = np.flatnonzero(valid_mask[leaf])
        if path.size == 0 or path[0] != 0:
            raise ValueError(
                f"request {batch_idx} has an invalid NGRAM root path at row {leaf}"
            )

        chain_len = int(path.size)
        chain_tokens[batch_idx, :chain_len] = tokens[batch_idx, path]
        chain_lens[batch_idx] = chain_len

        # Valid rows form a causal linear chain. Invalid rows remain isolated so
        # retrieve links derived from this mask terminate exactly at chain_len.
        chain_masks[batch_idx, np.arange(d), np.arange(d)] = 1
        chain_masks[batch_idx, :chain_len, :chain_len] = np.tri(
            chain_len, dtype=np.int64
        )

    return chain_tokens.reshape(-1), chain_masks.reshape(-1), chain_lens
