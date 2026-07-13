"""Incremental prompt tokenization cache (env ``SGLANG_INCREMENTAL_TOKENIZE``).

Coding-agent chat prompts are append-only: request N+1 renders to (request
N's full text + tool output + a small delta), yet ``tokenizer.encode`` reruns
from scratch every request (~2us/token: 19ms at 9k prompt, 50-120ms at
production 30-60k). This cache keeps per-conversation (rendered_text ->
token_ids) prefixes and only encodes the tail.

Modes (env):
  SGLANG_INCREMENTAL_TOKENIZE=            off (default) — call sites bypass
  SGLANG_INCREMENTAL_TOKENIZE=1|on        splice mode
  SGLANG_INCREMENTAL_TOKENIZE=verify      splice + shadow full encode; on any
                                          token mismatch: log details, USE THE
                                          FULL RESULT, and self-heal the cache
                                          (graduation gate before trusting on)
  SGLANG_INCREMENTAL_TOKENIZE_CAPACITY=N  LRU entries (default 32)

Correctness design (BPE boundary red line):
  * A splice point must be a provably char<->token aligned position:
      one synthetic near-tail checkpoint per store, built by decoding the
      last ~TOK_MARGIN tokens, verifying ``text.endswith(decoded)``, and then
      proving token-level seam closure against the original full encode.
      Request-end boundaries are not checkpoints: BPE can merge across an
      append boundary, invalidating both their char and token offsets.
  * The split additionally backs off MARGIN chars below the char-LCP so that
    a pretokenizer chunk crossing the split would have to exceed MARGIN chars
    to desynchronize the seam (possible only for pathological giant chunks).
  * The tail is re-encoded with add_special_tokens=False (prefix already
    carries any leading specials from its original full encode).
  * Residual pathological risk is exactly what verify mode measures; ON mode
    should only be enabled after a zero-mismatch verify canary.

Failure containment: every cache-path exception falls back to a full encode
(counted, logged at aggregate); the cache can only cost a log line, never a
request. Single-threaded by construction (serving handlers run on the event
loop; multi-tokenizer workers each own a private cache).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_MODE: Optional[str] = None

# Defaults; the cache constructor takes overrides for unit tests.
DEFAULT_MARGIN_CHARS = 256
DEFAULT_MIN_REUSE_CHARS = 1024
DEFAULT_TOK_MARGIN = 96
LOG_EVERY = 32


def inc_tok_mode() -> str:
    """'' (off) | 'on' | 'verify'; cached after first read."""
    global _MODE
    if _MODE is None:
        raw = os.environ.get("SGLANG_INCREMENTAL_TOKENIZE", "").strip().lower()
        if raw in ("", "0", "false", "off"):
            _MODE = ""
        elif raw == "verify":
            _MODE = "verify"
        else:
            _MODE = "on"
    return _MODE


def _lcp_len(a: str, b: str) -> int:
    """Longest common prefix length via binary search on C-speed slice compare."""
    lo, hi = 0, min(len(a), len(b))
    if hi == 0 or a[0] != b[0]:
        return 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class _Entry:
    __slots__ = ("text", "ids", "checkpoints")

    def __init__(self, text: str, ids: List[int], checkpoints: List[Tuple[int, int]]):
        self.text = text
        self.ids = ids
        # ascending [(char_len, tok_len)]: provably aligned splice points
        self.checkpoints = checkpoints


class IncrementalTokenizeCache:
    def __init__(
        self,
        capacity: Optional[int] = None,
        margin_chars: int = DEFAULT_MARGIN_CHARS,
        min_reuse_chars: int = DEFAULT_MIN_REUSE_CHARS,
        tok_margin: int = DEFAULT_TOK_MARGIN,
    ):
        self.capacity = capacity or int(
            os.environ.get("SGLANG_INCREMENTAL_TOKENIZE_CAPACITY", "32")
        )
        self.margin_chars = margin_chars
        self.min_reuse_chars = min_reuse_chars
        self.tok_margin = tok_margin
        self.entries: List[_Entry] = []  # LRU, front = most recent
        self.hits = 0
        self.misses = 0
        self.fallbacks = 0
        self.mismatches = 0
        self.reused_chars = 0
        self.total_chars = 0
        self.requests = 0

    # ---- public ----------------------------------------------------------
    def encode(self, tokenizer, text: str, encode_kwargs: dict) -> List[int]:
        self.requests += 1
        self.total_chars += len(text)
        try:
            ids, reused = self._spliced_encode(tokenizer, text, encode_kwargs)
        except Exception as e:
            self.fallbacks += 1
            logger.warning("[inc-tok] splice path failed (%r); full encode", e)
            ids, reused = list(tokenizer.encode(text, **encode_kwargs)), 0

        if inc_tok_mode() == "verify" and reused:
            full = list(tokenizer.encode(text, **encode_kwargs))
            if ids != full:
                self.mismatches += 1
                first = next(
                    (k for k, (x, y) in enumerate(zip(ids, full)) if x != y),
                    min(len(ids), len(full)),
                )
                logger.warning(
                    "[inc-tok] MISMATCH len(spliced)=%d len(full)=%d "
                    "first_diff_tok=%d reused_chars=%d text_len=%d "
                    "-> using full encode (self-heal)",
                    len(ids),
                    len(full),
                    first,
                    reused,
                    len(text),
                )
                ids, reused = full, 0

        self.reused_chars += reused
        self._store(tokenizer, text, ids, reused, encode_kwargs)
        if self.requests % LOG_EVERY == 0:
            self._log_stats()
        return ids

    # ---- internals -------------------------------------------------------
    def _find_best(self, text: str) -> Tuple[Optional[_Entry], int]:
        best, best_lcp = None, 0
        for e in self.entries:
            # cheap reject before the binary search
            if not e.text or e.text[0] != text[0]:
                continue
            lcp = _lcp_len(e.text, text)
            if lcp > best_lcp:
                best, best_lcp = e, lcp
        return best, best_lcp

    def _spliced_encode(
        self, tokenizer, text: str, encode_kwargs: dict
    ) -> Tuple[List[int], int]:
        best, lcp = self._find_best(text)
        if best is None or lcp < self.min_reuse_chars:
            self.misses += 1
            return list(tokenizer.encode(text, **encode_kwargs)), 0

        target = lcp - self.margin_chars
        ck_char, ck_tok = 0, 0
        for c, t in best.checkpoints:
            if c <= target:
                ck_char, ck_tok = c, t
            else:
                break
        if ck_char < self.min_reuse_chars:
            self.misses += 1
            return list(tokenizer.encode(text, **encode_kwargs)), 0

        self.hits += 1
        tail_kwargs = dict(encode_kwargs)
        tail_kwargs["add_special_tokens"] = False
        tail_ids = list(tokenizer.encode(text[ck_char:], **tail_kwargs))
        return best.ids[:ck_tok] + tail_ids, ck_char

    def _near_tail_checkpoint(
        self, tokenizer, text: str, ids: List[int], encode_kwargs: dict
    ) -> Optional[Tuple[int, int]]:
        """Synthesize a token-aligned checkpoint far enough from the end to
        clear the margin zone on the NEXT pure-append request (target is a
        CHAR distance — token counts alone break on char-dense tokenizers,
        e.g. CJK at ~1 char/token). decode() of byte-level BPE is
        concatenative, so char_pos = len(text) - len(decode(tail)) is exact —
        unless the split lands inside a multi-byte char; slide +-4 tokens and
        verify with text.endswith(decoded).

        ``endswith`` alone is insufficient: independently encoding a suffix
        can tokenize leading whitespace or a special-token boundary
        differently from the same characters inside the full prompt. A
        candidate is accepted only when prefix_ids + encode(suffix) exactly
        reconstructs the stored full token sequence.
        """
        n = len(ids)
        need_chars = self.margin_chars + 64
        k = max(self.tok_margin, 8)
        tail_kwargs = dict(encode_kwargs)
        tail_kwargs["add_special_tokens"] = False
        while k < n:
            split0 = n - k
            for delta in (0, -1, 1, -2, 2, -3, 3, -4, 4, -8, 8, -16, 16):
                split = split0 + delta
                if split <= 0 or split >= n:
                    continue
                try:
                    tail_txt = tokenizer.decode(ids[split:])
                except Exception:
                    continue
                if not tail_txt or not text.endswith(tail_txt):
                    continue
                if len(tail_txt) < need_chars:
                    continue
                char_pos = len(text) - len(tail_txt)
                try:
                    seam_tail_ids = list(
                        tokenizer.encode(text[char_pos:], **tail_kwargs)
                    )
                except Exception:
                    continue
                if ids[:split] + seam_tail_ids == ids:
                    return (char_pos, split)
            k *= 2
        return None

    def _store(
        self,
        tokenizer,
        text: str,
        ids: List[int],
        reused: int,
        encode_kwargs: dict,
    ) -> None:
        try:
            best, lcp = self._find_best(text)
            checkpoints: List[Tuple[int, int]] = []
            if best is not None and lcp == len(best.text) and reused:
                # pure append on the matched conversation: inherit its points
                checkpoints = [c for c in best.checkpoints if c[0] < len(text)]
            ck = self._near_tail_checkpoint(tokenizer, text, ids, encode_kwargs)
            if ck is not None and (not checkpoints or ck[0] > checkpoints[-1][0]):
                checkpoints.append(ck)
            checkpoints = checkpoints[-64:]

            entry = _Entry(text, list(ids), checkpoints)
            if best is not None and lcp >= self.min_reuse_chars:
                # same conversation grew (or diverged): replace in place
                self.entries.remove(best)
            self.entries.insert(0, entry)
            del self.entries[self.capacity :]
        except Exception as e:
            logger.warning("[inc-tok] store failed (%r); cache entry skipped", e)

    def _log_stats(self) -> None:
        pct = 100.0 * self.reused_chars / max(self.total_chars, 1)
        logger.info(
            "[inc-tok] req=%d hit=%d miss=%d fallback=%d mismatch=%d "
            "reused_chars=%.1f%% entries=%d mode=%s",
            self.requests,
            self.hits,
            self.misses,
            self.fallbacks,
            self.mismatches,
            pct,
            len(self.entries),
            inc_tok_mode(),
        )
