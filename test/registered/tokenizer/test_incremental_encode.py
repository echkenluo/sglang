import unittest

from sglang.srt.entrypoints.openai.incremental_encode import (
    IncrementalTokenizeCache,
)


class _BoundarySensitiveTokenizer:
    """Tiny greedy tokenizer whose newline token can merge across appends."""

    _SPECIAL = 3
    _DOUBLE_NEWLINE = 2
    _NEWLINE = 1
    _CHAR_BASE = 1000

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        ids = []
        i = 0
        while i < len(text):
            if text.startswith("<S>", i):
                ids.append(self._SPECIAL)
                i += 3
            elif text.startswith("\n\n", i):
                ids.append(self._DOUBLE_NEWLINE)
                i += 2
            elif text[i] == "\n":
                ids.append(self._NEWLINE)
                i += 1
            else:
                ids.append(self._CHAR_BASE + ord(text[i]))
                i += 1
        return ids

    def decode(self, ids):
        pieces = []
        for token_id in ids:
            if token_id == self._SPECIAL:
                pieces.append("<S>")
            elif token_id == self._DOUBLE_NEWLINE:
                pieces.append("\n\n")
            elif token_id == self._NEWLINE:
                pieces.append("\n")
            else:
                pieces.append(chr(token_id - self._CHAR_BASE))
        return "".join(pieces)


class TestIncrementalEncode(unittest.TestCase):
    def test_append_boundary_merge_stays_byte_exact(self):
        tokenizer = _BoundarySensitiveTokenizer()
        cache = IncrementalTokenizeCache(
            capacity=4,
            margin_chars=4,
            min_reuse_chars=1,
            tok_margin=32,
        )
        encode_kwargs = {"add_special_tokens": False}

        prompts = [
            "A" * 200 + "\n",
            "A" * 200 + "\n\nxxxx",
            "A" * 200 + "\n\nxxxxy",
            "A" * 200 + "\n\nxxxxy<S>\n",
        ]
        for prompt in prompts:
            expected = tokenizer.encode(prompt, **encode_kwargs)
            self.assertEqual(
                cache.encode(tokenizer, prompt, encode_kwargs),
                expected,
            )

        self.assertGreater(cache.hits, 0)
        for entry in cache.entries:
            self.assertNotIn((len(entry.text), len(entry.ids)), entry.checkpoints)


if __name__ == "__main__":
    unittest.main()
