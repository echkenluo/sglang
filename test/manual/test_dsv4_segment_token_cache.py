"""Exactness checks for the DeepSeek-V4 segment tokenization cache.

The cache is only worth having if it returns the very same ids as
``tokenizer.encode`` on the whole prompt: one different id would change the
model input and break radix-cache prefix matching for the rest of the
trajectory. These tests render agent-style conversations with the real DSV4
encoder and compare both paths, cold and warm, including the awkward cases
(marker text inside user content, whitespace next to markers, eviction).

Needs the DeepSeek-V4 tokenizer files; set DSV4_TOKENIZER_PATH. CPU only.
"""

import os
import random
import unittest

from sglang.srt.entrypoints.openai import encoding_dsv4
from sglang.srt.entrypoints.openai.segment_token_cache import (
    DSV4_SEGMENT_MARKERS,
    SegmentTokenCache,
    build_segment_token_cache,
)

TOKENIZER_PATH = os.environ.get("DSV4_TOKENIZER_PATH", "/models/DeepSeek-V4-Flash-0731")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]

SNIPPETS = [
    "def add(a, b):\n    return a + b\n",
    "  leading spaces and a tab\there  ",
    "\n\n两个换行开头，中文内容，结尾带空格 ",
    "literal markers in user text: <think> </think> <｜User｜> <｜Assistant｜>",
    "<｜end▁of▁sentence｜> pasted by a user, then more text",
    "diff --git a/x.py b/x.py\n@@ -1,3 +1,4 @@\n-import os\n+import os, sys\n",
    "emoji 🚀 and mixed ASCII/日本語/한국어 text 12345 3.14159e-7",
    "",
    " ",
]


def make_trajectory(rng, turns):
    messages = [
        {"role": "system", "content": "You are a coding agent. " + rng.choice(SNIPPETS)},
        {"role": "user", "content": "Fix the failing test.\n" + rng.choice(SNIPPETS)},
    ]
    for i in range(turns):
        call_id = "call_%d" % i
        messages.append(
            {
                "role": "assistant",
                "content": rng.choice(SNIPPETS),
                "reasoning_content": "thinking about step %d " % i + rng.choice(SNIPPETS),
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": '{"command": "sed -n %d,%dp x.py"}' % (i, i + 40),
                        },
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": rng.choice(SNIPPETS) * rng.randint(1, 30)}
        )
        if rng.random() < 0.3:
            messages.append({"role": "user", "content": rng.choice(SNIPPETS)})
    return messages


def render(messages, thinking_mode, with_tools):
    messages = [dict(m) for m in messages]
    if with_tools:
        messages[0]["tools"] = TOOLS
    return encoding_dsv4.encode_messages(messages, thinking_mode=thinking_mode)


@unittest.skipUnless(os.path.isdir(TOKENIZER_PATH), "DSV4 tokenizer files not found")
class TestSegmentTokenCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    def test_all_dsv4_markers_are_usable(self):
        cache = SegmentTokenCache(self.tokenizer, DSV4_SEGMENT_MARKERS, 1 << 20)
        self.assertEqual(set(cache.markers), set(DSV4_SEGMENT_MARKERS))

    def test_growing_trajectory_matches_plain_encode(self):
        rng = random.Random(7311)
        cache = SegmentTokenCache(self.tokenizer, DSV4_SEGMENT_MARKERS, 64 << 20)
        checked = 0
        for thinking_mode in ("thinking", "chat"):
            for with_tools in (True, False):
                messages = make_trajectory(rng, turns=12)
                # Replay the trajectory the way an agent does: every request
                # re-sends the conversation so far.
                for end in range(2, len(messages) + 1):
                    if messages[end - 1]["role"] == "assistant":
                        continue
                    text = render(messages[:end], thinking_mode, with_tools)
                    self.assertEqual(cache.encode(text), self.tokenizer.encode(text))
                    checked += 1
        self.assertGreater(checked, 40)
        # The point of the cache: later requests mostly hit.
        self.assertGreater(cache.hits, 5 * cache.misses)

    def test_warm_result_equals_cold_result(self):
        rng = random.Random(11)
        text = render(make_trajectory(rng, turns=6), "thinking", True)
        cache = SegmentTokenCache(self.tokenizer, DSV4_SEGMENT_MARKERS, 64 << 20)
        cold = cache.encode(text)
        warm = cache.encode(text)
        self.assertEqual(cold, warm)
        self.assertEqual(warm, self.tokenizer.encode(text))
        self.assertIsInstance(warm, list)
        self.assertTrue(all(type(t) is int for t in warm))

    def test_eviction_keeps_results_exact_and_budget_bounded(self):
        rng = random.Random(3)
        cache = SegmentTokenCache(self.tokenizer, DSV4_SEGMENT_MARKERS, 2000)
        for _ in range(5):
            text = render(make_trajectory(rng, turns=8), "thinking", True)
            self.assertEqual(cache.encode(text), self.tokenizer.encode(text))
            self.assertEqual(cache._cached_chars, sum(len(k) for k in cache._cache))
            # One oversized segment may stay; beyond that the budget holds.
            self.assertTrue(cache._cached_chars <= 2000 or len(cache._cache) == 1)

    def test_unsafe_tokenizer_disables_the_cache(self):
        class NoAddedVocab:
            def get_added_vocab(self):
                return {}

            def encode(self, text, add_special_tokens=True):
                return [1]

        self.assertIsNone(build_segment_token_cache(NoAddedVocab(), DSV4_SEGMENT_MARKERS, 1 << 20))


if __name__ == "__main__":
    unittest.main()
