import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_capture_input import build_token_stream, normalize_conversation


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return [len(message["content"]) for message in messages] + [99]


class BuildCaptureInputTest(unittest.TestCase):
    def test_normalize_conversation_maps_sharegpt_roles(self):
        messages = normalize_conversation(
            {
                "conversations": [
                    {"from": "human", "value": "question"},
                    {"from": "gpt", "value": "answer"},
                    {"from": "unknown", "value": "ignored"},
                ]
            }
        )
        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        )

    def test_build_is_exact_reproducible_and_records_truncation(self):
        records = [
            {
                "conversations": [
                    {"from": "human", "value": f"q{index}"},
                    {"from": "gpt", "value": f"answer{index}"},
                ]
            }
            for index in range(4)
        ]
        first_ids, first_selections = build_token_stream(
            records, FakeTokenizer(), target_tokens=5, seed=7
        )
        second_ids, second_selections = build_token_stream(
            records, FakeTokenizer(), target_tokens=5, seed=7
        )
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first_selections, second_selections)
        self.assertEqual(len(first_ids), 5)
        self.assertTrue(first_selections[-1]["truncated"])

    def test_rejects_insufficient_dataset(self):
        records = [
            {
                "conversations": [
                    {"from": "human", "value": "q"},
                    {"from": "gpt", "value": "a"},
                ]
            }
        ]
        with self.assertRaisesRegex(ValueError, "yielded only"):
            build_token_stream(records, FakeTokenizer(), target_tokens=10, seed=1)


if __name__ == "__main__":
    unittest.main()
