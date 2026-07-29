from array import array
import unittest

from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNgramWorkerTail(unittest.TestCase):
    def test_array_prompt_and_list_output_match_full_concat(self):
        prompt = array("q", range(100_000))
        output = [100_000, 100_001, 100_002]

        actual = NGRAMWorker._efficient_concat_last_n(prompt, output, 18)
        expected = (list(prompt) + output)[-18:]

        self.assertEqual(actual, expected)
        self.assertIsInstance(actual, list)

    def test_output_tail_alone_can_fill_window(self):
        prompt = array("q", range(100_000))
        output = tuple(range(200_000, 200_032))

        actual = NGRAMWorker._efficient_concat_last_n(prompt, output, 18)

        self.assertEqual(actual, list(output[-18:]))
        self.assertIsInstance(actual, list)

    def test_short_combined_sequence_preserves_all_tokens(self):
        prompt = array("q", [1, 2])
        output = array("q", [3, 4])

        actual = NGRAMWorker._efficient_concat_last_n(prompt, output, 18)

        self.assertEqual(actual, [1, 2, 3, 4])

    def test_empty_output_uses_only_prompt_suffix(self):
        prompt = array("q", range(64))

        actual = NGRAMWorker._efficient_concat_last_n(prompt, [], 18)

        self.assertEqual(actual, list(range(46, 64)))


if __name__ == "__main__":
    unittest.main()
