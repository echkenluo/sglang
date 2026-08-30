import unittest

import numpy as np

from sglang.srt.speculative.ngram_chain import (
    derive_chain_links,
    derive_tree_links,
    resolve_dsv4_chain_only,
    select_longest_ngram_chains,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNgramChain(CustomTestCase):
    def test_selects_earliest_longest_path(self):
        tokens = np.array([3, 4, 44, 5, 55, 6, 66, 77])
        mask = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 0, 0],
                [1, 1, 0, 1, 0, 1, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 0, 1, 1],
            ]
        )

        chain, linear_mask, lens = select_longest_ngram_chains(
            tokens, mask, np.array([8]), 8
        )

        self.assertEqual(lens.tolist(), [5])
        self.assertEqual(chain.tolist(), [3, 44, 55, 66, 77, 0, 0, 0])
        linear_mask = linear_mask.reshape(8, 8)
        np.testing.assert_array_equal(linear_mask[:5, :5], np.tri(5, dtype=int))
        self.assertEqual(linear_mask[5].tolist(), [0, 0, 0, 0, 0, 1, 0, 0])

        next_token, next_sibling = derive_tree_links(linear_mask, 1, 8)
        self.assertEqual(next_token.tolist(), [[1, 2, 3, 4, -1, -1, -1, -1]])
        self.assertEqual(next_sibling.tolist(), [[-1] * 8])
        direct_next_token, direct_next_sibling = derive_chain_links(lens, 8)
        np.testing.assert_array_equal(direct_next_token, next_token)
        np.testing.assert_array_equal(direct_next_sibling, next_sibling)

    def test_dsv4_capability_must_match_loaded_architecture(self):
        architectures = ["DeepseekV4ForCausalLM"]
        self.assertTrue(resolve_dsv4_chain_only(True, architectures))
        self.assertFalse(resolve_dsv4_chain_only(False, ["Qwen3ForCausalLM"]))
        with self.assertRaisesRegex(ValueError, "capability state"):
            resolve_dsv4_chain_only(False, architectures)
        with self.assertRaisesRegex(ValueError, "capability state"):
            resolve_dsv4_chain_only(True, ["Qwen3ForCausalLM"])

    def test_valid_length_distinguishes_real_zero_from_padding(self):
        tokens = np.array([[55, 0, 66, 0], [55, 0, 0, 0]])
        masks = np.array(
            [
                [
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 1, 0],
                    [1, 0, 0, 1],
                ],
                [
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [1, 0, 1, 0],
                    [1, 0, 0, 1],
                ],
            ]
        )

        chain, _, lens = select_longest_ngram_chains(tokens, masks, np.array([3, 1]), 4)

        self.assertEqual(chain.reshape(2, 4).tolist(), [[55, 0, 66, 0], [55, 0, 0, 0]])
        self.assertEqual(lens.tolist(), [3, 1])

    def test_rejects_invalid_producer_lengths(self):
        with self.assertRaises(ValueError):
            select_longest_ngram_chains(
                np.zeros(4), np.eye(4), np.array([0]), draft_token_num=4
            )


if __name__ == "__main__":
    unittest.main()
