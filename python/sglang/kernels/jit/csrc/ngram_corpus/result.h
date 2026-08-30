#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ngram {

struct Result {
  std::vector<int32_t> token;
  std::vector<uint8_t> mask;
  // Number of real nodes in token before fillResult pads the block. Padding
  // cannot be inferred later: token 0 is legal and a padded row has the same
  // mask as a real depth-1 child.
  int32_t num_valid = 0;

  void truncate(size_t n);
};

struct Node {
  std::unordered_map<int32_t, int32_t> next;
};

Result fillResult(int last_token, int draft_token_num, std::vector<Node>& tree, int root);
std::vector<std::vector<int32_t>> extractLeafPaths_(const Result& result);
Result buildResultFromLeafPaths_(int last_token, int draft_token_num, const std::vector<std::vector<int32_t>>& paths);
Result combineRootResults_(int last_token, int draft_token_num, const Result& primary, const Result& secondary);

}  // namespace ngram
