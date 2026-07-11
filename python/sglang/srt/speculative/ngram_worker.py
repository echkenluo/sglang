import logging
from time import perf_counter
from typing import List, Optional

import numpy as np
import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.observability.step_prof import StepProf
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    generate_token_bitmask,
    move_accept_tokens_to_target_kvcache,
    prepare_mamba_track_for_verify,
    record_stream_for_v2_verify,
)
from sglang.srt.speculative.triton_ops.cache_locs import (
    assign_extend_cache_locs_func as assign_extend_cache_locs_func,
)
from sglang.srt.utils.async_probe import maybe_detect_inf, maybe_detect_nan

logger = logging.getLogger(__name__)


USE_FULL_MASK = True


class NGRAMWorker(BaseSpecWorker):
    def alloc_memory_pool(self, **kwargs):
        # The target memory pool does not exist yet when __init__ runs.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self._target_worker.get_memory_pool()
        )
        self.max_batch_size = self.model_runner.max_running_requests
        self._init_preallocated_tensors()

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.enable_overlap = not server_args.disable_overlap_schedule
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.max_trie_depth: int = server_args.speculative_ngram_max_trie_depth
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        # req_to_token_pool / token_to_kv_pool_allocator are set in
        # alloc_memory_pool(), after the target pools are allocated.
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        self.adaptive_controller = None
        # rids of the last decode batch; used to erase corpus match state for
        # requests that left the batch (see forward_batch_generation).
        self._prev_decode_rids: set = set()

        # Variable-length drafting ("bet longer on strong match"): candidate
        # per-step draft lengths. The corpus is always queried at the cap
        # (= max tier = speculative_num_draft_tokens) and the draft tree is
        # truncated to the chosen tier. A single tier keeps static behavior.
        self.draft_tiers: List[int] = sorted(
            server_args.speculative_ngram_draft_tiers
            or [server_args.speculative_num_draft_tokens]
        )
        assert self.draft_tiers[-1] == self.draft_token_num
        # EMA of accepted drafts per verify step; drives tier selection.
        # Starts at 0 so cold start bets the smallest tier and ramps up.
        self._accept_len_ema = 0.0
        self._verify_step_count = 0
        self._last_step_stats = None  # (chosen_tier, can_run_cuda_graph)
        # Stride of the accept_tokens staging consumed from batch.spec_info in
        # _prepare_draft_tokens; _update_ngram_corpus reuses it after
        # batch.spec_info has been replaced by this step's verify input.
        self._prev_stride = self.draft_token_num

        # Sampling step-segment profiler (SGLANG_STEP_PROF=1). None when
        # disabled: every instrumentation site is guarded, zero overhead.
        self._step_prof = StepProf.maybe_create("ngram_spec", tp_rank)

        self.ngram_corpus = NgramCorpus(
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth,
            match_type=server_args.speculative_ngram_match_type,
            capacity=server_args.speculative_ngram_capacity,
            max_trie_depth=server_args.speculative_ngram_max_trie_depth,
            draft_token_num=server_args.speculative_num_draft_tokens,
            external_sam_budget=server_args.speculative_ngram_external_sam_budget,
            external_corpus_max_tokens=server_args.speculative_ngram_external_corpus_max_tokens,
        )
        if server_args.speculative_ngram_external_corpus_path is not None:
            from sglang.srt.speculative.cpp_ngram.external_corpus import (
                iter_external_corpus_chunks,
            )

            corpus_path = server_args.speculative_ngram_external_corpus_path
            chunks = list(
                iter_external_corpus_chunks(
                    corpus_path,
                    target_worker.tokenizer,
                    server_args.speculative_ngram_external_corpus_max_tokens,
                )
            )
            loaded = self.add_external_corpus(corpus_path, chunks)
            self.commit_corpus_load(corpus_path, loaded)
            logger.info(
                "Loaded external ngram corpus '%s' (%d tokens).",
                corpus_path,
                loaded,
            )

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> Optional[EagleDraftWorkerBase]:
        # NGRAM has no draft model; drafts come from the CPU-side corpus.
        return None

    def clear_cache_pool(self):
        self.ngram_corpus.reset()
        self._prev_decode_rids = set()

    def update_weights_from_tensor(self, recv_req):
        # NGRAM has no draft weights of its own — the n-gram corpus is a CPU
        # lookup structure built from request token streams — and its
        # `model_runner` is shared with the target worker. The scheduler
        # mixin dispatches via `self.draft_worker or self.tp_worker`, so
        # without this method any caller of `update_weights_from_tensor`
        # under `--speculative-algorithm NGRAM` raises AttributeError.
        return self.target_worker.update_weights_from_tensor(recv_req)

    def add_external_corpus(self, corpus_id: str, token_chunks: list[list[int]]) -> int:
        return self.ngram_corpus.load_external_corpus_named(corpus_id, token_chunks)

    def commit_corpus_load(self, corpus_id: str, loaded_token_count: int) -> None:
        self.ngram_corpus.commit_external_corpus_load(corpus_id, loaded_token_count)

    def remove_external_corpus(self, corpus_id: str) -> None:
        self.ngram_corpus.remove_external_corpus(corpus_id)

    def list_external_corpora(self) -> dict[str, int]:
        return self.ngram_corpus.list_external_corpora()

    def _efficient_concat_last_n(self, seq1: List[int], seq2: List[int], n: int):
        seq2_len = len(seq2)
        if seq2_len >= n:
            return seq2[-n:]

        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2

    def _init_preallocated_tensors(self):
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        self.draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrieve_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrieve_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )

        # Pinned host staging for the per-step numpy -> GPU uploads. A pageable
        # `torch.from_numpy(...)` source forces cudaMemcpyAsync through a
        # sync-ing driver staging buffer (profiling: ~8 pageable H2D + stream
        # syncs per step); writing into these reused pinned buffers first
        # (cheap CPU memcpy + dtype cast, <30KB) makes both H2D copies truly
        # async. Cross-step reuse is safe: before the next CPU write, the
        # accept tensors of the previous verify are read back with `.cpu()`
        # (in _prepare_draft_tokens under overlap, or in the result processor
        # in sync mode), which drains the stream past the previous H2D.
        self._drafts_pin = torch.empty(
            (max_total_drafts,), dtype=torch.int64, pin_memory=True
        )
        self._mask_pin = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, pin_memory=True
        )
        self._drafts_pin_np = self._drafts_pin.numpy()
        self._mask_pin_np = self._mask_pin.numpy()

    def _get_step_views(self, bs: int, d: int):
        """Carve contiguous (bs, d)/flat views over the cap-sized backings.

        The 2-D backings are (max_batch_size, cap); a [:bs, :d] slice would be
        non-contiguous for d < cap, so carve from the flat storage instead
        (downstream kernels assume dense (bs, d) layout).
        """
        n = bs * d
        return (
            self.retrieve_indexes.view(-1)[:n].view(bs, d),
            self.retrieve_next_token.view(-1)[:n].view(bs, d),
            self.retrieve_next_sibling.view(-1)[:n].view(bs, d),
            self.positions[:n],
            self.tree_mask[: n * d],
            self.draft_tokens[:n],
        )

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        # Signature must match BaseSpecWorker.on_verify_complete_cpu; the
        # result processor calls it with batch_size as a keyword argument.
        if num_correct_drafts_per_req and len(self.draft_tiers) > 1:
            batch_avg = sum(num_correct_drafts_per_req) / len(
                num_correct_drafts_per_req
            )
            self._accept_len_ema = 0.7 * self._accept_len_ema + 0.3 * batch_avg
            self._verify_step_count += 1
            if self._verify_step_count % 32 == 0 and self._last_step_stats is not None:
                # Under overlap this result lags the stashed step by one
                # iteration; good enough for low-frequency observability.
                tier, can_run_graph = self._last_step_stats
                logger.info(
                    "[ngram-var-draft] step=%d tier=%d accept_ema=%.2f "
                    "last_accept=%s cuda_graph=%s",
                    self._verify_step_count,
                    tier,
                    self._accept_len_ema,
                    num_correct_drafts_per_req,
                    can_run_graph,
                )
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(num_correct_drafts_per_req)

    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        bs = len(batch.reqs)
        # accept_tokens was laid out by the step that produced batch.spec_info;
        # with variable drafting its stride can differ from this step's tier,
        # so always read the producer's stride (and stash it for
        # _update_ngram_corpus, which runs after spec_info is replaced).
        stride = batch.spec_info.draft_token_num
        self._prev_stride = stride

        prev_token_ids, prev_accept_lens = (
            batch.spec_info.accept_tokens,
            batch.spec_info.accept_lens,
        )
        prof = self._step_prof
        t = perf_counter() if prof else 0.0
        if not prev_token_ids.is_cpu:
            prev_token_ids = prev_token_ids.cpu()
            prev_accept_lens = prev_accept_lens.cpu()
        # Worker-level staging: written here at draft prep, consumed by
        # _update_ngram_corpus after verify within the same forward call.
        self.prev_token_ids = prev_token_ids.tolist()
        self.prev_accept_lens = prev_accept_lens.tolist()
        if prof:
            # GPU wait: D2H sync on the previous step's accept tensors.
            prof.add("accept_sync", perf_counter() - t)
            t = perf_counter()

        self.ngram_corpus.synchronize()
        req_ids = []
        batch_tokens = []
        total_lens = []
        assert len(batch.reqs) == len(self.prev_accept_lens)
        # Overlap mode processes results one iteration behind, so the last
        # round's accepted tokens are not yet in req.output_ids and must be
        # spliced in from spec_info. Sync mode and grammar batches process
        # results before the next draft prep, so output_ids is already
        # complete and splicing would duplicate the tail.
        use_prev_tokens = self.enable_overlap and not batch.has_grammar
        i = 0
        for req in batch.reqs:
            prev_tokens = (
                self.prev_token_ids[i * stride : i * stride + self.prev_accept_lens[i]]
                if use_prev_tokens
                else []
            )
            check_token = self._efficient_concat_last_n(
                list(req.origin_input_ids),
                list(req.output_ids[-self.max_trie_depth :]) + prev_tokens,
                self.max_trie_depth,
            )
            req_ids.append(req.rid)
            batch_tokens.append(check_token)
            i += 1
            total_lens.append(
                len(req.origin_input_ids) + len(req.output_ids) + len(prev_tokens)
            )
        req_drafts, mask = self.ngram_corpus.batch_get(
            req_ids, batch_tokens, total_lens
        )
        if prof:
            # Corpus insert-thread sync + token-tail assembly + cpp batchMatch.
            prof.add("draft_lookup", perf_counter() - t)
        total_draft_token_num = len(req_drafts)

        # Check if speculative decoding is needed; here we always enforce it
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        return req_drafts, mask

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        # Decode-only: extend goes through the plain target forward, and an
        # IDLE batch must keep its forward_mode instead of being rewritten to
        # TARGET_VERIFY below (relevant once DP attention support lands).
        if not batch.forward_mode.is_decode():
            return

        bs = len(batch.reqs)
        cap = self.draft_token_num

        req_drafts, mask = self._prepare_draft_tokens(batch)
        prof = self._step_prof
        t = perf_counter() if prof else 0.0
        d = self._choose_draft_tier(bs, mask)
        if d < cap:
            # fillResult emits BFS order, so the first d nodes of each
            # request's tree form a prefix-closed subtree: plain truncation
            # keeps tokens/mask a valid draft tree.
            req_drafts = np.ascontiguousarray(
                req_drafts.reshape(bs, cap)[:, :d]
            ).reshape(-1)
            mask = np.ascontiguousarray(
                mask.reshape(bs, cap, cap)[:, :d, :d]
            ).reshape(-1)
        if prof:
            prof.add("tier_select", perf_counter() - t)
            t = perf_counter()

        (
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            positions,
            tree_mask,
            draft_tokens,
        ) = self._get_step_views(bs, d)

        n_tok = bs * d
        n_mask = n_tok * d
        # CPU write into pinned staging (int64 -> bool cast for the mask
        # happens here, on <=27K elements), then same-dtype async H2D.
        self._drafts_pin_np[:n_tok] = req_drafts
        self._mask_pin_np[:n_mask] = mask
        tree_mask.copy_(self._mask_pin[:n_mask], non_blocking=True)
        draft_tokens.copy_(self._drafts_pin[:n_tok], non_blocking=True)
        if prof:
            # Pinned staging write + async H2D of draft tokens + qlen mask.
            prof.add("staging_h2d", perf_counter() - t)
            t = perf_counter()

        # generate positions and some indices using tree_mask
        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,  # mutable
            retrieve_index,  # mutable
            retrieve_next_token,  # mutable
            retrieve_next_sibling,  # mutable
            bs,
            d,
        )
        if prof:
            prof.add("tree_reconstruct", perf_counter() - t)
            t = perf_counter()

        # NOTE: QLEN_MASK is faster than FULL_MASK, but requires corresponding changes in flashinfer.
        # Testing shows about 8% performance improvement (the effect is roughly proportional to batch size).
        if USE_FULL_MASK:
            # Assemble the full mask on GPU from the qlen mask already staged
            # in `tree_mask`. The old path re-uploaded each request's (d, d)
            # block from pageable numpy (sync-forcing cudaMemcpyAsync) and
            # paid per-req ones/cat/cast kernels; this is one fused ones-fill
            # plus bs strided D2D writes, bit-identical bool layout
            # (row r of request i = [True x seq_len_i | tree row r]).
            qlen_mask = tree_mask.view(bs, d, d)
            # [:bs] guard: total below sums the list, so any padded tail in
            # seq_lens_cpu (unlike the old per-index reads) must be excluded.
            seq_lens_list = batch.seq_lens_cpu[:bs].tolist()
            total = d * (sum(seq_lens_list) + bs * d)
            full_mask_out = torch.ones(total, dtype=torch.bool, device=self.device)
            off = 0
            for i in range(bs):
                s = seq_lens_list[i]
                full_mask_out[off : off + d * (s + d)].view(d, s + d)[
                    :, s:
                ] = qlen_mask[i]
                off += d * (s + d)
            tree_mask = full_mask_out
        if prof:
            # Full-mask GPU assembly (ones fill + bs strided D2D writes).
            prof.add("full_mask", perf_counter() - t)
            t = perf_counter()

        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.input_ids = draft_tokens
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=batch.seq_lens + d,
            batch_size=bs,
            draft_token_num=d,
            device=self.device,
        )

        prepare_mamba_track_for_verify(batch)

        batch.spec_info = NgramVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            draft_token_num=d,
        )
        if prof:
            prof.add("alloc_prep", perf_counter() - t)

    def _choose_draft_tier(self, bs: int, mask: np.ndarray) -> int:
        """Pick this step's draft length from self.draft_tiers.

        Signals: the draft tree's deepest chain (how far the corpus can
        continue right now) and the accept-length EMA (how much of recent
        bets was actually accepted). Bet roughly twice the recent accept
        length, bounded by the available continuation, then round up to the
        smallest tier that covers it.
        """
        if len(self.draft_tiers) == 1:
            return self.draft_tiers[0]
        cap = self.draft_token_num
        # A mask row sums root + ancestors + self, so node depth in draft
        # tokens = row sum - 1; zero-padded rows read as depth 1 and never
        # dominate a real chain.
        tree_depth = int(mask.reshape(bs, cap, cap).sum(axis=2).max()) - 1
        want = min(tree_depth, int(2.0 * self._accept_len_ema) + 2)
        for tier in self.draft_tiers:
            if tier >= want:
                return tier
        return self.draft_tiers[-1]

    def _update_ngram_corpus(self, batch: ScheduleBatch):
        batch_tokens = []
        # prev_token_ids was staged by _prepare_draft_tokens with the stride of
        # the PREVIOUS step's spec_info (batch.spec_info now holds this step's).
        i, stride = 0, self._prev_stride
        # Same splice condition as _prepare_draft_tokens: only overlap mode
        # has accepted tokens missing from req.output_ids.
        use_prev_tokens = self.enable_overlap and not batch.has_grammar
        for req in batch.reqs:
            # FIXME: Whether to insert 'extend' into the cache or not, after testing,
            # there is not much difference, so we will not insert it for now.
            # if batch.forward_mode.is_extend():
            #     put_ids = req.origin_input_ids + req.output_ids
            # else:
            prev_tokens = (
                self.prev_token_ids[i * stride : i * stride + self.prev_accept_lens[i]]
                if use_prev_tokens
                else []
            )
            put_ids = self._efficient_concat_last_n(
                list(req.origin_input_ids),
                list(req.output_ids[-self.max_trie_depth :]) + prev_tokens,
                self.max_trie_depth,
            )
            batch_tokens.append(put_ids)
            i += 1
        self.ngram_corpus.batch_put(batch_tokens)

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        fwd_stream = torch.get_device_module(self.device).current_stream()
        record_stream_for_v2_verify(batch, None, fwd_stream)
        bs = len(batch.reqs)

        prof = self._step_prof
        t_step = perf_counter() if prof else 0.0

        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
        self._prepare_for_speculative_decoding(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)

        verify_input: NgramVerifyInput = batch.spec_info
        accept_lens = torch.ones(bs, dtype=torch.int32, device=self.device)

        if batch.forward_mode.is_target_verify():
            # Prepare grammar data on CPU if needed
            if batch.has_grammar:
                retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
                retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
                draft_tokens_cpu = verify_input.draft_token.view(
                    verify_input.retrieve_next_token.shape
                ).cpu()

            t = perf_counter() if prof else 0.0
            batch_result = self.target_worker.forward_batch_generation(
                batch, is_verify=True
            )
            if prof:
                # Target verify forward, launch-to-return on this thread
                # (graph replay dispatch or eager launch; not kernel time).
                prof.add("verify_fwd", perf_counter() - t)

            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )

            verify_input: NgramVerifyInput = batch.spec_info
            vocab_mask = None
            if batch.has_grammar:
                # Generate the logit mask for structured output.
                # Overlap the CPU operations for bitmask generation with the forward pass.
                vocab_mask = generate_token_bitmask(
                    batch.reqs,
                    verify_input,
                    retrieve_next_token_cpu,
                    retrieve_next_sibling_cpu,
                    draft_tokens_cpu,
                    batch.sampling_info.vocab_size,
                )

                if vocab_mask is not None:
                    assert verify_input.grammar is not None
                    vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                    # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                    # and will be applied to produce wrong results
                    batch.sampling_info.vocab_mask = None

            # Sample
            maybe_detect_nan(
                logits_output.next_token_logits, "verify: target model logits"
            )
            maybe_detect_inf(
                logits_output.next_token_logits, "verify: target model logits"
            )
            t = perf_counter() if prof else 0.0
            (
                predict,
                accept_lens,
                accept_index,
            ) = eagle_sample(verify_input, batch, logits_output, vocab_mask)
            if prof:
                prof.add("verify_sample", perf_counter() - t)
                t = perf_counter()
            new_seq_lens = batch.seq_lens + accept_lens
            # This step's draft length; with variable drafting it can differ
            # from self.draft_token_num (the cap), so every layout-stride
            # consumer below must use it.
            step_draft_token_num = verify_input.draft_token_num
            self._last_step_stats = (step_draft_token_num, can_run_cuda_graph)
            commit_mamba_states_after_verify(
                self.target_worker,
                batch,
                accept_lens,
                accept_index,
                step_draft_token_num,
            )
            if prof:
                prof.add("mamba_commit", perf_counter() - t)
                t = perf_counter()
            accept_tokens = predict[accept_index].flatten()
            next_token_ids = accept_tokens

            # The KV mover expects drafts-only counts. NGRAM's
            # accept_lens includes the bonus token, matching scheduler output.
            num_correct_drafts_per_req = accept_lens - 1
            move_accept_tokens_to_target_kvcache(
                batch,
                accept_index,
                num_correct_drafts_per_req,
                self.token_to_kv_pool_allocator,
            )
            if prof:
                # Accept gather + accepted-KV relocation launches.
                prof.add("kv_move", perf_counter() - t)
            if batch.return_logprob:
                # The last arg is the accept_index row width minus 1. NGRAM's
                # accept_index is (bs, draft_token_num) -- the tree depth is not
                # bounded by spec_steps like EAGLE's (bs, spec_steps + 1).
                compute_spec_v2_logprobs(
                    batch,
                    logits_output,
                    predict,
                    accept_index,
                    step_draft_token_num - 1,
                )

            if on_publish is not None:
                on_publish(new_seq_lens)

            t = perf_counter() if prof else 0.0
            self._update_ngram_corpus(batch)
            # Erase match state of requests that left the decode batch.
            # req.finished() is unusable here: under overlap it flips at result
            # processing, one iteration after the request left the batch.
            # The last batch's entries persist while idle (bounded, small).
            cur_rids = {req.rid for req in batch.reqs}
            departed_rids = self._prev_decode_rids - cur_rids
            if departed_rids:
                self.ngram_corpus.erase_match_state(list(departed_rids))
            self._prev_decode_rids = cur_rids
            batch.forward_mode = ForwardMode.DECODE
            # Row stride of accept_tokens/next_token_ids in this result.
            result_stride = step_draft_token_num
            if prof:
                # CPU corpus insert enqueue + match-state GC.
                prof.add("corpus_update", perf_counter() - t)
                prof.end_step(perf_counter() - t_step)

        else:
            batch_result = self.target_worker.forward_batch_generation(batch)
            logits_output, predict, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )
            new_seq_lens = batch.seq_lens.clone()

            accept_tokens = torch.zeros(
                bs, self.draft_token_num, dtype=torch.int32, device=self.device
            )
            accept_tokens[:, 0] = predict
            accept_tokens = accept_tokens.flatten()
            next_token_ids = predict
            result_stride = self.draft_token_num

            if on_publish is not None:
                on_publish(new_seq_lens)

        # Construct the next draft input. draft_token_num must be the stride
        # accept_tokens was laid out with (this step's tier), NOT the cap: the
        # next step's _prepare_draft_tokens and the result processor both
        # unpack with it.
        next_draft_input = NgramVerifyInput(
            draft_token_num=result_stride,
            new_seq_lens=new_seq_lens,
            accept_tokens=accept_tokens,
            accept_lens=accept_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
            # Consumed by the non-overlap V2 scheduler branch to advance
            # batch.seq_lens after the isolation restore; overlap mode relays
            # it via on_publish instead.
            new_seq_lens=new_seq_lens,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=result_stride,
        )
