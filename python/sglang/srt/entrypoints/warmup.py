from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np
import tqdm

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.environ import envs
from sglang.srt.utils.mok_fault_progress import record_warmup_input, replay_warmup_input
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}


def warmup(name: str):
    def decorator(fn):
        _warmup_registry[name] = fn
        return fn

    return decorator


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)


@warmup("whisper_autodetect")
async def whisper_autodetect(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Pre-compile the xgrammar FSM for both Whisper auto-detect regexes.

    The first request that uses each structured-generation regex incurs a
    ~15-20s compilation cost. xgrammar caches compiled grammars by the
    exact regex string, so we warm both the notimestamps and timestamps
    variants here — otherwise the first ``language=None +
    timestamp_granularities`` request would still pay the full spike.
    """
    # A short silent audio encoded as base64 WAV (0.1s, 16kHz, mono) —
    # soundfile produces the WAV header + PCM data from a list of floats.
    import base64
    import io

    import soundfile as sf

    from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
        FUSED_AUTODETECT_FLAG,
        WHISPER_AUTODETECT_REGEX,
        WHISPER_AUTODETECT_TS_REGEX,
    )

    sr, dur = 16000, 0.1
    n = int(sr * dur)
    buf = io.BytesIO()
    sf.write(buf, [0.0] * n, sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

    for variant_name, regex in (
        ("notimestamps", WHISPER_AUTODETECT_REGEX),
        ("timestamps", WHISPER_AUTODETECT_TS_REGEX),
    ):
        logger.info(
            "Compiling Whisper auto-detect regex FSM (%s, one-time, ~15-20s)...",
            variant_name,
        )
        req = GenerateReqInput(
            text="",
            audio_data=audio_data_uri,
            sampling_params={
                "max_new_tokens": 4,
                "temperature": 0,
                "regex": regex,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                FUSED_AUTODETECT_FLAG: True,
            },
            modalities=["audio"],
        )
        # PD prefill servers assert req.bootstrap_room is not None in the
        # default follow_bootstrap_room scheduler; the fake values match
        # what the voice_chat warmup uses for the same reason.
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        # Drain the generator so the FSM is fully installed and any
        # downstream exception surfaces instead of being swallowed after
        # the first yield.
        async for _ in tokenizer_manager.generate_request(req, None):
            pass
    logger.info("Whisper auto-detect regex FSMs compiled.")


@warmup("voice_chat")
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


def _prefill_warmup_sizes(configured: str) -> List[int]:
    if configured:
        try:
            sizes = [int(value.strip()) for value in configured.split(",")]
        except ValueError as exc:
            raise ValueError(
                "SGLANG_PREFILL_WARMUP_SIZES must be comma-separated positive integers"
            ) from exc
        if any(size <= 0 for size in sizes):
            raise ValueError("SGLANG_PREFILL_WARMUP_SIZES must contain positive integers")
        # Ascending order creates each requested capacity before larger
        # workspaces can satisfy it through eager-prefill workspace reuse.
        return sorted(set(sizes))
    page_size = 64
    sizes = set()
    base = 64
    while base <= 32768:
        sizes.add(base)
        mid = base * 3 // 2
        mid = (mid + page_size - 1) // page_size * page_size
        if mid <= 32768:
            sizes.add(mid)
        base *= 2
    return sorted(sizes)


def _prefill_warmup_batches(configured: str) -> list[tuple[int, int]]:
    if not configured:
        return []
    try:
        batches = []
        for item in configured.split(","):
            count, tokens = (int(value.strip()) for value in item.split("x"))
            if count <= 0 or tokens <= 0:
                raise ValueError
            batches.append((count, tokens))
    except ValueError as exc:
        raise ValueError(
            "SGLANG_PREFILL_WARMUP_BATCHES must contain positive "
            "sequence-count x tokens-per-sequence pairs, e.g. 2x4096,3x4096"
        ) from exc
    return list(dict.fromkeys(batches))


@warmup("prefill_shapes")
async def prefill_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warm prefill shapes before serving, including lazy communication buffers.

    SGLANG_PREFILL_WARMUP_SIZES optionally selects exact token counts. Unset
    retains the power-of-two and intermediate sweep through 32K. Actual local
    shapes still depend on TP/EP, chunking, and admission, so callers should
    inspect activation records before claiming shape coverage.

    SGLANG_PREFILL_WARMUP_BATCHES optionally adds multi-sequence requests after
    the single-sequence sweep (e.g. 2x4096,3x4096). Equal total token counts do
    not necessarily warm the same metadata kernels. This option is restricted
    to standalone serving and does not guarantee the scheduler's batch shape.
    Each batch also takes one decode step to warm its KV allocation kernels.
    """
    sizes = _prefill_warmup_sizes(envs.SGLANG_PREFILL_WARMUP_SIZES.get())
    batches = _prefill_warmup_batches(envs.SGLANG_PREFILL_WARMUP_BATCHES.get())
    if batches and disaggregation_mode != "null":
        raise ValueError("Multi-sequence prefill warmup requires standalone serving")
    logger.info("Prefill startup warmup token counts: %s", sizes)

    for size in tqdm.tqdm(sizes, desc="Warmup prefill shapes"):
        input_ids = (np.random.randint(2**16, size=[size])).tolist()
        input_ids = replay_warmup_input(size, input_ids)
        record_warmup_input(size, input_ids)
        generate_req_input = GenerateReqInput(
            input_ids=input_ids,
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        # Complete the request and surface errors after its first yield before
        # advancing to another shape or marking server startup complete.
        async for _ in tokenizer_manager.generate_request(generate_req_input, None):
            pass

    if batches:
        logger.info("Prefill startup warmup batches (sequences, tokens each): %s", batches)
    for count, tokens in tqdm.tqdm(
        batches, desc="Warmup prefill batches", disable=not batches
    ):
        generate_req_input = GenerateReqInput(
            input_ids=np.random.randint(2**16, size=(count, tokens)).tolist(),
            # One token is produced by prefill. A second token exercises actual
            # decode allocation, which graph capture alone need not execute.
            sampling_params={
                "max_new_tokens": 2,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        )
        async for _ in tokenizer_manager.generate_request(generate_req_input, None):
            pass
