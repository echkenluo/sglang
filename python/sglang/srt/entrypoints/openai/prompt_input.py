from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.protocol import MessageProcessingResult


def request_carries_media(processed_messages: MessageProcessingResult) -> bool:
    """Return whether this request actually carries a multimodal payload."""
    return bool(
        processed_messages.image_data
        or processed_messages.video_data
        or processed_messages.audio_data
    )


def has_usable_prompt_ids(processed_messages: MessageProcessingResult) -> bool:
    """Return whether message processing produced token ids we can forward."""
    prompt_ids = processed_messages.prompt_ids
    return isinstance(prompt_ids, list) and bool(prompt_ids)


def should_use_text_prompt(
    is_multimodal: bool,
    processed_messages: MessageProcessingResult,
    *,
    prefer_prompt_ids: bool = False,
) -> bool:
    """Choose text only when media processing or missing token ids requires it.

    A multimodal-capable checkpoint does not make every request multimodal.
    Text-only requests with usable token ids should preserve those ids and avoid
    re-tokenizing the full context downstream.
    """
    if prefer_prompt_ids and has_usable_prompt_ids(processed_messages):
        return False
    return (
        is_multimodal and request_carries_media(processed_messages)
    ) or not has_usable_prompt_ids(processed_messages)
