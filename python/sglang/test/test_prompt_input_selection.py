"""Offline contract tests for OpenAI prompt input selection.

These tests load the small decision module directly, so they need neither torch
nor a running model server.
"""

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve()
MODULE = HERE.parents[1] / "srt" / "entrypoints" / "openai" / "prompt_input.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prompt_input_under_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prompt_input = _load_module()


def _processed(prompt_ids, *, image=None, video=None, audio=None):
    return SimpleNamespace(
        prompt_ids=prompt_ids,
        image_data=image,
        video_data=video,
        audio_data=audio,
    )


def test_multimodal_checkpoint_text_request_keeps_prompt_ids():
    processed = _processed([1, 2, 3])
    assert not prompt_input.should_use_text_prompt(True, processed)


def test_real_media_request_keeps_text_path():
    for media_field in ("image", "video", "audio"):
        processed = _processed([1, 2, 3], **{media_field: ["payload"]})
        assert prompt_input.should_use_text_prompt(True, processed)


def test_preencoded_media_request_can_prefer_prompt_ids():
    processed = _processed([1, 2, 3], image=["payload"])
    assert not prompt_input.should_use_text_prompt(
        True, processed, prefer_prompt_ids=True
    )


def test_missing_or_non_list_prompt_ids_keep_text_path():
    for prompt_ids in (None, [], "", "rendered prompt", (1, 2, 3)):
        processed = _processed(prompt_ids)
        assert prompt_input.should_use_text_prompt(True, processed)


def test_text_checkpoint_selection_is_unchanged():
    assert not prompt_input.should_use_text_prompt(False, _processed([1, 2, 3]))
    assert prompt_input.should_use_text_prompt(False, _processed("rendered prompt"))


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print("PASS", name)
            except AssertionError as error:
                failures += 1
                print("FAIL", name, error)
    sys.exit(1 if failures else 0)
