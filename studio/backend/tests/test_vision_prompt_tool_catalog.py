# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A picture must not cost the prompt its tool catalog or its conversation.

A vision turn renders through ``render_prompt_with_boundary`` rather than the text path's
``apply_chat_template_for_generation``, so a tool loop carrying an image is prompted by
this function. Without the catalog the model is asked to call tools it was never shown,
while the loop keeps parsing its output for calls.
"""

import os
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from core.inference.chat_template_helpers import render_prompt_with_boundary

try:
    # core.inference.inference imports unsloth at module scope, which requires
    # unsloth_zoo. The dependency-light backend CI job does not install it, so the
    # shape checks run only when the stack is importable.
    from core.inference.inference import InferenceBackend
except ImportError:
    InferenceBackend = None


TOOLS = [
    {
        "type": "function",
        "function": {"name": "mcp__fs__read_media_file", "description": "read", "parameters": {}},
    }
]

IMAGE_TURN = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}]


class _Processor:
    """Records the kwargs each render was given; optionally predates ``tools=``."""

    def __init__(self, rejects_tools = False):
        self.calls: list[dict] = []
        self.rendered: list[list] = []
        self._rejects_tools = rejects_tools

    def apply_chat_template(self, messages, **kwargs):
        if self._rejects_tools and "tools" in kwargs:
            raise TypeError("apply_chat_template() got an unexpected keyword argument 'tools'")
        self.calls.append(kwargs)
        self.rendered.append(messages)
        return "PROMPT"


def test_the_tool_catalog_reaches_the_processor_template():
    processor = _Processor()

    render_prompt_with_boundary(processor, IMAGE_TURN, tools = TOOLS)

    assert processor.calls[-1]["tools"], "a vision turn rendered without its tool catalog"


def test_a_render_without_tools_is_left_alone():
    processor = _Processor()

    render_prompt_with_boundary(processor, IMAGE_TURN)

    assert "tools" not in processor.calls[-1]
    assert processor.calls[-1]["add_generation_prompt"] is True


def test_a_continued_turn_keeps_its_boundary_when_tools_render():
    processor = _Processor()
    messages = IMAGE_TURN + [{"role": "assistant", "content": [{"type": "text", "text": "par"}]}]

    render_prompt_with_boundary(
        processor, messages, continue_final_message = True, tools = TOOLS
    )

    assert processor.calls[-1]["continue_final_message"] is True
    assert processor.calls[-1]["add_generation_prompt"] is False
    assert processor.calls[-1]["tools"]


def test_a_processor_predating_the_tools_kwarg_still_renders():
    processor = _Processor(rejects_tools = True)

    assert render_prompt_with_boundary(processor, IMAGE_TURN, tools = TOOLS) == "PROMPT"
    assert "tools" not in processor.calls[-1]


# ── Which shape the pixels are rendered against ───────────────────


class _Stopped(Exception):
    def __init__(self, pixels):
        self.pixels = pixels


class _VisionProcessor(_Processor):
    """Raises out of the tensor build, so the test reads the shape that was chosen
    without needing a real model."""

    def __call__(self, pixels, text, **kwargs):
        raise _Stopped(pixels)


def _backend_with(processor):
    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = "m"
    backend.models = {"m": {"model": object(), "processor": processor, "is_vision": True}}
    backend.last_generation_stats = None
    return backend


def _chosen(messages, image = None, images = None, tools = None):
    processor = _VisionProcessor()
    stream = _backend_with(processor)._generate_vision_response(
        messages, "", image, 0.7, 0.9, 40, 0.0, 16, 1.0, images = images, tools = tools
    )
    try:
        next(stream)
    except _Stopped as stopped:
        pixels = stopped.pixels
        return (pixels if isinstance(pixels, list) else [pixels]), processor
    raise AssertionError("the processor was never handed any pixels")


@pytest.mark.skipif(InferenceBackend is None, reason = "unsloth stack not installed")
def test_a_bare_attachment_renders_the_single_turn_shape():
    pixels, processor = _chosen([{"role": "user", "content": "what is this"}], image = "ATTACHED")

    assert pixels == ["ATTACHED"]
    assert len(processor.rendered[-1]) == 1


@pytest.mark.skipif(InferenceBackend is None, reason = "unsloth stack not installed")
def test_tool_returned_pictures_keep_the_conversation_they_came_from():
    history = [
        {"role": "user", "content": "what is in the file"},
        {"role": "tool", "content": "[1 image returned]"},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "describe it"}]},
    ]

    pixels, processor = _chosen(history, images = ["MCP"], tools = TOOLS)

    assert pixels == ["MCP"]
    assert [message["role"] for message in processor.rendered[-1]] == ["user", "tool", "user"]
    assert processor.calls[-1]["tools"], "the catalog was dropped by the chosen shape"


@pytest.mark.skipif(InferenceBackend is None, reason = "unsloth stack not installed")
def test_an_attachment_alongside_tool_pictures_keeps_the_conversation():
    history = [
        {"role": "user", "content": "what is in the file"},
        {"role": "tool", "content": "[1 image returned]"},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "describe it"}]},
        {"role": "assistant", "content": "a blue square"},
        {"role": "user", "content": "compare it with mine"},
    ]

    pixels, processor = _chosen(history, image = "ATTACHED", images = ["MCP"])

    # Document order: the tool's picture is marked in an earlier turn, the attachment on
    # the newest one, so the pixels have to arrive in that order too.
    assert pixels == ["MCP", "ATTACHED"]
    assert [message["role"] for message in processor.rendered[-1]] == [
        "user",
        "tool",
        "user",
        "assistant",
        "user",
    ]
