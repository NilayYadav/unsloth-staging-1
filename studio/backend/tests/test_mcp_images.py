# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import mcp_images
from core.inference.mcp_images import (
    IMAGE_TURN_TEXT,
    MAX_IMAGE_EDGE,
    MAX_MODEL_IMAGES,
    append_image_turn,
    content_parts,
    promote_history,
    split_images,
)


def _png(size = (8, 8), fmt = "PNG") -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(buffer, format = fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image(data = None, mime = "image/png") -> dict:
    return {"data": data if data is not None else _png(), "mimeType": mime}


def _envelope(text: str, *images: dict) -> str:
    return text + "\n" + mcp_images.SENTINEL + json.dumps(list(images))


def _decode(part: dict):
    from PIL import Image

    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_split_returns_text_and_images():
    text, images = split_images(_envelope("a screenshot", _image()))

    assert text == "a screenshot"
    assert len(images) == 1


def test_split_leaves_a_result_that_only_mentions_the_marker():
    result = "docs say the marker is\n__MCP_IMAGES__: and nothing follows"

    assert split_images(result) == (result, [])


def test_split_leaves_an_envelope_that_is_not_an_image_array():
    result = 'log\n__MCP_IMAGES__:["not", "image", "dicts"]'

    assert split_images(result) == (result, [])


def test_content_parts_reencode_to_png():
    parts = content_parts([_image(data = _png(fmt = "WEBP"), mime = "image/webp")])

    assert len(parts) == 1
    assert _decode(parts[0]).format == "PNG"


def test_content_parts_downscale_a_large_image():
    parts = content_parts([_image(data = _png(size = (MAX_IMAGE_EDGE * 2, MAX_IMAGE_EDGE)))])

    assert max(_decode(parts[0]).size) == MAX_IMAGE_EDGE


def test_content_parts_cap_how_many_reach_the_model():
    parts = content_parts([_image() for _ in range(MAX_MODEL_IMAGES + 3)])

    assert len(parts) == MAX_MODEL_IMAGES


def test_content_parts_drop_an_undecodable_payload():
    assert content_parts([_image(data = "not base64 at all")]) == []
    assert content_parts([_image(data = base64.b64encode(b"nope").decode())]) == []


def test_image_turn_is_its_own_user_message():
    conversation = [{"role": "tool", "name": "mcp__fs__read", "content": "[1 image returned]"}]

    append_image_turn(conversation, [_image()])

    assert conversation[-1]["role"] == "user"
    assert conversation[-1]["content"][0] == {"type": "text", "text": IMAGE_TURN_TEXT}
    assert conversation[-1]["content"][1]["type"] == "image_url"
    assert conversation[1] is conversation[-1]


def test_image_turn_merges_into_a_trailing_user_turn():
    conversation = [{"role": "user", "content": "a nudge"}]

    append_image_turn(conversation, [_image()])

    assert len(conversation) == 1
    assert conversation[0]["content"][0] == {"type": "text", "text": "a nudge"}
    assert conversation[0]["content"][1]["type"] == "image_url"


def test_image_turn_is_skipped_when_nothing_decodes():
    conversation = [{"role": "tool", "content": "[1 image returned]"}]

    append_image_turn(conversation, [_image(data = "///")])

    assert len(conversation) == 1


def test_history_promotes_a_replayed_envelope():
    messages = promote_history(
        [
            {"role": "user", "content": "what is in the file"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0"}]},
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": _envelope("[1 image returned]", _image()),
            },
            {"role": "assistant", "content": "a blue square"},
        ],
        vision = True,
    )

    assert messages[2]["content"] == "[1 image returned]"
    assert messages[3]["role"] == "user"
    assert messages[3]["content"][1]["type"] == "image_url"
    assert messages[4]["content"] == "a blue square"


def test_history_flushes_after_the_whole_batch_of_tool_results():
    messages = promote_history(
        [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "tool_call_id": "a", "content": _envelope("first", _image())},
            {"role": "tool", "tool_call_id": "b", "content": _envelope("second", _image())},
        ],
        vision = True,
    )

    assert [message["role"] for message in messages] == ["assistant", "tool", "tool", "user"]
    assert len(messages[3]["content"]) == 3


def test_history_strips_the_envelope_for_a_model_that_cannot_see_it():
    messages = promote_history(
        [{"role": "tool", "content": _envelope("[1 image returned]", _image())}],
        vision = False,
    )

    assert messages == [{"role": "tool", "content": "[1 image returned]"}]


def test_history_keeps_an_image_only_result_from_emptying_its_tool_message():
    messages = promote_history(
        [{"role": "tool", "content": _envelope("", _image())}],
        vision = False,
    )

    assert messages[0]["content"] == "[image returned]"


def test_history_leaves_other_messages_untouched():
    original = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "plain result"},
    ]

    assert promote_history(original, vision = True) == original


def test_history_merges_into_a_following_user_turn():
    messages = promote_history(
        [
            {"role": "tool", "content": _envelope("[1 image returned]", _image())},
            {"role": "user", "content": "what colour was it"},
        ],
        vision = True,
    )

    assert [message["role"] for message in messages] == ["tool", "user"]
    assert messages[1]["content"][0]["type"] == "image_url"
    assert messages[1]["content"][1] == {"type": "text", "text": "what colour was it"}


def test_local_history_carries_markers_and_payloads():
    messages, payloads = mcp_images.promote_history_local(
        [
            {"role": "user", "content": "what is in the file"},
            {"role": "tool", "content": _envelope("[1 image returned]", _image())},
            {"role": "assistant", "content": "a blue square"},
        ],
        vision = True,
    )

    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0] == {"type": "image"}
    assert len(payloads) == 1
    assert base64.b64decode(payloads[0])[:8] == b"\x89PNG\r\n\x1a\n"


def test_local_history_merges_markers_into_a_following_user_turn():
    messages, payloads = mcp_images.promote_history_local(
        [
            {"role": "tool", "content": _envelope("[1 image returned]", _image())},
            {"role": "user", "content": "what colour was it"},
        ],
        vision = True,
    )

    assert [message["role"] for message in messages] == ["tool", "user"]
    assert messages[1]["content"][0] == {"type": "image"}
    assert len(payloads) == 1


def test_local_history_strips_without_vision():
    messages, payloads = mcp_images.promote_history_local(
        [{"role": "tool", "content": _envelope("[1 image returned]", _image())}],
        vision = False,
    )

    assert messages == [{"role": "tool", "content": "[1 image returned]"}]
    assert payloads == []


def test_placeholder_turn_marks_one_image_per_payload():
    turn = mcp_images.placeholder_turn(2)

    assert [part["type"] for part in turn["content"]] == ["image", "image", "text"]


def test_a_turn_says_when_it_carries_fewer_than_the_tool_returned():
    returned = MAX_MODEL_IMAGES + 2
    conversation = [{"role": "tool", "content": f"[{returned} images returned]"}]

    append_image_turn(conversation, [_image() for _ in range(returned)])

    text = conversation[-1]["content"][0]["text"]
    assert text.startswith(IMAGE_TURN_TEXT)
    assert f"first {MAX_MODEL_IMAGES} of {returned}" in text


def test_a_turn_that_carries_them_all_says_nothing_extra():
    conversation = [{"role": "tool", "content": "[1 image returned]"}]

    append_image_turn(conversation, [_image()])

    assert conversation[-1]["content"][0]["text"] == IMAGE_TURN_TEXT


def test_a_placeholder_turn_reports_the_pictures_it_had_to_drop():
    turn = mcp_images.placeholder_turn(MAX_MODEL_IMAGES, MAX_MODEL_IMAGES + 3)

    assert f"first {MAX_MODEL_IMAGES} of {MAX_MODEL_IMAGES + 3}" in turn["content"][-1]["text"]


def _marker_count(conversation) -> int:
    return sum(
        1
        for message in conversation
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image"
    )


def test_a_long_loop_stops_accumulating_pictures():
    conversation = []
    payloads = []
    for _ in range(6):
        conversation.append(mcp_images.placeholder_turn(3))
        payloads.extend(["p"] * 3)
        mcp_images.trim_image_turns(conversation, payloads)

    assert len(payloads) == mcp_images.MAX_TOTAL_MODEL_IMAGES
    assert _marker_count(conversation) == len(payloads)


def test_trimming_keeps_the_newest_pictures():
    conversation = [mcp_images.placeholder_turn(2), mcp_images.placeholder_turn(2)]
    payloads = ["old_a", "old_b", "new_a", "new_b"]

    mcp_images.trim_image_turns(conversation, payloads, limit = 2)

    assert payloads == ["new_a", "new_b"]
    assert _marker_count(conversation) == 2


def test_trimming_drops_a_turn_that_has_no_pictures_left():
    conversation = [mcp_images.placeholder_turn(1), mcp_images.placeholder_turn(1)]
    payloads = ["old", "new"]

    mcp_images.trim_image_turns(conversation, payloads, limit = 1)

    assert len(conversation) == 1
    assert _marker_count(conversation) == 1


def test_trimming_keeps_a_users_own_words_when_it_takes_their_picture():
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "what is this"}],
        },
        mcp_images.placeholder_turn(1),
    ]
    payloads = ["theirs", "tools"]

    mcp_images.trim_image_turns(conversation, payloads, limit = 1)

    assert conversation[0]["content"] == [{"type": "text", "text": "what is this"}]
    assert _marker_count(conversation) == 1


def test_trimming_leaves_a_conversation_under_the_limit_alone():
    conversation = [mcp_images.placeholder_turn(2)]
    payloads = ["a", "b"]

    mcp_images.trim_image_turns(conversation, payloads)

    assert payloads == ["a", "b"]
    assert len(conversation) == 1
