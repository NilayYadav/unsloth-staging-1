# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Images an MCP tool returned, routed into the next model turn. _flatten_result
# appends them to the tool result as a frontend-only envelope; the tool text the
# model reads never carries base64, so the picture reaches a vision model as its
# own user turn instead.

from __future__ import annotations

import base64
import binascii
import io
import json
from typing import Any, Sequence

from loggers import get_logger

logger = get_logger(__name__)

SENTINEL = "__MCP_IMAGES__:"
IMAGE_TURN_TEXT = "Images returned by the tool call above:"

# A tool that answers with a contact sheet should not spend the whole window on
# it, and a local vision model pays for every tile at full resolution.
MAX_MODEL_IMAGES = 4
MAX_IMAGE_EDGE = 1024


def split_images(result: str) -> tuple[str, list[dict]]:
    """Split a tool result into the text the model reads and its image envelope.

    Validated rather than split on sight, so legit tool text that merely mentions
    the marker is not truncated.
    """
    head, sep, payload = result.rpartition("\n" + SENTINEL)
    if not sep:
        return result, []
    try:
        images = json.loads(payload)
    except (ValueError, RecursionError):
        return result, []
    if not isinstance(images, list) or not images:
        return result, []
    if not all(_is_image(image) for image in images):
        return result, []
    return head.rstrip(), images


def _is_image(image: Any) -> bool:
    return (
        isinstance(image, dict)
        and isinstance(image.get("data"), str)
        and isinstance(image.get("mimeType"), str)
    )


def has_images(result: str) -> bool:
    return bool(split_images(result)[1])


def content_parts(images: Sequence[dict]) -> list[dict]:
    """The images as OpenAI ``image_url`` parts, dropping any that will not decode."""
    parts = []
    for image in images[:MAX_MODEL_IMAGES]:
        url = _png_data_url(image.get("data", ""))
        if url:
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _png_data_url(data: str) -> str | None:
    # PNG regardless of what the server sent: llama-server's stb_image reads only
    # a few formats, and MCP servers commonly answer with WebP.
    try:
        raw = base64.b64decode(data, validate = True)
    except (binascii.Error, ValueError, TypeError):
        logger.debug("MCP image payload is not base64")
        return None
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(raw))
        image.load()
        if max(image.size) > MAX_IMAGE_EDGE:
            image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format = "PNG")
    except Exception:
        logger.debug("MCP image could not be decoded", exc_info = True)
        return None
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def append_image_turn(conversation: list, images: Sequence[dict]) -> None:
    """Append a batch's images as one user turn.

    A user turn, not the ``role=tool`` result they came with: OpenAI-shaped tool
    messages take no image parts, and local templates render tool content as a
    string. Appended after the whole batch so an image never splits an
    assistant's ``tool_calls`` from their results, and merged into a trailing
    user turn so the roles keep alternating.
    """
    parts = content_parts(images)
    if not parts:
        return
    last = conversation[-1] if conversation else None
    if (
        isinstance(last, dict)
        and last.get("role") == "user"
        and isinstance(last.get("content"), str)
    ):
        conversation[-1] = {**last, "content": [{"type": "text", "text": last["content"]}, *parts]}
        return
    conversation.append(
        {"role": "user", "content": [{"type": "text", "text": IMAGE_TURN_TEXT}, *parts]}
    )


def promote_history(messages: Sequence[dict], *, vision: bool) -> list[dict]:
    """Turn replayed envelopes back into image turns, so a follow-up question
    about an earlier tool's picture still has the picture.

    The envelope always leaves the tool text, vision or not: a model that cannot
    see the image must not be shown a megabyte of base64 either.
    """
    out: list[dict] = []
    pending: list[dict] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            text, images = split_images(content)
            pending.extend(images)
            out.append({**message, "content": text or "[image returned]"} if images else message)
            continue
        if pending and vision and message.get("role") == "user":
            # Merged rather than inserted ahead of it: a turn cancelled mid-tool
            # leaves the results as the newest history, and two user turns in a
            # row is what a strict template rejects.
            out.append(_with_image_parts(message, pending))
            pending.clear()
            continue
        _flush(out, pending, vision)
        out.append(message)
    _flush(out, pending, vision)
    return out


def _with_image_parts(message: dict, images: Sequence[dict]) -> dict:
    parts = content_parts(images)
    if not parts:
        return message
    content = message.get("content")
    own = list(content) if isinstance(content, list) else [{"type": "text", "text": content or ""}]
    return {**message, "content": [*parts, *own]}


def _flush(conversation: list, pending: list[dict], vision: bool) -> None:
    if pending and vision:
        append_image_turn(conversation, pending)
    pending.clear()
