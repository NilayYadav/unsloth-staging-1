# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

MAX_MODEL_IMAGES = 4
MAX_IMAGE_EDGE = 1024


def split_images(result: str) -> tuple[str, list[dict]]:
    """Validated, so tool text that merely mentions the marker is not truncated."""
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


def png_payloads(images: Sequence[dict]) -> list[str]:
    """Normalized PNG base64, for backends that take images as objects rather
    than as data URLs inside the prompt."""
    out = []
    for image in images[:MAX_MODEL_IMAGES]:
        url = _png_data_url(image.get("data", ""))
        if url:
            out.append(url.split(",", 1)[1])
    return out


def placeholder_turn(count: int) -> dict:
    """The user turn a local processor renders: ``{"type": "image"}`` markers the
    template turns into image tokens, with the pixels passed alongside."""
    return {
        "role": "user",
        "content": [
            *({"type": "image"} for _ in range(count)),
            {"type": "text", "text": IMAGE_TURN_TEXT},
        ],
    }


def append_image_turn(conversation: list, images: Sequence[dict]) -> None:
    """A user turn, not the ``role=tool`` result they came with: tool messages take
    no image parts, and local templates render tool content as a string."""
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


def mark_last_user_turn(messages: Sequence[dict], count: int) -> list[dict]:
    """Mark the newest user turn as carrying ``count`` images, where an
    attachment belongs."""
    out = list(messages)
    for index in range(len(out) - 1, -1, -1):
        if out[index].get("role") == "user":
            out[index] = _with_parts(out[index], [{"type": "image"} for _ in range(count)])
            break
    return out


def promote_history(messages: Sequence[dict], *, vision: bool) -> list[dict]:
    """Rebuild image turns from replayed envelopes. The envelope leaves the tool
    text either way: a text-only model must not be shown its base64."""
    return _promote(messages, vision, local = False)[0]


def promote_history_local(
    messages: Sequence[dict], *, vision: bool
) -> tuple[list[dict], list[str]]:
    """The same, for backends that take the pixels beside the prompt: the turns
    carry markers and the payloads come back with them."""
    return _promote(messages, vision, local = True)


def _promote(messages, vision: bool, *, local: bool) -> tuple[list[dict], list[str]]:
    out: list[dict] = []
    pending: list[dict] = []
    payloads: list[str] = []

    def flush(into: "dict | None" = None) -> "dict | None":
        if not pending or not vision:
            pending.clear()
            return into
        if local:
            encoded = png_payloads(pending)
            pending.clear()
            if not encoded:
                return into
            payloads.extend(encoded)
            markers = [{"type": "image"} for _ in encoded]
            if into is None:
                out.append(placeholder_turn(len(encoded)))
                return None
            return _with_parts(into, markers)
        images = list(pending)
        pending.clear()
        if into is None:
            append_image_turn(out, images)
            return None
        parts = content_parts(images)
        return _with_parts(into, parts) if parts else into

    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            text, images = split_images(content)
            pending.extend(images)
            out.append({**message, "content": text or "[image returned]"} if images else message)
            continue
        if pending and vision and message.get("role") == "user":
            # Merged, not inserted ahead of it: two user turns in a row is what
            # a strict template rejects.
            out.append(flush(message))
            continue
        flush()
        out.append(message)
    flush()
    return out, payloads


def _with_parts(message: dict, parts: list[dict]) -> dict:
    content = message.get("content")
    own = list(content) if isinstance(content, list) else [{"type": "text", "text": content or ""}]
    return {**message, "content": [*parts, *own]}
