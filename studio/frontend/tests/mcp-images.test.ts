// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  MCP_IMAGES_MARKER,
  mcpImagesEnvelope,
  splitMcpImages,
} from "../src/features/chat/api/mcp-images.ts";

const IMAGES = [{ data: "QUJD", mimeType: "image/png" }];
const RESULT = `[1 image returned]${mcpImagesEnvelope(IMAGES)}`;

const adapter = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);

test("a valid envelope splits into the text and its images", () => {
  assert.deepEqual(splitMcpImages(RESULT), {
    text: "[1 image returned]",
    images: IMAGES,
  });
});

test("text that only mentions the marker is left whole", () => {
  const result = `the marker is${MCP_IMAGES_MARKER} and nothing follows`;
  assert.deepEqual(splitMcpImages(result), { text: result, images: [] });
});

test("an envelope that is not an image array is left whole", () => {
  const result = `log${MCP_IMAGES_MARKER}["not", "image", "dicts"]`;
  assert.deepEqual(splitMcpImages(result), { text: result, images: [] });
});

test("an unparseable envelope is left whole", () => {
  const result = `log${MCP_IMAGES_MARKER}{oops`;
  assert.deepEqual(splitMcpImages(result), { text: result, images: [] });
});

test("replaying a tool result re-attaches its images for the backend", () => {
  assert.match(adapter, /content \+= mcpImagesEnvelope\(result\.images\);/);
});
