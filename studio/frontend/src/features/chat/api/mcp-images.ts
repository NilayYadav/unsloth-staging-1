// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The envelope the backend appends to a tool result that returned images. It is
// validated rather than split on sight, so legit tool text that merely mentions
// the marker is never truncated.

export const MCP_IMAGES_MARKER = "\n__MCP_IMAGES__:";

export interface McpImage {
  data: string;
  mimeType: string;
}

export function isMcpImageArray(value: unknown): value is McpImage[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (image) =>
        typeof image === "object" &&
        image !== null &&
        typeof (image as Record<string, unknown>).data === "string" &&
        typeof (image as Record<string, unknown>).mimeType === "string",
    )
  );
}

export function splitMcpImages(result: string): {
  text: string;
  images: McpImage[];
} {
  const idx = result.lastIndexOf(MCP_IMAGES_MARKER);
  if (idx === -1) return { text: result, images: [] };
  let images: unknown;
  try {
    images = JSON.parse(result.slice(idx + MCP_IMAGES_MARKER.length));
  } catch {
    return { text: result, images: [] };
  }
  if (!isMcpImageArray(images)) return { text: result, images: [] };
  return { text: result.slice(0, idx), images };
}

// Re-attached on replay so a follow-up question about the picture still has it:
// the backend promotes this into an image turn for a vision model, and strips it
// for every other one.
export function mcpImagesEnvelope(images: McpImage[]): string {
  return MCP_IMAGES_MARKER + JSON.stringify(images);
}
