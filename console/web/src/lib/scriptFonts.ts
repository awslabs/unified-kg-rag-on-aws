// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Faces a page language needs that the others do not: Pretendard JP's
 * stylesheet (its 119 @font-face rules) loads only when the page turns
 * Japanese, so no other language pays for it. Elsewhere Japanese text in a
 * document takes the system's Japanese face.
 */
let japanese: Promise<unknown> | null = null;

export function loadScriptFonts(tag: string): void {
  if (tag.toLowerCase().startsWith("ja") && !japanese) {
    japanese = import("pretendard-jp/dist/web/variable/pretendardvariable-jp-dynamic-subset.css");
  }
}
