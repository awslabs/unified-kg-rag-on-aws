// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * The citation markers of an answer, located for display. The grammar is
 * the server's (`unified_kg_rag.domain.retrieval.citations`): square,
 * full-width or lenticular brackets around one or more numbers (ASCII or
 * full-width digits) separated by commas, semicolons or `、`, or a short
 * range. Which numbers count is the server's word: the page links only the
 * markers `RAGOutput.citations` holds.
 */

// The longest range a marker may span ([1-20]), as on the server.
export const MAX_RANGE = 20;

const NUMBER = "[0-9０-９]{1,3}";
const ITEM = `${NUMBER}(?:\\s*[-–~～－]\\s*${NUMBER})?`;
const MARKER = new RegExp(`[\\[［【]\\s*${ITEM}(?:\\s*[,，;；、]\\s*${ITEM})*\\s*[\\]］】]`, "g");
const ITEMS = new RegExp(`(${NUMBER})(?:\\s*[-–~～－]\\s*(${NUMBER}))?`, "g");

export type AnswerSegment = { text: string } | { marker: string; numbers: number[] };

function toNumber(digits: string): number {
  return Number(digits.replace(/[０-９]/g, (d) => String(d.charCodeAt(0) - 0xff10)));
}

/** The numbers one marker cites, in order (a reversed or long range is its
 * two ends). */
export function markerNumbers(marker: string): number[] {
  const numbers: number[] = [];
  for (const match of marker.matchAll(ITEMS)) {
    const first = toNumber(match[1] ?? "");
    const last = match[2] ? toNumber(match[2]) : first;
    if (last < first || last - first >= MAX_RANGE) {
      numbers.push(first);
      if (last !== first) numbers.push(last);
      continue;
    }
    for (let n = first; n <= last; n += 1) numbers.push(n);
  }
  return numbers;
}

/** `text` cut into plain runs and markers. A marker still open at the end
 * of a streaming answer stays text until it closes. */
export function answerSegments(text: string): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  let last = 0;
  for (const match of text.matchAll(MARKER)) {
    const start = match.index ?? 0;
    if (start > last) segments.push({ text: text.slice(last, start) });
    segments.push({ marker: match[0], numbers: markerNumbers(match[0]) });
    last = start + match[0].length;
  }
  if (last < text.length) segments.push({ text: text.slice(last) });
  return segments;
}
