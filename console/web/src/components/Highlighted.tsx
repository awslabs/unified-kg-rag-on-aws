// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { Fragment } from "react";

const DENSE = /[぀-ヿ㐀-鿿가-힯]/;
// A word's runs of one script class: Hangul, Kana or Han, or anything else
// (a mixed word such as "Depot으로" is a Latin term and a Hangul one).
const SCRIPT_RUNS = /[぀-ヿ㐀-鿿가-힯]+|[^぀-ヿ㐀-鿿가-힯]+/gu;
// Words too common to be worth marking (English; other Latin-script
// languages are marked from four letters on).
const COMMON = new Set(
  "the and for with how who what when where which why does did was were are has have had this that from into about than then them they their there these those will would can could should".split(" "),
);

/** The terms of `query` worth marking: words of four or more letters (or
 * with a digit) but the most common ones, and
 * two-character runs of Hangul, Kana or Han (which have no spaces). A word
 * that mixes them is split where its script changes first. */
export function queryTerms(query: string): string[] {
  const terms = new Set<string>();
  for (const word of query.toLowerCase().split(/[^\p{L}\p{N}]+/u)) {
    for (const run of word.match(SCRIPT_RUNS) ?? []) {
      if (DENSE.test(run)) {
        for (let i = 0; i + 2 <= run.length; i += 1) terms.add(run.slice(i, i + 2));
      } else if ((run.length >= 4 || /\d/.test(run)) && !COMMON.has(run)) {
        terms.add(run);
      }
    }
  }
  return [...terms].sort((a, b) => b.length - a.length);
}

function escape(term: string): string {
  return term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** `text` split into runs, the query's terms marked; terms separated only
 * by spaces ("Kestrel Depot") join into one run, so a phrase reads as one
 * highlight instead of two abutting ones. */
export function highlightRuns(text: string, query: string): { text: string; marked: boolean }[] {
  const terms = queryTerms(query);
  if (!terms.length) return [{ text, marked: false }];
  // Every term is escaped, so the pattern is an alternation of literals and
  // cannot backtrack catastrophically.
  // nosemgrep: javascript.lang.security.audit.detect-non-literal-regexp.detect-non-literal-regexp
  const pattern = new RegExp(`(${terms.map(escape).join("|")})`, "giu");
  const runs: { text: string; marked: boolean }[] = [];
  const parts = text.split(pattern);
  parts.forEach((part, index) => {
    const marked = index % 2 === 1;
    const last = runs[runs.length - 1];
    if (marked && last?.marked) {
      last.text += part;
    } else if (!marked && /^[ \t]+$/.test(part) && last?.marked && index + 1 < parts.length && parts[index + 1]) {
      last.text += part;
    } else if (part) {
      runs.push({ text: part, marked });
    }
  });
  return runs;
}

/** `text` with the query's terms in the highlighter. */
export function Highlighted({ text, query }: { text: string; query: string }) {
  return (
    <>
      {highlightRuns(text, query).map((run, index) =>
        run.marked ? <mark key={index}>{run.text}</mark> : <Fragment key={index}>{run.text}</Fragment>,
      )}
    </>
  );
}
