// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * The language to mark user content with (a question, an answer, a passage,
 * an entity's name or description) when its script differs from the page's:
 * the page's line breaking follows its language (Korean keeps words whole,
 * Japanese and Chinese break strictly), so Korean text in an English page,
 * or Japanese text in a Korean one, needs its own `lang` to break right.
 *
 * The language is the dominant script's, weighed in words (as the backend's
 * `ScriptCounts` weighs letters by the tokens they carry, not one for one: a
 * Hangul syllable carries what four Latin letters do). A mostly English
 * answer that quotes one Korean name stays the page's, since a screen reader
 * picks its voice by `lang` and would read the whole answer in a Korean one;
 * a Korean or Japanese sentence that quotes English names is marked. A word
 * written onto a CJK one with nothing between (a Korean or Japanese particle
 * on a Latin name: "Labs가", "Sensorsは") takes part in the CJK sentence and
 * counts with it. On a tie the CJK script wins, as in the backend: the Latin
 * words are the names it quotes. Undefined when the content reads as the
 * page's language or its CJK words are fewer than its other words.
 */
const HANGUL = /[ᄀ-ᇿ㄰-㆏가-힯]/gu;
const KANA = /[぀-ヿ]/gu;
const HAN = /[㐀-鿿]/gu;
const CJK = /[ᄀ-ᇿ㄰-㆏가-힯぀-ヿ㐀-鿿]/u;
const LETTER = /\p{L}/u;

const words = new Intl.Segmenter(undefined, { granularity: "word" });

/** How many words of `text` are CJK and how many are not (words without
 * letters, such as numbers, count as neither). Words written together with
 * nothing between them form one group, which is CJK when any of its words
 * is. */
function wordCounts(text: string): { cjk: number; other: number } {
  let cjk = 0;
  let other = 0;
  let group = 0;
  let groupCjk = false;
  let end = -1;
  const close = () => {
    if (groupCjk) cjk += group;
    else other += group;
    group = 0;
    groupCjk = false;
  };
  for (const { segment, index, isWordLike } of words.segment(text)) {
    if (!isWordLike || !LETTER.test(segment)) continue;
    if (index !== end) close();
    group += 1;
    groupCjk ||= CJK.test(segment);
    end = index + segment.length;
  }
  close();
  return { cjk, other };
}

export function contentLang(text: string | null | undefined, page: string): string | undefined {
  if (!text || !CJK.test(text)) return undefined;
  const { cjk, other } = wordCounts(text);
  if (cjk < other) return undefined;
  const count = (pattern: RegExp) => text.match(pattern)?.length ?? 0;
  const [hangul, kana, han] = [count(HANGUL), count(KANA), count(HAN)];
  const pageLang = page.toLowerCase().split("-")[0];
  // Kana is Japanese only; Hangul Korean (Korean text may quote Hanja);
  // Han alone is read as Chinese, except on a Japanese page, where it may
  // be a Japanese name.
  const lang = kana && kana >= hangul ? "ja" : hangul ? "ko" : han && pageLang !== "ja" ? "zh" : undefined;
  return lang === pageLang ? undefined : lang;
}
