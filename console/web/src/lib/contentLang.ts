// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * The language to mark user content with (a question, an answer, a passage,
 * an entity's name or description) when its script differs from the page's:
 * the page's line breaking follows its language (Korean keeps words whole,
 * Japanese and Chinese break strictly), so Korean text in an English page,
 * or Japanese text in a Korean one, needs its own `lang` to break right.
 * Undefined when the content reads as the page's language or no CJK script.
 */
const HANGUL = /[ᄀ-ᇿ㄰-㆏가-힯]/gu;
const KANA = /[぀-ヿ]/gu;
const HAN = /[㐀-鿿]/gu;

export function contentLang(text: string | null | undefined, page: string): string | undefined {
  if (!text) return undefined;
  const count = (pattern: RegExp) => text.match(pattern)?.length ?? 0;
  const [hangul, kana, han] = [count(HANGUL), count(KANA), count(HAN)];
  const pageLang = page.toLowerCase().split("-")[0];
  // Kana is Japanese only; Hangul Korean (Korean text may quote Hanja);
  // Han alone is read as Chinese, except on a Japanese page, where it may
  // be a Japanese name.
  const lang = kana && kana >= hangul ? "ja" : hangul ? "ko" : han && pageLang !== "ja" ? "zh" : undefined;
  return lang === pageLang ? undefined : lang;
}
