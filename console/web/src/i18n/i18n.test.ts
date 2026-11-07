// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";
import { initialLocale, interpolate, LOCALES, makeI18n, translate, type Locale } from "./i18n";

const PLURAL = /_(zero|one|two|few|many|other)$/;
const en = LOCALES.en.catalog;

function base(key: string): string {
  return key.replace(PLURAL, "");
}
function placeholders(text: string): string[] {
  return [...text.matchAll(/\{\{\s*(\w+)\s*\}\}/g)].map((m) => m[1] ?? "").sort();
}
const pluralBases = new Set(Object.keys(en).filter((k) => PLURAL.test(k)).map(base));

describe.each(Object.keys(LOCALES) as Locale[])("the %s catalog", (locale) => {
  const catalog = LOCALES[locale].catalog;
  const categories = new Intl.PluralRules(LOCALES[locale].tag).resolvedOptions().pluralCategories;

  it("has every key of the English catalog and no other", () => {
    const keys = new Set(Object.keys(catalog).map(base));
    const expected = new Set(Object.keys(en).map(base));
    expect([...expected].filter((k) => !keys.has(k))).toEqual([]);
    expect([...keys].filter((k) => !expected.has(k))).toEqual([]);
  });

  it("has exactly the plural forms of its language", () => {
    for (const key of pluralBases) {
      const forms = Object.keys(catalog)
        .filter((k) => base(k) === key && PLURAL.test(k))
        .map((k) => k.match(PLURAL)?.[1])
        .sort();
      expect(forms, key).toEqual([...categories].sort());
      expect(catalog[key], key).toBeUndefined();
    }
  });

  it("keeps every placeholder of the English message", () => {
    for (const [key, text] of Object.entries(catalog)) {
      const source = en[key] ?? en[`${base(key)}_other`];
      expect(source, key).toBeDefined();
      // A plural form may leave the count out ("The gate passed").
      const plural = PLURAL.test(key);
      const strip = (names: string[]) => (plural ? names.filter((n) => n !== "count") : names);
      expect(strip(placeholders(text)), key).toEqual(strip(placeholders(source as string)));
      expect(text.trim(), key).not.toBe("");
    }
  });
});

describe("translate", () => {
  it("picks the plural form of the count and formats it", () => {
    expect(translate("en", "ask.trace.read", { count: 1 })).toBe("The answer read 1 source");
    expect(translate("en", "ask.trace.read", { count: 1200 })).toBe("The answer read 1,200 sources");
  });
  it("falls back to English, then to the key", () => {
    expect(translate("ko", "no.such.key")).toBe("no.such.key");
  });
  it("leaves an unknown placeholder visible", () => {
    expect(interpolate("a {{b}} {{c}}", { b: 1 })).toBe("a 1 {{c}}");
  });
});

describe("initialLocale", () => {
  it("prefers the saved choice, then the browser languages", () => {
    expect(initialLocale("ja", ["ko-KR"])).toBe("ja");
    expect(initialLocale(null, ["pt-BR", "en"])).toBe("pt");
    expect(initialLocale("xx", ["sv-SE"])).toBe("en");
  });
});

describe("formats", () => {
  it("follows the locale", () => {
    const de = makeI18n("de", () => undefined);
    expect(de.number(1234.5)).toBe("1.234,5");
    expect(de.usd(0.0042)).toContain("0,0042");
    expect(makeI18n("en", () => undefined).percent(0.25)).toBe("25%");
    expect(makeI18n("ko", () => undefined).number(null)).toBe("—");
  });
});
