// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * UI copy: one message catalog per locale (i18next-style keys, `{{name}}`
 * placeholders, plural forms as `key_one` / `key_other` / ... by
 * Intl.PluralRules). English is the source and the fallback: a key missing
 * from a locale shows its English text. Numbers, dates, currency and
 * relative times go through Intl in the chosen locale.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import de from "./locales/de.json";
import en from "./locales/en.json";
import es from "./locales/es.json";
import fr from "./locales/fr.json";
import it from "./locales/it.json";
import ja from "./locales/ja.json";
import ko from "./locales/ko.json";
import pt from "./locales/pt.json";
import ru from "./locales/ru.json";
import zh from "./locales/zh.json";

export type Catalog = Record<string, string>;

/** The UI locales: the languages the library supports (LanguageCode). */
export const LOCALES = {
  en: { name: "English", catalog: en as Catalog, tag: "en" },
  ko: { name: "한국어", catalog: ko as Catalog, tag: "ko" },
  ja: { name: "日本語", catalog: ja as Catalog, tag: "ja" },
  zh: { name: "简体中文", catalog: zh as Catalog, tag: "zh-Hans" },
  de: { name: "Deutsch", catalog: de as Catalog, tag: "de" },
  es: { name: "Español", catalog: es as Catalog, tag: "es" },
  fr: { name: "Français", catalog: fr as Catalog, tag: "fr" },
  it: { name: "Italiano", catalog: it as Catalog, tag: "it" },
  pt: { name: "Português", catalog: pt as Catalog, tag: "pt" },
  ru: { name: "Русский", catalog: ru as Catalog, tag: "ru" },
} as const;

export type Locale = keyof typeof LOCALES;
export const STORAGE_KEY = "ukr-console-locale";

export function isLocale(value: string | null | undefined): value is Locale {
  return !!value && Object.prototype.hasOwnProperty.call(LOCALES, value);
}

/** The locale to start in: the saved choice, else the browser's languages. */
export function initialLocale(saved: string | null, languages: readonly string[]): Locale {
  if (isLocale(saved)) return saved;
  for (const language of languages) {
    const base = language.toLowerCase().split("-")[0];
    if (isLocale(base)) return base;
  }
  return "en";
}

export type Vars = Record<string, string | number>;

/** `{{name}}` replaced by `vars.name`; an unknown name stays visible. */
export function interpolate(text: string, vars?: Vars): string {
  if (!vars) return text;
  return text.replace(/\{\{\s*(\w+)\s*\}\}/g, (match, name: string) =>
    name in vars ? String(vars[name]) : match,
  );
}

/** The message of `key` in `locale`: its plural form when `vars.count` is
 * given, English when the locale lacks it, the key itself as a last resort. */
export function translate(locale: Locale, key: string, vars?: Vars): string {
  const catalogs = [LOCALES[locale].catalog, LOCALES.en.catalog];
  const count = vars?.count;
  for (const [index, catalog] of catalogs.entries()) {
    if (typeof count === "number") {
      const tag = index === 0 ? LOCALES[locale].tag : "en";
      const form = new Intl.PluralRules(tag).select(count);
      const plural = catalog[`${key}_${form}`] ?? catalog[`${key}_other`];
      if (plural !== undefined) return interpolate(plural, { ...vars, count: formatNumber(locale, count) });
    }
    const text = catalog[key];
    if (text !== undefined) return interpolate(text, vars);
  }
  return key;
}

export function formatNumber(locale: Locale, value: number, options?: Intl.NumberFormatOptions): string {
  return new Intl.NumberFormat(LOCALES[locale].tag, options).format(value);
}

export interface I18n {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: string, vars?: Vars) => string;
  number: (value: number | null | undefined, options?: Intl.NumberFormatOptions) => string;
  percent: (value: number | null | undefined, digits?: number) => string;
  usd: (value: number | null | undefined) => string;
  seconds: (value: number | null | undefined) => string;
  date: (epochSeconds: number, options?: Intl.DateTimeFormatOptions) => string;
  ago: (epochSeconds: number) => string;
}

const I18nContext = createContext<I18n | null>(null);

const DASH = "—";

export function makeI18n(locale: Locale, setLocale: (l: Locale) => void): I18n {
  const tag = LOCALES[locale].tag;
  return {
    locale,
    setLocale,
    t: (key, vars) => translate(locale, key, vars),
    number: (value, options) => (value == null || Number.isNaN(value) ? DASH : formatNumber(locale, value, options)),
    percent: (value, digits = 0) =>
      value == null ? DASH : formatNumber(locale, value, { style: "percent", maximumFractionDigits: digits }),
    usd: (value) => {
      if (value == null) return DASH;
      const digits = value !== 0 && Math.abs(value) < 0.01 ? 4 : 2;
      return new Intl.NumberFormat(tag, { style: "currency", currency: "USD", minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
    },
    seconds: (value) =>
      value == null
        ? DASH
        : new Intl.NumberFormat(tag, { style: "unit", unit: "second", unitDisplay: "short", maximumFractionDigits: value < 10 ? 2 : 0 }).format(value),
    date: (epochSeconds, options) =>
      new Intl.DateTimeFormat(tag, options ?? { dateStyle: "medium", timeStyle: "short" }).format(new Date(epochSeconds * 1000)),
    ago: (epochSeconds) => {
      const delta = epochSeconds - Date.now() / 1000;
      const rtf = new Intl.RelativeTimeFormat(tag, { numeric: "auto" });
      const steps: [Intl.RelativeTimeFormatUnit, number][] = [
        ["second", 60],
        ["minute", 60],
        ["hour", 24],
        ["day", 30],
        ["month", 12],
      ];
      let value = delta;
      for (const [unit, size] of steps) {
        if (Math.abs(value) < size) return rtf.format(Math.round(value), unit);
        value /= size;
      }
      return rtf.format(Math.round(value), "year");
    },
  };
}

export function I18nProvider({ children, initial }: { children: ReactNode; initial?: Locale }) {
  const [locale, setLocaleState] = useState<Locale>(
    () => initial ?? initialLocale(safeStorage()?.getItem(STORAGE_KEY) ?? null, navigator.languages ?? [navigator.language]),
  );
  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
    safeStorage()?.setItem(STORAGE_KEY, next);
  }, []);
  useEffect(() => {
    document.documentElement.lang = LOCALES[locale].tag;
  }, [locale]);
  const value = useMemo(() => makeI18n(locale, setLocale), [locale, setLocale]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

function safeStorage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function useI18n(): I18n {
  const value = useContext(I18nContext);
  if (!value) throw new Error("useI18n outside I18nProvider");
  return value;
}
