// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * The light and dark themes (DESIGN.md). The choice is "system" (follow the
 * operating system's prefers-color-scheme), "light" or "dark", kept in the
 * browser; the page carries the theme it resolves to as
 * <html data-theme="light|dark">. public/theme.js applies the saved choice
 * before the first paint (the page's CSP allows no inline script), with
 * the same key and rule as here.
 */
import { useCallback, useEffect, useState } from "react";

export type ThemeChoice = "system" | "light" | "dark";
export type Theme = "light" | "dark";

export const THEME_KEY = "ukr-console-theme";
export const THEME_CHOICES: readonly ThemeChoice[] = ["system", "light", "dark"];
const DARK_QUERY = "(prefers-color-scheme: dark)";

interface KeyValue {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/** The saved choice; "system" when none is saved or it is not one. */
export function readChoice(storage: KeyValue | null): ThemeChoice {
  try {
    const saved = storage?.getItem(THEME_KEY);
    return saved === "light" || saved === "dark" ? saved : "system";
  } catch {
    return "system";
  }
}

/** Saves a choice; "system" removes the key, so the default follows the
 * operating system again. A storage that refuses (private mode) is
 * ignored: the choice then lasts for the page. */
export function saveChoice(storage: KeyValue | null, choice: ThemeChoice): void {
  try {
    if (choice === "system") storage?.removeItem(THEME_KEY);
    else storage?.setItem(THEME_KEY, choice);
  } catch {
    // The page keeps the choice; the next visit starts from the system's.
  }
}

/** The theme a choice resolves to. */
export function resolveTheme(choice: ThemeChoice, systemDark: boolean): Theme {
  return choice === "system" ? (systemDark ? "dark" : "light") : choice;
}

function storage(): KeyValue | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function systemDark(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia(DARK_QUERY).matches;
}

/** The choice, a setter that saves it, and the theme in effect; the page's
 * data-theme follows both the choice and, under "system", the operating
 * system's changes. */
export function useTheme(): { choice: ThemeChoice; theme: Theme; setChoice: (choice: ThemeChoice) => void } {
  const [choice, setChoiceState] = useState<ThemeChoice>(() => readChoice(storage()));
  const [dark, setDark] = useState(systemDark);
  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const query = window.matchMedia(DARK_QUERY);
    const onChange = () => setDark(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);
  const theme = resolveTheme(choice, dark);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  const setChoice = useCallback((next: ThemeChoice) => {
    setChoiceState(next);
    saveChoice(storage(), next);
  }, []);
  return { choice, theme, setChoice };
}
