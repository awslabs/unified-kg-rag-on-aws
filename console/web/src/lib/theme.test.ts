// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { readChoice, resolveTheme, saveChoice, THEME_KEY, useTheme } from "./theme";

function memory(initial: Record<string, string> = {}) {
  const items = new Map(Object.entries(initial));
  return {
    getItem: (key: string) => items.get(key) ?? null,
    setItem: (key: string, value: string) => void items.set(key, value),
    removeItem: (key: string) => void items.delete(key),
    items,
  };
}

function mockSystem(dark: boolean) {
  const listeners: (() => void)[] = [];
  const query = {
    matches: dark,
    addEventListener: (_: string, listener: () => void) => listeners.push(listener),
    removeEventListener: vi.fn(),
  };
  vi.stubGlobal("matchMedia", () => query);
  return {
    set(next: boolean) {
      query.matches = next;
      listeners.forEach((l) => l());
    },
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  window.localStorage.clear();
  delete document.documentElement.dataset.theme;
});

describe("the theme choice", () => {
  it("defaults to the system's and reads back what was saved", () => {
    const storage = memory();
    expect(readChoice(storage)).toBe("system");
    saveChoice(storage, "dark");
    expect(storage.items.get(THEME_KEY)).toBe("dark");
    expect(readChoice(storage)).toBe("dark");
    saveChoice(storage, "light");
    expect(readChoice(storage)).toBe("light");
  });
  it("forgets the key when the system's is chosen again", () => {
    const storage = memory({ [THEME_KEY]: "dark" });
    saveChoice(storage, "system");
    expect(storage.items.has(THEME_KEY)).toBe(false);
    expect(readChoice(storage)).toBe("system");
  });
  it("treats an unknown value or a refusing storage as the system's", () => {
    expect(readChoice(memory({ [THEME_KEY]: "sepia" }))).toBe("system");
    const refusing = {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("denied");
      },
      removeItem: () => undefined,
    };
    expect(readChoice(refusing)).toBe("system");
    expect(() => saveChoice(refusing, "dark")).not.toThrow();
    expect(readChoice(null)).toBe("system");
  });
  it("resolves the system's choice by the operating system", () => {
    expect(resolveTheme("system", true)).toBe("dark");
    expect(resolveTheme("system", false)).toBe("light");
    expect(resolveTheme("light", true)).toBe("light");
    expect(resolveTheme("dark", false)).toBe("dark");
  });
  it("sets the page's theme, saves a choice, and follows the system under system", () => {
    const system = mockSystem(false);
    const { result } = renderHook(() => useTheme());
    expect(result.current.choice).toBe("system");
    expect(document.documentElement.dataset.theme).toBe("light");
    act(() => system.set(true));
    expect(document.documentElement.dataset.theme).toBe("dark");
    act(() => result.current.setChoice("light"));
    expect(window.localStorage.getItem(THEME_KEY)).toBe("light");
    expect(document.documentElement.dataset.theme).toBe("light");
    // A later visit starts from the saved choice.
    const again = renderHook(() => useTheme());
    expect(again.result.current.choice).toBe("light");
  });
  it("is applied before the first paint by public/theme.js, with the same key", () => {
    const script = readFileSync(join(process.cwd(), "public", "theme.js"), "utf8");
    expect(script).toContain(`"${THEME_KEY}"`);
    window.localStorage.setItem(THEME_KEY, "dark");
    mockSystem(false);
    new Function(script)();
    expect(document.documentElement.dataset.theme).toBe("dark");
    window.localStorage.removeItem(THEME_KEY);
    mockSystem(true);
    new Function(script)();
    expect(document.documentElement.dataset.theme).toBe("dark");
    mockSystem(false);
    new Function(script)();
    expect(document.documentElement.dataset.theme).toBe("light");
  });
  it("is linked from index.html ahead of the app", () => {
    const html = readFileSync(join(process.cwd(), "index.html"), "utf8");
    expect(html.indexOf('src="/theme.js"')).toBeGreaterThan(0);
    expect(html.indexOf('src="/theme.js"')).toBeLessThan(html.indexOf("/src/main.tsx"));
  });
});
