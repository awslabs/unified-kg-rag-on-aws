// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { I18nProvider } from "./i18n/i18n";
import { TrailProvider } from "./lib/trail";

// The step the top bar is at, set by each test (jsdom lays nothing out).
let step = 3;
vi.mock("./lib/useCompactNav", async (original) => ({
  ...(await original<typeof import("./lib/useCompactNav")>()),
  useCompactNav: () => step,
}));

afterEach(() => vi.unstubAllGlobals());

describe("the top bar's menu", () => {
  it("starts closed when the row folds again", () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ config_path: "c.yaml", namespace: "default", jobs: [], sample_questions: [], strategies: [] }), { status: 200 })),
    );
    const app = () => (
      <I18nProvider initial="en">
        <TrailProvider>
          <App />
        </TrailProvider>
      </I18nProvider>
    );
    step = 3;
    const { rerender } = render(app());
    const menu = screen.getByRole("button", { name: "Menu" });
    fireEvent.click(menu);
    expect(menu).toHaveAttribute("aria-expanded", "true");
    // Regression (rv27): unfolded, the hidden button kept the menu open,
    // and narrowing the window again reopened it.
    step = 0;
    rerender(app());
    step = 3;
    rerender(app());
    expect(screen.getByRole("button", { name: "Menu" })).toHaveAttribute("aria-expanded", "false");
  });
});
