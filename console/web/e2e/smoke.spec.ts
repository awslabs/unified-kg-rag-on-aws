// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/** One pass through the console against a demo workspace: a question, its
 * provenance trail into the graph, then every page, with no request to
 * another host and no error in the page. Skipped without a browser, except
 * in CI: a CI run without one would pass having tested nothing. */
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { chromium, expect, test } from "@playwright/test";

const browser = process.env.CHROMIUM_PATH ?? chromium.executablePath();
// CI providers set CI=true (or 1); CI=false or an empty CI is a local run.
const inCI = ["true", "1"].includes((process.env.CI ?? "").trim().toLowerCase());
if (inCI && !existsSync(browser)) {
  throw new Error(`no browser at ${browser} in CI (npx playwright install chromium)`);
}
test.skip(!existsSync(browser), `no browser at ${browser} (npx playwright install chromium)`);

test("ask, follow the trail into the graph, visit every page", async ({ page, baseURL }) => {
  const origin = new URL(baseURL ?? "http://127.0.0.1:8501").origin;
  const foreign: string[] = [];
  const errors: string[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.protocol.startsWith("http") && url.origin !== origin) foreign.push(request.url());
  });
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.addInitScript(() => window.localStorage.setItem("ukr-console-locale", "en"));

  // The address run-console printed carries the access token.
  const token = new URL(process.env.CONSOLE_URL ?? origin).searchParams.get("token") ?? process.env.CONSOLE_TOKEN;
  await page.goto(token ? `/ask?token=${encodeURIComponent(token)}` : "/ask");
  await expect(page.getByRole("heading", { level: 1, name: "Ask" })).toBeVisible();
  await page.getByLabel("Question").fill("Who carries the lidar units to Kestrel Depot?");
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  await expect(page.locator(".answer-meta").first()).toBeVisible({ timeout: 60_000 });
  await expect(page.locator(".answer-text").first()).toContainText("Umbra Freight");
  // The answer's [n] opens the source it cites in the trace.
  const cite = page.locator(".answer-text .cite-link").first();
  await expect(cite).toBeVisible();
  const marker = (await cite.textContent()) ?? "";
  await cite.click();
  await expect(page.locator(".source.open .marker")).toHaveText(marker);
  await expect(page.locator(".citations-list li").first()).toBeVisible();
  await page.locator('.source[data-kind="entity"] > button').first().click();
  await expect(page.getByText("Stated in passages")).toBeVisible();
  await page.getByRole("button", { name: "Show in the graph" }).click();
  await expect(page).toHaveURL(/\/graph\?focus=/);
  await expect(page.locator(".trail-banner")).toContainText("From your question");
  await expect(page.locator(".graph-canvas")).toHaveAttribute("data-traced", "1");
  await expect(page.locator(".graph-canvas canvas").first()).toBeVisible();

  for (const [path, title] of <[string, string][]>[
    ["/ingest", "Ingest"],
    ["/evaluate", "Evaluate"],
    ["/dashboard", "Dashboard"],
    ["/settings", "Settings"],
  ]) {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1, name: title })).toBeVisible();
  }
  // Ingest starts from what the namespace holds; building less asks first.
  await page.goto("/ingest");
  await expect(page.getByText(/^The namespace default holds /)).toBeVisible();
  await page.locator("label.source").first().click();
  const graphrag = page.getByRole("checkbox", { name: /^GraphRAG/ });
  await expect(graphrag).toBeChecked();
  await graphrag.uncheck();
  await expect(page.getByText("This run removes methodologies from default")).toBeVisible();
  const start = page.getByRole("button", { name: "Start ingestion" });
  await expect(start).toBeDisabled();
  await page.getByRole("checkbox", { name: /^Remove GraphRAG/ }).check();
  await expect(start).toBeEnabled();
  await graphrag.check();
  await expect(page.getByText("This run removes methodologies from default")).toBeHidden();

  // An evaluation counts its answers, then compares with Holm-adjusted p-values.
  await page.goto("/evaluate");
  await page.getByRole("button", { name: "Start evaluation" }).click();
  await expect(page.getByText(/answers ·/).first()).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("2 of 2 runs done")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("columnheader", { name: "Holm p" })).toBeVisible({ timeout: 30_000 });
  await expect(page.locator("td .badge", { hasText: "primary" }).first()).toBeVisible();
  // Per question: every strategy's score, and the answers side by side.
  await page.getByText("Show each question’s score and answers").click();
  await page.locator(".question-toggle").first().click();
  await expect(page.locator(".answers-side .answer-card")).toHaveCount(3);

  // The theme: chosen in the shell, kept across a reload, back to the system's.
  await page.goto("/settings");
  const theme = page.getByRole("group", { name: "Theme" });
  await theme.getByRole("button", { name: "Dark" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(theme.getByRole("button", { name: "Dark" })).toHaveAttribute("aria-pressed", "true");
  await theme.getByRole("button", { name: "Light" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await theme.getByRole("button", { name: "System" }).click();
  expect(await page.evaluate(() => window.localStorage.getItem("ukr-console-theme"))).toBeNull();

  // One language picker: the top bar's.
  await expect(page.getByLabel("Language")).toHaveCount(1);
  await page.getByLabel("Language").selectOption("ko");
  await expect(page.locator("html")).toHaveAttribute("lang", "ko");

  expect(foreign).toEqual([]);
  expect(errors).toEqual([]);
});

// Regression (rv7): the top bar overflowed in English at 1200-1365 px and in
// German at every width up to 1440 (a fixed breakpoint). It compacts in
// steps (tagline, field labels) when its row does not fit, and folds into
// its menu only after them, so nothing is cut at any width (rv25: German,
// French and Russian folded at every width, the row being capped at 1440).
for (const locale of ["en", "de", "fr", "ru", "es"]) {
  for (const width of [1280, 1440]) {
    test(`the top bar fits at ${width} px in ${locale}`, async ({ page, baseURL }) => {
      const origin = new URL(baseURL ?? "http://127.0.0.1:8501").origin;
      await page.setViewportSize({ width, height: 900 });
      await page.addInitScript((code) => window.localStorage.setItem("ukr-console-locale", code), locale);
      const token = new URL(process.env.CONSOLE_URL ?? origin).searchParams.get("token") ?? process.env.CONSOLE_TOKEN;
      const statusRead = page.waitForResponse((response) => response.url().includes("/api/status"));
      await page.goto(token ? `/ask?token=${encodeURIComponent(token)}` : "/ask");
      const row = page.locator(".nav-inner");
      await expect(row).toBeVisible();
      // Checked once what changes the row's width has arrived: the status
      // (the namespace chip) and the web fonts.
      await statusRead;
      await page.evaluate(() => document.fonts.ready.then(() => undefined));
      await page.waitForTimeout(100);
      const { overflow, fold } = await row.evaluate((element) => ({
        overflow: element.scrollWidth - element.clientWidth,
        fold: element.closest("header")?.getAttribute("data-fold"),
      }));
      expect(overflow).toBeLessThanOrEqual(0);
      // At the widest content width every language keeps its links inline.
      if (width === 1440) expect(fold).not.toBe("3");
      if (fold === "3") {
        // Folded: the menu opens every page link.
        await page.locator(".nav-menu").click();
        await expect(page.locator(".nav-list .nav-link")).toHaveCount(6);
        await expect(page.locator(".nav-list .nav-link").last()).toBeVisible();
      } else {
        await expect(page.locator(".nav-list .nav-link").last()).toBeInViewport();
      }
    });
  }
}

// Regression (rv26): narrowing the window folded the top bar into its menu,
// skipping the compaction steps (each step was measured as the row's
// scrollWidth, never under its width at load). Loaded wide and narrowed,
// the row takes the step it takes when loaded at that width (rv27: loaded
// folded, Japanese measured every step 8 px wide, the folded row's padding,
// and folded at 1180 px where a narrowed window fits).
for (const locale of ["en", "ko", "de", "ja"]) {
  test(`the top bar compacts as the window narrows in ${locale}`, async ({ browser, baseURL }) => {
    const origin = new URL(baseURL ?? "http://127.0.0.1:8501").origin;
    const token = new URL(process.env.CONSOLE_URL ?? origin).searchParams.get("token") ?? process.env.CONSOLE_TOKEN;
    const open = async (width: number) => {
      const context = await browser.newContext({ viewport: { width, height: 900 } });
      await context.addInitScript((code) => window.localStorage.setItem("ukr-console-locale", code), locale);
      const page = await context.newPage();
      const statusRead = page.waitForResponse((response) => response.url().includes("/api/status"));
      await page.goto(token ? `/ask?token=${encodeURIComponent(token)}` : "/ask");
      await statusRead;
      await page.evaluate(() => document.fonts.ready.then(() => undefined));
      await page.waitForTimeout(100);
      return { context, page };
    };
    const fold = (page: import("@playwright/test").Page) => page.locator("header.nav").getAttribute("data-fold");
    const wide = await open(1920);
    for (const width of [1280, 1180]) {
      await wide.page.setViewportSize({ width, height: 900 });
      await wide.page.waitForTimeout(200);
      const direct = await open(width);
      expect(await fold(wide.page), `${width} px`).toBe(await fold(direct.page));
      const overflow = await wide.page.locator(".nav-inner").evaluate((row) => row.scrollWidth - row.clientWidth);
      expect(overflow).toBeLessThanOrEqual(0);
      await direct.context.close();
    }
    await wide.context.close();
  });
}

// Regression (rv27): at the compaction step that hides the pickers' field
// labels (1280 px in English and German), the language picker lost its
// accessible name with its label's text. axe-core finds no violation there.
const axeSource = readFileSync(createRequire(import.meta.url).resolve("axe-core/axe.min.js"), "utf8");
for (const locale of ["en", "de"]) {
  test(`no accessibility violation at 1280 px in ${locale}`, async ({ page, baseURL }) => {
    const origin = new URL(baseURL ?? "http://127.0.0.1:8501").origin;
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.addInitScript((code) => window.localStorage.setItem("ukr-console-locale", code), locale);
    const token = new URL(process.env.CONSOLE_URL ?? origin).searchParams.get("token") ?? process.env.CONSOLE_TOKEN;
    const statusRead = page.waitForResponse((response) => response.url().includes("/api/status"));
    await page.goto(token ? `/ask?token=${encodeURIComponent(token)}` : "/ask");
    await statusRead;
    await page.evaluate(() => document.fonts.ready.then(() => undefined));
    await page.waitForTimeout(100);
    expect(await page.locator("header.nav").getAttribute("data-fold")).toBe("2");
    await expect(page.locator("#locale")).toHaveAccessibleName(locale === "de" ? "Sprache" : "Language");
    // Evaluated, not a script tag: the page's CSP allows no inline script.
    await page.evaluate(axeSource);
    const violations = await page.evaluate(async () => {
      const axe = (window as unknown as { axe: { run: (context: Document) => Promise<{ violations: { id: string; nodes: { target: string[] }[] }[] }> } }).axe;
      const { violations } = await axe.run(document);
      return violations.map((violation) => `${violation.id}: ${violation.nodes.map((node) => node.target.join(" ")).join(", ")}`);
    });
    expect(violations).toEqual([]);
  });
}
