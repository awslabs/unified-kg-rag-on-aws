// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/** One pass through the console against a demo workspace: a question, its
 * provenance trail into the graph, then every page, with no request to
 * another host and no error in the page. Skipped without a browser, except
 * in CI: a CI run without one would pass having tested nothing. */
import { existsSync } from "node:fs";
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

  await page.getByLabel("Language").first().selectOption("ko");
  await expect(page.locator("html")).toHaveAttribute("lang", "ko");

  expect(foreign).toEqual([]);
  expect(errors).toEqual([]);
});

// Regression (rv7): the top bar overflowed in English at 1200-1365 px and in
// German at every width up to 1440 (a fixed breakpoint). It collapses into
// its menu when its row does not fit, so nothing is cut at any width.
for (const locale of ["en", "de"]) {
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
      const { overflow, compact } = await row.evaluate((element) => ({
        overflow: element.scrollWidth - element.clientWidth,
        compact: element.closest("header")?.getAttribute("data-compact"),
      }));
      expect(overflow).toBeLessThanOrEqual(0);
      if (compact === "true") {
        // Collapsed: the menu opens every page link.
        await page.locator(".nav-menu").click();
        await expect(page.locator(".nav-list .nav-link")).toHaveCount(6);
        await expect(page.locator(".nav-list .nav-link").last()).toBeVisible();
      } else {
        await expect(page.locator(".nav-list .nav-link").last()).toBeInViewport();
      }
    });
  }
}
