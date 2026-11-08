// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Screenshot every console page against a running console.
 *
 *   node scripts/screenshots.mjs --base-url "http://127.0.0.1:8501/?token=..." --out shots \
 *     [--locales en,ko] [--widths 1440,390] [--themes light,dark]
 *
 * It asks the first sample question (so the console must have a corpus
 * ingested; console/demo/demo.py prepares one), opens its first graph
 * source and walks every page. CHROMIUM_PATH points at a browser when the
 * one Playwright pins is not installed. With --themes, each shot is taken
 * in each theme and named <page>-<locale>-<theme>-<width>.png; without it,
 * in the browser's default (light) as <page>-<locale>-<width>.png.
 */
/* global window -- the init script runs in the page */
import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { chromium } from "@playwright/test";

function option(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index > 0 ? process.argv[index + 1] : fallback;
}

// The address run-console printed (its ?token=, or CONSOLE_TOKEN).
const address = new URL(option("base-url", "http://127.0.0.1:8501"));
const token = address.searchParams.get("token") ?? process.env.CONSOLE_TOKEN ?? null;
const base = address.origin;
const out = option("out", "shots");
const locales = option("locales", "en").split(",");
const widths = option("widths", "1440,390").split(",").map(Number);
const themes = option("themes", "") ? option("themes", "").split(",") : [null];
await mkdir(out, { recursive: true });

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
for (const theme of themes) {
  for (const locale of locales) {
    for (const width of widths) {
      const mobile = width < 700;
      const context = await browser.newContext({
        viewport: { width, height: mobile ? 844 : 900 },
        deviceScaleFactor: mobile ? 2 : 1,
        reducedMotion: "reduce",
      });
      await context.addInitScript(
        ([l, th]) => {
          window.localStorage.setItem("ukr-console-locale", l);
          if (th) window.localStorage.setItem("ukr-console-theme", th);
        },
        [locale, theme],
      );
      const page = await context.newPage();
      const shot = async (name) => {
        await page.waitForTimeout(300);
        const label = theme ? `${name}-${locale}-${theme}-${width}` : `${name}-${locale}-${width}`;
        await page.screenshot({ path: join(out, `${label}.png`), fullPage: true });
      };
      await page.goto(token ? `${base}/ask?token=${encodeURIComponent(token)}` : `${base}/ask`);
      await page.locator(".sample").first().click();
      await page.locator(".answer-meta").first().waitFor({ timeout: 60000 });
      await page.locator('.source[data-kind="entity"] > button').first().click();
      await shot("ask");
      const open = page.locator(".source-trail .button-primary").first();
      if (await open.count()) {
        await open.click();
        await page.locator(".graph-canvas").waitFor();
        await page.waitForTimeout(600);
        await shot("graph-trail");
        await page.getByRole("tab").nth(2).click();
        await shot("graph-analysis");
      }
      for (const name of ["ingest", "evaluate", "dashboard", "settings"]) {
        await page.goto(`${base}/${name}`);
        await page.waitForLoadState("networkidle");
        if (name === "settings") {
          const section = page.locator(".config-section > summary").filter({ hasText: /^search/ }).first();
          if (await section.count()) await section.click();
        }
        await shot(name);
      }
      await context.close();
    }
  }
}
await browser.close();
console.log(`screenshots in ${out}`);
