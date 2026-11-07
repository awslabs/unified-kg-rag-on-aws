// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Record a scripted tour of the console (about 25 s): a question streams
 * its answer, its sources open on the provenance trace, the trail leads
 * into the graph, an evaluation comparison, the dashboard.
 *
 *   node scripts/record-tour.mjs --base-url http://127.0.0.1:8501 --out tour \
 *     [--question "..."] [--width 1200] [--fps 12] [--locale en]
 *
 * Writes tour.webm (Playwright's recording), and with ffmpeg on the path
 * tour.gif (palettegen/paletteuse) and tour.mp4 of the same framing. The
 * console needs a corpus ingested (console/demo/demo.py prepares one); if
 * it has no saved evaluation, one is run first with the dataset it finds.
 * CHROMIUM_PATH points at a browser when Playwright's is not installed.
 */
/* global window, document -- page callbacks */
import { execFileSync, spawnSync } from "node:child_process";
import { mkdir, readdir, rename, rm, stat } from "node:fs/promises";
import { join } from "node:path";
import { chromium } from "@playwright/test";

function option(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index > 0 ? process.argv[index + 1] : fallback;
}

const base = option("base-url", "http://127.0.0.1:8501").replace(/\/$/, "");
const out = option("out", "tour");
const question = option("question", "Who carries the lidar units to Kestrel Depot, and how fast?");
const width = Number(option("width", "1200"));
const fps = Number(option("fps", "12"));
const locale = option("locale", "en");
const VIEW = { width: 1280, height: 800 };

async function api(path, body) {
  const response = await fetch(`${base}/api/${path}`, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json", "X-Console-Request": "1" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) throw new Error(`${path}: ${response.status} ${await response.text()}`);
  return response.json();
}

async function ensureEvaluation() {
  const { runs } = await api("evaluations");
  if (runs.length) return;
  const { datasets } = await api("datasets");
  if (!datasets.length) throw new Error("no saved evaluation and no dataset to run one with");
  const job = await api("jobs/evaluate", { dataset: datasets[0].ref, strategies: ["local", "mix"], repeats: 2 });
  console.log(`running evaluation ${job.id} first`);
  for (;;) {
    const state = await api(`jobs/${job.id}`);
    if (!["running", "stopping"].includes(state.status)) break;
    await new Promise((r) => setTimeout(r, 1000));
  }
}

await mkdir(out, { recursive: true });
await ensureEvaluation();
const raw = join(out, "raw");
await rm(raw, { recursive: true, force: true });
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
const context = await browser.newContext({ viewport: VIEW, recordVideo: { dir: raw, size: VIEW } });
await context.addInitScript((l) => window.localStorage.setItem("ukr-console-locale", l), locale);
const page = await context.newPage();
const pause = (ms) => page.waitForTimeout(ms);

await page.goto(`${base}/ask`);
await page.waitForLoadState("networkidle");
await pause(800);
await page.locator("#question").click();
await page.keyboard.type(question, { delay: 28 });
await pause(300);
await page.keyboard.press("Enter");
await page.locator(".answer-meta").first().waitFor({ timeout: 60000 });
await pause(1600);
const entity = page.locator('.source[data-kind="entity"] > button').first();
await entity.hover();
await pause(400);
await entity.click();
await pause(1800);
const passage = page.locator(".source-trail .link-button.mono").first();
if (await passage.count()) {
  await passage.hover();
  await pause(300);
  await passage.click();
  await pause(1500);
  await entity.click();
  await pause(600);
}
await page.locator(".source-trail .button-primary").first().click();
await page.locator(".graph-canvas").waitFor();
await pause(2600);
await page.getByRole("tab").nth(2).click();
await pause(1600);

await page.goto(`${base}/evaluate`);
await page.waitForLoadState("networkidle");
await pause(900);
const paired = page.locator(".sheet").filter({ has: page.locator("svg.chart") }).first();
await paired.evaluate((el) => el.scrollIntoView({ behavior: "smooth", block: "start" }));
await pause(3200);

await page.goto(`${base}/dashboard`);
await page.waitForLoadState("networkidle");
await pause(2600);
await page.evaluate(() => window.scrollTo({ top: document.body.scrollHeight / 3, behavior: "smooth" }));
await pause(1800);

await context.close();
await browser.close();

const [recorded] = (await readdir(raw)).filter((f) => f.endsWith(".webm"));
const webm = join(out, "tour.webm");
await rename(join(raw, recorded), webm);
await rm(raw, { recursive: true, force: true });
console.log(`recorded ${webm}`);

if (spawnSync("ffmpeg", ["-version"]).status !== 0) {
  console.log("ffmpeg not found: only the WebM was written");
  process.exit(0);
}
// Duplicate frames dropped (mpdecimate) and a 64-colour palette with no
// dithering: a UI stays crisp and the GIF small (about 3.5 MiB for 28 s).
const scale = `fps=${fps},mpdecimate=hi=1536:lo=640:frac=0.5,scale=${width}:-2:flags=lanczos`;
const gif = join(out, "tour.gif");
execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", webm, "-vf", `${scale},split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle`, "-fps_mode", "vfr", "-loop", "0", gif]);
execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", webm, "-vf", `scale=${width}:-2:flags=lanczos`, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "26", join(out, "tour.mp4")]);
const size = (await stat(gif)).size / 2 ** 20;
console.log(`wrote ${gif} (${size.toFixed(1)} MiB) and tour.mp4`);
if (size > 4) console.log("the GIF is over 4 MiB: try --fps 10 or --width 1000");
