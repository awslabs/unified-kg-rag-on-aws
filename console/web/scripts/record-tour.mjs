// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Record a scripted tour of the console (at most 25 s): a question streams
 * its answer, its sources open on the provenance trace, the trail leads
 * into the graph, an evaluation comparison, the dashboard.
 *
 *   node scripts/record-tour.mjs --base-url "http://127.0.0.1:8501/?token=..." --out tour \
 *     [--question "..."] [--width 1200] [--fps 12] [--locale en] [--max-seconds 25]
 *
 * Every step waits for the state it shows (the answer's result, its
 * sources, the graph drawn, the saved run's comparison, the dashboard's
 * figures) and fails, exit 1 with a screenshot of the page, instead of
 * recording a broken state. Waits on models and pages are cut from the
 * video: each idle span keeps a short head and tail and the rest plays
 * fast, so retrieval shows as a moment of "searching" and the answer as a
 * short stream, whatever the model's speed; a page's loading is cut.
 *
 * Writes tour.webm (the raw recording), tour.mp4 (the edit) and, with
 * ffmpeg on the path, tour.gif. The console needs a corpus ingested
 * (console/demo/demo.py prepares one); without a saved evaluation one is
 * run first with the first dataset it finds. CHROMIUM_PATH points at a
 * browser when Playwright's is not installed.
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

// The address run-console printed: its ?token= (or CONSOLE_TOKEN) is the
// access token, sent as a header to the API and set as the page's cookie.
const address = new URL(option("base-url", "http://127.0.0.1:8501"));
const token = address.searchParams.get("token") ?? process.env.CONSOLE_TOKEN ?? null;
const base = address.origin;
const out = option("out", "tour");
const question = option("question", "Who carries the lidar units to Kestrel Depot, and how fast?");
const width = Number(option("width", "1200"));
const fps = Number(option("fps", "12"));
const locale = option("locale", "en");
const maxSeconds = Number(option("max-seconds", "25"));
const VIEW = { width: 1280, height: 800 };
const MODEL_TIMEOUT = 180_000;
const PAGE_TIMEOUT = 30_000;

async function api(path, body) {
  const response = await fetch(`${base}/api/${path}`, {
    method: body ? "POST" : "GET",
    headers: {
      ...(body ? { "Content-Type": "application/json", "X-Console-Request": "1" } : {}),
      ...(token ? { "X-Console-Token": token } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) throw new Error(`${path}: ${response.status} ${await response.text()}`);
  return response.json();
}

/** A saved evaluation with a comparison, run first if there is none. */
async function ensureEvaluation() {
  const compared = (runs) => runs.filter((r) => r.comparisons.length);
  let { runs } = await api("evaluations");
  if (compared(runs).length) return compared(runs)[0];
  const { datasets } = await api("datasets");
  if (!datasets.length) throw new Error("no saved evaluation and no dataset to run one with");
  const job = await api("jobs/evaluate", { dataset: datasets[0].ref, strategies: ["local", "mix"], repeats: 2 });
  console.log(`running evaluation ${job.id} first`);
  for (;;) {
    const state = await api(`jobs/${job.id}`);
    if (!["running", "stopping"].includes(state.status)) {
      if (state.status !== "succeeded") throw new Error(`evaluation ${job.id} ${state.status}: ${state.error}`);
      break;
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  ({ runs } = await api("evaluations"));
  const mine = runs.find((r) => r.job_id === job.id);
  if (!mine?.comparisons.length) throw new Error(`evaluation ${job.id} saved no comparison`);
  return mine;
}

await mkdir(out, { recursive: true });
const run = await ensureEvaluation();
const raw = join(out, "raw");
await rm(raw, { recursive: true, force: true });
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
const context = await browser.newContext({ viewport: VIEW, recordVideo: { dir: raw, size: VIEW } });
await context.addInitScript((l) => window.localStorage.setItem("ukr-console-locale", l), locale);
const page = await context.newPage();
// The recording starts with the page.
const t0 = Date.now();
const now = () => (Date.now() - t0) / 1000;
const pause = (ms) => page.waitForTimeout(ms);
// [start, end, seconds it should take in the edit]
const squeezes = [];

/** Run `wait` and squeeze the time it took to `keep` seconds of video,
 * keeping `head` seconds as they were (the UI's waiting state) first. */
async function idle(label, wait, { head = 0, keep = 0.05 } = {}) {
  const start = now();
  try {
    await wait();
  } catch (error) {
    await fail(label, error);
  }
  const end = now();
  if (end - start > keep + 0.2) squeezes.push([start + head, end, Math.max(0.05, keep - head)]);
}

async function step(label, action) {
  try {
    await action();
  } catch (error) {
    await fail(label, error);
  }
}

async function fail(label, error) {
  const shot = join(out, `failed-${label}.png`);
  await page.screenshot({ path: shot, fullPage: true }).catch(() => undefined);
  await context.close().catch(() => undefined);
  await browser.close().catch(() => undefined);
  await rm(raw, { recursive: true, force: true });
  console.error(`step "${label}" failed (${shot}): ${error instanceof Error ? error.message : error}`);
  process.exit(1);
}

const visible = (selector, timeout = PAGE_TIMEOUT) => page.locator(selector).first().waitFor({ state: "visible", timeout });

// Ask: the strategies loaded, the question typed.
await idle("ask-page", async () => {
  await page.goto(token ? `${base}/ask?token=${encodeURIComponent(token)}` : `${base}/ask`);
  await visible(".chip");
});
await pause(700);
await step("type", async () => {
  await page.locator("#question").click();
  await page.keyboard.type(question, { delay: 26 });
});
await pause(250);
await page.keyboard.press("Enter");
// Retrieval: a moment of "searching", then the first chunk (or the whole
// answer, when it is not streamed).
await idle(
  "retrieval",
  () =>
    page.waitForFunction(() => {
      const text = document.querySelector(".answer-text");
      return (text && !text.querySelector(".muted") && text.textContent.trim().length > 0) || document.querySelector(".answer-meta");
    }, undefined, { timeout: MODEL_TIMEOUT }),
  { head: 0.7, keep: 1.0 },
);
// The stream, sped up to a couple of seconds, until the result event.
await idle("answer", () => visible(".answer-meta", MODEL_TIMEOUT), { head: 0.3, keep: 2.2 });
await step("answer-ok", async () => {
  if (await page.locator(".answer .notice-bad").count()) throw new Error(await page.locator(".answer .notice-bad").first().innerText());
  await visible(".source[data-rank]");
});
await pause(1400);

// The trace: open a graph source, follow one of its passages, back.
const graphSource = page.locator('.source[data-kind="entity"] > button, .source[data-kind="relationship"] > button').first();
await step("source", async () => {
  if (!(await graphSource.count())) throw new Error("the answer read no entity or relationship");
  await graphSource.hover();
  await pause(300);
  await graphSource.click();
  await visible(".source-trail");
});
await pause(1700);
const passage = page.locator(".source-trail .link-button.mono").first();
if (await passage.count()) {
  await passage.hover();
  await pause(250);
  await passage.click();
  await pause(1300);
  await graphSource.click();
  await visible(".source-trail .button-primary");
  await pause(400);
}

// Into the graph, on the trail.
await step("open-graph", () => page.locator(".source-trail .button-primary").first().click());
await idle("graph", async () => {
  await visible(".trail-banner");
  await visible(".graph-node circle");
});
await pause(2300);
await step("analysis", async () => {
  await page.getByRole("tab").nth(2).click();
  await visible('[role="tabpanel"] svg.chart');
});
await pause(1300);

// Evaluate: the saved run listed and its paired comparison drawn.
await idle("evaluate", async () => {
  await page.goto(`${base}/evaluate`);
  await visible(".chip");
  const listed = page.locator(".table .link-button.mono", { hasText: run.label });
  try {
    await listed.first().waitFor({ state: "visible", timeout: 8000 });
  } catch {
    await page.getByRole("button", { name: /refresh/i }).last().click();
    await listed.first().waitFor({ state: "visible", timeout: PAGE_TIMEOUT });
  }
  await listed.first().click();
  await visible("table svg.chart");
});
await pause(700);
await step("comparison", () =>
  page.locator(".sheet").filter({ has: page.locator("table svg.chart") }).first().evaluate((el) => el.scrollIntoView({ block: "start" })),
);
await pause(2600);

// Dashboard: its figures loaded.
await idle("dashboard", async () => {
  await page.goto(`${base}/dashboard`);
  await visible(".stat-value");
});
await pause(2000);
await page.evaluate(() => window.scrollTo({ top: document.body.scrollHeight / 3 }));
await pause(1200);
const end = now();

await context.close();
await browser.close();
const [recorded] = (await readdir(raw)).filter((f) => f.endsWith(".webm"));
const webm = join(out, "tour.webm");
await rename(join(raw, recorded), webm);
await rm(raw, { recursive: true, force: true });

// The edit: every span kept as recorded but the squeezed ones, sped up.
const pieces = [];
let cursor = 0;
for (const [start, stop, seconds] of squeezes.sort((a, b) => a[0] - b[0])) {
  if (start > cursor) pieces.push([cursor, start, 1]);
  pieces.push([start, stop, seconds / (stop - start)]);
  cursor = stop;
}
pieces.push([cursor, end, 1]);
const total = pieces.reduce((sum, [a, b, f]) => sum + (b - a) * f, 0);
console.log(`recorded ${end.toFixed(1)} s, edited to ${total.toFixed(1)} s (${squeezes.length} waits cut)`);
if (total > maxSeconds) {
  console.error(`the edit is ${total.toFixed(1)} s, over --max-seconds ${maxSeconds}`);
  process.exit(1);
}
if (spawnSync("ffmpeg", ["-version"]).status !== 0) {
  console.log("ffmpeg not found: only the raw WebM was written");
  process.exit(0);
}
const filters = pieces.map(
  ([a, b, f], i) => `[0:v]trim=start=${a.toFixed(3)}:end=${b.toFixed(3)},setpts=(PTS-STARTPTS)*${f.toFixed(5)}[p${i}]`,
);
const graph = `${filters.join(";")};${pieces.map((_, i) => `[p${i}]`).join("")}concat=n=${pieces.length}:v=1:a=0,fps=${fps},scale=${width}:-2:flags=lanczos[v]`;
const mp4 = join(out, "tour.mp4");
execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", webm, "-filter_complex", graph, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "24", mp4]);
// Duplicate frames dropped and a 64-colour palette without dithering: a UI
// stays crisp and the GIF small.
const gif = join(out, "tour.gif");
execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", mp4, "-vf", "mpdecimate=hi=1536:lo=640:frac=0.5,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle", "-fps_mode", "vfr", "-loop", "0", gif]);
const size = (await stat(gif)).size / 2 ** 20;
console.log(`wrote ${mp4} and ${gif} (${size.toFixed(1)} MiB)`);
if (size > 4) {
  console.error("the GIF is over 4 MiB: try --fps 10 or --width 1000");
  process.exit(1);
}
