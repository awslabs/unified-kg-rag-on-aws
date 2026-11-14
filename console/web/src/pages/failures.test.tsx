// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { I18nProvider, useI18n } from "../i18n/i18n";
import { TrailProvider } from "../lib/trail";
import type { Status } from "../lib/types";
import { EvaluatePage } from "./EvaluatePage";
import { IngestPage } from "./IngestPage";
import { SettingsPage } from "./SettingsPage";

// A failed request is kept as the error and said when the page renders, so
// a language switch says it again (as Ask's turns are).

const status = { config_path: "c.yaml", namespace: "default", jobs: [], methodologies: ["graphrag"] } as unknown as Status;

const answers: [string, unknown][] = [
  ["namespace/methodologies", { namespace: "default", recorded: false, indexed: false, held: [], dropped: [] }],
  ["corpora", { corpora: [{ ref: "corpus:a", name: "a", files: 1, unsupported: 0, bytes: 10, extensions: { ".txt": 1 }, modified: 1 }], extensions: [".txt"] }],
  ["datasets/preview", { items: 1, unanswerable: 0, feeds: {}, sample: [] }],
  ["datasets", { datasets: [{ ref: "dataset:a", name: "a", items: 1 }] }],
  ["evaluations", { runs: [] }],
  ["strategies", { strategies: [] }],
  ["jobs", { jobs: [] }],
  ["config/schema", { type: "object", properties: {} }],
  [
    "config",
    { path: "c.yaml", writable: true, sha: "s", text: "a: 1\n", data: { a: 1 }, secrets_in_file: [], parse_error: null, load_error: null, effective: null, methodologies: [], tracing_environment: {}, otel_installed: false },
  ],
];

/** Every read answers; every write finds the console gone. */
function serve() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method && init.method !== "GET") throw new TypeError("fetch failed");
      const path = String(url).replace(/^\/api\//, "").split("?")[0];
      const found = answers.find(([route]) => path === route);
      return new Response(JSON.stringify(found ? found[1] : {}), { status: 200 });
    }),
  );
}

function Switch() {
  const { setLocale } = useI18n();
  return <button onClick={() => setLocale("ko")}>to-ko</button>;
}

const renderPage = (page: ReactElement) =>
  render(
    <I18nProvider initial="en">
      <TrailProvider>
        <Switch />
        {page}
      </TrailProvider>
    </I18nProvider>,
  );

const english = "The console does not answer. Is run-console still running?";
const korean = "콘솔이 응답하지 않습니다. run-console이 실행 중인지 확인하세요.";

async function saysItAgainInKorean() {
  expect(await screen.findByText(english)).toBeInTheDocument();
  fireEvent.click(screen.getByText("to-ko"));
  expect(await screen.findByText(korean)).toBeInTheDocument();
  expect(screen.queryByText(english)).not.toBeInTheDocument();
}

afterEach(() => vi.unstubAllGlobals());

describe("a failed request is said in the page's language", () => {
  it("on Ingest", async () => {
    serve();
    renderPage(<IngestPage status={status} onJobsChanged={() => undefined} />);
    fireEvent.click(await screen.findByRole("radio"));
    fireEvent.click(screen.getByRole("button", { name: "Start ingestion" }));
    await saysItAgainInKorean();
  });

  it("on Evaluate", async () => {
    serve();
    renderPage(<EvaluatePage status={status} onJobsChanged={() => undefined} />);
    const start = await screen.findByRole("button", { name: "Start evaluation" });
    await vi.waitFor(() => expect(start).toBeEnabled());
    fireEvent.click(start);
    await saysItAgainInKorean();
  });

  it("on Settings", async () => {
    serve();
    renderPage(<SettingsPage onSaved={() => undefined} />);
    fireEvent.click(await screen.findByRole("tab", { name: "YAML" }));
    // The YAML tab's (the pending fields' one is off with nothing changed).
    const review = screen.getAllByRole("button", { name: "Review changes" }).find((b) => !(b as HTMLButtonElement).disabled);
    fireEvent.click(review as HTMLElement);
    await saysItAgainInKorean();
  });
});
