// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { I18nProvider } from "../i18n/i18n";
import { TrailProvider } from "../lib/trail";
import type { NamespaceHeld, Status } from "../lib/types";
import { IngestPage } from "./IngestPage";

// A namespace due for a rebuild under the config says why above the start
// button, each reason by its kind and what applies it (rebuild_due).

const status = { config_path: "c.yaml", namespace: "default", jobs: [], methodologies: ["graphrag"] } as unknown as Status;

function serve(held: NamespaceHeld) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const path = String(url).replace(/^\/api\//, "").split("?")[0];
      const body = path === "namespace/methodologies" ? held : path === "corpora" ? { corpora: [], extensions: [] } : { jobs: [] };
      return new Response(JSON.stringify(body), { status: 200 });
    }),
  );
}

const page = (locale: "en" | "ko") =>
  render(
    <I18nProvider initial={locale}>
      <TrailProvider>
        <IngestPage status={status} onJobsChanged={() => undefined} />
      </TrailProvider>
    </I18nProvider>,
  );

afterEach(() => vi.unstubAllGlobals());

describe("rebuild due", () => {
  const held: NamespaceHeld = {
    namespace: "default",
    recorded: true,
    indexed: true,
    held: ["graphrag"],
    dropped: [],
    rebuild: [
      { kind: "answer_settings", remedy: "reset", detail: "the community reports ...", settings: ["community_report"] },
      { kind: "extraction_settings", remedy: "next_run", detail: "changed", settings: ["processing.chunk_size"] },
    ],
  };

  it("lists each reason with what applies it", async () => {
    serve(held);
    page("en");
    expect(await screen.findByText("default is due for a rebuild")).toBeInTheDocument();
    const items = screen.getAllByRole("listitem").map((item) => item.textContent ?? "");
    const reports = items.find((text) => text.includes("community reports or description summaries"));
    expect(reports).toContain("Only a run with indexing.reset: true applies it.");
    expect(reports).toContain("community_report");
    const extraction = items.find((text) => text.includes("extraction settings changed"));
    expect(extraction).toContain("The next run applies it");
  });

  it("says it in the page's language", async () => {
    serve(held);
    page("ko");
    expect(await screen.findByText("default를 다시 빌드할 때가 되었습니다")).toBeInTheDocument();
  });

  it("says nothing for a namespace that is not due", async () => {
    serve({ ...held, rebuild: [] });
    page("en");
    await screen.findByRole("button", { name: "Start ingestion" });
    expect(screen.queryByText("default is due for a rebuild")).not.toBeInTheDocument();
  });
});
