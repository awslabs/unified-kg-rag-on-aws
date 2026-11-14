// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { I18nProvider, translate } from "../i18n/i18n";
import { TrailProvider } from "../lib/trail";
import type { JobRecord, NamespaceHeld, Status } from "../lib/types";
import { IngestPage, ResetAdvice } from "./IngestPage";

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
    // The console's own option, not a config key (W2 review E9).
    expect(reports).toContain("Only a run with “Rebuild the namespace” (More options) applies it.");
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

describe("a finished run that advises a rebuild", () => {
  // Regression (W2 review E1): the card said the communities had drifted
  // for any reason a reset applies (stale reports too): it names the run's
  // own reasons (its result's rebuild_due).
  const job = (reasons: { kind: string; remedy: string }[], extra: Partial<JobRecord> = {}) =>
    ({
      kind: "ingest",
      status: "succeeded",
      params: {},
      result: { rebuild_due: { default: reasons.map((r) => ({ ...r, detail: "d", settings: [] })) } },
      ...extra,
    }) as unknown as JobRecord;

  it("names the reasons the run reported, in the page's language", () => {
    let reset = 0;
    for (const locale of ["en", "ko"] as const) {
      const { unmount } = render(
        <I18nProvider initial={locale}>
          <ResetAdvice job={job([{ kind: "answer_settings", remedy: "reset" }, { kind: "methodologies", remedy: "next_run" }])} disabled={false} onReset={() => (reset += 1)} />
        </I18nProvider>,
      );
      expect(screen.getByText(translate(locale, "ingest.rebuild.answer_settings"))).toBeInTheDocument();
      expect(screen.queryByText(translate(locale, "ingest.rebuild.communities"))).not.toBeInTheDocument();
      expect(screen.queryByText(translate(locale, "ingest.rebuild.methodologies"))).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole("button", { name: translate(locale, "ingest.reset") }));
      unmount();
    }
    expect(reset).toBe(2);
    // No locale's advice claims drifted communities any more.
    for (const locale of ["en", "ko", "ja", "zh", "de", "es", "fr", "it", "pt", "ru"] as const) {
      // Nor that a warning says so: the run's result does (W2 re-check).
      expect(translate(locale, "ingest.resetAdvisedTitle"), locale).not.toMatch(/warn|경고|警告|avertissement|advertencia|avvis|aviso|предупрежд/i);
      expect(translate(locale, "ingest.resetAdvised"), locale).not.toMatch(/communit|커뮤니티|コミュニティ|社区|Communit|comunidad|communaut|comunit|сообществ/i);
    }
  });

  it("says nothing for a run with no reason a reset applies, or one that reset", () => {
    const { container, rerender } = render(
      <I18nProvider initial="en">
        <ResetAdvice job={job([{ kind: "methodologies", remedy: "next_run" }])} disabled={false} onReset={() => undefined} />
      </I18nProvider>,
    );
    expect(container).toBeEmptyDOMElement();
    rerender(
      <I18nProvider initial="en">
        <ResetAdvice job={job([{ kind: "communities", remedy: "reset" }], { params: { reset: true } })} disabled={false} onReset={() => undefined} />
      </I18nProvider>,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
