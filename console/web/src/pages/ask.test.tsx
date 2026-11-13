// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { I18nProvider, useI18n, type Locale } from "../i18n/i18n";
import { ApiError, errorMessage } from "../lib/api";
import { TrailProvider } from "../lib/trail";
import type { Status } from "../lib/types";
import { AskPage, turnErrorText } from "./AskPage";

const status = { config_path: "c.yaml", namespace: "default", jobs: [], sample_questions: ["Who ships parts?"], max_top_k: 40 } as unknown as Status;

function serve(recorded: boolean, indexed = recorded) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const body = String(url).includes("namespace/methodologies") ? { namespace: "default", recorded, indexed, held: recorded ? ["graphrag"] : [], dropped: [] } : { strategies: [] };
      return new Response(JSON.stringify(body), { status: 200 });
    }),
  );
}

const renderAsk = (locale: Locale = "en") =>
  render(
    <I18nProvider initial={locale}>
      <TrailProvider>
        <AskPage status={status} />
      </TrailProvider>
    </I18nProvider>,
  );

afterEach(() => vi.unstubAllGlobals());

describe("AskPage welcome", () => {
  it("says an empty namespace holds nothing yet and leads to Ingest", async () => {
    serve(false);
    renderAsk();
    await waitFor(() => expect(screen.getByText("Nothing is indexed in “default” yet")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Go to Ingest" })).toBeInTheDocument();
    expect(screen.queryByText("Who ships parts?")).not.toBeInTheDocument();
  });

  it("offers the sample questions once the namespace holds an index", async () => {
    serve(true);
    renderAsk();
    await waitFor(() => expect(screen.getByText("Who ships parts?")).toBeInTheDocument());
    expect(screen.queryByText(/Nothing is indexed/)).not.toBeInTheDocument();
  });

  it("reads an index without a build record as indexed: it answers", async () => {
    serve(false, true);
    renderAsk();
    await waitFor(() => expect(screen.getByText("Who ships parts?")).toBeInTheDocument());
    expect(screen.queryByText(/Nothing is indexed/)).not.toBeInTheDocument();
  });

  it("bounds top_k by the server's largest", async () => {
    serve(true);
    renderAsk();
    await waitFor(() => expect(screen.getByText("Who ships parts?")).toBeInTheDocument());
    const topK = document.querySelector<HTMLInputElement>('input[type="number"]');
    expect(topK?.max).toBe("40");
  });
});

/** The index answers as ``serve(true)`` does, and a question's stream sends
 * ``events``. */
function serveStream(events: object[]) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (String(url).endsWith("/api/ask")) {
        const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
        return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
      }
      const body = String(url).includes("namespace/methodologies") ? { namespace: "default", recorded: true, indexed: true, held: ["graphrag"], dropped: [] } : { strategies: [] };
      return new Response(JSON.stringify(body), { status: 200 });
    }),
  );
}

describe("a failure inside the server", () => {
  const masked = "The call failed inside the server (a store or model error); the server log has the details.";

  it("is said in the page's language, with the ids its log lines carry", async () => {
    serveStream([
      { kind: "start", query_id: "9b5c9b39e6d7", trace_id: null, trace_url: null },
      { kind: "error", error_kind: "failure", message: masked, query_id: "9b5c9b39e6d7", trace_id: "4bf92f3577b34da6" },
    ]);
    renderAsk("ko");
    fireEvent.click(await screen.findByText("Who ships parts?"));
    expect(await screen.findByText(/쿼리 ID 9b5c9b39e6d7/)).toHaveTextContent("트레이스 ID: 4bf92f3577b34da6");
    expect(screen.queryByText(masked)).not.toBeInTheDocument();
  });

  it("takes the query id from the start event when the error names none", async () => {
    serveStream([
      { kind: "start", query_id: "0a1b2c3d4e5f", trace_id: null, trace_url: null },
      { kind: "error", error_kind: "failure", message: masked },
    ]);
    renderAsk();
    fireEvent.click(await screen.findByText("Who ships parts?"));
    expect(await screen.findByText("The call failed inside the server (a store or model error). The server log has the details under query id 0a1b2c3d4e5f.")).toBeInTheDocument();
  });

  it("is said again in the new language when the page's language switches", async () => {
    serveStream([
      { kind: "start", query_id: "9b5c9b39e6d7", trace_id: null, trace_url: null },
      { kind: "error", error_kind: "failure", message: masked, query_id: "9b5c9b39e6d7", trace_id: "4bf92f3577b34da6" },
    ]);
    function Switch() {
      const { setLocale } = useI18n();
      return <button onClick={() => setLocale("ko")}>to-ko</button>;
    }
    render(
      <I18nProvider initial="en">
        <TrailProvider>
          <Switch />
          <AskPage status={status} />
        </TrailProvider>
      </I18nProvider>,
    );
    fireEvent.click(await screen.findByText("Who ships parts?"));
    expect(await screen.findByText(/under query id 9b5c9b39e6d7/)).toBeInTheDocument();
    fireEvent.click(screen.getByText("to-ko"));
    expect(await screen.findByText(/쿼리 ID 9b5c9b39e6d7/)).toHaveTextContent("트레이스 ID: 4bf92f3577b34da6");
    expect(screen.queryByText(/under query id/)).not.toBeInTheDocument();
  });

  it("says a question the network lost in the new language, too", () => {
    const t = (key: string) => key;
    expect(turnErrorText(t, { error: { kind: "failure", cause: new TypeError("fetch failed") } })).toBe("error.network");
    expect(turnErrorText(t, { error: { kind: "incomplete" } })).toBe("ask.incomplete");
    expect(turnErrorText(t, { error: { kind: "index_not_found", message: "no index" } })).toBe("no index");
  });

  it("of another request is not called a query's", () => {
    const t = (key: string, vars?: Record<string, string | number>) => `${key}${vars ? JSON.stringify(vars) : ""}`;
    expect(errorMessage(t, new ApiError(500, masked, { code: "failure", error: masked }))).toBe("error.failure");
    expect(errorMessage(t, new ApiError(500, masked, { code: "failure", trace_id: "ab12" }))).toBe('error.failure error.traceId{"trace_id":"ab12"}');
  });
});
