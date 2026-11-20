// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { I18nProvider, useI18n, type Locale } from "../i18n/i18n";
import { ApiError, errorMessage } from "../lib/api";
import { TrailProvider } from "../lib/trail";
import type { Status } from "../lib/types";
import { AskPage, graphParams, processingRows, searchTypeOf, turnErrorText } from "./AskPage";

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
    const t = (key: string, vars?: Record<string, string | number>) =>
      key === "common.sentenceSeparator" ? " " : `${key}${vars ? JSON.stringify(vars) : ""}`;
    expect(errorMessage(t, new ApiError(500, masked, { code: "failure", error: masked }))).toBe("error.failure");
    expect(errorMessage(t, new ApiError(500, masked, { code: "failure", trace_id: "ab12" }))).toBe('error.failure error.traceId{"trace_id":"ab12"}');
  });
});

describe("the filter box", () => {
  it("sends its lines as typed and shows the server's refusal", async () => {
    const refusal = "invalid filter 'attr_year 2024': expected 'key:value'";
    serveStream([
      { kind: "start", query_id: "q1", trace_id: null, trace_url: null },
      { kind: "error", error_kind: "invalid_filter", message: refusal, query_id: "q1" },
    ]);
    renderAsk();
    await screen.findByText("Who ships parts?");
    const box = screen.getByPlaceholderText("attr_<key>: value");
    fireEvent.change(box, { target: { value: "attr_code: 007\nattr_year 2024" } });
    fireEvent.click(screen.getByText("Who ships parts?"));
    expect(await screen.findByText(refusal)).toBeInTheDocument();
    const calls = (fetch as unknown as { mock: { calls: [string, RequestInit][] } }).mock.calls;
    const [, init] = calls.find(([url]) => String(url).endsWith("/api/ask"))!;
    const body = JSON.parse(String(init.body));
    // The value stays the string "007" (as a number, 7, it matched nothing).
    expect(body.filter_text).toBe("attr_code: 007\nattr_year 2024");
    expect(body).not.toHaveProperty("filters");
  });
});

describe("Show in the graph", () => {
  it("opens the graph of the namespace the answer searched", () => {
    const entity = { kind: "entity", id: "e1" } as unknown as Parameters<typeof graphParams>[0];
    expect(graphParams(entity, "v2")).toEqual({ namespace: "v2", focus: "e1" });
    const community = { kind: "community", community_id: "c7" } as unknown as Parameters<typeof graphParams>[0];
    expect(graphParams(community, "default")).toEqual({ namespace: "default", community: "c7" });
    expect(graphParams(entity, null)).toEqual({ focus: "e1" });
  });
});

describe("the namespace and search type a question sends", () => {
  it("asks the namespace the header picked and leaves the search type to the config", async () => {
    serveStream([{ kind: "start", query_id: "q1", trace_id: null, trace_url: null }]);
    render(
      <I18nProvider initial="en">
        <TrailProvider>
          <AskPage status={{ ...status, search_type: "lexical" } as Status} namespace="v2" />
        </TrailProvider>
      </I18nProvider>,
    );
    fireEvent.click(await screen.findByText("Who ships parts?"));
    const calls = (fetch as unknown as { mock: { calls: [string, RequestInit?][] } }).mock.calls;
    await waitFor(() => expect(calls.some(([url]) => String(url).endsWith("/api/ask"))).toBe(true));
    const [, init] = calls.find(([url]) => String(url).endsWith("/api/ask"))!;
    const body = JSON.parse(String(init!.body));
    expect(body.namespace).toBe("v2");
    // Not "hybrid": the server reads search.search_type.
    expect(body.search_type).toBeNull();
    // The page's other reads are the picked namespace's too.
    expect(calls.some(([url]) => String(url).includes("strategies?namespace=v2"))).toBe(true);
    expect(calls.some(([url]) => String(url).includes("namespace/methodologies?namespace=v2"))).toBe(true);
    // The picker shows what the config searches with until one is picked.
    expect(searchTypeOf(null, "lexical")).toEqual({ value: "lexical", sent: null });
    expect(searchTypeOf("vector", "lexical")).toEqual({ value: "vector", sent: "vector" });
  });
});

describe("the query processing panel", () => {
  it("lists what processing changed or extracted, in the order it ran", () => {
    const rows = processingRows({
      original_query: "and its supplier?",
      standalone_query: "Who supplies Acme Robotics?",
      translated_query: null,
      final_query: "Who supplies Acme Robotics?",
      entities: ["Acme Robotics"],
      hl_keywords: ["supply chain"],
      ll_keywords: [],
    });
    expect(rows.map((r) => r.key)).toEqual(["rewrite", "searched", "entities", "highKeywords"]);
  });

  it("shows nothing for a question searched as asked with nothing extracted", () => {
    expect(processingRows({ original_query: "q", final_query: "q", translated_query: null, entities: [] })).toEqual([]);
  });

  it("is a collapsed section under the answer", async () => {
    serveStream([
      { kind: "start", query_id: "q1", trace_id: null, trace_url: null },
      {
        kind: "result",
        answer: "Umbra Freight.",
        citations: [],
        unknown_citation_markers: [],
        strategy: "local",
        routing: null,
        abstained: false,
        abstention_reason: null,
        blocked: false,
        error: false,
        error_kind: null,
        error_message: null,
        rerank: null,
        usage: null,
        cost_usd: null,
        unpriced_models: [],
        prices_configured: false,
        latency_seconds: 1,
        total_results: 1,
        context_sections_included: 1,
        context_sections_excluded: 0,
        processed_query: { original_query: "Who ships parts?", final_query: "Who ships parts?", translated_query: null, entities: ["Umbra Freight"] },
        sources: [],
        namespace: "default",
        trace_id: null,
        trace_url: null,
      },
    ]);
    renderAsk();
    fireEvent.click(await screen.findByText("Who ships parts?"));
    const summary = await screen.findByText("Query processing");
    expect(summary.closest("details")).not.toHaveAttribute("open");
    expect(screen.getByText("Umbra Freight")).toBeInTheDocument();
  });
});
