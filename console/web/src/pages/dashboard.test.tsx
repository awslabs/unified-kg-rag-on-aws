// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { I18nProvider } from "../i18n/i18n";
import { DashboardPage, rowRequests } from "./DashboardPage";

function serve(refused: number, limited = 0, cancelled = 0, tokensPerQuery: number | null = 10) {
  const totals = { queries: 4, answered: 3, abstained: 0, decline_rate: 0, limited, errors: 0, refused, cancelled, latency_p50: 1, latency_p95: 2, cost_usd: null, priced_queries: 0 };
  const strategy = { strategy: "local", queries: 4 - refused, abstained: 0, limited, errors: 0, cancelled, latency_p50: 1, latency_p95: 2, tokens_per_query: tokensPerQuery, cost_usd: null, cost_per_query: null };
  const body = { totals, strategies: [strategy], models: [], roles: [], timeline: [], jobs: [], unpriced_models: [], prices_configured: false };
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })));
}

afterEach(() => vi.unstubAllGlobals());

describe("DashboardPage", () => {
  it("shows a failed read's error and no loading line under it", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ error: "The usage store could not be read.", code: "store" }), { status: 500 })));
    render(
      <I18nProvider initial="en">
        <DashboardPage />
      </I18nProvider>,
    );
    expect(await screen.findByText(/The usage store could not be read\./)).toBeInTheDocument();
    expect(screen.queryByText(/Loading/)).not.toBeInTheDocument();
  });

  it("says the Questions figure counts the refused ones the strategy rows leave out", async () => {
    serve(1);
    render(
      <I18nProvider initial="en">
        <DashboardPage />
      </I18nProvider>,
    );
    expect(await screen.findByText("Questions counts refused questions too; the strategy rows below do not.")).toBeInTheDocument();
  });

  it("says nothing of it when no question was refused", async () => {
    serve(0);
    render(
      <I18nProvider initial="en">
        <DashboardPage />
      </I18nProvider>,
    );
    await screen.findByText("Refused");
    expect(screen.queryByText(/counts refused questions too/)).not.toBeInTheDocument();
  });

  it("counts declines at a query limit apart from the decline rate (rv42)", async () => {
    serve(0, 2);
    render(
      <I18nProvider initial="ko">
        <DashboardPage />
      </I18nProvider>,
    );
    expect((await screen.findAllByText("한도로 거절")).length).toBeGreaterThan(0);
    expect(screen.getByText(/거절 비율에서 뺍니다/)).toBeInTheDocument();
  });

  it("counts a strategy's declines at a limit apart from its declines (rv43)", async () => {
    serve(0, 3);
    render(
      <I18nProvider initial="en">
        <DashboardPage />
      </I18nProvider>,
    );
    const header = await screen.findByRole("columnheader", { name: "Declined at a limit" });
    const table = header.closest("table") as HTMLTableElement;
    const cells = within(table).getAllByRole("cell").map((c) => c.textContent);
    const columns = within(table).getAllByRole("columnheader").map((c) => c.textContent);
    const at = (name: string) => cells[columns.indexOf(name)];
    expect([at("Declined"), at("Declined at a limit")]).toEqual(["0", "3"]);
  });

  it("shows no such figure when none declined at a limit", async () => {
    serve(0);
    render(
      <I18nProvider initial="en">
        <DashboardPage />
      </I18nProvider>,
    );
    await screen.findByText("Refused");
    expect(screen.queryByText("Declined at a limit")).not.toBeInTheDocument();
  });

  it("counts cancelled questions apart, their spend in the totals only (rv53)", async () => {
    serve(0, 0, 2);
    render(
      <I18nProvider initial="en">
        <DashboardPage />
      </I18nProvider>,
    );
    const labels = await screen.findAllByText("Cancelled");
    const tile = labels.map((l) => l.closest(".stat")).find(Boolean) as HTMLElement;
    expect(within(tile).getByText("2")).toBeInTheDocument();
    expect(screen.getByText(/Cancelled counts the questions left before their answer/)).toBeInTheDocument();
  });

  it("explains a strategy row of cancelled questions: its count, no per-question tokens (rv54)", async () => {
    serve(0, 0, 4, null);
    render(
      <I18nProvider initial="en">
        <DashboardPage />
      </I18nProvider>,
    );
    const header = await screen.findByRole("columnheader", { name: "Cancelled" });
    const table = header.closest("table") as HTMLTableElement;
    const cells = within(table).getAllByRole("cell").map((c) => c.textContent);
    const columns = within(table).getAllByRole("columnheader").map((c) => c.textContent);
    const at = (name: string) => cells[columns.indexOf(name)];
    expect([at("Cancelled"), at("Tokens per question")]).toEqual(["4", "—"]);
  });

  it("shows no cancelled figure when none was cancelled", async () => {
    serve(0);
    render(
      <I18nProvider initial="en">
        <DashboardPage />
      </I18nProvider>,
    );
    await screen.findByText("Refused");
    expect(screen.queryByText("Cancelled")).not.toBeInTheDocument();
  });
});

describe("a usage row", () => {
  it("counts every kind of request, the embedding model's too", () => {
    const row = { llm_calls: 0, input_tokens: 0, output_tokens: 0, cache_hits: 0, failed_calls: 0, rerank_calls: 0, embedding_calls: 3, embedding_input_tokens: 120 };
    expect(rowRequests(row)).toEqual({ calls: 3, inputTokens: 120 });
    expect(rowRequests({ ...row, llm_calls: 2, input_tokens: 10, rerank_calls: 1, embedding_calls: undefined, embedding_input_tokens: undefined })).toEqual({ calls: 3, inputTokens: 10 });
  });
});
