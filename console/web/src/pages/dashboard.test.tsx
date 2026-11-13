// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { I18nProvider } from "../i18n/i18n";
import { DashboardPage } from "./DashboardPage";

function serve(refused: number) {
  const totals = { queries: 4, answered: 3, abstained: 0, decline_rate: 0, errors: 0, refused, latency_p50: 1, latency_p95: 2, cost_usd: null, priced_queries: 0 };
  const strategy = { strategy: "local", queries: 4 - refused, abstained: 0, errors: 0, latency_p50: 1, latency_p95: 2, tokens_per_query: 10, cost_usd: null, cost_per_query: null };
  const body = { totals, strategies: [strategy], models: [], roles: [], timeline: [], jobs: [], unpriced_models: [], prices_configured: false };
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })));
}

afterEach(() => vi.unstubAllGlobals());

describe("DashboardPage", () => {
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
});
