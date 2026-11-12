// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { I18nProvider } from "../i18n/i18n";
import { TrailProvider } from "../lib/trail";
import type { Status } from "../lib/types";
import { AskPage } from "./AskPage";

const status = { config_path: "c.yaml", namespace: "default", jobs: [], sample_questions: ["Who ships parts?"] } as unknown as Status;

function serve(recorded: boolean) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const body = String(url).includes("namespace/methodologies") ? { namespace: "default", recorded, held: recorded ? ["graphrag"] : [], dropped: [] } : { strategies: [] };
      return new Response(JSON.stringify(body), { status: 200 });
    }),
  );
}

const renderAsk = () =>
  render(
    <I18nProvider initial="en">
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
});
