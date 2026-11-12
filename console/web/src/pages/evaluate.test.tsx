// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { higherIsBetter, MetricName } from "../components/charts";
import { I18nProvider } from "../i18n/i18n";
import type { QuestionScores } from "../lib/types";
import { outcomeOf, QuestionTable } from "./EvaluatePage";

const renderIn = (ui: ReactElement) => render(<I18nProvider initial="en">{ui}</I18nProvider>);

const view: QuestionScores = {
  metric: "answer_match/answer_contains",
  strategies: ["local", "mix"],
  questions: [
    { id: "q1", question: "Who ships parts?", ground_truth: "Vendor", scores: { local: 1, mix: 0 }, answers: { local: { text: "Vendor ships them.", error: null, abstained: false }, mix: { text: "Buyer.", error: null, abstained: false } } },
    { id: "q2", question: "Who pays?", ground_truth: "Buyer", scores: { local: 0, mix: 1 }, answers: { local: { text: "Nobody.", error: null, abstained: false }, mix: { text: "Buyer pays.", error: null, abstained: false } } },
    { id: "q3", question: "When?", ground_truth: "May", scores: { local: 1 }, answers: { local: { text: "In May.", error: null, abstained: false }, mix: { text: "", error: "timed out", abstained: false } } },
  ],
};

afterEach(() => vi.unstubAllGlobals());

describe("metric directions", () => {
  it("knows the rates where lower is better and the diagnostics", () => {
    expect(higherIsBetter("answer_match/token_f1")).toBe(true);
    expect(higherIsBetter("assertion/assertion_contradiction_rate")).toBe(false);
    expect(higherIsBetter("assertion/negative_assertion_violation_rate")).toBe(false);
    expect(higherIsBetter("retrieval/attributable_fraction")).toBeNull();
    expect(higherIsBetter("usage/cost_usd")).toBe(false);
  });

  it("describes a metric and its direction on hover", () => {
    const { container } = renderIn(<MetricName metric="retrieval/mrr" />);
    const name = container.querySelector(".metric-name") as HTMLElement;
    expect(name.title).toBe("Reciprocal rank of the first reference file among the reported sources. Higher is better.");
    expect(within(name).getByText("↑")).toBeInTheDocument();
  });

  it("calls a question's outcome by the metric's direction", () => {
    expect(outcomeOf("answer_match/token_f1", 0.5, 0.2)).toBe("win");
    expect(outcomeOf("assertion/assertion_contradiction_rate", 0.5, 0.2)).toBe("loss");
    expect(outcomeOf("answer_match/token_f1", 0.2, 0.2)).toBe("tie");
    expect(outcomeOf("answer_match/token_f1", undefined, 0.2)).toBeNull();
  });
});

describe("QuestionTable", () => {
  it("reads the questions once opened, filters by outcome and opens the answers side by side", async () => {
    const fetched = vi.fn(async (_url: string) => new Response(JSON.stringify(view), { status: 200 }));
    vi.stubGlobal("fetch", fetched);
    const { container } = renderIn(<QuestionTable refName="outputs:run" candidate="mix" baseline="local" />);
    expect(fetched).not.toHaveBeenCalled();
    const details = container.querySelector("details") as HTMLDetailsElement;
    details.open = true;
    fireEvent(details, new Event("toggle"));
    await waitFor(() => expect(screen.getByText("Who ships parts?")).toBeInTheDocument());
    expect(String(fetched.mock.calls[0]?.[0])).toContain("evaluations/questions?ref=outputs%3Arun");
    fireEvent.click(screen.getByRole("button", { name: "Loss 1" }));
    expect(screen.getByText("Who ships parts?")).toBeInTheDocument();
    expect(screen.queryByText("Who pays?")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "All 3" }));
    fireEvent.click(screen.getByRole("button", { name: "When?" }));
    expect(screen.getByText("In May.")).toBeInTheDocument();
    expect(screen.getByText("The answer failed: timed out")).toBeInTheDocument();
    expect(screen.getByText("May")).toBeInTheDocument();
  });
});
