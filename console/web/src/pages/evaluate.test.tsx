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
    { id: "q1", question: "Who ships parts?", ground_truth: "Vendor", scores: { local: 1, mix: 0 }, answers: { local: { text: "Vendor ships them.", error: null, declined: 0, repeats: 1 }, mix: { text: "Buyer.", error: null, declined: 0, repeats: 1 } } },
    { id: "q2", question: "Who pays?", ground_truth: "Buyer", scores: { local: 0, mix: 1 }, answers: { local: { text: "Nobody.", error: null, declined: 0, repeats: 1 }, mix: { text: "Buyer pays.", error: null, declined: 0, repeats: 1 } } },
    { id: "q3", question: "When?", ground_truth: "May", scores: { local: 1 }, answers: { local: { text: "In May.", error: null, declined: 0, repeats: 1 }, mix: { text: "", error: "timed out", declined: 0, repeats: 0 } } },
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

  it("reads a question marked unanswerable by whether each answer declined", async () => {
    const unanswerable: QuestionScores = {
      ...view,
      questions: [
        { id: "q4", question: "Who audits the moon?", ground_truth: "", unanswerable: true, scores: {}, answers: { local: { text: "I cannot tell.", error: null, declined: 1, repeats: 1 }, mix: { text: "The Vendor.", error: null, declined: 0, repeats: 1 } } },
      ],
    };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(unanswerable), { status: 200 })));
    const { container } = renderIn(<QuestionTable refName="outputs:run" candidate="mix" baseline="local" />);
    const details = container.querySelector("details") as HTMLDetailsElement;
    details.open = true;
    fireEvent(details, new Event("toggle"));
    await waitFor(() => expect(screen.getByText("Who audits the moon?")).toBeInTheDocument());
    // The scores' cells say how each did instead of an empty dash.
    const cells = [...container.querySelectorAll("tbody tr td.num")].map((td) => td.textContent);
    expect(cells).toEqual(["declined", "did not decline"]);
    fireEvent.click(screen.getByRole("button", { name: "Who audits the moon?" }));
    expect(screen.getByText("Should decline: the question is marked unanswerable.")).toBeInTheDocument();
    expect(container.querySelectorAll(".answers-side .badge-good")).toHaveLength(1);
    expect(container.querySelectorAll(".answers-side .badge-bad")).toHaveLength(1);
    // Its outcome is read from the declines: mix declined less than local.
    expect(screen.getByRole("button", { name: "Loss 1" })).toBeInTheDocument();
  });

  it("reads declines over every repeat and compares their shares", async () => {
    const repeated: QuestionScores = {
      ...view,
      questions: [
        { id: "q5", question: "Who audits the sun?", ground_truth: "", unanswerable: true, scores: {}, answers: { local: { text: "The Vendor.", error: null, declined: 1, repeats: 3 }, mix: { text: "The Buyer.", error: null, declined: 2, repeats: 3 } } },
      ],
    };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(repeated), { status: 200 })));
    const { container } = renderIn(<QuestionTable refName="outputs:run" candidate="mix" baseline="local" />);
    const details = container.querySelector("details") as HTMLDetailsElement;
    details.open = true;
    fireEvent(details, new Event("toggle"));
    await waitFor(() => expect(screen.getByText("Who audits the sun?")).toBeInTheDocument());
    const cells = [...container.querySelectorAll("tbody tr td.num")].map((td) => td.textContent);
    expect(cells).toEqual(["declined 1/3", "declined 2/3"]);
    expect(screen.getByRole("button", { name: "Win 1" })).toBeInTheDocument();
  });
});
