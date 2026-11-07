// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { ReactElement } from "react";
import { describe, expect, it, vi } from "vitest";
import { I18nProvider } from "../i18n/i18n";
import { fieldsOf, parseValue, type JsonSchema } from "../lib/schema";
import { answerSegments, markerNumbers } from "../lib/citations";
import type { AskResult, Citation, Comparison, Delta, Multiplicity, Source, StrategyInfo } from "../lib/types";
import { conversationHistory, parseFilters } from "../pages/AskPage";
import { intervalResamples } from "../pages/EvaluatePage";
import { AnswerText, CitationList } from "./AnswerText";
import { DeltaPlot, deltaInterval, deltaVerdict, higherIsBetter } from "./charts";
import { citedBy, EvidenceTrace, excerptBody } from "./EvidenceTrace";
import { queryTerms } from "./Highlighted";
import { StrategyPicker } from "./StrategyPicker";

function source(rank: number, kind: string, extra: Partial<Source> = {}): Source {
  return {
    rank,
    kind,
    id: `id-${rank}`,
    title: kind === "text" ? `doc-${rank}.txt` : `Source ${rank}`,
    score: 0.5,
    excerpt: `Excerpt ${rank}`,
    truncated: false,
    document_ids: [],
    file_name: kind === "text" ? `doc-${rank}.txt` : null,
    chunk_id: kind === "text" ? `id-${rank}` : null,
    text_unit_ids: [],
    source_kinds: [],
    iri: null,
    provenance: null,
    marker: null,
    ...extra,
  };
}

const sources = [
  source(1, "entity", { title: "Vendor", supported_by: [2], unread_passages: 3, text_unit_ids: ["id-2", "x", "y", "z"], node_ids: ["id-1"] }),
  source(2, "text"),
  source(3, "relationship", { title: "Vendor → Buyer", supported_by: [2], relation: "SUPPLIES" }),
  source(4, "entity", {
    title: "Depot",
    provenance: { kind: "seed_graph", seed_graphs: ["registry"], iri: "https://example.org/Depot" },
  }),
];

const renderIn = (ui: ReactElement) => render(<I18nProvider initial="en">{ui}</I18nProvider>);

describe("EvidenceTrace", () => {
  it("lists every source on the rail, filterable by kind", () => {
    renderIn(<EvidenceTrace sources={sources} question="What does Vendor supply?" />);
    expect(screen.getByText("The answer read 4 sources")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^Passage \d/ }));
    expect(screen.queryByText("Vendor")).not.toBeInTheDocument();
    expect(screen.getByText("doc-2.txt")).toBeInTheDocument();
  });

  it("opens a graph source's trail to the passages that state it", () => {
    const open = vi.fn();
    const { container } = renderIn(<EvidenceTrace sources={sources} question="Vendor" onOpenInGraph={open} />);
    fireEvent.click(screen.getByText("Vendor").closest("button") as HTMLElement);
    const trail = container.querySelector(".source-trail") as HTMLElement;
    expect(within(trail).getByText("Stated in passages")).toBeInTheDocument();
    expect(within(trail).getByText("and 3 passages the answer did not read")).toBeInTheDocument();
    expect(container.querySelector('[data-rank="2"]')).toHaveClass("linked");
    fireEvent.click(within(trail).getByRole("button", { name: "Show in the graph" }));
    expect(open).toHaveBeenCalledWith(sources[0]);
    // Following the passage link opens the passage, which names its citers.
    fireEvent.click(within(trail).getByRole("button", { name: "#2" }));
    const passage = container.querySelector('[data-rank="2"] .source-trail') as HTMLElement;
    expect(within(passage).getByText("doc-2.txt")).toBeInTheDocument();
    expect(within(passage).getByRole("button", { name: "#1" })).toBeInTheDocument();
  });

  it("shows seed-graph provenance and the IRI", () => {
    renderIn(<EvidenceTrace sources={sources} question="" />);
    fireEvent.click(screen.getByText("Depot").closest("button") as HTMLElement);
    expect(screen.getByText("registry")).toBeInTheDocument();
    expect(screen.getByText("https://example.org/Depot")).toBeInTheDocument();
    expect(screen.getByText("seed graph only")).toBeInTheDocument();
  });

  it("says when the answer read nothing", () => {
    renderIn(<EvidenceTrace sources={[]} question="" />);
    expect(screen.getByText("No sources")).toBeInTheDocument();
  });

  it("shows ten sources first", () => {
    const many = Array.from({ length: 14 }, (_, i) => source(i + 1, "text"));
    renderIn(<EvidenceTrace sources={many} question="" />);
    expect(screen.queryByText("doc-11.txt")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show all 14 sources" }));
    expect(screen.getByText("doc-14.txt")).toBeInTheDocument();
  });

  it("links passages back to their citers and drops repeated headers", () => {
    expect(citedBy(sources, 2)).toEqual([1, 3]);
    expect(excerptBody(source(9, "entity", { excerpt: "Entity: X\nType: org\nDescription: X ships." }))).toBe("X ships.");
  });
});

describe("StrategyPicker", () => {
  const strategies: StrategyInfo[] = [
    { name: "auto", methodology: null, routable: false, available: true, reason: null },
    { name: "local", methodology: "graphrag", routable: true, available: true, reason: null },
    { name: "mix", methodology: "lightrag", routable: true, available: true, reason: null },
    { name: "tree", methodology: "raptor", routable: false, available: false, reason: "not indexed" },
  ];
  it("disables what the index cannot serve and says why", () => {
    const change = vi.fn();
    renderIn(<StrategyPicker strategies={strategies} value={["auto"]} onChange={change} />);
    expect(screen.getByRole("button", { name: "tree" })).toBeDisabled();
    expect(screen.getByText(/picks one of local, mix/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "mix" }));
    expect(change).toHaveBeenCalledWith(["mix"]);
  });
  it("picks several in order for an evaluation", () => {
    const change = vi.fn();
    renderIn(<StrategyPicker strategies={strategies} value={["local"]} onChange={change} multiple />);
    fireEvent.click(screen.getByRole("button", { name: "mix" }));
    expect(change).toHaveBeenCalledWith(["local", "mix"]);
    fireEvent.click(screen.getByRole("button", { name: "local" }));
    expect(change).toHaveBeenCalledWith([]);
  });
});

describe("helpers", () => {
  it("marks words worth marking, CJK by bigrams", () => {
    expect(queryTerms("Who carries the units to Kestrel?")).toEqual(["carries", "kestrel", "units"]);
    expect(queryTerms("케스트럴 창고")).toEqual(["케스", "스트", "트럴", "창고"]);
    expect(queryTerms("in 2026")).toEqual(["2026"]);
  });
  it("calls a delta better or worse only outside zero, by direction", () => {
    const delta = { pairs: 5, mean_delta: 0.1, ci_low: 0.05, ci_high: 0.2, wins: 3, ties: 2, losses: 0 };
    expect(deltaVerdict("answer_match/token_f1", delta)).toBe("better");
    expect(deltaVerdict("latency/response_time_seconds", delta)).toBe("worse");
    expect(deltaVerdict("answer_match/token_f1", { ...delta, ci_low: -0.1 })).toBe("neutral");
    expect(higherIsBetter("graph_aware/answer_length")).toBeNull();
  });
  it("calls a delta by its Holm-adjusted p-value when the comparison has one", () => {
    // The plain interval excludes zero, but over the family it does not count.
    const delta: Delta = { pairs: 40, mean_delta: 0.04, ci_low: 0.01, ci_high: 0.07, wins: 9, ties: 28, losses: 3, p_value: 0.01, family: "secondary", adjusted_p_value: 0.2, adjusted_ci_low: -0.01, adjusted_ci_high: 0.09 };
    expect(deltaVerdict("answer_match/token_f1", delta, 0.05)).toBe("neutral");
    expect(deltaVerdict("answer_match/token_f1", { ...delta, adjusted_p_value: 0.01 }, 0.05)).toBe("better");
    expect(deltaVerdict("usage/total_tokens", { ...delta, adjusted_p_value: 0.01 }, 0.05)).toBe("worse");
    // An older comparison (no alpha) or a delta without one reads its interval.
    expect(deltaVerdict("answer_match/token_f1", delta)).toBe("better");
    expect(deltaVerdict("answer_match/token_f1", { ...delta, adjusted_p_value: undefined }, 0.05)).toBe("better");
    const multiplicity: Multiplicity = { method: "holm", alpha: 0.05, primary_metric: "answer_match/answer_contains", families: { secondary: { tests: 10, interval_confidence: 0.995 } } };
    expect(deltaInterval(delta, multiplicity, 0.95)).toEqual({ low: -0.01, high: 0.09, confidence: 0.995, adjusted: true });
    expect(deltaInterval(delta, null, 0.95)).toEqual({ low: 0.01, high: 0.07, confidence: 0.95, adjusted: false });
  });
  it("names the resamples the shown intervals read: the Holm families' own", () => {
    const comparison = {
      bootstrap: { resamples: 2000 },
      multiplicity: { method: "holm", alpha: 0.05, primary_metric: null, families: { primary: { tests: 3, interval_confidence: 0.98, resamples: 2000 }, secondary: { tests: 120, interval_confidence: 0.9996, resamples: 48000 } }, unadjusted: ["closed_book"] },
    } as unknown as Comparison;
    const plain = (n: number) => String(n);
    expect(intervalResamples(comparison, "mix", plain)).toBe("2000-48000");
    expect(intervalResamples(comparison, "closed_book", plain)).toBe("2000");
    expect(intervalResamples({ ...comparison, multiplicity: undefined }, "mix", plain)).toBe("2000");
  });
  it("shows the Holm p-value, the simultaneous interval and the primary metric", () => {
    const multiplicity: Multiplicity = {
      method: "holm",
      alpha: 0.05,
      primary_metric: "answer_match/answer_contains",
      families: { primary: { tests: 1, interval_confidence: 0.95 }, secondary: { tests: 10, interval_confidence: 0.995 } },
    };
    const base = { pairs: 40, wins: 9, ties: 28, losses: 3, p_value: 0.01 };
    const deltas: [string, Delta][] = [
      ["answer_match/answer_contains", { ...base, mean_delta: 0.1, ci_low: 0.02, ci_high: 0.18, family: "primary", adjusted_p_value: 0.01, adjusted_ci_low: 0.02, adjusted_ci_high: 0.18 }],
      ["answer_match/token_f1", { ...base, mean_delta: 0.04, ci_low: 0.01, ci_high: 0.07, family: "secondary", adjusted_p_value: 0.2, adjusted_ci_low: -0.01, adjusted_ci_high: 0.09 }],
      ["abstention/precision", { ...base, mean_delta: 0.1, ci_low: 0.05, ci_high: 0.2 }],
    ];
    renderIn(<DeltaPlot deltas={deltas} multiplicity={multiplicity} />);
    const rows = screen.getAllByRole("row");
    expect(screen.getByRole("columnheader", { name: "Holm p" })).toBeInTheDocument();
    const primary = rows.find((r) => r.textContent?.includes("answer_contains"))!;
    expect(within(primary).getByText("primary")).toBeInTheDocument();
    expect(within(primary).getByText("better")).toBeInTheDocument();
    const secondary = rows.find((r) => r.textContent?.includes("token_f1"))!;
    expect(within(secondary).queryByText("better")).toBeNull();
    expect(secondary.textContent).toContain("[-0.01, 0.09]");
    expect(secondary.textContent).toContain("99.5%");
    expect(secondary.textContent).toContain("0.2");
    const ratio = rows.find((r) => r.textContent?.includes("abstention/precision"))!;
    expect(ratio.textContent).toContain("unadjusted");
    expect(within(ratio).getByText("better")).toBeInTheDocument();
  });
  it("types filter values", () => {
    expect(parseFilters("file_name: a.pdf\nin_schema: false\nyear: 2026\n\nbad")).toEqual({ file_name: "a.pdf", in_schema: false, year: 2026 });
    expect(parseFilters("")).toBeNull();
  });
  it("reads the config schema's fields and values", () => {
    const schema: JsonSchema = {
      properties: { search: { $ref: "#/$defs/Search", description: "Search" }, key: { anyOf: [{ type: "string", format: "password" }, { type: "null" }] } },
      $defs: { Search: { properties: { top_k: { type: "integer", default: 10, description: "Results" }, mode: { enum: ["a", "b"] } } } },
    };
    const [search, key] = fieldsOf(schema);
    expect(search?.kind).toBe("section");
    expect(search?.children.map((f) => [f.path, f.kind])).toEqual([["search.top_k", "integer"], ["search.mode", "enum"]]);
    expect(key?.kind).toBe("secret");
    const topK = search!.children[0]!;
    expect(parseValue(topK, "12")).toEqual({ value: 12 });
    expect(parseValue(topK, "1.5").error).toBe("settings.errors.integer");
  });
});

describe("upload batches", () => {
  it("stay under the file and size limits", async () => {
    const { batches } = await import("../pages/IngestPage");
    const files = Array.from({ length: 1201 }, () => ({ size: 1 }));
    expect(batches(files).map((b) => b.length)).toEqual([500, 500, 201]);
    const big = [{ size: 80 * 2 ** 20 }, { size: 80 * 2 ** 20 }, { size: 1 }];
    expect(batches(big).map((b) => b.length)).toEqual([1, 2]);
    expect(batches([])).toEqual([]);
  });
});

function citation(marker: number, extra: Partial<Citation> = {}): Citation {
  return {
    marker,
    cited: true,
    section_type: "text",
    source_id: `id-${marker}`,
    title: null,
    document_id: null,
    file_name: `doc-${marker}.txt`,
    file_path: null,
    page: null,
    page_end: null,
    chunk_id: null,
    text_unit_ids: [],
    documents: [],
    iri: null,
    source_rank: marker,
    ...extra,
  };
}

describe("citation markers", () => {
  it("reads the server's marker grammar", () => {
    expect(markerNumbers("[1, 3]")).toEqual([1, 3]);
    expect(markerNumbers("[2-4]")).toEqual([2, 3, 4]);
    expect(markerNumbers("［２］")).toEqual([2]);
    expect(markerNumbers("【1、2】")).toEqual([1, 2]);
    // A reversed or long range is its two ends, not every number between.
    expect(markerNumbers("[5-2]")).toEqual([5, 2]);
    expect(markerNumbers("[1-500]")).toEqual([1, 500]);
  });

  it("cuts an answer into text and markers; an open marker stays text", () => {
    expect(answerSegments("Vendor ships [1][3].")).toEqual([
      { text: "Vendor ships " },
      { marker: "[1]", numbers: [1] },
      { marker: "[3]", numbers: [3] },
      { text: "." },
    ]);
    // A bracketed year is text (four digits), as is a marker still streaming.
    expect(answerSegments("In [2020] it shipped [1")).toEqual([{ text: "In [2020] it shipped [1" }]);
  });
});

describe("AnswerText", () => {
  it("links each cited marker to its source; an unknown one is marked", () => {
    const opened = vi.fn();
    renderIn(<AnswerText text="Vendor ships [1, 9] and stores [2]." citations={[citation(1), citation(2, { source_rank: null })]} onOpen={opened} />);
    fireEvent.click(screen.getByRole("button", { name: "Open source [1] in the provenance trail: doc-1.txt" }));
    expect(opened).toHaveBeenCalledWith(expect.objectContaining({ marker: 1 }));
    expect(screen.getByTitle("[9] is not a section of this answer's context")).toHaveClass("cite-unknown");
    // A summary the model read has no source to open.
    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(screen.getByText("[2]")).toHaveClass("cite");
  });

  it("shows a streaming answer's markers unlinked", () => {
    renderIn(<AnswerText text="Vendor ships [1]" />);
    expect(screen.getByText("[1]")).toHaveClass("cite-pending");
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("lists the cited sources with their pages", () => {
    const opened = vi.fn();
    renderIn(
      <CitationList
        citations={[citation(1, { file_path: "docs/manual.pdf", page: 3, page_end: 4 }), citation(2, { cited: false })]}
        onOpen={opened}
      />,
    );
    expect(screen.getByText("Cited in this answer")).toBeInTheDocument();
    expect(screen.getByText("docs/manual.pdf")).toBeInTheDocument();
    expect(screen.getByText(/p\. 3–4/)).toBeInTheDocument();
    expect(screen.queryByText("doc-2.txt")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button"));
    expect(opened).toHaveBeenCalledWith(expect.objectContaining({ marker: 1 }));
  });
});

describe("EvidenceTrace markers", () => {
  it("opens the source the page names and badges the cited markers", () => {
    const marked = [source(1, "text", { marker: 2 }), source(2, "text", { marker: 1 })];
    const { container } = renderIn(<EvidenceTrace sources={marked} question="" cited={new Set([2])} open={1} />);
    expect(container.querySelector('[data-rank="1"]')).toHaveClass("open");
    expect(screen.getByTitle("Context marker [2], cited by the answer")).toHaveClass("marker-cited");
    expect(screen.getByTitle("Context marker [1]")).not.toHaveClass("marker-cited");
  });
});

describe("conversationHistory", () => {
  const result = (answer: string, extra: Partial<AskResult> = {}) => ({ answer, blocked: false, error: false, ...extra }) as AskResult;
  it("carries each answered question and its final answer, oldest first", () => {
    expect(
      conversationHistory([
        { question: "Who ships?", status: "done", result: result("Vendor [1].") },
        { question: "Blocked?", status: "done", result: result("Refused.", { blocked: true }) },
        { question: "Stopped?", status: "stopped" },
        { question: "Where to?", status: "done", result: result("The depot [2].") },
      ]),
    ).toEqual([
      { role: "user", content: "Who ships?" },
      { role: "assistant", content: "Vendor [1]." },
      { role: "user", content: "Where to?" },
      { role: "assistant", content: "The depot [2]." },
    ]);
  });
});

describe("graph query result", () => {
  it("shows the table whole and its trail: query, rows, passages", () => {
    const result = source(1, "query_result", {
      title: "Count suppliers per depot",
      excerpt: "depot | suppliers\nKestrel | 2",
      operation: "aggregate",
      row_count: 1,
      exact: true,
      backend: "sqlite",
      node_ids: ["e-1", "e-2"],
      supported_by: [2],
    });
    const opened = vi.fn();
    const { container } = renderIn(<EvidenceTrace sources={[result, source(2, "text")]} question="" onOpenInGraph={opened} />);
    expect(container.querySelector(".query-table")?.textContent).toBe("depot | suppliers\nKestrel | 2");
    fireEvent.click(screen.getByText("Count suppliers per depot").closest("button") as HTMLElement);
    const trail = container.querySelector(".source-trail") as HTMLElement;
    expect(within(trail).getByText("aggregate")).toBeInTheDocument();
    expect(within(trail).getByText("exact")).toBeInTheDocument();
    expect(within(trail).getByRole("button", { name: "#2" })).toBeInTheDocument();
    fireEvent.click(within(trail).getByRole("button", { name: "Show in the graph" }));
    expect(opened).toHaveBeenCalledWith(result);
  });
});
