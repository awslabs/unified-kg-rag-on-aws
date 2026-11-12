// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { describe, expect, it } from "vitest";
import { drawnGraph, linkAlpha, placeLabels, type LabelBox } from "../components/GraphCanvas";
import { I18nProvider } from "../i18n/i18n";
import type { GraphData, GraphNode } from "../lib/types";
import { EntityTable, TABLE_PAGE } from "./GraphPage";

const renderIn = (ui: ReactElement) => render(<I18nProvider initial="en">{ui}</I18nProvider>);

function node(i: number, degree = i): GraphNode {
  return { id: `n${i}`, name: `Entity ${i}`, type: "thing", description: null, community_id: null, text_unit_ids: [], source_kinds: [], iri: null, type_iri: null, in_schema: null, degree };
}

/** A graph of `total` entities chained n0-n1-..., the server's layout
 * holding the first `drawn`. */
function graph(total: number, drawn: number): GraphData {
  const nodes = Array.from({ length: total }, (_, i) => node(i, total - i));
  const edges = nodes.slice(1).map((n, i) => ({ id: `e${i}`, source: `n${i}`, target: n.id, type: null, description: null, weight: null, in_schema: null, source_kinds: [] }));
  const layout = Object.fromEntries(nodes.slice(0, drawn).map((n, i) => [n.id, [i, i % 7] as [number, number]]));
  return { nodes, others: [], edges, layout, communities: [], stats: {} as GraphData["stats"] };
}

describe("drawnGraph", () => {
  it("draws only the laid-out nodes and the edges between them", () => {
    const { nodes, links } = drawnGraph(graph(601, 600));
    expect(nodes).toHaveLength(600);
    expect(nodes.some((n) => n.id === "n600")).toBe(false);
    expect(links).toHaveLength(599);
    expect(nodes.every((n) => Number.isFinite(n.x) && Number.isFinite(n.y))).toBe(true);
  });

  it("draws 600 of 20,000", () => {
    expect(drawnGraph(graph(20000, 600)).nodes).toHaveLength(600);
  });
});

describe("EntityTable", () => {
  it("builds no rows until opened, then a page at a time", () => {
    const nodes = graph(250, 0).nodes;
    const { container } = renderIn(<EntityTable nodes={nodes} communityName={(id) => id} onPick={() => undefined} />);
    expect(container.querySelectorAll("tbody tr")).toHaveLength(0);
    const details = container.querySelector("details") as HTMLDetailsElement;
    details.open = true;
    fireEvent(details, new Event("toggle"));
    expect(container.querySelectorAll("tbody tr")).toHaveLength(TABLE_PAGE);
    expect(screen.getByText("1–100 of 250")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(container.querySelectorAll("tbody tr")).toHaveLength(50);
    expect(screen.getByText("201–250 of 250")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
  });

  it("lists the undrawn entities by name and degree too", () => {
    const rows = [node(1, 9), { id: "x", name: "Undrawn", degree: 3 }];
    const { container } = renderIn(<EntityTable nodes={rows} communityName={(id) => id} onPick={() => undefined} />);
    const details = container.querySelector("details") as HTMLDetailsElement;
    details.open = true;
    fireEvent(details, new Event("toggle"));
    const last = container.querySelectorAll("tbody tr")[1] as HTMLElement;
    expect(last.textContent).toBe("Undrawn—3—");
  });
});

describe("graph legibility", () => {
  const box = (id: string, x: number, degree: number, strong = false): LabelBox => ({ id, x, y: 0, width: 40, height: 16, strong, degree });

  it("places labels most connected first, skipping overlaps", () => {
    const placed = placeLabels([box("low", 10, 1), box("high", 0, 9), box("apart", 100, 0)]);
    expect([...placed].sort()).toEqual(["apart", "high"]);
  });

  it("always places the focused label", () => {
    expect(placeLabels([box("hub", 0, 9), box("focus", 10, 1, true)]).has("focus")).toBe(true);
  });

  it("fades edges as they multiply, within bounds", () => {
    expect(linkAlpha(100)).toBe(0.85);
    expect(linkAlpha(300)).toBeCloseTo(0.85);
    expect(linkAlpha(1200)).toBeCloseTo(0.425);
    expect(linkAlpha(1_000_000)).toBe(0.06);
  });
});
