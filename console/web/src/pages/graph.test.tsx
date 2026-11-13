// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { drawnGraph, linkAlpha, placeLabels, settle, type LabelCandidate } from "../components/GraphCanvas";
import { I18nProvider } from "../i18n/i18n";
import { TrailProvider } from "../lib/trail";
import type { GraphData, GraphNode } from "../lib/types";
import { EntityTable, GraphPage, rankMatches, TABLE_PAGE } from "./GraphPage";

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

describe("settle", () => {
  // A ring of 12 entities laid out in the middle and 4 islands the server
  // packed around it, as the server's packed layout places them.
  function islands(): GraphData {
    const g = graph(16, 16);
    g.edges = [...g.edges.slice(0, 11), { id: "loop", source: "n11", target: "n0", type: null, description: null, weight: null, in_schema: null, source_kinds: [] }];
    const corners: [number, number][] = [[-300, -300], [300, -300], [-300, 300], [300, 300]];
    g.nodes.forEach((n, i) => {
      const angle = (i / 12) * 2 * Math.PI;
      g.layout[n.id] = i < 12 ? [150 * Math.cos(angle), 150 * Math.sin(angle)] : (corners[i - 12] as [number, number]);
    });
    return g;
  }
  const span = (ns: { x?: number; y?: number }[]) => {
    const xs = ns.map((n) => n.x ?? 0);
    const ys = ns.map((n) => n.y ?? 0);
    return Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  };

  it("keeps the islands beside the main component", () => {
    const { nodes } = settle(drawnGraph(islands()));
    // The main component fills most of the frame instead of shrinking
    // while the islands drift to its corners.
    // (Without the gravity: 0.59.)
    expect(span(nodes.slice(0, 12)) / span(nodes)).toBeGreaterThan(0.75);
  });

  it("starts warm from the server's layout", () => {
    const start = drawnGraph(islands()).nodes;
    const { nodes } = settle(drawnGraph(islands()));
    const moved = nodes.map((n, i) => Math.hypot((n.x ?? 0) - (start[i]?.x ?? 0), (n.y ?? 0) - (start[i]?.y ?? 0)));
    // Nodes only relax (a mean of about 60 px in the 600 px frame; a cold
    // start moves them about 80).
    expect(moved.reduce((sum, d) => sum + d, 0) / moved.length).toBeLessThan(70);
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
  // A node at (cx, 0) of radius 4 with a 40×16 label.
  const label = (id: string, cx: number, degree: number, strong = false): LabelCandidate => ({ id, cx, cy: 0, r: 4, width: 40, height: 16, strong, degree });
  const ids = (placed: { id: string }[]) => placed.map((p) => p.id).sort();

  it("places labels most connected first, on another side before skipping", () => {
    // "low" sits 10 px right of "high": its right side overlaps nothing.
    const placed = placeLabels([label("low", 10, 1), label("high", 0, 9), label("apart", 200, 0)]);
    expect(ids(placed)).toEqual(["apart", "high", "low"]);
    const high = placed.find((p) => p.id === "high");
    const low = placed.find((p) => p.id === "low");
    expect(high?.x).toBe(6);
    // The right of "low" overlaps "high"'s label: it goes to the left
    // (it used to be dropped).
    expect(low?.x).toBe(10 - 4 - 2 - 40);
  });

  it("skips a label only when all four sides are taken", () => {
    const wall: LabelCandidate = { ...label("wall", -100, 9), width: 400, height: 400 };
    expect(ids(placeLabels([wall, label("hidden", 0, 1)]))).toEqual(["wall"]);
  });

  it("stops at the density bound, but always places the strong ones", () => {
    const many = Array.from({ length: 10 }, (_, i) => label(`n${i}`, i * 100, 10 - i));
    expect(placeLabels(many, { area: 40 * 16 * 3 })).toHaveLength(3);
    const placed = placeLabels([...many, label("focus", 2000, 0, true)], { area: 40 * 16 * 3 });
    expect(ids(placed)).toContain("focus");
    expect(placed).toHaveLength(3);
  });

  it("keeps labels inside the canvas and off the drawn nodes", () => {
    // At the right edge: its right side would be cut, so it goes left.
    const [edge] = placeLabels([{ ...label("edge", 280, 1), cy: 50 }], { width: 300, height: 100, discs: [] });
    expect(edge?.x).toBe(280 - 4 - 2 - 40);
    // A node right of "hub" is under its right side: the label goes left.
    const discs = [
      { id: "hub", cx: 100, cy: 50, r: 4 },
      { id: "near", cx: 125, cy: 50, r: 4 },
    ];
    const [hub] = placeLabels([{ ...label("hub", 100, 9), cy: 50 }], { width: 300, height: 100, discs });
    expect(hub?.x).toBe(100 - 4 - 2 - 40);
    // No side inside the canvas and clear: skipped, unless strong.
    const cornered = { ...label("cornered", 2, 1), cy: 2 };
    expect(placeLabels([cornered], { width: 10, height: 10 })).toEqual([]);
    expect(ids(placeLabels([{ ...cornered, strong: true }], { width: 10, height: 10 }))).toEqual(["cornered"]);
  });

  it("names a hub over its leaves when no side is clear of nodes", () => {
    // Leaves on all four sides of a hub (rv27: a dense centre's hubs got no
    // label). Its label covers only less connected nodes.
    const ring = [
      [125, 50],
      [75, 50],
      [100, 30],
      [100, 70],
    ].map(([cx, cy], i) => ({ id: `leaf${i}`, cx: cx as number, cy: cy as number, r: 4, degree: 1 }));
    const discs = [{ id: "hub", cx: 100, cy: 50, r: 4, degree: 9 }, ...ring];
    const [hub] = placeLabels([{ ...label("hub", 100, 9), cy: 50 }], { width: 300, height: 100, discs });
    expect(hub?.x).toBe(100 + 4 + 2);
    // A leaf's label does not cover the hub, nor another leaf.
    const leaf = { ...label("leaf1", 75, 1), cy: 50 };
    const crowded = [...discs, { id: "rival", cx: 50, cy: 50, r: 4, degree: 1 }, { id: "up", cx: 75, cy: 30, r: 4, degree: 1 }, { id: "down", cx: 75, cy: 70, r: 4, degree: 1 }];
    expect(placeLabels([leaf], { width: 300, height: 100, discs: crowded })).toEqual([]);
  });

  it("always places the focused label", () => {
    expect(ids(placeLabels([label("hub", 0, 9), label("focus", 0, 1, true)]))).toContain("focus");
  });

  it("fades edges as they multiply, within bounds", () => {
    expect(linkAlpha(100)).toBe(0.85);
    expect(linkAlpha(300)).toBeCloseTo(0.85);
    expect(linkAlpha(1200)).toBeCloseTo(0.425);
    expect(linkAlpha(1_000_000)).toBe(0.06);
  });
});

describe("rankMatches", () => {
  it("puts the exact name first, then prefixes, then by connections", () => {
    const entities = [
      { id: "a", name: "Entity 779", degree: 40 },
      { id: "b", name: "Entity 77", degree: 2 },
      { id: "c", name: "Big Entity 77", degree: 90 },
      { id: "d", name: "entity 770", degree: 50 },
    ];
    expect(rankMatches(entities, "Entity 77").map((e) => e.id)).toEqual(["b", "d", "a", "c"]);
    expect(rankMatches(entities, "  ")).toEqual([]);
  });
});

describe("GraphPage without a graph", () => {
  const serve = (current: { state: string; namespace: string; built_at: null }, indexed: boolean) =>
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        const body = String(url).includes("namespace/methodologies") ? { namespace: "default", recorded: false, indexed, held: [], dropped: [] } : { graphs: [], current };
        return new Response(JSON.stringify(body), { status: 200 });
      }),
    );
  afterEach(() => vi.unstubAllGlobals());

  it("sends an off export to Settings, not to Ingest", async () => {
    serve({ state: "export_off", namespace: "default", built_at: null }, true);
    renderIn(<GraphPage params={new URLSearchParams()} />);
    expect(await screen.findByText("The graph is turned off")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open Settings" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Go to Ingest" })).not.toBeInTheDocument();
  });

  it("tells an empty namespace from one indexed before build records", async () => {
    serve({ state: "unrecorded", namespace: "default", built_at: null }, false);
    const { unmount } = renderIn(<GraphPage params={new URLSearchParams()} />);
    expect(await screen.findByText("Nothing is indexed in “default” yet")).toBeInTheDocument();
    unmount();
    serve({ state: "unrecorded", namespace: "default", built_at: null }, true);
    renderIn(<GraphPage params={new URLSearchParams()} />);
    expect(await screen.findByText("“default” has no build record")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Go to Ingest" })).toBeInTheDocument();
  });
});

describe("GraphPage when building the graph failed", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("says so over the older export it shows, marked outdated", async () => {
    const older = { ref: "cache:a/visualization/visualization_data.json", label: "default", namespace: "default", run: "a", built_at: "2026-11-13T00:00:00Z", modified: 1, bytes: 10, outdated: true };
    // The export's data never arrives: the notice and the picker are the
    // page's, drawn before it (the explorer and its canvas are tested apart).
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) =>
        String(url).includes("graphs/data")
          ? new Promise<Response>(() => undefined)
          : Promise.resolve(new Response(JSON.stringify({ graphs: [older], current: { state: "failed", namespace: "default", built_at: "2026-11-15T00:00:00Z" } }), { status: 200 })),
      ),
    );
    renderIn(
      <TrailProvider>
        <GraphPage params={new URLSearchParams()} />
      </TrailProvider>,
    );
    expect(await screen.findByText("The graph could not be built")).toBeInTheDocument();
    // The build time and the mark first: a narrow select cuts the end.
    expect(screen.getByRole("option", { name: /^[^·]+ \(outdated\) · default · a$/ })).toBeInTheDocument();
  });

  it.each([
    ["read", /Reading the namespace's stores failed/],
    ["write", /Writing the graph's export folder failed/],
    ["config", /The config could not be read/],
  ])("names what failed: %s", async (reason, text) => {
    const older = { ref: "cache:a/visualization/visualization_data.json", label: "default", namespace: "default", run: "a", built_at: "2026-11-13T00:00:00Z", modified: 1, bytes: 10, outdated: true };
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) =>
        String(url).includes("graphs/data")
          ? new Promise<Response>(() => undefined)
          : Promise.resolve(new Response(JSON.stringify({ graphs: [older], current: { state: "failed", namespace: "default", built_at: null, reason } }), { status: 200 })),
      ),
    );
    renderIn(
      <TrailProvider>
        <GraphPage params={new URLSearchParams()} />
      </TrailProvider>,
    );
    expect(await screen.findByText(text)).toBeInTheDocument();
  });

  it("names it without an export to show, too", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response(JSON.stringify({ graphs: [], current: { state: "failed", namespace: "default", built_at: null, reason: "write" } }), { status: 200 }))),
    );
    renderIn(<GraphPage params={new URLSearchParams()} />);
    expect(await screen.findByText(/Writing the graph's export folder failed/)).toBeInTheDocument();
  });
});
