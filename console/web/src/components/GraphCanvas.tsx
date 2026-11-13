// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * The graph as a live force-directed canvas (react-force-graph-2d), started
 * from the exported layout. Nodes are sized by their connections and
 * coloured by community or type; hovering or selecting one brings it and its
 * neighbours forward (labels, edges, arrows) and dims the rest; hovering an
 * edge names it. A traced node (from an answer's trail) wears a dashed ring
 * and its traced edge carries particles. Every colour comes from the page's
 * theme tokens, read again when the theme changes. Pan with a drag or the
 * arrow keys, zoom with the wheel, +/- or the buttons, 0 fits; Escape clears.
 */
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY, type SimulationLinkDatum, type SimulationNodeDatum } from "d3-force";
import ForceGraph2D, { type ForceGraphMethods, type LinkObject, type NodeObject } from "react-force-graph-2d";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n/i18n";
import type { GraphData } from "../lib/types";

// The forces: room for labels between neighbours, and a weak pull toward
// the centre (d3's remedy for a disconnected graph), so the islands the
// server packs around the main component stay beside it rather than drift
// to the corners.
const CHARGE = -260;
const DISTANCE = 78;
const GRAVITY = 0.08;
// The pre-layout warm-starts from the server's packed layout: a low alpha
// and few ticks only clear overlaps and relax the edges, keeping its shape
// (and the main thread free: a cold start of 300 ticks blocked 600 nodes for
// about 2 s).
const SETTLE_ALPHA = 0.3;
const SETTLE_TICKS = 80;
// Labels beyond the always-placed ones stop once they cover this share of
// the canvas: a density bound, so a big canvas or a zoomed-in view carries
// more of them and a small one fewer.
const LABEL_COVER = 0.1;
// The fit's margin around the graph, in pixels.
const FIT_PAD = 40;

export interface GraphCanvasProps {
  data: GraphData;
  colorOf: (nodeId: string) => string;
  selected: string | null;
  highlighted: Set<string> | null;
  traced: Set<string>;
  tracedEdge: [string, string] | null;
  onSelect: (id: string | null) => void;
  legend: { key: string; label: string; color: string }[];
  // The legend entry hovered or focused (null: none).
  onLegendFocus?: (key: string | null) => void;
}

interface Node {
  id: string;
  name: string;
  type: string | null;
  degree: number;
  x?: number;
  y?: number;
}
interface Link {
  id: string;
  source: string | NodeObject<Node>;
  target: string | NodeObject<Node>;
  type: string | null;
  description: string | null;
}

interface Palette {
  ink: string;
  graphite: string;
  rule: string;
  ruleStrong: string;
  canvas: string;
  nodeEdge: string;
  accent: string;
  font: string;
  resolve: (css: string) => string;
}

function readPalette(): Palette {
  const style = getComputedStyle(document.documentElement);
  const value = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  const cache = new Map<string, string>();
  return {
    ink: value("--ink", "#1c2128"),
    graphite: value("--graphite", "#59636e"),
    rule: value("--rule", "#d9dde3"),
    ruleStrong: value("--rule-strong", "#b8bfc8"),
    canvas: value("--canvas-bg", "#ffffff"),
    nodeEdge: value("--node-edge", "#848c96"),
    accent: value("--cobalt", "#4f46e5"),
    font: getComputedStyle(document.body).fontFamily || "sans-serif",
    resolve: (css) => {
      const name = css.match(/^var\((--[\w-]+)\)$/)?.[1];
      if (!name) return css;
      if (!cache.has(name)) cache.set(name, value(name, "#9aa7b1"));
      return cache.get(name) as string;
    },
  };
}

/** The page's theme, re-read when <html data-theme> changes. */
function usePalette(): Palette {
  const [palette, setPalette] = useState<Palette>(readPalette);
  useEffect(() => {
    const observer = new MutationObserver(() => setPalette(readPalette()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "lang"] });
    return () => observer.disconnect();
  }, []);
  return palette;
}

function withAlpha(color: string, alpha: number): string {
  const hex = color.match(/^#([0-9a-f]{6})$/i)?.[1];
  if (!hex) return color;
  return `#${hex}${Math.round(alpha * 255)
    .toString(16)
    .padStart(2, "0")}`;
}

const endId = (end: string | NodeObject<Node>) => (typeof end === "object" ? String(end.id) : end);

// Labels are set in screen pixels whatever the zoom (a font scaled by the
// canvas transform is laid out at its tiny nominal size, which breaks the
// letter spacing); a node is never drawn smaller than MIN_RADIUS pixels.
const LABEL_PX = 12;
const LINK_LABEL_PX = 11.5;
const MIN_RADIUS = 2.5;

/** The resting alpha of an edge among `edges`: fainter as they multiply, so
 * a dense graph's edges do not hide its nodes. */
export function linkAlpha(edges: number): number {
  return Math.min(0.85, Math.max(0.06, 0.85 * Math.sqrt(300 / Math.max(1, edges))));
}

/** A node's label to place, in screen pixels: the node's centre and radius
 * and the label's box size. */
export interface LabelCandidate {
  id: string;
  cx: number;
  cy: number;
  r: number;
  width: number;
  height: number;
  // Always placed (the focused, selected or traced node).
  strong: boolean;
  degree: number;
}

/** A placed label's box (its text is centred in it). */
export interface PlacedLabel {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

const GAP = 2;

/** The four boxes a label may take around its node, in order of preference:
 * right, left, above, below. */
function sides(c: LabelCandidate): PlacedLabel[] {
  const { id, cx, cy, r, width, height } = c;
  return [
    { id, x: cx + r + GAP, y: cy - height / 2, width, height },
    { id, x: cx - r - GAP - width, y: cy - height / 2, width, height },
    { id, x: cx - width / 2, y: cy - r - GAP - height, width, height },
    { id, x: cx - width / 2, y: cy + r + GAP, width, height },
  ];
}

const overlaps = (a: PlacedLabel, b: PlacedLabel) => !(a.x + a.width <= b.x || b.x + b.width <= a.x || a.y + a.height <= b.y || b.y + b.height <= a.y);

/** The labels to draw, greedily: the strong ones, then the most connected
 * first, each at the first of its four sides clear of those already placed
 * (skipped when none is), until the labels cover ``area`` square pixels (a
 * density bound; the strong ones always count and are always placed). */
export function placeLabels(candidates: LabelCandidate[], area = Infinity): PlacedLabel[] {
  const order = [...candidates].sort((a, b) => Number(b.strong) - Number(a.strong) || b.degree - a.degree);
  const placed: PlacedLabel[] = [];
  let covered = 0;
  for (const c of order) {
    const size = c.width * c.height;
    if (!c.strong && covered + size > area) continue;
    const options = sides(c);
    const clear = options.find((o) => placed.every((p) => !overlaps(o, p)));
    if (!clear && !c.strong) continue;
    placed.push(clear ?? (options[0] as PlacedLabel));
    covered += size;
  }
  return placed;
}

/** `(x, y)` of the graph in the canvas's CSS pixels, through the view's
 * transform (`ctx.getTransform()` while the graph is drawn). */
function toScreen(m: DOMMatrix, x: number, y: number, dpr: number): [number, number] {
  return [(m.a * x + m.c * y + m.e) / dpr, (m.b * x + m.d * y + m.f) / dpr];
}

/** The part of the graph the canvas draws: the nodes the server laid out
 * (its best-connected MAX_DRAWN_NODES), placed from that layout scaled to a
 * canvas-sized frame so the picture is stable from one visit to the next,
 * and the edges between them. */
export function drawnGraph(data: GraphData): { nodes: Node[]; links: Link[] } {
  const entries = Object.values(data.layout);
  let [minX, maxX, minY, maxY] = [Infinity, -Infinity, Infinity, -Infinity];
  for (const [x, y] of entries) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const span = Math.max(maxX - minX, maxY - minY) || 1;
  const nodes: Node[] = [];
  for (const n of data.nodes) {
    const p = data.layout[n.id];
    if (!p) continue;
    nodes.push({
      id: n.id,
      name: n.name,
      type: n.type,
      degree: n.degree,
      x: ((p[0] - (minX + maxX) / 2) / span) * 600,
      y: ((p[1] - (minY + maxY) / 2) / span) * 600,
    });
  }
  const ids = new Set(nodes.map((n) => n.id));
  const links: Link[] = data.edges
    .filter((e) => ids.has(e.source) && ids.has(e.target))
    .map((e) => ({ id: e.id, source: e.source, target: e.target, type: e.type, description: e.description }));
  return { nodes, links };
}

/** The forces besides the edges' (one set for the pre-layout and the live
 * engine, so the picture does not drift as it starts): repulsion, room
 * around each node (more for a busier one, and for its label), and the
 * centring and the weak gravity. */
function forces(nodes: Node[]) {
  const maxDeg = nodes.reduce((m, n) => Math.max(m, n.degree), 1);
  return {
    charge: forceManyBody().strength(CHARGE).distanceMax(700),
    collide: forceCollide<Node & SimulationNodeDatum>((n) => 16 + 5 * Math.sqrt(n.degree / maxDeg)),
    center: forceCenter(0, 0),
    x: forceX(0).strength(GRAVITY),
    y: forceY(0).strength(GRAVITY),
  };
}

/** Runs the pre-layout simulation on the drawn graph (in place), warm from
 * the server's layout. */
export function settle(graph: { nodes: Node[]; links: Link[] }): { nodes: Node[]; links: Link[] } {
  const { nodes, links } = graph;
  const simulation = forceSimulation(nodes as (Node & SimulationNodeDatum)[]).force(
    "link",
    forceLink(links as unknown as SimulationLinkDatum<Node & SimulationNodeDatum>[]).id((n) => (n as Node).id).distance(DISTANCE),
  );
  for (const [name, force] of Object.entries(forces(nodes))) simulation.force(name, force);
  simulation.alpha(SETTLE_ALPHA).stop().tick(SETTLE_TICKS);
  return graph;
}

export function GraphCanvas({ data, colorOf, selected, highlighted, traced, tracedEdge, onSelect, legend, onLegendFocus }: GraphCanvasProps) {
  const { t } = useI18n();
  const palette = usePalette();
  const graph = useRef<ForceGraphMethods<NodeObject<Node>, LinkObject<Node, Link>> | undefined>(undefined);
  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 800, height: 520, measured: false });
  const [hoverNode, setHoverNode] = useState<string | null>(null);
  const [hoverLink, setHoverLink] = useState<Link | null>(null);
  const fitted = useRef(false);
  const widths = useRef(new Map<string, number>());
  const reduced = useMemo(() => typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches, []);

  useEffect(() => {
    const element = box.current;
    if (!element) return;
    const measure = () => setSize({ width: element.clientWidth, height: element.clientHeight, measured: true });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // Only the laid-out nodes are drawn (the server lays out the best-connected
  // MAX_DRAWN_NODES), settled before the first paint, so the canvas opens on
  // a readable picture (and stays still under reduced motion); the live
  // engine then only eases what a drag disturbs.
  const graphData = useMemo(() => settle(drawnGraph(data)), [data]);

  // The live engine keeps the same forces as the settled layout, so the
  // picture does not drift as it starts, and a dragged node does not pull it
  // back into a knot.
  useEffect(() => {
    const g = graph.current;
    if (!g) return;
    g.d3Force("link")?.distance(DISTANCE);
    for (const [name, force] of Object.entries(forces(graphData.nodes))) g.d3Force(name, force);
  }, [graphData]);

  const neighbours = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const e of graphData.links) {
      const [a, b] = [endId(e.source), endId(e.target)];
      if (!map.has(a)) map.set(a, new Set());
      if (!map.has(b)) map.set(b, new Set());
      map.get(a)?.add(b);
      map.get(b)?.add(a);
    }
    return map;
  }, [graphData.links]);

  const maxDegree = useMemo(() => graphData.nodes.reduce((m, n) => Math.max(m, n.degree), 1), [graphData.nodes]);

  const focus = hoverNode ?? selected;
  const focusSet = useMemo(() => (focus ? new Set([focus, ...(neighbours.get(focus) ?? [])]) : null), [focus, neighbours]);
  const isLit = useCallback(
    (id: string) => {
      if (focusSet) return focusSet.has(id) || traced.has(id);
      if (highlighted && highlighted.size) return highlighted.has(id) || traced.has(id);
      return true;
    },
    [focusSet, highlighted, traced],
  );
  const isTraced = useCallback(
    (link: Link) => {
      if (!tracedEdge) return false;
      const [a, b] = [endId(link.source), endId(link.target)];
      return (tracedEdge[0] === a && tracedEdge[1] === b) || (tracedEdge[0] === b && tracedEdge[1] === a);
    },
    [tracedEdge],
  );
  const radius = useCallback((node: Node, scale = 1) => Math.max(3.5 + 5 * Math.sqrt(node.degree / maxDegree), MIN_RADIUS / scale), [maxDegree]);
  const restingAlpha = linkAlpha(graphData.links.length);

  const centre = useCallback(
    (id: string, ms = 500) => {
      const node = graphData.nodes.find((n) => n.id === id);
      if (!node || node.x === undefined || node.y === undefined) return;
      graph.current?.centerAt(node.x, node.y, ms);
      graph.current?.zoom(Math.max(graph.current.zoom() ?? 1, 1.8), ms);
    },
    [graphData.nodes],
  );

  // Fit the graph to the canvas between the overlays over it (the search
  // bar at its head, the legend and the zoom tools at its foot: every
  // element marked data-canvas-overlay beside or in it), so no node sits
  // under them.
  const fit = useCallback(
    (ms: number) => {
      const g = graph.current;
      const bounds = g?.getGraphBbox();
      const element = box.current;
      if (!g || !bounds || !element) return;
      const frame = element.getBoundingClientRect();
      let [top, bottom] = [0, 0];
      for (const overlay of (element.parentElement ?? element).querySelectorAll<HTMLElement>("[data-canvas-overlay]")) {
        const r = overlay.getBoundingClientRect();
        if (r.bottom <= frame.top || r.top >= frame.bottom || !r.height) continue;
        if (r.top + r.height / 2 < frame.top + frame.height / 2) top = Math.max(top, r.bottom - frame.top);
        else bottom = Math.max(bottom, frame.bottom - r.top);
      }
      const [bw, bh] = [Math.max(1, bounds.x[1] - bounds.x[0]), Math.max(1, bounds.y[1] - bounds.y[0])];
      const k = Math.min(12, Math.max(0.2, Math.min((size.width - 2 * FIT_PAD) / bw, (size.height - top - bottom - 2 * FIT_PAD) / bh)));
      g.zoom(k, ms);
      g.centerAt((bounds.x[0] + bounds.x[1]) / 2, (bounds.y[0] + bounds.y[1]) / 2 - (top - bottom) / (2 * k), ms);
    },
    [size],
  );

  // The layout is settled before the first paint: fit it (or centre the
  // selection) once, at once, when the canvas has its size. Only the Fit
  // button animates.
  useEffect(() => {
    if (!size.measured || fitted.current || !graph.current) return;
    const frame = requestAnimationFrame(() => {
      if (fitted.current) return;
      fitted.current = true;
      if (selected) centre(selected, 0);
      else fit(0);
    });
    return () => cancelAnimationFrame(frame);
  }, [size, selected, centre, fit]);

  // Bring a selection made elsewhere (search, a trail) into view.
  useEffect(() => {
    if (selected && fitted.current) centre(selected);
  }, [selected, centre]);

  const pan = (dx: number, dy: number) => {
    const g = graph.current;
    if (!g) return;
    const c = g.centerAt();
    const k = g.zoom() || 1;
    g.centerAt(c.x + dx / k, c.y + dy / k, 150);
  };

  const node = (n: NodeObject<Node>, ctx: CanvasRenderingContext2D, scale: number) => {
    const id = String(n.id);
    const lit = isLit(id);
    const r = radius(n as Node, scale);
    const fill = palette.resolve(colorOf(id));
    ctx.globalAlpha = lit ? 1 : 0.18;
    ctx.beginPath();
    ctx.arc(n.x ?? 0, n.y ?? 0, r, 0, 2 * Math.PI);
    ctx.fillStyle = fill;
    ctx.fill();
    // The edge keeps a light-theme node (some series are under 3:1 on the
    // canvas) at 3:1, and separates overlapping ones in the dark.
    ctx.lineWidth = 1.5 / scale;
    ctx.strokeStyle = palette.nodeEdge;
    ctx.stroke();
    if (id === selected || id === focus) {
      ctx.lineWidth = 2.5 / scale;
      ctx.strokeStyle = palette.ink;
      ctx.stroke();
    }
    if (traced.has(id)) {
      ctx.beginPath();
      ctx.arc(n.x ?? 0, n.y ?? 0, r + 5 / scale, 0, 2 * Math.PI);
      ctx.setLineDash([3 / scale, 3 / scale]);
      ctx.lineWidth = 2 / scale;
      ctx.strokeStyle = palette.accent;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.globalAlpha = 1;
  };

  // The labels, after every node and edge: in screen pixels, of the lit
  // nodes on screen, placed most connected first beside, then above or below
  // their node without overlapping, up to a share of the canvas (the
  // focused, selected and traced ones always).
  const labels = (ctx: CanvasRenderingContext2D, scale: number) => {
    const dpr = window.devicePixelRatio || 1;
    const { width, height } = size;
    const candidates: LabelCandidate[] = [];
    const names = new Map<string, { name: string; strong: boolean }>();
    const view = ctx.getTransform();
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.font = `600 ${LABEL_PX}px ${palette.font}`;
    for (const n of graphData.nodes) {
      const id = n.id;
      if (!isLit(id)) continue;
      const strong = id === focus || id === selected || traced.has(id);
      const [cx, cy] = toScreen(view, n.x ?? 0, n.y ?? 0, dpr);
      if (cx < 0 || cx > width || cy < 0 || cy > height) continue;
      const key = `${palette.font}\u0000${n.name}`;
      let textWidth = widths.current.get(key);
      if (textWidth === undefined) {
        textWidth = ctx.measureText(n.name).width;
        widths.current.set(key, textWidth);
      }
      candidates.push({ id, cx, cy, r: radius(n, scale) * scale, width: textWidth + 4, height: LABEL_PX + 4, strong, degree: n.degree });
      names.set(id, { name: n.name, strong });
    }
    const placed = placeLabels(candidates, width * height * LABEL_COVER);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.lineWidth = 3.5;
    for (const box of placed) {
      const { name, strong } = names.get(box.id) as { name: string; strong: boolean };
      const [x, y] = [box.x + box.width / 2, box.y + box.height / 2];
      ctx.font = `${strong ? 600 : 500} ${LABEL_PX}px ${palette.font}`;
      ctx.strokeStyle = palette.canvas;
      ctx.strokeText(name, x, y);
      ctx.fillStyle = palette.ink;
      ctx.fillText(name, x, y);
    }
    ctx.restore();
  };

  const linkColor = (l: LinkObject<Node, Link>) => {
    const link = l as Link;
    if (isTraced(link)) return palette.accent;
    const [a, b] = [endId(link.source), endId(link.target)];
    if (focus && (a === focus || b === focus)) return withAlpha(palette.ink, 0.75);
    if (hoverLink === link) return palette.ink;
    if (focusSet || (highlighted && highlighted.size)) return withAlpha(palette.ruleStrong, isLit(a) && isLit(b) ? Math.max(restingAlpha, 0.6) : Math.min(restingAlpha, 0.15));
    return withAlpha(palette.ruleStrong, restingAlpha);
  };
  const linkWidth = (l: LinkObject<Node, Link>) => {
    const link = l as Link;
    if (isTraced(link)) return 2.5;
    const [a, b] = [endId(link.source), endId(link.target)];
    if ((focus && (a === focus || b === focus)) || hoverLink === link) return 1.8;
    return 1;
  };

  // A hovered edge names its relationship (and its description) at its
  // middle, in screen pixels.
  const linkLabel = (l: LinkObject<Node, Link>, ctx: CanvasRenderingContext2D) => {
    const link = l as Link;
    if (link !== hoverLink) return;
    const s = link.source as NodeObject<Node>;
    const e = link.target as NodeObject<Node>;
    const raw = [link.type, link.description].filter(Boolean).join(" · ");
    if (!raw) return;
    const text = raw.length > 64 ? `${raw.slice(0, 63)}…` : raw;
    const dpr = window.devicePixelRatio || 1;
    const [x, y] = toScreen(ctx.getTransform(), ((s.x ?? 0) + (e.x ?? 0)) / 2, ((s.y ?? 0) + (e.y ?? 0)) / 2, dpr);
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.font = `500 ${LINK_LABEL_PX}px ${palette.font}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const width = ctx.measureText(text).width;
    const pad = 5;
    ctx.fillStyle = withAlpha(palette.canvas, 0.94);
    ctx.strokeStyle = palette.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x - width / 2 - pad, y - LINK_LABEL_PX / 2 - pad, width + pad * 2, LINK_LABEL_PX + pad * 2, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = palette.ink;
    ctx.fillText(text, x, y);
    ctx.restore();
  };

  const selectedName = selected ? graphData.nodes.find((n) => n.id === selected)?.name : null;

  return (
    <div
      ref={box}
      className="graph-canvas sheet"
      tabIndex={0}
      role="application"
      data-traced={traced.size}
      data-selected={selected ?? ""}
      data-drawn={graphData.nodes.length}
      aria-label={t("graph.canvasLabel", { nodes: graphData.nodes.length, edges: graphData.links.length })}
      onKeyDown={(event) => {
        const step = 60;
        if (event.key === "+" || event.key === "=") graph.current?.zoom((graph.current.zoom() || 1) * 1.25, 200);
        else if (event.key === "-") graph.current?.zoom((graph.current.zoom() || 1) * 0.8, 200);
        else if (event.key === "0") fit(300);
        else if (event.key === "ArrowLeft") pan(-step, 0);
        else if (event.key === "ArrowRight") pan(step, 0);
        else if (event.key === "ArrowUp") pan(0, -step);
        else if (event.key === "ArrowDown") pan(0, step);
        else if (event.key === "Escape") onSelect(null);
        else return;
        event.preventDefault();
      }}
    >
      <ForceGraph2D<Node, Link>
        ref={graph}
        graphData={graphData}
        width={size.width}
        height={size.height}
        backgroundColor="rgba(0,0,0,0)"
        nodeId="id"
        nodeLabel={(n) => `${n.name}${n.type ? ` (${n.type})` : ""} · ${t("graph.degree")} ${n.degree}`}
        nodeCanvasObjectMode={() => "replace"}
        nodeCanvasObject={node}
        nodePointerAreaPaint={(n, color, ctx, scale) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(n.x ?? 0, n.y ?? 0, radius(n as Node, scale) + 3 / scale, 0, 2 * Math.PI);
          ctx.fill();
        }}
        linkColor={linkColor}
        linkWidth={linkWidth}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        linkDirectionalArrowColor={linkColor}
        linkDirectionalParticles={(l) => (!reduced && isTraced(l as Link) ? 3 : 0)}
        linkDirectionalParticleWidth={3}
        linkDirectionalParticleColor={() => palette.accent}
        linkCanvasObjectMode={() => "after"}
        linkCanvasObject={linkLabel}
        onRenderFramePost={labels}
        linkHoverPrecision={6}
        onNodeHover={(n) => setHoverNode(n ? String(n.id) : null)}
        onLinkHover={(l) => setHoverLink((l as Link) ?? null)}
        onNodeClick={(n) => {
          onSelect(String(n.id));
          centre(String(n.id));
        }}
        onBackgroundClick={() => onSelect(null)}
        minZoom={0.2}
        maxZoom={12}
        warmupTicks={0}
        cooldownTicks={reduced ? 0 : 40}
        d3VelocityDecay={0.35}
      />
      <div className="graph-tools" data-canvas-overlay="">
        <button className="button button-small" onClick={() => graph.current?.zoom((graph.current.zoom() || 1) * 1.25, 200)} aria-label={t("graph.zoomIn")}>
          +
        </button>
        <button className="button button-small" onClick={() => graph.current?.zoom((graph.current.zoom() || 1) * 0.8, 200)} aria-label={t("graph.zoomOut")}>
          −
        </button>
        <button className="button button-small" onClick={() => fit(400)}>
          {t("graph.fit")}
        </button>
      </div>
      {legend.length ? (
        <div className="graph-legend" aria-label={t("graph.legend")} data-canvas-overlay="">
          {legend.map((item) => (
            <span
              key={item.key}
              className="legend-item"
              tabIndex={0}
              onMouseEnter={() => onLegendFocus?.(item.key)}
              onMouseLeave={() => onLegendFocus?.(null)}
              onFocus={() => onLegendFocus?.(item.key)}
              onBlur={() => onLegendFocus?.(null)}
            >
              <span className="swatch" style={{ background: item.color }} />
              {item.label}
            </span>
          ))}
        </div>
      ) : null}
      <span className="visually-hidden" aria-live="polite">
        {selectedName ?? ""}
      </span>
    </div>
  );
}
