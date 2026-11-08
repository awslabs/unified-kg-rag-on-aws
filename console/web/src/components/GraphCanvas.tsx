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
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, type SimulationLinkDatum, type SimulationNodeDatum } from "d3-force";
import ForceGraph2D, { type ForceGraphMethods, type LinkObject, type NodeObject } from "react-force-graph-2d";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n/i18n";
import type { GraphData } from "../lib/types";

// Past this many nodes, only the focused and the busiest carry labels and
// the traced edge carries no particles.
const LARGE = 1500;
// The forces: room for labels between neighbours, more pull on a big graph.
const forces = (large: boolean) => ({ charge: large ? -60 : -260, distance: large ? 30 : 78 });

export interface GraphCanvasProps {
  data: GraphData;
  colorOf: (nodeId: string) => string;
  selected: string | null;
  highlighted: Set<string> | null;
  traced: Set<string>;
  tracedEdge: [string, string] | null;
  onSelect: (id: string | null) => void;
  legend: { label: string; color: string }[];
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

export function GraphCanvas({ data, colorOf, selected, highlighted, traced, tracedEdge, onSelect, legend }: GraphCanvasProps) {
  const { t } = useI18n();
  const palette = usePalette();
  const graph = useRef<ForceGraphMethods<NodeObject<Node>, LinkObject<Node, Link>> | undefined>(undefined);
  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 800, height: 520 });
  const [hoverNode, setHoverNode] = useState<string | null>(null);
  const [hoverLink, setHoverLink] = useState<Link | null>(null);
  const fitted = useRef(false);
  const reduced = useMemo(() => typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches, []);
  const large = data.nodes.length > LARGE;

  useEffect(() => {
    const element = box.current;
    if (!element) return;
    const measure = () => setSize({ width: element.clientWidth, height: element.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // The force simulation starts from the exported layout (scaled to a
  // canvas-sized frame), so the picture is stable from one visit to the next.
  const graphData = useMemo(() => {
    const entries = Object.values(data.layout);
    const xs = entries.map((p) => p[0]);
    const ys = entries.map((p) => p[1]);
    const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const span = Math.max(maxX - minX, maxY - minY) || 1;
    const nodes: Node[] = data.nodes.map((n) => {
      const p = data.layout[n.id];
      return {
        id: n.id,
        name: n.name,
        type: n.type,
        degree: n.degree,
        ...(p ? { x: ((p[0] - (minX + maxX) / 2) / span) * 600, y: ((p[1] - (minY + maxY) / 2) / span) * 600 } : {}),
      };
    });
    const ids = new Set(nodes.map((n) => n.id));
    const links: Link[] = data.edges
      .filter((e) => ids.has(e.source) && ids.has(e.target))
      .map((e) => ({ id: e.id, source: e.source, target: e.target, type: e.type, description: e.description }));
    // Settled before the first paint, so the canvas opens on a readable
    // picture (and stays still under reduced motion); the live engine then
    // only eases what a drag disturbs.
    const large = nodes.length > LARGE;
    const { charge, distance } = forces(large);
    const maxDeg = Math.max(1, ...nodes.map((n) => n.degree));
    forceSimulation(nodes as (Node & SimulationNodeDatum)[])
      .force("link", forceLink(links as unknown as SimulationLinkDatum<Node & SimulationNodeDatum>[]).id((n) => (n as Node).id).distance(distance))
      .force("charge", forceManyBody().strength(charge).distanceMax(700))
      .force("collide", forceCollide((n) => 6 + 5 * Math.sqrt((n as Node).degree / maxDeg) + (large ? 0 : 10)))
      .force("center", forceCenter(0, 0))
      .stop()
      .tick(large ? 120 : 300);
    return { nodes, links };
  }, [data]);

  // The live engine keeps the same forces, so a dragged node does not pull
  // the picture back into a knot.
  useEffect(() => {
    const g = graph.current;
    if (!g) return;
    const { charge, distance } = forces(large);
    g.d3Force("charge")?.strength(charge);
    g.d3Force("link")?.distance(distance);
  }, [graphData, large]);

  const neighbours = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const e of data.edges) {
      if (!map.has(e.source)) map.set(e.source, new Set());
      if (!map.has(e.target)) map.set(e.target, new Set());
      map.get(e.source)?.add(e.target);
      map.get(e.target)?.add(e.source);
    }
    return map;
  }, [data.edges]);

  const maxDegree = useMemo(() => Math.max(1, ...data.nodes.map((n) => n.degree)), [data.nodes]);
  const busiest = useMemo(
    () => new Set([...data.nodes].sort((a, b) => b.degree - a.degree).slice(0, large ? 20 : 10).map((n) => n.id)),
    [data.nodes, large],
  );

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
  const radius = useCallback((node: Node) => 3.5 + 5 * Math.sqrt(node.degree / maxDegree), [maxDegree]);

  const centre = useCallback(
    (id: string, ms = 500) => {
      const node = graphData.nodes.find((n) => n.id === id);
      if (!node || node.x === undefined || node.y === undefined) return;
      graph.current?.centerAt(node.x, node.y, ms);
      graph.current?.zoom(Math.max(graph.current.zoom() ?? 1, 1.8), ms);
    },
    [graphData.nodes],
  );

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
    const r = radius(n as Node);
    const fill = palette.resolve(colorOf(id));
    ctx.globalAlpha = lit ? 1 : 0.18;
    ctx.beginPath();
    ctx.arc(n.x ?? 0, n.y ?? 0, r, 0, 2 * Math.PI);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.lineWidth = 1.5 / scale;
    ctx.strokeStyle = palette.canvas;
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
    const labelled =
      (focusSet?.has(id) ?? false) || traced.has(id) || (highlighted?.has(id) ?? false) || (!large && scale >= 1.6) || (!focusSet && busiest.has(id));
    if (labelled && lit) {
      const size = 12 / scale;
      const strong = id === focus || id === selected || traced.has(id);
      ctx.font = `${strong ? 600 : 500} ${size}px ${palette.font}`;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const x = (n.x ?? 0) + r + 4 / scale;
      const y = n.y ?? 0;
      ctx.lineJoin = "round";
      ctx.lineWidth = 3.5 / scale;
      ctx.strokeStyle = palette.canvas;
      ctx.strokeText(n.name, x, y);
      ctx.fillStyle = palette.ink;
      ctx.fillText(n.name, x, y);
    }
    ctx.globalAlpha = 1;
  };

  const linkColor = (l: LinkObject<Node, Link>) => {
    const link = l as Link;
    if (isTraced(link)) return palette.accent;
    const [a, b] = [endId(link.source), endId(link.target)];
    if (focus && (a === focus || b === focus)) return withAlpha(palette.ink, 0.75);
    if (hoverLink === link) return palette.ink;
    if (focusSet || (highlighted && highlighted.size)) return withAlpha(palette.ruleStrong, isLit(a) && isLit(b) ? 0.6 : 0.15);
    return withAlpha(palette.ruleStrong, 0.85);
  };
  const linkWidth = (l: LinkObject<Node, Link>) => {
    const link = l as Link;
    if (isTraced(link)) return 2.5;
    const [a, b] = [endId(link.source), endId(link.target)];
    if ((focus && (a === focus || b === focus)) || hoverLink === link) return 1.8;
    return 1;
  };

  // A hovered edge names its relationship (and its description) at its middle.
  const linkLabel = (l: LinkObject<Node, Link>, ctx: CanvasRenderingContext2D, scale: number) => {
    const link = l as Link;
    if (link !== hoverLink) return;
    const s = link.source as NodeObject<Node>;
    const e = link.target as NodeObject<Node>;
    const raw = [link.type, link.description].filter(Boolean).join(" · ");
    if (!raw) return;
    const text = raw.length > 64 ? `${raw.slice(0, 63)}…` : raw;
    const size = 11.5 / scale;
    ctx.font = `500 ${size}px ${palette.font}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const x = ((s.x ?? 0) + (e.x ?? 0)) / 2;
    const y = ((s.y ?? 0) + (e.y ?? 0)) / 2;
    const width = ctx.measureText(text).width;
    const pad = 5 / scale;
    ctx.fillStyle = withAlpha(palette.canvas, 0.94);
    ctx.strokeStyle = palette.rule;
    ctx.lineWidth = 1 / scale;
    ctx.beginPath();
    ctx.roundRect(x - width / 2 - pad, y - size / 2 - pad, width + pad * 2, size + pad * 2, 4 / scale);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = palette.ink;
    ctx.fillText(text, x, y);
  };

  const selectedName = selected ? data.nodes.find((n) => n.id === selected)?.name : null;

  return (
    <div
      ref={box}
      className="graph-canvas sheet"
      tabIndex={0}
      role="application"
      data-traced={traced.size}
      data-selected={selected ?? ""}
      aria-label={t("graph.canvasLabel", { nodes: data.nodes.length, edges: data.edges.length })}
      onKeyDown={(event) => {
        const step = 60;
        if (event.key === "+" || event.key === "=") graph.current?.zoom((graph.current.zoom() || 1) * 1.25, 200);
        else if (event.key === "-") graph.current?.zoom((graph.current.zoom() || 1) * 0.8, 200);
        else if (event.key === "0") graph.current?.zoomToFit(300, 40);
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
        nodePointerAreaPaint={(n, color, ctx) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(n.x ?? 0, n.y ?? 0, radius(n as Node) + 3, 0, 2 * Math.PI);
          ctx.fill();
        }}
        linkColor={linkColor}
        linkWidth={linkWidth}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        linkDirectionalArrowColor={linkColor}
        linkDirectionalParticles={(l) => (!large && !reduced && isTraced(l as Link) ? 3 : 0)}
        linkDirectionalParticleWidth={3}
        linkDirectionalParticleColor={() => palette.accent}
        linkCanvasObjectMode={() => "after"}
        linkCanvasObject={linkLabel}
        linkHoverPrecision={6}
        onNodeHover={(n) => setHoverNode(n ? String(n.id) : null)}
        onLinkHover={(l) => setHoverLink((l as Link) ?? null)}
        onNodeClick={(n) => {
          onSelect(String(n.id));
          centre(String(n.id));
        }}
        onBackgroundClick={() => onSelect(null)}
        enableNodeDrag={!large}
        minZoom={0.2}
        maxZoom={12}
        warmupTicks={0}
        cooldownTicks={reduced ? 0 : 40}
        d3VelocityDecay={0.35}
        onEngineStop={() => {
          if (fitted.current) return;
          fitted.current = true;
          if (selected) centre(selected, 0);
          else graph.current?.zoomToFit(reduced ? 0 : 400, 48);
        }}
      />
      <div className="graph-tools">
        <button className="button button-small" onClick={() => graph.current?.zoom((graph.current.zoom() || 1) * 1.25, 200)} aria-label={t("graph.zoomIn")}>
          +
        </button>
        <button className="button button-small" onClick={() => graph.current?.zoom((graph.current.zoom() || 1) * 0.8, 200)} aria-label={t("graph.zoomOut")}>
          −
        </button>
        <button className="button button-small" onClick={() => graph.current?.zoomToFit(400, 48)}>
          {t("graph.fit")}
        </button>
      </div>
      {legend.length ? (
        <div className="graph-legend" aria-label={t("graph.legend")}>
          {legend.map((item) => (
            <span key={item.label}>
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
