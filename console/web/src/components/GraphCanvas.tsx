// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/** The graph drawn from its exported layout: pan with a drag or the arrow
 * keys, zoom with the wheel, +/- or the buttons; a click selects a node. */
import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { useI18n } from "../i18n/i18n";
import type { GraphData } from "../lib/types";

const SERIES = 8;
const VIEW = 1000;

export function colorSlot(index: number): string {
  return index < SERIES ? `var(--series-${index + 1})` : "var(--series-other)";
}

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

export function GraphCanvas({ data, colorOf, selected, highlighted, traced, tracedEdge, onSelect, legend }: GraphCanvasProps) {
  const { t } = useI18n();
  const [view, setView] = useState({ k: 1, x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const svg = useRef<SVGSVGElement>(null);

  // Fitted to the layout's bounds (the server packs the components, so no
  // island stretches them).
  const positions = useMemo(() => {
    const entries = Object.entries(data.layout);
    const quantile = (values: number[], q: number) => {
      const sorted = [...values].sort((a, b) => a - b);
      return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))] ?? 0;
    };
    const xs = entries.map(([, p]) => p[0]);
    const ys = entries.map(([, p]) => p[1]);
    const [minX, maxX, minY, maxY] = [quantile(xs, 0), quantile(xs, 1), quantile(ys, 0), quantile(ys, 1)];
    const span = Math.max(maxX - minX, maxY - minY) || 1;
    const clamp = (v: number) => Math.min(VIEW - 30, Math.max(30, v));
    const map = new Map<string, [number, number]>();
    for (const [id, [x, y]] of entries) {
      map.set(id, [clamp(70 + ((x - minX) / span) * (VIEW - 140)), clamp(70 + ((y - minY) / span) * (VIEW - 140))]);
    }
    return map;
  }, [data.layout]);

  const byId = useMemo(() => new Map(data.nodes.map((n) => [n.id, n])), [data.nodes]);
  const maxDegree = Math.max(1, ...data.nodes.map((n) => n.degree));
  const labelled = useMemo(() => {
    const top = [...data.nodes].sort((a, b) => b.degree - a.degree).slice(0, 14).map((n) => n.id);
    return new Set([...top, ...(selected ? [selected] : []), ...(highlighted ?? []), ...traced]);
  }, [data.nodes, selected, highlighted, traced]);

  // Bring a selection made elsewhere (search, a trail) into view.
  useEffect(() => {
    if (!selected) return;
    const p = positions.get(selected);
    if (!p) return;
    setView((v) => {
      const k = Math.max(v.k, 1.6);
      return { k, x: VIEW / 2 - p[0] * k, y: VIEW / 2 - p[1] * k };
    });
  }, [selected, positions]);

  const zoom = (factor: number, cx = VIEW / 2, cy = VIEW / 2) =>
    setView((v) => {
      const k = Math.min(8, Math.max(0.3, v.k * factor));
      return { k, x: cx - ((cx - v.x) * k) / v.k, y: cy - ((cy - v.y) * k) / v.k };
    });

  const toView = (event: { clientX: number; clientY: number }) => {
    const box = svg.current?.getBoundingClientRect();
    if (!box) return [VIEW / 2, VIEW / 2] as const;
    const scale = VIEW / Math.min(box.width, box.height);
    const offsetX = (box.width * scale - VIEW) / 2;
    const offsetY = (box.height * scale - VIEW) / 2;
    return [(event.clientX - box.left) * scale - offsetX, (event.clientY - box.top) * scale - offsetY] as const;
  };

  const onPointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    drag.current = { x: event.clientX, y: event.clientY, moved: false };
    (event.target as Element).setPointerCapture?.(event.pointerId);
  };
  const onPointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const start = drag.current;
    if (!start) return;
    const box = svg.current?.getBoundingClientRect();
    const scale = box ? VIEW / Math.min(box.width, box.height) : 1;
    const dx = (event.clientX - start.x) * scale;
    const dy = (event.clientY - start.y) * scale;
    if (Math.abs(dx) + Math.abs(dy) > 3) start.moved = true;
    start.x = event.clientX;
    start.y = event.clientY;
    if (start.moved) setView((v) => ({ ...v, x: v.x + dx, y: v.y + dy }));
  };

  const near = (id: string) => !highlighted || highlighted.has(id) || id === selected;

  return (
    <div
      className="graph-canvas sheet"
      tabIndex={0}
      role="application"
      aria-label={t("graph.canvasLabel", { nodes: data.nodes.length, edges: data.edges.length })}
      onKeyDown={(event) => {
        const step = 40;
        if (event.key === "+" || event.key === "=") zoom(1.25);
        else if (event.key === "-") zoom(0.8);
        else if (event.key === "ArrowLeft") setView((v) => ({ ...v, x: v.x + step }));
        else if (event.key === "ArrowRight") setView((v) => ({ ...v, x: v.x - step }));
        else if (event.key === "ArrowUp") setView((v) => ({ ...v, y: v.y + step }));
        else if (event.key === "ArrowDown") setView((v) => ({ ...v, y: v.y - step }));
        else if (event.key === "Escape") onSelect(null);
        else return;
        event.preventDefault();
      }}
    >
      <svg
        ref={svg}
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        preserveAspectRatio="xMidYMid meet"
        onWheel={(event) => {
          const [cx, cy] = toView(event);
          zoom(event.deltaY < 0 ? 1.15 : 0.87, cx, cy);
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={(event) => {
          const moved = drag.current?.moved;
          drag.current = null;
          if (!moved && (event.target as Element).tagName === "svg") onSelect(null);
        }}
      >
        <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
          {data.edges.map((edge) => {
            const a = positions.get(edge.source);
            const b = positions.get(edge.target);
            if (!a || !b) return null;
            const isTraced =
              tracedEdge && ((tracedEdge[0] === edge.source && tracedEdge[1] === edge.target) || (tracedEdge[0] === edge.target && tracedEdge[1] === edge.source));
            const isNear = selected && (edge.source === selected || edge.target === selected);
            return (
              <line
                key={edge.id}
                className={`graph-edge${isTraced ? " traced" : isNear ? " near" : ""}`}
                x1={a[0]}
                y1={a[1]}
                x2={b[0]}
                y2={b[1]}
                strokeWidth={(isTraced ? 2.5 : isNear ? 1.5 : 1) / view.k}
                opacity={highlighted && !(near(edge.source) && near(edge.target)) ? 0.15 : 1}
              />
            );
          })}
          {data.nodes.map((node) => {
            const p = positions.get(node.id);
            if (!p) return null;
            const r = (5 + 9 * Math.sqrt(node.degree / maxDegree)) / Math.sqrt(view.k);
            const isSelected = node.id === selected;
            return (
              <g
                key={node.id}
                className={`graph-node${isSelected ? " selected" : ""}${near(node.id) || traced.has(node.id) ? "" : " dim"}`}
                transform={`translate(${p[0]} ${p[1]})`}
                onClick={(event) => {
                  event.stopPropagation();
                  onSelect(node.id);
                }}
              >
                <title>{`${node.name} (${node.type ?? "—"}) · ${t("graph.degree")} ${node.degree}`}</title>
                {traced.has(node.id) ? <circle className="ring" r={r + 6 / Math.sqrt(view.k)} /> : null}
                <circle r={r} fill={colorOf(node.id)} />
                {labelled.has(node.id) ? (
                  <text x={r + 4 / view.k} y={4 / view.k} style={{ fontSize: `${13 / Math.sqrt(view.k)}px`, strokeWidth: `${3.5 / Math.sqrt(view.k)}px` }}>
                    {byId.get(node.id)?.name}
                  </text>
                ) : null}
              </g>
            );
          })}
        </g>
      </svg>
      <div className="graph-tools">
        <button className="button button-small" onClick={() => zoom(1.25)} aria-label={t("graph.zoomIn")}>
          +
        </button>
        <button className="button button-small" onClick={() => zoom(0.8)} aria-label={t("graph.zoomOut")}>
          −
        </button>
        <button className="button button-small" onClick={() => setView({ k: 1, x: 0, y: 0 })}>
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
    </div>
  );
}
