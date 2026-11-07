// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/** Small hand-drawn SVG charts (no chart package): a paired-delta interval
 * plot, a bar series and a histogram. Every value they draw is also printed
 * next to its mark or in a table, so colour never carries it alone. */
import { useState } from "react";
import { useI18n } from "../i18n/i18n";
import type { Delta } from "../lib/types";

/** Whether a higher value of `metric` is better (true), worse (false) or
 * neither (null: answer length). */
export function higherIsBetter(metric: string): boolean | null {
  if (metric.startsWith("latency/") || metric.startsWith("usage/")) return false;
  if (metric.endsWith("/answer_length")) return null;
  return true;
}

export function deltaVerdict(metric: string, delta: Delta): "better" | "worse" | "neutral" {
  const direction = higherIsBetter(metric);
  if (direction === null) return "neutral";
  if (delta.ci_low > 0) return direction ? "better" : "worse";
  if (delta.ci_high < 0) return direction ? "worse" : "better";
  return "neutral";
}

/** One row per metric: the mean paired delta and its interval on the
 * row's own scale (metrics differ in unit), zero marked. */
export function DeltaPlot({ deltas, confidence = 0.95 }: { deltas: [string, Delta][]; confidence?: number }) {
  const { t, number, percent } = useI18n();
  const width = 220;
  const pad = 10;
  return (
    <div className="table-wrap">
      <table className="table">
        <caption className="visually-hidden">{t("chart.deltaCaption", { confidence: percent(confidence) })}</caption>
        <thead>
          <tr>
            <th>{t("evaluate.metric")}</th>
            <th>{t("chart.interval", { confidence: percent(confidence) })}</th>
            <th className="num">{t("evaluate.delta")}</th>
            <th className="num">{t("evaluate.ci")}</th>
            <th className="num">{t("evaluate.winsTiesLosses")}</th>
          </tr>
        </thead>
        <tbody>
          {deltas.map(([metric, delta]) => {
            const extent = Math.max(Math.abs(delta.ci_low), Math.abs(delta.ci_high), Math.abs(delta.mean_delta)) || 1;
            const x = (v: number) => pad + ((v + extent) / (2 * extent)) * (width - 2 * pad);
            const verdict = deltaVerdict(metric, delta);
            const digits = Math.abs(delta.mean_delta) < 0.01 && delta.mean_delta !== 0 ? 4 : 3;
            return (
              <tr key={metric}>
                <td className="mono">{metric}</td>
                <td>
                  <svg className="chart" width={width} height={22} role="img" aria-label={t("chart.deltaRow", { metric, delta: number(delta.mean_delta, { maximumFractionDigits: digits }) })}>
                    <line className="zero" x1={x(0)} x2={x(0)} y1={2} y2={20} />
                    <line className={`ci ${verdict}`} x1={x(delta.ci_low)} x2={x(delta.ci_high)} y1={11} y2={11} />
                    <circle className={`ci-dot ${verdict}`} cx={x(delta.mean_delta)} cy={11} r={4.5} />
                  </svg>
                </td>
                <td className="num">
                  {delta.mean_delta > 0 ? "+" : ""}
                  {number(delta.mean_delta, { maximumFractionDigits: digits })}
                  {verdict !== "neutral" ? <span className={`badge ${verdict === "better" ? "badge-good" : "badge-bad"}`} style={{ marginLeft: 6 }}>{t(`chart.${verdict}`)}</span> : null}
                </td>
                <td className="num">
                  [{number(delta.ci_low, { maximumFractionDigits: digits })}, {number(delta.ci_high, { maximumFractionDigits: digits })}]
                </td>
                <td className="num">
                  {delta.wins}/{delta.ties}/{delta.losses}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Advance of one character of the 11px data face, to centre a label.
const MONO_WIDTH = 6.6;
const tickText = (key: string) => (key.length > 6 ? key.slice(-5) : key);

/** Vertical bars over a category axis, one series, labelled with values. */
export function Bars({ data, format, label }: { data: { key: string; value: number }[]; format: (v: number) => string; label: string }) {
  const [hover, setHover] = useState<{ key: string; x: number; y: number } | null>(null);
  const height = 140;
  const barWidth = Math.max(10, Math.min(36, 560 / Math.max(1, data.length) - 6));
  const width = Math.max(220, data.length * (barWidth + 6) + 40);
  const max = Math.max(...data.map((d) => d.value), 0) || 1;
  return (
    <div style={{ position: "relative", overflowX: "auto" }}>
      <svg className="chart" width={width} height={height + 34} role="img" aria-label={label}>
        <line className="axis" x1={30} x2={width} y1={height} y2={height} />
        <line className="grid" x1={30} x2={width} y1={height / 2} y2={height / 2} />
        <text x={0} y={12}>{format(max)}</text>
        {data.map((d, i) => {
          const h = (d.value / max) * (height - 16);
          const x = 34 + i * (barWidth + 6);
          return (
            <g
              key={d.key}
              onMouseEnter={(e) => setHover({ key: d.key, x: e.clientX, y: e.clientY })}
              onMouseLeave={() => setHover(null)}
            >
              <rect x={x - 3} y={0} width={barWidth + 6} height={height} fill="transparent" />
              <path className="bar" d={roundedTop(x, height - h, barWidth, h)} />
              <text x={x + barWidth / 2 - (tickText(d.key).length * MONO_WIDTH) / 2} y={height + 14}>
                {tickText(d.key)}
              </text>
            </g>
          );
        })}
      </svg>
      {hover ? (
        <div className="tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          {hover.key}: {format(data.find((d) => d.key === hover.key)?.value ?? 0)}
        </div>
      ) : null}
    </div>
  );
}

/** A bar whose top corners are rounded (4px), standing on the baseline. */
function roundedTop(x: number, y: number, w: number, h: number): string {
  if (h <= 0) return "";
  const r = Math.min(4, w / 2, h);
  return `M${x},${y + h} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h} Z`;
}
