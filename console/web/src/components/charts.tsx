// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/** Small hand-drawn SVG charts (no chart package): a paired-delta interval
 * plot, a bar series and a histogram. Every value they draw is also printed
 * next to its mark or in a table, so colour never carries it alone. */
import { useState } from "react";
import { useI18n } from "../i18n/i18n";
import type { Delta, Multiplicity } from "../lib/types";

// Rates where less is better, and diagnostics with no better direction
// (the evaluation's own lists: comparison._UNFILLED, _DIAGNOSTICS).
const LOWER_IS_BETTER = new Set(["assertion_contradiction_rate", "negative_assertion_violation_rate"]);
const NO_DIRECTION = new Set(["answer_length", "attributable_fraction"]);

/** Whether a higher value of `metric` is better (true), worse (false) or
 * neither (null: answer length, the attributable fraction). */
export function higherIsBetter(metric: string): boolean | null {
  if (metric.startsWith("latency/") || metric.startsWith("usage/")) return false;
  const type = metric.split("/").pop() ?? metric;
  if (NO_DIRECTION.has(type)) return null;
  return !LOWER_IS_BETTER.has(type);
}

/** The catalog key of a metric's description: `metric.<type>` (the
 * abstention measures `metric.abstention_<type>`). */
export function metricKey(metric: string): string {
  const [evaluator, type] = metric.includes("/") ? metric.split("/", 2) : ["", metric];
  return evaluator === "abstention" ? `metric.abstention_${type}` : `metric.${type}`;
}

/** A metric's name, its description and which way is better on hover, and
 * an arrow for the direction (named for screen readers). */
export function MetricName({ metric }: { metric: string }) {
  const { t } = useI18n();
  const direction = higherIsBetter(metric);
  const hint = t(direction === null ? "evaluate.noDirection" : direction ? "evaluate.higherBetter" : "evaluate.lowerBetter");
  const key = metricKey(metric);
  const described = t(key);
  const help = described === key ? hint : `${described} ${hint}.`;
  return (
    <span className="metric-name" title={help}>
      <span className="mono">{metric}</span>
      {direction !== null ? (
        <span className="metric-direction" aria-hidden="true">
          {direction ? "↑" : "↓"}
        </span>
      ) : null}
      <span className="visually-hidden">{help}</span>
    </span>
  );
}

/** Whether a delta counts, and which way: with a Holm-adjusted p-value
 * (a comparison that adjusts for multiplicity), when it is below the
 * comparison's alpha; else (an older comparison, a delta without per-query
 * pairs, a closed-book arm) when its interval excludes zero. */
export function deltaVerdict(metric: string, delta: Delta, alpha?: number | null): "better" | "worse" | "neutral" {
  const direction = higherIsBetter(metric);
  if (direction === null) return "neutral";
  let sign: number;
  if (delta.adjusted_p_value != null && alpha != null) {
    sign = delta.adjusted_p_value < alpha ? Math.sign(delta.mean_delta) : 0;
  } else {
    sign = delta.ci_low > 0 ? 1 : delta.ci_high < 0 ? -1 : 0;
  }
  if (sign > 0) return direction ? "better" : "worse";
  if (sign < 0) return direction ? "worse" : "better";
  return "neutral";
}

/** The interval a row draws: the family's simultaneous one when the
 * comparison adjusted it, with its confidence, else the plain one. */
export function deltaInterval(delta: Delta, multiplicity: Multiplicity | null | undefined, confidence: number): { low: number; high: number; confidence: number; adjusted: boolean } {
  const family = delta.family ? multiplicity?.families[delta.family] : undefined;
  if (family && delta.adjusted_ci_low != null && delta.adjusted_ci_high != null) {
    return { low: delta.adjusted_ci_low, high: delta.adjusted_ci_high, confidence: family.interval_confidence, adjusted: true };
  }
  return { low: delta.ci_low, high: delta.ci_high, confidence, adjusted: false };
}

/** One row per metric: the mean paired delta and its interval on the
 * row's own scale (metrics differ in unit), zero marked; with a
 * comparison that adjusts for multiplicity, the Holm p-value and the
 * family's simultaneous interval, the primary metric marked. With
 * `noFinding` (a closed-book or A/A arm) no row is marked better or worse. */
export function DeltaPlot({ deltas, confidence = 0.95, multiplicity, noFinding = false }: { deltas: [string, Delta][]; confidence?: number; multiplicity?: Multiplicity | null; noFinding?: boolean }) {
  const { t, number, percent } = useI18n();
  const width = 180;
  const pad = 10;
  const adjusted = deltas.some(([, d]) => d.adjusted_p_value != null) && multiplicity != null;
  const alpha = multiplicity?.alpha ?? null;
  // The kind of interval the rows show, named in its header.
  const intervalKind = adjusted ? t("chart.intervalAdjusted") : t("chart.interval", { confidence: percent(confidence) });
  return (
    <div className="table-wrap" tabIndex={0}>
      <table className="table table-sticky table-stack">
        <caption className="visually-hidden">
          {adjusted ? t("chart.deltaCaptionHolm", { alpha: number(alpha ?? 0.05) }) : t("chart.deltaCaption", { confidence: percent(confidence) })}
        </caption>
        <thead>
          <tr>
            <th>{t("evaluate.metric")}</th>
            <th>{t("chart.plot")}</th>
            <th className="num">{t("evaluate.delta")}</th>
            <th className="num">{intervalKind}</th>
            {adjusted ? <th className="num">{t("chart.holmP")}</th> : null}
            <th className="num">{t("evaluate.winsTiesLosses")}</th>
          </tr>
        </thead>
        <tbody>
          {deltas.map(([metric, delta]) => {
            const interval = deltaInterval(delta, multiplicity, confidence);
            const extent = Math.max(Math.abs(interval.low), Math.abs(interval.high), Math.abs(delta.mean_delta)) || 1;
            const x = (v: number) => pad + ((v + extent) / (2 * extent)) * (width - 2 * pad);
            // A closed-book or A/A arm: its deltas bound or measure the
            // noise, so none reads as better or worse.
            const verdict = noFinding ? "neutral" : deltaVerdict(metric, delta, alpha);
            const digits = Math.abs(delta.mean_delta) < 0.01 && delta.mean_delta !== 0 ? 4 : 3;
            // An --arm run's primary metric is tested in arm_primary.
            const primary = delta.family === "primary" || delta.family === "arm_primary";
            return (
              <tr key={metric}>
                <td>
                  <MetricName metric={metric} />
                  {primary ? (
                    <span className="badge" style={{ marginLeft: 6 }} title={t("chart.primaryHelp")}>
                      {t("chart.primary")}
                    </span>
                  ) : null}
                </td>
                <td data-label={t("chart.plot")}>
                  <svg className="chart" width={width} height={22} role="img" aria-label={t("chart.deltaRow", { metric, delta: number(delta.mean_delta, { maximumFractionDigits: digits }) })}>
                    <line className="zero" x1={x(0)} x2={x(0)} y1={2} y2={20} />
                    <line className={`ci ${verdict}`} x1={x(interval.low)} x2={x(interval.high)} y1={11} y2={11} />
                    <circle className={`ci-dot ${verdict}`} cx={x(delta.mean_delta)} cy={11} r={4.5} />
                  </svg>
                </td>
                <td className="num" data-label={t("evaluate.delta")}>
                  {delta.mean_delta > 0 ? "+" : ""}
                  {number(delta.mean_delta, { maximumFractionDigits: digits })}
                  {verdict !== "neutral" ? <span className={`badge ${verdict === "better" ? "badge-good" : "badge-bad"}`} style={{ marginLeft: 6 }}>{t(`chart.${verdict}`)}</span> : null}
                </td>
                <td className="num" data-label={intervalKind}>
                  [{number(interval.low, { maximumFractionDigits: digits })}, {number(interval.high, { maximumFractionDigits: digits })}]
                  {adjusted ? <span className="muted"> {interval.adjusted ? percent(interval.confidence, 1) : t("chart.unadjusted")}</span> : null}
                </td>
                {adjusted ? (
                  <td className="num" data-label={t("chart.holmP")}>{delta.adjusted_p_value != null ? number(delta.adjusted_p_value, { maximumFractionDigits: 3 }) : "—"}</td>
                ) : null}
                <td className="num" data-label={t("evaluate.winsTiesLosses")}>
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

// Advance of one character of the 12px data face, to centre a label.
const MONO_WIDTH = 7.2;
const tickText = (key: string) => (key.length > 6 ? key.slice(-5) : key);
// Bars are capped at 24px: the band's leftover stays air (dataviz specs).
const BAR_MAX = 24;
const BAR_GAP = 10;

/** Vertical bars over a category axis, one series, on a scale with its
 * maximum and (when it is a clean value) its half marked. */
export function Bars({ data, format, label }: { data: { key: string; value: number }[]; format: (v: number) => string; label: string }) {
  const [hover, setHover] = useState<{ key: string; x: number; y: number } | null>(null);
  const height = 140;
  const top = 20;
  const barWidth = Math.max(10, Math.min(BAR_MAX, 560 / Math.max(1, data.length) - BAR_GAP));
  const max = Math.max(...data.map((d) => d.value), 0) || 1;
  const y = (v: number) => height - (v / max) * (height - top);
  const half = !Number.isInteger(max) || max % 2 === 0;
  // The scale's labels sit in a gutter left of the first bar, so no bar
  // ever covers one.
  const gutter = Math.ceil(Math.max(format(max).length, half ? format(max / 2).length : 0) * MONO_WIDTH) + 12;
  const width = Math.max(220, gutter + data.length * (barWidth + BAR_GAP) + 8);
  return (
    <div style={{ position: "relative", overflowX: "auto" }}>
      <svg className="chart" width={width} height={height + 34} role="img" aria-label={label}>
        <line className="grid" x1={0} x2={width} y1={y(max)} y2={y(max)} />
        <text x={0} y={y(max) - 5}>{format(max)}</text>
        {half ? (
          <>
            <line className="grid" x1={0} x2={width} y1={y(max / 2)} y2={y(max / 2)} />
            <text x={0} y={y(max / 2) - 5}>{format(max / 2)}</text>
          </>
        ) : null}
        <line className="axis" x1={0} x2={width} y1={height} y2={height} />
        {data.map((d, i) => {
          const h = height - y(d.value);
          const x = gutter + BAR_GAP / 2 + i * (barWidth + BAR_GAP);
          return (
            <g
              key={d.key}
              onMouseEnter={(e) => setHover({ key: d.key, x: e.clientX, y: e.clientY })}
              onMouseLeave={() => setHover(null)}
            >
              <rect x={x - BAR_GAP / 2} y={0} width={barWidth + BAR_GAP} height={height} fill="transparent" />
              <path className="bar" d={roundedTop(x, height - h, barWidth, h)} />
              <text x={x + barWidth / 2 - (tickText(d.key).length * MONO_WIDTH) / 2} y={height + 16}>
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
