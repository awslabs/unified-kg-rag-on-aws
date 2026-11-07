// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * The provenance trace: the sources an answer read, ranked, hung on one
 * rail. Opening a source draws its trail: the passages that state it (arcs
 * to their stations), the file and chunk they come from, seed-graph
 * evidence, and the way into the graph.
 */
import { useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useI18n } from "../i18n/i18n";
import type { Source } from "../lib/types";
import { Highlighted } from "./Highlighted";

const KINDS = ["all", "text", "entity", "relationship", "community"] as const;
// Sources shown before "Show all".
const FIRST = 10;

/** An excerpt without the header lines the card already shows (the
 * context's "Entity:", "Type:" and "Relationship:" lines). */
export function excerptBody(source: Source): string {
  if (source.kind !== "entity" && source.kind !== "relationship") return source.excerpt;
  return source.excerpt
    .split("\n")
    .filter((line) => !/^(Entity|Type|Relationship|Path):/.test(line))
    .map((line) => line.replace(/^Description:\s*/, ""))
    .join("\n");
}
type KindFilter = (typeof KINDS)[number];

export interface EvidenceTraceProps {
  sources: Source[];
  question: string;
  onOpenInGraph?: (source: Source) => void;
}

/** The ranks of the graph sources that cite passage `rank`. */
export function citedBy(sources: Source[], rank: number): number[] {
  return sources.filter((s) => s.supported_by?.includes(rank)).map((s) => s.rank);
}

export function EvidenceTrace({ sources, question, onOpenInGraph }: EvidenceTraceProps) {
  const { t, number } = useI18n();
  const [filter, setFilter] = useState<KindFilter>("all");
  const [open, setOpen] = useState<number | null>(null);
  const container = useRef<HTMLDivElement>(null);
  const [arcs, setArcs] = useState<string[]>([]);
  const [all, setAll] = useState(false);

  const counts = useMemo(() => {
    const c: Record<string, number> = { all: sources.length };
    for (const s of sources) c[s.kind] = (c[s.kind] ?? 0) + 1;
    return c;
  }, [sources]);

  const opened = sources.find((s) => s.rank === open) ?? null;
  const linked = useMemo(() => {
    if (!opened) return new Set<number>();
    if (opened.kind === "text") return new Set(citedBy(sources, opened.rank));
    return new Set(opened.supported_by ?? []);
  }, [opened, sources]);

  const matching = sources.filter((s) => filter === "all" || s.kind === filter || s.rank === open || linked.has(s.rank));
  const shown = all ? matching : matching.filter((s, i) => i < FIRST || s.rank === open || linked.has(s.rank));

  // Arcs from the opened source's station to each linked one, drawn left of
  // the rail so the rail stays readable.
  useLayoutEffect(() => {
    const root = container.current;
    if (!root || open === null) {
      setArcs([]);
      return;
    }
    const box = root.getBoundingClientRect();
    const station = (rank: number) => {
      const card = root.querySelector<HTMLElement>(`[data-rank="${rank}"]`);
      if (!card) return null;
      const r = card.getBoundingClientRect();
      return r.top - box.top + 21;
    };
    const from = station(open);
    if (from === null) {
      setArcs([]);
      return;
    }
    const paths: string[] = [];
    [...linked].forEach((rank, index) => {
      const to = station(rank);
      if (to === null) return;
      const bulge = 4 - Math.min(index, 3) * 1;
      paths.push(`M 19 ${from} C ${bulge} ${from}, ${bulge} ${to}, 19 ${to}`);
    });
    setArcs(paths);
  }, [open, linked, shown.length]);

  if (!sources.length) {
    return (
      <div className="empty">
        <div className="empty-title">{t("ask.trace.noSourcesTitle")}</div>
        <div>{t("ask.trace.noSources")}</div>
      </div>
    );
  }

  return (
    <div>
      <div className="filter-row" role="group" aria-label={t("ask.trace.filter")}>
        {KINDS.filter((k) => k === "all" || counts[k]).map((kind) => (
          <button key={kind} className="chip" aria-pressed={filter === kind} onClick={() => setFilter(kind)}>
            {t(`ask.kind.${kind}`)} <span className="chip-meta">{number(counts[kind] ?? 0)}</span>
          </button>
        ))}
      </div>
      <div className="trace" ref={container} style={{ paddingLeft: 38 }}>
        <span className="trace-rail" style={{ left: 18 }} aria-hidden="true" />
        <svg className="trace-svg" aria-hidden="true">
          {arcs.map((d) => (
            <path key={d} d={d} />
          ))}
        </svg>
        <div className="trace-origin">
          <span className="visually-hidden">{t("ask.trace.origin")}</span>
          {t("ask.trace.read", { count: sources.length })}
        </div>
        <ol className="stack" style={{ listStyle: "none", margin: 0, padding: 0, gap: 8 }}>
          {shown.map((source) => (
            <li key={source.rank}>
              <SourceCard
                source={source}
                question={question}
                open={open === source.rank}
                linked={linked.has(source.rank)}
                sources={sources}
                onToggle={() => setOpen(open === source.rank ? null : source.rank)}
                onJump={(rank) => setOpen(rank)}
                onOpenInGraph={onOpenInGraph}
              />
            </li>
          ))}
        </ol>
        {shown.length < matching.length ? (
          <button className="button button-small" style={{ marginTop: 8 }} onClick={() => setAll(true)}>
            {t("ask.trace.showAll", { count: matching.length })}
          </button>
        ) : null}
      </div>
    </div>
  );
}

function SourceCard({
  source,
  question,
  open,
  linked,
  sources,
  onToggle,
  onJump,
  onOpenInGraph,
}: {
  source: Source;
  question: string;
  open: boolean;
  linked: boolean;
  sources: Source[];
  onToggle: () => void;
  onJump: (rank: number) => void;
  onOpenInGraph?: (source: Source) => void;
}) {
  const { t, number } = useI18n();
  const id = `source-${source.rank}`;
  const title = source.title ?? source.file_name ?? source.id ?? t("ask.trace.untitled");
  const seeded = source.provenance;
  return (
    <div className={`source${linked ? " linked" : ""}`} data-rank={source.rank} data-kind={source.kind} aria-expanded={open}>
      <span className="station" aria-hidden="true" />
      <button
        className="link-button"
        style={{ textDecoration: "none", color: "inherit", textAlign: "left", display: "grid", gap: 6 }}
        aria-expanded={open}
        aria-controls={`${id}-trail`}
        onClick={onToggle}
      >
        <span className="source-head">
          <span className="source-rank">#{source.rank}</span>
          <span className="source-title">{title}</span>
          <span className="source-kind">{t(`ask.kind.${source.kind}`)}</span>
          {source.relation ? <span className="badge">{source.relation}</span> : null}
          {seeded ? <span className="badge badge-trace">{t("ask.trace.seed")}</span> : null}
          {source.truncated ? <span className="badge badge-warn">{t("ask.trace.truncated")}</span> : null}
        </span>
        <p className="source-excerpt">
          <Highlighted text={excerptBody(source)} query={question} />
        </p>
      </button>
      {open ? (
        <div className="source-trail" id={`${id}-trail`}>
          {source.kind === "text" ? (
            <>
              <Step label={t("ask.trail.file")}>
                <span className="mono">{source.file_name ?? "—"}</span>
                {source.chunk_id != null ? <span className="id">{t("ask.trail.chunk", { id: String(source.chunk_id) })}</span> : null}
              </Step>
              <Step label={t("ask.trail.citedBy")}>
                <RankLinks ranks={citedBy(sources, source.rank)} onJump={onJump} none={t("ask.trail.citedByNone")} />
              </Step>
            </>
          ) : null}
          {source.kind === "entity" || source.kind === "relationship" ? (
            <>
              <Step label={t("ask.trail.statedIn")}>
                <RankLinks ranks={source.supported_by ?? []} onJump={onJump} none={t("ask.trail.statedInNone")} />
                {source.unread_passages ? (
                  <span className="muted">{t("ask.trail.unread", { count: source.unread_passages })}</span>
                ) : null}
              </Step>
              {source.entity_type ? <Step label={t("ask.trail.type")}>{source.entity_type}</Step> : null}
            </>
          ) : null}
          {seeded ? (
            <Step label={t("ask.trail.seedGraph")}>
              <span>{seeded.seed_graphs.join(", ") || "—"}</span>
              <span className="badge">{seeded.kind === "seed_graph" ? t("ask.trail.seedOnly") : t("ask.trail.seedAndDocument")}</span>
            </Step>
          ) : null}
          {source.iri || seeded?.iri ? (
            <Step label="IRI">
              <span className="id">{source.iri ?? seeded?.iri}</span>
            </Step>
          ) : null}
          {source.source_kinds.length ? <Step label={t("ask.trail.sourceKinds")}>{source.source_kinds.join(", ")}</Step> : null}
          <Step label={t("ask.trail.ids")}>
            <span className="id">{source.id}</span>
            {source.text_unit_ids.length ? <span className="id">{t("ask.trail.units", { count: source.text_unit_ids.length })}</span> : null}
          </Step>
          {source.score != null ? <Step label={t("ask.trail.score")}>{number(source.score, { maximumFractionDigits: 4 })}</Step> : null}
          {onOpenInGraph && source.kind !== "text" ? (
            <div>
              <button className="button button-small button-primary" onClick={() => onOpenInGraph(source)}>
                {source.kind === "community" ? t("ask.trail.openCommunity") : t("ask.trail.openGraph")}
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function Step({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="trail-step">
      <span className="trail-step-label">{label}</span>
      {children}
    </div>
  );
}

function RankLinks({ ranks, onJump, none }: { ranks: number[]; onJump: (rank: number) => void; none: string }) {
  if (!ranks.length) return <span className="muted">{none}</span>;
  return (
    <span className="row" style={{ gap: 6 }}>
      {ranks.map((rank) => (
        <button key={rank} className="link-button mono" onClick={() => onJump(rank)}>
          #{rank}
        </button>
      ))}
    </span>
  );
}
