// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useRef, useState } from "react";
import { AnswerText, CitationList } from "../components/AnswerText";
import { EvidenceTrace } from "../components/EvidenceTrace";
import { StrategyPicker } from "../components/StrategyPicker";
import { Notice, PageHead, Sheet } from "../components/ui";
import { useI18n } from "../i18n/i18n";
import { errorMessage, get, stream } from "../lib/api";
import { navigate } from "../lib/router";
import { useTrail } from "../lib/trail";
import type { AskEvent, AskResult, Citation, Source, Status, StrategyInfo } from "../lib/types";
import { useApi } from "../lib/useApi";

interface Turn {
  id: number;
  question: string;
  requested: string;
  text: string;
  status: "streaming" | "done" | "error" | "stopped";
  result?: AskResult;
  error?: { kind: string; message: string };
  traceUrl?: string | null;
}

// The earlier turns a follow-up sends (the chain bounds them further by
// memory.max_messages_per_conversation and memory.max_history_tokens).
const HISTORY_TURNS = 25;

/** The conversation a follow-up carries (`RAGInput.history`): each earlier
 * question and its final answer, oldest first. A turn without an answer
 * (failed, stopped, blocked) is left out: it is no history. */
export function conversationHistory(turns: { question: string; result?: AskResult; status: string }[]): { role: "user" | "assistant"; content: string }[] {
  return turns
    .filter((turn) => turn.status === "done" && turn.result && !turn.result.blocked && !turn.result.error)
    .slice(-HISTORY_TURNS)
    .flatMap((turn) => [
      { role: "user" as const, content: turn.question },
      { role: "assistant" as const, content: turn.result?.answer ?? "" },
    ]);
}

/** Filters typed as `key: value` lines (a number or true/false is typed). */
export function parseFilters(text: string): Record<string, string | number | boolean> | null {
  const filters: Record<string, string | number | boolean> = {};
  for (const line of text.split("\n")) {
    const [key, ...rest] = line.split(":");
    const value = rest.join(":").trim();
    if (!key?.trim() || !value) continue;
    filters[key.trim()] = value === "true" ? true : value === "false" ? false : /^-?\d+(\.\d+)?$/.test(value) ? Number(value) : value;
  }
  return Object.keys(filters).length ? filters : null;
}

export function AskPage({ status }: { status: Status | null }) {
  const { t } = useI18n();
  const { setTrail } = useTrail();
  const strategies = useApi(() => get<{ strategies: StrategyInfo[] }>("strategies"), [status?.config_path]);
  const [query, setQuery] = useState("");
  const [strategy, setStrategy] = useState("auto");
  const [followUp, setFollowUp] = useState(true);
  const [topK, setTopK] = useState(10);
  const [searchType, setSearchType] = useState("hybrid");
  const [filterText, setFilterText] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  // The source open in the trace (its rank), set by an answer's [n].
  const [openRank, setOpenRank] = useState<number | null>(null);
  const abort = useRef<AbortController | null>(null);
  const counter = useRef(0);
  const busy = turns.some((turn) => turn.status === "streaming");

  const update = (id: number, change: (turn: Turn) => Turn) =>
    setTurns((all) => all.map((turn) => (turn.id === id ? change(turn) : turn)));

  async function ask(question: string) {
    const text = question.trim();
    if (!text || busy) return;
    counter.current += 1;
    const id = counter.current;
    const history = followUp ? conversationHistory(turns) : [];
    setTurns((all) => [...all, { id, question: text, requested: strategy, text: "", status: "streaming" }]);
    setSelected(id);
    setOpenRank(null);
    setQuery("");
    const controller = new AbortController();
    abort.current = controller;
    try {
      for await (const event of stream<AskEvent>(
        "ask",
        {
          query: text,
          search_strategy: strategy,
          search_type: searchType,
          top_k: topK,
          history: history.length ? history : null,
          filters: parseFilters(filterText),
        },
        controller.signal,
      )) {
        if (event.kind === "start") update(id, (turn) => ({ ...turn, traceUrl: event.trace_url }));
        else if (event.kind === "chunk") update(id, (turn) => ({ ...turn, text: turn.text + event.text }));
        else if (event.kind === "result") {
          update(id, (turn) => ({ ...turn, status: "done", result: event, text: event.answer }));
        } else if (event.kind === "error") update(id, (turn) => ({ ...turn, status: "error", error: { kind: event.error_kind, message: event.message } }));
      }
    } catch (e) {
      if (controller.signal.aborted) update(id, (turn) => ({ ...turn, status: "stopped" }));
      else update(id, (turn) => ({ ...turn, status: "error", error: { kind: "failure", message: errorMessage(t, e) } }));
    } finally {
      // A stream that ended with neither its result nor an error (the
      // server stopped) leaves no turn waiting.
      update(id, (turn) => (turn.status === "streaming" ? { ...turn, status: "error", error: { kind: "incomplete", message: t("ask.incomplete") } } : turn));
      abort.current = null;
    }
  }

  const current = turns.find((turn) => turn.id === selected) ?? turns[turns.length - 1];
  const select = (id: number, rank: number | null = null) => {
    setSelected(id);
    setOpenRank(rank);
  };
  const cited = new Set((current?.result?.citations ?? []).filter((c) => c.cited).map((c) => c.marker));
  const openInGraph = (source: Source) => {
    if (!current) return;
    setTrail({ question: current.question, source, sources: current.result?.sources ?? [] });
    const params: Record<string, string> = {};
    if (source.kind === "community" && source.community_id) params.community = source.community_id;
    else if (source.kind === "relationship" && source.node_ids?.[0]) {
      params.focus = source.node_ids[0];
      if (source.node_ids[1]) params.edge = source.node_ids[1];
    } else if (source.kind === "query_result") {
      // The graph query's first returned entity.
      if (source.node_ids?.[0]) params.focus = source.node_ids[0];
    } else if (source.id) params.focus = source.id;
    navigate("graph", params);
  };

  return (
    <>
      <PageHead
        title={t("ask.title")}
        lede={t("ask.lede")}
        aside={
          turns.length ? (
            <button
              className="button button-small"
              onClick={() => {
                setTurns([]);
                setSelected(null);
                setOpenRank(null);
              }}
            >
              {t("ask.newConversation")}
            </button>
          ) : null
        }
      />
      <div className="grid-2">
        <div className="conversation">
          <div className="thread" aria-live="polite">
            {!turns.length ? (
              <div className="welcome">
                <svg className="welcome-mark" viewBox="0 0 28 28" aria-hidden="true">
                  <line x1="6" y1="21" x2="14" y2="7" />
                  <line x1="14" y1="7" x2="22" y2="19" />
                  <line x1="6" y1="21" x2="22" y2="19" />
                  <circle cx="6" cy="21" r="3.5" />
                  <circle cx="22" cy="19" r="3.5" />
                  <circle className="lit" cx="14" cy="7" r="3.5" />
                </svg>
                <p className="welcome-title">{t("ask.evidenceEmptyTitle")}</p>
                {status?.sample_questions.length ? (
                  <>
                    <span>{t("ask.samples")}</span>
                    <div className="samples">
                      {status.sample_questions.map((sample) => (
                        <button key={sample} type="button" className="sample" onClick={() => void ask(sample)}>
                          {sample}
                        </button>
                      ))}
                    </div>
                  </>
                ) : null}
              </div>
            ) : null}
            {turns.map((turn) => (
              <TurnView
                key={turn.id}
                turn={turn}
                selected={turn.id === current?.id}
                onSelect={() => select(turn.id)}
                onOpenCitation={(citation) => select(turn.id, citation.source_rank)}
              />
            ))}
          </div>
          <div className="composer-dock">
            <section className="sheet">
              <form
                className="composer"
                onSubmit={(event) => {
                  event.preventDefault();
                  void ask(query);
                }}
              >
                <label className="visually-hidden" htmlFor="question">
                  {t("ask.question")}
                </label>
                <textarea
                  id="question"
                  className="textarea"
                  value={query}
                  placeholder={t("ask.placeholder")}
                  onChange={(event) => setQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                      event.preventDefault();
                      void ask(query);
                    }
                  }}
                  rows={3}
                />
                <div className="composer-options">
                  <details>
                    <summary>
                      {t("strategy.label")}: <strong>{strategy}</strong>
                    </summary>
                    {strategies.data ? (
                      <StrategyPicker strategies={strategies.data.strategies} value={[strategy]} onChange={(v) => setStrategy(v[0] ?? "auto")} />
                    ) : null}
                  </details>
                  <details>
                    <summary>{t("ask.options")}</summary>
                    <div className="grid-cols" style={{ gap: 12 }}>
                      <label className="field">
                        <span className="field-label">{t("ask.topK")}</span>
                        <input className="input" type="number" min={1} max={100} value={topK} onChange={(e) => setTopK(Math.max(1, Math.min(100, Number(e.target.value) || 10)))} />
                      </label>
                      <label className="field">
                        <span className="field-label">{t("ask.searchType")}</span>
                        <select className="select" value={searchType} onChange={(e) => setSearchType(e.target.value)}>
                          {["hybrid", "lexical", "vector"].map((v) => (
                            <option key={v} value={v}>
                              {t(`ask.searchType.${v}`)}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="field">
                        <span className="field-label">{t("ask.filters")}</span>
                        <textarea className="textarea mono" rows={2} value={filterText} placeholder="file_name: report.pdf" onChange={(e) => setFilterText(e.target.value)} />
                        <span className="field-help">{t("ask.filtersHelp")}</span>
                      </label>
                    </div>
                  </details>
                </div>
                {strategies.error ? (
                  <Notice tone="bad" title={t("ask.strategiesFailed")}>
                    {errorMessage(t, strategies.error)}
                  </Notice>
                ) : null}
                <div className="composer-actions">
                  <label className="check">
                    <input type="checkbox" checked={followUp} onChange={(e) => setFollowUp(e.target.checked)} />
                    <span>{t("ask.followUp")}</span>
                  </label>
                  <div className="row">
                    {busy ? (
                      <button type="button" className="button" onClick={() => abort.current?.abort()}>
                        {t("ask.stop")}
                      </button>
                    ) : null}
                    <button type="submit" className="button button-primary" disabled={!query.trim() || busy}>
                      {t("ask.submit")}
                    </button>
                  </div>
                </div>
              </form>
            </section>
          </div>
        </div>
        <div className="sticky">
        <Sheet title={t("ask.evidence")} sub={current?.result ? t("ask.evidenceFor", { question: truncate(current.question, 72) }) : null}>
          {current?.result ? (
            <div className="evidence-scroll" tabIndex={0} aria-label={t("ask.evidence")}>
              <EvidenceTrace
                key={current.id}
                sources={current.result.sources}
                question={current.question}
                onOpenInGraph={openInGraph}
                cited={cited}
                open={openRank}
                onOpenChange={setOpenRank}
              />
            </div>
          ) : (
            <div className="empty">
              <div className="empty-title">{current ? t("ask.evidenceWaiting") : t("ask.evidenceEmptyTitle")}</div>
              <div>{t("ask.evidenceEmpty")}</div>
            </div>
          )}
        </Sheet>
        </div>
      </div>
    </>
  );
}

function truncate(text: string, length: number): string {
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}

function TurnView({
  turn,
  selected,
  onSelect,
  onOpenCitation,
}: {
  turn: Turn;
  selected: boolean;
  onSelect: () => void;
  onOpenCitation: (citation: Citation) => void;
}) {
  const { t, number, usd, seconds } = useI18n();
  const result = turn.result;
  return (
    <article className="turn" aria-label={t("ask.turn", { question: truncate(turn.question, 60) })}>
      <div className="question">{turn.question}</div>
      <section className="sheet answer" style={selected ? { borderColor: "var(--trace)" } : undefined}>
        {result?.routing && turn.requested === "auto" ? (
          <p className="route-note">
            {t(`ask.routing.${result.routing.decided_by}`, {
              strategy: result.strategy ?? "",
              candidates: result.routing.candidates.join(", "),
            })}{" "}
            <span className="field-help">{t(`strategy.help.${result.strategy}`)}</span>
          </p>
        ) : null}
        {turn.error ? (
          <Notice tone="bad" title={t(`ask.error.${turn.error.kind}`)}>
            {turn.error.message}
          </Notice>
        ) : (
          <p className={`answer-text${turn.status === "streaming" ? " streaming" : ""}`}>
            {turn.text ? (
              <AnswerText text={turn.text} citations={result?.citations} onOpen={onOpenCitation} />
            ) : turn.status === "streaming" ? (
              <span className="muted">{t("ask.retrieving")}</span>
            ) : (
              ""
            )}
          </p>
        )}
        {result && !turn.error ? <CitationList citations={result.citations} onOpen={onOpenCitation} /> : null}
        {turn.status === "stopped" ? <p className="muted">{t("ask.stopped")}</p> : null}
        {result?.abstained ? (
          <div style={{ marginTop: 12 }}>
            <Notice tone="warn" title={t("ask.abstained")}>
              {t(`ask.abstention.${result.abstention_reason ?? "unknown"}`)}
            </Notice>
          </div>
        ) : null}
        {result?.blocked ? (
          <div style={{ marginTop: 12 }}>
            <Notice tone="warn" title={t("ask.blocked")} />
          </div>
        ) : null}
        {result?.error ? (
          <div style={{ marginTop: 12 }}>
            <Notice tone="bad" title={t("ask.degraded")}>
              {result.error_message}
            </Notice>
          </div>
        ) : null}
        {result ? (
          <div className="answer-meta">
            <span className="meta-item">
              {t("ask.meta.strategy")} <strong>{result.strategy}</strong>
            </span>
            <span className="meta-item">
              {t("ask.meta.latency")} <strong>{seconds(result.latency_seconds)}</strong>
            </span>
            {result.usage ? (
              <span className="meta-item" title={t("ask.meta.tokensHelp")}>
                {t("ask.meta.tokens")}{" "}
                <strong>
                  {number(result.usage.input_tokens)} / {number(result.usage.output_tokens)}
                </strong>
              </span>
            ) : null}
            {result.usage ? (
              <span className="meta-item">
                {t("ask.meta.calls")} <strong>{number(result.usage.llm_calls)}</strong>
              </span>
            ) : null}
            <span className="meta-item">
              {t("ask.meta.cost")}{" "}
              <strong>{result.cost_usd != null ? usd(result.cost_usd) : result.prices_configured ? t("ask.meta.unpriced") : t("ask.meta.noPrices")}</strong>
            </span>
            <span className="meta-item">
              {t("ask.meta.rerank")} <strong>{result.rerank ? t(`rerank.${result.rerank.status}`) : "—"}</strong>
            </span>
            {!selected ? (
              <button className="link-button meta-item" onClick={onSelect}>
                {t("ask.showEvidence")}
              </button>
            ) : null}
            {turn.traceUrl ?? result.trace_url ? (
              <a className="meta-item" href={(turn.traceUrl ?? result.trace_url) as string} target="_blank" rel="noreferrer noopener">
                {t("ask.meta.trace")}
              </a>
            ) : result.trace_id ? (
              <span className="meta-item">
                {t("ask.meta.traceId")} <strong>{result.trace_id.slice(0, 16)}</strong>
              </span>
            ) : null}
          </div>
        ) : null}
      </section>
    </article>
  );
}
