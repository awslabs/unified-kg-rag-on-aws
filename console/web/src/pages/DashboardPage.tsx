// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useState } from "react";
import { Bars } from "../components/charts";
import { Empty, ErrorText, JobStatusBadge, Loading, Notice, PageHead, Sheet } from "../components/ui";
import { useI18n } from "../i18n/i18n";
import { get, send } from "../lib/api";
import { navigate } from "../lib/router";
import { useApi } from "../lib/useApi";

interface Counts {
  llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  cache_hits: number;
  failed_calls: number;
  rerank_calls: number;
}

interface Dashboard {
  totals: {
    queries: number;
    answered: number;
    abstained: number;
    errors: number;
    latency_p50: number | null;
    latency_p95: number | null;
    cost_usd: number | null;
    priced_queries: number;
  };
  strategies: {
    strategy: string;
    queries: number;
    abstained: number;
    errors: number;
    latency_p50: number | null;
    latency_p95: number | null;
    tokens_per_query: number;
    cost_usd: number | null;
    cost_per_query: number | null;
  }[];
  models: ({ model: string; cost_usd: number | null } & Counts)[];
  roles: ({ role: string } & Counts)[];
  timeline: { day: string; queries: number; cost_usd: number; latency_p50: number | null }[];
  jobs: { id: string; kind: string; status: string; created: number; duration_seconds: number | null; llm_calls: number | null; tokens: number | null; cost_usd: number | null }[];
  unpriced_models: string[];
  prices_configured: boolean;
}

const RANGES = [1, 7, 30, 365];

export function DashboardPage() {
  const { t, number, usd, seconds, percent, date } = useI18n();
  const [days, setDays] = useState(30);
  const data = useApi(() => get<Dashboard>("dashboard", { days }), [days]);
  const d = data.data;

  return (
    <>
      <PageHead
        title={t("dashboard.title")}
        lede={t("dashboard.lede")}
        aside={
          <div className="chips" role="group" aria-label={t("dashboard.range")}>
            {RANGES.map((r) => (
              <button key={r} className="chip" aria-pressed={days === r} onClick={() => setDays(r)}>
                {t("dashboard.days", { count: r })}
              </button>
            ))}
          </div>
        }
      />
      <ErrorText error={data.error} />
      {!d ? (
        <Loading />
      ) : !d.totals.queries && !d.jobs.length ? (
        <Empty
          title={t("dashboard.emptyTitle")}
          action={
            <button className="button button-primary" onClick={() => navigate("ask")}>
              {t("dashboard.goAsk")}
            </button>
          }
        >
          {t("dashboard.empty")}
        </Empty>
      ) : (
        <div className="stack" style={{ gap: 20 }}>
          <div className="stat-row">
            <Stat label={t("dashboard.questions")} value={number(d.totals.queries)} />
            <Stat label={t("dashboard.latencyP50")} value={seconds(d.totals.latency_p50)} />
            <Stat label={t("dashboard.latencyP95")} value={seconds(d.totals.latency_p95)} />
            <Stat label={t("dashboard.declineRate")} value={percent(d.totals.queries ? d.totals.abstained / d.totals.queries : null, 1)} />
            <Stat label={t("dashboard.errors")} value={number(d.totals.errors)} />
            <Stat label={t("dashboard.cost")} value={d.totals.cost_usd != null ? usd(d.totals.cost_usd) : "—"} />
          </div>
          {!d.prices_configured ? (
            <Notice title={t("dashboard.noPricesTitle")}>{t("dashboard.noPrices")}</Notice>
          ) : d.unpriced_models.length ? (
            <Notice tone="warn" title={t("dashboard.unpricedTitle")}>
              {t("dashboard.unpriced", { models: d.unpriced_models.join(", ") })}
            </Notice>
          ) : null}
          <div className="grid-cols">
            <Sheet title={t("dashboard.perDay")}>
              <Bars data={d.timeline.map((r) => ({ key: r.day, value: r.queries }))} format={(v) => number(v)} label={t("dashboard.perDay")} />
            </Sheet>
            <Sheet title={t("dashboard.latencyPerDay")}>
              <Bars data={d.timeline.map((r) => ({ key: r.day, value: r.latency_p50 ?? 0 }))} format={(v) => seconds(v)} label={t("dashboard.latencyPerDay")} />
            </Sheet>
          </div>
          <Sheet title={t("dashboard.byStrategy")} pad={false}>
            <div className="table-wrap" tabIndex={0}>
              <table className="table">
                <thead>
                  <tr>
                    <th>{t("evaluate.strategy")}</th>
                    <th className="num">{t("dashboard.questions")}</th>
                    <th className="num">{t("dashboard.latencyP50")}</th>
                    <th className="num">{t("dashboard.latencyP95")}</th>
                    <th className="num">{t("dashboard.tokensPerQuestion")}</th>
                    <th className="num">{t("dashboard.costPerQuestion")}</th>
                    <th className="num">{t("dashboard.declined")}</th>
                    <th className="num">{t("dashboard.errors")}</th>
                  </tr>
                </thead>
                <tbody>
                  {d.strategies.map((s) => (
                    <tr key={s.strategy}>
                      <td className="mono">{s.strategy}</td>
                      <td className="num">{number(s.queries)}</td>
                      <td className="num">{seconds(s.latency_p50)}</td>
                      <td className="num">{seconds(s.latency_p95)}</td>
                      <td className="num">{number(s.tokens_per_query, { maximumFractionDigits: 0 })}</td>
                      <td className="num">{s.cost_per_query != null ? usd(s.cost_per_query) : "—"}</td>
                      <td className="num">{number(s.abstained)}</td>
                      <td className="num">{number(s.errors)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Sheet>
          <div className="grid-cols">
            <UsageTable title={t("dashboard.byModel")} rows={d.models.map((m) => ({ name: m.model, ...m }))} showCost />
            <UsageTable title={t("dashboard.byRole")} rows={d.roles.map((r) => ({ name: r.role, cost_usd: null, ...r }))} />
          </div>
          <Sheet title={t("dashboard.jobs")} pad={false}>
            {d.jobs.length ? (
              <div className="table-wrap" tabIndex={0}>
                <table className="table">
                  <thead>
                    <tr>
                      <th>{t("job.started")}</th>
                      <th>{t("dashboard.kind")}</th>
                      <th>{t("job.statusLabel")}</th>
                      <th className="num">{t("dashboard.duration")}</th>
                      <th className="num">{t("dashboard.calls")}</th>
                      <th className="num">{t("dashboard.tokens")}</th>
                      <th className="num">{t("dashboard.cost")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.jobs.map((j) => (
                      <tr key={j.id}>
                        <td>{date(j.created)}</td>
                        <td>{t(`dashboard.kind.${j.kind}`)}</td>
                        <td>
                          <JobStatusBadge status={j.status} />
                        </td>
                        <td className="num">{seconds(j.duration_seconds)}</td>
                        <td className="num">{number(j.llm_calls)}</td>
                        <td className="num">{number(j.tokens)}</td>
                        <td className="num">{j.cost_usd != null ? usd(j.cost_usd) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted" style={{ padding: "0 20px" }}>
                {t("dashboard.noJobs")}
              </p>
            )}
          </Sheet>
          <div>
            <button
              className="button button-small button-danger"
              onClick={async () => {
                if (!window.confirm(t("dashboard.clearConfirm"))) return;
                await send("history", undefined, "DELETE");
                await data.reload();
              }}
            >
              {t("dashboard.clear")}
            </button>
            <span className="field-help" style={{ marginLeft: 12 }}>
              {t("dashboard.clearHelp")}
            </span>
          </div>
        </div>
      )}
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span className="stat-value display">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

function UsageTable({ title, rows, showCost = false }: { title: string; rows: ({ name: string; cost_usd: number | null } & Counts)[]; showCost?: boolean }) {
  const { t, number, usd } = useI18n();
  return (
    <Sheet title={title} pad={false}>
      <div className="table-wrap" tabIndex={0}>
        <table className="table">
          <thead>
            <tr>
              <th>{t("dashboard.name")}</th>
              <th className="num">{t("dashboard.calls")}</th>
              <th className="num">{t("dashboard.inputTokens")}</th>
              <th className="num">{t("dashboard.outputTokens")}</th>
              <th className="num">{t("dashboard.cacheHits")}</th>
              {showCost ? <th className="num">{t("dashboard.cost")}</th> : null}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name}>
                <td className="mono">{r.name}</td>
                <td className="num">{number(r.llm_calls)}</td>
                <td className="num">{number(r.input_tokens)}</td>
                <td className="num">{number(r.output_tokens)}</td>
                <td className="num">{number(r.cache_hits)}</td>
                {showCost ? <td className="num">{r.cost_usd != null ? usd(r.cost_usd) : "—"}</td> : null}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Sheet>
  );
}
