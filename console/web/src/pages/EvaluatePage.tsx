// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useState } from "react";
import { DeltaPlot } from "../components/charts";
import { JobPanel } from "../components/JobPanel";
import { StrategyPicker } from "../components/StrategyPicker";
import { Empty, ErrorText, Loading, Notice, PageHead, Sheet } from "../components/ui";
import { useI18n } from "../i18n/i18n";
import { errorMessage, get, send } from "../lib/api";
import type { Comparison, JobRecord, SavedRun, Status, StrategyInfo } from "../lib/types";
import { useApi } from "../lib/useApi";

interface DatasetInfo {
  ref: string;
  name: string;
  items: number;
}
interface DatasetPreview {
  items: number;
  unanswerable: number;
  feeds: Record<string, number>;
  sample: { id: string | null; question: string; answer: string | null; answerable: boolean }[];
}
interface RunListing {
  ref: string;
  label: string;
  job_id: string | null;
  modified: number;
  summaries: number;
  strategies: string[];
}

export function EvaluatePage({ status, onJobsChanged }: { status: Status | null; onJobsChanged: () => void }) {
  const { t, date, number } = useI18n();
  const datasets = useApi(() => get<{ datasets: DatasetInfo[] }>("datasets"), []);
  const strategies = useApi(() => get<{ strategies: StrategyInfo[] }>("strategies"), [status?.config_path]);
  const runs = useApi(() => get<{ runs: RunListing[] }>("evaluations"), []);
  const jobs = useApi(() => get<{ jobs: JobRecord[] }>("jobs"), []);
  const [dataset, setDataset] = useState<string>("");
  const [chosen, setChosen] = useState<string[]>(["local", "mix"]);
  const [repeats, setRepeats] = useState(1);
  const [closedBook, setClosedBook] = useState(false);
  const [pairwise, setPairwise] = useState(false);
  const [gates, setGates] = useState("");
  const [budget, setBudget] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [run, setRun] = useState<string | null>(null);

  const datasetRef = dataset || datasets.data?.datasets[0]?.ref || "";
  const preview = useApi(() => (datasetRef ? get<DatasetPreview>("datasets/preview", { ref: datasetRef }) : Promise.resolve(null)), [datasetRef]);
  const evalJobs = (jobs.data?.jobs ?? []).filter((j) => j.kind === "evaluate");
  const active = evalJobs.find((j) => j.status === "running" || j.status === "stopping");
  const shownJob = jobId ?? active?.id ?? null;
  const selectedRun = run ?? runs.data?.runs[0]?.ref ?? null;
  const total = (chosen.length + (closedBook ? 1 : 0)) * repeats;

  async function start() {
    setError(null);
    try {
      const job = await send<JobRecord>("jobs/evaluate", {
        dataset: datasetRef,
        strategies: chosen,
        repeats,
        closed_book: closedBook,
        pairwise_judge: pairwise,
        fail_if: gates.split("\n").map((g) => g.trim()).filter(Boolean),
        max_failure_rate: budget.trim() ? Number(budget) : null,
      });
      setJobId(job.id);
      onJobsChanged();
      void jobs.reload();
    } catch (e) {
      setError(errorMessage(t, e));
    }
  }

  return (
    <>
      <PageHead title={t("evaluate.title")} lede={t("evaluate.lede")} />
      <div className="grid-2">
        <Sheet title={t("evaluate.newRun")}>
          <div className="stack">
            <label className="field">
              <span className="field-label">{t("evaluate.dataset")}</span>
              <select className="select" value={datasetRef} disabled={!datasets.data} onChange={(e) => setDataset(e.target.value)}>
                {!datasets.data ? <option value="">{t("common.loading")}</option> : null}
                {(datasets.data?.datasets ?? []).map((d) => (
                  <option key={d.ref} value={d.ref}>
                    {d.ref.replace(/^workspace:/, "")} ({t("evaluate.items", { count: d.items })})
                  </option>
                ))}
              </select>
              <span className="field-help">{t("evaluate.datasetHelp")}</span>
            </label>
            {datasets.data && !datasets.data.datasets.length ? <Empty title={t("evaluate.noDatasets")}>{t("evaluate.noDatasetsHelp")}</Empty> : null}
            {preview.data ? (
              <div className="notice">
                <div>
                  {t("evaluate.previewItems", { count: preview.data.items })}
                  {preview.data.unanswerable ? ` · ${t("evaluate.previewUnanswerable", { count: preview.data.unanswerable })}` : ""}
                </div>
                <div className="field-help">
                  {Object.entries(preview.data.feeds)
                    .filter(([, n]) => n)
                    .map(([evaluator, n]) => t("evaluate.feeds", { evaluator, count: n }))
                    .join(" · ") || t("evaluate.feedsNone")}
                </div>
              </div>
            ) : null}
            <ErrorText error={preview.error} />
            <div className="field">
              <span className="field-label">{t("evaluate.strategies")}</span>
              {strategies.data ? <StrategyPicker strategies={strategies.data.strategies} value={chosen} onChange={setChosen} multiple /> : <Loading />}
              <span className="field-help">{chosen[0] ? t("evaluate.baselineIs", { strategy: chosen[0] }) : t("evaluate.strategiesHelp")}</span>
            </div>
            <div className="row" style={{ alignItems: "end" }}>
              <label className="field" style={{ width: 140 }}>
                <span className="field-label">{t("evaluate.repeats")}</span>
                <input className="input" type="number" min={1} max={20} value={repeats} onChange={(e) => setRepeats(Math.max(1, Math.min(20, Number(e.target.value) || 1)))} />
              </label>
            </div>
            <details>
              <summary className="field-help">{t("evaluate.advanced")}</summary>
              <div className="stack" style={{ marginTop: 12 }}>
                <label className="field" style={{ width: 180 }}>
                  <span className="field-label">{t("evaluate.budget")}</span>
                  <input className="input" inputMode="decimal" placeholder="1.0" value={budget} onChange={(e) => setBudget(e.target.value)} />
                </label>
                <label className="check">
                  <input type="checkbox" checked={closedBook} onChange={(e) => setClosedBook(e.target.checked)} />
                  <span>
                    {t("evaluate.closedBook")} <span className="field-help">{t("evaluate.closedBookHelp")}</span>
                  </span>
                </label>
                <label className="check">
                  <input type="checkbox" checked={pairwise} onChange={(e) => setPairwise(e.target.checked)} />
                  <span>
                    {t("evaluate.pairwise")} <span className="field-help">{t("evaluate.pairwiseHelp")}</span>
                  </span>
                </label>
                <label className="field">
                  <span className="field-label">{t("evaluate.gates")}</span>
                  <textarea className="textarea mono" rows={2} value={gates} placeholder="answer_match/token_f1:ci_high<-0.02" onChange={(e) => setGates(e.target.value)} />
                  <span className="field-help">{t("evaluate.gatesHelp")}</span>
                </label>
              </div>
            </details>
            {error ? <Notice tone="bad">{error}</Notice> : null}
            <div className="row">
              <button className="button button-primary" disabled={!datasetRef || !chosen.length || !!active} onClick={() => void start()}>
                {t("evaluate.start")}
              </button>
              <span className="field-help">{t("evaluate.answersCount", { count: total * (preview.data?.items ?? 0), runs: number(total) })}</span>
            </div>
          </div>
        </Sheet>
        <div className="stack">
          <Sheet title={t("evaluate.running")}>
            {shownJob ? (
              <JobPanel
                key={shownJob}
                jobId={shownJob}
                onEnded={(job) => {
                  onJobsChanged();
                  void runs.reload();
                  void jobs.reload();
                  // The run the job saved, once the list is read again.
                  void get<{ runs: RunListing[] }>("evaluations").then(({ runs: listed }) => {
                    const found = listed.find((r) => r.job_id === job.id);
                    if (found) setRun(found.ref);
                  });
                }}
              >
                {(job) => <GateResult job={job} />}
              </JobPanel>
            ) : (
              <Empty title={t("evaluate.noRunTitle")}>{t("evaluate.noRun")}</Empty>
            )}
          </Sheet>
          <Sheet title={t("evaluate.saved")} aside={<button className="button button-small" onClick={() => void runs.reload()}>{t("common.refresh")}</button>}>
            {!runs.data ? (
              runs.error ? <ErrorText error={runs.error} /> : <Loading />
            ) : runs.data.runs.length ? (
              <div className="table-wrap" tabIndex={0}>
                <table className="table">
                  <thead>
                    <tr>
                      <th>{t("evaluate.runName")}</th>
                      <th>{t("evaluate.strategies")}</th>
                      <th>{t("evaluate.when")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.data.runs.slice(0, 15).map((r) => (
                      <tr key={r.ref} aria-selected={r.ref === selectedRun}>
                        <td>
                          <button className="link-button mono" onClick={() => setRun(r.ref)}>
                            {r.label}
                          </button>
                        </td>
                        <td className="mono">{r.strategies.join(", ")}</td>
                        <td>{date(r.modified)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Empty title={t("evaluate.noSaved")}>{t("evaluate.noSavedHelp")}</Empty>
            )}
          </Sheet>
        </div>
      </div>
      {selectedRun ? <RunView key={selectedRun} refName={selectedRun} /> : null}
    </>
  );
}

function GateResult({ job }: { job: JobRecord }) {
  const { t } = useI18n();
  const result = job.result as { gate_failures?: string[]; failure_budget_breaches?: string[] } | null;
  if (!result || job.status === "running") return null;
  const gates = (job.params.fail_if as string[] | undefined) ?? [];
  return (
    <div className="stack" style={{ gap: 8 }}>
      {gates.length ? (
        result.gate_failures?.length ? (
          <Notice tone="bad" title={t("evaluate.gateFailed")}>
            {result.gate_failures.join("; ")}
          </Notice>
        ) : (
          <Notice tone="good" title={t("evaluate.gatePassed", { count: gates.length })} />
        )
      ) : null}
      {result.failure_budget_breaches?.length ? (
        <Notice tone="bad" title={t("evaluate.budgetBreached")}>
          {result.failure_budget_breaches.join("; ")}
        </Notice>
      ) : null}
    </div>
  );
}

function RunView({ refName }: { refName: string }) {
  const { t, number, percent, usd, seconds } = useI18n();
  const run = useApi(() => get<SavedRun>("evaluations/run", { ref: refName }), [refName]);
  const [candidate, setCandidate] = useState<string | null>(null);
  if (run.error) return <ErrorText error={run.error} />;
  if (!run.data) return <Loading />;
  const comparison: Comparison | null = run.data.comparison;
  const strategies = comparison ? Object.keys(comparison.strategies) : [...new Set(run.data.summaries.map((s) => s.strategy))];
  const metricRows = comparison
    ? [...new Set(strategies.flatMap((s) => Object.keys(comparison.strategies[s]?.metrics ?? {})))].sort()
    : [...new Set(run.data.summaries.flatMap((s) => Object.keys(s.metric_statistics ?? {})))].sort();
  const mean = (strategy: string, metric: string): number | null => {
    if (comparison) return comparison.strategies[strategy]?.metrics[metric]?.mean ?? null;
    const values = run.data!.summaries.filter((s) => s.strategy === strategy).map((s) => s.metric_statistics?.[metric]?.mean).filter((v): v is number => v != null);
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  };
  const candidates = comparison ? Object.keys(comparison.deltas_vs_baseline) : [];
  const shown = candidate ?? candidates[0] ?? null;
  const deltas = shown && comparison ? Object.entries({ ...comparison.deltas_vs_baseline[shown], ...(comparison.cost?.deltas_vs_baseline?.[shown] ?? {}) }) : [];
  const repeatOf = new Map<string, number>();
  const abstention = run.data.summaries.map((s) => {
    const repeat = (repeatOf.get(s.strategy) ?? 0) + 1;
    repeatOf.set(s.strategy, repeat);
    return { strategy: s.strategy, repeat, stats: s.abstention_statistics as Record<string, number | null> | null };
  });
  const pairwise = shown ? comparison?.pairwise_vs_baseline?.[shown] : undefined;
  const lengths = (pairwise as { length?: { buckets?: Record<string, { pairs: number; criteria: Record<string, { candidate_win_rate: number | null }> }> } } | undefined)?.length?.buckets;

  return (
    <div className="stack" style={{ marginTop: 24 }}>
      <h2 className="display" style={{ margin: 0, fontSize: "var(--text-xl)" }}>
        {t("evaluate.runTitle", { name: run.data.label })}
      </h2>
      <Sheet title={t("evaluate.results")} pad={false}>
        <div className="table-wrap" tabIndex={0}>
          <table className="table">
            <thead>
              <tr>
                <th>{t("evaluate.metric")}</th>
                {strategies.map((s) => (
                  <th key={s} className="num">
                    {s}
                    {comparison?.baseline === s ? ` (${t("evaluate.baseline")})` : ""}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metricRows.map((metric) => (
                <tr key={metric}>
                  <td className="mono">{metric}</td>
                  {strategies.map((s) => (
                    <td key={s} className="num">
                      {number(mean(s, metric), { maximumFractionDigits: 3 })}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Sheet>
      {comparison && shown ? (
        <Sheet
          title={t("evaluate.paired", { candidate: shown, baseline: comparison.baselines?.[shown] ?? comparison.baseline })}
          aside={
            candidates.length > 1 ? (
              <select className="select" value={shown} onChange={(e) => setCandidate(e.target.value)} aria-label={t("evaluate.candidate")}>
                {candidates.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            ) : null
          }
        >
          <p className="field-help" style={{ marginTop: 0 }}>
            {t("evaluate.pairedHelp", { resamples: intervalResamples(comparison, shown, number) })}{" "}
            {comparison.multiplicity && !comparison.multiplicity.unadjusted?.includes(shown)
              ? t("evaluate.pairedHolmHelp", { alpha: number(comparison.multiplicity.alpha) })
              : t("evaluate.pairedPlainHelp")}
          </p>
          <DeltaPlot deltas={deltas} confidence={comparison.bootstrap?.confidence ?? 0.95} multiplicity={comparison.multiplicity && !comparison.multiplicity.unadjusted?.includes(shown) ? comparison.multiplicity : null} />
          {comparison.warnings?.length ? (
            <div style={{ marginTop: 12 }}>
              <Notice tone="warn" title={t("evaluate.warnings")}>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {comparison.warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              </Notice>
            </div>
          ) : null}
        </Sheet>
      ) : (
        <Notice title={t("evaluate.noComparison")}>{t("evaluate.noComparisonHelp")}</Notice>
      )}
      <div className="grid-wide">
        {comparison?.cost ? (
          <Sheet title={t("evaluate.cost")} pad={false}>
            <div className="table-wrap" tabIndex={0}>
              <table className="table">
                <thead>
                  <tr>
                    <th>{t("evaluate.strategy")}</th>
                    <th className="num">{t("evaluate.usdPerQuery")}</th>
                    <th className="num">{t("evaluate.tokensPerQuery")}</th>
                    <th className="num">{t("evaluate.latencyP50")}</th>
                    <th className="num">{t("evaluate.latencyP95")}</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(comparison.cost.strategies).map(([s, c]) => {
                    const usdQ = (c.usd_per_query as { mean?: number } | undefined)?.mean;
                    const tok = (c.total_tokens_per_query as { mean?: number } | undefined)?.mean;
                    const lat = c.latency_seconds as { p50?: number; p95?: number } | undefined;
                    return (
                      <tr key={s}>
                        <td className="mono">{s}</td>
                        <td className="num">{usdQ != null ? usd(usdQ) : "—"}</td>
                        <td className="num">{number(tok, { maximumFractionDigits: 0 })}</td>
                        <td className="num">{seconds(lat?.p50)}</td>
                        <td className="num">{seconds(lat?.p95)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Sheet>
        ) : null}
        <Sheet title={t("evaluate.abstention")} pad={false}>
          <div className="table-wrap" tabIndex={0}>
            <table className="table">
              <thead>
                <tr>
                  <th>{t("evaluate.strategy")}</th>
                  <th className="num">{t("evaluate.declined")}</th>
                  <th className="num">{t("evaluate.declineRate")}</th>
                  <th className="num">{t("evaluate.precision")}</th>
                  <th className="num">{t("evaluate.recall")}</th>
                </tr>
              </thead>
              <tbody>
                {abstention.map(({ strategy, repeat, stats }) => (
                  <tr key={`${strategy}-${repeat}`}>
                    <td className="mono">
                      {strategy}
                      {(repeatOf.get(strategy) ?? 1) > 1 ? <span className="muted"> · {t("evaluate.repeatN", { n: repeat })}</span> : null}
                    </td>
                    <td className="num">{number(stats?.abstained ?? null)}</td>
                    <td className="num">{percent(stats?.abstention_rate ?? null, 1)}</td>
                    <td className="num">{percent(stats?.precision ?? null, 1)}</td>
                    <td className="num">{percent(stats?.recall ?? null, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="field-help" style={{ padding: "0 20px" }}>
            {t("evaluate.abstentionHelp")}
          </p>
        </Sheet>
        <Sheet title={t("evaluate.lengths")} pad={!lengths}>
          {lengths ? (
            <div className="table-wrap" tabIndex={0}>
              <table className="table">
                <thead>
                  <tr>
                    <th>{t("evaluate.lengthRatio")}</th>
                    <th className="num">{t("evaluate.pairs")}</th>
                    {Object.keys(Object.values(lengths)[0]?.criteria ?? {}).map((c) => (
                      <th key={c} className="num">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(lengths).map(([bucket, row]) => (
                    <tr key={bucket}>
                      <td className="mono">{bucket}</td>
                      <td className="num">{number(row.pairs)}</td>
                      {Object.values(row.criteria).map((c, i) => (
                        <td key={i} className="num">
                          {percent(c.candidate_win_rate, 0)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted" style={{ margin: 0 }}>
              {t("evaluate.lengthsNone")}
            </p>
          )}
        </Sheet>
      </div>
    </div>
  );
}

/** The resamples the shown intervals read: a Holm-adjusted comparison draws
 * each family's from its own count (at least the base one, more for a large
 * family), so the help names those, else the base count. */
export function intervalResamples(
  comparison: Comparison,
  shown: string,
  number: (value: number) => string,
): string {
  const base = comparison.bootstrap?.resamples ?? 2000;
  const multiplicity = comparison.multiplicity;
  const counts =
    multiplicity && !multiplicity.unadjusted?.includes(shown)
      ? Object.values(multiplicity.families).map((f) => f.resamples ?? base)
      : [];
  const low = Math.min(...counts, ...(counts.length ? [] : [base]));
  const high = Math.max(...counts, ...(counts.length ? [] : [base]));
  return low === high ? number(low) : `${number(low)}-${number(high)}`;
}
