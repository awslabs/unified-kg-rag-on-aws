// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useState, type ReactNode } from "react";
import { useI18n } from "../i18n/i18n";
import { get, send } from "../lib/api";
import type { JobRecord, StageProgress } from "../lib/types";
import { useApi } from "../lib/useApi";
import { ErrorText, JobStatusBadge, Notice } from "./ui";

const ACTIVE = new Set(["running", "stopping"]);

/** One job, polled while it runs: progress, stop, log tail, the result. */
export function JobPanel({ jobId, onEnded, children }: { jobId: string; onEnded?: (job: JobRecord) => void; children?: (job: JobRecord) => ReactNode }) {
  const { t, seconds, date, number } = useI18n();
  const [stopping, setStopping] = useState(false);
  const [ended, setEnded] = useState(false);
  const job = useApi(
    async () => {
      const record = await get<JobRecord>(`jobs/${encodeURIComponent(jobId)}`);
      if (!ACTIVE.has(record.status) && !ended) {
        setEnded(true);
        onEnded?.(record);
      }
      return record;
    },
    [jobId],
    { intervalMs: 1500, poll: !ended },
  );
  const data = job.data;
  if (!data) return job.error ? <ErrorText error={job.error} /> : <p className="muted">{t("common.loading")}</p>;
  const progress = data.progress;
  // An evaluation moves by its answers (its runs are saved as they are
  // scored: an interleaved one only at the end).
  const bar = progress?.answers ?? progress;
  const interleaved = data.kind === "evaluate" && data.params.run_order === "interleaved";
  const running = ACTIVE.has(data.status);
  const elapsed = data.started && data.ended ? data.ended - data.started : null;
  return (
    <div className="stack">
      <div className="row">
        <JobStatusBadge status={data.status} />
        <span className="id">{data.id}</span>
        <span className="muted">{data.started ? date(data.started) : ""}</span>
        <span className="muted">{seconds(elapsed)}</span>
        {running ? (
          <button
            className="button button-small button-danger"
            style={{ marginLeft: "auto" }}
            disabled={data.status === "stopping" || stopping}
            onClick={async () => {
              setStopping(true);
              try {
                await send(`jobs/${encodeURIComponent(data.id)}/stop`);
                await job.reload();
              } finally {
                setStopping(false);
              }
            }}
          >
            {data.status === "stopping" ? t("job.stopping") : t("job.stop")}
          </button>
        ) : null}
      </div>
      {progress ? (
        <div className="stack" style={{ gap: 6 }}>
          <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={bar?.total || 1} aria-valuenow={bar?.done} aria-label={t("job.progress")}>
            <span style={{ width: `${bar?.total ? (100 * bar.done) / bar.total : 0}%` }} />
          </div>
          <span className="field-help">
            {data.kind === "ingest"
              ? t("job.stagesDone", { done: progress.done, total: progress.total })
              : [
                  progress.answers ? t("job.answersDone", { done: number(progress.answers.done), total: number(progress.answers.total) }) : null,
                  t("job.runsDone", { done: progress.done, total: progress.total }),
                ]
                  .filter(Boolean)
                  .join(" · ")}
          </span>
          {running && interleaved ? <span className="field-help">{t("job.interleavedHelp")}</span> : null}
        </div>
      ) : null}
      {progress?.stages ? <Stages stages={progress.stages} /> : null}
      {data.status === "stopped" ? (
        <Notice tone="warn" title={t("job.stoppedTitle")}>
          {data.kind === "ingest" ? t("job.stoppedIngest") : interleaved ? t("job.stoppedInterleaved") : t("job.stoppedEvaluate")}
        </Notice>
      ) : null}
      {data.status === "failed" ? (
        <Notice tone="bad" title={t("job.failedTitle")}>
          {data.error ?? t("job.failedNoMessage")}
        </Notice>
      ) : null}
      {children ? children(data) : null}
      {data.log?.length ? (
        <details open={data.status === "failed"}>
          <summary>{t("job.log")}</summary>
          <pre className="log" tabIndex={0}>{data.log.join("\n")}</pre>
        </details>
      ) : null}
      <details>
        <summary className="field-help">{t("job.command")}</summary>
        <pre className="log" tabIndex={0}>
          {[...Object.entries(data.env ?? {}).map(([k, v]) => `${k}=${v}`), data.command.join(" ")].join("\n")}
        </pre>
      </details>
    </div>
  );
}

export function Stages({ stages }: { stages: StageProgress[] }) {
  const { t, number, seconds } = useI18n();
  return (
    <ol className="stages" aria-label={t("job.stages")}>
      {stages.map((stage) => (
        <li key={stage.name} className="stage" data-status={stage.status}>
          <span className="stage-name">{stage.name}</span>
          <span className="stage-meta">
            {t(`stage.status.${stage.status}`)}
            {stage.duration_seconds != null ? ` · ${seconds(stage.duration_seconds)}` : ""}
            {stage.output_count != null ? ` · ${t("stage.out", { count: stage.output_count })}` : ""}
            {stage.llm_calls ? ` · ${t("stage.calls", { count: stage.llm_calls })}` : ""}
          </span>
          {stage.error ? <span className="stage-error">{stage.error}</span> : null}
          {stage.failed_items?.length ? (
            <span className="stage-error">
              {t("stage.failedItems", { count: stage.failed_items.length })}: {stage.failed_items.slice(0, 5).join("; ")}
              {stage.failed_items.length > 5 ? ` ${t("stage.andMore", { count: stage.failed_items.length - 5 })}` : ""}
            </span>
          ) : null}
          {stage.input_count != null && stage.output_count == null ? <span className="stage-meta">{number(stage.input_count)}</span> : null}
        </li>
      ))}
    </ol>
  );
}
