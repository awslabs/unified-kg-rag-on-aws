// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import type { ReactNode } from "react";
import { useI18n } from "../i18n/i18n";
import { errorMessage } from "../lib/api";
import type { JobRecord } from "../lib/types";

export function PageHead({ title, lede, aside }: { title: string; lede?: ReactNode; aside?: ReactNode }) {
  return (
    <header className="page-head">
      <div>
        <h1 className="page-title display">{title}</h1>
        {lede ? <p className="page-lede">{lede}</p> : null}
      </div>
      {aside ? <div className="page-head-aside">{aside}</div> : null}
    </header>
  );
}

export function Sheet({
  title,
  aside,
  sub,
  children,
  pad = true,
}: {
  title?: ReactNode;
  aside?: ReactNode;
  /** A line under the title (what the sheet is about). */
  sub?: ReactNode;
  children: ReactNode;
  pad?: boolean;
}) {
  return (
    <section className="sheet">
      {title ? (
        <div className="sheet-head">
          <h2 className="sheet-title">{title}</h2>
          {aside ? <div style={{ marginLeft: "auto" }}>{aside}</div> : null}
          {sub ? <p className="sheet-sub">{sub}</p> : null}
        </div>
      ) : null}
      <div className={pad ? "sheet-body" : undefined}>{children}</div>
    </section>
  );
}

export function Notice({ tone = "info", title, children }: { tone?: "info" | "warn" | "bad" | "good"; title?: ReactNode; children?: ReactNode }) {
  const className = tone === "info" ? "notice" : `notice notice-${tone}`;
  return (
    <div className={className} role={tone === "bad" ? "alert" : "status"}>
      {title ? <div className="notice-title">{title}</div> : null}
      {children ? <div>{children}</div> : null}
    </div>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty-title">{title}</div>
      {children ? <div>{children}</div> : null}
      {action}
    </div>
  );
}

const JOB_TONES: Record<string, string> = {
  running: "badge-trace",
  stopping: "badge-warn",
  succeeded: "badge-good",
  failed: "badge-bad",
  stopped: "badge-warn",
  // An ingestion stopped at its budget: a warning, as its notice is.
  budget: "badge-warn",
};

/** Whether an ingestion stopped at its budget (processing.max_run_*:
 * error class limit_exceeded), which the Ingest page says with its own
 * notice, offering to resume it, instead of the generic failure. */
export function isBudgetStop(job: JobStatusOf): boolean {
  return job.kind === "ingest" && job.status === "failed" && (job.result?.error_class ?? job.error_class) === "limit_exceeded";
}

/** What a job's status reads: its record (the error class in its result),
 * or a dashboard row (the error class beside it). */
export type JobStatusOf = Pick<JobRecord, "kind" | "status"> & {
  result?: JobRecord["result"];
  error_class?: string | null;
};

/** A job's status: an ingestion's budget stop as its own (a warning, not
 * a failure). */
export function JobStatusBadge({ job }: { job: JobStatusOf }) {
  const { t } = useI18n();
  const status = isBudgetStop(job) ? "budget" : job.status;
  return <span className={`badge ${JOB_TONES[status] ?? ""}`}>{t(`job.status.${status}`)}</span>;
}

export function Loading() {
  const { t } = useI18n();
  return (
    <p className="muted loading" role="status">
      {t("common.loading")}
    </p>
  );
}

export function ErrorText({ error }: { error: unknown }) {
  const { t } = useI18n();
  if (!error) return null;
  return (
    <Notice tone="bad" title={t("common.requestFailed")}>
      {errorMessage(t, error)}
    </Notice>
  );
}

/** The previous and next buttons of a paged table, and its row range. */
export function Pager({ page, size, total, onPage }: { page: number; size: number; total: number; onPage: (page: number) => void }) {
  const { t, number } = useI18n();
  const pages = Math.max(1, Math.ceil(total / size));
  if (pages <= 1) return null;
  return (
    <div className="row pager">
      <button className="button button-small" disabled={page === 0} onClick={() => onPage(page - 1)}>
        {t("common.previous")}
      </button>
      <span className="field-help" role="status">
        {t("common.range", { from: number(page * size + 1), to: number(Math.min(total, (page + 1) * size)), total: number(total) })}
      </span>
      <button className="button button-small" disabled={page >= pages - 1} onClick={() => onPage(page + 1)}>
        {t("common.next")}
      </button>
    </div>
  );
}

export function formatBytes(bytes: number, number: (v: number, o?: Intl.NumberFormatOptions) => string): string {
  const units = ["byte", "kilobyte", "megabyte", "gigabyte"] as const;
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  // "180 bytes": a byte count reads in full ("short" leaves "180 byte").
  return number(value, { style: "unit", unit: units[unit], unitDisplay: unit ? "short" : "long", maximumFractionDigits: unit ? 1 : 0 });
}
