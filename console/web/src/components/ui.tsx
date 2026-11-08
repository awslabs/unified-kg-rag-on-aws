// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import type { ReactNode } from "react";
import { useI18n } from "../i18n/i18n";
import { errorMessage } from "../lib/api";

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
};

export function JobStatusBadge({ status }: { status: string }) {
  const { t } = useI18n();
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

export function formatBytes(bytes: number, number: (v: number, o?: Intl.NumberFormatOptions) => string): string {
  const units = ["byte", "kilobyte", "megabyte", "gigabyte"] as const;
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return number(value, { style: "unit", unit: units[unit], unitDisplay: "short", maximumFractionDigits: unit ? 1 : 0 });
}
