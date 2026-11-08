// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useEffect, useRef, type ReactElement } from "react";
import { LocalePicker } from "./components/LocalePicker";
import { useI18n } from "./i18n/i18n";
import { get } from "./lib/api";
import { navigate, PAGES, useRoute, type Page } from "./lib/router";
import type { Status } from "./lib/types";
import { useApi } from "./lib/useApi";
import { AskPage } from "./pages/AskPage";
import { DashboardPage } from "./pages/DashboardPage";
import { EvaluatePage } from "./pages/EvaluatePage";
import { GraphPage } from "./pages/GraphPage";
import { IngestPage } from "./pages/IngestPage";
import { SettingsPage } from "./pages/SettingsPage";

export function App() {
  const { t } = useI18n();
  const route = useRoute();
  const status = useApi(() => get<Status>("status"), [], { intervalMs: 5000, poll: true });
  const main = useRef<HTMLElement>(null);
  const running = status.data?.jobs.length ?? 0;

  useEffect(() => {
    document.title = `${t(`nav.${route.page}`)} · ${t("app.name")}`;
    main.current?.focus({ preventScroll: true });
  }, [route.page, t]);

  // On a narrow screen the nav is a strip that scrolls sideways: keep the
  // current page's station in view.
  useEffect(() => {
    const link = document.querySelector<HTMLElement>('.nav-link[aria-current="page"]');
    const list = link?.closest<HTMLElement>(".nav-list");
    if (!link || !list || list.scrollWidth <= list.clientWidth) return;
    list.scrollLeft = link.offsetLeft - (list.clientWidth - link.offsetWidth) / 2;
  }, [route.page]);

  const page: Record<Page, ReactElement> = {
    ask: <AskPage status={status.data} />,
    ingest: <IngestPage status={status.data} onJobsChanged={status.reload} />,
    evaluate: <EvaluatePage status={status.data} onJobsChanged={status.reload} />,
    graph: <GraphPage params={route.params} />,
    dashboard: <DashboardPage />,
    settings: <SettingsPage onSaved={status.reload} />,
  };

  return (
    <div className="shell">
      <a className="skip-link" href="#main">
        {t("app.skip")}
      </a>
      <nav className="nav" aria-label={t("nav.label")}>
        <div className="brand">
          <svg className="brand-mark" viewBox="0 0 28 28" aria-hidden="true">
            <line x1="6" y1="21" x2="14" y2="7" />
            <line x1="14" y1="7" x2="22" y2="19" />
            <line x1="6" y1="21" x2="22" y2="19" />
            <circle cx="6" cy="21" r="3.5" />
            <circle cx="22" cy="19" r="3.5" />
            <circle className="lit" cx="14" cy="7" r="3.5" />
          </svg>
          <span className="brand-name">{t("app.name")}</span>
          <span className="brand-sub">{t("app.tagline")}</span>
        </div>
        <ul className="nav-list">
          {PAGES.map((name) => (
            <li key={name}>
              <a
                className="nav-link"
                href={`/${name}`}
                aria-current={route.page === name ? "page" : undefined}
                onClick={(event) => {
                  if (event.metaKey || event.ctrlKey || event.shiftKey) return;
                  event.preventDefault();
                  navigate(name);
                }}
              >
                {t(`nav.${name}`)}
                {name === "ingest" && running ? <span className="nav-note">{t("nav.running", { count: running })}</span> : null}
              </a>
            </li>
          ))}
        </ul>
        <div className="nav-foot">
          {status.data ? (
            <dl className="kv backend">
              <dt>{t("shell.namespace")}</dt>
              <dd className="mono">{status.data.namespace ?? "—"}</dd>
              <dt>{t("shell.models")}</dt>
              <dd className="mono">{status.data.backend?.llm ?? "—"}</dd>
              <dt>{t("shell.stores")}</dt>
              <dd className="mono">{status.data.backend ? `${status.data.backend.graph} · ${status.data.backend.vector}` : "—"}</dd>
            </dl>
          ) : null}
          <LocalePicker />
        </div>
      </nav>
      <main id="main" className="main" ref={main} tabIndex={-1}>
        {status.data?.config_error && route.page !== "settings" ? (
          <div className="notice notice-bad" role="alert" style={{ marginBottom: 16 }}>
            <div className="notice-title">{t("shell.configError")}</div>
            <div>{status.data.config_error}</div>
            <div>
              <button className="link-button" onClick={() => navigate("settings")}>
                {t("shell.openSettings")}
              </button>
            </div>
          </div>
        ) : null}
        {page[route.page]}
      </main>
    </div>
  );
}
