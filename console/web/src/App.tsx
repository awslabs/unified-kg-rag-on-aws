// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useEffect, useRef, useState, type ReactElement } from "react";
import { LocalePicker } from "./components/LocalePicker";
import { ThemePicker } from "./components/ThemePicker";
import { useI18n } from "./i18n/i18n";
import { get } from "./lib/api";
import { navigate, PAGES, useRoute, type Page } from "./lib/router";
import type { NamespaceHeld, Status } from "./lib/types";
import { useApi } from "./lib/useApi";
import { useCompactNav } from "./lib/useCompactNav";
import { AskPage } from "./pages/AskPage";
import { DashboardPage } from "./pages/DashboardPage";
import { EvaluatePage } from "./pages/EvaluatePage";
import { GraphPage } from "./pages/GraphPage";
import { IngestPage } from "./pages/IngestPage";
import { SettingsPage } from "./pages/SettingsPage";

export function App() {
  const { t, locale } = useI18n();
  const route = useRoute();
  const status = useApi(() => get<Status>("status"), [], { intervalMs: 5000, poll: true });
  const main = useRef<HTMLElement>(null);
  const running = status.data?.jobs.length ?? 0;
  const [menuOpen, setMenuOpen] = useState(false);
  const navRow = useRef<HTMLDivElement>(null);
  // The row compacts in steps (tagline, field labels) and folds the links
  // into the menu only when it still does not fit, whatever the width.
  const fold = useCompactNav(navRow, [locale, running > 0, status.data?.namespace]);
  // Whether the namespace holds an index (read again as jobs start and end).
  const held = useApi(
    () => (status.data?.namespace ? get<NamespaceHeld>("namespace/methodologies").catch(() => null) : Promise.resolve(null)),
    [status.data?.namespace, status.data?.config_path, running],
  );
  const indexed = held.data ? held.data.indexed : null;

  useEffect(() => {
    document.title = `${t(`nav.${route.page}`)} · ${t("app.name")}`;
    main.current?.focus({ preventScroll: true });
  }, [route.page, t]);


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
      <header className="nav" data-open={menuOpen} data-fold={fold}>
        <div className="nav-inner" ref={navRow}>
          <a
            className="brand"
            href="/ask"
            onClick={(event) => {
              if (event.metaKey || event.ctrlKey || event.shiftKey) return;
              event.preventDefault();
              navigate("ask");
            }}
          >
            <svg className="brand-mark" viewBox="0 0 28 28" aria-hidden="true">
              <line x1="6" y1="21" x2="14" y2="7" />
              <line x1="14" y1="7" x2="22" y2="19" />
              <line x1="6" y1="21" x2="22" y2="19" />
              <circle cx="6" cy="21" r="3.5" />
              <circle cx="22" cy="19" r="3.5" />
              <circle className="lit" cx="14" cy="7" r="3.5" />
            </svg>
            <span className="brand-text">
              <span className="brand-name">{t("app.name")}</span>
              <span className="brand-sub">{t("app.tagline")}</span>
            </span>
          </a>
          <button type="button" className="nav-menu" aria-expanded={menuOpen} aria-controls="nav-pages" onClick={() => setMenuOpen((open) => !open)}>
            {t("nav.menu")}
          </button>
          <nav aria-label={t("nav.label")} style={{ display: "contents" }}>
            <ul className="nav-list" id="nav-pages">
              {PAGES.map((name) => (
                <li key={name}>
                  <a
                    className="nav-link"
                    href={`/${name}`}
                    aria-current={route.page === name ? "page" : undefined}
                    onClick={(event) => {
                      if (event.metaKey || event.ctrlKey || event.shiftKey) return;
                      event.preventDefault();
                      setMenuOpen(false);
                      navigate(name);
                    }}
                  >
                    <PageIcon page={name} />
                    {t(`nav.${name}`)}
                    {name === "ingest" && running ? <span className="nav-note">{t("nav.running", { count: running })}</span> : null}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
          <div className="nav-tools">
            {status.data?.namespace ? (
              <span className="namespace-chip" data-indexed={indexed === null ? undefined : String(indexed)} title={indexed === false ? t("shell.namespaceEmpty") : t("shell.namespace")}>
                <span className="visually-hidden">{indexed === false ? t("shell.namespaceEmpty") : t("shell.namespace")}</span>
                {status.data.namespace}
              </span>
            ) : null}
            <ThemePicker />
            <LocalePicker />
          </div>
        </div>
      </header>
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
      {status.data ? (
        <footer className="shell-foot">
          <dl className="kv backend">
            <div>
              <dt>{t("shell.namespace")}</dt>
              <dd className="mono">{status.data.namespace ?? "—"}</dd>
            </div>
            <div>
              <dt>{t("shell.models")}</dt>
              <dd className="mono" title={Object.entries(status.data.models ?? {}).map(([role, id]) => `${role}: ${id ?? "—"}`).join("\n")}>
                {modelIds(status.data.models) || status.data.backend?.llm || "—"}
              </dd>
            </div>
            <div>
              <dt>{t("shell.stores")}</dt>
              <dd className="mono">{status.data.backend ? `${status.data.backend.graph} · ${status.data.backend.vector}` : "—"}</dd>
            </div>
          </dl>
        </footer>
      ) : null}
    </div>
  );
}

/** The distinct model ids the config's roles use, in role order. */
export function modelIds(models: Record<string, string | null> | null): string {
  return [...new Set(Object.values(models ?? {}).filter((id): id is string => !!id))].join(" · ");
}

/** A small line icon per page (drawn here, 24-unit grid). */
function PageIcon({ page }: { page: Page }) {
  const paths: Record<Page, string> = {
    ask: "M4 5h16v11H9l-5 4z M8 10h8 M8 13h5",
    ingest: "M12 15V4 M7 9l5-5 5 5 M4 15v4h16v-4",
    evaluate: "M5 19V11 M10 19V5 M15 19v-6 M20 19V8 M3 19h18",
    graph: "M6 18a2 2 0 1 0 0-.01 M18 18a2 2 0 1 0 0-.01 M12 6a2 2 0 1 0 0-.01 M7.5 16.5l3.5-8.5 M13 8l3.5 8.5 M8 18h8",
    dashboard: "M4 4h7v7H4z M13 4h7v4h-7z M13 10h7v10h-7z M4 13h7v7H4z",
    settings: "M4 7h10 M18 7h2 M16 5v4 M4 17h4 M12 17h8 M10 15v4",
  };
  return (
    <svg className="nav-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d={paths[page]} />
    </svg>
  );
}
