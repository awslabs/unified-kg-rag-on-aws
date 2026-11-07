// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useMemo, useState } from "react";
import { LocalePicker } from "../components/LocalePicker";
import { ErrorText, Loading, Notice, PageHead, Sheet } from "../components/ui";
import { useI18n } from "../i18n/i18n";
import { errorMessage, get, send } from "../lib/api";
import { fieldsOf, flatten, parseValue, valueAt, type JsonSchema, type SchemaField } from "../lib/schema";
import { useApi } from "../lib/useApi";

interface ConfigView {
  path: string | null;
  writable: boolean;
  sha: string;
  text: string | null;
  data: Record<string, unknown>;
  secrets_in_file: string[];
  parse_error: string | null;
  load_error: string | null;
  effective: Record<string, unknown> | null;
  methodologies: { available: string[]; all: string[] };
  tracing_environment: Record<string, boolean>;
  otel_installed: boolean;
}

interface Preview {
  text: string;
  diff: string;
  changed: boolean;
  comments_dropped: boolean;
  valid: boolean;
  errors: string[];
  warnings: string[];
  unknown_keys: string[];
  endpoint_changes: { path: string; from: unknown; to: unknown }[];
}

interface Change {
  path: string;
  value?: unknown;
  unset?: boolean;
}

type Tab = "fields" | "yaml" | "methodologies" | "observability" | "effective";

export function SettingsPage({ onSaved }: { onSaved: () => void }) {
  const { t } = useI18n();
  const view = useApi(() => get<ConfigView>("config"), []);
  const schema = useApi(() => get<JsonSchema>("config/schema"), []);
  const [tab, setTab] = useState<Tab>("fields");
  const [changes, setChanges] = useState<Record<string, Change>>({});
  const [yaml, setYaml] = useState<string | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [endpointsConfirmed, setEndpointsConfirmed] = useState(false);
  const fields = useMemo(() => (schema.data ? fieldsOf(schema.data) : []), [schema.data]);

  const setChange = (change: Change | null, path: string) =>
    setChanges((all) => {
      const next = { ...all };
      if (change) next[path] = change;
      else delete next[path];
      return next;
    });

  async function review(useText: boolean) {
    setPreviewing(true);
    setError(null);
    setSaved(false);
    setEndpointsConfirmed(false);
    try {
      setPreview(
        await send<Preview>("config/preview", useText ? { text: yaml ?? view.data?.text ?? "" } : { changes: Object.values(changes) }),
      );
    } catch (e) {
      setPreview(null);
      setError(errorMessage(t, e));
    } finally {
      setPreviewing(false);
    }
  }

  async function save() {
    if (!preview || !view.data) return;
    setError(null);
    try {
      await send("config/save", {
        text: preview.text,
        base_sha: view.data.sha,
        confirm_endpoints: endpointsConfirmed ? (preview.endpoint_changes ?? []).map((c) => c.path) : [],
      });
      setSaved(true);
      setChanges({});
      setYaml(null);
      setPreview(null);
      await view.reload();
      onSaved();
    } catch (e) {
      setError(errorMessage(t, e));
    }
  }

  if (view.error || !view.data || !schema.data)
    return (
      <>
        <PageHead title={t("settings.title")} />
        {view.error ? <ErrorText error={view.error} /> : <Loading />}
      </>
    );
  const data = view.data;
  const pending = Object.keys(changes).length;

  return (
    <>
      <PageHead
        title={t("settings.title")}
        lede={data.path ? t("settings.lede", { path: data.path }) : t("settings.ledeNoFile")}
        aside={<LocalePicker id="settings-locale" />}
      />
      {data.load_error ? (
        <Notice tone="bad" title={t("settings.loadError")}>
          {data.load_error}
        </Notice>
      ) : null}
      {data.secrets_in_file.length ? (
        <Notice tone="warn" title={t("settings.secretsTitle")}>
          {t("settings.secrets", { keys: data.secrets_in_file.join(", ") })}
        </Notice>
      ) : null}
      {saved ? <Notice tone="good" title={t("settings.saved")}>{t("settings.savedHelp")}</Notice> : null}
      <div className="grid-2" style={{ marginTop: 16 }}>
        <section className="sheet">
          <div className="tabs" role="tablist" aria-label={t("settings.sections")}>
            {(["fields", "yaml", "methodologies", "observability", "effective"] as Tab[]).map((name) => (
              <button key={name} role="tab" className="tab" aria-selected={tab === name} id={`settings-tab-${name}`} aria-controls="settings-panel" onClick={() => setTab(name)}>
                {t(`settings.tab.${name}`)}
              </button>
            ))}
          </div>
          <div className="sheet-body" role="tabpanel" id="settings-panel" aria-labelledby={`settings-tab-${tab}`}>
            {tab === "fields" ? <FieldsEditor fields={fields} data={data} changes={changes} setChange={setChange} /> : null}
            {tab === "yaml" ? (
              data.text === null ? (
                <Notice tone="warn">{t("settings.yamlHidden")}</Notice>
              ) : (
                <div className="stack">
                  <label className="visually-hidden" htmlFor="yaml">
                    {t("settings.tab.yaml")}
                  </label>
                  <textarea id="yaml" className="textarea mono" rows={22} spellCheck={false} value={yaml ?? data.text} onChange={(e) => setYaml(e.target.value)} />
                  <div className="row">
                    <button className="button button-primary" disabled={!data.writable || previewing} onClick={() => void review(true)}>
                      {t("settings.review")}
                    </button>
                    <span className="field-help">{t("settings.yamlHelp")}</span>
                  </div>
                </div>
              )
            ) : null}
            {tab === "methodologies" ? <MethodologyPanel data={data} changes={changes} setChange={setChange} /> : null}
            {tab === "observability" ? <ObservabilityPanel data={data} changes={changes} setChange={setChange} /> : null}
            {tab === "effective" ? (
              <div className="stack">
                <p className="field-help" style={{ margin: 0 }}>
                  {t("settings.effectiveHelp")}
                </p>
                <pre className="diff" tabIndex={0}>{JSON.stringify(data.effective, null, 2)}</pre>
              </div>
            ) : null}
          </div>
        </section>
        <div className="stack">
          <Sheet title={t("settings.pending")} aside={pending ? <span className="badge badge-trace">{t("settings.changes", { count: pending })}</span> : null}>
            {pending ? (
              <ul className="list-plain">
                {Object.values(changes).map((c) => (
                  <li key={c.path} className="row" style={{ justifyContent: "space-between" }}>
                    <span className="mono">{c.path}</span>
                    <span className="mono muted">{c.unset ? t("settings.unset") : JSON.stringify(c.value)}</span>
                    <button className="link-button" onClick={() => setChange(null, c.path)}>
                      {t("settings.undo")}
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted" style={{ margin: 0 }}>
                {t("settings.noChanges")}
              </p>
            )}
            <div className="row" style={{ marginTop: 12 }}>
              <button className="button button-primary" disabled={!pending || previewing || !data.writable} onClick={() => void review(false)}>
                {t("settings.review")}
              </button>
              {!data.writable ? <span className="field-help">{t("settings.notWritable")}</span> : null}
            </div>
          </Sheet>
          {error ? <Notice tone="bad" title={t("settings.refused")}>{error}</Notice> : null}
          {preview ? (
            <Sheet title={t("settings.diff")}>
              <div className="stack">
                {preview.valid ? (
                  <Notice tone="good" title={t("settings.valid")} />
                ) : (
                  <Notice tone="bad" title={t("settings.invalid")}>
                    <ul style={{ margin: 0, paddingLeft: 18 }}>
                      {preview.errors.map((e) => (
                        <li key={e}>{e}</li>
                      ))}
                    </ul>
                  </Notice>
                )}
                {preview.warnings.length ? (
                  <Notice tone="warn" title={t("settings.warnings")}>
                    <ul style={{ margin: 0, paddingLeft: 18 }}>
                      {preview.warnings.map((w) => (
                        <li key={w}>{w}</li>
                      ))}
                    </ul>
                  </Notice>
                ) : null}
                {preview.comments_dropped ? <Notice tone="warn">{t("settings.commentsDropped")}</Notice> : null}
                {(preview.endpoint_changes ?? []).length ? (
                  <Notice tone="warn" title={t("settings.endpointsTitle")}>
                    <ul style={{ margin: "0 0 8px", paddingLeft: 18 }}>
                      {(preview.endpoint_changes ?? []).map((c) => (
                        <li key={c.path} className="mono">
                          {c.path}: {JSON.stringify(c.from ?? null)} → {JSON.stringify(c.to ?? null)}
                        </li>
                      ))}
                    </ul>
                    <label className="check">
                      <input type="checkbox" checked={endpointsConfirmed} onChange={(e) => setEndpointsConfirmed(e.target.checked)} />
                      <span>{t("settings.endpointsConfirm")}</span>
                    </label>
                  </Notice>
                ) : null}
                <Diff text={preview.diff} />
                <div className="row">
                  <button
                    className="button button-primary"
                    disabled={!preview.valid || !preview.changed || ((preview.endpoint_changes ?? []).length > 0 && !endpointsConfirmed)}
                    onClick={() => void save()}
                  >
                    {t("settings.save", { file: data.path?.split("/").pop() ?? "config" })}
                  </button>
                  <button className="button" onClick={() => setPreview(null)}>
                    {t("settings.keepEditing")}
                  </button>
                </div>
              </div>
            </Sheet>
          ) : null}
        </div>
      </div>
    </>
  );
}

function Diff({ text }: { text: string }) {
  const { t } = useI18n();
  if (!text) return <p className="muted">{t("settings.noDiff")}</p>;
  return (
    <pre className="diff" tabIndex={0} aria-label={t("settings.diff")}>
      {text.split("\n").map((line, i) => {
        const cls = line.startsWith("+++") || line.startsWith("---") ? "hunk" : line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : line.startsWith("@@") ? "hunk" : undefined;
        return (
          <span key={i} className={cls}>
            {line}
            {"\n"}
          </span>
        );
      })}
    </pre>
  );
}

function FieldsEditor({
  fields,
  data,
  changes,
  setChange,
}: {
  fields: SchemaField[];
  data: ConfigView;
  changes: Record<string, Change>;
  setChange: (c: Change | null, path: string) => void;
}) {
  const { t } = useI18n();
  const [query, setQuery] = useState("");
  const all = useMemo(() => flatten(fields), [fields]);
  const q = query.trim().toLowerCase();
  const matches = q ? all.filter((f) => f.path.toLowerCase().includes(q) || f.description.toLowerCase().includes(q)).slice(0, 60) : [];
  return (
    <div className="stack">
      <label className="field">
        <span className="field-label">{t("settings.find")}</span>
        <input className="input" type="search" value={query} placeholder="search.top_k, rerank, language…" onChange={(e) => setQuery(e.target.value)} />
      </label>
      <div className="config-tree" tabIndex={0} aria-label={t("settings.tab.fields")}>
        {q ? (
          matches.length ? (
            matches.map((f) => <FieldEditor key={f.path} field={f} data={data} change={changes[f.path]} setChange={setChange} showPath />)
          ) : (
            <p className="muted">{t("settings.noMatch")}</p>
          )
        ) : (
          fields.map((f) => <Node key={f.path} field={f} data={data} changes={changes} setChange={setChange} />)
        )}
      </div>
    </div>
  );
}

function Node({ field, data, changes, setChange }: { field: SchemaField; data: ConfigView; changes: Record<string, Change>; setChange: (c: Change | null, path: string) => void }) {
  const [open, setOpen] = useState(false);
  if (field.kind !== "section") return <FieldEditor field={field} data={data} change={changes[field.path]} setChange={setChange} />;
  const changed = Object.keys(changes).some((p) => p.startsWith(`${field.path}.`));
  return (
    <details className="config-section" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary title={field.description}>
        {field.key}
        {changed ? " •" : ""}
      </summary>
      {open ? field.children.map((child) => <Node key={child.path} field={child} data={data} changes={changes} setChange={setChange} />) : null}
    </details>
  );
}

function display(value: unknown): string {
  if (value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function FieldEditor({
  field,
  data,
  change,
  setChange,
  showPath = false,
}: {
  field: SchemaField;
  data: ConfigView;
  change?: Change;
  setChange: (c: Change | null, path: string) => void;
  showPath?: boolean;
}) {
  const { t } = useI18n();
  const inFile = valueAt(data.data, field.path);
  const effective = valueAt(data.effective, field.path);
  const current = change ? (change.unset ? undefined : change.value) : inFile;
  const [text, setText] = useState(display(current ?? effective));
  const [problem, setProblem] = useState<string | null>(null);
  const id = `field-${field.path}`;

  const commit = (value: unknown) => {
    if (JSON.stringify(value) === JSON.stringify(inFile)) setChange(null, field.path);
    else setChange({ path: field.path, value }, field.path);
  };

  let editor;
  if (field.kind === "secret") {
    editor = <span className="field-help">{t("settings.secretField")}</span>;
  } else if (field.kind === "boolean") {
    editor = (
      <label className="check">
        <input id={id} type="checkbox" checked={Boolean(current ?? effective)} onChange={(e) => commit(e.target.checked)} />
        <span>{current ?? effective ? t("settings.on") : t("settings.off")}</span>
      </label>
    );
  } else if (field.kind === "enum") {
    editor = (
      <select id={id} className="select" value={display(current ?? effective)} onChange={(e) => commit(field.options.find((o) => display(o) === e.target.value) ?? e.target.value)}>
        {field.nullable ? <option value="">{t("settings.none")}</option> : null}
        {field.options.map((o) => (
          <option key={display(o)} value={display(o)}>
            {display(o)}
          </option>
        ))}
      </select>
    );
  } else {
    editor = (
      <input
        id={id}
        className="input mono"
        value={text}
        aria-invalid={problem ? true : undefined}
        aria-describedby={problem ? `${id}-error` : undefined}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => {
          if (text === display(current ?? effective)) return;
          const parsed = parseValue(field, text);
          if (parsed.error) setProblem(parsed.error);
          else {
            setProblem(null);
            commit(parsed.value);
          }
        }}
      />
    );
  }

  return (
    <div className={`config-field${change ? " changed" : ""}`}>
      <label className="config-key" htmlFor={id}>
        {showPath ? field.path : field.key}
      </label>
      {editor}
      {problem ? (
        <span className="field-help" id={`${id}-error`} style={{ color: "var(--bad)" }}>
          {t(problem)}
        </span>
      ) : null}
      {field.description ? <span className="field-help">{field.description}</span> : null}
      <span className="id">
        {inFile !== undefined ? t("settings.inFile") : t("settings.fromDefault")}
        {field.hasDefault ? ` · ${t("settings.default", { value: display(field.defaultValue) || "∅" })}` : ""}
        {inFile !== undefined && field.kind !== "secret" ? (
          <>
            {" · "}
            <button className="link-button" onClick={() => setChange({ path: field.path, unset: true }, field.path)}>
              {t("settings.reset")}
            </button>
          </>
        ) : null}
      </span>
    </div>
  );
}

function MethodologyPanel({ data, changes, setChange }: { data: ConfigView; changes: Record<string, Change>; setChange: (c: Change | null, path: string) => void }) {
  const { t } = useI18n();
  const path = "indexing.methodologies";
  const inFile = valueAt(data.data, path) as string[] | undefined;
  const current = (changes[path]?.value as string[] | undefined) ?? inFile ?? (valueAt(data.effective, path) as string[] | undefined) ?? [];
  return (
    <div className="stack">
      <p className="field-help" style={{ margin: 0 }}>
        {t("settings.methodologiesHelp")}
      </p>
      {data.methodologies.all.map((m) => {
        const available = data.methodologies.available.includes(m);
        return (
          <label key={m} className="check">
            <input
              type="checkbox"
              disabled={!available}
              checked={current.includes(m)}
              onChange={(e) => {
                const next = e.target.checked ? [...current, m] : current.filter((x) => x !== m);
                if (JSON.stringify(next) === JSON.stringify(inFile)) setChange(null, path);
                else setChange({ path, value: next }, path);
              }}
            />
            <span>
              <strong>{t(`methodology.${m}`)}</strong> <span className="field-help">{t(`methodology.help.${m}`)}</span>
              {!available ? <span className="field-help"> {t("settings.methodologyUnavailable")}</span> : null}
            </span>
          </label>
        );
      })}
      <Notice>{t("settings.methodologyRebuild")}</Notice>
    </div>
  );
}

function ObservabilityPanel({ data, changes, setChange }: { data: ConfigView; changes: Record<string, Change>; setChange: (c: Change | null, path: string) => void }) {
  const { t } = useI18n();
  const value = (key: string) => (changes[`observability.${key}`]?.value ?? valueAt(data.data, `observability.${key}`) ?? valueAt(data.effective, `observability.${key}`)) as unknown;
  const set = (key: string, v: unknown) => {
    const path = `observability.${key}`;
    if (JSON.stringify(v) === JSON.stringify(valueAt(data.data, path))) setChange(null, path);
    else setChange({ path, value: v }, path);
  };
  return (
    <div className="stack">
      <p className="field-help" style={{ margin: 0 }}>
        {t("settings.observabilityHelp")}
      </p>
      {!data.otel_installed ? <Notice tone="warn">{t("settings.otelMissing")}</Notice> : null}
      <label className="check">
        <input type="checkbox" checked={Boolean(value("enabled"))} onChange={(e) => set("enabled", e.target.checked)} />
        <span>{t("settings.tracingOn")}</span>
      </label>
      <label className="field">
        <span className="field-label">{t("settings.backend")}</span>
        <select className="select" value={String(value("backend") ?? "none")} onChange={(e) => set("backend", e.target.value)}>
          {["none", "otlp", "langfuse", "phoenix"].map((b) => (
            <option key={b}>{b}</option>
          ))}
        </select>
      </label>
      <label className="check">
        <input type="checkbox" checked={Boolean(value("capture_content"))} onChange={(e) => set("capture_content", e.target.checked)} />
        <span>
          {t("settings.captureContent")} <span className="field-help">{t("settings.captureContentHelp")}</span>
        </span>
      </label>
      <div>
        <h3 className="field-label" style={{ margin: "8px 0" }}>
          {t("settings.environment")}
        </h3>
        <table className="table">
          <tbody>
            {Object.entries(data.tracing_environment).map(([name, set]) => (
              <tr key={name}>
                <td className="mono">{name}</td>
                <td>{set ? <span className="badge badge-good">{t("settings.envSet")}</span> : <span className="badge">{t("settings.envUnset")}</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="field-help">{t("settings.environmentHelp")}</p>
      </div>
    </div>
  );
}
