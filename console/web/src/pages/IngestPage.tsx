// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useState } from "react";
import { JobPanel } from "../components/JobPanel";
import { Empty, ErrorText, formatBytes, JobStatusBadge, Loading, Notice, PageHead, Sheet } from "../components/ui";
import { useI18n } from "../i18n/i18n";
import { errorMessage, get, send, upload } from "../lib/api";
import type { Corpus, JobRecord, Status } from "../lib/types";
import { useApi } from "../lib/useApi";

const METHODOLOGIES = ["graphrag", "lightrag", "hipporag", "raptor"];

/** What an index namespace holds, as its build record names it. */
interface Held {
  namespace: string;
  recorded: boolean;
  held: string[];
  // Those an ingestion of the picked methodologies would no longer serve.
  dropped: string[];
}

interface FileWithPath extends File {
  webkitRelativePath: string;
}

export function IngestPage({ status, onJobsChanged }: { status: Status | null; onJobsChanged: () => void }) {
  const { t, number, date } = useI18n();
  const corpora = useApi(() => get<{ corpora: Corpus[]; extensions: string[] }>("corpora"), []);
  const jobs = useApi(() => get<{ jobs: JobRecord[] }>("jobs"), []);
  const [corpus, setCorpus] = useState<string | null>(null);
  const [methodologies, setMethodologies] = useState<string[] | null>(null);
  const [indexValue, setIndexValue] = useState("");
  const [forceRebuild, setForceRebuild] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const [confirmDeletions, setConfirmDeletions] = useState(false);
  const [confirmIntegrity, setConfirmIntegrity] = useState(false);
  const [confirmNarrowing, setConfirmNarrowing] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [files, setFiles] = useState<FileWithPath[]>([]);
  const [uploading, setUploading] = useState(false);
  const [over, setOver] = useState(false);

  const suffix = indexValue.trim();
  // The target namespace's methodologies: the picker starts from them, so
  // ingesting another corpus keeps what the namespace serves. Unknown (a
  // store error, a bad name) reads as no record.
  const [holds, setHolds] = useState<string[] | null>(null);
  const chosen = methodologies ?? holds ?? status?.methodologies ?? ["graphrag", "lightrag"];
  const held = useApi(
    () =>
      get<Held>("namespace/methodologies", { suffix: suffix || undefined, methodologies: chosen.join(",") || undefined })
        .then((found) => {
          setHolds(found.recorded && found.held.length ? found.held : null);
          return found;
        })
        .catch(() => {
          setHolds(null);
          return null;
        }),
    [suffix, chosen.join(","), status?.config_path],
  );
  // What this run would remove from the namespace (the server decides, by
  // the artifacts the picked methodologies build).
  const dropped = held.data?.recorded ? held.data.dropped : [];
  const ingestJobs = (jobs.data?.jobs ?? []).filter((j) => j.kind === "ingest");
  const active = ingestJobs.find((j) => j.status === "running" || j.status === "stopping");
  const shownJob = jobId ?? active?.id ?? null;

  async function start(answer?: { job: JobRecord; question: string }) {
    if (!corpus && !answer) return;
    setError(null);
    // Answering a run's question starts it again as it was, with that
    // question's flag added.
    const params = answer?.job.params as Record<string, unknown> | undefined;
    const body = params
      ? { ...params, [`confirm_${answer?.question}`]: true }
      : {
          corpus,
          methodologies: chosen,
          index_value: indexValue.trim() || null,
          force_rebuild: forceRebuild,
          confirm_reset: confirmReset,
          confirm_deletions: confirmDeletions,
          confirm_integrity: confirmIntegrity,
          confirm_narrowing: dropped.length > 0 && confirmNarrowing,
        };
    try {
      const job = await send<JobRecord>("jobs/ingest", body);
      setJobId(job.id);
      onJobsChanged();
      void jobs.reload();
    } catch (e) {
      setError(errorMessage(t, e));
    }
  }

  async function sendUpload() {
    if (!files.length || !uploadName.trim()) return;
    setUploading(true);
    setError(null);
    try {
      // In batches, so no request is over the server's file or size limit.
      let result = { corpus: "" };
      for (const batch of batches(files)) {
        const form = new FormData();
        form.append("name", uploadName.trim());
        for (const file of batch) {
          const path = file.webkitRelativePath ? file.webkitRelativePath.split("/").slice(1).join("/") || file.name : file.name;
          form.append("files", file, path);
        }
        result = await upload<{ corpus: string }>("uploads", form);
      }
      setFiles([]);
      await corpora.reload();
      setCorpus(result.corpus);
    } catch (e) {
      setError(errorMessage(t, e));
    } finally {
      setUploading(false);
    }
  }

  const pickFiles = (list: FileList | null) => {
    const picked = Array.from(list ?? []) as FileWithPath[];
    setFiles(picked);
    const folder = picked[0]?.webkitRelativePath?.split("/")[0];
    if (folder && !uploadName) setUploadName(folder);
  };

  return (
    <>
      <PageHead title={t("ingest.title")} lede={t("ingest.lede")} />
      <div className="grid-2">
        <div className="stack">
          <Sheet title={t("ingest.corpus")} aside={<button className="button button-small" onClick={() => void corpora.reload()}>{t("common.refresh")}</button>}>
            <ErrorText error={corpora.error} />
            {corpora.data && !corpora.data.corpora.length ? (
              <Empty title={t("ingest.noCorpora")}>{t("ingest.noCorporaHelp")}</Empty>
            ) : null}
            <div className="stack" role="radiogroup" aria-label={t("ingest.corpus")} style={{ gap: 8 }}>
              {(corpora.data?.corpora ?? []).map((item) => (
                <label key={item.ref} className="source" style={{ cursor: "pointer", gridTemplateColumns: "auto 1fr", alignItems: "start", columnGap: 10, borderColor: corpus === item.ref ? "var(--trace)" : undefined }}>
                  <input type="radio" name="corpus" checked={corpus === item.ref} onChange={() => setCorpus(item.ref)} style={{ marginTop: 5, accentColor: "var(--trace)" }} />
                  <span className="stack" style={{ gap: 2 }}>
                    <span className="source-title mono">{item.name}</span>
                    <span className="field-help">
                      {t("ingest.corpusFiles", { count: item.files })} · {formatBytes(item.bytes, number)} ·{" "}
                      {Object.entries(item.extensions)
                        .map(([ext, n]) => `${ext} ${number(n)}`)
                        .join(", ")}
                      {item.unsupported ? ` · ${t("ingest.unsupported", { count: item.unsupported })}` : ""}
                    </span>
                    <span className="field-help">{t("ingest.modified", { when: date(item.modified) })}</span>
                  </span>
                </label>
              ))}
            </div>
            <div
              className={`dropzone${over ? " over" : ""}`}
              style={{ marginTop: 16 }}
              onDragOver={(e) => {
                e.preventDefault();
                setOver(true);
              }}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setOver(false);
                pickFiles(e.dataTransfer.files);
              }}
            >
              <span className="field-label">{t("ingest.upload")}</span>
              <span className="field-help">{t("ingest.uploadHelp", { extensions: (corpora.data?.extensions ?? []).join(" ") })}</span>
              <div className="row">
                <label className="button button-small">
                  {t("ingest.chooseFolder")}
                  <input
                    type="file"
                    className="visually-hidden"
                    multiple
                    // @ts-expect-error -- the non-standard folder picker
                    webkitdirectory=""
                    onChange={(e) => pickFiles(e.target.files)}
                  />
                </label>
                <label className="button button-small">
                  {t("ingest.chooseFiles")}
                  <input type="file" className="visually-hidden" multiple onChange={(e) => pickFiles(e.target.files)} />
                </label>
                {files.length ? <span className="muted">{t("ingest.picked", { count: files.length })}</span> : null}
              </div>
              <div className="row" style={{ width: "100%" }}>
                <label className="field" style={{ flex: "1 1 220px" }}>
                  <span className="field-label">{t("ingest.uploadName")}</span>
                  <input className="input mono" value={uploadName} onChange={(e) => setUploadName(e.target.value)} placeholder="my-corpus" />
                </label>
                <button className="button" style={{ alignSelf: "end" }} disabled={!files.length || !uploadName.trim() || uploading} onClick={() => void sendUpload()}>
                  {uploading ? t("ingest.uploading") : t("ingest.uploadButton")}
                </button>
              </div>
            </div>
          </Sheet>
          <Sheet title={t("ingest.methodologies")}>
            <div className="stack" style={{ gap: 10 }}>
              <span className="field-help">{t("ingest.methodologiesHelp")}</span>
              {holds && held.data ? (
                <span className="field-help">
                  {t("ingest.namespaceHolds", { namespace: held.data.namespace, methodologies: holds.map((m) => t(`methodology.${m}`)).join(", ") })}
                </span>
              ) : null}
              <div className="stack" role="group" aria-label={t("ingest.methodologies")} style={{ gap: 6 }}>
                {METHODOLOGIES.map((name) => (
                  <label key={name} className="check">
                    <input
                      type="checkbox"
                      checked={chosen.includes(name)}
                      onChange={(e) => setMethodologies(e.target.checked ? [...chosen, name] : chosen.filter((m) => m !== name))}
                    />
                    <span>
                      <strong>{t(`methodology.${name}`)}</strong> <span className="field-help">{t(`methodology.help.${name}`)}</span>
                    </span>
                  </label>
                ))}
              </div>
              <details>
                <summary className="field-help">{t("ingest.advanced")}</summary>
                <div className="stack" style={{ marginTop: 10, gap: 10 }}>
                  <label className="field">
                    <span className="field-label">{t("ingest.indexValue")}</span>
                    <input className="input mono" value={indexValue} placeholder={status?.namespace ?? "default"} onChange={(e) => setIndexValue(e.target.value)} />
                    <span className="field-help">{t("ingest.indexValueHelp")}</span>
                  </label>
                  <label className="check">
                    <input type="checkbox" checked={forceRebuild} onChange={(e) => setForceRebuild(e.target.checked)} />
                    <span>
                      {t("ingest.forceRebuild")} <span className="field-help">{t("ingest.forceRebuildHelp")}</span>
                    </span>
                  </label>
                  <label className="check">
                    <input type="checkbox" checked={confirmReset} onChange={(e) => setConfirmReset(e.target.checked)} />
                    <span>
                      {t("ingest.confirmReset")} <span className="field-help">{t("ingest.confirmResetHelp")}</span>
                    </span>
                  </label>
                  <label className="check">
                    <input type="checkbox" checked={confirmDeletions} onChange={(e) => setConfirmDeletions(e.target.checked)} />
                    <span>
                      {t("ingest.confirmDeletions")} <span className="field-help">{t("ingest.confirmDeletionsHelp")}</span>
                    </span>
                  </label>
                  <label className="check">
                    <input type="checkbox" checked={confirmIntegrity} onChange={(e) => setConfirmIntegrity(e.target.checked)} />
                    <span>
                      {t("ingest.confirmIntegrity")} <span className="field-help">{t("ingest.confirmIntegrityHelp")}</span>
                    </span>
                  </label>
                </div>
              </details>
              {!chosen.length ? <Notice tone="warn">{t("ingest.noMethodology")}</Notice> : null}
              {dropped.length && chosen.length && held.data ? (
                <Notice tone="warn" title={t("ingest.narrowingTitle", { namespace: held.data.namespace })}>
                  <p style={{ margin: "0 0 8px" }}>
                    {t("ingest.narrowingHelp", { namespace: held.data.namespace, methodologies: dropped.map((m) => t(`methodology.${m}`)).join(", ") })}
                  </p>
                  <label className="check">
                    <input type="checkbox" checked={confirmNarrowing} onChange={(e) => setConfirmNarrowing(e.target.checked)} />
                    <span>{t("ingest.narrowingConfirm", { methodologies: dropped.map((m) => t(`methodology.${m}`)).join(", ") })}</span>
                  </label>
                </Notice>
              ) : null}
              {error ? <Notice tone="bad">{error}</Notice> : null}
              <div>
                <button className="button button-primary" disabled={!corpus || !chosen.length || !!active || (dropped.length > 0 && !confirmNarrowing)} onClick={() => void start()}>
                  {t("ingest.start")}
                </button>
                {!corpus ? <span className="field-help" style={{ marginLeft: 12 }}>{t("ingest.pickFirst")}</span> : null}
                {active ? <span className="field-help" style={{ marginLeft: 12 }}>{t("ingest.oneAtATime")}</span> : null}
              </div>
            </div>
          </Sheet>
        </div>
        <div className="stack">
          <Sheet title={t("ingest.run")}>
            {shownJob ? (
              <JobPanel
                key={shownJob}
                jobId={shownJob}
                onEnded={() => {
                  onJobsChanged();
                  void jobs.reload();
                  void held.reload();
                }}
              >
                {(job) => (
                  <>
                    {job.confirmation && job.status === "failed" ? (
                      <Notice tone="warn" title={t(`ingest.question.${job.confirmation}`)}>
                        <p style={{ margin: "0 0 8px" }}>{t(`ingest.question.${job.confirmation}Help`)}</p>
                        <button className="button button-primary button-small" disabled={!!active} onClick={() => void start({ job, question: job.confirmation as string })}>
                          {t(`ingest.answer.${job.confirmation}`)}
                        </button>
                      </Notice>
                    ) : null}
                    <IngestResult job={job} />
                  </>
                )}
              </JobPanel>
            ) : (
              <Empty title={t("ingest.noRunTitle")}>{t("ingest.noRun")}</Empty>
            )}
          </Sheet>
          <Sheet title={t("ingest.history")}>
            {!jobs.data ? (
              <Loading />
            ) : ingestJobs.length ? (
              <div className="table-wrap" tabIndex={0}>
                <table className="table">
                  <thead>
                    <tr>
                      <th>{t("job.started")}</th>
                      <th>{t("ingest.corpus")}</th>
                      <th>{t("job.statusLabel")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ingestJobs.slice(0, 12).map((job) => (
                      <tr key={job.id} aria-selected={job.id === shownJob}>
                        <td>
                          <button className="link-button" onClick={() => setJobId(job.id)}>
                            {date(job.created)}
                          </button>
                        </td>
                        <td className="mono">{String(job.params.corpus ?? "")}</td>
                        <td>
                          <JobStatusBadge status={job.status} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted">{t("ingest.noHistory")}</p>
            )}
          </Sheet>
        </div>
      </div>
    </>
  );
}

function IngestResult({ job }: { job: JobRecord }) {
  const { t, number, seconds } = useI18n();
  const result = job.result as Record<string, unknown> | null;
  if (!result || job.status === "running") return null;
  const usage = (result.token_usage ?? {}) as Record<string, number>;
  return (
    <dl className="kv">
      <dt>{t("ingest.result.pipeline")}</dt>
      <dd className="mono">{String(result.pipeline_id ?? job.pipeline_id)}</dd>
      <dt>{t("ingest.result.duration")}</dt>
      <dd>{seconds(result.duration_seconds as number)}</dd>
      <dt>{t("ingest.result.failedDocuments")}</dt>
      <dd>{number(result.failed_documents as number)}</dd>
      <dt>{t("ingest.result.excluded")}</dt>
      <dd>{number(result.documents_excluded as number)}</dd>
      <dt>{t("ingest.result.calls")}</dt>
      <dd>{number(usage.llm_calls)}</dd>
      <dt>{t("ingest.result.tokens")}</dt>
      <dd>
        {number(usage.input_tokens)} / {number(usage.output_tokens)}
      </dd>
    </dl>
  );
}

// What one upload request carries at most (under the server's limits).
const BATCH_FILES = 500;
const BATCH_BYTES = 100 * 1024 * 1024;

/** ``files`` in batches of at most BATCH_FILES files and BATCH_BYTES. */
export function batches<T extends { size: number }>(files: T[]): T[][] {
  const out: T[][] = [];
  let current: T[] = [];
  let bytes = 0;
  for (const file of files) {
    if (current.length && (current.length >= BATCH_FILES || bytes + file.size > BATCH_BYTES)) {
      out.push(current);
      current = [];
      bytes = 0;
    }
    current.push(file);
    bytes += file.size;
  }
  if (current.length) out.push(current);
  return out;
}
