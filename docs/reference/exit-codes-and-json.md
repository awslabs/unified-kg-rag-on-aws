# CLI exit codes and JSON output

What every CLI (`run-ingestion`, `run-rag`, `run-eval`, `run-visualization`,
`run-describe`, `run-prompt-tuning`, `prepare-benchmark`, and the servers
`run-mcp-server` and `run-console`) returns to a script or a scheduler:
its exit code, what a signal does, and the document `--output-format json`
prints. Every flag: [CLI Reference](./cli.md); what an error message means:
[Troubleshooting](../troubleshooting.md).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | The run failed: a failed stage or query (in `run-ingestion` also a `ConfigurationError` a stage meets, an unreachable store or an unreachable model server), a breached `--max-failure-rate`, a failed `run-eval --fail-if` gate, a failed upload, nothing rendered or a failed renderer |
| `2` | Usage or configuration error, before any work or outside a stage ([exit code 2](#exit-code-2)) |
| `130` | Interrupted with Ctrl-C (SIGINT) |
| `143` | Stopped by SIGTERM (ECS `StopTask`, a Step Functions timeout); `run-ingestion` checkpoints its finished stages first, and `run-eval` uploads the finished runs' results to an `s3://` output (or keeps them locally and logs the path) |

<a id="exit-code-2"></a>

Exit code `2` covers:

- **Usage.** A bad flag.
- **What the run reads.** A `run-rag` or `run-eval` `--suffix` nothing was
  ingested under (the message lists the ingested ones); a filter key no
  index declares; a `run-rag` or `run-eval`
  `--search-strategy` (or a `run-eval` query's own strategy) whose
  methodology the index namespace does not hold
  ([methodology profiles](../concepts/overview.md#methodology-profiles));
  a missing or invalid input (a `run-eval` dataset, a `run-visualization`
  data file, a `prepare-benchmark` data file).
- **Configuration**, all before any work: an invalid config file or
  override, an unset store endpoint, or a missing extra the selected
  providers need (`local`).
- **A `ConfigurationError` found outside a stage**, such as `run-ingestion`'s
  pre-run refusal before its first model call:
  - the search engine refused to create an index the run needs;
  - an incremental run's live indices (a deletion-only run's included) were
    built for other embedding, language or registry settings;
  - an incremental run would add a methodology to a namespace that also
    holds other source scopes' documents;
  - the embedding model returned vectors of another dimension than
    `models.embedding.dimension` for the probe text it embeds first.

Flags must be spelled out in full: a prefix such as `--output` is rejected.

A query's failure maps to its code by its class (`ErrorClass`, the same for
`run-rag`, `run-eval`, the MCP server and the console), also when
`processing.ignore_errors` turned it into the error answer
(`metadata.error_class`):

| Class | Raised as | Exit code |
|---|---|---|
| `invalid_filter` | `InvalidFilterError` | `2` |
| `invalid_argument` | `ToolArgumentError` | `2` |
| `not_indexed` | `MethodologyNotIndexedError` | `2` |
| `index_not_found` | `IndexNotFoundError` | `2` |
| `configuration` | `ConfigurationError` | `2` |
| `timeout` | `QueryTimeoutError` | `1` |
| `store_unreachable` | `StoreConnectionError` | `1` |
| `model_unreachable` | `ModelUnreachableError` | `1` |
| `failure` | any other | `1` |

The MCP server returns the message of the first four and `timeout` to the
caller (a generic one for the rest, whose message may name an endpoint); the
console's Ask page names the class. `run-rag`'s interactive session keeps
going after any of them.

`run-mcp-server` uses the same codes: `0` once the client closes stdio (or
the HTTP server stops), `2` for a usage or configuration error (the `mcp`
extra missing, a non-loopback `--host` without `--allow-remote`), `1`
for a failure, `130` and `143` for a signal. It has no `--output-format`:
over stdio its stdout carries the MCP messages only
([MCP server](../guides/mcp.md)).

`run-console` exits `0` when Ctrl-C stops it (the server finishes running
requests first, up to 60 s), `143` on SIGTERM, `2` for a usage error (the
`console` extra missing, a non-loopback `--host` without `--allow-remote`,
a `--source-directory` that is not a directory) and `1` when it cannot
listen on the address. A config file that does not load is no error: the
console starts and its Settings page shows why. With `--output-format json`
it prints one `{"status": "serving", "url": ..., "web_built": ...,
"version": ...}` document once it listens; the URL carries the access token
([Console](../guides/console.md)).

## Signals

A run stopped with `130` or `143` ends its process at once: model calls in
flight are abandoned rather than waited for, the logs, tracing and stdout
are flushed first, and a process pool's workers are ended with it.

- **Query CLIs.** A SIGTERM or Ctrl-C cancels a query CLI's work at its next
  await, so its cleanup (closing store connections, `run-eval`'s upload)
  runs. At the interactive prompt a SIGTERM ends the session at once and a
  Ctrl-C the conversation. A query CLI stopped by a SIGTERM, or by a Ctrl-C
  while its event loop runs, that is still running 100 seconds later is
  ended then (inside the 120 s StopTimeout of the app container), with the
  status document if none was printed, and a second signal ends it at once.
- **`run-ingestion`** stops at a SIGTERM or Ctrl-C after the checkpoint or
  metadata write in progress instead, bounded by the StopTimeout.
- **A CLI that finished** ends its process with its exit code as soon as its
  exit handlers ran, so a signal while it exits cannot turn `0`, `1` or `2`
  into `-15` or `-2`.

## JSON output

`--output-format json` prints one JSON document to stdout and moves the
banner, progress and logs to stderr, so the output pipes into `jq` (not with
`--metrics-sink cloudwatch`, which also writes to stdout). The document
holds:

- **`run-rag`**: `success` and the answer or the search results; a failed
  query has `"success": false` and its `error` (exit `1`). An answer's
  `search_results` leaves out `results`, which repeat its `sources`, unless
  `--verbose` (they also hold the results the context budget cut). From the
  one view of an answer every surface reads (`AnswerView`): `degraded` (the
  steps it was given without: `step`, `reason`, `error_type`),
  `degraded_error`, `citations_unresolved` (the lineage reads that failed,
  in the same shape), `error_class` (of the error answer under
  `processing.ignore_errors`, else `null`) and an answer's `written` (the
  model-written sections it may cite).
- **`run-ingestion`**: with `--verify-metadata`, `pipeline_id` and
  `"metadata_valid": true` (a missing or corrupt record exits `1` with the
  status document below). Otherwise the pipeline id, the status, the model
  calls and tokens of the stages the run executed (`token_usage`), the
  per-stage results, and these counts:
  - `failed_documents`: the documents the run recorded `FAILED`. A model
    outage that fails every unit records none; one that fails some units
    records their documents, retried by the next run without counting toward
    `indexing.max_document_failures`.
  - `units_failed`: the text units a stage failed on, indexed without that
    stage's output and extracted again by the next run. Counted in a full
    build too, where no registry records a document `FAILED` and
    `failed_documents` stays 0; the run ends with a WARNING naming them by
    stage and cause.
  - `documents_retrying`: the documents failing another consecutive run for
    an error not their own. An outage or a setting never excludes them; each
    file and its run count is also warned of.
  - `documents_without_claims`: the documents indexed without the claims of
    some text units, whose claim extraction failed for an error of their own.
    Not counted as failed; also in the summary.
  - `documents_excluded`: the documents left out because they failed
    `indexing.max_document_failures` consecutive runs unchanged, at the same
    extraction settings (a settings change starts the count over); also in
    the summary.

  And `build_record`, the suffix's build record after the run: `written`,
  `kept` (the run wrote nothing; the stored one stands), `failed` (no index
  took it; the next run writes it again), `missing` (none stored, none
  written), `unsupported` (a vector store without index metadata), or
  `null` when the indexing stage did not run. Until a record is stored,
  queries take the namespace to hold what their config builds; a run that
  exits `0` with `failed` or `missing` also says so in its summary.
- **`run-eval`**: the `output_directory`, `upload_error`, each run's summary
  (`runs`), the `comparison`, the failure-budget breaches
  (`failure_budget_breaches`) and the failed gates (`gate_failures`); a
  breach, a failed gate or an upload error exits `1`.
- **`run-visualization`**: the `output_directory`, the written `files` and
  the `failed_renderers` (each failed renderer's error); an empty list
  (nothing rendered) or a failed renderer exits `1`.
- **`run-describe`**: the namespace's description (`suffix`, `namespace`,
  `build`, `build_record` (`present` or `missing`: then `artifacts` are
  what the config builds), `artifacts`, `methodologies`, `strategies`, `auto_routes`,
  `indices`, `registry`) and `export` (its `path`, `counts`, `truncated`
  and `missing` kinds, or `null` without `--output-path`); a namespace that
  holds no index exits `2`
  ([Inspect and export](../guides/inspect.md)).
- **`run-prompt-tuning`**: the `output_path`, the `profile` and, with
  `--schema-out`, the `schema_path`.
- **`prepare-benchmark`**: the `output_directory`, the `benchmark` label
  and the counts written (`questions`, `paragraphs`); with `--hops`, the
  answerable questions per hop count (`hops`); with `--unanswerable`, the
  `answerable` and `unanswerable` counts and `unanswerable_filter`
  (`requested`, `kept`, `examined`, `excluded_twin`, `excluded_contaminated`,
  `excluded_no_twin`). Too few unanswerable questions passing the filters,
  `--unanswerable` with an answerable-only file, `--hops` quotas with
  `--limit` or above the questions of a hop count exit `2`.

### Failures and interruptions

Every failure has the same keys, in a status document or in a result that
shows the failure:

| Key | Value |
|---|---|
| `status` | `"failed"` (`"interrupted"` for a stopped run; a `run-ingestion` result keeps the run's own status) |
| `exit_code` | The process's exit code |
| `error` | The message, when there is one |
| `error_type` | What failed: the exception's class name (`ConfigurationError`, `QueryTimeoutError`, `StoreConnectionError`, `ModelUnreachableError`, `PipelineExecutionError`, ...), `UsageError` for a refused command line, `FailureBudgetExceeded` or `RegressionGateFailed` for `run-eval`, `RunFailed` for a run that failed without an exception (nothing rendered), `Interrupted` for a stopped run |

A run that printed its result prints nothing more; the result shows the
failure with these keys: a failed `run-rag` query (also with
`"success": false`), a `run-ingestion` run a stage failed, a `run-eval`
breach, gate or upload error, a `run-visualization` run that rendered
nothing or lost a renderer. A run that did not prints a status document
instead:

- A run stopped by a signal before it printed prints
  `{"status": "interrupted", "exit_code": 143, "error_type": "Interrupted"}`
  (`130` for Ctrl-C).
- A run that fails before it printed prints
  `{"status": "failed", "exit_code": 1, "error_type": ...}` (or `2`) with the
  `error` it failed with, an argparse error included.
- That holds from the first moment: a SIGTERM while the CLI still loads, or
  after it printed its result while the process exits, exits `143` too, and
  a Ctrl-C while it loads `130` (the status document only while loading; a
  SIGINT the parent ignores, as `nohup` does, stays ignored).
- A `run-ingestion` result whose run a stage failed carries the same `error`,
  `exit_code` and `error_type`. `run-ingestion` adds the `pipeline_id` to resume with
  (once the run resolved it) and, when a stopped run's next run on the
  namespace must wait for writes possibly still in flight (this run's, or
  those of the run whose expired lease it took over),
  `next_run_writes_from` (UTC, ISO 8601).

One run prints no JSON: a `run-ingestion` resume whose cached stages fail
the integrity check asks "Continue despite integrity issues?" on the
terminal, and an answer of no exits `0` with nothing on stdout. Pass `--yes`
to continue; without a terminal the question stops the run with exit `1` and
the status document.
