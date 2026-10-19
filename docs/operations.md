# Operator Runbook

> 🇰🇷 한국어판: [docs/operations.ko.md](./operations.ko.md)

Tasks an operator runs against an existing deployment, and what to do when a
run or a query fails. Every configuration key is in the
[Configuration Reference](./configuration.md) (how the configuration loads:
[User Guide §2](./user-guide.md#2-configuration)); the deployed stack and its
commands are in [`iac/README.md`](../iac/README.md#after-deploy).

| I want to | Go to |
|---|---|
| Check a deployment before the first run | [Before a run](#before-a-run) |
| Resume, recompute or rebuild an ingestion | [Re-ingest](#re-ingest) |
| Keep a corpus up to date | [Incremental runs](#incremental-runs) |
| Recover a run that stopped | [Interrupted runs](#interrupted-runs) |
| Find out why a document is `FAILED` or `QUARANTINED` | [Failed documents](#failed-documents), [Quarantined documents](#quarantined-documents) |
| Run two ingestions, or understand `NamespaceLeaseError` | [Concurrent runs](#concurrent-runs) |
| Act on a CloudWatch alarm | [Alarms](#alarms) |
| Restore the stores | [Backup and restore](#backup-and-restore) |
| Give a serving layer query access, or isolate tenants | [Query access and tenant isolation](#query-access-and-tenant-isolation) |
| Turn on tracing (Jaeger, Langfuse, Phoenix, CloudWatch) | [Observability](#observability) |
| Look up an error message | [Common errors](#common-errors) |

## Before a run

1. **Region and credentials.** `aws.region_name` in the config file wins;
   `AWS_REGION` fills it only when the file leaves it unset, and
   `UKR__AWS__REGION_NAME` overrides it (Bedrock follows it while
   `aws.bedrock.region_name` is unset). Check with
   `env | grep -E '^(AWS_REGION|UKR__AWS__)'` and `aws sts get-caller-identity`.
2. **Models.** The [required models](./user-guide.md#required-models) are
   enabled in the Bedrock region.
3. **Endpoints** are set ([Start-up endpoint check](#start-up-endpoint-check)).
4. **No other ingestion runs** against the same stores
   ([Concurrent runs](#concurrent-runs)). On the CDK stack:
   `aws stepfunctions list-executions --state-machine-arn <arn> --status-filter RUNNING`.

## Routine operations

### Re-ingest

| Goal | How | Stage cache | Stores and registry |
|---|---|---|---|
| Finish an interrupted run | Re-run with the same `--pipeline-id` (CDK: a new execution with the same `pipeline_id`); it resumes at the first failed or incomplete stage, or at `--resume-from-stage <stage>` | Reused | Written by the stages that run |
| Recompute every stage | `--force-rebuild` | Ignored and rewritten | Written as usual (incremental when enabled) |
| Rebuild the stores | `indexing.reset: true` for one run | Unchanged | Rebuilt without going offline; the namespace's other records are deleted at the end ([Design §5.2](./design.md#52-full-rebuild-indexingreset)) |

Use `indexing.reset: true` (plus `--force-rebuild` if stage outputs must
change too) after changing what is baked into the stored data: the embedding
model or `models.embedding.dimension`, `processing.translation.source_language`
or the language analyzers, or a built-in prompt template (code). Reset also
refreshes the communities incremental runs append to and gives entities their
current names' ids. Other extraction settings need no reset: the next
incremental run re-extracts the affected documents and logs how many
([User Guide §5](./user-guide.md#workflows)).

During a reset, vector queries read the previous indices until each new one
swaps in; graph queries see previous and rebuilt items mixed until the final
drop. Quarantined documents are skipped and keep their records. A failed reset
drops nothing: re-run it. One whose graph writes partly failed (even within
`indexing.max_failure_rate`) fails too (`neptune_sweep` in the indexing
results); the next run removes what the previous graph still holds.
**Set `indexing.reset` back to `false` afterwards**, or every run rebuilds.

### Incremental runs

With `providers.doc_status.enabled: true` (the container image sets it), each
run indexes only new and changed documents and removes what only a missing
file stated. The diff is scoped to the run's
[index suffix](./user-guide.md#index-suffix) and source scope
(`processing.document_parsing.source_scope`: the resolved source directory, or
the `s3://` URI in the container; with and without a trailing slash is one
scope, and a WARNING names the spelling used). So:

- **Keep a corpus at one source** (or a fixed `source_scope`). A moved corpus
  with the default scope leaves the old scope's content indexed: set
  `source_scope` before moving, or run once with
  `--retire-source-scope <old source directory>`
  ([Deletion scope](./user-guide.md#deletion-scope)). A run with new documents
  logs a WARNING naming the suffix's other local source directories.
- **Never let corpus files expire or disappear**: the next run removes their
  content.
- **One index suffix and config file per tenant or corpus version.**

**Rebuild periodically.** Delta runs append communities instead of
re-clustering, and stored communities and reports keep an entity's old id
after it took its current name's id; `indexing.reset: true` refreshes both.
Watch the indexing stage's `community_drift` metrics: `delta_communities`,
`clustered_entities`, `overlapping_entities`, `stale_communities`,
`added_roots` (each adds a report to every global-search cut) and
`overlap_share`. At `indexing.community_drift_threshold` (default `0.5`) the
run logs a WARNING and reports `community_rebuild_recommended: 1`, which an
alarm can watch (the counts are an upper bound). A run that removes text units
also reports `outdated_reports`, stored reports that still cite a removed
unit and may state what was removed: any such report recommends a rebuild
too. `unclustered_entities` counts stored entities without a community that
a small delta component the run clustered links to: a full build clusters
them, a delta run clusters only its delta.

**Registry unreadable** (missing table, denied access, lasting throttling):
`document_loading` fails with `DocStatusRegistryError` naming the table and
cause, rather than reindexing everything. Fix the registry and re-run with the
same `--pipeline-id`, or set `providers.doc_status.enabled: false`.

Coming from a pre-release build: see
[CHANGELOG: Upgrading from pre-release builds](../CHANGELOG.md#upgrading-from-pre-release-builds).

### Caches and prefixes

| Cache | Location | Notes |
|---|---|---|
| Stage checkpoints (local) | `<--cache-directory>/<pipeline_id>/` (default `cache/`) | Reused on resume |
| Stage checkpoints (S3) | `s3://<bucket>/<--s3-prefix>/<pipeline_id>/` (default `pipeline-runs`) | With `--s3-sync`: uploaded after every stage and on SIGTERM, so a fresh task resumes from it. A failed download or final upload fails the run with `CacheSyncError`; a failed per-stage or on-stop upload only warns (the next upload retries it), and a stop after a lost lease skips the upload. Expired by the bucket lifecycle, not `cache.ttl_seconds` |
| LLM response cache | `<stage checkpoint dir>/llm-cache/` | Reused by a re-run with the same pipeline id; `cache.llm_responses.enabled: false` turns it off, `--force-rebuild` ignores it |
| Extraction store | `graphrag-extraction-records-<suffix>` (`indexing.opensearch.extraction_records_index_prefix`) | Per-chunk extraction results reused by later runs while a stored text unit names them, and the note (`pending-strip`) a removal keeps until it finished, which a resumed removal needs. `clear` and `--force-rebuild` keep it; to drop it, delete the index only after a run that succeeded, or run `indexing.reset: true` next |
| Embedding cache | `s3://<aws.s3.bucket_name>/embedding-cache/cache.json` | Only with `indexing.opensearch.persist_embedding_cache: true` |
| Visualization data | `<cache directory>/<pipeline_id>/visualization/` unless `graph.visualization.outputs_directory` is set | Synced with the stage cache (`.json` only) |

The CDK bucket expires `pipeline-runs/` and `embedding-cache/` after 30
days: keep the corpus under another prefix (`corpus/`), since an expired file
looks deleted. Check a reused bucket's own lifecycle rules.

### OpenSearch replicas

On the CDK stack a multi-node domain (`opensearch_count` > 1) gets one replica
per index and a single-node domain none, through
`UKR__INDEXING__OPENSEARCH__INDEX_SETTINGS__NUMBER_OF_REPLICAS`, applied when
an index is created. Set older indices once from inside the VPC:

```bash
curl -X PUT "https://<OpenSearch endpoint>/_all/_settings" \
  --aws-sigv4 "aws:amz:<region>:es" --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
  -H "x-amz-security-token: $AWS_SESSION_TOKEN" \
  -H 'Content-Type: application/json' -d '{"index": {"number_of_replicas": 1}}'
```

### Backup and restore

The stores refer to each other (the registry lists each document's text
units, which the graph and index artifacts cite), so restore them to one
common point or not at all. The corpus in S3 is the source of truth.

| Store | Backup | Window |
|---|---|---|
| Neptune | Automated backups, point-in-time restore | `backup_retention_days` (default 7 days) |
| OpenSearch | Automated hourly snapshots (`cs-automated-enc`) | 14 days |
| DynamoDB doc-status table | Point-in-time recovery | 35 days |
| S3 cache bucket | None (checkpoints are re-creatable) | — |

**Within Neptune's window:**

1. Stop ingestion (no `RUNNING` execution).
2. Pick an OpenSearch snapshot (`GET _snapshot/cs-automated-enc/_all`) taken
   after the last run finished; its start time is the restore point.
3. Restore Neptune (`aws neptune restore-db-cluster-to-point-in-time`) and the
   table (`aws dynamodb restore-table-to-point-in-time`) to it. Both create
   new resources with default settings: add Neptune instances, re-enable PITR
   and deletion protection.
4. Delete the domain's current indices, then restore the snapshot's indices
   and aliases (`POST _snapshot/cs-automated-enc/<snapshot>/_restore`).
5. Point the data plane at the restored cluster and table
   (`UKR__AWS__NEPTUNE__ENDPOINT`, `UKR__AWS__DYNAMODB__TABLE_NAME`), and grant
   the task role, the DynamoDB gateway-endpoint policy and the alarms on them
   (they are scoped to the stack's own resources).

**Otherwise**, run one full rebuild with `indexing.reset: true`
([Re-ingest](#re-ingest)); restore the table first only if other namespaces
share it.

## When a run fails

### Interrupted runs

An incremental run that stops inside indexing (killed task, unreachable store
or registry, tripped failure gate) needs no repair: re-run. Indexing first
records every new and changed document `PENDING` with its old and new text
units, and every deleted one before its removal, and marks it `PROCESSED` or
`FAILED` (or drops a deleted one's record) only after the writes, so the next
run re-extracts each `PENDING` document still in the corpus, a deleted one
restored meanwhile included, and removes what was written for one that is gone
([Design §5.3](./design.md#53-crash-recovery-write-ahead-protocol)).

- **Leave `PENDING` records alone** (`status` `pending`, `content_hash`
  `pending`): their text units are how the next run finds the interrupted
  writes.
- An interruption is not a failure, but resets `failure_count`; deaths are
  counted in `attempt_count` ([Quarantined documents](#quarantined-documents)).
- A custom registry policy needs `dynamodb:BatchWriteItem` and
  `BatchGetItem` (the CDK grant covers both).

### Failed documents

A document is recorded `FAILED` in five cases:

- translation, graph extraction, gleaning or claim extraction failed on one
  of its text units;
- a base-level community covering one of its text units got no report (a
  failed parent-level report only logs a WARNING and counts in
  `reports_failed`; above `indexing.max_failure_rate` of communities without
  a report, community detection fails);
- an artifact stage (translation through community detection) crashed in a
  `--continue-on-error` run, which records every document of the run `FAILED`;
- writing one of its artifacts to the stores failed;
- the extraction record of one of its text units could not be written to the
  extraction store (`records_not_stored` in graph resolution and claim
  extraction).

The next run treats a `FAILED` document as changed: it removes what the failed
run wrote and processes it again. Any other stage failure records nothing and
stops the run before indexing, leaving the stores and registry as they were;
the next run processes the same delta again.

- **Retry limit.** After `indexing.max_document_failures` (default `3`)
  consecutive failures of unchanged content the document stays `FAILED`,
  keeps what it indexed, and each run logs a WARNING naming the file. Fix the
  cause (often an oversized or malformed file), edit the file, or raise the
  limit.
- **Too large for the registry.** A document whose text-unit ids exceed one
  DynamoDB item (400 KB, about 10,000 ids, half that for an edited document)
  fails indexing before anything is written, with an error naming the file:
  split it.
- **Write failures** are retried first: OpenSearch bulk items rejected with
  429, 502, 503 or 504 up to four times with backoff, every Neptune request up
  to `indexing.neptune.max_attempts` (default `4`) over a new connection. If
  more than `indexing.max_failure_rate` (default `0.2`) of one artifact type's
  writes still fail, indexing fails and records no document; below it, only
  the documents owning a failed artifact are `FAILED` (all of the run's when
  the backend names no item). The `IndexingFailures` and
  `ExtractionFailures` [alarms](#alarms) report both.

### Quarantined documents

A document that kills the task each time (out of memory, a crashing parser)
would stop every later run, so runs count deaths and isolate the culprit.

- **Only deaths count.** A run marks an attempt open on every document it
  processes and clears it however it ends in the process (finished, failed,
  SIGTERM, Ctrl-C). A run that lost its write lease retakes it to clear them
  when no other run holds it (none does, or its lease expired); while
  another run holds it they stay open and count as one death. An attempt
  left open by a dead task is counted (`attempt_count`, batch in
  `crash_batch`). `indexing.reset` counts nothing.
- **Quarantine.** After `indexing.max_document_attempts` (default `3`)
  deaths in a row, the last with the document alone, the next run logs a
  WARNING naming the file, removes what the attempts wrote, records it
  `QUARANTINED` (reason in `error_info`) and indexes the rest
  (`total_documents_quarantined`; CDK alarm `DocumentsQuarantined`).
- **Bisection.** When runs died with several documents, the next run
  processes half of the smallest dying batch and defers the other new and
  changed documents (a WARNING names the half; `total_documents_deferred`
  counts the rest, which keep their indexed version). One culprit among `n`
  is quarantined after about `log2(n) + 1` runs. **Start runs with a new
  pipeline id until `total_documents_deferred` is 0.**
- **Retrying.** A quarantined document is retried when its file changes, or
  when `indexing.max_document_attempts` is raised above its `attempt_count`
  (new extraction settings do not retry it); `indexing.reset` skips it too and
  keeps its record. A reset counts no attempts, so a document never indexed
  before that kills the task kills every reset: run incremental runs until
  it is quarantined. A changed document whose runs died before
  indexing keeps its indexed version; one that died while indexing is
  removed.
- **Limits.** A task that dies whatever it processes (a broken image)
  quarantines documents one at a time: fix the deployment, then edit the files
  or raise the limit. A one-off kill during a lone-suspect run quarantines it
  too. A run that closes its attempts within the SIGTERM stop timeout (120 s
  on ECS) is not counted, but one store request can take 180 s, so a run
  killed mid-request after the timeout can count as a death. A
  document producing no text units stays `PENDING` and is quarantined the
  same way.
- **Resumed runs** with the same pipeline id reuse their delta unless the
  registry no longer matches it or a culprit can be isolated; they then
  resume from `document_loading` (WARNING
  `... resuming from 'document_loading' ...`).

### Concurrent runs

With the registry, one run writes an index namespace at a time: it takes the
namespace's write lease before its first stage and renews it every sixth of
`indexing.lease_seconds` (default 600 s). A second run stops before it reads
the stores or writes anything:

```text
NamespaceLeaseError: Index namespace '<namespace>' is being written by
<host>:<pid>:<id> (its lease runs until <time> unless renewed) ...
```

Wait for that run and start again. A dead run's lease expires within
`lease_seconds` of its last renewal; nothing needs cleaning up. A run whose
renewals keep failing, or whose lease was taken, stops before its next write
with `NamespaceLeaseError: This run lost the write lease ...`: re-run once the
other run ended.

- **Fencing.** Every stage start, checkpoint upload, registry write, indexing
  phase and store write attempt (retries included) checks the lease first and
  stops once less than a third of the term is left since the last successful
  renewal. Keep `lease_seconds` at least three times the longest write
  request (180 s; less logs a WARNING)
  ([Design §5.6](./design.md#56-concurrency)).
- **Takeover after a crash or a stop.** A write already past the fence can
  land after the lease expired (its process was frozen, or the request was
  slow), but no later than that expiry plus the longest request (registry
  writes are fenced per attempt too). A run that takes over a lease that
  expired rather than was released (its holder died or froze, or was
  stopped: a stopped run leaves its lease expired, as requests may still
  be in flight) therefore holds its first write (not the stages before it,
  such as parsing) until then, with a WARNING
  (`Took over the expired write lease ... the first write waits until ...`):
  up to about 3 minutes (180 s plus a 5 s margin) after the old lease
  expired. A SIGTERM ends the wait. A process frozen for longer than that,
  with a request already built, can still write after the new holder; only
  a write fenced by the store itself would rule that out. With DynamoDB, an
  expired lease item its TTL already deleted reads as released (TTL
  deletion usually takes much longer than 3 minutes).
- **Errors in library use.** A failed indexing phase waits for the store
  requests its threads already sent before it raises, so a caller that runs
  again in the same process does not race them; a stop (SIGTERM, Ctrl-C)
  does not wait, as the process ends.
- **Checkpoints.** A run downloads its S3 checkpoint only once it holds the
  lease; a stopped run uploads its checkpoint before releasing it, one that
  lost the lease uploads none.
- **Where the lease lives.** DynamoDB item `namespace` = `<namespace>`,
  `doc_id` = `#lease` (`lease_owner`, `lease_expires_at`, the table's TTL
  attribute); SQLite: the namespace's row of `leases`. Delete it by hand only
  after making sure its run is gone.
- **Permissions.** A custom policy needs `dynamodb:PutItem`, `GetItem` and
  `DeleteItem` (and `UpdateTimeToLive` when the library creates the table).
- **Namespace** = index suffix (`index_value`, `default` when unset) plus
  `indexing.additional_suffix`, plus a tag of non-default
  `indexing.opensearch.*_index_prefix` / `indexing.neptune.*_label_prefix`
  values. Different namespaces can run side by side.

**Without the registry nothing locks** and concurrent runs on one index
suffix are not supported: a full reindex swaps the alias and deletes the older
indices, so one run can delete the index the other is writing. The direct
`IncrementalIndexer` API takes no lease either.

**On the CDK stack** nothing schedules runs and the state machine does not
stop a second execution (the lease fails it). Check for a `RUNNING` execution
first ([Before a run](#before-a-run)), and let a schedule you add fire only
when a previous execution cannot still run (up to `sfn_timeout_hours`,
default 6).

### Start-up endpoint check

Before the first model call, the CLIs check that the store endpoints the run
needs are set and otherwise exit 2:

```text
run-ingestion: error: Missing endpoint configuration for the indexing stage: aws.neptune.endpoint (env UKR__AWS__NEPTUNE__ENDPOINT), aws.opensearch.endpoint (env UKR__AWS__OPENSEARCH__ENDPOINT). Set them in the config file or the environment.
```

- `run-ingestion` checks both when the `indexing` stage is enabled (not for
  `--verify-metadata`).
- `run-rag` and `run-eval` check OpenSearch for every strategy and Neptune
  too for `local`, `drift`, `mix` and `hybrid`; `auto` checks every strategy
  it may route to (`search.auto_routable_strategies`, without `global` and
  `drift` when `graph.community_detection.enabled` is false).

A set but wrong or unreachable endpoint fails at the first connection with
an error, not empty results.

## Alarms

On the CDK stack, a failed ingestion task publishes the execution id and the
error to the SNS topic, and the alarms below report to the same topic: five
on pipeline runs, seven on store health (eight on Neptune Serverless) and two
on capacity. They have a subscriber only with `-c alarm_email=<address>`.
A failed task also fires `PipelineFailures`, a second notice without context
that also covers what the message cannot report (an error the task's handler
cannot catch, such as an input without `pipeline_id`, or a failed publish).

**Pipeline runs**

| Alarm | Fires when |
|---|---|
| `PipelineFailures` | A Step Functions execution failed |
| `PipelineTimedOutOrAborted` | An execution hit the state machine timeout or was stopped |
| `IndexingFailures` | Indexing reported failed items, even in a succeeded execution |
| `ExtractionFailures` | Extraction, gleaning or claim extraction failed on some units |
| `DocumentsQuarantined` | A run quarantined documents ([Quarantined documents](#quarantined-documents)) |

The last three read the EMF metrics of `run-ingestion --metrics-sink cloudwatch`,
emitted as each stage finishes (a stopped run still reports its completed
stages). The task tags them with an `Environment` dimension (`env_name`), and
the alarms and dashboard read only that series, so dev runs do not trip prod's
alarms; runs without `UKR_METRICS_ENVIRONMENT` publish a series no alarm reads.

**Store health**

These fire between runs too, so degradation shows before a run breaks on it:

| Alarm | Fires when | What to do |
|---|---|---|
| `OpenSearchClusterRed` | A primary shard is unallocated (5 minutes) | Check cluster health and the failed node; queries on the index miss data, so hold ingestion until green or yellow |
| `OpenSearchFreeStorageLow` | The fullest data node has 40% or less of `opensearch_volume_gib` free | Raise `opensearch_volume_gib` (or `opensearch_count`) and redeploy before the write-block watermark |
| `OpenSearchJvmPressure` | JVM memory pressure at least 95% for 3 minutes | Use a larger `opensearch_instance` or more data nodes |
| `DocStatusWriteThrottled` | The doc-status table throttled a write (5 minutes) | Registry calls retry with backoff; one still throttled fails the run with `DocStatusRegistryError`. Look for overlapping runs, rerun once it clears |
| `NeptuneHighCpu` | Average writer CPU at least 90% for 15 minutes | Sustained saturation: a larger `neptune_instance` (or `neptune_serverless_ncu` maximum), or, if you raised it, a lower `indexing.neptune.index_concurrency` (default `1`) |
| `NeptuneLowFreeableMemory` | Writer freeable memory at or below 512 MiB at any point in 5 minutes; on Serverless only while the writer is at maximum capacity | Out-of-memory failures ahead: a larger `neptune_instance` (or `neptune_serverless_ncu` maximum) |
| `NeptuneServerlessAtMaxCapacity` | Serverless only: the writer is at the `neptune_serverless_ncu` maximum for 3 consecutive 5-minute periods | Raise the maximum (or the minimum, so capacity is there before a burst) |
| `NeptuneGremlinErrors` | Gremlin server errors on any instance for 3 consecutive minutes | Persistent failures (retried write conflicts come in short bursts and do not fire it): find the failing request in the task log |

**Capacity**

| Alarm | Fires when | What to do |
|---|---|---|
| `BedrockThrottling` | At least 100 throttled calls in each of 3 consecutive 5-minute periods, summed over every model in the account and region (other Bedrock users count too) | Throttled calls are retried, so the run slows rather than fails: lower `processing.max_concurrency` or ask for a higher quota |
| `IngestionTaskMemory` | The ingestion task used at least 90% of its memory for 3 minutes (Container Insights) | Fargate kills a task over its memory, losing the running stage: raise `fargate_memory` |

The dashboard also plots stage durations, task memory and CPU, and Bedrock
calls, throttles and latency by model, and the incremental degradations no
alarm watches (`total_documents_deferred`, `community_rebuild_recommended`,
`total_records_not_stored`, `underived_items`).

## Query access and tenant isolation

Queries (`run-rag`, `run-eval`, an embedded `GraphRAGChain`) get their own
task definition and role (`QueryTaskDefinitionArn`, `QueryTaskRoleArn`; see
[`iac/README.md`](../iac/README.md#4-query-from-inside-the-vpc)): Neptune
`connect` and `ReadDataViaQuery`; OpenSearch `GET`/`HEAD` and `POST` only on
`<index>/_search`, with an explicit deny for deeper paths that end in
`_search` but write (`POST <index>/_doc/_search` creates a document);
Bedrock model, rerank and guardrail calls; and the cache bucket's `eval/`
prefix. It has no DynamoDB access and cannot write or delete in either store.
Do not query with the ingestion task definition (its role can delete); a
serving layer of your own should assume a role with the same grants.

**The index suffix is a naming separation, not an access-control boundary.**
`RAGInput.suffix` / `--suffix` selects the indices and labels a query reads
(validated so it cannot widen the target), but any holder of the query role
can name any suffix. A multi-tenant serving layer must set it server-side
from the authenticated principal (a tenant claim in the caller's token),
never from a request field. To isolate tenants in the infrastructure:

- OpenSearch: one query role per tenant whose `es:ESHttpGet`, `es:ESHttpHead`
  and `es:ESHttpPost` resources name only its indices
  (`<domain ARN>/graphrag-*-<suffix>` and `<domain ARN>/graphrag-*-<suffix>/*`;
  pick suffixes that never end in `-<another suffix>`, or distinct index
  prefixes), keeping the deny on the document and bulk APIs. For document- or
  field-level rules, enable fine-grained access control and map each tenant
  role to an OpenSearch role limited to `*-<suffix>*`.
- Neptune: IAM authorizes a whole cluster, so a tenant whose graph others must
  not read needs its own cluster (one deployment or `env_name` per tenant).
- Bedrock and the guardrail are shared and hold no tenant data.

## Observability

Tracing is off by default; the settings are in the
[User Guide](./user-guide.md#observability). To send traces to a local
Jaeger, which accepts OTLP on port 4318:

```bash
docker run --rm -d --name jaeger -p 16686:16686 -p 4318:4318 \
  jaegertracing/all-in-one:latest
uv sync --extra otel
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

```yaml
observability:
  enabled: true
  backend: "otlp"
```

Run a query and open <http://localhost:16686>: the service `unified-kg-rag`
shows one `kg_rag.query` trace per query and one `kg_rag.ingestion.run` trace
per ingestion run. For a collector or a vendor endpoint, point
`OTEL_EXPORTER_OTLP_ENDPOINT` at it, pass credentials in
`OTEL_EXPORTER_OTLP_HEADERS` from Secrets Manager or task secrets, and set
`OTEL_EXPORTER_OTLP_PROTOCOL=grpc` for a gRPC endpoint (port 4317).

To check it works: the start-up log has
`Observability enabled (backend: 'otlp', ...)`; log records written inside a
span carry its `trace_id` and `span_id`; a query trace has `kg_rag.retrieve`,
`kg_rag.fusion`, `kg_rag.context_build` and `chat <model>` children, whose
`gen_ai.usage.input_tokens` is the run's token usage.

### Langfuse

Langfuse and Phoenix ingest OTLP, so they need only the `otel` extra, no
vendor SDK; their settings come from the environment only, and a standard
`OTEL_EXPORTER_OTLP_*` variable wins over the vendor one. A local Langfuse
runs from its Docker Compose file (Langfuse v4; v3 works the same way):

```bash
git clone --depth 1 https://github.com/langfuse/langfuse.git
cd langfuse && docker compose up -d     # UI on http://localhost:3000
```

Create a project and an API key pair in the UI, then:

```bash
uv sync --extra otel
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-...   # deployments: from Secrets Manager
export LANGFUSE_SECRET_KEY=sk-lf-...   # as task secrets, never in a file
```

```yaml
observability:
  enabled: true
  backend: "langfuse"
```

- `LANGFUSE_HOST` is required (there is no default host); for Langfuse Cloud
  use the region's URL, for example `https://cloud.langfuse.com`. The
  exporter posts to `<host>/api/public/otel/v1/traces` over HTTP only (gRPC
  fails at start-up); with `OTEL_EXPORTER_OTLP_ENDPOINT` instead, give
  `<host>/api/public/otel` (the host root returns 404).
- Both keys are required and sent as a Basic `Authorization` header with
  `x-langfuse-ingestion-version: 4`. If you set `OTEL_EXPORTER_OTLP_HEADERS`
  yourself, include that version header, or Langfuse Cloud can take minutes
  to show traces.
- Traces group in Sessions by `RAGInput.conversation_id` and in Users by
  `RAGInput.user_id`. Langfuse v4's default `events_only` mode disables
  `/api/public/traces`, `/observations` and `/sessions`: read spans through
  `/api/public/v2/observations` or the UI.

### Arize Phoenix

```bash
docker run --rm -d --name phoenix -p 6006:6006 -p 4317:4317 \
  arizephoenix/phoenix:latest            # UI on http://localhost:6006
uv sync --extra otel
```

```yaml
observability:
  enabled: true
  backend: "phoenix"
```

With nothing else set, spans go to `http://localhost:6006/v1/traces` (`:4317`
with `OTEL_EXPORTER_OTLP_PROTOCOL=grpc`) and land in the project named by
`service_name` (`PHOENIX_PROJECT_NAME` wins). For a remote Phoenix set
`PHOENIX_COLLECTOR_ENDPOINT` to the server root and, when it needs one,
`PHOENIX_API_KEY` (sent as a Bearer `authorization` header). Spans carry
OpenInference kinds (`LLM`, `EMBEDDING`, `RETRIEVER`, `RERANKER`, `CHAIN`), and
the Sessions view groups a conversation's queries.

### CloudWatch through the ADOT collector sidecar

CloudWatch accepts OTLP traces signed with SigV4, so a collector next to the
app signs and forwards them. On the CDK stacks, `-c enable_cloudwatch_tracing=true`
adds an AWS Distro for OpenTelemetry (ADOT) collector sidecar and points the
app at it. The whole recipe (turning on Transaction Search, the image, the
first collector pull, where traces appear, cost, the collector configuration
without the CDK stacks, and Bedrock model invocation logging as an
account-level backstop) is in
[iac/README.md: CloudWatch tracing](../iac/README.md#cloudwatch-tracing-optional).

### Trace shape

| Span | Parent | Attributes |
|---|---|---|
| `kg_rag.ingestion.run` | root | Status `ERROR` when the run failed |
| `kg_rag.ingestion.stage` | run | One per stage: input/output counts, status, the stage's numeric metrics as `kg_rag.stage.<metric>`, `error.type` and the exception when it crashed |
| `kg_rag.batch` | stage, or root for `run-eval` answer generation | Item and failure counts |
| `kg_rag.query` | root, or the `run-eval` batch | `session.id`, `user.id`, `kg_rag.blocked`; with `capture_content` the redacted query as `input.value` and (outside `stream` / `astream`) the answer as `output.value` |
| `kg_rag.strategy` | query | `kg_rag.rerank_status` for the whole query |
| `kg_rag.retrieve` | strategy | Hits per result type, up to 50 source ids |
| `kg_rag.fusion` | strategy | In/out counts, the step's `kg_rag.rerank_status` |
| `kg_rag.rerank` | fusion | `kg_rag.rerank.status` (`ok`, `empty`, `error`), document counts |
| `kg_rag.context_build` | query (`drift`: the round's actions span) | Tokens used vs the token budget |
| `kg_rag.global.select`, `kg_rag.global.map` | strategy (`global`) | Select, once per query: `kg_rag.depth` (configured), `kg_rag.cut` (the cut mapped after falling back to a shallower one), `kg_rag.reports`, `kg_rag.dynamic`. Map: `kg_rag.reports`, `kg_rag.map_calls`, `kg_rag.report_tokens` |
| `kg_rag.drift.primer`, `kg_rag.drift.actions` | strategy (`drift`) | Primer: `kg_rag.reports`, `kg_rag.answers`. One actions span per round (`kg_rag.depth`, `kg_rag.actions`), with the round's local searches under it |
| `kg_rag.guardrail` | query | One per screened query or answer: `aws.bedrock.guardrail.id`, `.source` (`input`, `output`), `.action` (`none`, `masked`, `blocked`), `.intervened`, `.policies` |
| `kg_rag.evaluate` | root | One per evaluator: `kg_rag.evaluator`, `kg_rag.items`; the judge's model spans under it |
| `chat <model>`, `embeddings <model>` | the calling span | OpenTelemetry GenAI conventions, kind `CLIENT` (as Bedrock rerank spans): `gen_ai.request.model`, `.max_tokens`, `.temperature`, `gen_ai.usage.input_tokens` / `output_tokens`, `gen_ai.usage.cache_read.input_tokens` / `cache_creation.input_tokens` when Bedrock reports prompt-cache use, finish reasons, `error.type`. No cost |

With `capture_content`, the answer model's `chat <model>` span keeps the
answer as generated, even when the guardrail then blocks or masks it, and
the `kg_rag.guardrail` span comes from the Bedrock guardrail adapter only (a
custom `ContentGuardPort` emits none).

`kg_rag.rerank.status` is the outcome of one Rerank call (`error`: the query
fell back to the fused order; `empty`: nothing usable came back), while
`kg_rag.rerank_status` is what reranking did, set even without a call:
`applied`, `disabled`, `unavailable` (no rerank model), `skipped` (nothing to
rerank) or `failed`; on the strategy span it is the worst step, as in
[`metadata.rerank.status`](./user-guide.md#rerank-status).

On export, spans keep their `gen_ai.*` and `kg_rag.*` attributes and gain the
vendor's names: Langfuse `langfuse.observation.type` (`generation` for model
calls, `embedding`, `retriever` for retrieval and rerank, `span` otherwise)
and, with `capture_content`, `langfuse.observation.input` / `output`; Phoenix
`openinference.span.kind`, `llm.model_name`, `llm.provider`,
`llm.token_count.*` and, with `capture_content`, `input.value` /
`output.value`.

Known limits:

- A streamed answer (`stream` / `astream`) is its own trace; its model span
  takes `session.id` / `user.id` from the answer step's run metadata, so it
  still groups with its session.
- The async batch path records one span per batch, not per chunk.
- A model call cancelled during `ainvoke` (for example by `asyncio.wait_for`)
  leaves no model span: LangChain reports no callback for it. A stream the
  caller stops reading early ends its span normally.
- With `otlp`, `langfuse` or `phoenix` the first backend the process installs
  also becomes OpenTelemetry's global tracer provider, which can be set only
  once: after a later configuration with other settings, the package's spans
  use the new exporter, but spans other libraries emit through the global API
  are dropped.

### Export limits, private VPCs and privacy

- **Export never blocks a run.** Spans leave from a background thread; each
  request is bounded by `observability.export_timeout_seconds` (10 s;
  `OTEL_EXPORTER_OTLP_TIMEOUT` wins), a full queue drops spans, and failures
  are logged at `ERROR`. Against an unreachable collector the flush at exit
  takes about one timeout per remaining batch of 512 spans, at most about
  30 s, then drops the rest.
- **Private VPCs.** The endpoint must be reachable from the tasks: a
  collector sidecar on `localhost`, a Langfuse or Phoenix inside the VPC, or
  the X-Ray interface endpoint for the collector; hosted Langfuse or Phoenix
  needs internet egress, which `private` mode does not have. Build the image
  with `--build-arg UV_EXTRAS="--extra otel"`.
- **Credentials** (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`,
  `PHOENIX_API_KEY`, `OTEL_EXPORTER_OTLP_HEADERS`) come from the environment
  only: inject them as ECS task secrets. The start-up line
  `Observability exporter for backend '...'` shows the endpoint and header
  names, never values.

Spans carry counts, ids, model names and token usage only.
`capture_content: true` adds prompt and completion text to model spans, and
the query and answer to the `kg_rag.query` span (after `redact_patterns`):
enable it only where the trace backend may hold corpus data. With `otlp`,
`langfuse` or `phoenix`, `redact_patterns` also applies at export to every
string attribute, event attribute (an exception message can quote a query)
and status message, `session.id` and `user.id` included; with `none` the
host's pipeline exports spans as recorded, so redact in its collector.
`RAGInput.user_id` is exported as given: pass an opaque id, not an e-mail
address. With LangSmith tracing on (`LANGSMITH_TRACING=true`), LangChain
uploads content to LangSmith independently of these settings.

## Common errors

Messages are abridged; the log line names the key, file or stage. A store,
model or credential error fails a query rather than answering from no context.

### Configuration and start-up

| Message (abridged) | Meaning | Action |
|---|---|---|
| `Missing endpoint configuration for ...` / `Configuration validation error: ...` | An endpoint the run needs is unset, or a value has the wrong type | Set or fix the named key |
| `ConfigurationError: Unknown config keys: '<path>' ...` (exit 2) | A typo, or a key dropped without a replacement (a renamed key fails with `settings moved; replace <old> -> <new>`; a removed one, `indexing.cross_run_merge` or `aws.bedrock.guardrail.apply_to`, only logs `Config key ... is removed and ignored`) | Fix or delete the key; `--allow-unknown-keys` loads anyway with a WARNING per key |
| Usage errors (exit 2): `--source-directory is required for pipeline execution`, `--s3-sync needs --s3-bucket-name or aws.s3.bucket_name`, `--pipeline-id is required for --resume-from-stage`, `--enabled-stages: invalid stage names ...`, `argument --resume-from-stage: Unknown pipeline stage ...` | A required flag is missing, or a stage name is unknown | Pass the flag, or a stage name from the message ([CLI Reference](./cli.md#run-ingestion)) |
| `... needs confirmation, but stdin is not a terminal` | Completed stages' outputs are no longer cached under the current inputs | `--yes` (recomputes them), or a new `--pipeline-id` |
| `Stage '<stage>', upstream of this run's enabled stages, has not completed` | An earlier stage failed or lost its output | Run the earlier stages first |
| `observability.enabled is true but OpenTelemetry is not installed` | The `otel` extra is missing | `uv sync --extra otel`, or turn tracing off |
| `Skipping N '.md' file(s) ...` / `No supported source files found in '<dir>'` | The `unstructured` extra is not installed | Install it (Python 3.11+) or convert the files |

### Credentials and models

| Message (abridged) | Meaning | Action |
|---|---|---|
| `No valid AWS credentials for Amazon Bedrock in region ...` | Missing or expired credentials | Refresh them; check `AWS_PROFILE` / `aws.profile_name` |
| `Model '<id>' is only available through a cross-region inference profile, but none resolved ...` | No on-demand throughput and no profile found | Keep `aws.bedrock.enable_global_profile: true`, grant `bedrock:ListInferenceProfiles`, or use another region |
| `Reranking failed: ...` (per query, `metadata.rerank.status: failed`) / `Reranking initialization failed; queries keep the fused order ...` (`unavailable`) | The Rerank call was denied or the model is not in the region; or the rerank model could not be built | Grant `bedrock:Rerank` on `Resource: "*"`, use a region with the model, check `models.rerank` / `providers.rerank`, or set `search.reranking.enabled: false` |
| `RerankModelError: ...` | As above, with `search.reranking.required: true` | Fix the cause, or `required: false` to answer from the fused order |
| `ClientError: ... (ValidationException) ... InvokeModel ...` / `(ResourceNotFoundException)` / `NoCredentialsError` during a query | The query embedding call cannot succeed | Check `models.embedding.model_id`, `aws.bedrock.region_name`, model access and credentials |
| `EmbeddingModelError: Query embedding failed: ...` | The embedding call failed after retries (a crashed Ollama runner or vLLM engine, a persistent `InternalServerException`) | Check the embedding server and its logs; retry |
| Bedrock calls hang in a private VPC | Bedrock is called in another region than the VPC endpoint's | Keep `aws.bedrock.region_name` unset or the VPC's region |

### Stores: connection, permissions, requests

| Message (abridged) | Meaning | Action |
|---|---|---|
| `StoreConnectionError: ... Cannot connect to host ...` / `Neptune is not reachable ...` / `... status=429 ... (still failing after the client's retries)` | Refused or unresolved, closed by Neptune (restart, failover; reopened next time), or OpenSearch 429/502/503/504 for all `search.store_max_attempts` attempts | Check endpoint, port, that the store runs, security groups and VPC endpoints |
| `OpenSearch did not answer a request on <index> within N s in M attempt(s)` / `Neptune did not answer within N s` | No answer within `search.store_timeout_seconds` (ingestion: Neptune fixed 180 s, item retried) | Check the store's health and load, or raise the timeout |
| `OpenSearch denied the search on <index>: ... status=403 ...` (or `401`) | Fine-grained access control: the role lacks `indices:data/read/search`; or bad credentials | Grant it (and `indices:admin/mappings/get`), or fix the credentials |
| `Neptune refused the connection (HTTP 403)` | No `neptune-db:connect` on the cluster, or signed for another region | Grant it (and the needed `neptune-db:*` data actions); set `aws.region_name` to the cluster's region |
| OpenSearch authentication fails with `use_iam: false` | No basic-auth credentials | Set `UKR__AWS__OPENSEARCH__USERNAME` / `__PASSWORD` (for example in `.env`) |
| `OpenSearch rejected the search on <index>: ...` | 400 `query_shard_exception`: a filter value of the wrong type. 500 `too_many_clauses` / `too_many_nested_clauses`: `indices.query.bool.max_clause_count` below 1,024 | Fix the value, or keep the limit at 1,024 or more |
| `Cannot create the OpenSearch index '<index>': the cluster has no analyzer '<name>'` (`mapper_parsing_exception`) | An analyzer in `indexing.opensearch.language_analyzers` is missing, typically `ko: nori` without the Nori package | Use one the cluster has (`ko: cjk`, the CDK default), or associate the package; re-run |

### Ingestion runs and the doc-status registry

| Message (abridged) | Meaning | Action |
|---|---|---|
| `Pipeline failed at stage(s): ...` | A stage failed; exit 1 | Fix from the stage's log and resume with the same `--pipeline-id` (`--force-rebuild` starts clean) |
| `CacheSyncError` | The S3 stage-cache download or upload failed | Check bucket permissions and `aws.s3.encryption`; resume |
| `DocStatusRegistryError` | The registry table is missing, not accessible, or throttled | Create it or fix `aws.dynamodb.table_name`, grant the permissions the message names; resume |
| `Incremental indexing is enabled but no document delta was computed` | Indexing ran without a registry diff (loading failed under `continue_on_error`) | Re-run from `document_loading` once the registry is reachable |
| `NamespaceLeaseError: Index namespace '...' is being written by ...` | Another run holds the write lease | Wait; a dead run's lease expires within `indexing.lease_seconds` ([Concurrent runs](#concurrent-runs)) |
| `NamespaceLeaseError: This run lost the write lease ...` | Renewals failed until the lease expired, or another run took it | Re-run after the other run; check the registry is reachable |
| `ConfigurationError: Doc-status registry: DynamoDB table '...' has the key schema ... of an earlier release` | A table keyed by `doc_id` alone | Point `aws.dynamodb.table_name` at a new table (or the stack's), and run once with `indexing.reset: true` ([Incremental runs](#incremental-runs)) |
| `ConfigurationError: The OpenSearch index '...' records no embedding settings ...` | An earlier release built the index | Run once with `indexing.reset: true` |
| `The doc-status registry lists indexed documents of this namespace whose entities the graph store does not hold` (or `whose text units the vector store does not hold`) | A store lost what the registry records (a restarted Gremlin Server, `down -v`, a restored graph, deleted indices, a moved `vector.sqlite3`), or, in an index a pre-#383 build wrote, a deleted document was restored while its removal was unfinished | Re-run with `indexing.reset: true`, or delete the namespace's records; locally delete the three store files together or restore the moved one. For the restored document, deleting that one document's registry record is enough: the next run indexes it again |

### Queries

| Message (abridged) | Meaning | Action |
|---|---|---|
| `No indices found for suffix '<suffix>' ...` | Nothing was ingested under that suffix | Ingest, or query the corpus's suffix |
| `InvalidFilterError: Unknown filter key(s) ...` / `... has no value (None)` / `... range bound of None` / `... not a scalar` / `... lists N values ...` | No store the strategy reads declares the filter key; a filter value is `None` (alone, in a list or as a range bound: OpenSearch rejects such a query); a list holds a list or mapping; or a list filter holds more values than one `terms` query takes (`index.max_terms_count`, default 65,536) | Use a key the message lists; drop the `None` or give the value to match; list strings, numbers or booleans; narrow the list (or raise `max_terms_count` in `indexing.opensearch.index_settings` and reindex) |
| LightRAG `mix` / `hybrid` returns nothing | No relationship index, or no keywords extracted | Check the relationships index exists; short queries fall back to the raw query only under `raw_query_fallback_max_len` |

### Local mode

| Message (abridged) | Meaning | Action |
|---|---|---|
| `... stayed locked ...` (`database is locked`) | Another process held the SQLite write lock longer than `search.store_timeout_seconds` | One ingestion per store file at a time, or raise the timeout; use a local disk |
| `... processed N inputs but produced 0 outputs`, with `Connection refused` in the log | Ollama or the OpenAI-compatible server is not reachable at `providers.llm.base_url` / `providers.embedding.base_url` | Start it (`ollama serve`, or `docker compose -f docker/compose.local.yaml --profile models up -d --wait`), check `base_url` (`--model-base-url`), pull the models |
| `AuthenticationError` / `NotFoundError` (401 / 404), `APIConnectionError` / `APITimeoutError`, `ResponseError: model ... not found` | Key refused, model not served, or server unreachable | Check `base_url`, `api_key` and the model names under `providers.llm` / `providers.embedding`; pull the models |
