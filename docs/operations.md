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
   `AWS_REGION` sets it only when the file leaves it unset, and
   `UKR__AWS__REGION_NAME` overrides it (Bedrock follows it while
   `aws.bedrock.region_name` is unset). Check with
   `env | grep -E '^(AWS_REGION|UKR__AWS__)'` and `aws sts get-caller-identity`.
2. **Models.** The models in
   [Required models](./user-guide.md#required-models) are enabled in the
   Bedrock region.
3. **Endpoints.** `run-ingestion`, `run-rag` and `run-eval` check at start-up,
   before any model call, that the store endpoints they need are set (see
   [Start-up endpoint check](#start-up-endpoint-check)).
4. **No other ingestion is running** against the same stores. With the
   doc-status registry a second run on one index namespace stops with
   `NamespaceLeaseError`; without it concurrent runs are not supported (see
   [Concurrent runs](#concurrent-runs)). On the CDK stack:
   `aws stepfunctions list-executions --state-machine-arn <arn> --status-filter RUNNING`.

## Routine operations

### Re-ingest

Three operations look alike but act on different state:

| Goal | How | Stage cache | Stores and registry |
|---|---|---|---|
| Finish an interrupted run | Re-run with the same `--pipeline-id` (on the CDK stack, start a new execution with the same `pipeline_id`); it resumes at the first failed or incomplete stage and skips the stages that completed. `--resume-from-stage <stage>` picks the stage. | Reused | Written by the stages that run |
| Recompute every stage | `--force-rebuild` | Ignored and rewritten | Written as usual (incremental when enabled) |
| Rebuild the stores from scratch | `indexing.reset: true` for one run | Unchanged | Rebuilt without going offline: new vector indices replace the old ones by an alias swap, the graph is rewritten in place and what the rebuild did not write is dropped at the end; every document is recorded again, then the namespace's other records are deleted (other namespaces in the table keep theirs). A failed reset drops nothing |

Use `indexing.reset: true` (and `--force-rebuild` if stage outputs must change
too) after changing anything that is baked into the stored data:

- the embedding model or `models.embedding.dimension`;
- `processing.translation.source_language` or the language analyzers (they
  change the text-unit mapping);
- to refresh the communities incremental runs append to, and to give
  entities the ids of their current names (see
  [Incremental runs](#incremental-runs));
- a built-in prompt template (code) you edited. Other extraction settings
  (prompts, `entity_types`, extraction or chunking models, chunk sizes...)
  need no reset: an incremental run re-extracts the unchanged documents
  whose stored fingerprint differs from the current settings on its own and
  logs how many (see [User Guide §5](./user-guide.md#workflows)).

During a reset, queries keep reading the previous content: the vector
indices swap aliases one index type at a time, and the graph holds a mix of
previous and rebuilt vertices and edges until the rebuild drops what it did
not write. A reset that fails leaves the previous graph and registry records
in place (beside what it wrote); re-run it.

Set `indexing.reset` back to `false` afterwards; otherwise every run rebuilds.

`--verify-metadata` (needs `--pipeline-id`) checks a run's stage metadata and
exits non-zero when it is missing or corrupt.

### Incremental runs

With `aws.dynamodb.enabled: true` (the container image sets it), every
`run-ingestion` diffs the corpus against the doc-status registry by content
hash and indexes only the new and changed documents. A file missing from the
source is treated as deleted, and what only it stated is removed.

The diff is scoped to the run's [index suffix](./user-guide.md#index-suffix)
and its source scope (`processing.document_parsing.source_scope`; by default
the resolved source directory, and the `s3://` source URI in the container).
Both are part of each document's registry key, so two corpora on one suffix
never share a record, even for files with the same relative path. An `s3://`
URI with and without a trailing slash is one scope: a run uses the spelling
the registry holds and logs a WARNING. Therefore:

- **Keep the same source directory** (or a fixed `source_scope`) for a
  corpus. Moving a corpus with the default scope changes the scope, and the
  old scope's records are never adopted or deleted on their own, so the old
  content stays indexed. Set a fixed `source_scope` before moving it, or run
  once from the new location with
  `--retire-source-scope <old source directory>` ([User Guide: Deletion scope](./user-guide.md#deletion-scope)).
  A run with new documents logs a WARNING naming the other local source
  directories of its suffix.
- **Never let corpus files expire or disappear** unintentionally: the next
  run removes their graph and vector content.
- **Index each tenant or corpus version under its own index suffix**, with
  its own config file.

**What incremental runs do not refresh.** Delta runs append communities
rather than re-clustering, and stored communities and reports keep listing an
entity's old id after the entity took its current name's id. A periodic full
rebuild (`indexing.reset: true`) refreshes both; shared descriptions, names,
ids, types, ranks and claims are derived from the current documents on every
run ([User Guide §5](./user-guide.md#5-incremental-indexing)).

**When to rebuild: community drift.** A delta run's indexing stage reports
`community_drift` metrics: the communities it appended (`delta_communities`),
the entities it clustered (`clustered_entities`), how many of those an
earlier clustering already placed (`overlapping_entities`, now in two or more
level-0 communities), the earlier communities whose reports miss what the run
added (`stale_communities`), and `overlap_share` (overlapping / clustered).
When `overlap_share` reaches `indexing.community_drift_threshold` (default
`0.5`) the run logs a WARNING recommending `indexing.reset: true` and reports
`community_rebuild_recommended: 1` next to `community_drift_overlap_share`,
which an alarm can watch. The counts accumulate over the runs that touch an
entity, so read them as an upper bound. A run without communities or with
`indexing.cross_run_merge: false` reports no overlap.

**When the registry cannot be read** (missing table, denied access,
throttling that outlasts the client's retries), the `document_loading` stage
fails with `DocStatusRegistryError`, naming the table and the cause. The run
does not fall back to indexing every document, which would replace the
suffix's content with this run's documents only. Fix the registry and re-run
with the same `--pipeline-id`, or set `aws.dynamodb.enabled: false` to run
without incremental indexing.

Coming from a pre-release build: nothing it stored is read; see
[CHANGELOG: Upgrading from pre-release builds](../CHANGELOG.md#upgrading-from-pre-release-builds).

### Caches and prefixes

| Cache | Location | Notes |
|---|---|---|
| Stage checkpoints (local) | `<--cache-directory>/<pipeline_id>/` (default `cache/`) | Reused on resume; `--force-rebuild` ignores it and writes the recomputed outputs |
| Stage checkpoints (S3) | `s3://<bucket>/<--s3-prefix>/<pipeline_id>/` (default prefix `pipeline-runs`) | With `--s3-sync`; uploaded after every stage and on SIGTERM; a re-run in a fresh task resumes from it. A failed initial download or final upload fails the run with `CacheSyncError`. Expired by the bucket lifecycle, not `cache.ttl_seconds` |
| LLM response cache | `<stage checkpoint dir>/llm-cache/` (local, and in S3 with `--s3-sync`) | Ingestion model responses reused by a re-run with the same pipeline id; `cache.llm_responses.enabled: false` turns it off, `--force-rebuild` ignores it |
| Embedding cache | `s3://<aws.s3.bucket_name>/embedding-cache/cache.json` (`indexing.opensearch.embedding_cache_s3_key`) | Only with `indexing.opensearch.persist_embedding_cache: true` |
| Visualization data | `<cache directory>/<pipeline_id>/visualization/` unless `graph.visualization.outputs_directory` is set | Synced with the stage cache (`.json` only) |

The CDK cache bucket expires objects after 30 days under `pipeline-runs/` and
`embedding-cache/` only. Put the corpus under another prefix (for example
`corpus/`): an expired corpus file looks deleted to the next incremental run.
A reused bucket (`cache_bucket_name`) keeps its own lifecycle rules, so check
them.

A `pipeline_id` names the cache directory and S3 prefix, so it accepts only
letters, digits, `.`, `_` and `-`: up to 128 characters, starting with a
letter or digit, with no `..`.

### OpenSearch replicas

On the CDK stack, a multi-node domain (`opensearch_count` > 1, the non-dev
default) gets one replica per index, so losing a node or an AZ keeps every
shard available; a single-node dev domain gets none. The compute stack sets
`UKR__INDEXING__OPENSEARCH__INDEX_SETTINGS__NUMBER_OF_REPLICAS`, which overrides
`indexing.opensearch.index_settings.number_of_replicas` when an index is
created. Indices created before that keep their replica count, so set it on
them once from inside the VPC:

```bash
curl -X PUT "https://<OpenSearch endpoint>/_all/_settings" \
  --aws-sigv4 "aws:amz:<region>:es" --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
  -H "x-amz-security-token: $AWS_SESSION_TOKEN" \
  -H 'Content-Type: application/json' -d '{"index": {"number_of_replicas": 1}}'
```

### Backup and restore

The three stores are backed up on different schedules, and their contents
refer to each other (the registry lists each document's text units, which the
graph and index artifacts cite), so restore them to one common point or not at all:

| Store | Backup | Window |
|---|---|---|
| Neptune | Automated backups, point-in-time restore | `backup_retention_days` (default 7 days) |
| OpenSearch | Automated hourly snapshots (`cs-automated-enc` repository) | 14 days |
| DynamoDB doc-status table | Point-in-time recovery | 35 days |
| S3 cache bucket | None (unversioned; stage checkpoints are re-creatable) | — |

The corpus in S3 is the source of truth: every store can be rebuilt from it.

**Within Neptune's window**, restore all three to the same time:

1. Stop ingestion (no `RUNNING` execution) so nothing writes during the restore.
2. Pick an OpenSearch snapshot (`GET _snapshot/cs-automated-enc/_all`) taken
   after the last ingestion run finished, and use its start time as the
   restore point.
3. Restore Neptune to that time with
   `aws neptune restore-db-cluster-to-point-in-time`, and the table with
   `aws dynamodb restore-table-to-point-in-time`. Both create *new* resources
   with default settings (add instances to the Neptune cluster, re-enable
   PITR and deletion protection on the table).
4. Delete the domain's current indices, then restore the snapshot's indices
   and aliases (`POST _snapshot/cs-automated-enc/<snapshot>/_restore`).
5. Point the data plane at the restored cluster endpoint and table
   (`UKR__AWS__NEPTUNE__ENDPOINT`, `UKR__AWS__DYNAMODB__TABLE_NAME`). The task role, the
   DynamoDB gateway-endpoint policy and the alarms are scoped to the stack's
   own cluster and table, so grant them on the restored resources too.

**Outside Neptune's window** (or when step 5 is not worth it), there is no
common point: run one full rebuild with `indexing.reset: true` (see
[Re-ingest](#re-ingest)). It re-indexes the whole corpus into this index
namespace and drops what the rebuild did not write. Restore the table first
only if other namespaces share it.

## When a run fails

### Interrupted runs

An incremental run that stops inside the indexing stage (the task is killed,
a store or the registry becomes unreachable, the failure gate trips) needs no
manual repair: re-run. Before it removes or writes anything, the indexing
stage records every new and changed document `PENDING`, listing both the text
units it had and the ones the run is about to write; the records become
`PROCESSED` (or `FAILED`) only after the writes. The next run therefore:

- re-extracts each `PENDING` document still in the corpus, even if its file
  was reverted to the version indexed before;
- removes everything the interrupted run may have written for a `PENDING`
  document that is no longer in the corpus.

**Leave `PENDING` records alone.** They (`status` `pending`, `content_hash`
`pending`, `error_info` naming the write-ahead record) are expected after a
failed run, and their text units are what lets the next run find the
interrupted run's writes, since every artifact cites the text units it was
stated in.

- An interruption is not counted as a failure, but it resets the document's
  `failure_count` toward `indexing.max_document_failures`. The `PENDING`
  record counts instead the consecutive runs that started the same content
  without finishing it (`attempt_count`, see
  [Quarantined documents](#quarantined-documents)).
- A document with more text units than one DynamoDB item holds (about
  10,000; half that for an edited document, whose `PENDING` record lists the
  old and the new ones) fails the run before anything is written: split it.
- Extraction records that only removed text units named are deleted after
  each successful commit; a run that stops between its removal and its
  commit leaves them behind, which costs space only.
- The stage writes the registry in batches, so a role with a custom policy
  needs `dynamodb:BatchWriteItem` as well as `BatchGetItem` (the CDK stack's
  grant covers both).

### Failed documents

A document is recorded `FAILED` in four cases:

- translation, graph extraction, gleaning or claim extraction failed on one
  of its text units;
- a base-level community covering one of its text units got no report (a
  failed parent-level report marks no document; see
  [User Guide: Failed documents](./user-guide.md#failed-documents));
- an artifact stage (translation through community detection) crashed in a
  `--continue-on-error` run, which records every document of the run
  `FAILED`;
- writing one of its artifacts (text unit, entity, relationship, claim,
  community, report) to the stores failed.

The next run treats a `FAILED` document as changed: it removes what the
failed run wrote and processes the document again. Any other stage failure
records nothing: when parsing, loading or chunking crashed in a
`--continue-on-error` run, or any stage failed without it, the run stops
before indexing and the stores and the registry keep their previous state,
because a stage failure does not say whether one document or the backend
(throttling, an outage, expired credentials) is at fault. The next run
processes the same new and changed documents again.

- **Retry limit.** After `indexing.max_document_failures` (default `3`)
  consecutive failures with unchanged content, the document is no longer
  retried: it stays `FAILED`, keeps what it indexed, and each run logs a
  WARNING naming the file. To retry it, fix the cause (often an oversized or
  malformed file), edit the file so its content hash changes, or raise the
  limit.
- **Too large for the registry.** A document whose text-unit ids exceed one
  DynamoDB item (400 KB, roughly 10,000 ids, half that for an edited
  document) fails the indexing stage before anything is written for the
  delta, with an error naming the file and its text-unit count. Split the
  file.

**Write failures** are retried first. OpenSearch bulk items rejected with a
retryable status (429, 502, 503, 504), and a whole bulk request still
answered with one after the client's own retries, are resent up to four times
with backoff. Every Neptune request of a run is retried up to
`indexing.neptune.max_attempts` times (default `4`) over a new connection,
and each is safe to repeat. If more than `indexing.max_failure_rate` (default
`0.2`) of one artifact type's writes still fail, the indexing stage fails and
records no document, so the next run retries them all. Below that rate the
run succeeds and only the documents that own a failed artifact are recorded
`FAILED` (a failure the backend reports without an item id marks every
document of the run). The `IndexingFailures` and `ExtractionFailures`
[alarms](#alarms) report both.

### Quarantined documents

A document whose indexing kills the run each time (the task runs out of
memory or hits its timeout on it) would stop every later run the same way.
Once runs have started its unchanged content `indexing.max_document_attempts`
times in a row (default `3`) without finishing it, the next run quarantines
it: it logs a WARNING naming the file, removes what the attempts wrote,
records it `QUARANTINED` (no text units, the reason in `error_info`) and
indexes the rest. The run metric `total_documents_quarantined` counts them,
and on the CDK stack the `DocumentsQuarantined` alarm fires.

- A quarantined document is skipped while its content stays. It is retried
  when the file changes, when the extraction settings change, or when
  `indexing.max_document_attempts` is raised.
- Every document of a delta that keeps dying counts an attempt, so the
  documents indexed together with the culprit are quarantined with it.
  Retry them by touching their files (or raising the limit) once the
  culprit is found, for example by running its folder alone.
- A document that produces no text units stays `PENDING` and is quarantined
  after the same number of runs.

### Concurrent runs

With the doc-status registry, one run writes an index namespace at a time. A
run takes its namespace's write lease before its first stage and holds it to
its end, renewing it every third of `indexing.lease_seconds` (default 300 s).
A second run on the same namespace stops at once, before it reads or writes
anything, with:

```text
NamespaceLeaseError: Index namespace '<namespace>' is being written by
<host>:<pid>:<id> (its lease runs until <time> unless renewed) ...
```

Wait for that run to finish and start again. A run that dies (a killed task,
a lost host) stops renewing, so its lease expires at most `lease_seconds`
after its last renewal and the next run proceeds; nothing needs cleaning up.
A run whose renewals keep failing until its lease expires, or whose lease
another run then took, stops before its next write with `NamespaceLeaseError:
This run lost the write lease ...`: re-run once the other run ended.

- **Where the lease lives.** In DynamoDB, the item with partition key
  `namespace` = `<namespace>` and sort key `doc_id` = `#lease` (owner in
  `lease_owner`, expiry in `lease_expires_at`, the table's TTL attribute, so
  DynamoDB also deletes it once expired); in the SQLite registry, the
  namespace's row of the `leases` table. Delete it by hand only after making
  sure the run named in it is gone.
- **Permissions.** A registry on a custom policy needs `dynamodb:PutItem`,
  `GetItem` and `DeleteItem` for the lease (the CDK stack's grant covers
  them), and `dynamodb:UpdateTimeToLive` when the library creates the table.
- **What a namespace is.** The index suffix (`index_value`, `default` when
  unset) plus `indexing.additional_suffix`, plus a tag of the
  `indexing.opensearch.*_index_prefix` and `indexing.neptune.*_label_prefix`
  values when any is not the default. Two configurations that differ only in
  those prefixes write different stores, so they keep separate records even
  in one table. Runs on different namespaces lease apart and can run side by
  side.

**Without the registry nothing locks**, and concurrent runs are not
supported. Do not start a run while another one runs against the same index
suffix (the same OpenSearch aliases and Neptune labels): a full reindex writes
a new timestamped index, swaps the alias onto it and deletes the alias's
older indices, so one run can delete the index the other is writing. The
direct `IncrementalIndexer` API takes no lease either.

**On the CDK stack**, each `start-execution` starts one execution that runs
the ingestion task, and the stack defines no schedule or trigger. The state
machine does not stop a second execution from starting while one runs; the
lease then fails the second one. Check for a `RUNNING` execution first
([Before a run](#before-a-run)), and let a schedule you add fire only when a
previous execution cannot still be running (up to `sfn_timeout_hours`, 6
hours by default).

### Start-up endpoint check

Before the first paid model call, the CLIs check that the store endpoints the
run needs are set, and exit with status 2 (a configuration error) otherwise:

```text
run-ingestion: error: Missing endpoint configuration for the indexing stage: aws.neptune.endpoint (env UKR__AWS__NEPTUNE__ENDPOINT), aws.opensearch.endpoint (env UKR__AWS__OPENSEARCH__ENDPOINT). Set them in the config file or the environment.
```

- `run-ingestion` checks both endpoints when the `indexing` stage is enabled
  (not for `--verify-metadata`).
- `run-rag` and `run-eval` check the endpoints the chosen strategy's retrievers
  need: OpenSearch for every strategy, and Neptune as well for `local`,
  `drift`, `mix` and `hybrid`. `auto` checks every strategy it may route to: `search.auto_routable_strategies` without `global` and `drift` when `graph.community_detection.enabled` is false.

The check only tests that a value is set. An endpoint that is set but wrong or
unreachable fails at the first connection; the retrievers report such
authentication, configuration and connection errors instead of returning empty
results.

## Alarms

On the CDK stack, a failed ingestion task publishes a message with the
execution id and the error to the SNS topic, and the alarms below report to
the same topic: five on pipeline runs, seven on store health and two on
capacity. The alarms have a subscriber only when the stack was deployed with
`-c alarm_email=<address>`.

`PipelineFailures` also fires for a failed task, so such a failure arrives
twice, the second time without context; the alarm stays because it also
covers failures the message cannot report (an error the task's handler cannot
catch, such as a start input without `pipeline_id`, or a publish that fails
after its retries).

**Pipeline runs**

| Alarm | Fires when |
|---|---|
| `PipelineFailures` | A Step Functions execution failed |
| `PipelineTimedOutOrAborted` | An execution hit the state machine timeout or was stopped |
| `IndexingFailures` | Indexing reported failed items, even in a succeeded execution |
| `ExtractionFailures` | Extraction, gleaning or claim extraction failed on some units |
| `DocumentsQuarantined` | A run quarantined documents whose indexing kept dying (see [Quarantined documents](#quarantined-documents)) |

The pipeline-run alarms other than the Step Functions ones read the EMF metrics
`run-ingestion --metrics-sink cloudwatch` emits, each stage's as the stage
finishes (so a run that is stopped or times out still reports the stages it
completed) and the run's own (duration, cache hit rate) when it ends. The
task tags them with an `Environment` dimension, the stack's `env_name`, and
the alarms and dashboard read only that series: a dev deployment's runs in
the same account and region do not trip a prod deployment's alarms. Runs
without `UKR_METRICS_ENVIRONMENT` (outside the CDK task) publish a series
without dimensions, which no alarm reads.

**Store health**

The store-health alarms fire between runs too, so degradation shows before
the next run breaks on it:

| Alarm | Fires when | What to do |
|---|---|---|
| `OpenSearchClusterRed` | A primary shard is unallocated (5 minutes) | Check the domain's cluster health and the failed node; queries on the affected index miss data, so hold ingestion until it is green or yellow |
| `OpenSearchFreeStorageLow` | The fullest data node has 40% or less of `opensearch_volume_gib` free | Raise `opensearch_volume_gib` (or `opensearch_count`) and redeploy before the node reaches the write-block watermark |
| `OpenSearchJvmPressure` | JVM memory pressure is at least 95% in 3 consecutive minutes | Out-of-memory risk: use a larger `opensearch_instance` or more data nodes |
| `DocStatusWriteThrottled` | The doc-status table throttled a write (5 minutes) | The app retries throttled registry calls with backoff; one still throttled after that fails the run with `DocStatusRegistryError`. Check for overlapping runs on the table and rerun a failed run once the throttling clears |
| `NeptuneHighCpu` | Average writer CPU is at least 90% for 15 minutes (3 × 5 minutes) | Sustained saturation, not a bulk-write peak: use a larger `neptune_instance` (or a higher `neptune_serverless_ncu` maximum) or lower `indexing.neptune.index_concurrency` |
| `NeptuneLowFreeableMemory` | Writer freeable memory falls to 512 MiB or less at any point in 5 minutes (minimum) | Queries and writes may fail with out-of-memory errors: use a larger `neptune_instance` (or a higher `neptune_serverless_ncu` minimum, so capacity is there before the burst) |
| `NeptuneGremlinErrors` | Gremlin server errors on any instance in 3 consecutive minutes | Persistent write or query failures (retried write conflicts come in short bursts and do not fire it): check the task log for the failing request |

**Capacity**

The capacity alarms catch a run that slows down or is about to be killed:

| Alarm | Fires when | What to do |
|---|---|---|
| `BedrockThrottling` | Bedrock throttled at least 100 calls in each of 3 consecutive 5-minute periods, summed over every model in the account and region (Bedrock publishes per model only, and the stack does not know which models the app resolves) | The run retries each throttled call, so it slows down rather than fails: lower `processing.max_concurrency` or ask for a higher Bedrock quota. Other Bedrock users in the account and region count too |
| `IngestionTaskMemory` | The ingestion task used at least 90% of its memory for 3 minutes (Container Insights) | Fargate kills a task over its memory, which loses the running stage: raise `fargate_memory` |

The dashboard also plots each stage's duration, the ingestion task's memory
and CPU use, and Bedrock invocations, throttles and latency by model.

## Query access and tenant isolation

Queries (`run-rag`, `run-eval`, or an application that embeds
`GraphRAGChain`) need far less than ingestion. The CDK stack gives them their
own task definition and role (`QueryTaskDefinitionArn`, `QueryTaskRoleArn`
outputs of `GraphRagCompute`; see
[`iac/README.md`](../iac/README.md#4-query-from-inside-the-vpc)): Neptune
`connect` and `ReadDataViaQuery`, OpenSearch `GET`/`HEAD` on the domain and
`POST` only on `_search` paths (OpenSearch authorizes by HTTP method and URL
path, so `_bulk`, `_doc`, `_delete_by_query` and index changes are denied),
Bedrock model, rerank and guardrail calls, and the cache bucket's `eval/`
prefix. It has no DynamoDB access and cannot write or delete in either store.
Run queries with it, not with the ingestion task definition, whose role
holds `neptune-db:DeleteDataViaQuery`, `es:ESHttp*` and read/write on the
doc-status table. A serving layer of your own should assume a role with the
same grants.

**The index suffix is a naming separation, not an access-control boundary.**
`RAGInput.suffix` / `--suffix` selects which indices and labels a query
reads, and the chain validates it so it cannot widen the index target, but
any caller that holds the query role can name any suffix. A multi-tenant
serving layer must therefore set the suffix server-side from the
authenticated principal (for example a tenant claim in the caller's token),
never from a request field. When tenants must be isolated by the
infrastructure rather than by that code:

- OpenSearch: give each tenant its own query role whose `es:ESHttpGet`,
  `es:ESHttpHead` and `es:ESHttpPost` resources name only its indices, such
  as `<domain ARN>/graphrag-*-<suffix>` and `<domain ARN>/graphrag-*-<suffix>/*`
  (aliases are `<index prefix>-<suffix>`; pick suffixes that never end in
  `-<another suffix>`, or set distinct index prefixes per tenant). For
  document- or field-level rules, enable fine-grained access control on the
  domain and map each tenant role to an OpenSearch role whose index
  permissions cover only `*-<suffix>*`.
- Neptune: IAM authorizes a whole cluster, with no per-label scope, so a
  tenant whose graph must be unreadable by others needs its own Neptune
  cluster (one deployment, or one `env_name`, per tenant).
- Bedrock and the guardrail are shared and hold no tenant data.

## Observability

Tracing is off by default (`observability.enabled: false`). To send traces to
a local Jaeger, which accepts OTLP on port 4318:

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

Run a query (`run-rag --config-path config.yaml --query "..."`) and open
<http://localhost:16686>: the service `unified-kg-rag` shows one
`kg_rag.query` trace per query and one `kg_rag.ingestion.run` trace per
ingestion run. For an OpenTelemetry Collector or a vendor endpoint, point
`OTEL_EXPORTER_OTLP_ENDPOINT` at it and pass any credential in
`OTEL_EXPORTER_OTLP_HEADERS` from Secrets Manager or task secrets, never in the
config file. Set `OTEL_EXPORTER_OTLP_PROTOCOL=grpc` for a gRPC endpoint (port
4317).

To check it works:

- The start-up log has `Observability enabled (backend: 'otlp', ...)`.
- Log records written inside a span carry `trace_id` and `span_id`; search the
  trace backend for that id.
- A query trace has `kg_rag.retrieve`, `kg_rag.fusion`, `kg_rag.context_build`
  and `chat <model>` children; `gen_ai.usage.input_tokens` on the model spans
  is the token usage of the run.
- `kg_rag.rerank.status` on the `kg_rag.rerank` span is the outcome of that
  one Rerank call: `ok`, `error` when the call failed and the query fell back
  to the fused order, or `empty` when the model returned nothing usable.
- `kg_rag.rerank_status` is what reranking did, set even when no Rerank call
  was made: on a `kg_rag.fusion` span for that fusion step, and on the
  `kg_rag.strategy` span for the whole query (the worst step, the same value
  as `metadata.rerank.status`). Values: `applied`, `disabled`, `unavailable`
  (the rerank model could not be built), `skipped` (nothing to rerank) or
  `failed`.

### Langfuse

A local Langfuse runs from its Docker Compose file (it pulls Langfuse v4; v3
works the same way):

```bash
git clone --depth 1 https://github.com/langfuse/langfuse.git
cd langfuse && docker compose up -d     # UI on http://localhost:3000
```

Create a project in the UI and an API key pair for it, then:

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

Each query is a `kg_rag.query` trace, grouped under its conversation in
Sessions when `RAGInput.conversation_id` is set (and under `RAGInput.user_id`
in Users); model calls show as generations with their token usage. Langfuse v4
runs in `events_only` mode by default, where the public `/api/public/traces`,
`/api/public/observations` and `/api/public/sessions` endpoints are disabled:
read spans through `/api/public/v2/observations` or the UI. For Langfuse Cloud set `LANGFUSE_HOST` to
the region's URL (for example `https://cloud.langfuse.com`). Langfuse accepts
OTLP over HTTP only. If you set `OTEL_EXPORTER_OTLP_ENDPOINT` instead of
`LANGFUSE_HOST`, give it `<host>/api/public/otel` (the exporter appends
`/v1/traces`; the host root returns 404). If you set
`OTEL_EXPORTER_OTLP_HEADERS` yourself instead of the two keys, include
`x-langfuse-ingestion-version=4`: Langfuse Cloud can otherwise take minutes to
show traces (a self-hosted v4 server writes them at once either way).

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

With nothing else set, spans go to `http://localhost:6006/v1/traces` and land
in the project named by `service_name` (`PHOENIX_PROJECT_NAME` wins). For a
remote or authenticated Phoenix set `PHOENIX_COLLECTOR_ENDPOINT` (the server
root, e.g. `https://phoenix.example.com`) and `PHOENIX_API_KEY`. Spans carry
OpenInference kinds (`LLM`, `EMBEDDING`, `RETRIEVER`, `RERANKER`, `CHAIN`), so
model calls show their token counts and the Sessions view groups a
conversation's queries.

### CloudWatch through the ADOT collector sidecar

CloudWatch accepts OTLP traces at `https://xray.<region>.amazonaws.com/v1/traces`
signed with SigV4, so no key is involved. The app's OTLP exporter does not
sign requests, so a collector next to it signs and forwards them. On the CDK
stacks, `-c enable_cloudwatch_tracing=true` adds an AWS Distro for
OpenTelemetry (ADOT) collector sidecar to the task and points the app at it
(`backend: otlp`, `OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318`); the
collector signs with the task role. What the option changes, including IAM and
the private-mode `xray` endpoint, is in
[iac/README.md](../iac/README.md#cloudwatch-tracing-optional).

1. **Turn on Transaction Search** (once per account and region). The traces
   endpoint rejects spans until it is on. It is an account-wide setting: it
   moves all X-Ray span ingestion in the region, not only this stack's, to
   CloudWatch Logs collection (the `aws/spans` log group, billed as CloudWatch
   Logs ingestion) and indexes 1% of spans as trace summaries for free. Check
   it first; another workload may already rely on the current destination:

   ```bash
   aws xray get-trace-segment-destination --region "$REGION"
   ```

   `"Destination": "CloudWatchLogs", "Status": "ACTIVE"` means it is on.
   Otherwise turn it on in the CloudWatch console (**Application Signals,
   Transaction Search, Enable Transaction Search**), which also sets up the
   log-group resource policy X-Ray needs, or follow the API steps of the
   [Transaction Search guide](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Enable-TransactionSearch.html)
   (a CloudWatch Logs resource policy for `xray.amazonaws.com`, then
   `aws xray update-trace-segment-destination --destination CloudWatchLogs`).
   Spans can take 10 minutes to become searchable after it is turned on.
   Turning it on creates the `aws/spans` log group (30-day retention) and the
   `/aws/application-signals/data` log group, and sets the `Default` indexing
   rule to 1%. To turn it off, run
   `aws xray update-trace-segment-destination --destination XRay`; that
   removes neither the log groups nor the resource policy, so if nothing else
   uses them, delete the policy (`aws logs delete-resource-policy
   --policy-name <name>`) and the two log groups yourself.
2. **Build the image with the `otel` extra**:
   `docker build --build-arg UV_EXTRAS="--extra otel" -f docker/Dockerfile ...`.
   Without it the run fails at start-up with
   `observability.enabled is true but OpenTelemetry is not installed`.
3. **Deploy** with `cdk deploy --all -c enable_cloudwatch_tracing=true` (add
   the flag to every later deploy, or set it in `cdk.json`, or the next deploy
   removes the sidecar).
4. **Pull the collector image once** from a workstation before the first run
   (step 4 of [CloudWatch tracing](../iac/README.md#cloudwatch-tracing-optional)).
   The image comes through an ECR pull-through cache whose first pull may need
   an internet route when ECR is reached through an interface endpoint, as in
   `private` mode, and ECS fails a task when it cannot pull any of its images.

**Where to look.** Run an ingestion or a query, then open the CloudWatch
console, **Application Signals, Transaction Search**, and filter on the
service `unified-kg-rag`. Each ingestion run is one `kg_rag.ingestion.run`
trace and each `run-rag` query one `kg_rag.query` trace, with the children
listed under [Observability](#observability). A `run-eval` run is one
`kg_rag.batch` trace: its queries are `kg_rag.query` children of that root, so
the list shows the batch, not each query; each evaluator adds one
`kg_rag.evaluate` trace holding its judge model calls. The spans are also in the `aws/spans` log
group, for Logs Insights (if a field comes back empty, run
`fields @message | limit 5` first to see the exact field paths):

```text
fields @timestamp, name, durationNano, attributes.gen_ai.usage.input_tokens, traceId
| filter resource.attributes.service.name = "unified-kg-rag"
| sort @timestamp desc
| limit 50
```

If no trace arrives, read the `otel-collector/` log streams of the task log
group (`/<prefix>/tasks`): a `403` means a missing X-Ray permission or the
wrong SigV4 service, a rejection naming the destination means Transaction
Search is off, and a connection timeout in `private` mode means the `xray`
interface endpoint is missing (a reused VPC must provide it). A collector
that crashes or cannot export never fails the run, since its container is not
essential; a collector image ECS cannot pull does (`CannotPullContainerError`
in the stopped task's reason), so pull it once as in step 4.

**Cost.** Transaction Search bills spans as CloudWatch Logs ingestion and
storage in `aws/spans`, and charges for indexed spans above the free 1%. An
ingestion run emits a span per stage, batch and model call, so a large corpus
produces many spans: lower `observability.sample_rate` in the image config to
sample root traces, and keep the indexing percentage at 1% unless you search
trace summaries. The `xray` interface endpoint (private mode) bills per
AZ-hour like the others.

**Without the CDK stacks**, run the same collector configuration yourself; the
one the stack renders (`compute_stack.py`, after the CloudWatch OTLP guide) is,
for `us-east-1`. The memory limit fits a 512 MiB container, and the exporter
has no queue and no retry so a shutdown flush ends inside ECS's 30 s stop
timeout:

```yaml
extensions:
  health_check:
    endpoint: 127.0.0.1:13133
  sigv4auth/traces:
    region: us-east-1
    service: xray
receivers:
  otlp:
    protocols:
      http:
        endpoint: 127.0.0.1:4318
processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 400
    spike_limit_mib: 100
  batch:
    send_batch_size: 512
    send_batch_max_size: 1000
    timeout: 5s
exporters:
  otlp_http/traces:
    traces_endpoint: https://xray.us-east-1.amazonaws.com/v1/traces
    compression: gzip
    auth:
      authenticator: sigv4auth/traces
    timeout: 10s
    sending_queue:
      enabled: false
    retry_on_failure:
      enabled: false
service:
  extensions: [health_check, sigv4auth/traces]
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlp_http/traces]
  telemetry:
    metrics:
      level: none
```

The collector's role needs `xray:PutTraceSegments`, the one action the OTLP
traces endpoint authorizes (`xray:PutSpans` alone gets a 403). The
`AWSXrayWriteOnlyAccess` managed policy the CloudWatch guide names includes
it.

**Account-level backstop: Bedrock model invocation logging.** It records
every `InvokeModel` / `Converse` request and response in a region, embeddings
included, to CloudWatch Logs or S3, whatever this package's tracing does:
use it as the audit record and traces for the per-query breakdown. It does not
cover the Rerank call (`bedrock-agent-runtime`), and it holds full prompt and
response text, so protect its destination like captured content (restricted
readers, the CMK, a retention period). It is one setting per account and
region, shared by every workload there, so the CDK stacks do not manage it:
turning it on or off from a stack would change logging for everything else in
the account. Set it in the Bedrock runtime region (`bedrock_region`):

```bash
# Check first: an existing configuration is replaced by the put below.
aws bedrock get-model-invocation-logging-configuration --region "$UKR__AWS__BEDROCK__REGION_NAME"

aws bedrock put-model-invocation-logging-configuration --region "$UKR__AWS__BEDROCK__REGION_NAME" \
  --logging-config '{
    "cloudWatchConfig": {
      "logGroupName": "/aws/bedrock/model-invocation-logs",
      "roleArn": "arn:aws:iam::<account>:role/<bedrock-logging-role>"
    },
    "textDataDeliveryEnabled": true,
    "embeddingDataDeliveryEnabled": false
  }'
```

The log group must exist, and the role must trust `bedrock.amazonaws.com` and
allow `logs:CreateLogStream` and `logs:PutLogEvents` on it; the console
(**Amazon Bedrock, Settings, Model invocation logging**) can create both. See
[Monitor model invocation using CloudWatch Logs and Amazon S3](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html).
`aws bedrock delete-model-invocation-logging-configuration` turns it off.
In the CloudWatch console, **GenAI Observability, Model Invocations** lists
these logs next to Bedrock's own invocation, token and latency metrics.

### Export limits, private VPCs and privacy

Spans are exported from a background thread, so an unreachable collector does
not slow runs down: each export request is bounded by
`observability.export_timeout_seconds` (10 s; `OTEL_EXPORTER_OTLP_TIMEOUT`
wins when set), a full queue drops spans instead of blocking, and the exporter
logs its failures (at `ERROR`; `logging.library_levels` hides its per-retry
warnings). The flush at exit sends what is still queued, one request per batch
of up to 512 spans, so against an unreachable collector it takes about one
timeout per remaining batch, and at most about 30 s (the batch processor's
shutdown wait) before the process exits and drops the rest.
In a private VPC the endpoint must be reachable from the tasks: a collector
sidecar on `localhost`, a Langfuse or Phoenix hosted inside the VPC, or the
X-Ray interface VPC endpoint for the collector's CloudWatch export. Hosted
Langfuse or Phoenix needs internet egress, which the private-mode stacks do
not have. The data-plane image includes the extra only when built with
`--build-arg UV_EXTRAS="--extra otel"`.

Credentials (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`, `PHOENIX_API_KEY`,
`OTEL_EXPORTER_OTLP_HEADERS`) are read from the environment only: inject them
as ECS task secrets from Secrets Manager. The start-up line
`Observability exporter for backend '...'` shows the endpoint without
credentials and the header names, never their values.

Privacy defaults: spans carry counts, ids, model names and token usage only.
`capture_content: true` adds prompt and completion text to model spans, and
the query and answer to the `kg_rag.query` span (after `redact_patterns`); enable it only where the trace backend may hold corpus
data. With `otlp`, `langfuse` or `phoenix`, `redact_patterns` also applies at
export to every string attribute, event attribute (an exception message can
quote a query) and status message, `session.id` and `user.id` included; with
`none` the host's pipeline exports spans as recorded, so redact in its
collector. `RAGInput.user_id` is exported as given: pass an opaque id. With
LangSmith tracing also on (`LANGSMITH_TRACING=true`), LangChain uploads
content to LangSmith independently of this setting.

## Common errors

Messages are abridged; the log line names the key, file or stage involved. A store, model or credential error during a query fails the query instead of answering from no context ("could not find"), since every later query would fail the same way.

### Configuration and start-up

| Message (abridged) | Meaning | Action |
|---|---|---|
| `Missing endpoint configuration for ...` | An endpoint the run needs is unset | Set the named key or environment variable |
| `Configuration validation error: ...` | A config value has the wrong type or an unsupported value | Fix the named key |
| `ConfigurationError: Unknown config keys: '<path>' ...` (exit code 2) | A typo, or a key removed or renamed in this release | Correct or delete the key ([CHANGELOG](../CHANGELOG.md) names the replacement). `--allow-unknown-keys` loads the file anyway, logging `Unknown config key '<path>' is ignored` (WARNING) per key |
| `--source-directory is required for pipeline execution` (exit 2, like the usage errors below) | No `--source-directory` and no `$UKR_SOURCE_DIRECTORY` | Pass it; `--verify-metadata` needs `--pipeline-id` instead |
| `--s3-bucket-name must be specified for --s3-sync` | `--s3-sync` without a bucket | Pass `--s3-bucket-name` |
| `--pipeline-id is required for --resume-from-stage` | Resuming needs the earlier run | Pass that run's `--pipeline-id` |
| `--enabled-stages: invalid stage names ...` | `--enabled-stages` names an unknown stage | Use a name from the list in the message |
| `argument --resume-from-stage: Unknown pipeline stage ...` (exit 2) | `--resume-from-stage` names an unknown stage | Use a name from the list in the message (any case) |
| `... needs confirmation, but stdin is not a terminal` | A re-run found completed stages whose output is no longer cached under the current inputs, without a terminal to ask on | Pass `--yes` to go on (the resume recomputes those stages), or use a new `--pipeline-id` |
| `Stage '<stage>', upstream of this run's enabled stages, has not completed` | An `--enabled-stages` window resumed while an earlier stage failed or lost its cached output | Run the earlier stages first |
| `observability.enabled is true but OpenTelemetry is not installed` | Tracing is on but the `otel` extra is missing | `uv sync --extra otel`, or set `observability.enabled: false` |
| `Skipping N '.md' file(s) ...` / `No supported source files found in '<dir>'` | The `unstructured` extra is not installed | Install it (Python 3.11+) or convert the files |

### Credentials and models

| Message (abridged) | Meaning | Action |
|---|---|---|
| `No valid AWS credentials for Amazon Bedrock in region ...` | Credentials are missing or expired | Refresh credentials; check `AWS_PROFILE` / `aws.profile_name` |
| `Model '<id>' is only available through a cross-region inference profile, but none resolved ...` | The model has no on-demand throughput and no profile was found | Keep `aws.bedrock.enable_global_profile: true`, grant `bedrock:ListInferenceProfiles`, or use another region |
| `Reranking failed: ...` (ERROR, per query; `metadata.rerank.status: failed`) | The Rerank call was denied or the model is unavailable in the region | Grant `bedrock:Rerank` on `Resource: "*"`, move Bedrock to a region with the rerank model, or set `search.reranking.enabled: false` |
| `RerankModelError: Reranking failed: ...` / `... returned no usable results` / `Rerank model could not be built: ...` | `search.reranking.required: true`, and the rerank call failed, returned nothing usable, or the model could not be built | Fix the cause as for `Reranking failed`, or set `required: false` to answer from the fused order (`metadata.rerank.status` `failed` / `unavailable`) |
| `Reranking initialization failed; queries keep the fused order ...` (ERROR, when a strategy is first built) | Reranking is enabled but the rerank model could not be built; every query reports `metadata.rerank.status: unavailable` | Check `models.rerank` and the rerank provider (`providers.rerank`), or set `search.reranking.enabled: false` |
| `ClientError: ... (ValidationException) ... InvokeModel ...` / `(ResourceNotFoundException)` / `NoCredentialsError` during a query | The query embedding call cannot succeed: an invalid model id, no on-demand throughput in the region, no model access, or no credentials | Check `models.embedding.model_id`, `aws.bedrock.region_name`, the account's model access and the credentials |
| `EmbeddingModelError: Query embedding failed: ...` | The query embedding call failed after the client's retries (a crashed Ollama runner, a dead vLLM engine, a persistent Bedrock `InternalServerException`); every vector search of the query needs it | Check the embedding server or model (`providers.embedding`, `models.embedding.model_id`) and its logs, then retry |
| Bedrock calls hang in a private VPC | Bedrock is called in a region the VPC endpoint does not serve | Keep `aws.bedrock.region_name` unset or equal to the VPC's region |

### Stores: connection, permissions, requests

| Message (abridged) | Meaning | Action |
|---|---|---|
| `StoreConnectionError: ... Cannot connect to host ...` / `Neptune is not reachable ...` / `... status=429 ... (still failing after the client's retries)` | The store refused the connection or could not be resolved, Neptune closed it (a restart or failover), or OpenSearch still answered 429/502/503/504 after `search.store_max_attempts` attempts. A lost Neptune connection is reopened by the next query or retry | Check the endpoint and port, that the store runs, and the network path (security groups, VPC endpoints) |
| `OpenSearch did not answer a request on <index> within N s in M attempt(s)` / `Neptune did not answer within N s` | The store accepted the connection but did not answer within `search.store_timeout_seconds`. During ingestion the Neptune bound is a fixed 180 s and the item is retried | Check the store's health and load (CPU, queues, a failover); raise `search.store_timeout_seconds` for a cluster that is slow by design |
| `OpenSearch denied the search on <index>: ... status=403 ...` (or `401`) | The query principal may not search the index: under fine-grained access control, a role without `indices:data/read/search`; or invalid credentials | Grant the role `indices:data/read/search` on the indices (and `indices:admin/mappings/get` for the embedding check), or fix the credentials |
| `Neptune refused the connection (HTTP 403)` | IAM authentication denied the websocket: the policy lacks `neptune-db:connect` on the cluster, or the request was signed for another region | Grant `neptune-db:connect` (and the `neptune-db:*` data actions the run needs); set `aws.region_name` to the cluster's region |
| OpenSearch authentication fails with `use_iam: false` | No basic-auth credentials | Set `UKR__AWS__OPENSEARCH__USERNAME` / `UKR__AWS__OPENSEARCH__PASSWORD` (for example in `.env`) |
| `OpenSearch rejected the search on <index>: ...` | HTTP 400: a filter value of another type than the field's (`query_shard_exception`). HTTP 500 `too_many_clauses` / `too_many_nested_clauses`: `indices.query.bool.max_clause_count` is below the default 1,024 | Fix the filter value, or keep `indices.query.bool.max_clause_count` at 1,024 or more |
| `Cannot create the OpenSearch index '<index>': the cluster has no analyzer '<name>'` (raw: `mapper_parsing_exception`) | `indexing.opensearch.language_analyzers` names an analyzer whose plugin the cluster lacks, typically `ko: nori` on an Amazon OpenSearch Service domain without the Nori package | Map the language to an analyzer the cluster has (`ko: cjk` is built in; the CDK stack's default, `-c opensearch_korean_analyzer`), or associate the package; then re-run |

### Ingestion runs and the doc-status registry

| Message (abridged) | Meaning | Action |
|---|---|---|
| `Pipeline failed at stage(s): ...` | A stage failed; the CLI exits 1 | Read the stage's log lines, fix, resume with the same `--pipeline-id`. `--verify-metadata` checks the run's metadata, `--force-rebuild` starts clean |
| `CacheSyncError` | The S3 stage-cache download or upload failed | Check bucket permissions and `aws.s3.encryption`, then resume with the same `--pipeline-id` |
| `DocStatusRegistryError` | The doc-status registry table is missing, not accessible, or throttled | Create the table or fix `aws.dynamodb.table_name`, grant the DynamoDB permissions named in the message, then resume with the same `--pipeline-id` |
| `Incremental indexing is enabled but no document delta was computed` | The indexing stage ran without a registry diff (the loading stage failed under `continue_on_error`) | Re-run from `document_loading` once the registry is reachable |
| `NamespaceLeaseError: Index namespace '...' is being written by ...` | Another run holds the namespace's write lease | Wait for it to finish; a dead run's lease expires within `indexing.lease_seconds` (see [Concurrent runs](#concurrent-runs)) |
| `NamespaceLeaseError: This run lost the write lease ...` | The run's lease renewals failed until it expired, or another run took it | Re-run once the other run ended; check the registry's reachability |
| `ConfigurationError: Doc-status registry: DynamoDB table '...' has the key schema ... of an earlier release` | The table was created by a release that keyed records by `doc_id` alone | Point `aws.dynamodb.table_name` (`UKR__AWS__DYNAMODB__TABLE_NAME`) at a new table or deploy the stack's new one, and run once with `indexing.reset: true` (see [Incremental runs](#incremental-runs)) |
| `ConfigurationError: The OpenSearch index '...' records no embedding settings ...` | An earlier release built the index | Run once with `indexing.reset: true` |
| `The doc-status registry lists indexed documents of this namespace whose entities the graph store does not hold` | The graph lost what the registry records (a restarted Gremlin Server, `docker compose ... down -v`, a restored or replaced graph) | Re-run with `indexing.reset: true`, or delete the namespace's registry records (local mode: the SQLite file) |
| `The doc-status registry lists indexed documents of this namespace whose text units the vector store does not hold` | The vector store lost what the registry records (deleted indices; local mode: `vector.sqlite3` moved or deleted, or `providers.vector.sqlite_path` changed) | As above; in local mode delete the three store files together or restore the vector file |

### Queries

| Message (abridged) | Meaning | Action |
|---|---|---|
| `No indices found for suffix '<suffix>' ...` | Nothing was ingested under that index suffix | Run ingestion, or query with the suffix the corpus was ingested under |
| `InvalidFilterError` | No store the strategy reads declares a filter key | Use a key from the list in the message |
| LightRAG `mix` / `hybrid` returns nothing | No relationship index, or keyword extraction produced no keywords | Check that ingestion built the relationships index; very short queries fall back to the raw query only under `raw_query_fallback_max_len` |

### Local mode

| Message (abridged) | Meaning | Action |
|---|---|---|
| `... stayed locked ...` (`database is locked`) from an embedded store or the SQLite registry | Another process (a second ingestion, say) held the SQLite file's write lock longer than `search.store_timeout_seconds`; SQLite allows one writer at a time | Run one ingestion per store file at a time, or raise `search.store_timeout_seconds`. Keep the files on a local disk, not a network filesystem |
| `... processed N inputs but produced 0 outputs`, with `Connection refused` in the log (local mode) | Ollama or the OpenAI-compatible server is not reachable at `providers.llm.base_url` / `providers.embedding.base_url` | Start it (`ollama serve`, or `docker compose -f docker/compose.local.yaml --profile models up -d --wait`), check `base_url` (or `--model-base-url`), and pull the models |
| A query fails with `AuthenticationError` / `NotFoundError` (401 / 404), `APIConnectionError` / `APITimeoutError`, or `ResponseError: model ... not found` | The OpenAI-compatible or Ollama server refused the key, does not serve the model, or could not be reached after the client's retries | Check `base_url` and `api_key` under `providers.llm` / `providers.embedding`, the model names (`models.default`, `models.embedding.model_id`), that the server runs, and pull the models |
