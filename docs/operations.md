# Operator Runbook

> 🇰🇷 한국어판: [docs/operations.ko.md](./operations.ko.md)

Tasks an operator runs against an existing deployment: re-ingesting, running
incremental updates, handling failed documents, managing the caches, and
reading the errors the CLIs report. Configuration keys are described in the
[User Guide §2](./user-guide.md#2-configuration); the deployed stack and its
commands are in [`iac/README.md`](../iac/README.md#after-deploy).

## Before a run

1. **Region and credentials.** `AWS_REGION` in the environment overrides
   `aws.region_name` (and Bedrock's region when `aws.bedrock.region_name` is
   unset). Check with `env | grep -E '^(AWS_REGION|BEDROCK_REGION)='` and
   `aws sts get-caller-identity`.
2. **Models.** The models in
   [Required models](./user-guide.md#required-models) are enabled in the
   Bedrock region.
3. **Endpoints.** `run-ingestion`, `run-rag` and `run-eval` check at start-up,
   before any model call, that the store endpoints they need are set (see
   [Start-up endpoint check](#start-up-endpoint-check)).
4. **No other ingestion is running** against the same stores. Concurrent runs
   are not supported (see [Concurrent runs](#concurrent-runs)). On the CDK
   stack:
   `aws stepfunctions list-executions --state-machine-arn <arn> --status-filter RUNNING`.

## Concurrent runs

Concurrent ingestion runs are not supported. Do not start a run while another
one is running against:

- **the same index suffix** (the same OpenSearch aliases and Neptune labels).
  A full reindex writes a new timestamped index, swaps the alias onto it and
  deletes the alias's older indices, so one run can delete the index the other
  is writing or leave the alias on the other run's index. A reset clears the
  suffix's stores under the other run.
- **the same doc-status table namespace.** An incremental run reads the
  registry once to plan what to remove and which records to keep, so records
  another run writes in the meantime can be pruned and their artifacts removed
  or kept against the plan.

Runs on different index suffixes that use separate doc-status tables
(`aws.dynamodb.table_name`) do not share either and can run side by side.
Otherwise, wait for one run to finish before starting the next.

A doc-status table namespace is the index suffix (`index_value`, `default`
when unset) plus `indexing.additional_suffix`, plus a tag of the
`indexing.opensearch.*_index_prefix` and `indexing.neptune.*_label_prefix`
values when any of them is not the default. Two configurations that differ
only in those prefixes write different stores, so they keep separate records
even in one table (the CDK stack injects one fixed table name): artifact ids
are derived from content, and records of the other store set would otherwise
keep a changed document's old text units and entities in this run's stores.
A default configuration keeps the namespace it always had. A configuration
with custom prefixes whose records were written before the prefixes joined
the namespace sees its corpus as new: its first run re-indexes it (the stage
cache is re-keyed from parsing on), and the old records stay in the table
under the untagged namespace. Delete those records once no configuration with
the default prefixes and the same suffixes uses the table.

On the CDK stack, each `start-execution` starts one execution that runs the
four phases in sequence. The stack defines no schedule or event trigger, so
executions start only when someone (or an automation you add) calls
`start-execution`, but the state machine does not stop a second execution from
starting while one is running. Check for a `RUNNING` execution first (the
command above) and do not start overlapping executions; a schedule you add
should not fire while a previous execution can still be running (up to the
state machine's timeout, `sfn_timeout_hours`, 6 hours by default).

## Re-ingest

Three operations look alike but act on different state:

| Goal | How | Stage cache | Stores and registry |
|---|---|---|---|
| Finish an interrupted run | Re-run with the same `--pipeline-id` (on the CDK stack, start a new execution with the same `pipeline_id`); it resumes at the first failed or incomplete stage, and each Step Functions phase skips the stages of its window that completed. `--resume-from-stage <stage>` picks the stage. | Reused | Written by the stages that run |
| Recompute every stage | `--force-rebuild` | Ignored and rewritten | Written as usual (incremental when enabled) |
| Rebuild the stores from scratch | `indexing.reset: true` for one run | Unchanged | Cleared for the run's index namespace (other namespaces in the table keep their records), then the whole corpus is indexed and every document recorded again |

Use `indexing.reset: true` (and `--force-rebuild` if stage outputs must change
too) after changing anything that is baked into the stored data:

- the embedding model or `indexing.opensearch.embedding_dimension`;
- `processing.translation.source_language` or the language analyzers (they
  change the text-unit mapping);
- to refresh descriptions and communities that incremental runs keep (see
  [Incremental runs](#incremental-runs));
- the extraction settings (prompts, `entity_types`, extraction or chunking
  models, chunk sizes...) when the registry holds records written before
  records carried an extraction fingerprint, or after editing a built-in
  prompt template (code). Otherwise an incremental run re-extracts the
  unchanged documents whose stored fingerprint differs from the current
  settings on its own and logs how many (see
  [User Guide §5](./user-guide.md#workflows)); records without a fingerprint
  are compared by content only, so they keep their old extraction until a
  reset.

Set `indexing.reset` back to `false` afterwards; otherwise every run rebuilds.

`--verify-metadata` (needs `--pipeline-id`) checks a run's stage metadata and
exits non-zero when it is missing or corrupt.

## Incremental runs

With `aws.dynamodb.enabled: true` (the container image sets it), every
`run-ingestion` diffs the corpus against the doc-status registry by content
hash and indexes only new and changed documents. The diff is scoped to the
run's [index suffix](./user-guide.md#index-suffix) and its source scope
(`processing.document_parsing.source_scope`, by default the resolved source
directory; the container entrypoint sets it to the `s3://` source URI), and
both are part of each document's registry key, so two corpora on one suffix
never share a record even for files with the same relative path. An `s3://`
URI with and without a trailing slash (`s3://bucket/corpus/`,
`s3://bucket/corpus`) is one scope: a run whose spelling the registry does
not hold, but the other one, uses the stored spelling and logs a WARNING. A
file missing from the source is treated as deleted, and its exclusive
artifacts are removed. Therefore:

- keep the same source directory (or a fixed `source_scope`) for a corpus.
  Moving a corpus with the default scope changes the scope; the old scope's
  records are never adopted or deleted on their own, so they keep the old
  content indexed (files removed during the move included). Either set a
  fixed `source_scope` before moving it, or run once from the new location
  with `--retire-source-scope <old source directory>`, which removes the old
  scope's exclusive content and records in that run (see
  [User Guide §5](./user-guide.md#deletion-scope)). A run with new documents
  logs a WARNING naming the other local source directories of its suffix;
- never let corpus files expire or disappear unintentionally: the next run
  removes their graph and vector content;
- index each tenant or corpus version under its own index suffix, with its own
  config file.

Upgrading a registry written by a pre-release build: see
[CHANGELOG: Upgrading from pre-release builds](../CHANGELOG.md#upgrading-from-pre-release-builds).

What incremental runs do not refresh: a shared entity or relationship keeps
the description text a changed or deleted document contributed, and delta
runs append communities rather than re-clustering. A periodic full rebuild
(`indexing.reset: true`) refreshes both. Details: [User Guide §5](./user-guide.md#5-incremental-indexing).

If the registry cannot be read (missing table, denied access, throttling that
outlasts the client's retries), the `document_loading` stage fails with
`DocStatusRegistryError`, naming the table and the cause. The run does not fall
back to indexing every document: that path replaces the suffix's index content
with this run's documents only. Fix the registry and re-run with the same
`--pipeline-id`, or set `aws.dynamodb.enabled: false` to run without
incremental indexing.

### Interrupted runs

An incremental run that stops inside the indexing stage (the task is killed,
a store or the registry becomes unreachable, the failure gate trips) needs no
manual repair: re-run. Before it removes or writes anything, the indexing
stage records every new and changed document `PENDING`, listing both the
artifacts it had and the ones the run is about to write; the records become
`PROCESSED` (or `FAILED`) only after the writes. The next run therefore:

- re-extracts each `PENDING` document still in the corpus, even if its file
  was reverted to the version indexed before;
- removes everything the interrupted run may have written for a `PENDING`
  document that is no longer in the corpus.

An interruption is not counted as a failure, but it resets the document's
consecutive-failure count for `indexing.max_document_failures`: the `PENDING`
record has `failure_count` 0, so a document that had failed before starts
counting again from its next failure. `PENDING` records (`status` `pending`,
`content_hash` `pending`, `error_info` naming the write-ahead record) are
expected after a failed run. Do not delete them by hand: their lineage is what
lets the next run find the interrupted run's writes. When a large document's
old and new artifact ids do not fit one DynamoDB item together, the ids the
record lacks are kept in overflow items of the same table, keyed
`<doc_id>#pending#<n>` with `record_kind` `lineage_overflow`; leave them too.
The run deletes them when it commits or removes the document; overflow a
run left next to a committed record (it stopped between the commit and the
delete) is deleted by the next incremental run of the same namespace, even
when the document is unchanged. The stage writes the registry in batches, so a role with a
custom policy needs `dynamodb:BatchWriteItem` as well as `BatchGetItem`
(the CDK stack's `grant_read_write_data` covers both).

## Failed documents

A document is recorded `FAILED` when translation, graph extraction, gleaning
or claim extraction failed on any of its text units, when one of the stages
that build its artifacts (translation through community detection) crashed in
a `--continue-on-error` run, or when the write of one
of its artifacts (text unit, entity, relationship, claim, community, report)
to OpenSearch or Neptune failed. The next run treats it as changed: it removes
what the failed run wrote and processes the document again. When parsing,
loading or chunking crashed in a
`--continue-on-error` run, the indexing stage does not run at all, so the
stores and the registry keep their previous state. The same holds when any
stage fails without `--continue-on-error`: the run stops before indexing and
records nothing, because a stage failure does not say whether one document or
the backend (throttling, an outage, expired credentials) is at fault, so the
next run processes the same new and changed documents again.

- After `indexing.max_document_failures` (default `3`) consecutive failures
  with unchanged content, the document is no longer retried. It stays
  `FAILED`, keeps what it indexed, and each run logs a WARNING naming the file.
- To retry it, fix the cause (often an oversized or malformed file), edit the
  file so its content hash changes, or raise `indexing.max_document_failures`.
- A document whose artifact ids exceed one DynamoDB item (400 KB, roughly
  10,000 ids) fails the indexing stage, before anything is written for the
  delta, with an error naming the file and its id count. Split the file.

Write failures are gated separately. OpenSearch bulk items rejected with a
retryable status (429, 502, 503, 504), and the items of a whole bulk request
still answered with one after the client's own retries, are resent up to four
times with backoff first. Every Neptune request of a run (writes, drops and
the reads of removals and merges) is retried up to
`indexing.neptune.max_attempts` times (default `4`) over a new connection,
and each one is safe to repeat: a request whose response was lost is not
applied twice. If more than `indexing.max_failure_rate` (default `0.2`) of one
artifact type's writes still fail, the indexing stage fails and the documents
are not recorded, so the next run retries them. Below that rate the run
succeeds and only the documents that own a failed artifact are recorded
`FAILED`; a failure the backend reports without an item id marks every
document of the run `FAILED`. On the CDK stack, a failed phase publishes a
message with the execution id and the error to the SNS topic, and four alarms
report problems to the same topic. `PipelineFailures` also fires for a failed
phase, so such a failure arrives twice, the second time without context; the
alarm stays because it also covers failures the message cannot report (an
error the phase's handler cannot catch, such as a start input without
`pipeline_id`, or a publish that fails after its retries):

| Alarm | Fires when |
|---|---|
| `PipelineFailures` | A Step Functions execution failed |
| `PipelineTimedOutOrAborted` | An execution hit the state machine timeout or was stopped |
| `IndexingFailures` | Indexing reported failed items, even in a succeeded execution |
| `ExtractionFailures` | Extraction, gleaning or claim extraction failed on some units |

The alarms have a subscriber only when the stack was deployed with
`-c alarm_email=<address>`.

## Caches and prefixes

| Cache | Location | Notes |
|---|---|---|
| Stage checkpoints (local) | `<--cache-directory>/<pipeline_id>/` (default `cache/`) | Reused on resume; `--force-rebuild` ignores it and writes the recomputed outputs |
| Stage checkpoints (S3) | `s3://<bucket>/<--s3-prefix>/<pipeline_id>/` (default prefix `pipeline-runs`) | With `--s3-sync`; how the Step Functions phases hand off. A failed sync fails the run with `CacheSyncError` |
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

## OpenSearch replicas

On the CDK stack, a multi-node domain (`opensearch_count` > 1, the non-dev
default) gets one replica per index, so losing a node or an AZ keeps every
shard available; a single-node dev domain gets none. The compute stack sets
`OPENSEARCH_NUMBER_OF_REPLICAS`, which overrides
`indexing.opensearch.index_settings.number_of_replicas` when an index is
created. Indices created before that keep their replica count, so set it on
them once from inside the VPC:

```bash
curl -X PUT "https://<OpenSearch endpoint>/_all/_settings" \
  --aws-sigv4 "aws:amz:<region>:es" --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
  -H "x-amz-security-token: $AWS_SESSION_TOKEN" \
  -H 'Content-Type: application/json' -d '{"index": {"number_of_replicas": 1}}'
```

## Backup and restore

The three stores are backed up on different schedules, and their contents
refer to each other (the registry lists each document's graph and index
artifacts), so restore them to one common point or not at all:

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
   (`NEPTUNE_ENDPOINT`, `GRAPHRAG_DOC_STATUS_TABLE`). The task role, the
   DynamoDB gateway-endpoint policy and the alarms are scoped to the stack's
   own cluster and table, so grant them on the restored resources too.

**Outside Neptune's window** (or when step 5 is not worth it), there is no
common point: run one full rebuild with `indexing.reset: true` (see
[Re-ingest](#re-ingest)). It clears this index namespace's stores and
registry records and re-indexes the whole corpus. Restore the table first
only if other namespaces share it.

## Start-up endpoint check

Before the first paid model call, the CLIs check that the store endpoints the
run needs are set, and exit with status 1 otherwise:

```text
Error: Missing endpoint configuration for the indexing stage: aws.neptune.endpoint (env NEPTUNE_ENDPOINT), aws.opensearch.endpoint (env OPENSEARCH_ENDPOINT). Set them in the config file or the environment.
```

- `run-ingestion` checks both endpoints when the `indexing` stage is enabled
  (not for `--verify-metadata`).
- `run-rag` and `run-eval` check the endpoints the chosen strategy's retrievers
  need: OpenSearch for every strategy, and Neptune as well for `local`,
  `drift`, `mix` and `hybrid`. `auto` checks every strategy in `search.auto_routable_strategies`.

The check only tests that a value is set. An endpoint that is set but wrong or
unreachable fails at the first connection; the retrievers report such
authentication, configuration and connection errors instead of returning empty
results.

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
- `kg_rag.rerank.status` on the `kg_rag.rerank` span is `error` when the
  Rerank call failed and the query fell back to the fused order, and `empty`
  when the model returned nothing usable.

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
service `unified-kg-rag`. Each run is one `kg_rag.ingestion.run` trace and each
query one `kg_rag.query` trace, with the children listed under
[Observability](#observability). The spans are also in the `aws/spans` log
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
aws bedrock get-model-invocation-logging-configuration --region "$BEDROCK_REGION"

aws bedrock put-model-invocation-logging-configuration --region "$BEDROCK_REGION" \
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
`capture_content: true` adds prompt and completion text to model spans (after
`redact_patterns`); enable it only where the trace backend may hold corpus
data. With `otlp`, `langfuse` or `phoenix`, `redact_patterns` also applies at
export to every string attribute, event attribute (an exception message can
quote a query) and status message, `session.id` and `user.id` included; with
`none` the host's pipeline exports spans as recorded, so redact in its
collector. `RAGInput.user_id` is exported as given: pass an opaque id. With
LangSmith tracing also on (`LANGSMITH_TRACING=true`), LangChain uploads
content to LangSmith independently of this setting.

## Common errors

| Message (abridged) | Meaning | Action |
|---|---|---|
| `Missing endpoint configuration for ...` | An endpoint the run needs is unset | Set the named key or environment variable |
| `Configuration validation error: ...` | A config value has the wrong type or an unsupported value | Fix the named key |
| `Unknown config key '<path>' is ignored` (WARNING) | Typo or removed key; the run continues without it | Correct or delete the key |
| `No valid AWS credentials for Amazon Bedrock in region ...` | Credentials are missing or expired | Refresh credentials; check `AWS_PROFILE` / `aws.profile_name` |
| `Model '<id>' is only available through a cross-region inference profile, but none resolved ...` | The model has no on-demand throughput and no profile was found | Keep `aws.bedrock.enable_global_profile: true`, grant `bedrock:ListInferenceProfiles`, or use another region |
| `Reranking failed: ...` (ERROR, per query; `metadata.rerank.status: failed`) | The Rerank call was denied or the model is unavailable in the region | Grant `bedrock:Rerank` on `Resource: "*"`, move Bedrock to a region with the rerank model, or set `search.reranking.enabled: false` |
| `RerankModelError: Reranking failed: ...` / `... returned no usable results` / `Rerank model could not be built: ...` | `search.reranking.required: true` and the rerank call failed, returned nothing usable, or the rerank model could not be built; the query fails instead of answering from the fused order | Fix the cause as for `Reranking failed` above, or set `required: false` to keep fusion-only answers (reported as `metadata.rerank.status` `failed` / `unavailable`) |
| `Reranking initialization failed; queries keep the fused order ...` (ERROR, when a strategy is first built) | Reranking is enabled but the rerank model could not be built; every query reports `metadata.rerank.status: unavailable` | Check `search.reranking.rerank_model_id` and the rerank provider (`providers.rerank`), or set `search.reranking.enabled: false` |
| `No indices found for suffix '<suffix>' ...` | Nothing was ingested under that index suffix | Run ingestion, or query with the suffix the corpus was ingested under |
| `InvalidFilterError` | No store the strategy reads declares a filter key | Use a key from the list in the message |
| `OpenSearch rejected the search on <index>: ...` | The request is invalid for that index (HTTP 400): a filter value of another type than the field's (`query_shard_exception`), or a setting naming something the cluster lacks, such as `indexing.opensearch.hybrid_search_pipeline_name` (`Pipeline ... is not defined`). `too_many_nested_clauses` / `too_many_clauses` (HTTP 500): the query has more clauses than the cluster allows, typically an `indices.query.bool.max_clause_count` below the default 1,024 | Fix the filter value or the setting; ingestion creates the hybrid search pipeline. For the clause limit, keep `indices.query.bool.max_clause_count` at 1,024 or more |
| `Cannot create the OpenSearch index '<index>': the cluster has no analyzer '<name>'` (raw: `mapper_parsing_exception: analyzer [<name>] has not been configured in mappings`) | `indexing.opensearch.language_analyzers` maps a corpus or target language to an analyzer whose plugin the cluster lacks, typically `ko: nori` on an Amazon OpenSearch Service domain without the optional Nori package | Map the language to an analyzer the cluster has (`ko: cjk` is built in; the CDK deployment does this by default, `-c opensearch_korean_analyzer`), or associate the plugin package with the domain; then re-run |
| `StoreConnectionError: ... Cannot connect to host ...` / `Neptune is not reachable ...` / `... status=429 ... (still failing after the client's retries)` | OpenSearch or Neptune refused the connection or could not be resolved, Neptune closed the connection (a restart or failover), or OpenSearch still answered 429/502/503/504 after `search.store_max_attempts` attempts (overloaded or restarting); the query fails rather than answering from no context. A lost Neptune connection is dropped: the next query, or ingestion's next retry, reconnects | Check the endpoint and port, that the store is running, and the network path (security groups, VPC endpoints) |
| `OpenSearch did not answer a request on <index> within N s in M attempt(s)` / `Neptune did not answer within N s` | The store accepted the connection but did not answer within `search.store_timeout_seconds` (OpenSearch: in `search.store_max_attempts` attempts); the query fails | Check the store's health and load (CPU, queues, a failover in progress); raise `search.store_timeout_seconds` for a cluster that is slow by design |
| `OpenSearch denied the search on <index>: ... status=403 ...` (or `401`) | The query principal may not search the index: under fine-grained access control, a role without `indices:data/read/search`; or invalid credentials | Grant the role `indices:data/read/search` on the indices (and `indices:admin/mappings/get` for the embedding check), or fix the credentials |
| `ClientError: ... (ValidationException) ... InvokeModel ...` / `(ResourceNotFoundException)` / `NoCredentialsError` during a query | The query embedding call cannot succeed: an invalid model id, a model without on-demand throughput in the region, model access not granted, or no credentials; the query fails rather than answering from no context | Check `indexing.opensearch.embedding_model_id`, `aws.bedrock.region_name` and the model access of the account, or the credentials |
| `EmbeddingModelError: Query embedding failed: ...` | The query embedding model call failed after the client's retries (a crashed Ollama runner, a dead vLLM engine, a Bedrock `InternalServerException` that persisted); every vector search of the query needs it, so the query fails instead of answering "could not find" | Check the embedding server or model (`providers.embedding`, `indexing.opensearch.embedding_model_id`) and its logs, then retry |
| `Neptune refused the connection (HTTP 403)` | IAM authentication denied the websocket: the policy lacks `neptune-db:connect` on the cluster, or the request was signed for another region | Grant `neptune-db:connect` (and the `neptune-db:*` data actions the run needs); set `aws.region_name` to the cluster's region |
| `Skipping N '.md' file(s) ...` / `No supported source files found in '<dir>'` | The `unstructured` extra is not installed | Install it (Python 3.11+) or convert the files |
| `DocStatusRegistryError` | The doc-status registry table is missing, not accessible, or throttled | Create the table or fix `aws.dynamodb.table_name`, grant the DynamoDB permissions named in the message, then resume with the same `--pipeline-id` |
| `Incremental indexing is enabled but no document delta was computed` | The indexing stage ran without a registry diff (the loading stage failed under `continue_on_error`) | Re-run from `document_loading` once the registry is reachable |
| `CacheSyncError` | The S3 stage-cache download or upload failed | Check bucket permissions and `aws.s3.encryption`, then resume with the same `--pipeline-id` |
| `Pipeline failed at stage(s): ...` | A stage failed; the CLI exits 1 | Read the stage's log lines, fix, resume with the same `--pipeline-id`. `--verify-metadata` checks the run's metadata, `--force-rebuild` starts clean |
| `--source-directory is required for pipeline execution` | No `--source-directory` and no `$GRAPHRAG_SOURCE_DIRECTORY` | Pass it; `--verify-metadata` needs `--pipeline-id` instead |
| `--s3-bucket-name must be specified for S3 sync` | `--s3-sync` without a bucket | Pass `--s3-bucket-name` |
| `--pipeline-id is required for --resume-from-stage` | Resuming needs the earlier run | Pass that run's `--pipeline-id` |
| `Invalid stage names provided: ...` | `--enabled-stages` names an unknown stage | Use a name from the list the CLI prints |
| `argument --resume-from-stage: Unknown pipeline stage ...` (exit 2) | `--resume-from-stage` names an unknown stage | Use a name from the list in the message (any case) |
| `... needs confirmation, but stdin is not a terminal` | A re-run found completed stages whose output is no longer cached under the current inputs, without a terminal to ask on | Pass `--yes` to go on (the resume recomputes those stages), or use a new `--pipeline-id` |
| `Stage '<stage>', upstream of this run's enabled stages, has not completed` | An `--enabled-stages` window (a Step Functions phase) resumed while an earlier stage failed or lost its cached output | Run the earlier stages first; on the CDK stack, start a new execution with the same `pipeline_id` |
| `The doc-status registry lists indexed documents of this namespace whose entities the graph store does not hold` | The graph lost what the registry records (a restarted Gremlin Server, `docker compose ... down -v`, a restored or replaced graph) | Re-run with `indexing.reset: true`, or delete the namespace's registry records (local mode: the SQLite file) |
| `The doc-status registry lists indexed documents of this namespace whose text units the vector store does not hold` | The vector store lost what the registry records (local mode: `vector.sqlite3` deleted or moved, or `providers.vector.sqlite_path` changed; deleted OpenSearch indices) | Same as above; in local mode delete the three store files together or restore the vector file |
| `... stayed locked ...` (`database is locked`) from the embedded graph or vector store (local mode) | Another process is writing to the same SQLite file (a second ingestion, say) and held its lock longer than `search.store_timeout_seconds`; SQLite allows one writer at a time | Wait for the other run and retry, run one ingestion per store file at a time, or raise `search.store_timeout_seconds`. Keep the files on a local disk: network filesystems are not supported |
| `... processed N inputs but produced 0 outputs`, with `Connection refused` in the log (local mode) | Ollama or the OpenAI-compatible server is not reachable at `providers.llm.base_url` / `providers.embedding.base_url` | Start it (`ollama serve`, or `docker compose -f docker/compose.local.yaml --profile models up -d --wait`), check `base_url` (or `--model-base-url`), and pull the models |
| A query fails with `AuthenticationError` / `NotFoundError` (`Error code: 401` / `404`), `APIConnectionError` / `APITimeoutError`, or `ResponseError: model ... not found` (local mode) | The OpenAI-compatible or Ollama server refused the key or does not serve the model, or could not be reached or did not answer after the client's retries; every later query would fail the same way, so the query fails instead of answering "could not find" | Check `providers.llm` / `providers.embedding` (`base_url`, `api_key`, `model_id`), that the server is running, and pull the models |
| `observability.enabled is true but OpenTelemetry is not installed` | Tracing is on but the `otel` extra is missing | `uv sync --extra otel`, or set `observability.enabled: false` |
| Bedrock calls hang in a private VPC | Bedrock is called in a region the VPC endpoint does not serve | Keep `aws.bedrock.region_name` unset or equal to the VPC's region |
| OpenSearch authentication fails with `use_iam: false` | No basic-auth credentials | Set `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD` (for example in `.env`) |
| LightRAG `mix` / `hybrid` returns nothing | No relationship index, or keyword extraction produced no keywords | Check that ingestion built the relationships index; very short queries fall back to the raw query only under `raw_query_fallback_max_len` |
