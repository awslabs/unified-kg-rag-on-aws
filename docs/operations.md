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
4. **No other ingestion is running** against the same stores. Two runs race on
   the doc-status registry and on the graph and vector writes. On the CDK stack:
   `aws stepfunctions list-executions --state-machine-arn <arn> --status-filter RUNNING`.

## Re-ingest

Three operations look alike but act on different state:

| Goal | How | Stage cache | Stores and registry |
|---|---|---|---|
| Finish an interrupted run | Re-run with the same `--pipeline-id`; it resumes at the first failed or incomplete stage. `--resume-from-stage <stage>` picks the stage. | Reused | Written by the stages that run |
| Recompute every stage | `--force-rebuild` | Ignored and rewritten | Written as usual (incremental when enabled) |
| Rebuild the stores from scratch | `indexing.reset: true` for one run | Unchanged | Cleared, then the whole corpus is indexed and every document recorded again |

Use `indexing.reset: true` (and `--force-rebuild` if stage outputs must change
too) after changing anything that is baked into the stored data:

- the embedding model or `indexing.opensearch.embedding_dimension`;
- `processing.translation.source_language` or the language analyzers (they
  change the text-unit mapping);
- to refresh descriptions and communities that incremental runs keep (see
  [Incremental runs](#incremental-runs)).

Set `indexing.reset` back to `false` afterwards; otherwise every run rebuilds.

`--verify-metadata` and `--repair-metadata` (both need `--pipeline-id`) check
and repair a run's stage metadata; both exit non-zero on failure.

## Incremental runs

With `aws.dynamodb.enabled: true` (the container image sets it), every
`run-ingestion` diffs the corpus against the doc-status registry by content
hash and indexes only new and changed documents. The diff is scoped to the
run's [index suffix](./user-guide.md#index-suffix) and its source scope
(`processing.document_parsing.source_scope`, by default the resolved source
directory; the container entrypoint sets it to the `s3://` source URI). A file
missing from the source is treated as deleted, and its exclusive artifacts are
removed. Therefore:

- keep the same source directory (or a fixed `source_scope`) for a corpus;
  moving it changes the scope;
- never let corpus files expire or disappear unintentionally: the next run
  removes their graph and vector content;
- index each tenant or corpus version under its own index suffix, with its own
  config file.

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

## Failed documents

A document is recorded `FAILED` when translation, graph extraction, gleaning
or claim extraction failed on any of its text units, or when the write of one
of its artifacts (text unit, entity, relationship, claim, community, report)
to OpenSearch or Neptune failed. The next run treats it as changed: it removes
what the failed run wrote and processes the document again.

- After `indexing.max_document_failures` (default `3`) consecutive failures
  with unchanged content, the document is no longer retried. It stays
  `FAILED`, keeps what it indexed, and each run logs a WARNING naming the file.
- To retry it, fix the cause (often an oversized or malformed file), edit the
  file so its content hash changes, or raise `indexing.max_document_failures`.
- A document whose artifact ids exceed one DynamoDB item (400 KB, roughly
  10,000 ids) fails the indexing stage with an error naming the file. Split
  the file.

Write failures are gated separately. OpenSearch bulk items rejected with a
retryable status (429, 502, 503, 504) are resent up to four times with
backoff first. If more than `indexing.max_failure_rate` (default `0.2`) of one
artifact type's writes still fail, the indexing stage fails and the documents
are not recorded, so the next run retries them. Below that rate the run
succeeds and only the documents that own a failed artifact are recorded
`FAILED`; a failure the backend reports without an item id marks every
document of the run `FAILED`. On the CDK stack, three alarms report problems
to the SNS topic:

| Alarm | Fires when |
|---|---|
| `PipelineFailures` | A Step Functions execution failed |
| `IndexingFailures` | Indexing reported failed items, even in a succeeded execution |
| `ExtractionFailures` | Extraction, gleaning or claim extraction failed on some units |

The alarms have a subscriber only when the stack was deployed with
`-c alarm_email=<address>`.

## Caches and prefixes

| Cache | Location | Notes |
|---|---|---|
| Stage checkpoints (local) | `<--cache-directory>/<pipeline_id>/` (default `cache/`) | Reused on resume; `--force-rebuild` ignores it |
| Stage checkpoints (S3) | `s3://<bucket>/<--s3-prefix>/<pipeline_id>/` (default prefix `pipeline-runs`) | With `--s3-sync`; how the Step Functions phases hand off. A failed sync fails the run with `CacheSyncError` |
| Embedding cache | `s3://<aws.s3.bucket_name>/embedding-cache/cache.json` (`indexing.opensearch.embedding_cache_s3_key`) | Only with `indexing.opensearch.persist_embedding_cache: true` |
| Visualization data | `<cache directory>/<pipeline_id>/visualization/` unless `graph.visualization.outputs_directory` is set | Synced with the stage cache (`.json` only) |

The CDK cache bucket expires objects after 30 days under `pipeline-runs/` and
`embedding-cache/` only. Put the corpus under another prefix (for example
`corpus/`): an expired corpus file looks deleted to the next incremental run.
A reused bucket (`cache_bucket_name`) keeps its own lifecycle rules, so check
them.

A `pipeline_id` names the cache directory and S3 prefix, so it accepts only
lowercase letters, digits, hyphens and underscores.

## Start-up endpoint check

Before the first paid model call, the CLIs check that the store endpoints the
run needs are set, and exit with status 1 otherwise:

```text
Error: Missing endpoint configuration for the indexing stage: aws.neptune.endpoint (env NEPTUNE_ENDPOINT), aws.opensearch.endpoint (env OPENSEARCH_ENDPOINT). Set them in the config file or the environment.
```

- `run-ingestion` checks both endpoints when the `indexing` stage is enabled
  (not for `--verify-metadata` / `--repair-metadata`).
- `run-rag` and `run-eval` check the endpoints the chosen strategy's retrievers
  need: OpenSearch for every strategy, and Neptune as well for `local`,
  `drift`, `mix` and `hybrid`. `auto` checks every strategy in `search.auto_routable_strategies`.

The check only tests that a value is set. An endpoint that is set but wrong or
unreachable fails at the first connection; the retrievers report such
authentication, configuration and connection errors instead of returning empty
results.

## Common errors

| Message (abridged) | Meaning | Action |
|---|---|---|
| `Missing endpoint configuration for ...` | An endpoint the run needs is unset | Set the named key or environment variable |
| `Configuration validation error: ...` | A config value has the wrong type or an unsupported value | Fix the named key |
| `Unknown config key '<path>' is ignored` (WARNING) | Typo or removed key; the run continues without it | Correct or delete the key |
| `No valid AWS credentials for Amazon Bedrock in region ...` | Credentials are missing or expired | Refresh credentials; check `AWS_PROFILE` / `aws.profile_name` |
| `Model '<id>' is only available through a cross-region inference profile, but none resolved ...` | The model has no on-demand throughput and no profile was found | Keep `aws.bedrock.enable_global_profile: true`, grant `bedrock:ListInferenceProfiles`, or use another region |
| `Reranking failed: ...` (ERROR, per query) | The Rerank call was denied or the model is unavailable in the region | Grant `bedrock:Rerank` on `Resource: "*"`, move Bedrock to a region with the rerank model, or set `search.reranking.enabled: false` |
| `No indices found for suffix '<suffix>' ...` | Nothing was ingested under that index suffix | Run ingestion, or query with the suffix the corpus was ingested under |
| `InvalidFilterError` | No store the strategy reads declares a filter key | Use a key from the list in the message |
| `Skipping N '.md' file(s) ...` / `No supported source files found in '<dir>'` | The `unstructured` extra is not installed | Install it (Python 3.11+) or convert the files |
| `DocStatusRegistryError` | The doc-status registry table is missing, not accessible, or throttled | Create the table or fix `aws.dynamodb.table_name`, grant the DynamoDB permissions named in the message, then resume with the same `--pipeline-id` |
| `Incremental indexing is enabled but no document delta was computed` | The indexing stage ran without a registry diff (the loading stage failed under `continue_on_error`) | Re-run from `document_loading` once the registry is reachable |
| `CacheSyncError` | The S3 stage-cache download or upload failed | Check bucket permissions and `aws.s3.encryption`, then resume with the same `--pipeline-id` |
| `Pipeline failed at stage(s): ...` | A stage failed; the CLI exits 1 | Read the stage's log lines, fix, resume with the same `--pipeline-id` |
| Bedrock calls hang in a private VPC | Bedrock is called in a region the VPC endpoint does not serve | Keep `aws.bedrock.region_name` unset or equal to the VPC's region |

More messages and their fixes are in the
[User Guide §10](./user-guide.md#10-operations--troubleshooting).
