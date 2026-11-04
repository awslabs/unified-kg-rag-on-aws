# Operator Runbook

Tasks an operator runs against an existing deployment, and what to do when a
run or a query fails. Every configuration key is in the
[Configuration Reference](./configuration.md) (how the configuration loads:
[User Guide §2](./user-guide.md#2-configuration)); the deployed stack and its
commands are in [`iac/README.md`](../iac/README.md#after-deploy).

| I want to | Go to |
|---|---|
| Check a deployment before the first run | [Before a run](#before-a-run) |
| Size the ingestion task and estimate a run's time | [Capacity sizing](#capacity-sizing) |
| Resume, recompute or rebuild an ingestion | [Re-ingest](#re-ingest) |
| Keep a corpus up to date | [Incremental runs](#incremental-runs) |
| Recover a run that stopped | [Interrupted runs](#interrupted-runs) |
| Find out why a document is `FAILED` or `QUARANTINED` | [Failed documents](#failed-documents), [Quarantined documents](#quarantined-documents) |
| Run two ingestions, or understand `NamespaceLeaseError` | [Concurrent runs](#concurrent-runs) |
| Act on a CloudWatch alarm | [Alarms](#alarms) |
| Restore the stores | [Backup and restore](#backup-and-restore) |
| Upgrade the OpenSearch engine version | [Upgrading OpenSearch](#upgrading-opensearch) |
| Upgrade Neptune to a new major version, or move the maintenance windows | [Upgrading Neptune](#upgrading-neptune), [Maintenance windows](#maintenance-windows) |
| Analyze Korean with Nori instead of `cjk` | [Korean with Nori](#korean-with-nori) |
| Switch Neptune to I/O-Optimized storage | [Neptune I/O-Optimized storage](#neptune-io-optimized-storage) |
| Look Neptune vertices up by custom vertex ids | [Custom Neptune vertex ids](#custom-neptune-vertex-ids) |
| Cut the vector memory of a large corpus | [On-disk vector storage](#on-disk-vector-storage) |
| Ship an OWL/RDFS ontology schema in the image | [Ontology schema in the image](#ontology-schema-in-the-image) |
| Ship a seed graph (a knowledge graph to merge), or reset a namespace that holds one | [Seed graphs in the image or S3](#seed-graphs-in-the-image-or-s3) |
| Tune how long a query may take, or act on `QueryTimeoutError` | [Query deadlines and model timeouts](#query-deadlines-and-model-timeouts) |
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

### Capacity sizing

Peak memory of the ingestion task grows linearly with the text units of a
run. A full local build of 5,000 to 60,000 short units (about 200
characters, two or three entities each) with fake models and the embedded
stores measured about 42 KiB of resident memory per text unit on top of
about 0.2 GiB, and about 11 KiB of stage cache per unit. Real units are
longer (`fallback_chunk_size` 4,800 characters) and name more entities and
relationships, so plan for 2-3 times that per unit (an estimate, not
measured):

| `fargate_memory` | Text units per run (estimate) |
|---|---|
| 8192 (default) | about 50,000-80,000 |
| 16384 | about 100,000-160,000 |
| 30720 | about 200,000-300,000 |

Keep a margin: the `IngestionTaskMemory` alarm fires at 90%, and a task over
its memory is killed with the running stage lost. A larger corpus can also
be ingested in several incremental runs, each within the budget. The task's
disk (`ingestion_ephemeral_storage_gib`) holds the synced corpus, the parsed
documents and the stage cache: about twice the corpus plus the cache
(`IngestionTaskEphemeralStorage` fires at 90%).

Run time is dominated by model calls, which run `processing.max_concurrency`
(default 40) at a time:

```text
time ~ calls x average call latency / max_concurrency + about 8 ms per unit
calls ~ units x (1 extraction + gleaning rounds that add something + 1 claim
        extraction, when claims are on) + 1 report per community
```

The 8 ms per unit is the rest of the pipeline (parsing, resolution,
community detection, indexing) measured with fake models and the embedded
stores; Neptune and OpenSearch writes over the network add to it. With a
call latency of 10 s and 40 calls in flight, 50,000 units with one
gleaning round and no claims take about 50,000 x 2 x 10 / 40 s, roughly 7
hours: above the default `sfn_timeout_hours` (6), so raise it or split the
run. `BedrockThrottling` means the account's quota, not `max_concurrency`,
is the limit; the stage durations on the dashboard show where a run spends
its time.

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
results); the next run removes what the previous graph still holds. A
reset stopped or killed before it swapped an index in leaves that index
(`<alias>-<timestamp>`, no alias) until the next reset deletes it; an
incremental run does not. Delete it by hand, or re-run the reset.
A reset pipeline id run again resumes from its cached stages like any
other; when another pipeline id has since deleted extraction records its
text units name (an incremental run with other settings deletes the
records of the units it re-extracts), it resumes from the stage that
stores them instead (WARNING `Extraction records this pipeline stored are
gone ...`) and extracts those units again: the LLM response cache answers
them, and with `cache.llm_responses.enabled: false` the rerun costs what
the reset did.
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
  logs a WARNING naming the suffix's other local source directories (a
  document only a refused or failed run recorded counts as new).
- **Never let corpus files expire or disappear**: the next run removes their
  content. A file the run cannot read keeps it: one that fails to parse or
  load, and one skipped because no loader reads its type (an optional extra
  such as `unstructured` dropped from the image, a loader plugin not
  imported; `Skipping N file(s) ... of a type no loader reads ...`,
  `total_files_skipped`).
- **One index suffix and config file per tenant or corpus version.**

**Mass deletions ask first.** A run that would delete more than
`indexing.max_deletion_share` (default `0.5`) of the documents the registry
holds for its source scope stops at `document_loading`, before it writes
(`This run deletes N of the M documents ... ; continue?`, exit 1), unless
the deletion is confirmed: `run-ingestion` asks on a terminal, and
`--confirm-deletions`, `UKR_CONFIRM_DELETIONS=1` or `--yes` answers yes. A
partial S3 sync, an expired prefix and a volume not mounted look the same
as files removed on purpose. Only scopes of at least 10 documents are
checked; a retired scope's records (`--retire-source-scope`) do not count,
nor do seed graph shards, in either count (`N` and `M` are documents: the
configuration adds and removes shards, and a large seed graph's many shard
records would otherwise hide a partial sync); a reset, and a run with the registry off, are not checked (without the
registry every run is a full build of what it reads, so keep the registry
on for a corpus that may arrive partial; an empty corpus with seeds asks as
in [Seeds with the registry off](#seed-graphs-in-the-image-or-s3)). When the files are really gone, re-run once with
`UKR_CONFIRM_DELETIONS=1` as an environment override, as for the reset in
[Seed graphs in the image or S3](#seed-graphs-in-the-image-or-s3); set the
share to `1.0` or `null` to turn the check off. `total_documents_deleted`
counts what a run removed.

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

**When to reset.** Each suffix's build record (the `build` entry of its
vector indices' `_meta`, which `run-eval` manifests copy) keeps community
counters: `clustered_at` and `communities_at_clustering` from the last full
build, then `communities_now` (each incremental commit adds the communities
it appended and subtracts those its removals deleted) and `runs_since`
(incremental commits since). A commit is counted once: a resumed run that
writes the record again does not count it twice. After a commit that
reaches `indexing.community_compaction.max_runs_since_clustering` (default
`20`) or `max_community_growth` (default `1.5` times the clustered
communities), the run logs a WARNING:

```
Communities of 'default' are due for re-clustering: 24 incremental runs since
the last full clustering (max_runs_since_clustering: 20); 163 communities
against the 10 it clustered, 16.3x (max_community_growth: 1.5). ...
```

It means global search now maps the appended communities' reports next to
the original ones, and their reports miss what later runs added. Run once
with `indexing.reset: true`: it re-clusters the whole graph and reuses the
stored summaries and reports whose input is unchanged, so it asks the
model mostly for the communities it clusters differently (on the cost
harness, 17-20 calls after 23 incremental runs against 197 for the first
build). The counters start again at that build. Alarm on the
`community_growth_ratio` or `runs_since_clustering` metric (the largest
over the run's suffixes; `0` for a run that recorded no counters).
Nothing is reset automatically. `null` turns a threshold off;
`on_drift_recommendation: true` also gives the recommendation when the
run's `community_rebuild_recommended` is 1. `communities_now` is an upper
bound (a delta community written under a stored community's id counts
again, and a run killed after its removal deleted communities does not
subtract them). A suffix whose record an earlier version wrote has no
counters, and no recommendation, until its next full build.

**Registry unreadable** (missing table, denied access, lasting throttling):
`document_loading` fails with `DocStatusRegistryError` naming the table and
cause, rather than reindexing everything. Fix the registry and re-run with the
same `--pipeline-id`, or set `providers.doc_status.enabled: false`.

Coming from 0.2.3 or an earlier `main` checkout: see
[CHANGELOG: Upgrading](../CHANGELOG.md#upgrading).

### Caches and prefixes

| Cache | Location | Notes |
|---|---|---|
| Stage checkpoints (local) | `<--cache-directory>/<pipeline_id>/` (default `cache/`) | Reused on resume |
| Stage checkpoints (S3) | `s3://<bucket>/<--s3-prefix>/<pipeline_id>/` (default `pipeline-runs`) | With `--s3-sync`: uploaded after every stage and on SIGTERM, so a fresh task resumes from it. A failed download or final upload fails the run with `CacheSyncError`; a failed per-stage or on-stop upload only warns (the next upload retries it), and a stop after a lost lease skips the upload. Expired by the bucket lifecycle, not `cache.ttl_seconds` |
| LLM response cache | `<stage checkpoint dir>/llm-cache/` | Reused by a re-run with the same pipeline id; `cache.llm_responses.enabled: false` turns it off, `--force-rebuild` ignores it |
| Extraction store | `graphrag-extraction-records-<suffix>` (`indexing.opensearch.extraction_records_index_prefix`) | Per-chunk extraction results and the intelligent chunker's per-pre-chunk boundary answers, reused by later runs while a stored text unit names them, the community report answers of full builds, kept while a stored report names them (a full build drops the others), and the notes a removal (`pending-strip`) or a commit (`pending-commit`) keeps until it finished, which a resumed run needs. `clear` and `--force-rebuild` keep it; to drop it, delete the index only after a run that succeeded, or run `indexing.reset: true` next |
| Visualization data | `<cache directory>/<pipeline_id>/visualization/` unless `graph.visualization.outputs_directory` is set | Synced with the stage cache (`.json` only) |

The CDK bucket expires `pipeline-runs/` after 30 days: keep the corpus under another prefix (`corpus/`), since an expired file
looks deleted. The bucket is versioned: an overwritten or deleted object
keeps its earlier version for 30 days (under `pipeline-runs/`, which every
stage overwrites, for 1 day). A `--s3-prefix` other than `pipeline-runs`
gets neither the expiry nor the 1-day rule. Check a reused bucket's own
lifecycle rules and versioning.

### Schema sources in S3

A `schema_source` of kind `file` or `owl` may name an `s3://` object (or,
for `owl`, an `s3://bucket/prefix/` directory). Every entry point that loads
the schema reads it with the task's role: ingestion, prompt tuning and
evaluation at start-up, and `run-rag` on the first query with a schema
filter key (`in_schema`, `type_iri`, `predicate_iri`), which it checks
against the schema. When `run-rag` cannot load the source it skips that
check with one WARNING and does not retry on later queries.
The loader needs `s3:GetObject` on the objects and, for a prefix,
`s3:ListBucket` on the bucket (without `s3:ListBucket` a missing object is
reported as access denied too), plus `kms:Decrypt` on the key of a
KMS-encrypted object (the stacks grant it for the deployment's own key).

The CDK stacks grant no schema-specific permission; place the schema where
the roles already read:

| Location | Ingestion task role | Query task role (`run-rag`, `run-eval`) |
|---|---|---|
| The `corpus_bucket_name` bucket, outside the ingestion source prefix (`UKR_SOURCE_DIRECTORY`, whose files are ingested as documents) | Read | Read |
| The cache bucket, outside `pipeline-runs/` (which expires after 30 days) | Read | No |
| The cache bucket under `eval/` | Read | Read |

In private mode the S3 gateway endpoint also reaches only these two
buckets, so a schema in any other bucket fails with access denied even
after you grant the role `s3:GetObject` on it. To keep it elsewhere, grant
both roles read on it and add the bucket to the endpoint policy in your own
stack.

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
| S3 cache bucket | Versioning: earlier versions of overwritten or deleted objects (the corpus under `corpus/`, `eval/`); checkpoints are re-creatable | 30 days (1 day under `pipeline-runs/`); none on a reused bucket unless it has its own |

A deleted or overwritten corpus file looks deleted or changed to the next
incremental run, which then removes or replaces its documents' artifacts.
Restore it before that run: list its versions and copy the one you want
back, which makes it the current version again.

```bash
aws s3api list-object-versions --bucket <cache bucket> --prefix corpus/<file>
aws s3api copy-object --bucket <cache bucket> --key corpus/<file> \
  --copy-source "<cache bucket>/corpus/<file>?versionId=<version id>"
```

A file deleted without a version id leaves a delete marker; deleting the
marker (`aws s3api delete-object --bucket <cache bucket> --key corpus/<file>
--version-id <marker id>`) restores it as well, but the task roles have no `s3:DeleteObject`, so
run it with an operator role.

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

### Upgrading OpenSearch

The CDK domain runs OpenSearch 2.13 and carries the CloudFormation
`EnableVersionUpgrade` update policy, so a changed engine version upgrades
the domain in place: same endpoint, same indices, no reindex. Without the
policy CloudFormation would replace the domain with a new, empty one.
OpenSearch Service upgrades 2.13 to 2.19 in one step, as a blue/green
deployment (every shard copied; latency and rejections may rise until it
ends, from minutes to hours). An upgrade cannot be paused or rolled back.

1. Stop ingestion (no `RUNNING` execution) and plan for slower queries.
2. Take a manual snapshot: the way back to 2.13 is restoring it on a new
   2.13 domain (a manual snapshot needs a registered S3 repository; see
   [Creating index snapshots](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/managedomains-snapshots.html)).
   The hourly automated snapshots stay with the domain.
3. Run the eligibility check:
   `aws opensearch upgrade-domain --domain-name <domain> --target-version OpenSearch_2.19 --perform-check-only`,
   then `aws opensearch get-upgrade-status --domain-name <domain>`. With the
   optional Nori package associated, also check that a Nori package for 2.19
   exists (`aws opensearch describe-packages --filters
   Name=PackageName,Value=analysis-nori Name=EngineVersion,Value=OpenSearch_2.19`):
   optional plugins are version-specific.
4. Set `OPENSEARCH_VERSION = opensearch.EngineVersion.OPENSEARCH_2_19` in
   `iac/stacks/storage_stack.py` (and the local image in
   `docker/compose.local.yaml`, so local runs test the same engine), then
   `cdk diff GraphRagStorage`: the only change must be the domain's
   `EngineVersion`, with no replacement, and with `opensearch_nori_package`
   the `EngineVersion` of its `Custom::OpenSearchNoriPackage`. Deploy with
   `cdk deploy GraphRagStorage`. The package resource updates after the
   domain: it associates the 2.19 package unless the upgrade already did,
   and the update's cleanup dissociates the 2.13 one if it is still there
   ([Korean with Nori](#korean-with-nori)).
5. When the stack update completes, run a query and one incremental
   ingestion before the next large run.

### Maintenance windows

All times are UTC; set them to the deployment's quiet hours.

| Store | When | Key (default) |
|---|---|---|
| Neptune cluster and instances | Weekly maintenance: minor engine upgrades (`auto_minor_version_upgrade`), patches | `neptune_maintenance_window` (`sun:20:00-sun:21:00`) |
| Neptune | Daily automated backup | `neptune_backup_window` (`18:00-19:00`; apart from the maintenance window) |
| OpenSearch | Daily 10-hour off-peak window: service software updates, Auto-Tune | `opensearch_off_peak_start` (unset: the service default, 10:00 P.M. local time of the region) |

Changing a window takes effect on the next deploy without downtime.
Maintenance restarts Neptune instances (writes fail until the writer is
back, and the run's retries may not outlast it), so keep ingestion runs out
of the Neptune window. A pending OpenSearch service software update is
listed by `aws opensearch list-scheduled-actions --domain-name <domain>`.

### Upgrading Neptune

Minor versions upgrade in the maintenance window. A major version (1.3 to
1.4, say) is never applied by itself, and the stack does not pin
`engine_version` (a pin would change every deployed cluster), so you start
it with the CLI and CloudFormation sees no change:

1. Check the path and the client: `aws neptune describe-db-engine-versions
   --engine neptune --engine-version <current> --query
   'DBEngineVersions[].ValidUpgradeTarget[].EngineVersion'`, and that the
   target supports the TinkerPop version of the pinned `gremlinpython`
   ([Neptune engine releases](https://docs.aws.amazon.com/neptune/latest/userguide/engine-releases.html)).
2. Try it on a copy first: restore the latest cluster snapshot to a new
   cluster (`aws neptune restore-db-cluster-from-snapshot`, then add an
   instance), upgrade the copy, and run a query and an incremental ingestion
   against it with `UKR__AWS__NEPTUNE__ENDPOINT` pointed there. Delete the
   copy afterwards.
3. Stop ingestion (no `RUNNING` execution) and take a manual snapshot:
   `aws neptune create-db-cluster-snapshot --db-cluster-identifier
   <cluster> --db-cluster-snapshot-identifier <cluster>-pre-<version>`.
   It is the way back: a major upgrade cannot be rolled back in place.
4. Upgrade: `aws neptune modify-db-cluster --db-cluster-identifier <cluster>
   --engine-version <target> --allow-major-version-upgrade
   --apply-immediately`. Every instance restarts; the cluster is
   unavailable until the upgrade ends (typically minutes, longer for a large
   graph), so queries fail meanwhile. The cluster uses the default parameter
   groups, which follow the new engine family.
5. When `aws neptune describe-db-clusters` shows the new `EngineVersion` and
   `available`, run a query and one incremental ingestion before the next
   large run.

### Korean with Nori

The CDK deployment maps Korean (`ko`) to the built-in `cjk` analyzer
(overlapping character pairs), because a domain has the morphological `nori`
analyzer only once the AWS-provided `analysis-nori` plugin package is
associated with it. `-c opensearch_nori_package=true` associates it: a
custom resource looks up the package of the domain's engine version
(`DescribePackages`), associates it (`AssociatePackage`, a blue/green
deployment) and waits until it is `ACTIVE`, up to two hours; the compute
stack, deployed after storage, then maps `ko` to `nori`.

1. Stop ingestion (no `RUNNING` execution).
2. `cdk deploy GraphRagStorage GraphRagCompute -c opensearch_nori_package=true`.
   A large domain whose blue/green deployment takes longer than two hours
   fails the resource: associate the package by hand first
   (`aws opensearch associate-package --package-id <id> --domain-name <domain>`,
   then wait for `ACTIVE` in `aws opensearch list-packages-for-domain`); the
   resource finds it associated and does nothing.
3. Run once with `indexing.reset: true` ([Re-ingest](#re-ingest)): the
   field analyzers of a live index cannot change, so an incremental run
   fails until then with `The OpenSearch index '<index>' was built for
   other language settings: '<field>' uses 'cjk' (needs 'nori')`. Set it
   back to `false` afterwards.

To keep `cjk` while the package is associated (to rebuild later), deploy
with `-c opensearch_korean_analyzer=cjk` as well.

**Turning it off.** OpenSearch refuses to dissociate a package an index
still uses, so move the Korean indices off `nori` first: deploy with
`-c opensearch_korean_analyzer=cjk` (package still on), run one
`indexing.reset: true`, then deploy without `opensearch_nori_package`. The
resource dissociates the package (another blue/green deployment) only when
a stack update removes or replaces it; deleting the stack leaves it
associated, so a retained domain keeps its `nori` indices usable. A failed
dissociation does not fail the update (CloudFormation ignores cleanup
failures); `aws opensearch list-packages-for-domain` shows it.

### Neptune I/O-Optimized storage

Neptune bills Standard storage per I/O request; I/O-Optimized (`iopt1`)
bills no I/O but costs more per instance-hour and per GiB-month. AWS's rule
of thumb: switch when I/O is above 25% of the cluster's Neptune spend
(compare the I/O line of the Neptune bill in Cost Explorer with its total;
the `VolumeReadIOPs` and `VolumeWriteIOPs` metrics show when the I/O
happens). The engine must be 1.3.0.0 or later. CloudFormation has no
property for the storage type, so switch the deployed cluster with the CLI,
with no ingestion running:

```bash
aws neptune modify-db-cluster --db-cluster-identifier <cluster> \
  --storage-type iopt1 --apply-immediately
aws neptune describe-db-clusters --db-cluster-identifier <cluster> \
  --query 'DBClusters[0].[Status,StorageType,IOOptimizedNextAllowedModificationTime]'
```

The switch is done when the cluster is `available` again with storage type
`iopt1`. `--storage-type standard` switches back.

- **Once per 30 days.** A switch to I/O-Optimized is allowed once per 30
  days (`IOOptimizedNextAllowedModificationTime`). AWS documents a switch
  back to Standard as allowed at any time in one place and as limited to
  once per 30 days in another, so plan either switch as a monthly decision.
- **Outside the stack.** The setting is not in the template, so CDK keeps
  it on stack updates, but a cluster the stack replaces (or a restore,
  which creates a new cluster) starts on Standard: run the command again.

### Custom Neptune vertex ids

`indexing.neptune.vertex_ids` (default `property`) picks how the indexer and
the graph retriever find vertices. With `property`, Neptune assigns each
vertex id and every lookup filters on the item's `id` property. With
`custom`, each vertex id is `<label>:<item id>`, for example
`Entity-default:3f2a...` or `Community-tenant1-v2:c12`. Neptune serves a
lookup by vertex id faster than a property filter, and concurrent writes
by vertex id raise fewer `ConcurrentModificationException`s
([Neptune guide](https://docs.aws.amazon.com/neptune/latest/userguide/best-practices-content-15.html)).
The label is part of the id because a vertex id is unique in the whole
graph, while the same entity id is stored once per suffix. Labels hold
only letters, digits, `-` and `_`, so the first `:` separates the two parts.
The `id` property is still written. Edges keep their ids, and each edge is
still written in its own request. The embedded SQLite graph store ignores
the setting.

When to enable: for a new graph, or a large or write-heavy one (many
concurrent ingestion writers, `index_concurrency` above 1, or CME retries
in the logs). On a small graph the gain is small.

Switching an existing graph needs one full build:

1. Set `vertex_ids` to the new mode for ingestion (`UKR__INDEXING__NEPTUNE__VERTEX_IDS`).
   An incremental run in one mode refuses a label built in the other mode,
   with a `StoreConfigurationError` that names the label.
2. Run ingestion once with `indexing.reset: true` (with incremental indexing
   off, any run is a full build). The build first writes a
   `VertexIdSwitch` vertex for the label and then writes the graph in the new
   mode. At the end, the sweep drops the vertices of the old mode and the
   switch vertex.
3. Set the same mode for queries, before, during or after the build. Graph
   expansion never refuses a label: while the label is in the other mode or
   holds a switch vertex, it looks vertices up by their `id` property (both
   modes write it) and logs a WARNING once per label. During the build it
   reads each entity once: of its old and its rewritten copy, the one nearer
   the seeds (the rewritten one when both are as near), and the old copy
   where the build has not rewritten it yet. The expansion then reads a
   graph split between old and new copies: the build replaces each edge's
   old copy with one between the rewritten endpoints, so a path runs over
   old or over rewritten vertices only, and results may be partial (fewer
   or farther entities) until the sweep ends. Once the label is settled in
   the query's mode, custom-mode queries go back to vertex-id lookups. A
   running query service needs no restart: in `custom` mode each query
   reads the label's switch vertex (one indexed read), and a lookup that
   finds none of its seeds checks the label's mode again. In `property` mode
   a label found in property mode is not read again (its lookups find the
   vertices of both modes, and the rewritten copy is told by its custom
   vertex id), so a running property-mode service logs no WARNING for a
   switch to `custom` it started before.
4. If a write fails, the build keeps the old vertices for a retry, and the
   switch vertex stays. In that state both modes refuse incremental runs
   ("did not finish") until a full build succeeds; queries go on reading by
   the `id` property.

Gremlin Server stores string vertex ids only when its graph uses
`vertexIdManager=ANY`, which `docker/compose.local.yaml` sets; on a stock
server each custom-id write fails at once with a `StoreConfigurationError`
naming the setting.

### On-disk vector storage

By default every vector field keeps full-precision vectors in memory
(OpenSearch's `in_memory` mode; no key is sent, so every version takes it).
OpenSearch 2.17+ can store them quantized instead:

```yaml
indexing:
  opensearch:
    vector_mode: on_disk     # in_memory (default) | on_disk
    compression_level: 4x    # 1x 2x 4x 8x 16x 32x; unset = the mode's default
```

- **When it pays off.** Only when data-node memory limits the corpus, at
  millions of vectors (each document fills up to 8 vector fields). `on_disk`
  keeps compressed vectors in memory and rescores each search's candidates
  with the full-precision ones read from disk: less memory, slightly more
  latency and a little recall, and no disk saved. faiss takes 2x, 8x, 16x and
  32x (`on_disk`'s default), lucene 4x. On a synthetic 50,000-vector set
  (256 dimensions, OpenSearch 2.19.3,
  `tests/integration/test_opensearch_vector_storage.py`) recall@10 was 0.95
  for faiss 8x, 0.93 for lucene 4x and 0.78 for faiss 32x (1.0
  unquantized), and faiss's native memory fell from 61 MB to 17 MB (8x) or
  12 MB (32x). Measure recall on your own corpus before choosing a level.
- **Version.** The keys need OpenSearch 2.17 or later, and faiss with the
  default `cosinesimil` space 2.19. The CDK domain runs 2.13 until upgraded
  ([Upgrading OpenSearch](#upgrading-opensearch)); there the probe before the
  first model call refuses the run and names the 2.17 requirement.
  Combinations no version takes (lucene 2x, faiss 4x, `on_disk` with 1x,
  lucene `on_disk` without a level, another engine, method or space type)
  fail at config load.
- **Reindex.** Both are fixed when an index is created. An incremental run
  into indices built otherwise stops before any model call (`... stores the
  vectors of '...' as ...`); run once with `indexing.reset: true`.
- **Local mode.** The embedded engine ignores both (one warning per process)
  and keeps searching exactly at full precision.

### Ontology schema in the image

A config with `schema_source: {kind: owl}`
([User Guide: From an OWL/RDFS ontology](./user-guide.md#from-an-owlrdfs-ontology))
needs the `ontology` extra and the ontology files in every image that runs
it: ingestion, and also the query and evaluation tasks, which read the
schema to check filter keys. (To keep the ontology out of the image, put
it in S3 instead of steps 1-3: [Schema sources in S3](#schema-sources-in-s3);
the extra is still needed.) The data-plane image has neither by default,
and a task without the extra exits 2 before any model call (`Not
installed: 'rdflib' (for processing.graph_extraction.schema_source.kind
'owl')`).

1. Put the ontology next to the baked-in config, for example
   `docker/ontology/procure.ttl` (a directory of files is read as one
   ontology). Give it absolute IRIs or a declared base (`@base`,
   `xml:base`): a relative IRI resolves against the file's path in the
   image, so the same file read elsewhere names other terms.
2. Copy it into the image after the config in `docker/Dockerfile`:

   ```dockerfile
   COPY docker/config.yaml /app/config.yaml
   COPY docker/ontology /app/ontology
   ```

3. Select it in `docker/config.yaml` (a relative path is resolved against
   `/app/config.yaml`, so this reads `/app/ontology`), or with
   `UKR__PROCESSING__GRAPH_EXTRACTION__SCHEMA_SOURCE__KIND=owl` and
   `UKR__PROCESSING__GRAPH_EXTRACTION__SCHEMA_SOURCE__PATH=ontology` on the
   task:

   ```yaml
   processing:
     graph_extraction:
       schema_source:
         kind: owl
         path: ontology
   ```

4. Build with the extra, then push and deploy as for any image change
   ([`iac/README.md`](../iac/README.md#after-deploy)); several extras go in
   one argument (`"--extra ontology --extra unstructured"`):

   ```bash
   docker build --platform linux/amd64 --build-arg UV_EXTRAS="--extra ontology" \
     -f docker/Dockerfile -t "$REGISTRY/graphrag-app:<tag>" .
   ```

The ontology is read at start-up and keyed by the schema it maps to, so
rebuilding the image with an unchanged ontology, or moving it to another
path with absolute IRIs, re-extracts nothing. Before the first full
ingest, preview the mapped schema locally (User Guide, *Preview*) and
settle the options: a label or description change re-extracts the
corpus. With `on_collision: qualify`, bind the colliding namespaces in
`options.prefixes`, so a re-serialised ontology keeps its suffixes.

Every task that loads the schema parses the ontology into memory: about
3 s and 85 MB per MB of Turtle (a 14 MB file took 45 s and 1.2 GB), so at
the default `max_bytes` of 64 MiB a load can take about 5 GB of the task's
memory before ingestion starts. Lower `options.max_bytes` to the size you
ship, or raise `fargate_memory`, when the ontology is large (User Guide,
*Load cost*).

### Seed graphs in the image or S3

A config with `processing.seed_graphs`
([User Guide: Seed graphs](./user-guide.md#d-seed-graphs-bring-your-own-knowledge-graph))
is read by the ingestion task only, once at start-up: a seed that cannot be
read or imported exits 2 before any model call, naming the seed. Put the
files in one of two places.

- **In the image**, next to the baked-in config: copy them in
  `docker/Dockerfile` (`COPY docker/seed /app/seed` after the config) and
  give a path relative to `/app/config.yaml` (`path: seed/procure.jsonl`).
- **In S3**: `path: s3://bucket/seed/procure.jsonl`, or a prefix read as a
  directory (`s3://bucket/seed/`, the trailing `/` required; kind `csv`
  always names a prefix). The ingestion task role needs `s3:GetObject` on
  the objects, `s3:ListBucket` on the bucket for a prefix (and to tell a
  missing object from a denied one), and `kms:Decrypt` on the key of a
  KMS-encrypted object. The CDK stacks grant nothing seed-specific; put
  the files where the role already reads:

| Location | Ingestion task role |
|---|---|
| The `corpus_bucket_name` bucket, outside the ingestion source prefix (`UKR_SOURCE_DIRECTORY`: its files are ingested as documents, a `.csv` seed too) | Read |
| The cache bucket, outside `pipeline-runs/` (which expires after 30 days) | Read |
| The cache bucket under `eval/` | Read |

In private mode the S3 gateway endpoint reaches only these two buckets
(see [Schema sources in S3](#schema-sources-in-s3)). Moving a seed between
the image and S3 re-indexes nothing: what it converts to is keyed, not
where it is read from. Size the task for it: the whole seed is held in
memory, and each node costs about two embeddings and each edge one
(User Guide, *Cost and limits*).

A seed of kind `rdf` needs the `ontology` extra in the ingestion image:
`run-ingestion` checks the extras a configured seed kind needs before
any model call (`Not installed: 'rdflib' (for processing.seed_graphs
'procure' kind 'rdf')`, exit 2). Build it with `--build-arg
UV_EXTRAS="--extra ontology"` as in
[Ontology schema in the image](#ontology-schema-in-the-image), step 4.
The query and evaluation images load no seed and need no seed kind's
extra (an `owl` schema source still needs it wherever it is read).

**A reset where another scope holds seeds.** `indexing.reset: true`
rebuilds the namespace from the run's corpus and drops the other source
scopes' documents, as always. When one of those scopes holds seed graph
documents (a seed-only scope), or the run's corpus is its seed graphs
alone (a seed-only scope with seed records of its own) while another
scope holds documents, `run-ingestion` asks first,
and an ECS task has no terminal to ask on: the run stops at
`document_loading`, before it writes (`... needs confirmation, but stdin
is not a terminal. Re-run with --confirm-reset (or UKR_CONFIRM_RESET=1)
to proceed.`, exit 1). `UKR_CONFIRM_RESET=1` answers that question only
(not a resume's integrity check, which `--yes` answers). The Step
Functions pipeline passes only the source directory and pipeline id, so
run that one reset as a one-off task of the ingestion task definition
(`.States.Ingest.Parameters.TaskDefinition`, in the network of the query
example in [`iac/README.md`](../iac/README.md#4-query-from-inside-the-vpc)):
the state machine's command unchanged (the image's own command only
prints the help), and the reset and its answer as environment overrides,
so the config in the image stays as it is:

```json
{"containerOverrides": [{"name": "app",
  "command": ["run-ingestion", "--config-path", "/app/config.yaml",
              "--metrics-sink", "cloudwatch", "--s3-sync",
              "--s3-bucket-name", "<cache bucket>"],
  "environment": [{"name": "UKR__INDEXING__RESET", "value": "true"},
                  {"name": "UKR_CONFIRM_RESET", "value": "1"},
                  {"name": "UKR_CONFIRM_DELETIONS", "value": "1"},
                  {"name": "UKR_SOURCE_DIRECTORY", "value": "s3://<corpus bucket>/<prefix>/"},
                  {"name": "UKR_PIPELINE_ID", "value": "reset-001"}]}]}
```

Each answer covers one question; keep only the ones the run asks.
`UKR_CONFIRM_RESET=1` is for this reset (`UKR__INDEXING__RESET=true`) when
another scope holds seeds or the corpus is seeds alone, and for an
empty-corpus seed build with the registry off (**Seeds with the registry
off**, below). `UKR_CONFIRM_DELETIONS=1` is for an
incremental run, without the reset, that deletes more than
`indexing.max_deletion_share` of its scope's documents ([Mass deletions
ask first](#incremental-runs)): for that re-run drop the two reset lines.
A reset is never checked for mass deletions, so with the reset on
`UKR_CONFIRM_DELETIONS` changes nothing; neither answers a resume's
integrity check (`--yes` does).

Two resets from an empty corpus (seeds alone) fail whatever the answer,
`UKR_CONFIRM_RESET=1` included: of a scope that holds documents, under
any spelling of its source (`s3://bucket/prefix` and
`s3://bucket/prefix/` are one scope; `No source document reached this
run, only its seed graphs, ...`), and of a scope the registry holds no
record of while other scopes hold documents (`... holds no record of
source scope ...`: a new, moved or mistyped prefix). In both the corpus
is likelier missing than empty, as an incremental run assumes too. To
rebuild a namespace from seeds alone, run the seed-only scope once
without the reset, then reset it.

**Seeds with the registry off.** Without the doc-status registry every
run is a full build, so an empty corpus beside seeds would replace the
whole index with the seeds. The run builds from seeds alone unasked only
while the index holds no entity, relationship or text unit (a first seed-only
build); otherwise it asks (`... rebuild the index '...' from the seed
graphs alone, replacing what it holds; continue?`) and an ECS task stops
at `document_loading` before it writes, exit 1, unless
`UKR_CONFIRM_RESET=1` (or `--confirm-reset`, `--yes`) answers it. A
store that cannot tell or cannot be read counts as holding items. Keep
the registry on (`providers.doc_status.enabled`) for a corpus that may
arrive empty: it tells a seed-only scope from a missing corpus.

Or avoid the question: configure the seed on the corpus's own scope
instead of a seed-only scope, so the reset re-imports it and asks nothing.

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
- An interruption is not a failure. A stop or death during extraction leaves
  `failure_count` and `retry_count` as they were; one during indexing, once
  the `PENDING` records are written, resets both. Deaths are counted in
  `attempt_count` ([Quarantined documents](#quarantined-documents)).
- A custom registry policy needs `dynamodb:BatchWriteItem` and
  `BatchGetItem` (the CDK grant covers both).
- On the CDK stack with `ingestion_fargate_spot`, the state machine re-runs
  a task Spot interrupted by itself, after `spot_retry_wait_seconds`; the
  execution history shows the `SpotInterrupted` and `WaitBeforeSpotRetry`
  states ([Fargate Spot](../iac/README.md#fargate-spot)).

### Failed documents

A document is recorded `FAILED` in five cases:

- translation, graph extraction, gleaning or claim extraction failed on one
  of its text units (claim extraction for an error of the unit's own: see
  below);
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
run wrote and processes it again. A run of the same pipeline id without
`--resume-from-stage` resumes from the earliest completed stage enabled in
this run that recorded failed text units and retries those units (WARNING
`Stage '...' failed on N text units; resuming from it ...`); the extraction
store keeps the units that succeeded. An explicit `--resume-from-stage` past
it is kept, with a WARNING: the indexing stage records those documents
`FAILED` again without retrying them, one more run toward
`indexing.max_document_failures` (past it, resuming from the failed stage
excludes them instead of retrying them). Failures of a stage that is not
enabled in the rerun are dropped from the restored state when the stage is
at or after the resume point. Before it, a stage the run does not enable
keeps them: an indexing-only phase records them once, and that phase run
again after it finished is refused (`PipelineResumeError`: enable the named
stage to retry them; an explicit `--resume-from-stage` is kept, with a
WARNING, and counts them again). A document that
reached the retry limit makes the rerun diff the corpus again, which skips it.

A stage failure stops the run before indexing and leaves the stores as they
were; in the registry it only adds one to the `retry_count` of the documents
whose units failed for an error not their own (see below), and the next run
processes the same delta again. The exception: when graph extraction fails on all of its units
(`Critical stage 'graph_extraction' ... produced 0 outputs. It failed on N
files: ...`), the documents whose units failed with an error of their own
are recorded `FAILED`, so the retry limit below applies.

Only an error of the unit's own counts toward the retry limit, on this path
and when the indexing stage records a document `FAILED`: a response that does
not parse or was cut off (or an empty translation), or a request the model rejects for its content (a
Bedrock `ValidationException`, HTTP 400/413/422 from a local server) other
than a rejection of the model or a setting (an invalid model id, a model that
needs an inference profile, an unsupported parameter). Nothing else counts:
throttling, 5xx, timeouts, connections, credentials (an expired SSO token,
`NoCredentialsError`), a missing region, quotas
(`ServiceQuotaExceededException`), stream errors, bugs and failed store writes
fail every document alike. Such a document stays `PENDING` (or `FAILED`,
retried by the next run), uncounted, until the cause is fixed: an outage or a
bad setting never excludes it. Its consecutive failed runs of the same
content are kept (`retry_count`; new content, a success or an
error of its own starts it over; a stop or death during extraction leaves
it, one during indexing, once the `PENDING` records are written, resets it):
from the second, the run warns with the file and the count and reports it in
`documents_retrying`. When every unit sent fails and its errors of their own
are of one cause (the same error type, code and HTTP status, whatever the
message says; or all output that does not parse or was cut off; a unit
throttled or timed out alongside is left out), output errors on the units of
more than one document count as each document's own, without a known-good
request: a server-side output cap cuts them all, and a short known-good
request cannot tell it from each document's own output. Each run they fail
counts once (`FAILED`, `1/N` from the first run; an outage run in between
neither counts nor ends the streak), so a batch that always fails is
excluded after `indexing.max_document_failures` runs and cannot stop every
run with the delta's deletions. The trade-off: a server-side output cap that
persists for that many runs excludes those documents too, until their
content or the extraction settings change (fix the cap, then edit the files
or change a setting). Otherwise a known-good request decides (sent through the same chain,
never served from the LLM response cache): failed too (for output errors,
an answer that does not parse is a failure), nothing counts; answered, the
errors count, so a delta of documents the model rejects is still excluded,
and so is a single document whose output never parses. A content error that
later clears with no change can thus exclude a document. Graph extraction
output that lists entities but none with a name, with a section left
unclosed (cut off without a stop reason), is output cut off (not retried, as
at the token limit); with every section closed it is malformed output,
retried like output that does not parse and counted the same way when every
attempt fails. Output cut later keeps the entities before the cut, with a
WARNING (`XML output ends with <entities> unclosed`). The response behind a
failed graph or claim extraction check is logged at DEBUG (its start and end
when long). An answer the model corrects
(the answer, a line such as "Correction: ... final output:", the answer
again) is parsed by its last complete answer, each section repeated keeping
its last closed occurrence (INFO `XML output repeats <entities>, ...`; a
WARNING when the last one is unclosed), and a JSON answer by the last object
that parses; a single answer cut off is flagged as before. An OpenAI-compatible server's
context-length rejection that counts the requested `max_tokens` ("you
requested N tokens", "'max_tokens' or 'max_completion_tokens' is too large")
is a setting and counts for none.

- **Claims are optional.** When only claim extraction failed on some of a
  document's units, for an error of their own (a request rejected for its
  content, output cut off or unparseable, alike across documents too), the
  document is indexed without those units' claims and counted for none. It
  is recorded `FAILED` once (`error_info`: `indexed without the claims of
  some text units ...`, counted in `documents_without_claims`, not in
  `failed_documents`) so the next run extracts
  the missing claims again, reusing the stored graph and claim records;
  failing so again, it is recorded `PROCESSED` without them, and later runs
  do not extract them again until its content or the extraction settings
  change (a re-run or resume with the same pipeline id still does).
  A server-side output cap on claims thus re-indexes such documents once and
  excludes none. Both runs warn with the files. Any other claim extraction
  failure (an outage, a record not stored) fails the document, uncounted, as
  in the other stages. Gleaning is not treated so: a unit whose gleaning
  failed has no final graph record to store, so its document must be
  extracted again.
- **Retry limit.** After `indexing.max_document_failures` (default `3`)
  consecutive runs that recorded unchanged content `FAILED` for an error of
  its own, output that failed alike on several documents included (see
  above; with the same extraction settings: a settings change starts the
  count over) the document stays
  `FAILED`, keeps what it indexed, and is skipped: each run logs a WARNING
  naming the file and goes on with the rest of the delta, deletions included.
  Fix the cause (often an oversized or malformed file), edit the file, or
  raise the limit.
- **Too large for the registry.** A document whose text-unit ids exceed one
  DynamoDB item (400 KB, about 10,000 ids, half that for an edited document)
  fails indexing before anything is written, with an error naming the file:
  split it.
- **Write failures** are retried first: OpenSearch bulk items rejected with
  429, 502, 503 or 504 up to four times with backoff, every Neptune request up
  to `retry.graph_store_max_attempts` (default `4`) over a new connection. If
  more than `indexing.max_failure_rate` (default `0.2`) of one artifact type's
  writes still fail, indexing fails and records no document; below it, only
  the documents owning a failed artifact are `FAILED` (all of the run's when
  the backend names no item), retried by the next run without counting
  toward the retry limit. The `IndexingFailures` and
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
  `... resuming from 'document_loading' ...`). So do they when
  `document_loading`'s inputs changed since the delta was computed (a reset,
  retry or corpus setting; WARNING `Rewinding resume point from ... to
  'document_loading'`). A finished pipeline id run again counts as resumed:
  it resumes from `document_loading` when another pipeline id has since
  written a record of one of its documents (every record names the pipeline
  id that wrote it last; a record written before that was recorded names
  none: a `PENDING` or `FAILED` one, or one with an open attempt, counts as
  another pipeline id's, as does a `PROCESSED` one at other content than the
  delta's; one at the delta's content is left to the other checks),
  committed one of its documents with other extraction settings (a record
  naming no writer counts even for a document the delta re-extracts for a
  settings change), or added, changed or deleted a document outside its
  delta. A run diffs within its own scope (index namespace and source
  scope), so the last check reads only that scope; a document the delta
  deletes that another pipeline id (or a record naming none) has since
  recorded in another scope (one the delta does not retire) makes it diff
  again too, so a rerun never deletes another scope's live document. A run
  resumed after its own kill or stop keeps its delta.

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
  expired. The stopped run's own message says until when (`the next run on
  its namespace waits until ...`). A SIGTERM ends the wait. With every
  store embedded (`backend: local`, SQLite) a stopped run releases its
  lease: every embedded write checks the lease once it holds the database's
  lock, so a write the stop left running fails instead of landing after
  the next run's, and a takeover waits at most one store write
  (`search.store_timeout_seconds`). A process frozen for longer than that,
  with a request already built, can still write after the new holder; only
  a write fenced by the store itself would rule that out. With DynamoDB, an
  expired lease item its TTL already deleted reads as released (TTL
  deletion usually takes much longer than 3 minutes).
- **Errors in library use.** A failed indexing phase waits for the store
  requests its threads already sent before it raises, so a caller that runs
  again in the same process does not race them; a stop (SIGTERM, Ctrl-C)
  does not wait, as the process ends.
- **Checkpoints.** A run downloads its S3 checkpoint only once it holds the
  lease; a stopped run uploads its checkpoint before giving the lease up
  (even during its takeover wait, which holds back only store and registry
  writes), one that lost the lease uploads none.
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
error to the SNS topic, and the alarms below report to the same topic: six
on pipeline runs, eleven on store health (more on a multi-node domain, with
`use_cmk` and on Neptune Serverless), three on capacity and one on queries.
They have a subscriber only with `-c alarm_email=<address>`.
A failed task also fires `PipelineFailures`, a second notice without context
that also covers what the message cannot report (an error the task's handler
cannot catch, such as an input without `pipeline_id`, or a failed publish).

**Pipeline runs**

| Alarm | Fires when | What to do |
|---|---|---|
| `PipelineFailures` | A Step Functions execution failed | Read the failure notification (execution id, error) and the task log; fix the cause and start a run with the same `pipeline_id`, which resumes from its checkpoint ([Interrupted runs](#interrupted-runs)) |
| `PipelineTimedOutOrAborted` | An execution hit the state machine timeout or was stopped | A timeout: raise `sfn_timeout_hours` ([Capacity sizing](#capacity-sizing)) and re-run with the same `pipeline_id`. A stop nobody intended: check CloudTrail for `StopExecution`/`StopTask` |
| `IndexingFailures` | Indexing reported failed items, even in a succeeded execution | Find the failed writes in the task log (store errors, rejections), fix the store, and re-run with the same `pipeline_id` ([Stores](#stores-connection-permissions-requests)) |
| `ExtractionFailures` | Extraction, gleaning or claim extraction failed on some units | The graph is partial. Look for throttling or model errors in the log; the next incremental run retries the affected documents ([Failed documents](#failed-documents)) |
| `DocumentsQuarantined` | A run quarantined documents ([Quarantined documents](#quarantined-documents)) | Inspect the document; fix or remove it, then release it as that section describes |
| `DocumentsNotIndexed` | Source files failed to parse or load, or documents were recorded `FAILED` | Check the run summary's file and document counts and the warnings naming each file; fix the files, then run again ([Failed documents](#failed-documents)). Files no loader reads (a missing extra, an unsupported type) do not alarm: their count, `total_files_skipped`, is on the dashboard's "Document outcomes (EMF)" widget, so keep unsupported files out of the synced source prefix and watch that count after an image change |

The last four read the EMF metrics of `run-ingestion --metrics-sink cloudwatch`,
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
| `OpenSearchClusterYellow` | Multi-node only: a replica shard is unallocated for 5 minutes | Usually a node being replaced or a blue/green deployment; if it lasts, check `_cluster/allocation/explain`, since losing another node loses data |
| `OpenSearchIndexWritesBlocked` | The domain blocks index writes (5 minutes) | Free storage or memory (larger volume or instance), then re-run the failed ingestion; writes fail until the block lifts |
| `OpenSearchAutomatedSnapshotFailure` | An automated snapshot failed | Often a red cluster: fix that first. Until a snapshot succeeds the hourly restore point is missing ([Backup and restore](#backup-and-restore)) |
| `OpenSearchWriteRejections` | Write thread-pool rejections in 3 consecutive minutes | Rejected bulk items are retried, so indexing slows: more data nodes or a larger `opensearch_instance` |
| `OpenSearchSearchRejections` | Search thread-pool rejections in a minute | Searches were refused under load (the client retries a 429, then the query fails): fewer concurrent queries (`search.max_concurrency`, `run-eval`'s `processing.max_concurrency`) or more data nodes |
| `OpenSearchKmsKeyError`, `OpenSearchKmsKeyInaccessible` | `use_cmk` only: the data key is disabled, or deleted or its grants revoked | Re-enable the key. A deleted key cannot be recovered: restore a manual snapshot to a new domain |
| `DocStatusWriteThrottled` | The doc-status table throttled a write (5 minutes) | Registry calls retry with backoff; one still throttled fails the run with `DocStatusRegistryError`. Look for overlapping runs, rerun once it clears |
| `NeptuneHighCpu` | Average writer CPU at least 90% for 15 minutes | Sustained saturation: a larger `neptune_instance` (or `neptune_serverless_ncu` maximum), or, if you raised it, a lower `indexing.neptune.index_concurrency` (default `1`) |
| `NeptuneLowFreeableMemory` | Writer freeable memory at or below 512 MiB at any point in 5 minutes; on Serverless only while the writer is at maximum capacity | Out-of-memory failures ahead: a larger `neptune_instance` (or `neptune_serverless_ncu` maximum) |
| `NeptuneServerlessAtMaxCapacity` | Serverless only: the writer is at the `neptune_serverless_ncu` maximum for 3 consecutive 5-minute periods | Raise the maximum (or the minimum, so capacity is there before a burst) |
| `NeptuneGremlinErrors` | Gremlin server errors on any instance for 3 consecutive minutes | Persistent failures (retried write conflicts come in short bursts and do not fire it): find the failing request in the task log |

**Capacity**

| Alarm | Fires when | What to do |
|---|---|---|
| `BedrockThrottling` | At least 100 throttled calls in each of 3 consecutive 5-minute periods, summed over every model in the account and region (other Bedrock users count too) | Throttled calls are retried, so the run slows rather than fails: lower `processing.max_concurrency` or ask for a higher quota |
| `IngestionTaskMemory` | The ingestion task used at least 90% of its memory for 3 minutes (Container Insights) | Fargate kills a task over its memory, losing the running stage: raise `fargate_memory` ([Capacity sizing](#capacity-sizing)) |
| `IngestionTaskEphemeralStorage` | The ingestion task used at least 90% of its ephemeral storage (image included) for 5 minutes (Container Insights) | A full disk fails the running stage: raise `ingestion_ephemeral_storage_gib` (up to 200) |

**Queries**

| Alarm | Fires when | What to do |
|---|---|---|
| `QueryLatencyHigh` | The p90 latency of `run-rag --metrics-sink cloudwatch` queries is above `query_latency_alarm_seconds` (default 60) in 2 of 3 5-minute periods | The dashboard's per-strategy widget shows which strategy is slow; check Bedrock latency and throttles and OpenSearch search latency. `global` and `drift` make many model calls per query by design |

`run-rag` publishes `query_latency_ms` only with `--metrics-sink cloudwatch`
(`--mode rag` queries, failed ones included: a query that ran past
`search.query_timeout_seconds` records about the deadline, so timeouts raise
the p90 rather than vanishing from it), under `unified_kg_rag/query`
with the task's `Environment` and the requested `Strategy`. A serving layer
that embeds `GraphRAGChain` publishes nothing; emit the same metric from it
to use the alarm.

The dashboard also plots stage durations, task memory, CPU and disk, the
run's LLM tokens and calls by stage and its response-cache hits, Bedrock
calls, throttles and latency by model, query latency, the document outcomes
of each run (failed, skipped, excluded, deleted, retrying, without claims),
and the incremental degradations no alarm watches
(`total_documents_deferred`, `community_rebuild_recommended`,
`community_growth_ratio`, `runs_since_clustering`,
`total_records_not_stored`, `underived_items`).

Corpus coverage metrics, one value per run (Sum):

| Metric | Emitted after | Counts |
|---|---|---|
| `total_files_failed` | `document_loading` | Corpus files that failed to parse or load (kept, not deleted) |
| `total_files_skipped` | `document_parsing` | Corpus files of a type no loader reads (kept, not deleted) |
| `total_documents_excluded` | `document_loading` | Changed documents no longer retried (`indexing.max_document_failures`) |
| `total_documents_deleted` | `indexing` | Documents removed as gone from the corpus or of a retired scope |
| `total_documents_failed` | run end | Documents recorded `FAILED` (JSON `failed_documents`) |
| `total_documents_retrying` | run end | Documents failed again for an error not their own (JSON `documents_retrying`) |
| `total_documents_without_claims` | run end | Documents indexed without some claims (JSON `documents_without_claims`) |
| `total_units_failed` | run end | Text units a stage failed on, full builds included (JSON `units_failed`) |

Model usage metrics, emitted right after each stage that called a model or
was served from the LLM response cache (Sum; a model-free stage emits none,
so it creates no metric series). They come from the app's own usage
accounting, tracing on or off, and count only this deployment's runs, unlike
the account-wide Bedrock widgets; `run-ingestion --output-format json` has the
same counts in `token_usage`:

| Metric | Counts |
|---|---|
| `llm_calls_<stage>` | Model calls that returned (retries included, failed calls not) |
| `llm_input_tokens_<stage>` | Input tokens the responses reported, prompt-cache reads and writes included |
| `llm_output_tokens_<stage>` | Output tokens the responses reported |
| `llm_cache_hits` | Responses the run's LLM response cache served instead of a call (a resumed stage), chunk-boundary and community report answers the extraction store served, and description summaries a full build took from their stored copies, summed over the stages |

To check them against Bedrock's own counts, compare a run's sums with the
metrics of the `AWS/Bedrock` namespace for its models over the run's window
(other callers of the same models in the account and region count there
too): `llm_output_tokens_<stage>` with `OutputTokenCount`, and
`llm_input_tokens_<stage>` with `InputTokenCount` +
`CacheReadInputTokenCount` + `CacheWriteInputTokenCount`. The app's input
tokens include the prompt-cache reads and writes, which Bedrock reports
apart from `InputTokenCount` (`token_usage` has them apart too:
`cache_read_input_tokens`, `cache_write_input_tokens`), so with prompt
caching on `InputTokenCount` alone falls well short: about half of the
input on a run whose calls share a long prompt prefix. Calls compare less
exactly. `Invocations` counts every request Bedrock accepted, those that
later failed included, and not throttled ones, while `llm_calls` counts
each returned call once: a call botocore resent after a read timeout or a
server error is one call but several invocations (`request_retries` in
`token_usage`), a call that failed in the end is in `failed_calls` only,
and calls made outside a usage scope (the `run-eval` warm-up queries,
scripts) are not counted. In the package's November 2026 benchmark runs
Bedrock counted 13% more Sonnet 5.5 invocations than recorded calls while
the tokens agreed within 4.6% (input, cache included) and 0.9% (output),
which points to resent and failed requests that spent few tokens; those
runs predate `request_retries` and `failed_calls`, so the split was not
measured. Haiku 5.5 agreed within 0.2%.

## Query deadlines and model timeouts

A query has two bounds ([User Guide](./user-guide.md#query-deadline)):

- **The query deadline**, `search.query_timeout_seconds` (default 180 s):
  everything from the input guardrail to the saved answer. Past it the query
  raises `QueryTimeoutError` (or, under `processing.ignore_errors`, returns
  the error answer) and its asyncio tasks are cancelled. A Bedrock call
  already sent from a worker thread cannot be interrupted: it runs on
  through its botocore attempts (at most `model_max_attempts x
  model_timeout_seconds`), the framework's transient retries start no new
  call past the deadline, and `run-rag` exits without waiting for it.
- **The query's model clients**: `search.model_timeout_seconds` (default
  90 s) per request and `search.model_max_attempts` (default 2) botocore
  attempts, for query-time LLM calls, query embeddings and Bedrock rerank. A
  stalled call fails after about `model_max_attempts x model_timeout_seconds`.
  Before these, queries used the ingestion client (330 s, 6 attempts): one
  stalled call held a query for about 33 minutes. A stream must also send its
  first chunk within `model_timeout_seconds`.

Ingestion is unchanged: its calls keep the 330 s read timeout and 6 botocore
attempts, bounded by `BatchProcessor`'s 300 s call timeout.

**Tuning.**

| Symptom | Change |
|---|---|
| Answers that need long generation (high `aws.bedrock.default_effort`, long `default_max_output_tokens` answers) fail with a read timeout | Raise `search.model_timeout_seconds` above the longest answer call, and `search.query_timeout_seconds` to at least about twice it |
| `drift`, or `global` over many reports, fails with `QueryTimeoutError` | Raise `search.query_timeout_seconds` (DRIFT makes up to 27 LLM calls); if those calls are throttled, lower `search.max_concurrency` instead |
| `drift` runs with the comprehensive answer (`search.answer_prompt_style: {drift: comprehensive}`) | Raise `search.query_timeout_seconds` to about 300 s first (see *DRIFT near the deadline* below) |
| A serving layer must answer or fail fast | Lower `search.query_timeout_seconds` to its own request timeout, minus the time it needs to respond |
| `run-eval` reports queries failed with `QueryTimeoutError` | Throttling, usually: see below |

**Why these defaults.** In the package's real-AWS evaluation runs (5,846
queries) the p99 query took 47 s and the slowest 61 s, in all. A
non-streamed Converse answer at `default_effort: high` can spend most of
that in one read, so `model_timeout_seconds` is 90 s: about 1.5x the
slowest whole query and 1.9x the p99, for one request. The deadline,
180 s, is `model_max_attempts x model_timeout_seconds`, so a call that
stalls once can still be answered by its second attempt; it is about 3x
the slowest query and 3.8x the p99. Two attempts, not more: a third
would start only after the deadline.

**DRIFT near the deadline.** Those runs did not include DRIFT. On a
3,318-document MuSiQue index (embedded SQLite stores, 50 sensemaking
questions, two repeats, one run-eval at a time) DRIFT with the concise answer
took 147 s at p95; with the comprehensive answer 168 s at p95 and 178 s at
most, 2 s inside the 180 s deadline. A second evaluation running at the same
time pushed 20 of 50 comprehensive DRIFT answers past it. So DRIFT keeps
the concise answer by default, and a deployment that routes to DRIFT
(`search.auto_routable_strategies`), serves it under load or gives it the
comprehensive answer should set `search.query_timeout_seconds: 300` (with
`model_timeout_seconds` 90 s and two attempts, a stalled call still fits).
The deadline is one setting for every strategy: a per-strategy deadline is
not offered, because an `auto` query learns its strategy only after the
router call, inside the deadline that has already started, and the streamed
path fixes its deadline before the first chunk. A service that wants a
short deadline for `local`/`mix` and a long one for `drift` runs two chains
with two configurations (or two `run-rag` invocations) and sends each
strategy to its own.

**With the Bedrock quota.** Bedrock reserves input plus `max_tokens` of each
request against the model's tokens-per-minute (TPM) quota when the request
starts, and throttles requests past it. A throttled query call is retried by
botocore's adaptive mode within `search.model_max_attempts`, then with
backoff by the query-chain retry (`retry.max_attempts`,
`retry.max_total_seconds`), and every retry waits inside the query deadline.
So a quota too small for the query load shows up as `QueryTimeoutError`
rather than as slower answers: `run-eval` keeps `processing.max_concurrency`
queries in flight, each with up to `search.max_concurrency` model calls (the
global map step, DRIFT). Lower `processing.max_concurrency` for the
evaluation, lower `aws.bedrock.default_max_output_tokens` (it is reserved
whether used or not; the short query calls reserve `search.output_ceilings`
instead), or ask for a higher quota; raising the deadline only
lets queries queue longer. The `BedrockThrottling` alarm and the dashboard's
throttle widget show whether this is the cause.

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
serving layer of your own should assume a role with the same grants. Under
fine-grained access control, give it `indices:data/read/search`,
`indices:admin/mappings/get` (the embedding-model check) and
`indices:monitor/settings/get` (searches sized by the live index's
`max_result_window` / `max_terms_count`) on the package's indices; without
the last two a query still runs and logs one warning for each.

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
| `kg_rag.batch` | stage, or root for `run-eval` answer generation | Item and failure counts; in `run-eval` the run's `kg_rag.eval.strategy` and `kg_rag.eval.repeat` (1-based), as on every span of the run |
| `kg_rag.query` | root, or the `run-eval` batch | `session.id`, `user.id`, `kg_rag.blocked`; with `capture_content` the redacted query as `input.value` and (outside `stream` / `astream`) the answer as `output.value` |
| `kg_rag.strategy` | query | `kg_rag.rerank_status` for the whole query |
| `kg_rag.retrieve` | strategy | Hits per result type, up to 50 source ids |
| `kg_rag.fusion` | strategy | In/out counts, the step's `kg_rag.rerank_status` |
| `kg_rag.rerank` | fusion | `kg_rag.rerank.status` (`ok`, `empty`, `error`), document counts |
| `kg_rag.context_build` | query (`drift`: the round's actions span) | Tokens used vs the token budget |
| `kg_rag.global.select`, `kg_rag.global.map` | strategy (`global`) | Select, once per query: `kg_rag.depth` (configured), `kg_rag.cut` (the cut mapped after falling back to a shallower one), `kg_rag.reports`, `kg_rag.dynamic`. Map: `kg_rag.reports`, `kg_rag.map_calls`, `kg_rag.report_tokens` |
| `kg_rag.drift.primer`, `kg_rag.drift.actions` | strategy (`drift`) | Primer: `kg_rag.reports`, `kg_rag.answers`. One actions span per round (`kg_rag.depth`, `kg_rag.actions`), with the round's local searches under it |
| `kg_rag.guardrail` | query | One per screened query or answer: `aws.bedrock.guardrail.id`, `.source` (`input`, `output`), `.action` (`none`, `masked`, `blocked`), `.intervened`, `.policies`; with the contextual grounding check, `aws.bedrock.guardrail.grounding.checked`, `.grounding.failed`, `.grounding.score`, `aws.bedrock.guardrail.relevance.score` |
| `kg_rag.evaluate` | root | One per evaluator: `kg_rag.evaluator`, `kg_rag.items`, `kg_rag.eval.strategy`, `kg_rag.eval.repeat`; the judge's model spans under it |
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
- A model call cancelled during `ainvoke` (its task cancelled, or the call cut
  off by `asyncio.wait_for` / `asyncio.timeout`, as RAGAS's `ragas_timeout`
  does) gets no LangChain callback; its model span is ended as `ERROR`
  (`error.type: CancelledError`) when the task that started it ends. A stream
  the caller stops reading early ends its span normally.
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
  only: inject them as ECS task secrets. On the CDK stacks, store them as
  JSON keys of one Secrets Manager secret and deploy with
  `-c observability_secret_arn=<complete secret ARN>` and, for keys other
  than the two Langfuse ones,
  `-c observability_secret_keys=OTEL_EXPORTER_OTLP_HEADERS,...` (in
  `private` mode the stacks then add a Secrets Manager interface endpoint,
  which a reused VPC must provide, or the tasks fail to start; it reaches
  the deploy region only, so synth refuses a secret of another region); the
  non-secret settings (`LANGFUSE_HOST`, `UKR__OBSERVABILITY__*`) go in the
  image config or the run's container overrides. The start-up line
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
| `ConfigurationError: Unknown config keys: '<path>' ...` (exit 2) | A typo, or a key dropped without a replacement (a renamed key fails with `settings moved; replace <old> -> <new>`; a removed one, for example `indexing.cross_run_merge` or `aws.bedrock.guardrail.apply_to`, only logs `Config key ... is removed and ignored`; the configuration reference lists the [moved](./configuration.md#moved-keys) and [removed](./configuration.md#removed-keys) keys) | Fix or delete the key; `--allow-unknown-keys` loads anyway with a WARNING per key |
| Usage errors (exit 2): `--source-directory is required for pipeline execution`, `--s3-sync needs --s3-bucket-name or aws.s3.bucket_name`, `--pipeline-id is required for --resume-from-stage`, `--enabled-stages: invalid stage names ...`, `argument --resume-from-stage: Unknown pipeline stage ...` | A required flag is missing, or a stage name is unknown | Pass the flag, or a stage name from the message ([CLI Reference](./cli.md#run-ingestion)) |
| `... needs confirmation, but stdin is not a terminal` | Completed stages' outputs are no longer cached under the current inputs | `--yes` (recomputes them), or a new `--pipeline-id` |
| `Stage '<stage>', upstream of this run's enabled stages, has not completed` | An earlier stage failed or lost its output | Run the earlier stages first |
| `observability.enabled is true but OpenTelemetry is not installed` | The `otel` extra is missing | `uv sync --extra otel`, or turn tracing off |
| `Skipping N file(s) in '<dir>' of a type no loader reads (2 '.md', ...)` / `No supported source files found in '<dir>'` | The `unstructured` extra is not installed (`.md`, `.markdown`, `.htm`, `.html`), or no loader is registered for the type. The files are not indexed; an incremental run keeps what they indexed before and lists them as `failed` (`total_files_skipped`) | Install the extra the message names (rebuild the image with it), or convert the files. Until then the skipped files' indexed content stays as it was, edits to them included |
| `Not installed: 'rdflib' (for processing.graph_extraction.schema_source.kind 'owl')` (exit 2) | The `ontology` extra is missing | `uv sync --extra ontology`, or build the image with it ([Ontology schema in the image](#ontology-schema-in-the-image)) |
| `Ontology file '...' is OWL/XML ...` / `... names a remote JSON-LD context ...` / `... more than max_entity_types ...` (exit 2) | A format the `owl` source does not read, or a selection over a cap | Follow the message: convert the file, inline the context, or narrow the selection ([User Guide](./user-guide.md#from-an-owlrdfs-ontology)) |
| `Ontology file '...' holds ... (MAX_XML_TEXT_CHUNKS)` / `... (MAX_XML_LITERAL_ELEMENTS)` / `... has a line of N bytes ... (MAX_LINE_BYTES)` / `... limit on input amplification factor ...` (exit 2) | A shape the RDF/XML or N-Triples parser takes quadratic time over (refused before parsing) | Convert the file to Turtle (`robot convert --input onto.rdf --output onto.ttl`) |
| `Ontology IRIs with whitespace: ...` / `Ontology rdf:List is cyclic ...` (exit 2) | An IRI the RDF/XML or JSON-LD parser let through, or a malformed list | Fix the named IRI (`%20` for a space), or end the list with `rdf:nil` |
| `SeedGraphError: seed graph '<name>': '<file>' line N: ...` (`duplicate node id ...`, `property ... is not a string, number, boolean or null`, `not valid JSON`, `"kind" is ...`) (exit 2) | A seed record the import refuses | Fix the named line ([User Guide: Seed graphs](./user-guide.md#d-seed-graphs-bring-your-own-knowledge-graph)) |
| `seed graph '<name>': edges name unknown nodes ...` / `... the edge from ... has no type ...` (exit 2) | An edge to a node the seed does not state, or with neither a `type` nor a `predicate_iri` a relationship type declares | Add the node, set `options.dangling: skip`, or type the edge |
| `seed graph '<name>': N names are shared by distinct nodes (ids ...); set on_name_collision to merge or qualify` (exit 2) | Two nodes of the seed are one entity by name | Rename them, or set `on_name_collision` |
| `seed graph '<name>': ... more nodes than max_nodes (N)` (or edges, `max_edges`) / `... exceed max_property_keys ...` / `... more than max_bytes ...` / `shard N would hold M text units, over 2000 ...` (exit 2) | The seed is over a cap (nothing is truncated) | Raise the cap, or for a shard `shards` or `max_items_per_unit` |
| `Unknown processing.seed_graphs[].kind '<kind>'; registered: csv, jsonl, rdf` / `Not installed: 'rdflib' (for processing.seed_graphs '<name>' kind 'rdf')` (exit 2) | A kind no source registers (a custom source starts from Python after importing its module), or the `ontology` extra missing | Fix the kind, or install the extra ([Seed graphs in the image or S3](#seed-graphs-in-the-image-or-s3)) |
| `Access to seed source '<path>' denied (...)` / `Seed source '<path>' not found (no such S3 object)` (exit 2) | The ingestion task role cannot read the seed | Grant `s3:GetObject` (and `s3:ListBucket`), or move the seed ([Seed graphs in the image or S3](#seed-graphs-in-the-image-or-s3)) |

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
| `StoreConnectionError: ... Cannot connect to host ...` / `Neptune is not reachable ...` / `... status=429 ... (still failing after the client's retries)` | Refused or unresolved, closed by Neptune (restart, failover; reopened next time), or OpenSearch 429/502/503/504 for all `retry.query_store_max_attempts` attempts. An unreachable vector store fails `run-ingestion` at its store check, before any model call | Check endpoint, port, that the store runs, security groups and VPC endpoints |
| `OpenSearch did not answer a request on <index> within N s in M attempt(s)` / `Neptune did not answer within N s` | No answer within `search.store_timeout_seconds` (ingestion: Neptune fixed 180 s, item retried) | Check the store's health and load, or raise the timeout |
| `OpenSearch denied the search on <index>: ... status=403 ...` (or `401`) | Fine-grained access control: the role lacks `indices:data/read/search`; or bad credentials | Grant it (and `indices:admin/mappings/get`, `indices:monitor/settings/get`), or fix the credentials |
| `Neptune refused the connection (HTTP 403)` | No `neptune-db:connect` on the cluster, or signed for another region | Grant it (and the needed `neptune-db:*` data actions); set `aws.region_name` to the cluster's region |
| OpenSearch authentication fails with `use_iam: false` | No basic-auth credentials | Set `UKR__AWS__OPENSEARCH__USERNAME` / `__PASSWORD` (for example in `.env`) |
| `OpenSearch rejected the search on <index>: ...` | 400 `query_shard_exception`: a filter value of the wrong type. 500 `too_many_clauses` / `too_many_nested_clauses`: `indices.query.bool.max_clause_count` below 1,024 | Fix the value, or keep the limit at 1,024 or more |
| `Cannot create the OpenSearch index '<index>': the cluster has no analyzer '<name>'` (`mapper_parsing_exception`) | An analyzer in `indexing.opensearch.language_analyzers` is missing, typically `ko: nori` without the Nori package | Use one the cluster has (`ko: cjk`, the CDK default), or associate the package ([Korean with Nori](#korean-with-nori)); re-run |
| `The search engine refused to create an index: ...` (exit 2, before any model call) | Before its first model call, `run-ingestion` creates and deletes a probe index per index the run will create (all of them on a full build or reset, the missing ones on an incremental run), named `<extraction-records alias>-settings-probe-<12 hex digits>`; the store refused one (the error names the index it stood in for, `(a probe of '<alias>')`): an `indexing.opensearch.index_settings` key, the `vector_search` method, `vector_mode` / `compression_level` (`unknown parameter [mode]`: OpenSearch before 2.17) or an analyzer, or (`maximum shards open`) no room for the new indices' shards | Fix or remove the named setting, or delete unneeded indices or raise `cluster.max_shards_per_node`; re-run. A probe a stopped run left (an unaliased index of that name) is deleted by the namespace's next run |

### Ingestion runs and the doc-status registry

| Message (abridged) | Meaning | Action |
|---|---|---|
| `Pipeline failed at stage(s): ...` | A stage failed; exit 1 | Fix from the stage's log and resume with the same `--pipeline-id` (`--force-rebuild` starts clean) |
| `Pipeline resume failed: ...` (`PipelineResumeError`, exit 1, before any model call) | The pipeline id cannot be resumed as asked: a stage this run must rerun is not enabled (the message names it), an explicit `--resume-from-stage` past a stage whose cached output is gone, or its saved state could not be read | Enable the named stage or resume from it (`--resume-from-stage <stage>`); for unreadable state, a new `--pipeline-id` or `--force-rebuild` |
| `CacheSyncError` | The S3 stage-cache download or upload failed | Check bucket permissions and `aws.s3.encryption`; resume |
| `DocStatusRegistryError` | The registry table is missing, not accessible, or throttled | Create it or fix `aws.dynamodb.table_name`, grant the permissions the message names; resume |
| `The reset of index namespace '...' drops the seed graph documents of source scope(s) ...` (or `rebuilds it from this run's seed graphs alone ...`, or, with the registry off, `No source document reached this run, ... rebuild the index '...' from the seed graphs alone, replacing what it holds`) `; continue?` with `... needs confirmation, but stdin is not a terminal` or `No one can confirm it (run-ingestion --confirm-reset does)` (exit 1, before any write) | A reset would drop another scope's seed graph documents, or, from seeds alone, another scope's documents; or, with the registry off, seeds alone would replace an index that holds items; and no one answered yes | Re-run with `--confirm-reset` or `UKR_CONFIRM_RESET=1` ([Seed graphs in the image or S3](#seed-graphs-in-the-image-or-s3)), or pass `confirm=` to `DataIngestionPipeline` (a no raises `ConfirmationDeclinedError`) |
| `This run deletes N of the M documents the doc-status registry holds for source scope '...' (P%, over indexing.max_deletion_share S) ...; continue?` with `... needs confirmation, but stdin is not a terminal` or `No one can confirm it (run-ingestion --confirm-deletions does)` (exit 1, at `document_loading`, before any write) | More than `indexing.max_deletion_share` of the scope's indexed documents are missing from the corpus | Check the corpus (sync, prefix, mount) and re-run. If the files are really gone, re-run with `--confirm-deletions` (or `UKR_CONFIRM_DELETIONS=1`, or `--yes`) ([Incremental runs](#incremental-runs)) |
| `No source document reached this run, only its seed graphs, but the doc-status registry holds N documents of source scope '...'` (exit 1, at `document_loading`, before any write) | The corpus is empty (a volume not mounted, an S3 sync that copied nothing) while seeds are configured and the scope has documents | Put the corpus in place and re-run. To delete the scope's documents, retire it from a run of another scope (`--retire-source-scope`); for seeds of their own, use a seed-only scope (User Guide, *Lifecycle*) |
| `Incremental indexing is enabled but no document delta was computed` | Indexing ran without a registry diff (loading failed under `continue_on_error`) | Re-run from `document_loading` once the registry is reachable |
| `NamespaceLeaseError: Index namespace '...' is being written by ...` | Another run holds the write lease | Wait; a dead run's lease expires within `indexing.lease_seconds` ([Concurrent runs](#concurrent-runs)) |
| `NamespaceLeaseError: This run lost the write lease ...` | Renewals failed until the lease expired, or another run took it | Re-run after the other run; check the registry is reachable |
| `ConfigurationError: Doc-status registry: DynamoDB table '...' has the key schema ... of an earlier release` | A table keyed by `doc_id` alone | Point `aws.dynamodb.table_name` at a new table (or the stack's), and run once with `indexing.reset: true` ([Incremental runs](#incremental-runs)) |
| `ConfigurationError: The OpenSearch index '...' records no embedding settings ...` | An earlier release built the index | Run once with `indexing.reset: true` |
| `ConfigurationError: The OpenSearch index '...' maps 'attributes.seed_properties' field by field ...` (exit 2, before any model call) | A build before seed properties were left unmapped wrote seed graph properties into the live entities index | Run once with `indexing.reset: true` |
| `ConfigurationError: The OpenSearch index '...' stores the vectors of '...' as ...` (exit 2, before any model call) | `indexing.opensearch.vector_mode` / `compression_level` changed since the live index was built | Run once with `indexing.reset: true`, or restore the previous values ([On-disk vector storage](#on-disk-vector-storage)) |
| `ConfigurationError: The OpenSearch index '...' holds embeddings of ...` / `... was built for other language settings ...` (exit 2, before any model call) | An incremental run's embedding model or dimension, translation languages or `language_analyzers` differ from those the live index was built with (also a run with nothing to extract: no change, or only deletions) | Run once with `indexing.reset: true` (it re-embeds or re-translates the whole corpus once), or restore the previous settings |
| `The doc-status registry lists indexed documents of this namespace whose entities the graph store does not hold` (or `whose text units the vector store does not hold`) | A store lost what the registry records (a restarted Gremlin Server, `down -v`, a restored graph, deleted indices, a moved `vector.sqlite3`), or, in an index a pre-#383 build wrote, a deleted document was restored while its removal was unfinished, or a reset run by a build before #410 that dropped documents was interrupted after it swapped an index in | Re-run with `indexing.reset: true`, or delete the namespace's records; locally delete the three store files together or restore the moved one. For the restored document, deleting that one document's registry record is enough: the next run indexes it again |

### Queries

| Message (abridged) | Meaning | Action |
|---|---|---|
| `No indices found for suffix '<suffix>' ...` | Nothing was ingested under that suffix | Ingest, or query the corpus's suffix |
| `QueryTimeoutError: The query did not finish within N s (search.query_timeout_seconds)` / `The answer stream sent no first chunk within N s (search.model_timeout_seconds)` | The query, or a stream's first chunk, ran past its deadline: a stalled or throttled model call, a slow store, or a long strategy | [Query deadlines and model timeouts](#query-deadlines-and-model-timeouts) |
| `Read timeout on endpoint URL ...` (`ReadTimeoutError`) on a query | A query-time Bedrock request got no answer within `search.model_timeout_seconds`, on every attempt | Check Bedrock's health and the VPC endpoint; raise `search.model_timeout_seconds` if answers are legitimately long |
| `InvalidFilterError: Unknown filter key(s) ...` / `... has no value (null)` / `... not a scalar` / `... lists N values ...` | No store the strategy reads declares the filter key; a filter value is null (`None` from Python; alone, in a list or as a range bound: OpenSearch rejects such a query); a list holds a list or mapping; or a list filter holds more values than one `terms` query takes (`index.max_terms_count`, default 65,536) | Use a key the message lists; drop the null or give the value to match; list strings, numbers or booleans; narrow the list (or raise `max_terms_count` in `indexing.opensearch.index_settings` and reindex) |
| LightRAG `mix` / `hybrid` returns nothing | No relationship index, or no keywords extracted | Check the relationships index exists; short queries fall back to the raw query only under `raw_query_fallback_max_len` |

### Local mode

| Message (abridged) | Meaning | Action |
|---|---|---|
| `... stayed locked ...` (`database is locked`) | Another process held the SQLite write lock longer than `search.store_timeout_seconds` | One ingestion per store file at a time, or raise the timeout; use a local disk |
| `... processed N inputs but produced 0 outputs`, with `Connection refused` in the log | Ollama or the OpenAI-compatible server is not reachable at `providers.llm.base_url` / `providers.embedding.base_url` | Start it (`ollama serve`, or `docker compose -f docker/compose.local.yaml --profile models up -d --wait`), check `base_url` (`--model-base-url`), pull the models |
| `AuthenticationError` / `NotFoundError` (401 / 404), `APIConnectionError` / `APITimeoutError`, `ResponseError: model ... not found` | Key refused, model not served, or server unreachable | Check `base_url` and `api_key` under `providers.llm` / `providers.embedding` and the model names (`models.default`, `models.embedding.model_id`); pull the models |
