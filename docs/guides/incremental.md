# Incremental indexing

> 🇰🇷 한국어판: [incremental.ko.md](./incremental.ko.md)

Incremental (delta) indexing re-indexes only the documents that are **new or
changed** since the last run and merges them into the live graph, instead of
rebuilding everything; a deleted document takes along what only its text
units stated. Which documents a run compares, and may delete, follows from its
index suffix and source scope
([Scopes and namespaces](../concepts/scopes-and-namespaces.md)). How it works
inside: [Design §5](../design/incremental.md#5-incremental-indexing). Resuming a run,
recovering an interrupted one and the write lease are in the
[Operator Runbook](../operations.md#incremental-runs).

## Enable it

```yaml
providers:
  doc_status:
    enabled: true          # kind: dynamodb (the default) or sqlite
aws:
  dynamodb:
    table_name: "unified-kg-rag-on-aws-doc-status-v2"
    create_table_if_missing: true
```

With this on (the container image sets it), each `run-ingestion` diffs the
corpus against the DynamoDB document-status registry by **content hash**. In
local mode the registry is a SQLite file instead
(`providers.doc_status.kind: sqlite`; see [Local mode](./local-mode.md)).

The registry is partitioned by index namespace (the `<namespace>#` prefix of
each document key), so a run reads only its own namespace's records. A table
of the earlier `doc_id`-only key schema is rejected at first use: switch to a
new table and run once with `indexing.reset: true`. A registry the run cannot
read stops it before anything is indexed
([Operator Runbook: Incremental runs](../operations.md#incremental-runs)).

## Workflows

- **Add a document:** drop it into the source directory and re-run
  `run-ingestion`. Only the new file is parsed, extracted and indexed; its
  entities and relationships merge into the existing graph.
- **Modify a document:** edit it and re-run. Its old artifacts are removed and
  the new version is indexed. Only its changed chunks go to the model: the
  extraction store (`indexing.opensearch.extraction_records_index_prefix`)
  serves every chunk whose text and extraction settings it already holds a
  record for, in any document (`--force-rebuild` re-extracts them all).
- **Delete a document:** remove it and re-run. What only its chunks stated
  (entities, relationships, claims) is deleted; shared artifacts are kept
  without its chunks, and a community or report goes once none of its chunks
  is left.
- **Change the extraction settings** (a tuned `custom_prompts` set,
  `entity_types`, an extraction, gleaning or chunking model, chunk sizes,
  translation, resolution): the next run re-extracts the unchanged documents
  extracted with other settings and logs a WARNING with their count and the
  settings that changed. Settings are compared field by field, so a setting a
  newer release adds is no change while it holds its default. A re-extracted
  chunk still takes its stored record when the settings that shape a record
  (models, extraction, gleaning and claim settings, the target language, the
  prompt templates) are unchanged, e.g. after a resolution or chunking change.
  Edits to the built-in prompt templates (code) are not seen: reset after
  editing one. A description summary setting (`description_summarization`,
  its `custom_prompts` or `models.roles.description_summary`) is not an
  extraction setting: changing it re-extracts nothing. The descriptions a
  run merges are summarized the new way, and `indexing.reset: true`
  summarizes every one again (a stored summary records the settings it was
  made with and is reused only under the same ones).
- **Change the embedding model or its dimension**
  (`models.embedding.model_id` / `dimension`): each OpenSearch index records
  the model and dimension it was built with, and an incremental run stops
  before its first model call (exit 2) when they differ from the run's (an
  index that records none is refused too). Re-run once with
  `indexing.reset: true`. `run-rag` and `run-eval` with another model or
  dimension fail with a `ConfigurationError` instead of answering from empty
  search results.
- **Change the corpus or translation languages** (`processing.translation`
  languages, `indexing.opensearch.language_analyzers`): live text fields keep
  their analyzers, so an incremental run stops before its first model call
  (exit 2); re-run once with `indexing.reset: true`. Dropping a language does
  not stop it.
- **Index a suffix with the registry off** (as the `backend: local` preset
  does): a later incremental run on that suffix stops before its first model
  call. Re-run once with `indexing.reset: true` and the registry on, or keep
  it off for that suffix.

## Rebuild (`indexing.reset`)

Set `indexing.reset: true` for one run to index the whole corpus again
without taking the suffix offline. Quarantined documents are skipped and keep
their records. A failed reset drops nothing, so it is safe to re-run. **Set
`indexing.reset` back to `false` afterwards**, or every run rebuilds.

During a reset, vector queries read the previous indices until each new one
swaps in, and graph queries see previous and rebuilt items mixed until the
final drop. A reset whose graph writes partly failed (even within
`indexing.max_failure_rate`) fails too (`graph_sweep` in the indexing
results), and the next run removes what the previous graph still holds. One
stopped or killed before it swapped an index in leaves that index
(`<alias>-<timestamp>`, no alias) until the next reset deletes it (an
incremental run does not): delete it by hand, or re-run the reset. A reset
pipeline id run again resumes from its cached stages. When another pipeline
id has since deleted extraction records its text units name (an incremental
run with other settings deletes the records of the units it re-extracts), it
resumes from the stage that stores them (WARNING `Extraction records this
pipeline stored are gone ...`) and extracts those units again: the LLM
response cache answers them, and with `cache.llm_responses.enabled: false`
the rerun costs what the reset did.

A reset reuses the stored chunk boundaries, translations, extraction records
and vectors of unchanged text, and the stored description summaries and community report
answers whose input is unchanged. A rebuild with the same models and settings
therefore asks the model only for what the corpus changed, plus the reports
of the communities it clusters differently from the incremental runs before
it (an incremental run clusters only its delta). After a summarization or
report setting change it asks again for every summary or report, and after an
embedding model or dimension change it embeds every text again. Add
`--force-rebuild` to ask and embed everything again.

A reset re-clusters the whole graph, which replaces the communities the
incremental runs appended, and gives the stored communities and reports the
current ids of their entities ([Cross-run merge](#cross-run-merge)).

A run that adds a methodology the namespace does not hold (`graphrag` to a
`[lightrag]` namespace: communities and reports; `lightrag` where the
relationship vector index was off), or turns claim extraction on for a
namespace without claims, is a reset by itself, for that run only:
a delta would build them for its own documents only. It logs a WARNING and
reuses what a reset reuses, so its new model calls are the added
methodology's. When the namespace also holds another source scope's
documents, which the rebuild would delete, the run is refused before any
model call instead ([Methodology profiles](./ingest.md#methodology-profiles)).
An incremental run with claim extraction off keeps the claim records of the
units it extracts again, as a reset does, so turning claims back on asks
the model for none of them.

## Rebuild signals

An incremental run clusters only its delta and appends its communities, and
stored communities and reports keep an entity's old id after it took its
current name's id. Two signals tell you when a rebuild is due. Neither resets
anything on its own.

### Community drift (per run)

The indexing stage reports `community_drift` metrics: `delta_communities`,
`clustered_entities`, `overlapping_entities`, `stale_communities`,
`added_roots` (each adds a report to every global-search cut) and
`overlap_share`. When at least `indexing.community_drift_threshold`
(default `0.5`) of the entities the run clustered also belong to an earlier
level-0 community, the run logs a WARNING and reports
`community_rebuild_recommended: 1` (the counts are an upper bound). A run that
removes text units also reports `outdated_reports`, stored reports that still
cite a removed unit and may state what was removed: any such report
recommends a rebuild too. `unclustered_entities` counts stored entities
without a community that a small delta component the run clustered links to:
a full build clusters them, a delta run clusters only its delta.

### Community growth (across runs)

Each suffix's build record (the `build` entry of its vector indices' `_meta`,
which `run-eval` manifests copy) keeps community counters: `clustered_at` and
`communities_at_clustering` from the last full build, then `communities_now`
(each incremental commit adds the communities it appended and subtracts those
its removals deleted) and `runs_since` (incremental commits since). A commit
is counted once: a resumed run that writes the record again does not count it
twice. After a commit that reaches
`indexing.community_compaction.max_runs_since_clustering` (default `20`) or
`max_community_growth` (default `1.5` times the clustered communities), the
run logs a WARNING:

```
Communities of 'default' are due for re-clustering: 24 incremental runs since
the last full clustering (max_runs_since_clustering: 20); 163 communities
against the 10 it clustered, 16.3x (max_community_growth: 1.5). ...
```

The `community_growth_ratio` and `runs_since_clustering` metrics carry the
same numbers (the largest over the run's suffixes; `0` for a run that
recorded no counters). `null` turns a threshold off;
`on_drift_recommendation: true` also gives this recommendation when the run's
`community_rebuild_recommended` is 1. `communities_now` is an upper bound (a
delta community written under a stored community's id counts again, and a run
killed after its removal deleted communities does not subtract them). A
suffix whose build record holds no counters gets none, and no
recommendation, until its next full build.

### When to reset

Either signal means global search now maps the appended communities' reports
next to the original ones, and their reports miss what later runs added (or,
for `outdated_reports`, may still state what was removed). Run once with
`indexing.reset: true` ([Rebuild](#rebuild-indexingreset)): it asks the model
mostly for the communities it clusters differently
([measured](../concepts/defaults-and-evidence.md#reset-cost-after-incremental-runs)),
and the counters start again at that build. A reset is needed too after the
setting changes in [Workflows](#workflows) that stop an incremental run. The
CDK dashboard plots `community_rebuild_recommended`, `community_growth_ratio`
and `runs_since_clustering`; no alarm watches them
([Operator Runbook: Alarms](../operations.md#alarms)).

## Deletion scope

A run only diffs, and treats as deleted, the documents of its own scope: its
index suffix and its corpus source
([Scopes and namespaces: Source scope](../concepts/scopes-and-namespaces.md#source-scope)).
From Python, `DataIngestionPipeline.insert_documents` and `delete_documents`
run documents given from memory as named changes instead: what a call
names is added, updated or deleted, and the scope's other documents are
kept ([Python library: Documents from memory](./python-library.md#documents-from-memory)).

### Keep the corpus in place

- **Keep a corpus at one source** (or a fixed `source_scope`). A moved corpus
  with the default scope leaves the old scope's content indexed
  ([Moving a corpus](#moving-a-corpus)).
- **Never let corpus files expire or disappear**: the next run removes their
  content. A file the run cannot read keeps it: one that fails to parse or
  load, and one skipped because no loader reads its type (an optional extra
  such as `docx` not installed or dropped from the image, a loader
  plugin not imported; `Skipping N file(s) ... of a type no loader reads ...`,
  `total_files_skipped`). Such a file is reported as `failed` and keeps its
  indexed content until a run reads it again. On the CDK stack, keep the
  corpus out of the expiring `pipeline-runs/` prefix
  ([Caches and prefixes](../operations.md#caches-and-prefixes)).
- **One index suffix and config file per tenant or corpus version**
  ([Scopes and namespaces: Index suffix](../concepts/scopes-and-namespaces.md#index-suffix)).

### Mass deletions

A run that would delete more than `indexing.max_deletion_share` (default
`0.5`) of the documents the registry holds for its scope stops at
`document_loading`, before it writes, and asks (`This run deletes N of the M
documents ...; continue?`): a partial S3 sync, an expired prefix and a volume
not mounted look the same as files removed on purpose. `run-ingestion` asks
on a terminal; `--confirm-deletions` (or `UKR_CONFIRM_DELETIONS=1`) answers
yes to this question only, `--yes` to every question, and a no, or no
terminal, stops the run (exit 1).

- Only scopes of at least 10 documents are checked.
- `N` and `M` count documents only. A retired scope's records
  (`--retire-source-scope`) do not count, nor do seed graph shards: they
  follow `processing.seed_graphs`, so removing or resharding a seed graph
  never asks, and a large seed graph's many shard records do not hide a
  partial sync.
- A reset and a run with the registry off are not checked. Without the
  registry every run is a full build of what it reads, so keep the registry
  on for a corpus that may arrive partial (an empty corpus with seeds asks as
  in [Seed graphs: Lifecycle](./seed-graphs.md#lifecycle)).
- `1.0` or `null` turns the check off.

When the files are really gone, check the corpus first (sync, prefix, mount),
then confirm once; on a deployed task, re-run with `UKR_CONFIRM_DELETIONS=1`
as an environment override
([Operator Runbook: Run a one-off ingestion task](../operations.md#run-a-one-off-ingestion-task)
shows how). `total_documents_deleted` counts what a run removed. From
Python, pass `DataIngestionPipeline(..., confirm_deletions=callable)`;
without it, or on a no, `run` raises `ConfirmationDeclinedError`.

### Moving a corpus

Moving a corpus with the default scope changes its scope, and no run deletes
another scope's documents (a directory missing here may be another host's
corpus), so the old content stays indexed. Either:

- set `document_parsing.source_scope` to a stable name before moving it; or
- run once from the new location with
  `--retire-source-scope <old source directory>` (or
  `indexing.retire_source_scopes`). That run treats every record of the old
  scope in its suffix as deleted and indexes the new location as new
  documents. Give the scope as the registry stores it: the absolute
  directory (trailing and repeated slashes ignored) or the former
  `source_scope`, `s3://` URIs included; a non-URI value is also tried as a
  path resolved like `--source-directory`. A bare value (`old`) is a name or a
  relative path: the reading that has records is retired, and when both do
  the run stops before any write (give `./old` or an absolute path). Naming
  the run's own scope fails before the registry is read, a bare value one of
  whose readings is the run's own included: when the own scope is its name,
  give the directory as `./<value>`; when it is the run's own directory, the
  name could be another corpus's `source_scope`, so retire that name from a
  working directory where the value is not the run's directory, with the
  same configuration (the same stores and registry, e.g. the same
  `--config-path`). A scope without records logs a WARNING.

A local run with the default scope and new documents logs a WARNING naming the
other local source directories with records in its suffix (a document only a
refused or failed run recorded counts as new); it deletes nothing. Set
`source_scope` per corpus to silence it when they are separate corpora.

## Failed documents

A document is recorded `FAILED` when one of its text units failed in
translation, extraction, gleaning or claim extraction, when a base-level
community covering it got no report, when an artifact stage crashed in a
`--continue-on-error` run, when writing one of its text units failed, or when
the extraction record of one of its text units could not be stored. A store
that rejects an entity, relationship, claim or report write fails no
document: the next run writes the item again without extracting anything.
The next run processes a `FAILED` document again, even if the file did not change, until
`indexing.max_document_failures` (default `3`) consecutive failures of the
same content; editing the file or raising the limit retries it after that.
Only an error of the document's own counts toward that limit (a response
that does not parse or was cut off, a request the model rejects for its
content); an outage or a bad setting fails every document alike and excludes
none. Claims are optional: when only claim extraction failed, for
an error of the unit's own, the document is indexed without those claims,
counted for none, and the next run extracts them once more. Which errors
count, resuming a failed stage and write failures:
[Operator Runbook: Failed documents](../operations.md#failed-documents).

## Quarantined documents

A document whose processing kills the task (out of memory, a crashing parser)
would stop every later run the same way. After
`indexing.max_document_attempts` (default `3`) runs in a row died processing
its unchanged content, the last of them with the document alone, the next run
removes what the attempts wrote, records it `QUARANTINED`, logs a WARNING
naming the file and indexes the rest; it is skipped, by a reset too, until its
file changes or the limit is raised above its attempt count (new extraction
settings do not retry it).

- Only deaths count: a run that finishes, fails or is stopped (SIGTERM,
  Ctrl-C) clears its in-progress mark, and a reset counts no attempt. A run
  that lost its write lease counts as a death only when another run held the
  lease as it ended.
- When runs died processing several documents, the next runs halve the
  suspects until the culprit is alone, deferring the other new and changed
  documents meanwhile (`total_documents_deferred`): about `log2(n) + 1` runs
  for one culprit among `n`. Start each run with a new pipeline id until
  `total_documents_deferred` is 0.

Attempt bookkeeping, resumed runs and the limits of the scheme:
[Operator Runbook: Quarantined documents](../operations.md#quarantined-documents).

An incremental run's community report answers live only in its stage cache
(only a full build stores them in the extraction store, where a later reset
finds them): a run killed after community detection and started again under
a new pipeline id asks those reports again. Run the same pipeline id again to
reuse them.

## Concurrent runs

With the registry, a run holds its index namespace's write lease from its
first stage to its end; a second run on the same namespace stops with
`NamespaceLeaseError` before it reads the stores or writes anything. Wait for
the other run and start again. Without the registry nothing locks, so do not
run two ingestions on one index suffix. The lease, its renewal and takeover
after a crash: [Operator Runbook: Concurrent runs](../operations.md#concurrent-runs).

## Document size limit

The DynamoDB registry stores a document's text-unit (chunk) ids in one item,
limited to 400 KB (roughly 10,000 ids, about half that for a document you
edit, whose in-progress record holds old and new ids). A document over it
fails the indexing stage, naming the file, before anything is written: split
it. The SQLite registry of local mode has no such limit.

## Cross-run merge

A delta run merges what it extracts into the stored entities, relationships
and claims it touches, so an entity shared with unchanged documents keeps
their descriptions and chunk lineage, and what a changed or deleted document
shared is derived again from the chunks left; the rules are in
[Design §5.4](../design/incremental.md#54-identity-derived-content-and-merge-rules).
With `processing.fuzzy_entity_resolution` on, a delta entity merges into
its one most similar stored entity and two stored entities stay apart even
when a new name is similar to both, which a full build groups together
([Design §4.2](../design/ingestion.md#42-graph-identity-and-merging-stages-5-7)); run
`indexing.reset: true` to regroup the corpus.
