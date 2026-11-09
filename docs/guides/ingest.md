# Ingestion

> 🇰🇷 한국어판: [ingest.ko.md](./ingest.ko.md)

Ingestion turns a directory of documents into a knowledge graph in the graph
and vector stores. `run-ingestion --source-directory <dir>` runs the 13 stages ([Design §4](../design/ingestion.md#4-ingestion-pipeline) shows them),
checkpoints each one, and resumes after the last valid stage when re-run with
the same `--pipeline-id`. With the doc-status registry on it is incremental
([Incremental indexing](./incremental.md)).

Every flag with its default: [CLI Reference](../reference/cli.md#run-ingestion).
Which indexes a run writes, and which documents it may delete, follow from its
index suffix and source scope ([Scopes and namespaces](../concepts/scopes-and-namespaces.md)).
A Korean or mixed-language corpus: [Multilingual corpora](./multilingual.md).

<a id="pipeline-stages"></a>

## The 13 pipeline stages

In run order; `--enabled-stages` / `--resume-from-stage` take the **names**
(case-insensitive):

1. **`document_parsing`**: text per format (`.pdf`, `.txt`, `.csv`, `.json`,
   `.md`, `.html`; `.docx` with the `docx` extra), see
   [source formats](../troubleshooting.md#which-source-formats-are-read).
   With the doc-status registry on, only the files whose bytes or parser
   changed since they were indexed are parsed.
2. **`document_loading`**: the run's corpus (MinHash dedup, incremental
   filter); without `document_parsing`, pre-parsed `Document` `.json` files.
3. **`text_chunking`**: text units (`processing.chunking`).
4. **`translation`** (optional): to `target_language`; a no-op when source
   equals target and there are no extra targets.
5. **`graph_extraction`**: entities and relationships per chunk (LLM).
6. **`gleaning`** (optional): refinement passes (`processing.gleaning`).
7. **`graph_resolution`**: merges duplicate entities and relationships (name
   variants; similar names with `processing.fuzzy_entity_resolution`).
8. **`claim_extraction`** (optional, off by default): factual claims.
9. **`claim_resolution`** (optional): dedupes the claims.
10. **`graph_analysis`**: builds the graph, logs its size, density and connected components.
11. **`community_detection`**: Leiden clustering and LLM community reports
    (only when `indexing.methodologies` selects `graphrag`).
12. **`indexing`**: writes to OpenSearch and Neptune (and the registry when
    enabled).
13. **`summary_tree`** (only when `indexing.methodologies` selects
    `raptor`): RAPTOR's summary tree over the stored text units, which the
    `tree` strategy searches ([below](#a-raptor-summary-tree-opt-in)).

A run includes the stages its [methodology profile](#methodology-profiles)
needs. An explicit `--enabled-stages` list wins: a listed stage runs and an
unlisted one does not, and a WARNING names each selected methodology that an
unlisted stage leaves out of the index. A listed stage that no selected
methodology needs (`community_detection` without `graphrag`, `summary_tree`
without `raptor`) stops the run before any model call (exit 2): the message
names the methodology to add to `indexing.methodologies` (`--methodologies`).

## Examples

```bash
# Full build
run-ingestion --source-directory ./documents --config-path config.yaml

# Keep the stage cache in S3 so a fresh task can resume
run-ingestion --source-directory ./documents --config-path config.yaml \
  --s3-sync --s3-bucket-name your-bucket

# Resume an interrupted run from a stage
run-ingestion --source-directory ./documents --config-path config.yaml \
  --pipeline-id <id> --resume-from-stage graph_extraction

# Run only specific stages
run-ingestion --source-directory ./documents --config-path config.yaml \
  --enabled-stages DOCUMENT_PARSING,TEXT_CHUNKING,GRAPH_EXTRACTION

# Write the namespace tenant-a with LightRAG only, rebuilt from the whole
# corpus once (query it with run-rag --suffix tenant-a)
run-ingestion --source-directory ./documents --config-path config.yaml \
  --suffix tenant-a --methodologies lightrag --reset
```

`--suffix`, `--methodologies` and `--reset` set
`processing.document_parsing.index_value`, `indexing.methodologies` and
`indexing.reset` for that run only; the banner prints the run's
**Index Namespace** as `run-rag` does.

## Resume vs. force-rebuild

Without `--force-rebuild`, completed stages are cached and skipped on a
re-run:

- `--pipeline-id <id>` resumes that run at the first failed or incomplete
  stage (within `--enabled-stages`); a stage switched on since (gleaning,
  claims) counts as incomplete. Without `--pipeline-id` a run gets a new id.
- If the corpus or config changed, the CLI lists the completed stages whose
  output is no longer cached and asks before recomputing them (`--yes`
  without a terminal).
- With incremental indexing, a resume whose extraction settings changed since
  its delta was computed restarts at `document_loading`, so the delta follows
  the new settings.
- `--force-rebuild` recomputes every stage and rewrites the cache; it also
  asks the model for every pre-chunk's boundaries, description summary and
  community report again and embeds every text again (see
  [S3 sync](#s3-sync)).

## S3 sync

`--s3-sync` keeps the stage cache in `s3://<bucket>/<prefix>/...`, uploaded
after every stage, so a fresh task resumes there. On SIGTERM (ECS
`StopTask`, a Step Functions timeout) the run uploads the running stage's
checkpoint and exits 143. A failed upload between stages is retried by the
next one; a failed initial download or final upload fails the run with
`CacheSyncError`. Expiry and the other caches:
[Caches and prefixes](../operations.md#caches-and-prefixes).

Across runs, the doc-status registry (`providers.doc_status.enabled`) keeps
unchanged documents from being embedded again: an incremental run embeds
only what its delta changes. Within what a run writes, unchanged text keeps
its stored vector, and unchanged chunk boundaries, translations, description
summaries and community reports keep their stored model answers, so a changed document or
a reset of an unchanged corpus pays only for what changed
([Design §4.4](../design/ingestion.md#44-pipeline-infrastructure),
[§5.2](../design/incremental.md#52-full-rebuild-indexingreset)).

## LLM response cache

`cache.llm_responses` (on by default). Every ingestion model response is
kept under `llm-cache/` in the run's cache directory (synced with it), so a
re-run with the same pipeline id repays only the calls a stopped or crashed
stage had not made. The key is the rendered prompt plus the model id, max
tokens, temperature and reasoning.

- Only earlier runs' responses are served, each at most once per run; a
  response cut off at the output-token limit is not kept. Above temperature 0
  a re-run replays the earlier sample. The responses of a call that failed
  every attempt (one that never parsed), or whose graph or claims record
  failed to build, are dropped, so a re-run asks the model again for those
  units only; the late answer of a call abandoned on its timeout is not kept
  either.
- `--force-rebuild` serves nothing; query and evaluation calls are never
  cached.
- The cache grows by a few KB per model call and is read into memory at
  start; it is written in segments of `cache.llm_responses.flush_every`
  responses, so a killed task loses at most one segment. A stop also ends
  the retries of the model calls still running.

## Large and heterogeneous corpora

- **Large:** raise `processing.max_concurrency` as far as the Bedrock quota
  allows; keep `indexing.neptune.index_concurrency` at `1` unless you measured
  a gain (concurrent write batches conflict on shared vertices and retry);
  the doc-status registry
  (`providers.doc_status.enabled`) makes later runs pay only for changed
  documents. On Neptune, `indexing.neptune.vertex_ids: custom` looks vertices
  up by vertex id rather than by the `id` property. Switching an existing
  graph needs one full build ([Custom Neptune vertex ids](../operations.md#custom-neptune-vertex-ids)).
  Graph expansion takes a hub's neighbours in id order
  (`indexing.neptune.deterministic_order: true`, the default), so a rebuild
  returns the same results, at the cost of reading every neighbour of each
  expanded hub (on Gremlin Server, 6.8 to 23.2 ms per query for
  2,500-neighbour hubs; unmeasured on Neptune). If hub-heavy queries are
  too slow, `false` takes the first neighbours Neptune returns instead.
- **Community reports:** a report's prompt packs the community's
  relationships edge by edge, each with its two entities (and their claims
  with claim extraction on), into
  `graph.community_detection.report_generation.max_report_context_tokens`
  (4000). MS GraphRAG uses 8000 with `max_cluster_size` 10: if reports of
  large communities cover too few relationships, raise the budget (more input
  tokens per report call) or lower `max_cluster_size` (more, smaller
  communities and report calls).
- **Multilingual:** see [Multilingual corpora](./multilingual.md).
- **Heterogeneous domains:** tune `entity_types` to the union of the domains,
  or give each its own [index suffix](../concepts/scopes-and-namespaces.md#index-suffix).

## Methodology profiles

`indexing.methodologies` (default `[graphrag, lightrag]`) selects what a run
builds ([profiles](../concepts/overview.md#methodology-profiles)):

```yaml
indexing:
  methodologies: [lightrag]   # mix, hybrid, naive, ppr, iterative: no communities, no reports
```

- **What it skips.** Without `graphrag` no run does the Leiden pass or a
  report call; everything else (extraction, gleaning, summaries, the
  relationship vectors) is what LightRAG reads. A config from before this
  setting with `graph.community_detection.enabled: false` builds the same
  profile; one with `indexing.opensearch.build_relationship_vector_index:
  false` builds `graphrag` alone. An unknown name fails the load naming the
  valid ones, and so does a selection those switches leave nothing of.
- **Without reports.** `graph.community_detection.report_generation.enabled:
  false` keeps the Leiden pass and skips the report calls: the namespace
  holds `graphrag` for `local`, while `global` and `drift`, which answer
  from the reports, are refused naming the switch (a WARNING at load says
  so) and `auto` leaves them out. Turning the switch off for a namespace
  that holds reports, or claim extraction off for one that holds claims,
  stops maintaining them like removing a methodology (below): refused while
  the namespace holds another source scope's documents. Turning claim
  extraction on for a namespace without claims rebuilds it once, as adding
  a methodology does (the stored claim records are reused, so only the
  units without one are asked).
- **What the namespace holds.** Each run records it in the build record
  (`artifacts`, `methodologies`, and with the registry the `source_scopes`
  whose documents it holds). A query whose strategy reads a methodology
  the namespace does not hold is refused before any model call
  (`MethodologyNotIndexedError`; `run-rag` and `run-eval` exit 2), naming what it holds
  and how to add it: a re-run over the corpus of its one source scope, or,
  for a namespace of several, a namespace of its own or a reset.
- **Adding one later**, say `graphrag` after a `[lightrag]` build, builds
  only what it adds. With the doc-status registry the next run finds the
  namespace without communities and rebuilds it once (`indexing.reset`, a
  WARNING says so), reusing the stored chunk boundaries, extractions,
  description summaries and vectors: its new model calls are the community
  reports. A namespace that also holds another source scope's documents is
  refused instead (exit 2): the rebuild from one corpus would delete them.
  Without the registry every run is a full build and reuses the
  same; a re-run of the same `--pipeline-id` runs only `community_detection`
  and `indexing` from its stage cache.
- **Removing one** stops the runs from maintaining its artifacts: the next
  run of the namespace, one with an unchanged corpus too, records them gone
  and its strategies are refused from then on. The methodologies are the
  namespace's, so while it also holds another source scope's documents a
  run with fewer is refused before any model call (`ConfigurationError`,
  exit 2, nothing written), naming the scopes: it would stop maintaining
  the methodology for their documents too, and adding it back needs the
  rebuild a run of one corpus is refused (not for `raptor`: its tree is
  built again from every stored text unit on the next run that selects it,
  so the refusal says no rebuild is needed). Run with the namespace's
  methodologies, give the corpus a namespace of its own
  (`processing.document_parsing.index_value`), or rebuild from one corpus
  with `indexing.reset: true`, deleting the others. `insert_documents` and
  `delete_documents` are refused alike.
  Nothing is deleted. To delete them, run once with `indexing.reset: true`
  and the remaining methodologies (the extraction records stay, so adding
  the methodology back later extracts nothing again): the rebuild replaces the community-reports
  index with an empty one, drops the graph's community vertices and, without
  `raptor`, deletes the summary tree's indices (those of a build killed
  before it swapped in too). Without the registry every run is a full build,
  so the next run deletes them.

## A RAPTOR summary tree (opt-in)

`raptor` (not a default) builds a summary tree after indexing: the stored
text units are clustered by their vectors, each cluster is summarized
(`models.roles.summary_tree`, the default tier), the summaries are
clustered and summarized again, up to `indexing.summary_tree.max_levels`
(5), and the tree is written into its own index for the `tree` strategy
([Retrieval](../concepts/retrieval.md#search-strategies--when-to-use-which)).
Opt-in: on multi-hop questions it answered better than `naive` but well
below the graph strategies, in one repeat on one corpus ([evidence](../concepts/defaults-and-evidence.md#raptor-summary-tree)).

```yaml
indexing:
  methodologies: [raptor]                     # text units and the tree only
  # methodologies: [graphrag, lightrag, raptor]  # the tree beside the graph
  summary_tree:
    algorithm: kmeans          # or gmm (RAPTOR's Gaussian mixture, slower)
    partition_by: [category]   # optional: one tree per filter value
```

- **What `raptor` alone runs.** Parsing, loading, chunking, translation
  (when on), indexing and the tree: no graph extraction, gleaning,
  resolution, claims or communities, so a `raptor`-only namespace pays
  for the summaries alone: about one call per five or six text units, so
  the cost per corpus token grows as the chunks shrink and with the
  reasoning effort. On the MuSiQue benchmark index the tree on the fast
  tier (`models.roles.summary_tree: fast`) cost well under a dollar, against
  hundreds for its graph ingestion
  ([evidence](../concepts/defaults-and-evidence.md#raptor-summary-tree); the
  default tier's estimate: [Design §6.5](../design/retrieval-hipporag-raptor.md#65-raptor-summary-tree-raptor-tree-opt-in)).
  `tree`, `naive` and `simple` serve it; `auto` falls back to `tree`. It
  writes no graph store, so it needs no Neptune endpoint. A graph the
  namespace held before is left as it is by an incremental run; a full
  build (`indexing.reset: true`, or any run without the registry) deletes
  it, as it empties the vector indices of what it no longer builds, when a
  graph endpoint is configured. Without one the graph stays: run that reset
  once with the endpoint set to delete it.
- **Adding `raptor` later** builds only the tree, on the next run (an
  unchanged corpus included) or with `run-ingestion --enabled-stages
  summary_tree`, from the stored text units: no re-extraction and no
  rebuild of the namespace. The stage reads only the stores, so the
  `--enabled-stages summary_tree` run works under the pipeline id that
  built the namespace even when an upgrade changed the earlier stages'
  cache keys, and it runs no earlier stage. Either way `raptor` must be in
  `indexing.methodologies` of that run (`run-ingestion --methodologies
  graphrag,lightrag,raptor --enabled-stages summary_tree`): an
  `--enabled-stages summary_tree` run without it is refused before it starts.
- **Incremental runs** rebuild the tree from the stored text units and swap
  it in. A cluster whose children are unchanged takes its stored summary (no
  model call); the clustering starts from a sample of the node ids that a
  small delta barely moves, so most clusters survive one. A removed
  document's summaries, and every summary above them, are deleted with its
  text units, before the tree is built again, so no summary of removed text
  is served even when the run stops in between. `refresh: on_reset` keeps
  the tree until a reset (removals still delete their summaries), unless
  some of its summaries failed: that tree is built again on the next run.
- **Filters.** A summary keeps a document filter key only when all its
  documents share the value, so under a filter it is kept only when all its
  documents pass; `partition_by` keeps whole trees pure on those keys.
- **Languages.** Summaries are written in the index language (the target
  language when translation runs, the source language otherwise), and their
  text is analyzed with that language's analyzer.
- **Limits.** A tree holds at most `max_leaves_in_memory` (100,000) text
  units, and so do the trees read at once (a corpus split into several
  trees is read a batch of them at a time; the summaries of every tree,
  fewer than their text units, are held until the stage writes them all):
  their texts and vectors are clustered in memory, about 50 KiB of
  task memory per text unit at the peak (1,024 dimensions), so the default
  takes about 5 GiB of the task's 8 ([Capacity
  sizing](../operations.md#capacity-sizing)). Split larger corpora with
  `partition_by` or `scope: document` (one tree per document, RAPTOR's own
  setting; documents of fewer than three text units get none), or raise the
  limit with `fargate_memory`.

## Cost levers

LLM calls dominate cost: extraction (one or more calls per chunk), gleaning
(up to `max_rounds` per chunk), community reports, claims (one call per text
unit, off by default) and one answer per query. Levers: a
[methodology profile](#methodology-profiles) without `graphrag` (no
community reports), Haiku-class models for
the mechanical roles (the default for chunking, translation, map-reduce and
summarization), fewer gleaning rounds, a lower ingestion effort, claims
off, the caches and incremental indexing.

Gleaning is the largest of these levers (its share of a measured
ingestion: [evidence](../concepts/defaults-and-evidence.md#gleaning-rounds)).
`processing.gleaning.max_rounds` is 3; round 1 adds most of gleaning's
relationships and rounds 2-3 little more, and the answers measured with
round 1 alone favoured neither setting. Set `max_rounds: 1` (MS GraphRAG's
default) to keep most of that gain at about a third of the gleaning cost.

`aws.bedrock.ingestion_effort: low` lowers the reasoning effort of the
default tier's ingestion calls alone (queries keep `default_effort`): on a
measured slice it cut about a fifth of the ingestion cost with no
measurable difference in the answers
([evidence](../concepts/defaults-and-evidence.md#ingestion-effort)).
