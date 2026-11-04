# Ingestion

> 🇰🇷 한국어판: [ingest.ko.md](./ingest.ko.md)

Ingestion turns a directory of documents into a knowledge graph in the graph
and vector stores. `run-ingestion --source-directory <dir>` runs the 12 stages ([Design §4](../design.md#4-ingestion-pipeline) shows them),
checkpoints each one, and resumes after the last valid stage when re-run with
the same `--pipeline-id`. With the doc-status registry on it is incremental
([Incremental indexing](../user-guide.md#5-incremental-indexing)).

Every flag with its default: [CLI Reference](../reference/cli.md#run-ingestion).
Which indexes a run writes, and which documents it may delete, follow from its
index suffix and source scope ([Scopes and namespaces](../concepts/scopes-and-namespaces.md)).
A Korean or mixed-language corpus: [Multilingual corpora](./multilingual.md).

## The 12 pipeline stages

In run order; `--enabled-stages` / `--resume-from-stage` take the **names**
(case-insensitive):

1. **`document_parsing`**: text per format (`.pdf`, `.txt`, `.csv`, `.json`;
   `.md`/`.html` with the `unstructured` extra).
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
10. **`graph_analysis`**: builds the graph, logs `graph.analysis.statistics`.
11. **`community_detection`**: Leiden clustering and LLM community reports.
12. **`indexing`**: writes to OpenSearch and Neptune (and the registry when
    enabled).

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
```

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
its stored vector, and unchanged chunk boundaries, description summaries and
community reports keep their stored model answers, so a changed document or
a reset of an unchanged corpus pays only for what changed
([Design §4.4](../design.md#44-pipeline-infrastructure),
[§5.2](../design.md#52-full-rebuild-indexingreset)).

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

## Cost levers

LLM calls dominate cost: extraction (one or more calls per chunk), gleaning
(up to `max_rounds` per chunk), community reports, claims (one call per text
unit, off by default) and one answer per query. Levers: Haiku-class models for
the mechanical roles (the default for chunking, translation, map-reduce and
summarization), fewer gleaning rounds, claims off, the caches and incremental
indexing.

Gleaning is the largest of these levers. `processing.gleaning.max_rounds`
stays 3 for the graph recall it measured
([evidence](../concepts/defaults-and-evidence.md#gleaning-rounds)). Set
`max_rounds: 1` (MS GraphRAG's default) when ingestion cost matters more
than graph recall.
