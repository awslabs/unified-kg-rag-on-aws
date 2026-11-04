# Unified Knowledge Graph RAG on AWS — User Guide

> 🇰🇷 한국어판: [docs/user-guide.ko.md](./user-guide.ko.md)

This is the practical, how-to-use guide for **unified-kg-rag-on-aws** — an AWS-native
knowledge-graph RAG framework that builds knowledge graphs from large,
multilingual document corpora and answers questions over them. It reimplements
two retrieval methodologies on one stack: **Microsoft GraphRAG**
(community-summary) and **LightRAG** (dual-level keyword), selectable per query.

- For the *what / why* and a one-minute quickstart, see [README.md](../README.md).
- For a map of every page, see the [documentation hub](./index.md); for the
  concepts and what the framework adds beyond the papers, the
  [Overview](./concepts/overview.md).
- For *internals / architecture* (hexagonal layers, ports & adapters, the
  dependency rule), see [docs/design.md](./design.md).

The six console entry points (defined as `pyproject` scripts) all start
through `application.cli.launch`, which then loads the module below:

| Script | Module | Purpose |
|---|---|---|
| `run-ingestion` | `application.cli.run_ingestion_pipeline` | Build / update the knowledge graph |
| `run-rag` | `application.cli.run_rag_chain` | Query the graph |
| `run-eval` | `application.cli.run_evaluation` | Evaluate retrieval + generation |
| `run-visualization` | `application.cli.run_visualization` | Render an exported graph (no ingestion) |
| `run-prompt-tuning` | `application.cli.run_prompt_tuning` | Generate domain-adapted prompts |
| `prepare-benchmark` | `application.cli.prepare_benchmark` | Turn a public benchmark (MuSiQue) into a corpus + `run-eval` dataset |

Every CLI's `--help` ends with an example invocation and a link to its guide; every flag is in the [CLI Reference](./reference/cli.md).

## Contents

1. Prerequisites & Installation: moved to [Getting started on AWS](./getting-started/aws.md) (including [Required models](./getting-started/aws.md#required-models)), [Getting started locally](./getting-started/local.md) and [Local mode](./guides/local-mode.md)
2. Configuration: moved to [Configuration](./guides/configure.md) (every key: [Configuration Reference](./reference/configuration.md))
3. Ingestion: moved to [Ingestion](./guides/ingest.md) and [Multilingual corpora](./guides/multilingual.md)
4. Querying: moved to [Querying](./guides/query.md), [Retrieval](./concepts/retrieval.md) and the [Filters reference](./reference/filters.md)
5. Incremental indexing: moved to [Incremental indexing](./guides/incremental.md)
6. Evaluation: moved to [Evaluation](./guides/evaluate.md), the [Evaluation output reference](./reference/evaluation-output.md) and [Benchmarks](./guides/benchmarks.md)
7. Visualization: moved to [Visualization](./guides/visualize.md)
8. Prompt tuning: moved to [Domain adaptation: Prompt tuning](./guides/domain-adaptation.md#prompt-tuning-run-prompt-tuning)
9. Domain adaptation: moved to [Domain adaptation](./guides/domain-adaptation.md), [Extraction schema](./guides/extraction-schema.md), [Ontology](./guides/ontology.md) and [Seed graphs](./guides/seed-graphs.md)
10. [Operations & troubleshooting](#10-operations--troubleshooting)
11. [Using the library from Python](#11-using-the-library-from-python)
12. [Limitations](#12-limitations)

Related pages: [CLI Reference](./reference/cli.md) · [Configuration Reference](./reference/configuration.md) · [Model Catalog](./reference/models.md) · [Operator Runbook](./operations.md) ·
[Design Doc](./design.md) · [`iac/README.md`](../iac/README.md)

---

## 10. Operations & troubleshooting

[IAM permissions](./getting-started/aws.md#iam-permissions) are in Getting started on AWS; error messages and their fix
are in [Operator Runbook: Common errors](./operations.md#common-errors).

### CLI exit codes and JSON output

Every CLI (`run-ingestion`, `run-rag`, `run-eval`, `run-visualization`,
`run-prompt-tuning`, `prepare-benchmark`) exits with:

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | The run failed: a failed stage or query (in `run-ingestion` also a `ConfigurationError` a stage meets, or an unreachable store), a breached `--max-failure-rate`, a failed `run-eval --fail-if` gate, a failed upload, nothing rendered |
| `2` | Usage or configuration error: a bad flag, a missing or invalid input (a `run-eval` dataset, a `run-visualization` data file, a `prepare-benchmark` data file), an invalid config file or override, an unset store endpoint, a missing extra the selected providers need (`local`) (all before any work), or a `ConfigurationError` found while running outside a stage, such as `run-ingestion`'s pre-run refusal before its first model call (the search engine refused to create an index the run needs, or an incremental run's live indices, a deletion-only run's included, were built for other embedding, language or registry settings) |
| `130` | Interrupted with Ctrl-C (SIGINT) |
| `143` | Stopped by SIGTERM (ECS `StopTask`, a Step Functions timeout); `run-ingestion` checkpoints its finished stages first, and `run-eval` uploads the finished runs' results to an `s3://` output (or keeps them locally and logs the path) |

A run stopped with `130` or `143` ends its process at once: model calls in flight are abandoned rather than waited for, the logs, tracing and stdout are flushed first, and a process pool's workers are ended with it. A SIGTERM or Ctrl-C cancels a query CLI's work at its next await, so its cleanup (closing store connections, `run-eval`'s upload) runs (at the interactive prompt a SIGTERM ends the session at once and a Ctrl-C the conversation); a query CLI stopped by a SIGTERM, or by a Ctrl-C while its event loop runs, that is still running 100 seconds later is ended then (inside the 120 s StopTimeout of the app container), with the status document if none was printed, and a second signal ends it at once. `run-ingestion` stops at a SIGTERM or Ctrl-C after the checkpoint or metadata write in progress instead, bounded by the StopTimeout. A CLI that finished ends its process with its exit code as soon as its exit handlers ran, so a signal while it exits cannot turn `0`, `1` or `2` into `-15` or `-2`.

Flags must be spelled out in full: a prefix such as `--output` is rejected.

`--output-format json` prints one JSON document to stdout and moves the
banner, progress and logs to stderr, so the output pipes into `jq` (not with
`--metrics-sink cloudwatch`, which also writes to stdout): the answer or
search results (`run-rag`), the pipeline id, status, the number of documents
the run recorded `FAILED` (`failed_documents`; a model outage that fails
every unit records none, one that fails some units records their documents,
retried by the next run without counting toward
`indexing.max_document_failures`), the number of text units a stage
failed on (`units_failed`: indexed without that stage's output and
extracted again by the next run; counted in a full build too, where no
registry records a document `FAILED` and `failed_documents` stays 0; the
run ends with a WARNING naming them by stage and cause), the number of documents failing
another consecutive run for an error not their own (`documents_retrying`:
an outage or a setting never excludes them; also warned of with each file
and its run count), the number of documents indexed without the claims of
some text units, whose claim extraction failed for an error of their own
(`documents_without_claims`: counted for none, also in the summary), the
number of documents left out because they failed
`indexing.max_document_failures` consecutive runs unchanged, at the same
extraction settings (a settings change starts the count over)
(`documents_excluded`, also in the summary), the model calls and tokens of
the stages the run executed (`token_usage`) and
per-stage results (`run-ingestion`), each run's summary, the comparison, the failure-budget
breaches and the failed gates (`gate_failures`) (`run-eval`), the written files (`run-visualization`), the output
path and profile (`run-prompt-tuning`), or the counts written
(`prepare-benchmark`). A run stopped by a
signal before it printed prints `{"status": "interrupted", "exit_code": 143}`
(`130` for Ctrl-C), and one that fails before it printed prints
`{"status": "failed", "exit_code": 1}` (or `2`) with the `error` it failed
with (an argparse error included), so stdout is always JSON. That holds
from the first moment: a SIGTERM while the CLI still loads, or after it
printed its result while the process exits, exits `143` too, and a Ctrl-C
while it loads `130` (the status document only while loading; a SIGINT the
parent ignores, as `nohup` does, stays ignored). A
`run-ingestion` result whose run a stage failed carries the same `error` and
`exit_code`; `run-ingestion` adds the `pipeline_id` to resume with (once the run resolved it) and, when a stopped
run's next run on the namespace must wait for writes possibly still in
flight (this run's, or those of the run whose expired lease it took over),
`next_run_writes_from` (UTC, ISO 8601).

---

## 11. Using the library from Python

The CLIs wrap two public classes. Import them, and everything else, from the
package root: the names in `unified_kg_rag.__all__` are the public API, other
module paths are internal and may change. `get_config` applies the same file
and environment layers as the CLIs ([Configuration](./guides/configure.md#how-configuration-is-loaded)).

```python
from unified_kg_rag import (
    DataIngestionPipeline,
    GraphRAGChain,
    PipelineConfig,
    PipelineStageStatus,
    RAGInput,
    get_config,
)

config = get_config("config.yaml")

# Ingest. PipelineConfig carries the run-ingestion run options (pipeline_id,
# cache directory, S3 sync, enabled stages, force_rebuild). run() ingests the
# directory it is given: parsing, loading and the registry scope all use it.
pipeline = DataIngestionPipeline(config, PipelineConfig(pipeline_id="run-001"))
try:
    context = pipeline.run("./source")
    if context.status == PipelineStageStatus.FAILED:
        raise SystemExit("ingestion failed")
finally:
    pipeline.close()  # releases the indexers' Neptune/OpenSearch connections

# Query. One chain serves many queries, including concurrent ones.
chain = GraphRAGChain(config)
try:
    output = chain.invoke(RAGInput(query="How are Alice and Acme related?", search_strategy="mix"))
    print(output.answer)
    for source in output.sources:  # only what the answer model saw
        print(source)
finally:
    chain.close()
```

- A failed stage sets the returned `PipelineContext`'s `status` to `FAILED`
  rather than raising, so check it as above.
- The errors to tell apart are exported from the package root:
  `ConfigurationError` (a setting to fix; `StoreRequestRejectedError` among
  them), `InvalidFilterError`, `IndexNotFoundError`, `StoreConnectionError`,
  `NamespaceLeaseError`, `PipelineExecutionError` and `PipelineInterrupted`
  (a `KeyboardInterrupt`); all but the last are `GraphRAGException`s.
  `DataIngestionPipeline(...)` raises `SeedGraphError` (a
  `ConfigurationError`) for a seed graph it cannot import.
- `DataIngestionPipeline(..., confirm=callable)` answers the questions a run
  asks (`callable(question) -> bool`), today only a reset that would drop
  another source scope's seed graph documents; without it such a run fails
  ([Seed graphs](#d-seed-graphs-bring-your-own-knowledge-graph)).
  `DataIngestionPipeline.run` raises `ConfigurationError`,
  `StoreConnectionError` and `NamespaceLeaseError` as they are, and wraps
  any other crash outside a stage in `PipelineExecutionError`. A
  model client's own transport error (`httpx.ConnectError` from a local
  model server, say) is raised as is.
- `RAGInput` takes the `run-rag` options as fields (`search_strategy`,
  `search_type`, `top_k`, `suffix` ([index suffix](./concepts/scopes-and-namespaces.md#index-suffix)),
  `filters`, `conversation_id`, `use_memory`, `target_language`); a plain dict
  works too.
- `chain.search(...)` (`await chain.asearch(...)`) retrieves without
  answering and returns a `SearchOutput` (`search_results`,
  `processed_query`, `metadata`).
- In async code use `await chain.ainvoke(...)` and `await chain.aclose()`, and
  size the loop's executor once at startup:
  `configure_event_loop(asyncio.get_running_loop(), config.processing.io_workers)`.
- Always `close()` / `aclose()` what you build: both classes open connections
  on first use and keep them until garbage collection otherwise. A sync call
  still running on another thread when the chain closes raises
  `StoreConnectionError`; a later call reconnects.
- Importing the package configures no logging handler;
  `unified_kg_rag.setup_logging(config)` gives the CLI's logging. Without it,
  structured fields (including `structlog.contextvars` bindings) reach your
  handlers as record attributes, a trailing `_` added to one named like a
  `LogRecord` attribute (`name`, `msg`, ...). `setup_logging` changes
  process-wide state, so call it only where the package owns logging: it
  replaces the structlog configuration, sets the root logger's level to
  `logging.level` and the `logging.library_levels` loggers' levels, and adds
  a stdout handler (plus the dated file with `logging.log_to_file`) to the
  root logger. Handlers already on the root logger stay: they receive every
  record too, so the console shows each line twice when one of them writes
  there, and for the package's records they get the plain message and the
  exception, not the structured fields. A repeated call replaces only the
  handlers it added.

Extensions import from the package root too: the ports (`LLMFactoryPort`,
`GraphIndexer`, `DocStatusPort`, ...), `Providers`, `register_provider`,
`BaseSearchStrategy` with `register_strategy`, `ParserFactory` and
`register_renderer`. Both constructors take injected backends, either as a
`providers=Providers(config, ...)` bundle or as individual arguments
(`doc_status=`, `vector_indexer=`, `model_factory=`, ...), not both; see
[Design Doc §15](./design.md#15-extension-guide). An indexer passed as
`vector_indexer=` / `graph_indexer=` stays yours to close: the pipeline never
closes it.

---

## 12. Limitations

- **Extraction can hallucinate** entities or relationships the text does not
  support, which then reach retrieval.
  `processing.graph_extraction.entity_grounding.enabled` (off by default)
  drops or down-weights items whose quoted `source_text` is not in the chunk.
- **Runs are not reproducible**: LLM outputs (extraction, reports, the
  `auto` router, answers) vary, so compare configurations over several runs.
  Given the same model responses and the same source scope, everything else
  is deterministic in any process, whatever its `PYTHONHASHSEED`. With the
  doc-status registry on, text-unit ids derive from the source scope, which
  defaults to the absolute source directory: the same corpus ingested from
  another directory gets other ids. Set
  `processing.document_parsing.source_scope` for ids that do not depend on
  where the corpus sits.
- **Reranking is regional** ([Required models](./getting-started/aws.md#required-models)); without it
  results keep the fusion order, as `metadata.rerank` shows
  ([Rerank status](./guides/query.md#rerank-status)).
- **Shipped backends.** Models: Bedrock (default), Ollama or
  OpenAI-compatible servers; reranking: Bedrock or a sentence-transformers
  cross-encoder ([Local mode](./guides/local-mode.md)). Stores: Neptune (or a
  Gremlin Server) with OpenSearch, or the embedded SQLite stores (one host,
  about 100,000 documents per index). Others need a `register_provider`
  builder or custom adapters ([Design Doc §15](./design.md#15-extension-guide)).
- **No serving layer**: no HTTP API, end-user authentication or rate limiting.
- **Filters are not access control**
  ([Scopes and namespaces](./concepts/scopes-and-namespaces.md#not-an-access-control-boundary)):
  separate tenants by index suffix.
- **A lighter ingestion limits the strategies.** Without community detection
  (`graph.community_detection.enabled: false`) `global` and `drift` have no
  reports; without the relationship vector index
  (`indexing.opensearch.build_relationship_vector_index: false`) `mix` and
  `hybrid` lose high-level keywords, `local` its relationship section, and
  graph expansion under a document filter keeps entities on the
  entities-index check alone, not on the relationships that reach them.
- **Incremental indexing approximates**
  ([Rebuild signals](./guides/incremental.md#rebuild-signals)): delta runs append communities
  instead of re-clustering, and stored communities and reports keep an
  entity's old id after it took its current name's; a full rebuild
  (`indexing.reset: true`) refreshes both. With DynamoDB one document's chunk
  ids must fit one 400 KB item ([*Document size limit*](./guides/incremental.md#document-size-limit)).
- **Concurrent runs need the registry**, whose write lease stops a second run
  on a namespace; without it (or through the direct `IncrementalIndexer` API)
  nothing locks, and two ingestions on one suffix can delete each other's
  index ([Concurrent runs](./operations.md#concurrent-runs)).
- **Isolating a document that kills the task takes runs**
  ([*Quarantined documents*](./guides/incremental.md#quarantined-documents)): about `log2(n) + 1` for one culprit among `n` documents that
  died together, while the deferred ones wait.
- **Markdown and HTML** need the `unstructured` extra, which
  the container image leaves out by default.
- **Seed graphs link by name only** ([*Precedence*](#seed-precedence),
  [§9 D](#d-seed-graphs-bring-your-own-knowledge-graph)): a seed node and an
  unrelated document entity of the same name are one entity. IRIs are
  stored and filterable but are no identity, `owl:sameAs` is not followed,
  and nothing is inferred.
- **Changing the embedding model** or dimension needs `indexing.reset: true`;
  an incremental run refuses to write into another model's indices
  ([Workflows](./guides/incremental.md#workflows)).

---

*See also: [README.md](../README.md) for the overview and quickstart, and
[docs/design.md](./design.md) for architecture and internals.*

## Moved sections

Sections 1 to 9 of this guide, and part of section 10, moved to their own pages. The headings
below keep older links, including the `--help` output of earlier
versions, working.

### 1. Prerequisites & Installation

Moved to [Getting started on AWS](./getting-started/aws.md) and
[Getting started locally](./getting-started/local.md).

### Runtime

Moved to [Getting started on AWS: Runtime](./getting-started/aws.md#runtime).

### AWS services

Moved to [Getting started on AWS: AWS services](./getting-started/aws.md#aws-services).

### Required models

Moved to [Getting started on AWS: Required models](./getting-started/aws.md#required-models).

### Install

Moved to [Getting started on AWS: Install](./getting-started/aws.md#install).

### Authentication

Moved to [Getting started on AWS: Authentication](./getting-started/aws.md#authentication).

### IAM permissions

Moved to [Getting started on AWS: IAM permissions](./getting-started/aws.md#iam-permissions).

### Local mode (no AWS)

Moved to [Getting started locally](./getting-started/local.md) (a first
run) and [Local mode](./guides/local-mode.md) (providers, differences from
AWS mode, containers).

### Differences from AWS mode

Moved to [Local mode: Differences from AWS mode](./guides/local-mode.md#differences-from-aws-mode).

### Provider flags

Moved to [Local mode: Provider flags](./guides/local-mode.md#provider-flags).

### Local stores in containers

Moved to [Local mode: Local stores in containers](./guides/local-mode.md#local-stores-in-containers).

### Local stores with Bedrock models

Moved to [Local mode: Local stores with Bedrock models](./guides/local-mode.md#local-stores-with-bedrock-models).

### 2. Configuration

Moved to [Configuration](./guides/configure.md).

### How configuration is loaded

Moved to [Configuration: How configuration is loaded](./guides/configure.md#how-configuration-is-loaded).

### Settings most runs change

Moved to [Configuration: Settings most runs change](./guides/configure.md#settings-most-runs-change).

### Model selection notes

Moved to [Model Catalog: Model selection notes](./reference/models.md#model-selection-notes).

### Guardrail

Moved to [Guardrail](./guides/guardrail.md).

### S3 cache encryption

Moved to [Configuration: S3 cache encryption](./guides/configure.md#s3-cache-encryption).

### What the logs contain

Moved to [Configuration: What the logs contain](./guides/configure.md#what-the-logs-contain).

### Custom prompts

Moved to [Domain adaptation: Custom prompts](./guides/domain-adaptation.md#custom-prompts).

### Environment variable overrides

Moved to [Configuration: Environment variable overrides](./guides/configure.md#environment-variable-overrides).

### Observability

Moved to [Observability](./guides/observability.md) (settings and
recipes) and the [Telemetry reference](./reference/telemetry.md) (trace
shape, usage counters).

### 3. Ingestion (`run-ingestion`)

Moved to [Ingestion](./guides/ingest.md).

### The 12 pipeline stages

Moved to [Ingestion: The 12 pipeline stages](./guides/ingest.md#the-12-pipeline-stages).

### Index suffix

Moved to [Scopes and namespaces: Index suffix](./concepts/scopes-and-namespaces.md#index-suffix).

### Multilingual ingestion

Moved to [Multilingual corpora](./guides/multilingual.md).

### 4. Querying (`run-rag`)

Moved to [Querying](./guides/query.md) and [Retrieval](./concepts/retrieval.md).

### Search strategies — when to use which

Moved to [Retrieval: Search strategies](./concepts/retrieval.md#search-strategies--when-to-use-which).

### Answer style

Moved to [Retrieval: Answer style](./concepts/retrieval.md#answer-style).

### Context budget

Moved to [Retrieval: Context budget](./concepts/retrieval.md#context-budget).

### Rerank status

Moved to [Querying: Rerank status](./guides/query.md#rerank-status).

### Query deadline

Moved to [Querying: Query deadline](./guides/query.md#query-deadline).

### Filters

Moved to [Querying: Filters](./guides/query.md#filters) (how to use them)
and the [Filters reference](./reference/filters.md) (every field, value
typing, per-strategy applicability).

### Interactive mode & conversation memory

Moved to [Querying: Interactive mode & conversation memory](./guides/query.md#interactive-mode--conversation-memory).

### 5. Incremental indexing

Moved to [Incremental indexing](./guides/incremental.md).

### Enable it

Moved to [Incremental indexing: Enable it](./guides/incremental.md#enable-it).

### Workflows

Moved to [Incremental indexing: Workflows](./guides/incremental.md#workflows)
(rebuilding: [Rebuild](./guides/incremental.md#rebuild-indexingreset) and
[Rebuild signals](./guides/incremental.md#rebuild-signals)).

### Deletion scope

Moved to [Incremental indexing: Deletion scope](./guides/incremental.md#deletion-scope).

### Failed documents

Moved to [Incremental indexing: Failed documents](./guides/incremental.md#failed-documents).

### Quarantined documents

Moved to [Incremental indexing: Quarantined documents](./guides/incremental.md#quarantined-documents).

### Concurrent runs

Moved to [Incremental indexing: Concurrent runs](./guides/incremental.md#concurrent-runs).

### Document size limit

Moved to [Incremental indexing: Document size limit](./guides/incremental.md#document-size-limit).

### Cross-run merge

Moved to [Incremental indexing: Cross-run merge](./guides/incremental.md#cross-run-merge).

### Large / multilingual / heterogeneous corpora

Moved to [Ingestion: Large and heterogeneous corpora](./guides/ingest.md#large-and-heterogeneous-corpora).

### Cost notes

Moved to [Ingestion: Cost levers](./guides/ingest.md#cost-levers).

### 6. Evaluation (`run-eval`)

Moved to [Evaluation](./guides/evaluate.md) (evaluators, dataset format,
comparisons, rebuilding and gating), the
[Evaluation output reference](./reference/evaluation-output.md) (result
files, summary, run manifest, comparison fields, evaluator details) and
[Benchmarks](./guides/benchmarks.md).

### Evaluators

Moved to [Evaluation: Evaluators](./guides/evaluate.md#evaluators).

### `ragas` contexts

Moved to [Evaluation output: `ragas` contexts](./reference/evaluation-output.md#ragas-contexts).

### `graph_aware`

Moved to [Evaluation output: `graph_aware`](./reference/evaluation-output.md#graph_aware).

### `retrieval`

Moved to [Evaluation output: `retrieval`](./reference/evaluation-output.md#retrieval).

### `answer_match`

Moved to [Evaluation output: `answer_match`](./reference/evaluation-output.md#answer_match).

### Eval data format

Moved to [Evaluation: Eval data format](./guides/evaluate.md#eval-data-format).

### Examples

Moved to [Evaluation: Examples](./guides/evaluate.md#examples) (what the
result files hold: [Evaluation output](./reference/evaluation-output.md)).

### Comparing strategies

Moved to [Evaluation: Comparing strategies](./guides/evaluate.md#comparing-strategies)
(every field: [Evaluation output: Comparison](./reference/evaluation-output.md#comparison)).

### Rebuilding, re-judging and gating a comparison

Moved to [Evaluation: Rebuilding, re-judging and gating a comparison](./guides/evaluate.md#rebuilding-re-judging-and-gating-a-comparison).

### Benchmarks (MuSiQue)

Moved to [Benchmarks](./guides/benchmarks.md).

### 7. Visualization (`run-visualization`)

Moved to [Visualization](./guides/visualize.md).

### 8. Prompt tuning (`run-prompt-tuning`)

Moved to [Domain adaptation: Prompt tuning](./guides/domain-adaptation.md#prompt-tuning-run-prompt-tuning).

### A. `entity_types` (cheapest, highest-impact)

Moved to [Domain adaptation: Entity types](./guides/domain-adaptation.md#entity-types).

### B. `custom_prompts` overrides

Moved to [Domain adaptation: Custom prompts](./guides/domain-adaptation.md#custom-prompts)
(the five prompts without an override:
[Prompts without an override](./guides/domain-adaptation.md#prompts-without-an-override)).

### 9. Domain adaptation

Moved to [Domain adaptation](./guides/domain-adaptation.md) (entity types,
prompts, prompt tuning), [Extraction schema](./guides/extraction-schema.md),
[Ontology](./guides/ontology.md) and [Seed graphs](./guides/seed-graphs.md).

### C. Extraction schema

Moved to [Extraction schema](./guides/extraction-schema.md).

### From an OWL/RDFS ontology

Moved to [Ontology](./guides/ontology.md) (install, configure, preview,
freeze) and the [Ontology mapping reference](./reference/ontology-mapping.md)
(options, what is read, mapping rules, worked example).

- <a id="ontology-load-cost"></a>Load cost: [Ontology: Load cost](./guides/ontology.md#load-cost).
- <a id="ontology-what-a-change-re-runs"></a>What a change re-runs: [Ontology: What a change re-runs](./guides/ontology.md#what-a-change-re-runs).
- <a id="ontology-preview"></a>Preview: [Ontology: Preview](./guides/ontology.md#preview).

### D. Seed graphs (bring your own knowledge graph)

Moved to [Seed graphs](./guides/seed-graphs.md) and the
[Seed graph formats reference](./reference/seed-graph-formats.md) (JSONL,
CSV, RDF).

- <a id="seed-precedence"></a>Precedence: [Seed graphs: Precedence](./guides/seed-graphs.md#precedence-documents-decide).
- <a id="seed-lifecycle"></a>Lifecycle: [Seed graphs: Lifecycle](./guides/seed-graphs.md#lifecycle).
- <a id="seed-cost-and-limits"></a>Cost and limits: [Seed graphs: Cost and limits](./guides/seed-graphs.md#cost-and-limits).
