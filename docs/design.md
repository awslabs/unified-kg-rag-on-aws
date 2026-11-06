# Unified Knowledge Graph RAG on AWS — Technical Documentation

This is the **design reference for contributors and advanced users**: how the
library is built and why. Usage is in the guides of the [documentation hub](./index.md), every
configuration key in the [Configuration Reference](./reference/configuration.md), the
contribution workflow in [CONTRIBUTING.md](../CONTRIBUTING.md), and extension
recipes in [§15](#15-extension-guide).

## Table of Contents

1. [Overview and Design Philosophy](#1-overview-and-design-philosophy)
2. [Hexagonal Architecture (Ports & Adapters)](#2-hexagonal-architecture-ports--adapters)
3. [Domain Model](#3-domain-model)
4. [Ingestion Pipeline](#4-ingestion-pipeline)
5. [Incremental Indexing](#5-incremental-indexing)
6. [Retrieval: Two Methodologies](#6-retrieval-two-methodologies)
7. [Hybrid Scoring and Token Management](#7-hybrid-scoring-and-token-management)
8. [AWS Service Integration](#8-aws-service-integration)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Visualization & Analytics](#10-visualization--analytics)
11. [Prompts and Prompt Tuning](#11-prompts-and-prompt-tuning)
12. [Configuration System](#12-configuration-system)
13. [Testing Strategy](#13-testing-strategy)
14. [CI/CD and Security](#14-cicd-and-security)
15. [Extension Guide](#15-extension-guide)
16. [Further Reading](#16-further-reading)

---

## 1. Overview and Design Philosophy

`unified-kg-rag-on-aws` reimplements the Microsoft GraphRAG and LightRAG
methodologies on an AWS-native stack (Bedrock, Neptune, OpenSearch, S3,
DynamoDB). Its design principles:

- **Two methodologies, one infrastructure.** GraphRAG (community summaries)
  and LightRAG (dual-level keywords) share ingestion, indexing, caching,
  multilingual processing and hybrid search; **only the retrieval algorithm
  differs**.
- **Generalization first.** No hardcoded domain rules or regex heuristics:
  semantic judgments go to the LLM or to authoritative data, token counts to
  the model's tokenizer API where it has one (a script-aware estimate
  otherwise), and thresholds and weights are configuration.
- **Hexagonal boundaries.** Domain and algorithm code depends on abstract
  ports; the AWS (and local) adapters sit behind them.
- **Registry-based extension.** Search strategies, renderers, parsers and
  providers register through registries, with no change to dispatch code;
  evaluators map from `EvaluatorType` in one method (§2.4).

Where the framework extends the GraphRAG and LightRAG papers (triple-hybrid
retrieval, both methodologies on one index, incremental indexing, multilingual
ingestion, bring-your-own graphs and schemas, measured defaults) and where it
deliberately departs from them is summarized in
[Overview: Beyond the papers](./concepts/overview.md#beyond-the-papers); the
sections below give the design behind each.

![Architecture at a glance](../assets/architecture-overview.drawio.png)

---

## 2. Hexagonal Architecture (Ports & Adapters)

### 2.1 Dependency Rule and Layer Map

Imports point **inward**. The layers, outermost first; a module imports its
own layer and the layers after it, never one before it:

| Layer | Package | Role |
|---|---|---|
| Entry points | `application/cli/` | the console scripts: parse flags, load config, wire, run |
| Evaluation | `evaluation/` | drives the RAG chain over a dataset; of the application it imports only the chain (`application/retrieval/`) and the composition root (`application/bootstrap.py`) |
| Application | `application/` | orchestration: ingestion pipeline, indexing manager, RAG chain, prompt tuner, config loading, composition root |
| Adapters | `adapters/` | technology bindings and port implementations |
| Ports | `ports/` | the interfaces the application wires and the adapters implement |
| Domain | `domain/` | models, pure algorithms, prompts |
| Shared kernel | `shared/` | logging, errors, metrics, tracing hooks, write fence, pure helpers |

The rule holds for every import: at module level, inside a function, and
under `TYPE_CHECKING` (a type-only import is still a dependency). Inside the
package the root facade is imported only for `__version__`.

```mermaid
flowchart TB
    cli["<b>application/cli/</b><br/>entry points"]
    evaluation["<b>evaluation/</b><br/>evaluators, strategy comparison"]
    application["<b>application/</b><br/>ingestion pipeline, indexing manager, RAG chain, bootstrap"]
    adapters["<b>adapters/</b><br/>AWS and local bindings, search strategies, retrievers"]
    ports["<b>ports/</b><br/>abstract interfaces"]
    domain["<b>domain/</b><br/>models, pure algorithms, prompts"]
    shared["<b>shared/</b><br/>logging, errors, metrics, pure helpers"]
    cli --> evaluation --> application --> adapters --> ports --> domain --> shared
```

An arrow reads "may import"; a layer may also import every layer further
down (the application imports the domain directly, every layer the kernel).

- **Ports sit outside the domain.** A port is written in terms of the domain
  models (`DocStatusPort.diff` returns a `DocumentDelta`), so the domain
  cannot import it back. Domain code that reads a store declares the subset
  it reads as its own `Protocol` (`domain.ingestion.delta_detector.
  DocStatusReader`, the read half of `DocStatusPort`); every port
  implementation satisfies it structurally, mypy checks each caller, and
  `tests/unit/test_fake_port_signatures.py` pins the two shapes together.
  The application hands the port in.
- **`shared/` is a leaf kernel.** It imports no other layer, so every layer,
  the domain included, uses it without a cycle. Code that needs the domain
  models lives in the layer it belongs to: config loading in
  `application/config.py`, the stage cache and resume managers in
  `application/ingestion/`, stage cache keys and document identity in
  `domain/ingestion/`, store names in `domain/retrieval/`, the LangChain
  helpers in `adapters/llm/langchain.py` and
  `adapters/ingestion/document_converter.py`, the rich console helpers in
  `application/cli/display.py`. `setup_logging` takes the configuration
  structurally (`shared.logging.HasLoggingSettings`).
- **Evaluation reaches the stores through the chain** (its retrievers,
  `GraphRAGChain.check_filter_values` / `check_filter_keys`) and imports no
  store adapter.
- **Backends are injected.** Vendor bindings (`adapters/aws/`,
  `adapters/local/`) and port implementations (the `storage/` indexers,
  `retrievers/`, `doc_status/`, `guardrail/`) are built by the provider
  builders and the composition roots: `application/bootstrap.py`
  (`build_providers`, `build_query_guard`, `build_checkpoint_sync`) and the
  CLIs. The orchestrators receive them (`Providers`, those builders, or a
  constructor argument).
- **Plugins through their registry.** Outside its package, a renderer,
  search strategy, evaluator, schema source or seed source is reached
  through the package root only, or, for renderers and evaluators, also
  their `base` module (§2.4); `tests/unit/test_layering.py` (`_PLUGINS`)
  lists the importable modules. The search-strategy registry lives with the
  strategy bases it stores (`adapters/retrieval/strategy_registry.py`), as
  the provider and renderer registries live with what they register.

No `domain/` module may load LangChain, LangSmith, boto3/botocore, lxml,
opensearch-py or gremlin-python, even transitively, so `shared/utils/`
holds only dependency-light helpers; `tests/unit/test_domain_purity.py`
checks this in a clean interpreter.


```
unified_kg_rag/
├─ __init__.py         # public API facade: lazy re-exports of the supported names
├─ domain/              # technology-agnostic core (no boto3/LangChain/backend imports)
│  ├─ models/           #   Pydantic domain models
│  ├─ ingestion/        #   pure algorithms: delta_detector, graph_builder/resolver/analyzer,
│  │  │                 #   claim_resolver, contributions, derivation, cache_keys,
│  │  │                 #   document_identity, seed_graph, visualization_data (+ layout helpers),
│  │  │                 #   community_clustering (Leiden hierarchy, content-named communities)
│  │  └─ merge/         #   cross-run merger
│  ├─ retrieval/        #   store_names, index_prefixes, hybrid_normalization
│  └─ prompts/          #   version-controlled prompt templates
├─ ports/               # interfaces the application wires, adapters implement (DocStatusPort,
│                       #   BaseIndexer/GraphIndexer/VectorIndexer, ModelFactoryPort,
│                       #   ContentGuardPort — ports/__init__ is the port catalog)
├─ adapters/            # concrete technology bindings
│  ├─ aws/              #   Bedrock model factories, Neptune/OpenSearch clients, S3 transfer and caches
│  ├─ local/            #   Ollama/OpenAI-compatible models, cross-encoder, embedded SQLite graph/search stores
│  ├─ llm/              #   provider-neutral chain assembly, transient retry, token estimate,
│  │                    #   LangChain batch/XML helpers (langchain.py)
│  ├─ observability/    #   OpenTelemetry tracer, exporters, LangChain callback
│  ├─ storage/          #   Neptune/OpenSearch/SQLite-graph indexers (write-side port implementations)
│  ├─ retrievers/       #   Neptune/OpenSearch/SQLite-graph retrievers
│  ├─ doc_status/       #   DynamoDB/SQLite document-status registries (DocStatusPort)
│  ├─ guardrail/        #   Bedrock ApplyGuardrail (ContentGuardPort)
│  ├─ search_strategies/#   simple/local/global/drift + lightrag(mix/hybrid/naive)
│  ├─ retrieval/        #   abstract retriever/strategy bases, strategy_registry, hybrid scorer,
│  │                    #   token/memory managers
│  ├─ ingestion/        #   LLM/IO coupled: chunker, *_extractor, loader, parser,
│  │                    #   translator, gleaner, community_detector, document_converter
│  ├─ renderers/        #   graph visualization renderers
│  ├─ schema_sources/   #   @register_schema_source: extraction-schema loaders (file, owl)
│  ├─ seed_sources/     #   @register_seed_source: seed graph loaders (jsonl, csv, rdf)
│  ├─ source_files/     #   shared readers of a source's files: local/S3 locations, RDF files, checks, parse
│  ├─ evaluators/       #   evaluator bases + langchain/ragas evaluators
│  ├─ store_errors.py   #   is_fatal_retrieval_error: the fatal-vs-degraded policy (§8.1)
│  ├─ providers.py      #   Providers: the per-run bundle of model and store providers
│  └─ provider_registry.py # @register_provider: builders keyed by role and kind
├─ application/         # orchestration + entry points
│  ├─ cli/              #   run-ingestion/rag/eval/visualization/prompt-tuning, prepare-benchmark,
│  │                    #   display (rich console helpers)
│  ├─ ingestion/        #   DataIngestionPipeline + pipeline_stages, IncrementalIndexer,
│  │                    #   namespace lease, extraction-record store, stage cache
│  │                    #   (cache_manager), resume (pipeline_manager; the resume
│  │                    #   point and rewinds in resume) and run metrics
│  │                    #   (pipeline_metrics)
│  ├─ storage/          #   IndexingManager (indexer fan-out) and its collaborators:
│  │                    #   cross_run_merge, removal, reannotation, stored_items
│  ├─ retrieval/        #   rag_chain (GraphRAGChain, RAGInput/Output), its query
│  │                    #   processing (query_processing) and filter-key checks
│  │                    #   (filter_validation)
│  ├─ prompts/          #   PromptTuner (LLM-based corpus profiling)
│  ├─ bootstrap.py      #   the composition root: build_providers, build_query_guard,
│  │                    #   build_checkpoint_sync, resolve_extraction_schema,
│  │                    #   load_seed_graphs
│  ├─ config.py         #   get_config / ConfigLoader: file + environment -> Config
│  └─ observability.py  #   configure_observability: installs the tracing backend
├─ shared/              # leaf kernel (logging, exceptions, metrics, observability hooks,
│                       #   write fence, utils/: hashing, ids, concurrency, scripts)
└─ evaluation/          # real logic package: evaluation_manager / graph_aware /
                        #   retrieval / answer_match, strategy comparison + pairwise
                        #   judge, benchmarks/ (MuSiQue)
```

- **Real logic package.** `evaluation/` (the manager and the LLM-free
  evaluators) is a layer of its own (above). Everything else is imported
  from its real location (`application.*`, `adapters.*`, `domain.*`).
- **Vendor and role packages.** Vendor packages (`adapters/aws/`,
  `adapters/local/`) hold technology bindings (clients, embedded engines and
  stores, model factories, token counters, caches); role packages
  (`storage/`, `retrievers/`, `doc_status/`, `guardrail/`) hold the port
  implementations the application wires. A role package imports a vendor
  package, never the reverse: `SQLiteGraphIndexer` writes to
  `SQLiteGraphStore` as `NeptuneIndexer` writes through `NeptuneClient`. Of
  the adapters, a vendor package imports only the other vendor package and
  the vendor-neutral kernel (`adapters/llm/`, `adapters/observability/`),
  which imports no other adapter package.

### 2.2 Ports (Abstract Interfaces)

| Port | Location | Adapters | Notes |
|---|---|---|---|
| `DocStatusPort` | `ports/doc_status.py` | `adapters/doc_status/dynamodb.py`, `sqlite.py`; `FakeDocStatusStore` for tests | Document status and each document's text units, partitioned by index namespace; the domain's delta reads its `DocStatusReader` subset |
| `ContentGuardPort` | `ports/guardrail.py` (with `GuardVerdict`) | `adapters/guardrail/bedrock.py` (Bedrock `ApplyGuardrail`) | Screens the raw query and the final answer (§14) |
| `GraphIndexer` (write) | `ports/indexer.py` | `adapters/storage/neptune_indexer.py`; local `sqlite_graph_indexer.py` | One contract for full builds and deltas (`upsert_*` / `delete_by_id`) |
| `VectorIndexer` (write) | `ports/indexer.py` | `adapters/storage/opensearch_indexer.py` (on OpenSearch, or on the embedded engine through `SearchEngineClient`) | Same |
| `BaseGraphRAGRetriever` (read) | `adapters/retrieval/base.py` | `adapters/retrievers/{neptune,opensearch,sqlite_graph}_retriever.py` | Retrieval adapters |
| Model factories, token counter | `ports/model_factory.py` (`ModelFactoryPort`, `TokenCounterPort`, the `*ModelInfo` records) | `adapters/aws/bedrock.py`, `bedrock_models.py`, `token_counter.py`; local `adapters/local/{models,embeddings,rerank,token_counter}.py` | Carried by one `Providers` bundle (§15) |

The read-side bases (`BaseGraphRAGRetriever`, `BaseSearchStrategy`) and
`BaseGraphRAGEvaluator` are *adapter bases*: they build infrastructure
(scorer, token manager) in `__init__`, so they live in `adapters/` and
`ports/__init__` only names them.

### 2.3 Role-Based Retriever Injection

Strategies receive retrievers **by role, not by backend**:
`RetrieverRole.GRAPH` (graph expansion) and `RetrieverRole.DOCUMENT`
(vector and lexical lookup). They reach them only through
`self.graph_retriever` / `self.document_retriever`, and the chain's role →
builder map binds the implementation, so swapping the graph backend changes
no strategy code.

```python
# adapters/retrieval/strategy_registry.py
@register_strategy(SearchStrategy.LOCAL, required_roles=(RetrieverRole.DOCUMENT, RetrieverRole.GRAPH))
class LocalSearchStrategy(BaseSearchStrategy): ...
```

The write path has two fixed store roles: `IndexingManager` always writes a
graph store and a vector store, because the fan-out (entities to both stores,
entities before edges, the orphan-edge cascade) is domain knowledge; which
backend fills each role is open (`providers.graph.kind` /
`providers.vector.kind`, or an injected indexer).

### 2.4 Registries

- **Search strategies**: `@register_strategy(...)`
  (`adapters/retrieval/strategy_registry.py`): a class, its required roles and
  query inputs, keyed by `SearchStrategy`; a wrapper (`iterative`) names the
  strategy it runs under a config (`wraps`, §6.3).
- **Providers**: `@register_provider(ProviderRole.X, "kind")`
  (`adapters/provider_registry.py`): the builder `providers.<role>.kind`
  selects, for every model, store and registry role.
- **Renderers**: `@register_renderer("name")` (`adapters/renderers/base.py`);
  **parsers**: `ParserFactory.register_loader(".ext", Loader)`.
- **Schema sources**: `@register_schema_source("kind")`
  (`adapters/schema_sources/registry.py`): the loader
  `processing.graph_extraction.schema_source.kind` selects (§12).
- **Seed sources**: `@register_seed_source("kind")`
  (`adapters/seed_sources/registry.py`): the loader each
  `processing.seed_graphs[].kind` selects (§4.5).
- **Evaluators**: one lazy-import branch per `EvaluatorType` in
  `EvaluationManager._resolve_evaluator_class`.

### 2.5 Dependency Rule Verification and Public API

`tests/unit/test_layering.py` enforces §2.1 on every import of every
module (lazy and `TYPE_CHECKING` ones included, relative ones resolved):
the layer order, evaluation's reach into the application, injected
backends (also inside the adapters: the search strategies, the retrieval
kernel and the ingestion adapters import none), store-free evaluation,
vendor packages as leaves, plugins through their registries,
and the root facade imported only for `__version__`. Importing `pkg.mod`
runs `pkg/__init__.py`, so the imports a package's `__init__` makes count
for every module that imports from the package; outside the facade and the
console-script launcher, a package module is never imported by a name
string (`importlib.import_module`, `__import__`, `pkgutil.resolve_name`,
`importlib.util.find_spec`, also under an alias, through `getattr` or with
`name=`), and a dynamic import of a name the scan cannot read (a variable,
a concatenation, a relative name), `exec` or
`importlib.util.spec_from_file_location` needs an entry saying what it
imports. Its list of allowed exceptions is empty, and each rule has
a synthetic case it must catch. A clean-interpreter check pins the runtime
side, which the scan of edges cannot see (an allowed adapter module passing
a backend on): importing any module of the application (the CLIs and the
composition root aside), evaluation, the search strategies, the retrieval
kernel or the ingestion adapters loads no vendor binding, port
implementation, gremlin-python or opensearch-py.
`tests/unit/test_domain_purity.py` checks the third-party side: no domain
module loads an infrastructure package, even transitively.

**Public API.** Callers outside the package import from `unified_kg_rag`,
whose `__all__` is the supported surface: the names for using the framework
(`Config`, `get_config`, `DataIngestionPipeline`, `GraphRAGChain`, `RAGInput`
/ `RAGOutput` / `SearchOutput`, `EvaluationManager`, ...) and for extending it
(the ports, `Providers` / `build_providers`, the registries). Other module
paths are internal; only a change to the exported names, configuration keys,
CLI flags and exit codes, or stored formats is breaking. Names resolve on
first access (`import unified_kg_rag` loads nothing), and
`tests/unit/test_public_api.py` pins the list.

---

## 3. Domain Model

`domain/models/` holds pure Pydantic models with no infrastructure
dependency:

- `Entity` (`name`, `description`, `type`, `text_unit_ids`, `community_ids`,
  `rank`, `confidence`, embedding fields, the schema annotations
  `in_schema` / `type_iri`, the stated `iri` and the derived
  `source_kinds`), `Relationship` (`source_id` / `target_id`,
  `description`, `weight`, `text_unit_ids`, `in_schema` / `predicate_iri`,
  `source_kinds`; §4.2, §4.5),
  `Community` / `CommunityReport`, `TextUnit`, `Covariate` (claim);
- `DocStatus` (a `PENDING` write-ahead record, then `PROCESSED` / `FAILED` /
  `QUARANTINED`), `DocStatusRecord` (content hash, the document's text units,
  suffix), `DocumentDelta` (new / changed / unchanged / deleted, plus failed,
  quarantined, quarantined-unchanged, deferred and settings-changed ids),
  `DocumentLineage` (the text units one document produced);
- `SearchQuery` / `SearchResult` / `RetrievalResult`, `SearchStrategy` /
  `SearchType` / `RetrieverRole`;
- `SeedGraph` / `SeedNode` / `SeedEdge` (`seed_graph.py`): what a seed source
  reads, before conversion (§4.5).

**Lineage is the core data.** Entities and relationships record the
`text_unit_ids` they were stated in. This lineage, not a token-overlap
heuristic, decides whether an entity relates to a text unit, in any
language, and incremental removal follows it (§5).

**Rank (MS GraphRAG parity).** An entity's `rank` is its degree, a
relationship's the sum of its endpoints' degrees (`domain/ingestion/graph_rank.py`).
A full build sets them; an incremental merge recomputes them for what it
writes over the stored graph's neighbours. Only document-backed edges
count: an edge every text unit of which is a seed graph's adds to no
degree (§4.5), so with no seed graph the rank is the plain degree.

**Index suffix.** Every OpenSearch index and Neptune label is named
`<prefix>-<suffix>[-<additional_suffix>]`: ingestion writes
`processing.document_parsing.index_value`, queries select `RAGInput.suffix`
(`run-rag` / `run-eval`: `--suffix`, else the config's `index_value`), all
`default` when unset; one suffix per tenant or corpus version.

**Display names and identity keys.**

- `Entity.name` keeps the display form the source used, only trimmed and
  whitespace-collapsed (`clean_display_name`), so context and reports quote
  "$1,000 penalty" or "C++" verbatim.
- Ids hash `entity_key(name)` (`shared/utils/common.py`): NFKC, casefold, `_`
  and `-` as spaces, quote marks and commas dropped, whitespace collapsed,
  trailing sentence punctuation stripped, **every other symbol and the
  letters and digits of all scripts kept**. "ACME  Corp." and "Acme Corp"
  share an id; "C++" and "C#" do not; a non-empty name never yields an empty
  key. Every exact-name match (extraction, gleaning, claim resolution,
  incremental merges) and the fuzzy matcher's shingles use the same key.
- **Changing the key changes every id**, so it requires a full re-index.

---

## 4. Ingestion Pipeline

`DataIngestionPipeline` (`application/ingestion/pipeline.py`) runs 12
checkpointed stages, so a re-run resumes after the last valid one. Extraction
works per text unit and the graph is resolved from the merged per-unit
records, so a full build does not depend on batching and an incremental run
can re-derive what it touches (§5).

![Ingestion pipeline](../assets/ingestion-pipeline.drawio.png)

| # | Stage | Module | Notes |
|---|---|---|---|
| 1 | Document parsing | `parser.py` (`ParserFactory`) | PDF/TXT/CSV/JSON (+ MD/HTML with the `unstructured` extra) |
| 2 | Document loading | `loader.py` | optional MinHash deduplication; the incremental diff |
| 3 | Chunking | `chunker.py` | intelligent (LLM boundaries, default) or simple |
| 4 | Translation | `translator.py` | optional; skipped when source = target language |
| 5 | Graph extraction | `graph_extractor.py` | entities and relationships per text unit |
| 6 | Gleaning | `gleaner.py` | optional; re-asks a unit while its last answer added something |
| 7 | Graph resolution | `graph_resolver.py`, `description_summarizer.py` | merges name variants, re-summarizes long descriptions (calls the LLM) |
| 8-9 | Claim extraction and resolution | `claim_extractor.py`, `claim_resolver.py` | optional (off by default) |
| 10 | Graph analysis | `graph_analyzer.py` | graph build and statistics |
| 11 | Community detection | `community_detector.py`, `domain/ingestion/community_clustering.py` | hierarchical Leiden (the clustering is pure, in the domain) and structured reports |
| 12 | Indexing | `application/storage/indexing_manager.py` (+ `cross_run_merge.py`, `removal.py`, `reannotation.py`) | graph and vector stores; full build or delta |

The stage order lives in `DataIngestionPipeline.STAGE_CLASSES`; every stage
gets its models and stores from the run's `Providers` bundle.

### 4.1 Extraction records and the extraction store (stages 5-8)

- **One record per text unit.** Extraction and gleaning write one
  `TextUnitExtraction` per unit (`domain/models/extraction.py`): the
  entities and relationships that unit states, in its own words, plus the
  entity keys rejected for it. Gleaning edits one record at a time: its
  prompt lists the unit's own items, and a correction never reaches what
  another unit says (MS GraphRAG and LightRAG likewise glean each chunk's own
  extraction). It lists the entities with their types and gives the
  extraction prompt's entity and relationship type blocks and its 1-10
  confidence and strength scales, so a gleaned item is typed and scored as
  an extracted one (a missing score takes extraction's default too).
- **Output format and gleaning (deliberate divergences).** Extraction asks
  for XML records (`GraphExtractionPrompt`): each entity with its type,
  description, a 1-10 `<confidence>` and a verbatim `<source_text>` span,
  each relationship with a strength and its span, where MS GraphRAG and
  LightRAG parse delimited tuples. The span is what entity grounding checks
  an item against its chunk with, and the confidence what
  `entity_confidence_threshold` filters on. Gleaning is a refinement prompt
  of its own (`GraphRefinementPrompt`), sent without the extraction's
  conversation: it lists the unit's text and current items and asks for
  missing items and corrections. Upstream continues the extraction
  conversation ("MANY entities were missed ...") with its history. A round
  here costs one prompt of the unit's text and items rather than a growing
  conversation, and it can correct an item (`ENTITY_CORRECTION`,
  `RELATIONSHIP_CORRECTION`), not only add one.
- **Resolution starts from the merge of all records**
  (`domain/ingestion/contributions.merge_contributions`: items merged by id,
  weights kept per unit), so the merged graph does not depend on which units
  shared a batch. Claim prompts list the unit's own entities, so a unit's
  claims depend on the unit alone. A claim's subject and object resolve
  only against the entities its own text units stated (`ClaimResolver`, in
  a full build as incrementally): a claim whose subject its units' entity
  extraction did not list is dropped, and an object not listed there is
  kept as the literal name the claim states (`object_id` empty), though
  another document states that entity. Each unit's statement is resolved
  on its own, and resolution merges the statements that resolve to one
  claim id.
- **The extraction store** keeps each record, keyed by
  `<fingerprint>:<SHA-256 of the processed text>`, in the vector store's
  extraction-records index. The fingerprint
  (`cache_keys.unit_record_fingerprint`) covers the models, the extraction,
  gleaning or claim settings, the target language and the prompt templates in
  effect. A unit whose text and settings have a stored record takes it
  instead of an LLM call, so an edited document re-extracts only its changed
  chunks, and identical chunks elsewhere or in a rebuild reuse theirs.
  `--force-rebuild` re-extracts and refreshes every record; an unreadable
  store is logged and the run re-extracts. A record that cannot be written
  fails its unit (`records_not_stored`), so its document is recorded `FAILED`
  and extracted again: incremental runs derive from the records (§5.4).
- **A text unit names its stored records** in `attributes.extraction_records`,
  computed from the run's records when the unit is indexed (not set on the
  cached units), so a resumed run writes them too.
- **A record stays while a stored text unit names it.** A full build or
  reset whose text units were all written keeps only the records they name
  (`VectorIndexer.retain_extraction_records` copies them into a new index and
  swaps the alias); an incremental run deletes the records only removed text
  units named.
- **Chunk boundaries are records too.** The intelligent chunker stores each
  pre-chunk's boundary answer in the same index, keyed by
  `<cache_keys.chunk_boundary_fingerprint>:<SHA-256 of the pre-chunk>` (the
  every-call and chunking settings, the chunking model and the boundary
  prompt), and the text units cut from it name the record (`chunks` in
  `attributes.extraction_records`), so it is kept and removed like the
  others. A pre-chunk with a stored answer takes it instead of a model call:
  a reset of an unchanged corpus asks the boundary model nothing, and a
  model that answers otherwise each time (no temperature control) cannot
  move the boundaries, and with them every downstream cache key, of
  unchanged text. The stage cache could not hold them: it is per pipeline
  id and keyed by the whole corpus. Only an answer that cuts its pre-chunk
  is stored: one it falls back from (lines that do not exist, a miss rate
  over the limit) would split it by size on every later run, so it is asked
  again, and a stored one of that kind is replaced. `--force-rebuild` asks
  again.
- **A full build reuses the corpus-wide answers of unchanged input.** The
  LLM response cache keys a call by its prompt and model, which are stable
  (community ids are content ids), but it lives in the run's cache
  directory: a reset under a new pipeline id starts it empty, so before
  this every reset summarized every long description and reported every
  community again. When the run's graph is the corpus's (no registry, a
  reset, or no unchanged document), graph resolution reads the stored
  graph copies of the items over the summarization threshold (one read
  per suffix) and keeps a summary whose `summarized_from` is the hash of
  the item's description and whose `summarized_with` is
  `cache_keys.description_summary_fingerprint` (every-call settings, the
  summary model, the summarization section, the prompt), so a reset after
  a summarization setting change summarizes again; a copy the graph capped
  is not reused. The report answers are extraction-store records keyed by
  `<cache_keys.community_report_fingerprint>:<SHA-256 of the report input>`
  (the chain input as packed, with the community's name, which titles the
  report), stored with their generation time and named by the report
  (`attributes.report_record`), so the rebuild's retention keeps the ones
  its reports name. A stored answer gives the same report a call would
  (rank and lineage are recomputed), and its vectors are reused by digest.
  An incremental run that leaves documents unchanged neither reads nor
  writes them: its communities cluster the delta, which a full build
  clusters otherwise, so the first reset after incremental runs asks for
  those reports. `--force-rebuild` asks
  again and refreshes the records.
- **Gleaning stops on observed output**: a unit is re-sent (up to
  `max_rounds`) only while its previous answer added something new. An empty
  answer is the "nothing missing" signal MS GraphRAG asks for with a separate
  yes/no prompt, so no call is spent on it. `max_rounds` stays 3, against
  MS GraphRAG's 1, for the measured graph recall (§9.1).

### 4.2 Graph identity and merging (stages 5-7)

- **Entity identity (deliberate divergence).** An entity id hashes
  `entity:{entity_key(name)}`: the type is not part of it, so a name stated
  with two types is one entity, typed by the most frequent of its statements
  (document statements first, the smallest type on a tie;
  `contributions.merge_entity_contributions`). MS GraphRAG groups extracted
  entities by name and type; LightRAG keys them by name, as here.
- **Relationship identity (deliberate divergence).** A relationship id hashes
  `relationship:{entity_key(source)}:{entity_key(target)}:{type}`, lowercased,
  so the model-written `type` is part of the identity and the edge is
  directed: two types between one pair are two edges here and one in MS
  GraphRAG and LightRAG (the community graph still joins them, §4.3). The
  parts are joined with `:` unescaped, so a `:` in a name or type can collide;
  the format is kept because fixing it would re-key every stored relationship.
- **Endpoints cite their relationships' text units (deliberate
  divergence).** An entity also cites the text units of each relationship it
  is an endpoint of, and an endpoint no unit listed becomes a stub entity
  citing them (`relationship_endpoints.cite_relationship_endpoints`). Upstream, an
  entity's text units depend on which chunks were extracted together; here
  they depend only on the chunks that name it, so an incremental run cites
  what a full build cites and a removed document strips an entity to exactly
  the surviving text units that name it. The cost: local search and
  community lineage also reach chunks that mention an entity only through a
  relationship. With entity grounding on, gleaning drops a relationship naming
  an entity rejected as ungrounded for its chunk, so no stub brings it back.
- **Entity merging.** Two names merge when their `compact_name_key` is equal:
  `entity_key` without a short, fixed list of legal-form designators (`(주)`,
  `주식회사`, `株式会社`, `有限公司`, a trailing `Inc` / `Ltd` / `LLC` / `Co` /
  `Corp`, ...) and without spaces next to Han, Hangul or Kana, so
  "(주)가나다", "가나다 상사" / "가나다상사" and "Acme Inc" / "Acme" merge. A
  merged group takes the name cited by the most text units (ties: the
  smallest), so the result does not depend on input order.
- **Fuzzy matching is opt-in** (`processing.fuzzy_entity_resolution`), because
  name similarity also joins different organizations that share most of their
  name ("Northfield University" / "Northfield University Hospital"). When it is
  on, names link above `processing.similarity_threshold` only when their
  *discriminator tokens* (tokens with a digit, single letters, Roman numerals)
  are identical, their types are compatible, and, for names ending in Han,
  Hangul or Kana, their final (head) characters are equal ("가나다연구원" and
  "가나다연구소" stay apart). A type-aware union-find never groups two
  differently typed names. Without fuzzy matching, full and incremental
  builds group the same names (both merge by compact name key). With it, the
  incremental merge uses the same matcher but folds each delta entity into
  its one best-matching stored entity and does not join stored entities with
  each other: a delta name that links two stored names a full build chains
  into one group ("Northfield Medical Centre" between "... Center" and
  "... Centre Annex") merges with one of them only, until
  `indexing.reset: true` regroups the corpus.
- **Schema annotations are written, not extracted.** `IndexingManager`
  sets `in_schema` and `type_iri` on each entity and `in_schema` and
  `predicate_iri` on each relationship it writes, on every write path (full
  build, delta, the strip of a removal, the stored edges it rewrites for
  their names or rank), from the item's stored type and the schema
  (`domain/ingestion/schema_annotation.SchemaAnnotator`, matched by
  `schema_key`, so the lowercased type vote still matches). A pure function
  of the merged type, it gives a full build and an incremental run the same
  values; pattern conformance is not stored, since it depends on endpoint
  types other documents can change. Where the schema does not apply
  (`SchemaConformance.applies`: `off`, `align` with entity types alone),
  `in_schema` is `None`; where it applies, both kinds get it, and every item
  of a kind with no declared type is in schema (nothing of it is off schema,
  so `in_schema: true` never empties a kind); an IRI is written
  whenever the stored type names a declared type with an `iri` and the mode
  is not `off` (so `align` with entity types alone still links them), and is
  `None` otherwise. `None` is never written: models drop it from their dumps
  (`SchemaAnnotated`), extraction records never hold annotations, the
  Neptune codec and the OpenSearch documents omit it. A default config (no
  `iri` anywhere) therefore writes no annotation field on its items; only the index mappings gain
  `in_schema` (boolean) and the IRI keyword fields, added to live indices
  by the incremental preflight (`check_index_settings` →
  `_add_missing_fields`). Ids are unchanged; the IRIs are the seam for
  linking items to an external ontology. A schema edit that changes only
  the annotations (the digest, §12 "Extraction settings") is applied by
  `IndexingManager.reannotate`: each store lists its items' annotations
  with their types (`read_schema_annotations`; a relationship's vector copy,
  which holds no type, takes its edge's), and only those that differ are
  rewritten in place (`write_schema_annotations`: Neptune and the SQLite
  graph drop and set the two properties, OpenSearch a partial update with
  null for an unset one), so the stores end as a full build leaves them.
  Neptune lists the ids in one traversal and reads the annotations in
  `batch_size` batches; OpenSearch pages by id. A store that inherits the
  port's defaults (`rewrites_schema_annotations`, which compares the
  functions its methods are bound to, so a forwarding wrapper answers for
  the store it wraps) makes the pass read nothing; a store error ends the
  pass with a warning, not the run.
- **Merged descriptions** are concatenated, so `DescriptionSummarizer`
  re-summarizes those over a token budget with a cheap model (map-reduce when
  one exceeds a call's input); a failed call keeps the concatenation. The
  trigger is a cost variant of upstream, not parity: MS GraphRAG
  `summarize_descriptions` summarizes every item with more than one
  description and LightRAG also at `force_llm_summary_on_merge` (8)
  fragments; here only the token budget triggers it.

### 4.3 Communities (stages 10-11)

- **Community graph.** `GraphBuilder` builds an undirected graph with one edge
  per entity pair; relationships of other types or directions between the pair
  share it, with the summed weight, so Leiden sees the pair's total strength.
- **Hierarchy (deliberate divergence).** Level 0 is one Leiden partition of the
  whole entity graph; each coarser level partitions the graph of the previous
  level's communities, up to `max_levels`. Because modularity's resolution
  limit grows level-0 communities with the corpus beyond what one report can
  describe, a community over `max_cluster_size` (default 20) members is
  partitioned again on its own subgraph. Level-0
  communities under `min_community_size` merge into their strongest neighbour.
  A connected component smaller than `min_community_size` (an isolated
  entity, a pair) gets no community or report; only these are skipped, unlike
  MS GraphRAG's `use_lcc`, which drops every component but the largest. On an
  incremental run, whose graph is only the delta, a small component is
  clustered when it reaches `min_community_size` in the whole graph a full
  build would cluster: walking the stored relationships that cite text units
  of documents the run neither processes nor removes, at most
  `min_community_size` batched reads. Stored entities without a community it
  links to stay without one until a rebuild (`community_drift`
  `unclustered_entities`). A
  parent identical to its only child is never created, so each distinct
  community is reported once. A Leiden failure fails the stage instead of
  truncating the hierarchy.
- **Determinism.** Leiden's partition depends on insertion order and
  `graspologic_native` returns labels in a per-process order, so the clustering
  (`domain/ingestion/community_clustering.py`) sorts nodes and edges and
  renumbers each level by its smallest member id:
  the same graph and `random_state` always yield the same communities and ids.
- **Reports (MS GraphRAG parity, two deliberate deviations below).** A report's input takes the community's
  entities by degree (document-backed, §4.5), capped at
  `max_entities_per_report`, and packs them into `max_report_context_tokens`
  edge by edge, as MS GraphRAG's `sort_context`: the relationships among them
  by combined endpoint degree (then weight and id), each with its endpoints
  and, with claim extraction on, the claims about them; one that does not fit
  is left out and the next tried; the entities with no relationship among
  them come last; the top entity always fits. Packing the entities first left
  the relationships whatever budget remained (with 200-token descriptions, 2
  of 40 edges). MS GraphRAG's 8000-token budget with `max_cluster_size` 10 is
  the tuning point for larger reports: the defaults (4000, 20) are kept for
  cost. Reports are generated bottom-up. The prompt returns a structured
  report (summary, 0-10 rating with explanation, findings); `full_content`
  for embedding and search is rendered from it, with no second call.
- **Sub-community roll-up (deviation).** When a parent's own context was
  cut, its children's report summaries (biggest child first, each with its
  rating) are added to that context under a second budget of
  `max_report_context_tokens` (`enable_sub_community_rollup`), so the
  prompt stays within about twice the raw budget. MS GraphRAG substitutes
  sub-community reports for the raw context that does not fit; adding them
  keeps the parent's best-connected entities and relationships verbatim, at
  up to twice the input tokens of a parent that overflows.
- **Report rank (deviation).** A report's `rank` is the sum of its members'
  document-backed degrees (`_compute_report_rank`, no model call; the member
  count without a graph), where MS GraphRAG's is the model's rating. The
  model's rating is kept as `rating` (in the report text, and a filter key
  of the reports index); global search's map step rates each report against
  the query instead. `rank` orders a search of the reports index without
  query text.
- **Report lineage.** Each report carries `text_unit_ids` and `document_ids`
  (the union over its members: provenance, not citation support) and its
  place in the hierarchy, which global search selects by (§6.1.1).

### 4.4 Pipeline infrastructure

- **Resume.** Each stage's output is checkpointed under the run's pipeline id
  (`application/ingestion/pipeline_manager.py`, `application/ingestion/cache_manager.py`); the stage cache
  key is described in §12. Where a resumed run starts, and the rewinds that
  move it back, are in `application/ingestion/resume.py`.
- **S3 sync** (`adapters/aws/s3_cache.py`) uploads the changed files after
  every stage, data before the metadata that refers to it. On SIGTERM,
  `run-ingestion` stops the running stage (`request_stop`: deferred while a
  checkpoint is written, no I/O in the handler) and uploads its checkpoint.
- **Reuse across runs.** Within what a run writes, a text embedded before
  with the same embedding provider, model and dimension keeps its stored
  vector (each vector stores a digest of its text, §5.2), so a changed
  document re-embeds only the chunks, descriptions and reports whose text
  changed, and a reset of an unchanged corpus embeds again only what it
  regenerates. The intelligent chunker's boundary answers are kept per
  pre-chunk in the extraction store (§4.1), so a reset or an edit asks the
  model only for pre-chunks whose text or chunking settings changed, and an
  unchanged pre-chunk keeps its boundaries even with a model that answers
  otherwise each time. Only an answer the chunker used is kept: a pre-chunk
  split by size instead (the answer named lines that do not exist) asks the
  model again on the next run. A full build (a reset, a first run, or a run
  without the registry) does the same for the corpus-wide answers: a
  description over the summarization threshold whose stored copy was
  summarized from the same text with the same summarization settings keeps
  that summary, and a community whose report input (its members,
  descriptions, edges and claims as packed, its roll-up text and name) and
  report settings are unchanged takes the stored answer (the extraction
  store keeps it, and the report names it in `attributes.report_record`). A
  community's name (`Level 0 Community 1a2b3c4d`) derives from its members,
  so a document that adds a community leaves the others' names, and their
  stored answers, as they were.
- **LLM response cache** (`adapters/llm/response_cache.py`): a LangChain
  `BaseCache` on every ingestion model, keyed by the rendered messages and the
  model and generation settings, stored with the stage cache. An earlier
  run's entry is served at most once per run and truncated responses are not
  kept, so a retry after a bad response asks the model again. A
  `BatchProcessor` item that failed every attempt drops the responses its
  calls were served or stored (`discard_if_item_fails`), so a re-run asks
  again for that item only instead of replaying them; so does an item whose
  answer the stage rejects afterwards (a graph or claims record that fails
  to build), through the `discards` the batch hands back. A call abandoned on
  its call timeout (`call_timeout_seconds`) is neither served nor stored: its late answer
  is not its item's, which was retried or failed meanwhile.
- **Output fixing.** Structured outputs are parsed with an output-fixing
  parser (the model of `models.roles.fixing`, always on). The XML parser unwraps one
  element wrapping the expected sections and reads a section holding only
  "None" (or "N/A", "null", "-") as empty; the extraction, claims and
  gleaning parsers raise `OutputParserException` for any other answer that
  parses but is not the prompt's shape (no section, a section holding
  prose, nameless entities in a complete answer), so the fixer repairs it
  as it repairs output that does not parse.
- **Resources.** The pipeline releases indexer and client resources in
  `close()`, which `run-ingestion` calls in `finally` (§8.2).

### 4.5 Seed graphs as virtual documents

A seed graph (`processing.seed_graphs`,
[Seed graphs](./guides/seed-graphs.md))
is a knowledge graph the user brings. It is imported as virtual documents
of the run's own corpus rather than through a second write path, so
lineage, removal, derivation, the extraction store and the write-ahead
protocol apply unchanged, and an incremental run converges to a full build
of the documents plus the configured seeds. No model reads a seed.

- **Load and convert.** `DataIngestionPipeline.__init__` calls
  `bootstrap.load_seed_graphs` right after `resolve_extraction_schema`,
  before any model is built: each item's registered loader returns a
  `SeedGraph`, and `convert_seed(graph, SeedConversionSettings, schema)`
  (`domain/ingestion/seed_graph.py`, pure) turns it into shard documents.
  It checks (duplicate ids, dangling edges, the caps, which fail instead
  of truncating), maps types (a `type_iri` / `predicate_iri` a schema type
  declares, the last-declared winning, else the label through
  `SchemaConformance`), conforms the whole seed at once (both endpoint
  types are known, so patterns always apply; under `qualify` each
  identity of a shared name, its IRI else its id, is checked by its own
  type, as the distinct entity it becomes), applies
  `on_name_collision`, shards nodes by `int(sha256(id)[:8], 16) % shards`
  (an edge goes with its source node), packs node cards (a node and its
  outgoing edges, by `(entity_key(name), id)`) into units of at most
  `max_items_per_unit` statements (a hub card is split; a shard over 2,000
  units fails naming its largest card's node, keeping a shard's registry
  record small) and renders
  `seed_graphs/<name>/<k>-of-<n>.jsonl`: a header line (name,
  `SEED_FORMAT_VERSION`, filters), then the units' canonical JSON lines,
  a blank line before each unit. The content hash therefore covers the
  name, the filters, the format, the schema's effect and every statement,
  and the output does not depend on the input's order. Every check is
  linear in the seed (an edge's endpoints are dict lookups). The models
  refuse what a store cannot write: a lone surrogate in any text, a name
  over 1,000 characters, an IRI over 4,000 or a type over 256, so each
  stays far under an indexed keyword's 32,766 bytes.
- **Keying.** `processing.seed_graphs` is neutral in `CONFIG_KEYING`: the
  conversions' digests are folded into the corpus fingerprint
  (`cache_keys.seed_corpus_fingerprint`), which keys every stage, and
  nothing enters `extraction_settings`, so where a seed is read from keys
  nothing and a seed edit re-extracts no document. Without seeds the
  fingerprint is the one it was.
- **Stages.** Loading appends the shards after deduplication
  (`assign_document_identity` keeps their path), so the registry diffs
  them like files. Chunking cuts them with `seed_text_units` into `sg-`
  units (the chunker's stable id, prefixed: they sort after every hex
  document unit id, and `is_seed_unit` is the one test anywhere).
  Translation, extraction, gleaning and claims leave them out; a seed
  unit's record is `parse_seed_unit(unit)`, a pure function of its text,
  stored under `SEED_RECORD_FINGERPRINT` (the format only, never a model
  setting), and its text unit names only that graph record. A failed
  report never fails a seed unit. With seeds configured an empty corpus
  is a valid run.
- **Document-first merge.** Every merge path (the records' merge, graph
  resolution, the incremental merge, derivation) applies one rule keyed
  on `is_seed_unit` (`document_first`): the type vote counts document
  statements and falls back to seed votes only when none gives a type;
  a resolved group's name is keyed `(-document units, -seed units,
  name)`, and its type is the named member's unless that is unknown
  (none, blank, `unknown`), then the known type the most units cite; confidence is the maximum over document statements when any
  exist. Descriptions stay a union, document lines first because records
  merge in unit-id order. `Entity.iri` is a stated field, kept in
  records, merged as the smallest non-null value (groups with several
  are counted). Seed properties live in `attributes.seed_properties`,
  merged per key, never over an extracted attribute; the OpenSearch
  entities index leaves them unmapped (`enabled: false`), so no key or
  value fails a write and none is filterable, and an incremental strip
  compares them (`_vector_content`), as a lineage update leaves stored
  attributes as written. The rules hold per
  name key: names that only resolve together (a legal-form variant, or
  fuzzy matching) let seed units break a tie between document spellings,
  and a document entity named only as a relationship endpoint (no type)
  takes the seed's type and confidence, as a gap filled.
- **Source kinds.** `annotate_sources` sets `source_kinds` where indexing
  sets the schema annotations: `None` for an item no seed unit cites,
  else the sorted kinds (`["document", "seed_graph"]` or
  `["seed_graph"]`). It is derived, left out of records, and written only
  when set, so an item no seed states is stored byte for byte as before
  and nothing is embedded again. Seed text units carry a `source_kind`
  keyword (no filter key). Graph edges store the list as a sorted JSON
  string (`encoded_source_kinds`); `SEED_ONLY_SOURCE_KINDS` is the one
  encoding of a seed-only edge.
- **Ranks.** `with_graph_ranks(counts=is_document_backed)` counts an edge
  unless every text unit it cites is a seed unit, in the full build,
  derivation and both incremental rank paths, which read stored
  neighbours through `GraphIndexer.read_neighbor_ids(document_backed=True)`
  (Neptune `bothE().not(has("source_kinds", '["seed_graph"]')).otherV()`;
  the SQLite graph and the fakes apply the same predicate; an edge
  without the property is document-backed). `count_neighbors` keeps every
  edge: it sizes the community walk, which clusters seeds too. The
  community detector ranks and selects report entities by
  document-backed degree, while Leiden sees every edge at its full
  weight. A large catalog seed therefore enriches the corpus's entities
  without turning its hubs into the top-ranked entities. Query-time
  `entity_importance_source: degree` counts the live edges, seeds
  included.
- **Communities.** A community whose text units are all seed units is
  stored without a report and counts as neither generated nor failed
  (`communities_seed_only`); a parent with one evidenced child is
  evidenced, and a child without a report leaves the parent its raw
  context. An incremental run's graph holds its delta only, so a
  community seed units alone state there is checked against the stored
  members (`_evidenced_by_stored_members`, one `read_entities` batch per
  suffix): a member citing a document unit the run does not replace, or
  stored without lineage and without seed-only `source_kinds`, evidences
  it. A seed added beside unchanged documents thus reports their
  communities as a full build does: the report reads each such member's
  stored description followed by the run's lines, and a member citing a
  unit the run replaces (an edited seed's earlier shard) has its stored
  description derived again from the records of its other units first,
  as the indexing's strip does (`stripped_descriptions`, one read of
  those units' records). The report then reads what the run indexes:
  that derived description, and the merge with the run's lines, are
  each summarized when over the threshold (`needs_summary`), as the
  strip and the merge summarize them, and the summaries go to the
  indexing through `PipelineContext.member_summaries`, marked
  `summarized_from`, so `reuse_summaries` takes them and the model is
  asked once (a resumed indexing summarizes again; the context does not
  persist them). A summary the indexing reuses, or an item already holds,
  is not summarized again even when it is over the threshold itself, so
  the run stores the description its report read. Without that, a hub's report entry was every line its
  documents state and could crowd it out of its own report. A full build
  summarizes the member's lines in one pass, so the two summaries are
  equivalent, not equal; with summarization off both read the lines
  whole (no cap: a cap would make the incremental input differ from the
  full build's). A member the run states with neither a type nor a
  description (the endpoint of a seed edge into a kept shard) reads its
  stored copy's type and description; an unread member store counts the
  communities in `reports_failed`, not `communities_seed_only`. The
  relationship lines are the run's graph's
  only (the delta's seed edges; a stored document edge between members
  is not read): clustering the delta only, the community itself can
  differ from a full build's. Global search answers from the
  reports, so content only seeds state is in none, while a mixed
  community's report describes its seed members too.
- **Scope (decided).** The shards belong to the source scope of the run
  that configures them. A separate `seed:<name>` scope per seed was
  rejected: every run-level registry path (the diff, attempts,
  write-ahead, commit, the per-scope extraction-settings record) assumes
  one scope per run, a reset clears every scope of the namespace, and a
  registry-off build has no scope at all. A seed-only scope is still a
  run whose corpus is only seeds, with a fixed `source_scope`. The
  trade-off: one seed configured on two scopes of a namespace is imported
  twice (the same ids, seed edge weights counted twice).
- **Model settings never re-import a seed**
  (`delta_detector.keep_seed_settings_unchanged`, in `detect_delta` and
  `delta_still_applies`): a shard the diff lists only as
  `settings_changed` is unchanged, since no model setting changes what it
  states; re-importing would only delete and re-embed its items. Its
  record keeps the earlier fingerprint, so later runs exempt it again. A
  schema change that changes the converted statements changes the content
  hash and re-imports the shard.
- **Reset.** A reset (or a registry-off full build) processes the whole
  corpus, configured seeds included, so the sweep keeps them. Before any
  write, `incremental._confirm_reset_of_seed_scopes` lists the namespace
  once (the quarantine check reads the same listing), counts as the run's
  own every record of a spelling of its source scope
  (`normalize_source_scope`, as the incremental diff's `_stored_spelling`
  does) and, when another scope holds seed paths (`is_seed_path`), or the
  corpus is seeds alone, the scope holds seed records of its own and
  another scope holds documents, asks the
  pipeline's `confirm` (the CLI's terminal prompt, `--confirm-reset` /
  `UKR_CONFIRM_RESET`, or `--yes`); a no, or no `confirm`, raises
  `ConfirmationDeclinedError`, which the stage and the pipeline pass on
  untouched (a cancellation, not a failed stage). Refusing every such
  reset was rejected: it would make the common case, a seed on the
  corpus, impossible to rebuild. The answer has its own input because
  `--yes` also accepts a resume's integrity issues, which an ECS task
  answering a reset must not accept blindly.
- **Empty corpus.** A corpus of seeds alone is a seed-only run only for a
  scope the registry holds no document of
  (`incremental._refuse_seeds_over_documents`, before any write, also
  under a reset): otherwise an unmounted volume or an empty S3 sync would
  delete the scope's documents with exit 0. An incremental run reads the
  scope's records from its delta (every record of the scope a seeds-only
  corpus does not match is `deleted` or `failed`), not the namespace. A
  reset from seeds alone of a scope with no record at all, beside other
  scopes' documents, fails too (`_refuse_seeds_over_other_scopes`): a
  seed-only scope's earlier runs recorded its seeds, so this is a new,
  moved or mistyped prefix, and the reset would rebuild the namespace
  from the seeds. Keyed on the registry, not on a flag: a seed-only scope
  needs no extra setting, and the mistake it guards against carries none.
  A scope's first run warns when another scope already holds a seed of
  the same name (its weights would count twice). A registry-off build
  cannot tell a seed-only corpus from a missing one, so
  `DocumentLoadingStage._confirm_seeds_alone` asks `confirm` before seeds
  alone replace an index that holds items (`holds_any_item`, one bounded
  read per store, and the vector store's `holds_any_text_unit`, so a
  corpus that stated no item counts; unknown or unreadable counts as
  holding). An index that
  holds none builds unasked: a first seed-only build loses nothing, and
  asking there would demand a flag for nothing.
- **Not in v1.** IRI-based identity and aliases, `owl:sameAs`, a
  retrievable fact card (seed units as TEXT context), and federation
  (§15, *New seed source*).

---

## 5. Incremental Indexing

With the doc-status registry on, a run diffs the corpus against the registry
by content hash and processes only new, changed and failed documents. The
registry lists each document's text units, and removal asks the stores what
cites them. A write-ahead protocol lets the next run repair any
interruption, and the result converges to a full build of the current corpus.
How to use it: [Incremental indexing](./guides/incremental.md).

![Incremental indexing](../assets/incremental-indexing.drawio.png)

### 5.1 The steps

Names are `IncrementalIndexer` methods (`application/ingestion/incremental.py`)
unless noted.

1. **Delta detection** (`domain/ingestion/delta_detector.py`).
   `DocStatusPort.diff(incoming, scope)` classifies documents as new, changed,
   unchanged or deleted by content hash. A `doc_id` is
   `<namespace>#<hash of namespace, source scope and relative path>`, and the
   namespace is the registry partition, so a run reads only its namespace (one
   projected DynamoDB `Query`) and two corpora or tenants never share a record.
   Only records of the run's scope are deletion candidates; files that failed
   to parse or load are reported `failed`, not deleted. A moved corpus's old
   records are never adopted (a move cannot be told from another host's corpus
   on the same path) until a run retires the old scope
   (`indexing.retire_source_scopes`). Documents whose processing keeps killing
   the run are quarantined or deferred (§5.3). A seed graph's shards are
   documents of the run's scope; one only an extraction setting changed
   for reads unchanged (§4.5).
2. **Write-ahead** (`write_ahead`). Before any store write, every new and
   changed document is recorded `PENDING` (content hash `"pending"`, which no
   document hashes to) with the union of its stored text units and the ones
   this run will write, in one batched write. A record over the registry's
   item limit (DynamoDB 400 KB, about 10,000 text units) fails the run here,
   before anything is written.
3. **Stale cleanup** (`remove_changed_and_deleted`). The text units of changed
   and deleted documents are removed in one pass per suffix (step 5), so an
   artifact only they state goes too. Deleted documents are first recorded
   `PENDING`, so one restored before its removal finished reads changed.
4. **Delta upsert** (`commit`). The stored entities and relationships the
   delta touches (found by compact name key, every stored name with fuzzy
   matching on) and the stored edges of its entities are read back, the delta
   is merged into them and their content derived from the extraction records
   (§5.4); delta claims merge with the stored claims of the same id. Neptune
   writes with an idempotent `coalesce(unfold, addV)` upsert, OpenSearch by id
   into the live alias index; old ids of re-keyed items are deleted only after
   the writes succeeded. Before the first write the commit notes the
   entities it splits, re-keys or rewires in one record of the suffix's
   extraction store (`pending-commit`), deleted once the old ids are: those
   decisions live in memory only, and the next commit of the suffix derives
   the noted entities, their stored edges and the claims about them again,
   so a killed commit's half-written split is finished. It also retires
   the noted entities' copies only the vector store holds, with the vector
   edges at them (the killed commit got to its vector writes only), and a
   run that only removes in the suffix reads the note all the same.
5. **Deletion propagation** (`IndexingManager.remove_text_units`). The stores
   are asked what cites the removed text units (`GraphIndexer.find_ids_citing`
   for entities, `VectorIndexer.find_documents`); edges are found among the
   cited entities' incident edges, and only a resumed removal (some removed
   text units already gone from the vector store, or a vector store that
   cannot tell) looks them up by text unit with
   `find_relationship_ids_citing`, on Neptune a scan of the suffix's edges.
   An item citing only removed text units is
   deleted from both stores, the vector copy first (a retry finds what to
   delete through the graph, so an interrupted run leaves a graph copy, not
   an orphaned vector copy); one citing others too is stripped of them, its
   weight recomputed and its content derived again from the text units left.
   A strip that merges or re-keys items (another spelling of a stripped
   entity's name, which need not cite the removed text units), or derives
   claims under new ids, notes before its first write the entities on both
   sides of the merge, the edges it re-keyed or dropped, the claims about a
   split's entities and the new claim ids, in one record of the
   suffix's extraction store (`pending-strip`, with those of a note a retry
   finishes) and deletes it once their old ids are gone; every removal reads
   it, so a resumed one strips them again. The claims about the other stored
   spellings of a re-keyed, renamed or split entity's name are derived again
   too: one may resolve to it now.
   Before writing them, a strip also notes the items it summarized or wrote
   unsummarized (`stripped-summaries`, with those of a note a resumed run
   reads), whose stored summary the commit summarizes again from the lines
   rather than extending it; the note is deleted once a commit's writes
   succeed, so a run that dies between its removal and its commit still
   converges with a full build.
   A community or report goes once none of its text units is stored. A failed
   removal keeps the deleted documents' records for a retry and fails the
   stage (before the commit when the delta has changed documents).
6. **Registry update** (`commit`). The `PENDING` records are replaced, in one
   batched write, by `PROCESSED` or `FAILED` records with each document's text
   units, scope and relative path (the five causes of `FAILED`:
   [runbook](./operations.md#failed-documents)). `failure_count` counts
   consecutive failures of the same content; after
   `indexing.max_document_failures` the document reads unchanged. A run that
   stops before this step at a stage that failed on all of its units still
   records FAILED the documents whose units failed for an error of their own,
   so a document that always fails cannot block every later run. Only such
   errors count, here and at this commit: unparseable or cut-off output, and
   a request the model rejects for its content (the factory's
   `is_item_error`: a Bedrock `ValidationException`, HTTP 400/413/422, minus
   rejections of the model or a setting). Any other error (an outage,
   credentials, a quota, a setting, a bug) counts for none. When every unit
   sent fails and its own errors are of one cause (alike in type, code and
   HTTP status, not message; or all unparseable or cut-off output; a unit
   that failed otherwise, throttled say, is left out), output errors on the
   units of more than one document count as each document's own, once a
   run, with no known-good request: a server-side output cap cuts them all
   and a short request cannot tell, but a batch that always fails must not
   stop every run with its deletions (the trade-off: a cap that persists
   `indexing.max_document_failures` runs excludes them, until their content
   or the extraction settings change). Otherwise the stage sends one
   known-good request through the same chain, bypassing the LLM response
   cache: failed too (for output errors, unparseable counts as failed), the
   cause is the model or a setting and none counts; answered, the errors are
   the documents' own. Claim extraction's errors of a unit's own count for
   none: claims are optional, so a document whose other stages succeeded is
   indexed without those claims and recorded `FAILED` once, for the next run
   to extract them again from the stored records, then `PROCESSED` without
   them (`incremental.run_failures`). A failure that counts for none
   adds one to the record's `retry_count` instead (new content, a success
   or an error of its own resets it; a stop or death during extraction
   leaves it, one during indexing resets it once the `PENDING` records are
   written), and from the second the run warns (`documents_retrying`).
   What counts as unparseable or cut-off output: graph extraction output
   naming no entity with a section left unclosed (cut off without a stop
   reason) is cut off, not retried; with every section closed it is
   malformed and retried like unparseable output. Output cut later keeps
   the entities before the cut (WARNING `XML output ends with <entities>
   unclosed`). An answer the model corrects is parsed by its last complete
   answer, each section by its last closed occurrence (INFO `XML output
   repeats <entities>, ...`), and a JSON answer by its last object that
   parses. An OpenAI-compatible server's context-length rejection that
   counts the requested `max_tokens` is a setting. The response behind a
   failed graph or claim extraction check is logged at DEBUG. Extraction
   records only removed text units named are then deleted (best effort).

**Limitation:** shared communities and reports keep what a changed or deleted
document contributed, and a re-keyed entity's old id, until a full rebuild
(§5.5); descriptions, names, ids, types, filters, ranks and claims are
derived again on every run.

### 5.2 Full rebuild (`indexing.reset`)

`indexing.reset` skips the diff and rebuilds the whole corpus with the
full-index path, without taking the suffix offline. Quarantined documents
(§5.3) are skipped and keep their records; every other document, and every
indexed one the corpus dropped, is recorded `PENDING`; the vector indices are written into new timestamped indices whose
aliases swap when each is built; the graph is rebuilt in place
(`GraphIndexer.begin_rebuild` tags every vertex and edge the build writes, and
`end_rebuild` drops the suffix's entities, communities and edges without the
tag). The writes upsert in place, so graph queries see previous and rebuilt
items mixed until that final drop. Only after every
document is recorded again are the namespace's other records deleted, so a
failed rebuild drops nothing. A rebuild whose graph writes partly failed
(within `indexing.max_failure_rate`) skips the drop, so a retry still finds the
previous copies, and is reported failed (`graph_sweep`) rather than recorded:
clearing the replaced documents' records would orphan their vertices and edges.

The corpus a reset rebuilds includes the run's configured seed graphs;
when another scope of the namespace holds seed graph shards, the run asks
before its first write (§4.5).

A generation switch (new labels plus a pointer) was rejected: Neptune labels
cannot be renamed, so every retriever would resolve the current generation on
each query and the graph would be stored twice during the rebuild.

**Unchanged text is not embedded again**, by a rebuild or a delta run. Each
vector field stores the digest of the text it embeds next to the vector
(`embedding_digests.<field>`), with the embedding provider, model, dimension
and (outside Bedrock) input cap hashed in. A write reads the stored digests
of its items, one search per write slice of 800 ids, from the alias's live
index: the one an upsert writes, or the one a rebuild is about to replace
(still the alias's target until the swap). It then reads only the vectors
whose digest is that of a text it writes, only those fields, 100 ids per
search and four searches in flight, and embeds the texts whose digest it
does not find. Reading every stored vector of a slice whose texts changed
held over 60 MB of response and parsed floats per search for vectors no
write took: rebuilding 2,000 reports at 1,024 dimensions with a third of the
texts changed peaked at 388 MiB, and peaks at 165 MiB this way. Reusing across a reset is safe because
the digest names the embedding settings: a reset after a model or dimension
change finds no matching digest (nor reads at all: the live index's `_meta`
records its model) and embeds everything. A vector an upsert replaced is
read as well (up to 2,048) and kept for a later write of its text (a
removal rewrites items the commit writes back); the commit, the run's last
write, and a rebuild read none (an upsert of 2,000 changed descriptions
reads 2,000 vector fields instead of 4,000). An index without the digest
field gets it with the other missing ones (§5.1) and its items are embedded
once more.
`--force-rebuild` embeds every text again.

### 5.3 Crash recovery (write-ahead protocol)

The registry is not updated atomically with the stores, so a run can stop
between any two writes. One invariant holds: **before the stores hold
anything the registry cannot account for, the registry already lists it.**
Write-ahead records precede every removal and upsert, deleted documents' rows
go only after their artifacts, and a `PENDING` record is replaced only after
the document's writes landed. The next run repairs an interrupted one:

- `PENDING` never equals a content hash, so a document still in the corpus
  reads changed, even if its file went back to the indexed version; one no
  longer in the corpus reads deleted, and its text units cover everything the
  interrupted run may have written.
- The removal plans from the record stored before the current run's own
  `PENDING` record, so a recovery run removes the partial writes first.
- An interruption is not counted toward `indexing.max_document_failures`
  (a `PENDING` record has `failure_count` 0).
- Old ids of re-keyed items are deleted only after their new copies are
  written, so an interruption leaves at worst both copies, which the retry
  derives together.

**Attempts and quarantine** (`domain/ingestion/attempts.py`). A document
whose processing kills the task would stop every later run the same way.

- **Open and close.** After the diff (and when a run resumes past
  document_loading) the run opens an attempt on every document it processes
  (`incremental.open_attempts`: `attempt_batch` = documents processed
  together, `attempted_hash`; a new document gets a `PENDING` record without
  text units) and closes them however the process ends: finished, failed,
  SIGTERM or Ctrl-C (`close_attempts`). An attempt still open on the next run
  is one more death of that content (`attempt_count`) in a batch of
  `attempt_batch` (`crash_batch`). A reset opens none (the write-ahead record
  keeps the stored attempt fields); a document that produces no text units in
  a committing run counts as a death alone.
- **Plan** (`plan_attempts`, at the diff). A changed document whose deaths
  reach `indexing.max_document_attempts` is a suspect; one whose last death
  was alone (`crash_batch` 1) is `quarantined`. Otherwise the run processes
  the first half (by doc id) of the suspects that died in the smallest batch
  and classifies every other new or changed document `deferred` (neither
  processed nor deleted). A death narrows the suspects to that half and a run
  that ends commits it, so one culprit among `n` is isolated in about
  `log2(n) + 1` runs and no document is quarantined for dying with it.
- **Quarantined record**: `QUARANTINED` with the attempted content hash, its
  deaths and the reason in `error_info`. When the attempts reached indexing
  (the record is the write-ahead `PENDING` one), the removal pass removes what
  they wrote and no text units are kept; otherwise the indexed version and its
  text units stay. It reads unchanged (`diff` lists it in
  `DocumentDelta.quarantined_unchanged`) until the content changes or the
  limit is raised above its deaths (`release_quarantined` then classifies it
  changed); deaths count per content, so new extraction settings quarantine
  it again, and a reset skips it too.
- **Resume check** (`delta_still_applies`). A resumed run whose restored
  delta names a document whose record is gone, holds other content, reached
  the limit or was written by another pipeline id since (`written_by`; a
  record without the field counts as another pipeline
  id's unless it is `PROCESSED` at the delta's content without an open
  attempt, and even then when
  it was committed under other extraction settings for a document the delta
  re-extracts for a settings change), or deletes a document another
  pipeline id (or a record without a writer) has recorded in another scope
  since, resumes from document_loading instead. So does one for whose
  scope another pipeline id has since added, changed or deleted a document
  outside the delta, and one whose `document_loading` inputs changed since
  the delta was computed (a reset, retry or corpus setting; WARNING
  `Rewinding resume point from ... to 'document_loading'`). A finished
  pipeline id run again counts as resumed; a run resumed after its own kill
  or stop keeps its delta. A resume past
  a stage that recorded failed text units starts at that stage, so the units
  are retried rather than recorded `FAILED` again; a reset resumed past the
  stages that store extraction records starts at the first whose records
  another run deleted. Metrics:
  `total_documents_quarantined` (CDK `DocumentsQuarantined` alarm) and
  `total_documents_deferred`.
- **Unread files and mass deletions.** The diff never deletes a file the
  run could not read: `failed_doc_ids` holds the files that failed to
  parse or load and those the parsing stage skipped because no loader
  reads their type (`PipelineContext.skipped_source_files`; otherwise an
  optional extra dropped from an image would read as the deletion of every
  such file). Skipped files are kept apart from failed ones only for the empty
  corpus rule: a seed-only scope with a stray unsupported file is still a
  seed-only run. Before its first write (`open_attempts`),
  `incremental._confirm_mass_deletion` asks the pipeline's
  `confirm_deletions` when the run's own deletions exceed
  `indexing.max_deletion_share` of `DocumentDelta.scope_documents` (the
  scope's records the diff read, counted by `detect_delta` before any
  document moves between the lists) in a scope of at least 10 documents.
  The retired scopes' records (`DocumentDelta.retired`) are left out: the
  run was told to delete them. So are seed graph shards, from both counts
  (`DocumentDelta.seed_deleted`, by the records' `is_seed_path`): the
  configuration removes them (a seed graph dropped or resharded), and a
  seed graph of many shards would dilute the share until a partial sync of
  most documents passed unasked. Its own input (`--confirm-deletions` /
  `UKR_CONFIRM_DELETIONS`) for the same reason as the reset's. A run with
  the registry off is not checked: the stores hold no per-document count
  to compare the corpus with (a port method on every backend for one
  check), and its empty-corpus case, the likeliest mistake, is already
  asked (`_confirm_seeds_alone` with seeds; without them an empty corpus
  fails the parsing stage). Metrics: `total_files_failed`,
  `total_files_skipped`, `total_documents_excluded`,
  `total_documents_deleted`.

The cost is one extra batched registry write per run (two more writes and
reads for the attempts) and a few batched store lookups per suffix.
`tests/unit/test_incremental_write_ahead.py` and
`tests/property/test_incremental_interruption_properties.py` interrupt a run at
every write and check that the next run converges to a fresh full build.

### 5.4 Identity, derived content and merge rules

- **Document identity** (`domain/ingestion/document_identity.py`). A document
  version's id hashes its path relative to the corpus root and its full text,
  so without the registry a corpus yields the same ids wherever it is checked
  out; an incremental run keys text units by the registry `doc_id`, which
  also holds the source scope (`document_parsing.source_scope`, else the
  resolved source directory), so each belongs to one record and the ids hold
  only from the same directory or with a fixed scope. Changing either rule
  requires a re-index.
- **Derived content** (`domain/ingestion/derivation.py`). The merge and the
  strip keep each touched item's structure (id, text units, per-unit
  weights); its content is then derived from the records of the text units it
  cites now, by the full build's own code (`merge_contributions`,
  `GraphResolver`, `ClaimResolver`), so the item gets what a full build of the
  current corpus gives it: a deleted document's text leaves shared
  descriptions and claims, names and types follow the current votes, and an
  entity whose derived name has another id takes that id
  (`Derived.id_remap`), its edges and claims following. An item citing a text
  unit without a stored record keeps its stored content (a WARNING and the
  indexing stage's `underived_items` metric).
- **Merges a full build undoes.** A stored entity may hold what a full build
  keeps apart (two spellings merged while their types agreed): every resolved
  entity sharing a name key and a text unit with it is derived, the best
  match under its id and each other as its own entity (`Derived.split_from`).
  The stored edges of a split or re-keyed entity are derived with it
  (`Derived.rewired`), so each follows the part that states it and a
  re-pointed edge merges with the derived edge it now equals. An edge a full
  build draws that no stored edge is (an earlier merge dropped it as a
  self-loop) is added; a stored edge whose endpoints now resolve to one
  entity is deleted with the retired ids (`Derived.dropped`), as is a delta
  id the cross-run merge folded into another (an interrupted run may have
  left its vector copy). Edges take the resolution's per-unit strengths, and
  claim statements merge in text-unit id order, as in a full build.
- **Cost.** Derivation runs only where something is stored (a first build
  writes the delta as graph resolution made it) and matches through a key
  index. A stored summary that only grew (the item still cites every text unit
  it did) is summarized from itself and the new description, as LightRAG
  merges into the stored one, and a stored description that is no summary is
  extended with the delta's summary when graph resolution made one (which
  would otherwise go unused). An item that lost a text unit is summarized from
  its derived description, so removed text goes, and so is an item this
  run's removal summarized; the removal leaves the summary of an item the
  delta derives again under its id to the commit (`expect_delta`), so an
  edited document's shared entity is summarized once, from its lines, as in a
  full build. A stored edge rewritten for its rank only is updated in place,
  not embedded again.
- **Merge rules** (`domain/ingestion/merge/merger.py`, ported from MS GraphRAG
  `update/*`), all idempotent:

| Artifact | Matched by | Merged as |
|---|---|---|
| Entity | id, else identity key | description lines unioned; `text_unit_ids` and `community_ids` unioned; max confidence and rank; attributes unioned (delta wins) except `filters`, whose values are unioned per key; the stored name and type are kept (the derivation then sets what a full build would) |
| Relationship | id, else (source, target, type) | as entities; weight = sum of the strengths per supporting text unit (`attributes.text_unit_weights`), so the result equals a full build over the union; endpoint names re-derived from the merged entities |
| Claim | id | full-build `ClaimResolver` rules: descriptions and source texts unioned, `text_unit_ids` unioned, the stored status kept |
| Community | content-derived id | appended (§5.5) |

### 5.5 Communities on delta runs

A delta run clusters only the delta subgraph, as MS GraphRAG's update path
appends rather than re-clusters. Community ids are therefore content-derived
(a hash of the level and the sorted member ids), so a delta's communities and
reports are appended next to the corpus's instead of overwriting them. The
display name (`Level {level} Community {id[:8]}`, short id `L{level}_C{id[:8]}`)
derives from that id too: a positional `C{i}` would rename every community
sorted after an added one, and the name is part of a report's record key and
of its parent's roll-up text, so a reset would re-ask the model for
unchanged communities. A corpus community goes once none of
its text units is stored; shared ones keep their report until a full rebuild,
the only path that re-partitions the whole graph
(`indexing.community_drift_threshold` reports when one is due). Each delta's
root communities join every cut of the hierarchy, so global search's cut
grows until a rebuild; the cap and the shallower-cut fallback (§6.1.1) bound
it, and the drift metrics report it (`added_roots`). Rejected alternatives:
leaving delta communities of already clustered entities out of the cuts
(their reports are the only ones describing the delta, and stored community
ids still name removed communities), and re-detecting the full graph on every
delta run (it would regenerate every report, an LLM call each).
Across runs, the suffix's build record carries community counters
(`IndexBuildRecord.communities`): a full build starts them with the
communities it wrote, and `IndexBuildRecord.over` adds each delta commit's
appended communities minus those its removals deleted, keyed by a commit
identity (pipeline id and delta, `community_drift.commit_identity`) so a
resumed commit is counted once. `indexing.community_compaction` recommends a
rebuild from them (`community_drift.reset_reasons`); it never resets on its
own.
A seed graph added by a delta run is clustered on its own shards' entities
the same way, until a rebuild; a delta community only seed units evidence
gets no report (§4.5).

### 5.6 Concurrency

With the registry, a run takes its namespace's write lease before its first
stage and holds it to its end (`application/ingestion/namespace_lease.py`):
one conditional registry write, renewed by a heartbeat every sixth of
`indexing.lease_seconds` (lease requests: short timeouts, 3 attempts). A
second run fails with `NamespaceLeaseError` before it reads or writes.

- **Write fence** (`shared/write_fence.py`). Every incremental registry
  write, indexing phase, rebuild sweep, Neptune write batch or drop and
  OpenSearch indexing request other than a GET or HEAD checks the lease
  first. The check fails once the lease was released, and once less than a
  third of the term is left since the last successful renewal was sent (the
  store's expiry runs from then), so a run that lost the lease stops while
  it still holds it, provided no single write request outlasts a third of
  the term. A second fence fails the worker threads' writes once a stop
  (SIGTERM or Ctrl-C) was requested; the thread running the pipeline still
  writes while it unwinds.
- **No conditional writes.** `BatchWriteItem` takes no condition and store
  writes cannot be conditioned on the registry, so the margin is the fence.
  The DynamoDB registry checks it before every attempt of a write request
  (a botocore before-send hook), not once per call, so its re-sends and
  retries stop with the lease too; an embedded (SQLite) write transaction
  checks it before it waits for the database's write lock and again once it
  holds it.
- **Takeover wait.** A write past the fence lands by the holder's expiry plus
  the longest request (its attempts are bounded by the request timeout), so
  a run that takes over an expired, unreleased lease holds its first write
  (not its reads) until then; a frozen process that sends a built request
  later than that is the one case the margin cannot cover.
- **Giving the lease up.** A run that ended or failed released it (its pools
  waited for their writes), and the next run writes at once. A stopped run
  (SIGTERM or Ctrl-C) leaves it expired, owner kept, since its pools do not
  wait for the requests in flight: the next run waits as after a crash, and
  the stop message says until when. With every store embedded (SQLite; told
  by the adapters the run builds, so an injected remote store counts as
  remote) the lease is released: a worker the stop left running fails at its next write
  transaction, and one already holding the database's lock commits before
  the next run can write that store; a takeover waits at most one store
  write (`search.store_timeout_seconds`). A
  run that ends during its own takeover wait hands the wait on: the lease
  goes back expired at the previous holder's expiry.
- **Checkpoint.** The lease is taken before the S3 checkpoint is downloaded
  and given up only after a stopped run uploaded it; the upload checks the
  lease is held (`check_fences_held`), not the takeover wait, so a run
  stopped during the wait still keeps its checkpoint.

Without the registry nothing locks, and concurrent runs on one suffix race on
its aliases and labels ([runbook](./operations.md#concurrent-runs)).

**Lookups read once per run** (`shared/store_lookups.py`). Only a run's own
writes move its namespace's aliases and mappings while it holds the lease,
so the pipeline installs a run-scoped cache of which index each alias points
at and of each index's mappings, shared by the run's vector indexers (and
their worker threads). An alias swap or removal, an index deletion or a
mapping or `_meta` update through any of them drops the entries it changes
before its request is sent. A run that indexes nothing (an empty delta)
writes no build record and no extraction settings, so it writes nothing but
its lease.

---

## 6. Retrieval: Two Methodologies

`GraphRAGChain` (`application/retrieval/rag_chain.py`, an LCEL Runnable) runs
one chain for every strategy: conversation memory (a follow-up is rewritten as
a standalone question) → strategy resolution → query processing (translation,
entity or keyword extraction) → retrieval → fusion and rerank → (RAG) context
budget and answer. `RAGInput.search_strategy` picks the methodology.

![Retrieval chain](../assets/retrieval-pipeline.drawio.png)

![Search strategies](../assets/retrieval-strategies.drawio.png)

- **Where the chain stops.** `search` / `asearch` return the fused, ranked
  results (a `SearchOutput`); `invoke` / `ainvoke` / `stream` run the same
  steps, pack the results into the token budget (§7) and return a `RAGOutput`
  whose sources are exactly the sections the answer model saw. An empty
  budgeted context is answered "cannot answer" without calling the LLM.
- **Graph expansion** is part of `local` and `drift`, and opt-in for `mix` /
  `hybrid` (`search.lightrag_search.enable_graph_expansion`). Strategies find
  the seed entities in the vector store and pass their ids to the graph
  retriever, which expands from those ids only.
- **Under a document filter, expansion runs on the search store.** A graph
  holds no document attribute, so the walk goes over the relationships
  index instead (`BaseSearchStrategy._expand_over_passing_relationships`):
  each hop asks for the passing edges incident to the frontier vertices
  whose seed is short of its share, two requests (source and target side;
  plus one text-unit check under two or more keys), cached per query, and
  one entities-index read at the end (and, under two or more keys, a
  second text-unit check of the entities it reached). It keeps the graph walk's bounds
  (`max_hops`, `max_results_per_hop`, the budget split per seed, each seed
  at most its share) and scoring, but `degree` importance reads as `rank`
  (the index holds no degree). A side returns at most `min(max_query_size,
  live max_result_window)` edges, best first (a frontier past
  `max_terms_count` is asked in batches, each with its own page); when a
  side's page is full, the vertices still open are asked together once
  more at that side, and when that page is full too, each on its own, in
  waves of `search.max_concurrency`: each asks for what its wave may still
  take, and a wave leaves out what the waves before it reached. A hop costs
  one request per side, one more per full side, then one per open vertex
  and full side; under two or more keys each page adds a text-unit check.
  Under a caller `id` list the walk stops once it has reached every listed
  id (the results keep only those, and a reached vertex's path is fixed).
  Edges back to a vertex already reached are left out by the store (a
  `must_not` on the other endpoint), so they cannot fill a page; past
  `max_terms_count` the exclusion keeps the vertices reached last, so edges
  back to the others can. Without the relationships index the graph walk
  runs, checked against the entities index. So is a graph walk
  under an entity key entity vertices do not store (`name.keyword`,
  `attributes.<path>`): the graph drops it, the entities index applies it
  (`_graph_unapplied_filters`), to LightRAG's seeds too (its relationship
  endpoints were found under no entity key; `local`'s and `drift`'s seeds
  were found under the filters).
- **Seed graph facts are entities and relationships, never text.** A
  seed graph's text units are stored (removal and derivation read
  them) but never retrieved as chunks: the
  OpenSearch retriever (OpenSearch and the embedded engine) adds one
  `must_not` on their `source_kind` to every text-units search, a fetch
  by id included, inside the kNN filter (efficient filtering on lucene
  and faiss, so seed units take none of the k). OpenSearch 2.13 refuses a
  kNN filter on an unmapped field, so unless the alias's mappings show
  `source_kind`, a kNN search first probes once per query for a unit
  holding it; an index that holds none gets no clause. An ids-only
  filter check keeps them: it returns no text. Local search ranks and
  fetches document units only and its `entity_frequency_threshold` counts
  document units; the LightRAG chunk follow drops seed units. The
  `source_kinds` and `iri` filter keys are probed like the schema
  annotations (`LATE_MAPPED_FIELDS`); `source_kinds: document` means
  document evidence, items without the field included (OpenSearch
  `should[term, must_not exists]`, Gremlin `or(has, hasNot)`, the SQLite
  vertex filter alike), and is dropped on an index none of whose
  documents holds the field, where it filters nothing. Community reports
  carry no source kind (a report summarizes its members, seed-only ones
  included; deriving one from the members would go stale whenever a seed
  edit changes a member's kinds without regenerating the report), so a
  `source_kinds` filter keeping one kind skips them in `local`, the DRIFT
  primer and `simple`'s sweep
  (`BaseSearchStrategy._source_kinds_skip_reports`), and `global`
  refuses the key (no index it reads declares it). A reported source
  with seed evidence gets `provenance` (`rag_chain.seed_provenance`:
  kind, the seed graphs its document filters name, the entity IRI);
  graph expansion projects a vertex's `iri` and `source_kinds` (kept a
  list, `graph_expansion._LIST_PROPERTIES`, where a value map's
  one-element list is otherwise read as its value), but vertices hold no
  document filters: `local` copies a candidate entity hit's
  `attributes.filters` onto the graph's copy of it
  (`LocalSearchStrategy._with_hit_filters`), and a source a graph walk
  alone reached names no seed graph.
- **One index, both methodologies (deliberate extension).** Ingestion builds
  the union of what both need (MS GraphRAG builds no relationship vector
  index, LightRAG no communities). A LightRAG-only deployment can set
  `graph.community_detection.enabled: false` to skip Leiden and every report
  call, leaving `global` and `drift` unavailable.

### 6.1 GraphRAG Methodology (`adapters/search_strategies/`)

- **simple** — vector and lexical search over every built index, no graph; the
  claims index joins the sweep only when claim extraction is on.
- **local** — candidate entities → graph expansion → frequency filter → their
  text units, plus a community-report section and a relationship section
  (MS GraphRAG local search). Each query entity is looked up in the entities
  index on its own, concurrently: the entities of its name
  (`compact_name_key`) seed the expansion, else its top hit when that scores
  at least `search.local_search.seed_score_ratio` (0.6) of the best hit of
  any query entity (`seed_floor`: on a hybrid search the ratio is read at
  equal weights, where a hit only the kNN side found scores at most 0.5 of
  one both found, and rescaled piecewise linearly so 0.5 lands on
  `search.hybrid.vector_weight`, that hit's cap at any weights; a vector or
  lexical search compares raw scores, where the floor separates less); the seeds share the expansion budget in proportion to
  their scores (`seed_shares`; Neptune walks the seeds of each share in a
  request of its own). One lookup of the joined names seeded a wrong
  neighbour beside every right seed on a synthetic multi-hop corpus. With
  `search.local_search.include_bridge_relationships` (default on), the
  relationships incident to the expanded entities are fetched too, edges
  between two retrieved entities first: they carry the hops of a multi-hop
  chain the relationship vector query rarely matches. With claims on,
  matching claims are injected (MS GraphRAG covariates). The section lookups
  run concurrently with the entity chain. The community reports are, by
  default (`search.local_search.report_selection: similarity`), those most
  similar to the joined query entities, at every level, which always fill
  the report slots (`top_k / 2`). `membership` follows MS GraphRAG local's
  (`mixed_context._build_community_context`) with one difference: it takes
  the reports of the level-0 (leaf) communities the expanded entities name,
  where upstream takes each entity's community at the `community_level` cut
  (default 2, the entity's deepest level not past it; upstream numbers
  levels from the root), so the measurement below compares leaf reports,
  not upstream's cut, with similarity. The communities come from
  `community_ids`, which graph
  expansion projects from the vertex (the entities index holds none), most
  matched entities first, then by report rank, cut to the report slots and
  fetched alongside the text units. A community without a report (a seed
  graph's alone has none) adds nothing; when no expanded entity names a
  community (the walk under a document filter, which reads the entities
  index, or a failed expansion), or a `community_id` or `level` filter
  admits none of those communities, the similar reports stand in (under the
  filter). The search metadata's `report_selection` says which ran; DRIFT
  counts its follow-ups by each. On the
  synthetic 12-query multi-hop harness, membership put 3.8 reports a query
  in the context instead of 5.0, 42% of them naming an entity of the
  query's chain instead of 38% (19 such reports instead of 23), for 5%
  fewer report tokens (about 1,450 against 1,530); every query kept a
  relevant report and its gold text units. A real-model run then measured
  the answers: on 200 MuSiQue questions, two repeats each, membership
  scored exact match 4.0 points lower (95% CI -7.75 to -0.25) and token F1
  4.1 points lower (-7.7 to -0.6) than similarity, for about 0.5% fewer
  tokens. So similarity stays the default.
- **global** — MS GraphRAG global search over one cut of the community
  hierarchy (below), map-reduced into scored key points.
- **drift** — MS GraphRAG DRIFT search (`drift_search.py`).
  *Primer:* a hypothetical answer shaped like the most important report
  (`DriftHydePrompt`) selects the `k_followups` most similar reports, split
  into `primer_folds` folds, each returning an intermediate answer, a score
  and follow-up queries. *Actions:* for `n_depth` rounds the best-scored open
  follow-ups each run the local search (without rerank) and are answered with
  a score and new follow-ups. *Reduce:* the intermediate answers (one
  synthesized section, never a reported source) and the evidence (each
  primer fold's reports, each action's local search results), reranked
  once, go to the answer model. Without reports or primer follow-ups,
  the question itself is the one follow-up.
  - Defaults `k_followups` 10, `n_depth` 2: at most 27 LLM calls a query (MS
    GraphRAG's 20 and 3: 67). Primer calls and actions run
    `search.max_concurrency` at a time.
  - The action local searches pack their context by the local type shares
    (`search.token_manager.type_budgets`, renormalized over the types
    present: about 0.60 text, 0.24 entities, 0.16 relationships), not MS GraphRAG's DRIFT
    `local_search_text_unit_prop` 0.9 / `local_search_community_prop` 0.1.
  - The primer searches reports of every depth, not MS GraphRAG's one
    `community_level`, so similarity picks the granularity (on a synthetic
    six-hop corpus a depth-2 cut answered 16 of 21 sub-questions, every depth
    21).
  - A fatal error fails the query and cancels the calls in flight; any other
    failure leaves only its own answer out (`degraded_error`). The query fails
    only when a whole step failed and `processing.ignore_errors` is off; a
    round whose actions all failed ends the rounds.
- **auto** — an LLM router (`StrategySelectionPrompt`) picks among
  `search.auto_routable_strategies` minus those the pipeline cannot serve
  (`global` and `drift` need community reports) and, with filters, those
  that do not apply every filter key (none left: `InvalidFilterError`,
  before any LLM call, so a filter never fails by route); the prompt
  describes exactly these candidates, and unrecognized output falls back to
  `local` (else the first candidate). The default set is `local`, `mix`,
  `global` (`drift` costs up to 27 LLM calls); with one candidate no router
  call is made. The query processing the candidates may read (entities for
  `local`, keywords for `mix`/`hybrid`, the translation) starts alongside
  the router (`search.auto_speculative_extraction`), so `auto`'s critical
  path is the router or the extraction, whichever is slower, then the
  strategy: two model round trips before retrieval became one. A routed
  strategy never waits for an extraction it does not read, nor does the
  query: it keeps running in the background (`_BackgroundExtractions`),
  uncancelled, so its result is cached, and its call counts in the query's
  usage from its start (`estimated_calls` until it answers; cancelling it
  saved nothing, since a cancelled task does not stop a request already
  sent). It is cancelled at the query's deadline or when the chain closes.
  We chose this over one router
  prompt that also returns entities and keywords because the router prompt
  and its decision stay byte-identical (no re-validation of the routing),
  `custom_prompts.strategy_selection_*` keeps working, and the router's
  output stays one word, at the price of one or two short fast-model
  calls a query. The extractions are cached per question, target language,
  model and prompt text (`search.query_extraction_cache_size`), so the
  repeats and arms of a comparison read the same ones. The
  prompt sends questions that chain from a named entity to `local` (synthetic
  multi-hop harness: `local` 0.925, `mix` 0.40 exact match).

#### 6.1.1 Global search (`global_search.py`)

**Which reports.** Each report carries its place in the hierarchy (`level`
bottom-up, `depth` from the root, parent and children, `max_cut_depth`). The
cut at depth `D` is every community at depth `D` plus the leaves that stop
above it (`depth <= D <= max_cut_depth`), MS GraphRAG's roll-up that covers
every entity once; MS GraphRAG counts `community_level` from the root, so the
depth selects the cut.

- **Fixed cut (default)** — every report of the cut is mapped, up to a cap:
  the cut is browsed by rank, one report more than the cap
  (`max_reports_per_level`, at most `max_query_size` - 1, and the live
  `max_result_window` - 1, so the probe fits one request; 99 by default) so a
  cut of exactly the cap is complete. A larger cut gives way to the deepest
  shallower cut that fits (`global_cut_depth`), covering the same entities in
  broader reports; MS GraphRAG has no cap and maps the whole cut. Only when even the root cut does not fit, or at once under a filter on a report key (`community_id`, `level`, `rating`: a shallower report passes it on its own, not through its members), does a hybrid search,
  restricted to the configured cut, pick that many most similar reports
  (`global_level_capped`): the only place similarity enters global search.
  The cap counts reports, which bounds the map calls.
- **Dynamic selection** (`use_dynamic_selection`) — MS GraphRAG's
  `DynamicCommunitySelection`: rate the root reports for relevance, follow
  the children of the relevant ones to the leaves, and rate the whole next
  depth when a depth yields none, down to `community_depth` (MS GraphRAG's
  `max_level`, which bounds only this fallback). The walk starts at the roots
  as upstream's does but follows relevant reports to the leaves, where
  upstream's stops at `community_level`. Ratings run
  `search.max_concurrency` at a time. A rating is the first number of the
  model's answer, clamped to the prompt's 1-10; an answer without one, like
  a failed call, keeps its report unrated (MS GraphRAG keeps a report it
  could not rate), recorded in `degraded_error` and
  `global_unrated_report_count`.

**Map-reduce.** At least `map_reduce_min_results` reports are packed, in the
order the selection gives them (rank for the fixed cut, rating for dynamic
selection, similarity for the similarity fallback; MS GraphRAG shuffles
them), into batches of `map_context_tokens` (MS
GraphRAG's `data_max_tokens`); each map
call (`search.max_concurrency` at a time, awaited on the event loop, so the
query deadline cancels them and starts no queued one) extracts key points scored 0-100,
each naming its `report_ids` (MS GraphRAG's `[Data: Reports (ids)]`; a
report's id in the prompt is its ordinal in the batch, mapped back). Points
at or below `map_relevance_threshold` are dropped, the rest ranked, packed
into `max_map_reduce_tokens` and handed to the answer model as one
synthesized section (no separate reduce call: it only added a rewrite that
could drop facts), followed by the reports they cite; a
point citing none of its batch's reports credits the whole batch.
`evidence_tokens` caps the cited report text (whole reports, best point
first) at 4000 tokens by default; null keeps the reports the answer's
context budget holds, the largest part of a global query's input, while MS
GraphRAG's reduce reads the key points only (`0`). Global answers with the
comprehensive prompt by default, modelled on MS GraphRAG's reduce prompt
(multiple paragraphs, markdown); every other strategy with the concise one
(`search.answer_prompt_style`, merged over that default). Both defaults are
measured choices (§9.1).

**Degradation.** A batch whose map call failed or returned unparseable output
is *unrated*; when no point passed but some reports were never rated, those
reports pass through, so global search still answers, and the search
metadata records it (`degraded_error`). When every map call failed (other than
on a prompt longer than the model's context) and `processing.ignore_errors` is
off, the query fails with the first error, as DRIFT does; a fatal error (no
model access) fails it even under `ignore_errors`. When every batch was
rated and every point rejected, it returns nothing
(`map_reduce_no_relevant_points`) and the chain says it cannot answer, as MS
GraphRAG does.

### 6.2 LightRAG Methodology (`lightrag_search.py`)

Dual-level keyword retrieval (`KeywordsExtractionPrompt` extracts high- and
low-level keywords) on the shared hybrid infrastructure:

- **naive** — chunks only, no graph, under the request's `search_type`
  (`hybrid`: BM25 + vector by default; `vector` approximates upstream's
  baseline and MS GraphRAG's `basic` search).
- **hybrid** — low-level keywords → entity index, high-level keywords →
  relationship index (LightRAG's `relationships_vdb`), one-hop expansion
  (entity hits → incident relationships, relationship hits → endpoint
  entities), and the source chunks the matched entities and relationships
  cite (LightRAG's `_find_related_text_unit_from_entities` /
  `_from_relations`).
- **mix** — hybrid plus a naive chunk retrieval.

The cited chunks are picked per side, entities first, as upstream does: the
relationship side drops only the chunks the entity side picked
(`entity_chunks`), and each side is cut to `chunk_stream_top_k`. The two
sides are separate fusion streams (`lightrag_linked_chunks`,
`lightrag_relation_chunks`) beside mix's naive chunks, so RRF interleaves
them as upstream's round robin does, and the text quota cuts the merged
chunks after the rerank (upstream's `chunk_top_k` cut). `kg_chunk_pick_method`
picks within a side: `weight` (the default) is upstream's `WEIGHT` method,
weighted polling with every matched item at least one chunk, without
upstream's per-item sort by how many matched items cite a chunk, and both
sides fetched by id in one request; `vector` is its `VECTOR` method (upstream's
default), one search per side of the cited chunks for the query, under the
request's `search_type`, keeping `related_chunk_number * hits / 2`. A lineage
is read in chunk-id order, the order a full build stores; an incremental
merge appends the delta's chunks in arrival order, so reading the stored
order made the weighted pick depend on how the graph was built. `weight`
stays the default because the two measured no significant difference
(§9.1).

The independent queries run concurrently, then the expansions, then the cited
chunks. Short queries without keywords use the raw query as low-level
keywords (`raw_query_fallback_max_len`). Upstream LightRAG has no multi-hop
traversal, hence graph expansion is opt-in.

### 6.3 Iterative bridge follow-up (`iterative_search.py`, opt-in)

`iterative` is opt-in and unmeasured: no default selects it, `auto` never
routes to it (`search.auto_routable_strategies` refuses it), and whether it
helps is for a measured experiment to decide, not assumed. It targets one
failure: on 200 MuSiQue questions (v15 benchmark index) about 53 of `mix`'s
96 failures lacked a supporting passage, and the missing one was a later
hop's far more often than the first's (the last hop's 14-16% against the
first's 0-5%): round 0 finds the passage that names the bridge entity, not
the passage about it.

**Data flow.** The strategy wraps a base strategy by composition
(`search.iterative_search.base`: `local` or `mix`, default `mix`), built
through the strategy registry, so there is no dispatch on the base:

1. *Round 0* is the base's own search, unchanged (its fusion, rerank and
   quotas). A round 0 the controller calls sufficient returns the base's
   results as they are, so `iterative` then answers as the base does.
2. *Control.* A fast-model call (`models.roles.query_controller`,
   `IterativeControllerPrompt`) reads the question and the candidates packed
   as the answer's context is (the base's `spare_context_budget_first`),
   within `controller_context_tokens` (8000), plus what was already
   searched. It returns JSON (`parse_llm_json` with answer keys):
   `sufficient`, `known_facts`, `bridge_entities` (as the context writes
   them) and at most `max_next_queries` (2) `next_queries` naming them.
3. *Follow-up round.* The base search reruns seeded with the new bridge
   entities (local's `entity_focus`, so each is looked up by name and seeds
   the graph walk; mix's `ll_keywords`; the first follow-up query is the
   search text) without its own rerank, and each new follow-up query
   searches the text units (`round_top_k`, 5) under the request's
   `search_type`. A query already searched (case-folded) or a bridge already
   followed (`compact_name_key`, so `동부 창고` and `동부창고` are one) is
   not searched again, and the controller is shown the searched list: the
   redundant searches ReaRAG (arXiv 2503.21729) reports for open-ended
   search agents, which run on until their iteration limit, are what the
   caps and the dedupe prevent.
4. *Fusion.* Every round's candidates are fused once more by RRF, one
   bucket per round: `iterative_round_<n>` (round n's base search, round 0
   the first) and `iterative_round_<n>_text` (its follow-up queries' text
   units, interleaved), each weighable through `search.fusion.fusion_weights`
   (the base's own buckets were fused, with their weights, inside each
   round). The base's rerank types (`text` for both bases) are reranked
   against the question and the follow-up queries joined, and each section
   type keeps the larger of round 0's count of it and the type quota
   (`TYPE_QUOTA`), so a round-0 item leaves only when a new one outranks
   it. Back to 2 while rounds remain (`max_rounds`, 2); the last round is
   judged too, so the final verdict is known.
5. *Answer.* The chain packs the final candidates into the usual token
   budget and the answer model answers once (the chain's answer step,
   output guardrail, grounding check and sources, unchanged).

`decompose: true` replaces 2-4 with one KAG-style pass: one call
(`QueryDecompositionPrompt`) writes at most `max_rounds + 1` single-hop
steps in MuSiQue's notation (`#1` is step 1's answer); each step, with its
references replaced by the earlier answers (an unanswered one by its
question), runs one follow-up round (the step's answers are its bridge
entities) and is answered from those candidates by one call
(`DecompositionStepPrompt`, an empty answer when they state none). The steps'
candidates are fused with round 0's once, as in 4. The step answers are not
passed to the answer model: it reads retrieved evidence only.

**Layers.** The strategy is an adapter (`adapters/search_strategies/`),
registered with `@register_strategy(SearchStrategy.ITERATIVE,
wraps=...)`. `StrategySpec.wraps` names, per config, the strategy a wrapper
runs; `wrapped_strategy()` resolves it, and the chain reads the wrapped
strategy's query inputs (`query_processing.needs_query_entities` /
`needs_dual_keywords` with the config) and context packing
(`spare_context_budget_first`) through it, so `iterative` extracts and
packs exactly what its base does, and needs no index, store or community
report the base does not (same retriever roles, `needs_community_reports`
false). Whatever else is declared per strategy (a methodology a namespace
must have indexed) should be read through `wrapped_strategy()` too. Its
filter checks delegate to the base (`filtered_index_prefixes`,
`filtered_retrievers`, `reserved_filter_keys`). The prompts are in
`domain/prompts/retrieval.py` (overridable as `custom_prompts.
iterative_controller_*`, `query_decomposition_*`, `decomposition_step_*`),
the settings in `search.iterative_search`.

**Languages.** The chain passes the query's language in
`SearchQuery.metadata["query_language"]` (`query_processing.query_language`:
the requested language, else the index's, which is the source language when
translation is off and the target language when it is on); the controller,
decomposition and step prompts write their queries and bridge names in it,
names as the context writes them. Nothing in the strategy parses text but
JSON and `#n`; dedupe uses case folding and `compact_name_key`, which drops
spaces next to Han, Hangul and Kana. The text searches go through the
configured analyzers (the embedded engine's bigrams for dense scripts).

**Filters, scopes and provenance.** Every round's base search is the
caller's query with seeds added (`model_copy`, so its filters, suffix and
`search_type` are kept), and every text search uses `_scoped_filters`, so
a document filter or tenant scope applies in every round with the same
code-injected predicates as the base's. Fusion copies results, so a cited
follow-up passage keeps the provenance the base reports (`text_unit_ids`,
document attributes, `iri`, `source_kinds`, `attributes.filters`); no
synthesized item is added, so every source is retrieved evidence and the
retrieval evaluator scores it as any other.

**Deadline.** A follow-up round starts only while the query deadline
(`search.query_timeout_seconds`) leaves `search.model_timeout_seconds` for
the answer, and the rounds run in one `asyncio.timeout` that ends there:
past it the round in flight (its controller call, base search and text
searches) is cancelled and the answer reads what the rounds before it
found (`stop_reason: deadline`). Without a deadline the round cap alone
bounds it.

**Failure modes.**

| Event | Outcome |
|---|---|
| Controller or step call raises a fatal error (no model access) | The query fails, as in every strategy |
| Any other failed call (throttled past its retries) | The rounds stop; the answer reads the earlier rounds' candidates; `degraded` names the step and `degraded_error` the error type |
| Unreadable controller JSON (no `sufficient`), or a plan with no steps | As above, without `degraded_error` (`controller_unparsed`, `decomposition_unparsed`) |
| A store request of a round fails | Degrades that sub-search to nothing (`_safe_aretrieve`), as in the base; fatal store errors fail the query |
| Deadline reached | The round in flight is cancelled (`deadline`) |
| Insufficient after the round cap, or nothing new to search | `SearchResult.metadata["context_insufficient"] = True` |

**Contract: `context_insufficient`.** Set (to `True`, absent otherwise)
only on a positive verdict: the controller's last verdict was insufficient
and no further round could run (`max_rounds` reached or no new lead), or,
with `decompose`, the last step found no answer. A degraded or timed-out run
never sets it, its verdict being unknown. The strategy does not abstain; an
abstention policy reads the flag from `RAGOutput.metadata` (the chain
copies the search metadata there). `metadata["iterative"]` records the base,
mode, rounds, `stop_reason`, round 0's own rerank outcome (`rerank` is the
last fusion's), the searched queries, each verdict (sufficient,
bridge entities, queries, the number of known facts), the steps, the
results added over round 0, and the rounds' own model calls and tokens
(`usage`, by model; the query's `metadata["usage"]` counts them too).

**Cost.** Per query at most `max_rounds + 1` controller calls (3), or
`max_rounds + 2` decomposition and step calls (4), on the fast tier;
`max_rounds` reruns of the base search without rerank, up to
`max_rounds * max_next_queries` text searches, and one rerank call per
round. At the 2026-11 list prices (Haiku 5.5 $0.10 / $0.50 per million
tokens, Cohere Rerank 3.5 $2 per 1,000 search units), a controller call
reading 8,000 context tokens costs about $0.001 and a rerank $0.002, so
the worst case adds about $0.007 a query to `mix`'s (most queries stop
after round 0 or 1, KAG reports 84-98% after its first round); latency adds
one fast-model call per round plus a base search and a rerank, an estimated
3-12 s for two rounds (unmeasured). The answer is generated once.

**Fidelity: what the papers do and what we changed.** Neither IRCoT nor
KAG is ported as a library (there is nothing to diff against algorithm by
algorithm; their published settings are compared instead):

| | IRCoT (Trivedi et al., ACL 2023, arXiv 2212.10509) | KAG (Liang et al., arXiv 2409.13731) | `iterative` |
|---|---|---|---|
| Loop | Interleaves one chain-of-thought sentence with one retrieval: the first query is the question, each later one the last generated sentence (code: `query_source="question_or_last_generated_sentence"`) | Logical-form planner splits the question into sub-queries with operators (Retrieval, Sort, Math, Deduce, Output) linked by variables; each answered by graph retrieval, falling back to hybrid chunk and subgraph retrieval below a threshold; a Judge decides answerability, SupplyQuery writes a supplementary question (reflection, adopted from ReSP) | Controller loop: a sufficiency verdict, then follow-up queries and bridge entities; `decompose`: natural-language steps with `#n` references, no operators |
| Query per step | Last CoT sentence | Sub-query or supplementary question | Controller-written queries naming the bridge entities, plus the bridge entities as graph seeds |
| Retrieval per step | BM25 (Elasticsearch), K paragraphs, K tuned over {2, 4, 6, 8} (configs default 6) | Graph, then hybrid | The full base search (BM25 + kNN + graph walk for local; mix's dual-level streams) plus `round_top_k` (5) text units, fused by RRF and reranked |
| Collection cap | 15 paragraphs (`global_max_num_paras`) | — | The answer's token budget, each section type at round 0's share |
| Steps cap | 8 in the paper (footnote 4), 10 in the code default (`max_num_sentences`) | Reflection n = 3 rounds (`ref3`; benchmark configs `max_iteration: 3`) | `max_rounds` 2 follow-up rounds (3 retrieval rounds, as `ref3`); `decompose`: 3 steps |
| Stop | The sentence says "answer is", or the cap | Judge says answered, or the cap | Controller says sufficient, nothing new to search, the cap, or the deadline |
| Final answer | A separate reader always (reading the CoT's own answer scored lower: MuSiQue F1 30.8 vs 24.9, Flan-T5-XXL) | Generator over the sub-answers | The answer model once, from retrieved evidence only |
| Models | Flan-T5 (base to XXL), GPT-3 code-davinci-002 | DeepSeek-V2 (baselines mixed backbones) | Controller on the fast tier (Haiku 5.5), answer on the default tier |

Why the changes: an LLM call per sentence (IRCoT) or per logical form (KAG)
costs more per query than a bounded number of fast-model verdicts, and
Adaptive-RAG measured IRCoT-style multi-step retrieval at 2.1-4.7 times the
steps and 3.3-8.8 times the time of single-step retrieval; the graph and
dense signals we already have make the bridge entity a better next query
than the last CoT sentence, whose text is a statement rather than a
question; the logical-form operators target numeric and comparative
questions our benchmarks do not exercise. The reported effects that
motivate it are modest and setting-dependent: IRCoT's QA F1 gain on
MuSiQue was +5.0 (Flan-T5-XXL) and +7.1 (GPT-3), retrieval recall +3.5 and
+12.5; KAG's reflection (ref1 to ref3) gained +4.4 F1 on MuSiQue for its
hybrid variant (44.3 to 48.7) but +0.2 and +1.2 on HotpotQA and 2Wiki. KAG's
current code re-plans the original question on a retry rather than writing
a supplementary question as the paper describes; ours follows the paper's
supplementary-question form. Our expected gain is the own-data ceiling of
about +11 points of answer F1 if every missing last-hop passage were
recovered, realistically +3 to +5, below the 6-8 point minimum effect 200
questions detect: hence the measured experiment before any default.

---

## 7. Hybrid Scoring and Token Management

- **HybridScorer** (`adapters/retrieval/hybrid_scorer.py`) fuses the
  per-source results by RRF (`rrf_k`) or weights, reranks, applies MMR
  diversity (`diversity_lambda`) and cuts to the per-type quota or `top_k`.
  Reranking comes before diversity (as upstream LightRAG reranks every chunk
  candidate before truncating), so a cut on fused scores cannot drop what the
  reranker ranks first. Each query records its rerank outcome on its own
  `RerankTrace`, reported as `SearchResult.metadata["rerank"]`; a rerank model
  that cannot be built leaves the fused order (`unavailable`) or raises under
  `search.reranking.required`.
- **TokenManager** (`adapters/retrieval/token_manager.py`) packs the context
  into the answer model's budget. Each section type gets a hard-capped share
  (`search.token_manager.type_budgets`); within it sections go in score order
  and the first that overflows is truncated. Budget a type leaves unused goes
  to the types that still have candidates (first to the strategy's
  `spare_context_budget_first` types, as LightRAG gives chunks what its graph
  sections leave), so shares, never cross-type score comparisons, decide the
  mix. The shares are of `max_context_tokens`, not upstream's numbers: at
  30000, LightRAG mix/hybrid get shares of text 18072, entities 7228 and
  relationships 4698 tokens (upstream: about 16000 / 6000 / 8000); after the
  512-token prompt buffer, the question and the section headers about 17700
  / 7050 / 4600 are packed. Synthesized global
  sections are seated first. `RAGOutput.sources` is
  built from this selection, so it lists only what the answer model saw,
  truncated text included. Passages, claims and reports render with an
  `[id]` prefix; entity and relationship lines without one, since nothing
  reads those ids and a UUID cost about a quarter of their tokens. An entity
  line shows the entity's type, from either store.
- **Token counting** (`adapters/aws/token_counter.py`,
  `adapters/local/token_counter.py`). The Bedrock `count_tokens` API is used
  for language models that accept it, with a script-aware estimate otherwise
  (`adapters/llm/token_estimate.py`); CountTokens rejects the default Claude
  5.x models and every embedding and rerank model, so with the defaults every
  count is the estimate. A non-transient failure marks the model unsupported
  for the process. Local models use the estimate or tiktoken.

---

## 8. AWS Service Integration

| Service | Module | Purpose |
|---|---|---|
| **Bedrock** | `adapters/aws/bedrock.py`, `bedrock_models.py` (capability catalog) | LLM, embedding and rerank; inference-profile resolution, provider-aware request shaping (adaptive thinking and `effort` on Claude 4.6+, `reasoning.effort` on OpenAI GPT), prompt caching where supported; any model id works through curated rows, provider-family defaults and `aws.bedrock.model_overrides` |
| **Neptune** | `adapters/aws/neptune.py` | Gremlin over `wss://` with SigV4 (`ws://` for a local Gremlin Server); batch upsert and delete, `indexing.neptune.index_concurrency` concurrent batches; every indexer request is retried over a new connection and safe to repeat (vertices and edges are created or matched by id); an edge whose endpoint vertex is missing fails and keeps its stored copy; communities are vertices only (no `MemberOf` edges: an entity names its communities in `community_ids`) |
| **OpenSearch** | `adapters/aws/opensearch.py` | kNN (HNSW, `lucene` engine by default, which supports `cosinesimil` on OpenSearch 2.13) and BM25, sync and async SigV4 clients, alias management, bulk upsert and delete, per-language analyzers |
| **S3** | `adapters/aws/s3_cache.py` | Stage-cache sync (the bucket's default encryption unless `aws.s3.encryption` forces one) |
| **DynamoDB** | `adapters/doc_status/dynamodb.py` | The doc-status registry |

The AWS session is the bundle's (`Providers(boto_session=...)`, so a fake or
moto session can be injected), created from `aws.profile_name` when an AWS
client first asks for it.

### 8.1 Retrieval Error Visibility

Authentication, configuration and store failures are never disguised as "no
results". One policy, `is_fatal_retrieval_error()` (`adapters/store_errors.py`),
decides **by exception type alone**: fatal errors fail the query, the rest
degrade only the section that raised them. Every sub-retrieval goes through
`BaseSearchStrategy._safe_aretrieve`, so a section handler cannot swallow a
fatal error.

| Failure | Raised as | Outcome |
|---|---|---|
| Store unreachable, connection closed, or 429/502/503/504 after the client's retries | `StoreConnectionError` | fatal |
| No answer within `search.store_timeout_seconds` | `StoreTimeoutError` | fatal |
| HTTP 401/403, Neptune `AccessDeniedException` | `StoreAccessDeniedError` | fatal |
| HTTP 400 (a filter value of another type, a vector of another dimension) | `StoreRequestRejectedError` with the server's reason | fatal |
| Bedrock error that is not transient (bad model id, no access, no credentials) | botocore error | fatal |
| Query embedding failed, whatever the cause | `EmbeddingModelError` | fatal (every vector section needs it) |
| Local model server 4xx other than 408/429, or unreachable after retries | `openai` / `httpx` error | fatal |
| A prompt longer than the model's context | `ContextOverflowError` (or a plain 400 from vLLM / llama.cpp) | not fatal: that global-search map batch is left unrated, unless every batch fails |
| One failed request (an HTTP 500 for one search) | any other exception | degrades that section |
| The whole query past `search.query_timeout_seconds`, or a stream's first chunk past `search.model_timeout_seconds` | `QueryTimeoutError`, raised by `GraphRAGChain` once its tasks (and, through `gather_or_cancel`, every section's) are cancelled | fatal (the error answer under `processing.ignore_errors`) |

Neptune specifics: a lost connection is dropped so the next query or retry
opens a new one; a query traversal runs on retriever-owned threads, bounded by
twice `search.store_timeout_seconds`; closing the client fails requests still
waiting for a pooled connection or a late read; the indexer's read timeout is
a fixed 180 s, above Neptune's own 120 s query timeout.

### 8.2 Client Lifecycle / Resource Release

Retrievers open a Neptune WebSocket and thread pool and OpenSearch HTTP
pools, which leak until garbage collection unless closed, so every layer
exposes a best-effort `close()` / `aclose()` that never raises:

- `OpenSearchClient` and `NeptuneClient` close their pools (OpenSearch also
  when the event loop changes, awaiting the old client on its own loop).
  A closed Neptune retriever opens no connection or worker pool again: a
  traversal still running at the close, or started after it, fails with
  `StoreConnectionError`.
- `GraphRAGChain.close()` / `aclose()` close what the chain built, then
  stop the chain's loop (cancelling the sync calls still in flight), even
  when a retriever close or the wait for one fails.
  `IndexingManager.close()` closes both its indexers, injected ones too.
  Indexers passed to `DataIngestionPipeline(vector_indexer=...,
  graph_indexer=...)` are borrowed: the pipeline never closes them, their
  owner does. A `Providers` indexer factory returns a new instance per call,
  which its caller closes. Whoever builds a chain must close it; a finalizer
  is only a safety net. The CLIs close in `finally`.
- `GraphRAGChain` caches retrievers and strategies per event loop; its sync
  entry points run on one chain-owned loop thread, so repeated sync calls reuse
  one set of clients. Blocking calls run in `asyncio.to_thread`, on a default
  executor sized to `processing.io_workers` (which bounds concurrent Bedrock
  calls; `configure_event_loop` for async hosts).

### 8.3 Multilingual Processing

- **Analyzers.** `indexing.opensearch.language_analyzers` maps a language to an
  OpenSearch analyzer. Each supported language has a default: the built-in
  language analyzers (`german`, `english`, `spanish`, `french`, `italian`,
  `portuguese`, `russian`), `cjk` (character bigrams) for `ja` and `zh`, and
  `nori` for `ko`. An Amazon OpenSearch
  Service domain has `nori` only once its Nori package is associated, so the
  CDK deployment maps `ko` to the built-in `cjk` unless
  `opensearch_nori_package` associates the package (a custom resource) and
  maps `ko` to `nori`; an analyzer the cluster lacks fails index creation
  with an error naming it.
- **Identity keys** keep the letters of all scripts (§3).
- **Translation** is skipped when the source and target languages match and
  no other target is set (`TranslationConfig.is_noop`). A query is
  translated only when ingestion translated: not when the translation stage
  is disabled or a no-op, unless the caller asks for a language
  (`RAGInput.target_language`). Its entities and keywords are extracted in
  the language the index was built in: `source_language` when the stage is
  disabled, `target_language` otherwise.
- **Encoding.** A text, CSV or JSON file is first read with the locale's
  preferred encoding, as `open()` reads (UTF-8 in the container image; it
  can be another on a host whose locale is not UTF-8). Only when that read
  fails with a decode error is the file read again with the encoding
  `charset-normalizer` detects (latin-1 when it detects none). A file that
  decodes under the locale's encoding is kept as read, even when it was
  written in another encoding.

---

## 9. Evaluation Framework

`EvaluationManager` (`evaluation/`) builds the evaluators through
`_resolve_evaluator_class` (one lazy-import branch per `EvaluatorType`) and
scores every result after all answers are generated. What each evaluator
measures is in [Evaluation: Evaluators](./guides/evaluate.md#evaluators) (how it
scores: [Evaluation output](./reference/evaluation-output.md#evaluator-details)); the design
choices:

- **LLM-free evaluators are first-class.** `answer_match`, `retrieval` and
  `graph_aware` are deterministic and free, so they run by default next to the
  RAGAS judge. The LangChain judge is opt-in: its correctness metrics repeat
  RAGAS' `answer_correctness`.
- **`graph_aware` emits recall only.** Precision and F1 would need every
  entity of a free-text answer enumerated, so they would only re-label recall.
  Because answer-text coverage grows with length, it also reports the
  coverage of the retrieved sources and the answer length.
- **`retrieval` counts files, not sources.** A community report is attributed
  to every file of its community, so counting it as one source let random
  reports score near-perfect hit@k.
- **Strategy comparison** (`evaluation/comparison.py`, `pairwise.py`). Several
  strategies or `--repeats` run the same queries; each metric is averaged per
  query over repeats and compared per query with a seeded paired bootstrap
  interval (pooled means would mix query difficulty and noise into the
  difference). The optional pairwise judge (comprehensiveness, diversity,
  empowerment, directness, as in the GraphRAG and LightRAG papers) judges both
  answer orders, keeps only order-stable verdicts and reports how often the
  orders agreed (`position_consistency`), so a position-biased judge's ties
  are not read as equal answers. A failed answer is never dropped (a strategy
  failing on the queries it would lose would look as good as one answering
  them): it scores 0 on every metric another run scored the query on and its
  own strategy scores somewhere (a metric the strategy never produces is no
  loss to invent), and a pair with one failed answer goes to the other side
  unjudged.
- **Cost is measured, not estimated.** `shared/usage.py` keeps contextvar
  scopes (one per ingestion stage, per query, per `run-eval` run and answer
  attempt) that a
  callback on every factory-built chat model fills from the tokens each
  response reports, tracing on or off (one context-variable read outside a
  scope); cache-served responses count as hits, not calls. The comparison's
  `cost` block pairs each query's calls, tokens and latency like a metric,
  and `--fail-if` gates on either. A failed answer keeps what its attempts
  spent (its own scope and timer), so failing cannot make a strategy look
  cheaper or faster by dropping out of the cost; an answer without a measure
  fails a cost gate rather than leaving the delta to the others. Prices are user-supplied
  (`evaluation.prices`): a shipped table would be stale. Input tokens
  include the prompt-cache reads and writes, also counted apart
  (`cache_read_input_tokens`, `cache_write_input_tokens`) and priced at
  their own rates when a price gives them. A call counts once however many
  requests botocore sent for it; those resends (`request_retries`, read from
  `ResponseMetadata.RetryAttempts`) and calls that failed (`failed_calls`)
  are counted apart, since Bedrock counts each accepted request as an
  invocation.
- **The judge is not the answer model.** `models.roles.evaluation` defaults
  to Opus 5.5 on Bedrock, a model other than the default answer model, since
  a judge favours its own answers (§9.1); when the answer model is Opus 5.5
  the unset judge falls back to the default tier's model, else Sonnet 5.5
  (`Config.judge_model_choice`, logged at INFO); a re-judge
  (`--from-results`, `--baseline-results`) also avoids the saved runs'
  answer models. `judge_is_answer_model` in
  the run manifest and a warning still flag a configured judge that is the
  answer model.
- **A comparison is a pure function of saved files.** `paired_comparison`
  reads only the runs' results and reports, so `run-eval --from-results`
  (`evaluation/saved_results.py`) rebuilds it without a model call, an
  interrupted comparison finishes by running only the missing strategies,
  and runs scored with different `answer_scoring` are refused rather than
  compared. Against `--baseline-results`, each strategy pairs with the
  earlier run of the same strategy (`baseline_comparison`: one paired
  comparison per pair, merged): pairing every strategy with one baseline
  would compare strategies, not runs, and hide a regression of all but the
  first.
- **EM/F1 read a span, not the whole answer.** The reading-comprehension
  metrics were defined over a short predicted span; over a sentence plus its
  support they reward brevity. `evaluation.answer_span` picks the scored
  text: the first answer line (default, free), a fast-model extraction
  (cached, falling back to the first line) or the whole answer. The choice is
  recorded in the manifest and each report, and a comparison warns when its
  runs differ.
- **Run manifest.** Ingestion records an `IndexBuildRecord` (pipeline id,
  mode, build time, documents, extraction fingerprint and settings, package
  version, community counters) in the vector index mappings' `_meta` (`VectorIndexer.record_build`),
  read back per queried suffix. The config hash covers only the settings that
  shape answers (§12), so it is equal across environments.
- **Benchmarks.** `prepare-benchmark` (`evaluation/benchmarks/musique.py`)
  builds a MuSiQue corpus and dataset at run time;
  `evaluation.answer_normalization: official_squad` with
  `evaluation.answer_span: extracted` brings EM/F1 close to published
  numbers (a RAG answer still is not a trained reader's span).

### 9.1 Measured defaults

The defaults set or kept from real-model measurements, with the method, the
numbers and dated prices, are in [Defaults and evidence](./concepts/defaults-and-evidence.md).

### 9.2 Unanswerable questions and abstention (opt-in)

Whether the chain declines a question its corpus cannot answer is measured
with a benchmark split built for it and an explicit decline marker. Both are
opt-in: with no new flag and `search.abstention.mode: off` the corpus, the
dataset, the answer prompts and the scores are those of earlier releases.

**Benchmark** (`evaluation/benchmarks/musique.py`, driven by
`prepare-benchmark musique --unanswerable N [--hops ...]`). MuSiQue-Full
pairs every answerable question with an unanswerable twin of the same id:
the same question with one or more supporting paragraphs replaced by
distractors. Two leaks make a naive pool answerable:

1. *The twin.* An unanswerable item pooled with its own answerable twin
   gets the removed paragraph back from the twin's paragraphs. Any
   unanswerable item whose twin is in the answerable sample is excluded.
2. *Other questions.* MuSiQue reuses Wikipedia paragraphs across questions:
   in a sample of 200 unanswerable plus 200 non-twin answerable items,
   28-42% of the unanswerable ones still find every removed paragraph in
   the pool (measured offline on the dev split). Selection is greedy over a
   seeded shuffle of the candidates: an item is kept only when no removed
   paragraph's exact text (whitespace-normalized) is in the pool so far, and
   none of its own paragraphs is a removed paragraph of an item already kept;
   a kept item's paragraphs join the pool. Either rule excludes it as
   `contaminated`. The pool starts from the answerable sample, so the
   answerable set is the one the same seed draws without `--unanswerable`.

Unanswerable rows carry `metadata.answerable: false`, no answer or
reference sources, `metadata.benchmark: musique-full-unanswerable` and the
twin's answer (`metadata.twin_answer`, for auditing an answer that names it).
The counts (`unanswerable_filter`: candidates, kept, excluded twins,
contaminated, no twin) are printed and in the JSON output. `--hops` filters
both pools by hop count (`3,4`) or draws an exact answerable quota per hop
(`2=50,3=50,4=100`); sampling stays deterministic per seed. The exact-text
filter does not catch near-duplicate paragraphs or facts a model knows
without the corpus; `twin_answer` exposes the latter. Cost is ingestion's:
about 19 paragraphs a question.

**Decline marker** (`search.abstention`, applied by `GraphRAGChain`). The
chain already answered an empty context with a fixed reply
(`metadata.abstained`), but a model that declines in its own words was
invisible to evaluation. With `mode: mark` the answer step appends
`ABSTENTION_INSTRUCTIONS` (`domain/prompts/retrieval.py`) to the system
prompt of whichever answer prompt runs (concise, comprehensive or a
`custom_prompts` override): when the context does not support the answer,
start with the fixed token `[NO_ANSWER]`, then one sentence in the query's
language naming what is missing. Detection
(`domain/retrieval/abstention.py`) is a pure function over the first
characters after NFKC normalization, so full-width brackets and leading
Markdown emphasis are accepted and the answer's language does not matter
(the token stays ASCII). It sets `metadata.abstained` and
`metadata.abstention_reason` (`no_context`, `marker`, `insufficient_context`)
and strips the token unless `strip_marker: false`; a streamed answer holds
back at most the first 32 characters until the token is ruled in or out.
An abstained answer keeps its sources: the model saw them, and evaluators
audit them (only a guardrail block or an empty context leaves none). The
marker reaches the output guardrail stripped, like any answer.

**Insufficiency hook** (`search.abstention.on_insufficient`). A strategy
that judges its own context (an iterative or sufficiency-checking strategy)
sets `SearchResult.metadata["context_sufficient"] = False` (constant
`CONTEXT_SUFFICIENT_KEY`) and optionally `insufficiency_reason`. Only then
does the answer step apply the policy: `hedge` (default) appends
`HEDGE_INSTRUCTIONS` (answer what the context supports and say what it
does not), `abstain` appends `DECLINE_INSTRUCTIONS` (do not answer; give the
token and the missing part in the query's language) and marks the output
abstained with reason `insufficient_context` whatever the model writes.
Both are LLM-written so the wording follows the query language; neither
costs an extra call. Strategies that never set the flag are unaffected; no
shipped strategy sets it yet.

**Scoring** (`EvaluationManager._calculate_abstention_statistics`). An
abstention on an answerable item is a wrong answer: `answer_match` scores it
0 on every metric (no span extraction call) and records `abstained` in the
report. Unanswerable items are scored only by abstention:
`abstention_statistics` adds `answerable` (same shape as `unanswerable`),
`precision` (abstentions that were on unanswerable items), `recall`
(unanswerable items abstained on) and `f1`, and per `answer_match` metric
`coverage` (answerable items answered), `accuracy` (over all answerable
items, abstentions as 0), `selective_accuracy` (over the answered ones) and,
with unanswerable items, `overall_accuracy` (a correct answer or a correct
decline, over both). Coverage and accuracy are reported together so a
strategy cannot look precise by declining more.

Failure modes: a model that ignores the marker instruction looks like
`mode: off` (recall falls, nothing breaks); one that writes the token
before a real answer is scored as a decline (precision on answerable items
shows it). The marker instruction changes the answer prompt, so `mark`
runs are a separate arm, not comparable EM-for-EM with `off` runs; the
manifest's config hash differs. Whether `mark` should become the default
waits on a measured run (`unanswerable` 100-200 + matched answerable,
`local` and `mix`, `off` vs `mark`).

---

## 10. Visualization & Analytics

Ingestion renders nothing: the community-detection stage writes the graph
and its community hierarchy as JSON (`graph.visualization.export_data`,
`domain/ingestion/visualization_data.py`). `run-visualization` renders that
file without ingestion through the renderers of `adapters/renderers/`
(`BaseRenderer`, `@register_renderer`, `RenderContext`, `resolve_renderers`,
`render_resolved`):
`InteractiveRenderer` (pyvis network and community hierarchy) and
`StaticRenderer` (degree, centrality and community-size charts as HTML with
inline SVG, written without a plotting package and with every name
HTML-escaped), computing a seeded spring layout and the configured
centrality first. The interactive renderer is the `viz` extra: the adapters
import the renderers on use, so the registry, `BaseRenderer` and `static`
work on a core install, and a missing extra fails the run
(`BaseRenderer.check_dependencies`) before anything renders.
Using it: [Visualization](./guides/visualize.md).

---

## 11. Prompts and Prompt Tuning

- **Prompts** (`domain/prompts/`) are `BasePrompt` classes whose templates and
  variables are class attributes, version-controlled as code. All but five
  can be overridden from config (`custom_prompts`); the
  [Domain adaptation guide](./guides/domain-adaptation.md#prompts-without-an-override)
  lists the five.
- **Prompt tuning** (`application/prompts/tuner.py`, after MS GraphRAG
  `prompt_tune`) profiles a corpus sample (domain, language, persona, entity
  types) and writes a domain-adapted `custom_prompts` fragment to review and
  apply; nothing is applied at run time. A tuned prompt replaces only the
  persona preamble and keeps the built-in output rules (the XML schema, the
  verbatim `source_text` grounding), so the parser and grounding guard keep
  working. The rules carry the `{entity_types}` and `{relationship_types}`
  variables, rendered from config at call time
  (`GraphExtractionPrompt.config_variables`). A variable that stands for
  rule text the template would otherwise spell out declares it
  (`inlined_slots`): the extraction-record fingerprint hashes the template
  with it filled in, so moving rule text into a variable re-keys no record.
- **Schema drafts** (`run-prompt-tuning --schema-out`, `SchemaDraft` in
  `application/prompts/tuner.py`): the profile's entity types (the
  configured specs when the config declares them), the configured
  relationship types, then those the grounded examples state most often,
  counted by `schema_key` (at most `MAX_DRAFT_RELATIONSHIP_TYPES` more).
  Configured patterns are kept, and a type that has them gets no other;
  any other type gets patterns only when every (source type, target type)
  the examples give it between two draft entity types appears
  `MIN_PATTERN_SUPPORT` times, so `align` never counts the draft's own
  examples off pattern. A draft over the loaded schema therefore drops
  nothing it declares; the CLI still refuses to write over that file or,
  without `--force`, any existing one. The file is a
  `schema_source: {kind: file}` document validated as a schema before it is
  written, and names `align`, never `strict`: a draft from a few chunks
  would drop what they did not show. A pure function of the profile and the
  records, so a fixed model gives the same file.

---

## 12. Configuration System

The configuration is one nested Pydantic tree (`domain/models/config.py`,
root `Config`) loaded by `application/config.py` (`get_config`).
[docs/reference/configuration.md](./reference/configuration.md) is generated from it by
`scripts/gen_config_reference.py`, and `tests/unit/test_config_reference.py`
fails when it is stale; `config-template.yaml` is a short starting file with
the keys most deployments set (`tests/unit/test_config_template_defaults.py`
checks that its values are the defaults and each key is in the reference).
[docs/reference/cli.md](./reference/cli.md) is generated the same way from the CLI parsers
(`scripts/gen_cli_reference.py`, `tests/unit/test_cli_reference.py`).

- **Loading** (`Config.as_input`). The config is validated once from its
  layers: defaults < `backend` preset < file < environment. `UKR__<SECTION>__<KEY>`
  sets any key, resolved against the model tree (no variable table to keep in
  step); `AWS_PROFILE` / `AWS_REGION` fill only keys the file leaves unset. Unknown keys fail the
  load unless `allow_unknown_keys`, a moved key (`MOVED_KEYS`) fails
  naming its replacement, and a removed key (`REMOVED_KEYS`) logs a warning
  and is ignored. `Config` keeps the input it was validated from, so
  `with_provider_overrides` (the CLI provider flags) rebuilds from it plus
  every later change, and switching `--backend` drops only what the old preset
  filled in.
- **Model selection** (`ModelsConfig`). One `models` section names every
  model: the two tiers, per-role overrides (`ModelRole`), the embedding model
  with its dimension and the rerank model. Each family's provider stays
  `providers.<family>.kind`, and `Config._check_models` validates the ids
  against it. Values stay as written and resolve on read (`Config.model_id`,
  `embedding_model_id`, `rerank_model_id`); a setting of a provider the config
  does not select is reported at load.
- **Stage cache keys** (`domain/ingestion/cache_keys.py`). `CONFIG_KEYING`
  classifies every field: keyed from the first stage whose output it shapes,
  or unkeyed with the reason. A stage's key hashes the keyed fields in effect
  for it and every upstream stage (a disabled block contributes its switch
  alone), the package version and the corpus manifest. An unclassified field
  keys every stage (a miss, never a stale hit), and
  `tests/unit/test_cache_key_coverage.py` fails until it is classified. The
  same table decides what the `run-eval` manifest's `retrieval_config_sha256`
  hashes (`answer_settings`): every field except the unkeyed ones marked
  `shapes_answers=False` (call routing, throughput, telemetry, provisioning,
  cache storage) and a few location fields (corpus and registry paths, the
  table name, the guardrail id).
- **Extraction settings.** The keyed fields marked `per_document` form a
  document's extraction settings: their fingerprint is recorded on each
  committed document and the settings once per registry scope, compared field
  by field on the next run (a setting a release adds is no change while it
  holds its default). A field added to a keyed section whose default keeps
  the earlier behaviour is listed in `_KEYED_ONCE_SET`: it counts for no key
  until set otherwise, so an upgrade re-keys no stage cache or extraction
  record. The description summary settings key graph resolution and are
  no extraction setting (`per_document=False`: they summarize merged
  descriptions, in no unit's record, and a summary records them in
  `summarized_with`). A unit record's key holds them as the default
  configuration has them (`_summary_key`, the same whatever the run's
  `models` tiers), and a unit without a record under it is looked up under
  the keys earlier `main` builds stored it under too (no release stored
  the extraction store; `legacy_unit_record_fingerprints`: them at their
  defaults resolved through the run's tiers, and at the run's values, the
  key of a record stored while they keyed graph extraction):
  `ExtractionStore.load` stores a record found there under the current
  key, the one its unit names. `changed_settings` ignores them in a
  scope's earlier record. So stored records are found, documents read as
  unchanged, and a summary change asks no unit again. The lookup is
  kept until the first release after 0.2.3 and removed in it.
  The schema keys count as extraction sees them
  (`cache_keys._as_extracted`): a type by its label and description (a
  string item as written, a mapping as the string that reads back the
  same), a pattern by its relation and ends, `schema_mode` only where
  `SchemaConformance.applies` would differ, never an IRI. What only
  indexing reads of the schema (the IRIs, `off` against `align` with entity
  types alone) is the annotation digest (`SchemaAnnotator.digest`), keyed
  into the indexing stage. The registry records it once per index
  namespace (`DocStatusPort.get/put_annotation_digest`), not per scope:
  a pass rewrites every scope's items, so a per-scope record misses a
  schema reverted on one scope after another scope's pass. Before its
  first store write, an incremental run whose digest differs from the
  record sets it to `REANNOTATING_DIGEST` (`incremental.annotations_due`),
  rewrites the namespace's annotations without extracting
  (`reannotate_on_schema_change` → `IndexingManager.reannotate`) and
  records its digest only once the pass completed, so a pass that stopped
  is made again by the next run under any schema. A namespace without a
  record is unknown (one pass), unless the registry held none of its
  documents before the run and neither store holds an item of the suffix
  (`holds_any_item`, one bounded read each; recorded without reading): a
  lost or replaced registry is no proof the stores are empty. A reset marks its
  namespaces before the rebuild and records its digest when every item
  was written (`record_rebuilt_annotations`). Adding a section: §15.
- **Extraction schema** (`domain/models/schema.py`). The entity and
  relationship types and the allowed patterns live inline in
  `processing.graph_extraction` (`entity_types`, `relationship_types`,
  `relationship_patterns`, `schema_mode`), read as one frozen
  `ExtractionSchema` (types with an optional ontology `iri`, patterns, mode,
  a content `digest`). Labels compare by `schema_key` (NFKC, each run of
  whitespace and `_` one `_`, case-folded and normalized again, so a key is
  its own key) and nothing looser: no synonyms, plurals or edit distance. `schema_source` names where the schema comes
  from: `inline`, or a registered source (`file`, `owl`) that
  `application/bootstrap.resolve_extraction_schema` loads once at start-up
  (`DataIngestionPipeline`, `EvaluationManager`, `PromptTuner`) into the
  inline keys of a config copy, whose source is then `inline`.
  `GraphExtractionConfig.extraction_schema()` raises `ConfigurationError`
  on a config whose source is not loaded yet, so a component built straight
  from one (an `IndexingManager`, an extractor) fails instead of reading
  the default schema. The extractor, the prompts
  and the cache keys read only the inline keys, so the loaded content is
  keyed and its location (`schema_source.path`) is not: a moved file
  re-extracts nothing, an edited one re-extracts. The extractor and the
  gleaner apply it through `domain/ingestion/schema_conformance.py`
  (`SchemaConformance`, built by `BaseProcessor`): it applies in `strict`,
  and in `align` once relationship types are declared (entity types alone
  only shape the prompt, so configs from before the schema keys keep their
  records). An in-schema label is written as declared when parsed, before
  the ids are built; `conform` then checks each unit's statements after
  grounding and the confidence threshold, where `strict` drops off-schema
  entities (rejection `off_schema`, lifted by an in-schema statement of the
  same entity) with their relationships, off-schema relationship types, and
  relationships a pattern of their type disallows between two endpoints the
  unit types with declared types. The check is per unit and order-free, so
  incremental runs converge with a full build; its drops are not failures.
  Indexing then stores, from the same schema, whether each merged item's
  type is declared (where the schema applies) and its type's IRI (whenever
  one is declared and the mode is not `off`; §4.2, "Schema annotations"); the
  filter schema (`adapters/storage/filter_schema.py`) declares `in_schema`
  and `type_iri` on entities and Neptune entity vertices and `in_schema` and
  `predicate_iri` on relationships, but no `type` on relationships, which
  would empty that index under `mix` and `hybrid`. `GraphRAGChain` refuses a
  filter on an annotation the configured schema writes on no item
  (`SchemaAnnotator.fields`), which could only empty the streams reading it,
  and the OpenSearch retriever probes the annotation keys as it probes
  `attr_*` keys, so an index that does not map them matches no value of
  them instead of failing the kNN filter.

---

## 13. Testing Strategy

Tests (`tests/{unit,integration,property,fixtures/fakes}/`) run **without AWS
by default**.

- **Port-based fakes** (`tests/fixtures/fakes/`) implement the stores and the
  registry in memory; `moto` covers the DynamoDB and S3 adapters.
- **Property tests** (hypothesis) cover merge laws, incremental-versus-full
  convergence (under a `strict` extraction schema too), interrupted runs,
  fusion monotonicity, MinHash and token-estimate equivalence, and the
  extraction schema (`strict` keeps a subset of `align`, the per-unit check
  is idempotent and order-free, `schema_key` is its own key); `tests/integration/test_doc_status_conformance.py` checks the
  `DocStatusPort.diff` partition against every registry.
- **Store conformance suites**
  (`tests/integration/test_{graph_store,vector_store,retrieval}_conformance.py`)
  write each store contract once and run it on every backend through the
  `StoreHarness` (`tests/fixtures/store_backends.py`): the fakes, the embedded
  SQLite stores, the `docker/compose.local.yaml` containers (`LOCAL_STORES=1`)
  and Neptune and OpenSearch Service (`-m aws`); `test_store_parity.py`
  checks that SQLite and an OpenSearch container rank a synthetic corpus alike
  (same lexical and hybrid top 10, kNN recall@10 ≥ 0.98).
- **Markers**: `unit`, `integration`, `property`, `aws` (real AWS, excluded in
  CI), `slow`. Run `uv run pytest -m "not aws" --cov=unified_kg_rag`.

---

## 14. CI/CD and Security

- **CI** (`.github/workflows/`): `quality` gates pull requests and pushes to
  `main` (lint, types, tests with the coverage floor, the image build, the IaC
  synth with cdk-nag), `licenses` checks dependency licenses, `security` runs
  an ASH scan that fails on an unsuppressed finding at MEDIUM or above
  (configuration and path-scoped suppressions in `.ash/.ash.yaml`); the local
  checks are in [CONTRIBUTING.md](../CONTRIBUTING.md#checks-to-run-before-pushing).
- **Dependabot** updates the `uv` locks (root and `iac/`) and the SHA-pinned
  Actions weekly, 7 days after a release at the earliest (security updates
  at once); known-breaking bumps are held back with a recorded reason.
- **Hardening.** Content hashes use SHA-256 only; credentials come from the
  environment, never from code.
- **Guardrail at the query's edges** (`ports/guardrail.py`,
  `adapters/guardrail/bedrock.py`). `GraphRAGChain` screens the raw query
  (source `INPUT`) before memory, routing and retrieval, and the final answer
  (`OUTPUT`) after generation; a block short-circuits, a mask replaces the
  text, and a guardrail error fails the query closed. The model calls in
  between run unguarded: an attached guardrail judges the whole prompt, so on
  Converse it treats the retrieved corpus and the instructions as user input
  (false prompt-attack blocks, cost of context times calls), and on
  InvokeModel its prompt-attack filter is inert without input tags. Ingestion
  and evaluation models are never guarded. The opt-in contextual grounding
  check rides on the answer's check: the chain hands the guard a
  `GroundingContext` (the standalone question and the context string the
  answer prompt carried, plus its sections by priority) and the adapter
  sends it as `grounding_source` / `query` blocks, cut to Bedrock's limits
  by rank, with the answer first so a masked reply's first output is the
  answer.

---

## 15. Extension Guide

This section is the single reference for extending the framework. Extensions come in two kinds:

- **Plugins**, which live in your own package and import only from `unified_kg_rag` (§2.5): backends through the provider registry or constructor injection, and renderers, parsers, schema sources and seed sources through their registries. None needs a change to the framework's code.
- **In-tree changes**, made in a fork or contributed upstream: a search strategy and an evaluator. `register_strategy` is keyed by the closed `SearchStrategy` enum, which `RAGInput.search_strategy`, `--search-strategy` and `search.auto_routable_strategies` validate against, so a new strategy adds an enum member; an evaluator adds an `EvaluatorType` member and a branch in `EvaluationManager._resolve_evaluator_class`. The registry still keeps the chain's dispatch code unchanged.

- **New search strategy** (in-tree): Add a `SearchStrategy` enum member (`domain/models/retrieval.py`; the CLI choices follow it) + subclass `BaseSearchStrategy` + `@register_strategy(SearchStrategy.X, required_roles=(...), query_inputs=frozenset({QueryInput.ENTITIES}))` + export from `adapters/search_strategies/__init__.py` + an entry in `STRATEGY_ROUTING_GUIDE` (`domain/prompts/retrieval.py`; a unit test requires one for every strategy but `auto`). `query_inputs` declares the query-side LLM extractions it reads (`ENTITIES` for `entity_focus`, `DUAL_KEYWORDS` for `hl_keywords`/`ll_keywords`); the chain skips the rest. A strategy answering from community reports also passes `needs_community_reports=True`, so `auto` skips it when none are built. A strategy that runs another one's search (as `iterative` runs `local` or `mix`) passes `wraps=` a function of the config naming it: the chain then extracts what that one reads and packs its context as that one does (`wrapped_strategy`), and the wrapper builds its base through `get_strategy_spec(...).strategy_class`, not by name. Fetch by id through `BaseSearchStrategy._fetch_by_ids` (batched by the live limits), and cut or batch any other list filter (`terms`) to the live `max_terms_count` from `aindex_limits` of the index it reads: a longer list is refused (`InvalidFilterError`). Override `filtered_index_prefixes` with the indexes it searches with the caller's filters, so a filter key none of them declares is refused instead of dropped (the default checks every index), and `filtered_retrievers` with the retrievers it applies them to (the default: every one it holds; `mix` and `hybrid` leave out the graph without graph expansion); build each sub-search's filters with `_scoped_filters`, which keeps the caller's and intersects a key both set (`id`/`source_id`/`target_id`: the scope's values the caller allows; the chain refuses a range on them); `reserved_filter_keys` names the keys it scopes by with values a caller value cannot be intersected with (`global`: its cut ranges or browse from the roots, and the multi-valued `community_ids`, where a text unit in a caller's community and a scope's one passes both filters but the two lists may not intersect), which the chain refuses. No edit to `rag_chain`; add it to `search.auto_routable_strategies` if `auto` may pick it. Regenerate the references (the new member is a `--search-strategy` choice and a config value): `uv run python scripts/gen_cli_reference.py` and `uv run python scripts/gen_config_reference.py`. A minimal strategy, a document search fused and reranked like `simple` (`adapters/search_strategies/simple_search.py` is the full version, with the index gates):

  ```python
  # unified_kg_rag/adapters/search_strategies/keyword_search.py
  # (after adding SearchStrategy.KEYWORD = "keyword" to domain/models/retrieval.py)
  import time

  from langchain_core.runnables import RunnableConfig

  from unified_kg_rag.adapters.retrieval.base import BaseSearchStrategy
  from unified_kg_rag.adapters.retrieval.strategy_registry import register_strategy
  from unified_kg_rag.domain.models import (
      RetrieverRole,
      SearchQuery,
      SearchResult,
      SearchStrategy,
  )


  @register_strategy(SearchStrategy.KEYWORD, required_roles=(RetrieverRole.DOCUMENT,))
  class KeywordSearchStrategy(BaseSearchStrategy):
      async def asearch(
          self, query: SearchQuery, config: RunnableConfig | None = None
      ) -> SearchResult:
          start, rerank = time.time(), self._new_rerank_trace()
          hits = []
          if self.document_retriever:  # bound through required_roles
              hits = await self._safe_aretrieve(
                  self.document_retriever, query, "Document retrieval"
              )
          results = await self._fuse_and_rerank(
              {"documents": hits},
              rerank_trace=rerank,
              top_k=query.top_k,
              retrieval_multiplier=query.retrieval_multiplier,
              query=query.query,
          )
          return SearchResult(
              query=query,
              results=results,
              total_results=len(results),
              search_strategy="keyword_search",
              processing_time=time.time() - start,
              metadata={"rerank": rerank.summary()},
          )
  ```
- **New storage/LLM backend**: Implement the relevant port, then either register a builder for a new `providers.<role>.kind` with `@register_provider(ProviderRole.X, "kind")` (`adapters/provider_registry.py`) and select it in config, or pass the port to the constructor that uses it (see "Custom backends" below). Do not hardcode it into a manager's `__init__`.
- **New evaluator** (in-tree): Subclass `BaseGraphRAGEvaluator` (`from unified_kg_rag.adapters.evaluators.base import BaseGraphRAGEvaluator`) + add a branch in `EvaluationManager._resolve_evaluator_class` + an `EvaluatorType` enum. Regenerate the configuration reference, whose `evaluation.enabled_evaluators` lists the types: `uv run python scripts/gen_config_reference.py`.
- **New renderer**: Subclass `BaseRenderer` + `@register_renderer("name")`. Registration happens on import: `run-visualization` sees those that `adapters/renderers/__init__.py` imports. Override `check_dependencies` to fail fast when an optional package the renderer imports is missing, and set `needs_centrality = True` when `render` reads `RenderContext.centrality`: `run-visualization` computes the centrality metrics (betweenness alone is O(V·E)) only when a selected renderer sets it. Regenerate the CLI reference, where the name is a `--renderers` choice: `uv run python scripts/gen_cli_reference.py` (and `uv run python scripts/gen_config_reference.py` when it adds a `graph.visualization` settings section).
- **New parser / file format**: `ParserFactory.register_loader(".ext", MyLangChainLoader, loader_kwargs=..., file_type_name=...)` — any `langchain_core` `BaseLoader` subclass (the built-in loaders in `adapters/ingestion/file_loaders.py` are examples); no edit to the factory, and the extension is then auto-discovered + parseable. Override a built-in by registering its extension.
- **New schema source** (an ontology format or service): register a loader
  for a new `schema_source.kind` with `@register_schema_source("kind")`
  (exported from `unified_kg_rag`; the built-in `file` and `owl` loaders
  live in `adapters/schema_sources/`, imported by the package `__init__`,
  and read an `s3://` path through `adapters/source_files/locations.py`:
  HeadObject size check, paged prefix listing, S3 errors as
  `SchemaSourceError`, or the error type a `SourceTerms` names; what the
  built-in sources read: [Extraction schema](./guides/extraction-schema.md),
  [Ontology mapping](./reference/ontology-mapping.md)).
  A loader that imports an optional package declares it:
  `@register_schema_source("kind", requires=("module", "how to install
  it"))`, and the CLIs check that the module is importable before any model
  call, exiting 2 with the install hint (`owl` declares `("rdflib", "the
  ontology extra: uv sync --extra ontology ...")`). Import the package
  inside the loader, not at module level, so registering the kind needs
  nothing, and raise `SchemaSourceError` naming the extra when the import
  fails; re-registering a kind replaces its loader and its requirement. It
  receives a `SchemaSourceContext` and returns a `SchemaDocument`: the
  `entity_types` (or `None` to keep the config's), `relationship_types` and
  `relationship_patterns` it supplies, each a string or a spec. The context
  holds:
  - `source`: the `schema_source` settings: `kind`, `path` (left as written
    when it is a URL), `options` (a free mapping for the loader's own
    settings, not keyed: the loaded content is) and `secret_arn` (a Secrets
    Manager secret to read credentials from at load time).
  - `config`: the run's config (its schema keys are not loaded yet).
  - `boto_session()`: the run's AWS session (the injected `Providers`
    one, else one from `aws.profile_name` and `aws.region_name`), built on
    the first call only, and `client(service)`, a client in the region the
    config sends that service to (`aws.bedrock_region` for Bedrock,
    `aws.region_name` for the rest).
  - `location`: `path` without the userinfo, query and fragment of a URL.
    Log and raise with it, never with `path` or the secret's value.

  The loader must return the types in the same order for the same content
  (sort what a set, a graph or a paged API yields): the order is part of
  the prompt and of the cache keys, so a reordered snapshot re-extracts the
  corpus. It translates, it does not reason.

  **Reusing the ontology mapping.** The `owl` source is split so that its
  mapping does not depend on RDF:
  - `ontology_model.py`: `OntologyModel`, a pure Pydantic model of the named
    classes and properties a reader found and what it stated about them
    (annotations as language-tagged `Text`, parents, equivalents,
    deprecation; per property its kind, domains and ranges as sets of
    unions, inverses and symmetry) plus the prefix bindings. Statements,
    not inferences, and no blank nodes.
  - `ontology_mapping.py`: `MappingOptions` (selection, labels, collisions,
    descriptions and hints, patterns, caps) and `map_ontology(model,
    options) -> OntologyMapping` (`document`, the `SchemaDocument`, and
    `report`, a `MappingReport` of what was dropped, lifted, merged,
    skipped or qualified). It owns every rule: the RDFS-style closure, the
    flat hierarchy with push-down and lift, labels, order and the caps,
    and depends only on `domain`.
  - `owl_reader.py` builds the model from parsed statements; `owl.py` is
    the loader: it locates and checks the files (formats, size, and the
    shapes the parsers take quadratic time over, refused before parsing),
    parses their bytes, reads the model and maps it with
    `OwlSourceOptions`, which is `MappingOptions` plus the reader's
    `format` and `max_bytes`.
  - The file handling is not owl's own: `adapters/source_files/` holds it
    for any source that reads RDF. `rdf_files.locate`/`locate_s3` pick the
    files and the format of each, `rdf_files.check_bytes` refuses what a
    parser must not see, and `rdf_parse.parse` (the only module there
    importing rdflib) parses the bytes; `rdf_terms` holds the RDF
    vocabulary IRIs and the pure naming helpers (`local_name`,
    `upper_snake`, `best_text`, `Text`, ...) the mapping shares with the
    `rdf` seed source. A `SourceTerms` record carries the exception type
    and how messages name the source, its options, its size cap
    (`max_bytes`, which a seed's terms name `processing.seed_graphs[].max_bytes`)
    and its kind, and any `SourceClientProvider` (`location`, `client(service)`)
    gives `locations.SourceReader` its S3 client.

  A loader for an ontology held elsewhere (an ontology service, or a
  non-RDF format such as LinkML) added to `adapters/schema_sources/` builds
  the same `OntologyModel` from its snapshot and calls `map_ontology`, so
  its users get the same labels, collisions, closure, patterns, hints,
  order and caps as `kind: owl`, and the same IRIs end up in `type_iri` and
  `predicate_iri`. Its options model subclasses `MappingOptions` with the
  reader's own settings, validated from `context.source.options` (wrap a
  `ValidationError` in `SchemaSourceError`, as `owl._options` does).
  These modules are unstable internals, not part of the public API (§2.5):
  `unified_kg_rag` does not export `map_ontology`, `MappingOptions`,
  `OntologyModel`, `OntologyClass`, `OntologyProperty` or the vocabulary
  constants, and their names, fields and mapping rules may change in any
  release without a "Breaking" entry. A loader that imports them pins the
  package version and runs its own tests on each upgrade:

  ```python
  from pydantic import Field

  from unified_kg_rag import SchemaDocument, SchemaSourceContext, register_schema_source
  from unified_kg_rag.adapters.schema_sources.ontology_mapping import (
      MappingOptions,
      map_ontology,
  )
  from unified_kg_rag.adapters.schema_sources.ontology_model import (
      RDFS_LABEL,
      OntologyClass,
      OntologyModel,
      OntologyProperty,
      Text,
  )


  class ServiceOptions(MappingOptions):
      graph: str = Field(description="The service's graph to read")


  @register_schema_source("ontology_service")
  def load_ontology_service(context: SchemaSourceContext) -> SchemaDocument:
      options = ServiceOptions.model_validate(context.source.options)
      snapshot = fetch_snapshot(context.source.path, graph=options.graph)
      model = OntologyModel(
          classes=tuple(
              OntologyClass(
                  iri=c["iri"],
                  annotations={RDFS_LABEL: (Text(value=c["name"], language="en"),)},
                  parents=frozenset(c.get("parents", ())),
              )
              for c in snapshot["classes"]
          ),
          properties=tuple(
              OntologyProperty(
                  iri=p["iri"],
                  domains=frozenset({frozenset(p["domain"])}),
                  ranges=frozenset({frozenset(p["range"])}),
              )
              for p in snapshot["properties"]
          ),
      )
      return map_ontology(model, options).document
  ```

  The order of the model's terms does not matter (`map_ontology` sorts).
  A loader that maps its snapshot itself returns a `SchemaDocument`
  directly, as a SHACL loader mapping node shapes would. A service loader
  (an ontology API, or an MCP server's schema tool) fetches its snapshot
  once:

  ```python
  import json

  from unified_kg_rag import (
      SchemaDocument,
      SchemaSourceContext,
      SchemaSourceError,
      register_schema_source,
  )

  @register_schema_source("ontology_service")
  def load_ontology_service(context: SchemaSourceContext) -> SchemaDocument:
      source = context.source
      token = None
      if source.secret_arn:
          secret = context.client("secretsmanager").get_secret_value(
              SecretId=source.secret_arn
          )
          token = json.loads(secret["SecretString"])["token"]
      try:
          # fetch_snapshot: your client of the service (HTTP, or an MCP
          # session calling the server's schema tool).
          snapshot = fetch_snapshot(
              source.path, graph=source.options.get("graph"), token=token
          )
      except OSError as e:
          raise SchemaSourceError(
              f"Cannot read the ontology service at '{context.location}': {e}"
          ) from e
      return SchemaDocument(
          entity_types=[
              {"label": c["name"], "description": c.get("doc", ""), "iri": c["iri"]}
              for c in sorted(snapshot["classes"], key=lambda c: c["iri"])
          ],
          relationship_types=[
              {"label": p["name"], "iri": p["iri"]}
              for p in sorted(snapshot["properties"], key=lambda p: p["iri"])
          ],
      )
  ```

  selected with `schema_source: {kind: ontology_service, path:
  "https://ontology.example.com/v1", options: {graph: products},
  secret_arn: ...}`. Registration happens on import, and the CLIs import
  only the built-in sources, so a run with a custom source starts from
  Python after importing its module. Nothing else changes: start-up writes
  the result into the inline keys, the config validates it (unique labels,
  patterns naming declared types), and the content keys the caches, so the
  snapshot a run used is the one it is keyed by. Raise `SchemaSourceError`
  for anything unreadable; it is a `ConfigurationError`, so the CLIs exit 2.
- **New seed source** (a graph format, or a service holding a graph):
  register a loader for a new `processing.seed_graphs[].kind` with
  `@register_seed_source("kind")` (exported from `unified_kg_rag` with
  `SeedSourceContext`, `SeedSourceOptions`, `SeedGraph`, `SeedNode`,
  `SeedEdge` and `SeedGraphError`; the built-in `jsonl`, `csv` and `rdf` loaders live in
  `adapters/seed_sources/`, imported by the package `__init__`; their
  formats: [Seed graph formats](./reference/seed-graph-formats.md)). A loader
  that imports an optional package declares it, `requires=("module", "how
  to install it")`, and the CLIs check it before any model call, as for
  schema sources (`rdf` declares rdflib). The loader receives a
  `SeedSourceContext` and returns a `SeedGraph` of `SeedNode`s and
  `SeedEdge`s:
  - `source`: the `processing.seed_graphs` item (`name`, `kind`, `path`,
    left as written when it is a URL, `options`, `secret_arn` and the
    caps). Validate `options` with `context.options(Model)`, a subclass of
    `SeedSourceOptions` (`from unified_kg_rag import SeedSourceOptions`),
    which carries `dangling` and `qualifier_property` and forbids unknown
    keys.
  - `config`: the run's config, the extraction schema already in the
    inline keys (`config.processing.graph_extraction.extraction_schema()`),
    for a loader that maps class IRIs itself.
  - `boto_session()`, `client(service)` and `location`, as on
    `SchemaSourceContext`; and `terms`, whose call builds a
    `SeedGraphError` naming the seed (`raise context.terms("...")`). Log
    and raise with `location`, never with `path` or the secret's value.

  What it returns must depend only on what it read and the options, never
  on the order (`convert_seed` sorts, applies the schema, the collisions,
  the caps and the sharding). Ids are the seed's own (edges name them);
  identity in the graph is the node's name. Nothing else changes: the
  conversion's digest keys the stages, and the shards are documents of the
  run's corpus (§4.5).

  **Reusing the readers.** `adapters/source_files/` serves seed sources as
  it serves schema sources: `locations.resolve_source_path` turns a `path`
  into a local `Path` (a plain path or a `file://` URL) or the `s3://` URL,
  refusing other schemes, as every built-in kind does;
  `locations.SourceReader(context, terms=context.terms)` heads, lists and
  reads S3 objects within a byte cap, and `rdf_files` / `rdf_parse` locate, check and parse RDF files
  with the `owl` source's safety checks (pass `context.terms`);
  `rdf_terms` names RDF terms as both built-in RDF readers do. The
  property-graph helpers of the built-ins (`property_graph.FileReader`,
  `GraphBuilder`: sized reads, line caps, refusals naming the file and
  line) are for kinds added inside `adapters/seed_sources/`. A reader of
  RDF from elsewhere (a SPARQL endpoint's `CONSTRUCT` result, say) hands
  its triples as `abox_mapping.RdfTerm` tuples to
  `abox_mapping.map_abox(triples, options, schema, terms=context.terms)`
  (pure, no rdflib) and gets the `rdf` kind's rules: instances, names and
  descriptions by language, the most specific mapped class, CURIE
  property keys, nothing inferred. Its options model subclasses
  `AboxMappingOptions`.

  A service loader fetches a snapshot once per run:

  ```python
  import json

  from pydantic import Field

  from unified_kg_rag import (
      SeedEdge,
      SeedGraph,
      SeedNode,
      SeedSourceContext,
      SeedSourceOptions,
      register_seed_source,
  )


  class ServiceOptions(SeedSourceOptions):
      graph: str = Field(description="The service's graph to read")


  @register_seed_source("ontology_service")
  def load_ontology_service(context: SeedSourceContext) -> SeedGraph:
      options = context.options(ServiceOptions)
      source = context.source
      token = None
      if source.secret_arn:
          secret = context.client("secretsmanager").get_secret_value(
              SecretId=source.secret_arn
          )
          token = json.loads(secret["SecretString"])["token"]
      try:
          # fetch_instances: your client of the service (HTTP, SPARQL, or an
          # MCP session calling the server's tool).
          snapshot = fetch_instances(source.path, graph=options.graph, token=token)
      except OSError as e:
          raise context.terms(f"cannot read '{context.location}': {e}") from e
      return SeedGraph(
          nodes=tuple(
              SeedNode(id=n["iri"], iri=n["iri"], name=n["label"], type_iri=n["class"])
              for n in snapshot["instances"]
          ),
          edges=tuple(
              SeedEdge(source=e["from"], target=e["to"], predicate_iri=e["property"])
              for e in snapshot["links"]
          ),
      )
  ```

  selected with `seed_graphs: [{name: catalog, kind: ontology_service,
  path: "https://ontology.example.com/v1", options: {graph: products},
  secret_arn: ...}]`, from Python after importing the module (the CLIs
  import only the built-in sources).

  **Copy, or federate.** Such a loader copies: the snapshot becomes part
  of the graph, embedded, clustered and keyed like the rest, and its IRIs
  line up with the service's through `Entity.iri`, `type_iri` and
  `predicate_iri`. Federation, querying the service at retrieval time
  instead, is not built. Its seam would be a retrieval-time port (look up
  the IRIs of names, a node's neighbours, a node's description) that
  expands stored entities by `iri` without copying; nodes it returns would
  be in neither Leiden nor the vectors, so global search would not see
  them, and the `source_kinds` vocabulary can name them. The registry,
  `Entity.iri` and `source_kinds` are the seams either way.
- **New config section**: Define a Pydantic `BaseModel` → attach it to its parent via `Field(default_factory=...)` → describe every field and regenerate `docs/reference/configuration.md`; add a key to `config-template.yaml` only when most deployments set it (see §12).


### Custom backends (run without AWS)

Every cross-service dependency is behind a port. Each port is a role in the
provider bundle whose adapter `providers.<role>.kind` selects through the
provider registry, and the orchestrators also accept any port by
**constructor injection**, so a custom backend needs no subclassing or
dispatch edit. The ports, their default adapters and the local ones (the
`[local]` extra):

| Port | Contract | Default adapter (local adapters) | Inject via |
|---|---|---|---|
| `LLMFactoryPort` / `EmbeddingFactoryPort` / `RerankFactoryPort` (`ports/model_factory.py`, `Protocol`) | `get_model()` / `get_model_info()` returning a LangChain-compatible model; optional LLM-factory hooks: `is_transient_error(exc)` (which errors the ingestion batches retry; default `adapters.llm.retry.looks_transient`: network errors, HTTP 408/429/5xx), `is_item_error(exc)` (which errors count against a document, `indexing.max_document_failures`; default `looks_like_item_error`: HTTP 400/413/422 other than a rejected model or setting, read from `status_code` or `response.status_code`) and `transient_retry` (a `TransientRetryConfig`: with `is_transient_error`, query chains are retried under it) | `BedrockLanguageModelFactory` / `BedrockEmbeddingModelFactory` / `BedrockRerankModelFactory` (`openai_compatible`, `ollama`; `cross_encoder`, `none`) | `providers=` (below) on `GraphRAGChain`, `DataIngestionPipeline`, `EvaluationManager`; `GraphRAGChain(model_factory=...)` is an LLM-only bundle |
| `TokenCounterPort` (`ports/model_factory.py`, `Protocol`) | `count_tokens()` / `truncate_to_token_limit()` | `BedrockTokenCounter` (`LocalTokenCounter`: estimate or tiktoken; follows the LLM kind) | `Providers(token_counter_factory=...)` |
| `VectorIndexer` / `GraphIndexer` (`ports/indexer.py`, ABC) | `index_*` / `upsert_*` / `delete_by_id`; the incremental lookups (`GraphIndexer.find_ids_citing` / `find_relationship_ids_citing` / `find_entity_ids_by_name_key`, `VectorIndexer.find_documents` / `delete_extraction_records` / `retain_extraction_records` / `prune_text_unit_communities`) default to "cannot tell" (`None`: nothing is found through them, the merge reads every name) and to deleting nothing; the extraction-record store (`VectorIndexer.read_extraction_records` / `write_extraction_records` / `stored_extraction_record_keys`, by record key) defaults to keeping none (`{}`: every text unit is extracted, nothing is stored), and `stored_extraction_record_keys` to the keys `read_extraction_records` returns; `VectorIndexer.check_index_creation(suffix, rebuild=)` (called once per run before the first model call, for the indices the run will create: all of them on a `rebuild` (full build or reset), the missing ones otherwise; raises `StoreRequestRejectedError` for settings the store refuses) defaults to accepting them; `VectorIndexer.read_vector_copies` defaults to `{}` (every item an incremental run touches is written in full); `GraphIndexer.read_entities` / `read_relationships` default to `[]`, so without overrides the delta merge overwrites the stored descriptions, lineage (`text_unit_ids`) and weights; `GraphIndexer.begin_rebuild` / `end_rebuild` (in-place full build, §5.2) default to `False` (the full build clears each label first); `initialize()` (no-op by default) / `clear()` return `None` and raise a typed store error (`StoreConnectionError`, `StoreConfigurationError`, ...) on failure: a returned `False` is ignored; an `in_process = True` attribute (on the indexer and on the registry) says the store is embedded in this process, and only when all three say so does a stopped run release its lease at once (absent: remote) | `OpenSearchIndexer` / `NeptuneIndexer` (`gremlin`: the same adapter on Gremlin Server; `sqlite`: `OpenSearchIndexer` on `SQLiteSearchEngine` / `SQLiteGraphIndexer`) | `Providers(vector_indexer_factory=..., graph_indexer_factory=...)` (a new instance per call; its caller closes it), `DataIngestionPipeline(..., vector_indexer=..., graph_indexer=...)` (borrowed: never closed by the pipeline) or `IndexingManager(vector_indexer=..., graph_indexer=...)` (closed by `IndexingManager.close()`; §8.2) |
| retriever (role-keyed builder) | `BaseGraphRAGRetriever.aretrieve`; a document retriever must honour `SearchQuery.exclude` (leave out what it matches before the page is cut: the filtered walk's edge pages rely on it); `aindex_limits(prefix, suffix)` (the hits one search returns, the values one `terms` clause takes) defaults to the configured `index_settings`, so a backend whose live limits can be smaller overrides it; the strategies size every list they send by it (fetches by id in batches of `IndexLimits.id_batch_size`, walk frontiers and exclusions by `max_terms_count`, at most `search.max_concurrency` batches at once), so `aretrieve` may cut a search to its limits and need not batch; `filter_fields()` declares the filter keys the retriever applies (`None`, the default, accepts every key) and `filter_fields_for(prefixes)` those of the indexes a strategy filters (`BaseSearchStrategy.filtered_index_prefixes`), by default `filter_fields()`; `GraphRAGChain` rejects a caller key none of a strategy's `filtered_retrievers` declares, checking a custom retriever against the union of its keys | `OpenSearchRetriever` / `NeptuneRetriever` (`sqlite`: `OpenSearchRetriever` on `SQLiteSearchEngine` / `SQLiteGraphRetriever`) | `Providers(retriever_builders=...)` or `GraphRAGChain(retriever_builders={RetrieverRole.GRAPH: lambda: MyGraphRetriever(...)})` |
| `ContentGuardPort` (`ports/guardrail.py`, `Protocol`) | `async acheck(text, source, *, grounding=None)` returning a `GuardVerdict` (`none` / `masked` / `blocked`, the text to continue with, and `grounding`, a `GroundingCheck`, when a `GroundingContext` was given); raises when it cannot decide. `grounding` is passed only while `aws.bedrock.guardrail.contextual_grounding.enabled` is set | `BedrockGuardrail` (built when `aws.bedrock.guardrail.identifier` is set) | `GraphRAGChain(guard=...)` |
| `DocStatusPort` (`ports/doc_status.py`, `Protocol`) | `get` / `get_many` / `put` / `put_many` / `delete` / `list_namespace` / `diff` (which must take `extraction_fingerprint` and fill `DocumentDelta.quarantined_unchanged`), `record_fits`, the write lease (`try_acquire_lease` / `lease_holder` / `release_lease`) a scope's last extraction settings (`get_extraction_settings` / `put_extraction_settings`) and the schema annotations a namespace's stores hold (`get_annotation_digest` / `put_annotation_digest`), all called directly; a record's namespace is the `<namespace>#` prefix of its `doc_id` (`doc_id_namespace`), and `diff` with a scope reads only that namespace. A subclass inherits `get_many` / `put_many` / `record_fits` defaults (one record per call, no size limit) | `DynamoDBDocStatusStore` (`SQLiteDocStatusStore`) | `Providers(doc_status=...)` or `DataIngestionPipeline(..., doc_status=...)` (either turns incremental indexing on); pass `tests/integration/test_doc_status_conformance.py` |

**Errors from a custom backend** are classified by type, never by message
(§8.1): raise a `ConnectionError` or `ConfigurationError` subclass (the typed
store errors in `shared/exceptions.py`) to fail the query; any other
exception degrades only the section that raised it.

**Protocols, not base classes.** The model-factory and doc-status ports are
`runtime_checkable` `Protocol`s, so a custom class only needs the right method
shapes:

```python
from unified_kg_rag import GraphRAGChain

# Not exported from unified_kg_rag: an internal path that may change in any
# release (§2.5); pin the package version when you import it.
from unified_kg_rag.domain.models.config import TransientRetryConfig

class MyLLMFactory:                       # structurally an LLMFactoryPort
    def get_model(self, model_id, **kwargs): ...   # returns a LangChain chat model
    def get_model_info(self, model_id): ...        # returns a LanguageModelInfo
    # Optional hooks (defaults: adapters.llm.retry.looks_transient /
    # looks_like_item_error; no query-chain retry without transient_retry):
    def is_transient_error(self, exc): ...         # retry this error?
    def is_item_error(self, exc): ...              # the document's own error?
    transient_retry = TransientRetryConfig()       # query-chain retry policy

chain = GraphRAGChain(config=cfg, model_factory=MyLLMFactory())
```

A factory must keep the `**kwargs` catch-all and ignore keys it does not
understand: callers pass framework arguments such as
`model_purpose=ModelPurpose.INGESTION` (the per-path policy: response cache,
ingestion effort, query retry), and a strict signature raises `TypeError` on
the first one.

**One bundle, every component.** `Providers` (`adapters/providers.py`) holds
the boto3 session, the model and token-counter providers and the store and
registry builders; `application/bootstrap.py` (`build_providers`) is the
composition root. Each orchestrator builds or receives one bundle and passes
it to everything it constructs (the chain's prompts, strategies, reranker,
token managers and retrievers; the pipeline's stages and indexers; the
evaluation judges, which reuse the chain's bundle by default).

```python
providers = Providers(
    cfg,
    llm_factory=MyLLMFactory(),
    embedding_factory=MyEmbeddingFactory(),
    # Reranking is on by default (providers.rerank.kind: bedrock) and would
    # otherwise call Bedrock; pass a factory or set providers.rerank.kind: none.
    rerank_factory=MyRerankFactory(),
    token_counter_factory=lambda model_id, **_: MyTokenCounter(model_id),
)
chain = GraphRAGChain(config=cfg, providers=providers)
pipeline = DataIngestionPipeline(cfg, pipeline_config, providers=providers)
```

- Orchestrator arguments (`retriever_builders=`, `doc_status=`,
  `vector_indexer=` / `graph_indexer=`, `GraphRAGChain(model_factory=...)`)
  are shorthand for a bundle built with them; passing one together with
  `providers=` raises `ValueError`.
- Anything not supplied is built lazily by the configured kind's builder (the
  model factories at most once per bundle); a component built without a bundle
  builds a default one. There is deliberately no DI container: the bundle
  covers exactly what the framework consumes.
- Conversation memory belongs to the chain and keys each conversation by index
  suffix, `user_id` and `conversation_id`; pass one `MemoryManager(cfg)` as
  `memory_manager=` to share history across chains.

**Provider registry.** `adapters/provider_registry.py` maps
`(ProviderRole, kind)` to a builder that receives the bundle; a new kind needs
no edit to the bundle, the config model or any call site:

```python
from unified_kg_rag import ProviderRole, register_provider

@register_provider(ProviderRole.LLM, "my_server")
def _build(providers):
    return MyServerFactory(providers.config)
```

`providers.llm.kind: my_server` then selects it (a non-Bedrock LLM kind also
needs a token-counter builder of the same kind). A kind that connects to a
server declares its setting (`endpoint=("<config key>", "<env var>")`), and
one that imports an optional package declares it
(`requires=("<module>", "<how to install it>")`): the CLIs refuse an unset
endpoint or a missing package at start-up (exit 2). `backend: local` is a config
preset (`domain/models/provider_config.py`), not a code path: it fills in
`providers`, `aws.neptune`, `aws.opensearch` and the context budget only
where the config leaves them unset.

**Reference implementations.** The in-memory fakes of `tests/fixtures/fakes/`
implement the indexer ports and run the whole pipeline
(`tests/integration/test_ingestion_stages.py`): the starting point for a
custom store. The embedded SQLite stores show the two ways to add a backend:
the vector store (`adapters/local/search_engine/`) implements the OpenSearch
client calls (`SearchEngineClient`), so the unchanged OpenSearch indexer and
retriever run on it; Gremlin cannot be emulated that way, so the graph store
is its own `GraphIndexer` (`SQLiteGraphIndexer`) plus a
`GraphExpansionRetriever` subclass implementing only the traversal.

### Deliberate Design Boundaries

Two boundaries are intentional:

- **`SearchQuery` carries adapter vocabulary (index and label prefixes).** A
  prefix names a logical collection each store pairing maps to its own storage
  (the embedded SQLite stores reuse them). The boundaries that matter, the
  indexer and model-factory ports, are abstracted and injected; a
  backend-neutral query model would be a large, risky churn that only a
  backend unable to map named collections would justify.
- **One physical OpenSearch index per index suffix and artifact type.** Fine
  for a handful of tenants; tens of thousands of suffixes multiply the shard
  count. The scale-out fix (one index per artifact type with an enforced
  tenant field and routing) changes the index, search and delete paths and is
  deferred. Metadata filters are relevance filters, not access control:
  isolation comes from separate suffixes
  ([runbook](./operations.md#query-access-and-tenant-isolation)).

---

## 16. Further Reading

Microsoft Research posts on GraphRAG and its follow-up methods, useful background for §6:

- [GraphRAG: Unlocking LLM Discovery on Narrative Private Data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
- [GraphRAG: New Tool for Complex Data Discovery Now on GitHub](https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/)
- [GraphRAG Auto-Tuning Provides Rapid Adaptation to New Domains](https://www.microsoft.com/en-us/research/blog/graphrag-auto-tuning-provides-rapid-adaptation-to-new-domains/)
- [Introducing DRIFT Search: Combining Global and Local Search Methods to Improve Quality and Efficiency](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/)
- [GraphRAG: Improving Global Search via Dynamic Community Selection](https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/)
- [LazyGraphRAG: Setting a New Standard for Quality and Cost](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
- [Introducing GraphRAG 1.0](https://www.microsoft.com/en-us/research/blog/moving-to-graphrag-1-0-streamlining-ergonomics-for-developers-and-users/)
