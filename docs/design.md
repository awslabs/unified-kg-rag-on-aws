# Unified Knowledge Graph RAG on AWS — Technical Documentation

> 🇰🇷 한국어판: [docs/design.ko.md](./design.ko.md)

This document is a **design reference for contributors and advanced users**, covering the architecture, algorithms, data model, and operational aspects of the `unified-kg-rag-on-aws` library. For the "what/why" and a quick start, see [README.md](../README.md); for "how to use it," see the [User Guide](user-guide.md); for the contribution workflow, see [CONTRIBUTING.md](../CONTRIBUTING.md). Extension recipes are in [§15](#15-extension-guide).

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

`unified-kg-rag-on-aws` is a library that reimplements the Microsoft GraphRAG and LightRAG methodologies on top of an AWS-native stack (Bedrock + Neptune + OpenSearch + S3 + DynamoDB). The core design principles are as follows.

- **Two methodologies, one infrastructure**: GraphRAG (community-summary) and LightRAG (dual-level keyword) share the same ingestion, indexing, caching, multilingual, and hybrid-search infrastructure, and **only the retrieval algorithm layer is swapped**.
- **Generalization first**: We avoid hardcoding, regex heuristics, and overfitting. Semantic judgments are delegated to the LLM or to authoritative data, token counting uses the Bedrock `count_tokens` API where the model accepts it and a script-aware estimate otherwise, and thresholds/weights are config-driven.
- **Hexagonal boundaries**: Domain/algorithm code depends on abstract ports, with concrete AWS adapters placed behind them.
- **Registry-based extension**: Search strategies and renderers are registered via decorator registries, so they can be extended without modifying dispatch code. Evaluators are mapped from `EvaluatorType` in one method (`EvaluationManager._resolve_evaluator_class`), so a new evaluator adds one branch there.

![Architecture at a glance](../assets/architecture-overview.drawio.png)

---

## 2. Hexagonal Architecture (Ports & Adapters)

### 2.0 Dependency Rule and Layer Map

Imports point **inward** (left imports right, never the reverse). `shared/` is a cross-cutting kernel any layer may use, so the package roots it imports stay dependency-light: importing any `domain/` module must not load LangChain, LangSmith, boto3/botocore, lxml, opensearch-py or gremlin-python, even transitively. LangChain-coupled helpers are imported from their own submodules (`shared.utils.langchain`, `shared.utils.document_converter`; the rich console helpers from `shared.utils.display`), never re-exported from `shared.utils`. `tests/unit/test_domain_purity.py` checks this in a clean interpreter. The two RAG methodologies (GraphRAG community-summary, LightRAG dual-level keyword) share one ingestion/indexing/caching/hybrid-search infrastructure and diverge only at the algorithm layer.

```mermaid
flowchart TB
    application["<b>application/</b><br/>CLIs, ingestion pipeline, indexing manager, RAG chain"]
    adapters["<b>adapters/</b><br/>AWS and local bindings, search strategies, retrievers"]
    ports["<b>ports/</b><br/>abstract interfaces"]
    domain["<b>domain/</b><br/>models, pure algorithms, prompts"]
    shared["<b>shared/</b><br/>config, logging, exceptions, utils"]
    application --> adapters --> ports --> domain
    application -.-> shared
    adapters -.-> shared
    domain -.-> shared
```

Solid arrows are the dependency rule (a layer imports only the layers it points to); dotted arrows show that any layer may import `shared/`. Search strategies are adapters: the domain only holds their registry, so both methodologies plug into the same ports.

```
unified_kg_rag/
├─ domain/              # technology-agnostic core (no boto3/LangChain/backend imports)
│  ├─ models/           #   Pydantic domain models
│  ├─ ingestion/        #   pure algorithms: delta_detector, graph_analyzer/
│  │  └─ merge/         #   builder/resolver, claim_resolver, merge/merger
│                       #   (IncrementalIndexer is orchestration, so it lives in application/)
│  ├─ retrieval/        #   strategy_registry, index_prefixes
│  └─ prompts/          #   version-controlled prompt templates
├─ ports/               # abstract interfaces the domain depends on (DocStatusPort,
│                       #   BaseIndexer/GraphIndexer/VectorIndexer,
│                       #   ModelFactoryPort — ports/__init__ is the port catalog)
├─ adapters/            # concrete technology bindings
│  ├─ aws/              #   Bedrock model factories, Neptune/OpenSearch clients, S3 transfer and caches
│  ├─ local/            #   Ollama/OpenAI-compatible models, cross-encoder, embedded SQLite graph/search stores
│  ├─ llm/              #   provider-neutral chain assembly, transient retry, token estimate
│  ├─ observability/    #   OpenTelemetry tracer, exporters, LangChain callback
│  ├─ storage/          #   Neptune/OpenSearch/SQLite-graph indexers (write-side port implementations)
│  ├─ retrievers/       #   Neptune/OpenSearch/SQLite-graph retrievers
│  ├─ doc_status/       #   DynamoDB/SQLite document-status registries (DocStatusPort)
│  ├─ search_strategies/#   simple/local/global/drift + lightrag(mix/hybrid/naive)
│  ├─ retrieval/        #   abstract retriever/strategy bases, hybrid scorer, token/memory managers
│  ├─ ingestion/        #   LLM/IO coupled: chunker, *_extractor, loader, parser,
│  │                    #   translator, gleaner, community_detector
│  ├─ renderers/        #   graph visualization renderers
│  ├─ evaluators/       #   evaluator bases + langchain/ragas evaluators (the LLM-free graph_aware,
│                       #   retrieval and answer_match evaluators live in evaluation/)
│  ├─ providers.py      #   Providers: the per-run bundle of model and store providers
│  └─ provider_registry.py # @register_provider: builders keyed by role and kind
├─ application/         # orchestration + entry points
│  ├─ cli/              #   run-ingestion/rag/eval/visualization/prompt-tuning
│  ├─ ingestion/        #   DataIngestionPipeline + pipeline_stages
│  ├─ storage/          #   IndexingManager (indexer fan-out)
│  ├─ retrieval/        #   rag_chain (GraphRAGChain, RAGInput/Output)
│  ├─ prompts/          #   PromptTuner (LLM-based corpus profiling)
│  ├─ bootstrap.py      #   build_providers: the composition root
│  └─ observability.py  #   configure_observability: installs the tracing backend
├─ shared/              # cross-cutting kernel (config, logging, exceptions, metrics,
│                       #   cache/pipeline manager, utils)
├─ evaluation/          # real logic package: evaluation_manager / graph_aware /
│                       #   retrieval / answer_match
└─ visualization/       # real logic package: render loop + embeddings/exporters
```

> Layout note: `evaluation/` and `visualization/` are **real logic packages** —
> `evaluation/` holds `evaluation_manager` and the LLM-free
> `graph_aware_evaluator` / `retrieval_evaluator` / `answer_match_evaluator`;
> `visualization/` holds the render loop plus the `embeddings/` and
> `exporters/` subpackages (the renderers are in `adapters/renderers/`). Everything else is imported from its real
> location (`application.retrieval.rag_chain`,
> `application.storage.indexing_manager`, `application.ingestion.pipeline`,
> `adapters.*`, `domain.*`).

> Adapter packages: vendor packages (`adapters/aws/`, `adapters/local/`) hold
> technology bindings, that is transport clients, embedded engines and stores,
> model factories (LLM, embedding, rerank), token counters and caches. Role
> packages hold the port implementations the application wires:
> `adapters/storage/` (`GraphIndexer` / `VectorIndexer`), `adapters/retrievers/`
> (retrievers), `adapters/doc_status/` (`DocStatusPort`) and
> `adapters/guardrail/` (`ContentGuardPort`). A role package
> imports a vendor package, never the other way round: `SQLiteGraphIndexer`
> (`storage/`) writes to `SQLiteGraphStore` (`local/`) as `NeptuneIndexer`
> writes through `NeptuneClient` (`aws/`).

### 2.1 Ports (Abstract Interfaces)

| Port | Location | Adapter | Notes |
|---|---|---|---|
| `DocStatusPort` | `ports/doc_status.py` | `adapters/doc_status/dynamodb.py` (`DynamoDBDocStatusStore`), `adapters/doc_status/sqlite.py` (`SQLiteDocStatusStore`), `FakeDocStatusStore` for tests | Persists incremental-indexing document status and each document's text units, partitioned by index namespace |
| `ContentGuardPort` | `ports/guardrail.py` (with the `GuardVerdict` record) | `adapters/guardrail/bedrock.py` (`BedrockGuardrail`, Bedrock `ApplyGuardrail`) | Screens the raw query and the final answer (§14) |
| `GraphIndexer` (write-side) | `ports/indexer.py` | `adapters/storage/neptune_indexer.py`; local: `adapters/storage/sqlite_graph_indexer.py` (on the `adapters/local/sqlite_graph.py` store) | Single contract for full + delta (`upsert_*`/`delete_by_id`) |
| `VectorIndexer` (write-side) | `ports/indexer.py` | `adapters/storage/opensearch_indexer.py` (on OpenSearch, or on `adapters/local/search_engine/` through `SearchEngineClient`) | Same |
| `BaseGraphRAGRetriever` (read-side) | `adapters/retrieval/base.py` | `adapters/retrievers/{neptune,opensearch,sqlite_graph}_retriever.py` (the graph ones on `graph_expansion.py`) | Retrieval adapters |
| LLM/Embedding/Rerank factories, token counter | `ports/model_factory.py` (`ModelFactoryPort`, `TokenCounterPort`, and the `LanguageModelInfo` / `EmbeddingModelInfo` / `RerankModelInfo` records `get_model_info` returns) | `adapters/aws/bedrock.py`, `adapters/aws/bedrock_models.py`, `adapters/aws/token_counter.py`; local: `adapters/local/{models,embeddings,rerank,token_counter}.py` | Carried by one `Providers` bundle (`adapters/providers.py`, built by `application/bootstrap.py`) that each orchestrator builds once and passes to every component; each role's adapter is the one `providers.<role>.kind` selects through `adapters/provider_registry.py` (default Bedrock; §15) |

> Design note: Pure ports (`DocStatusPort`, the write-side indexer ABCs) are gathered in `ports/`. The read-side abstract bases (`BaseGraphRAGRetriever`/`BaseSearchStrategy`) are "adapter bases" that construct infrastructure (HybridScorer/TokenManager) in `__init__`, so they live in `adapters/retrieval/base.py` and are imported from there; `ports/__init__` does not export them and only names them in its catalog docstring (no duplicate Protocol definition is kept). `BaseGraphRAGEvaluator` (`adapters/evaluators/base.py`) is an adapter base of the same kind.

### 2.2 Role-Based Retriever Injection

Search strategies are injected with retrievers **by abstract role, not by concrete backend name**.

- `RetrieverRole.GRAPH` → graph traversal/expansion (currently Neptune)
- `RetrieverRole.DOCUMENT` → vector/lexical lookup (currently OpenSearch)

Strategies access retrievers only via `self.graph_retriever` / `self.document_retriever` (base-class properties), and the role→adapter builder map in `rag_chain` binds the actual implementation. As a result, swapping the graph backend requires no changes to the strategy code.

```python
# domain/retrieval/strategy_registry.py
@register_strategy(SearchStrategy.LOCAL, required_roles=(RetrieverRole.DOCUMENT, RetrieverRole.GRAPH))
class LocalSearchStrategy(BaseSearchStrategy): ...
```

### 2.3 Registries

- **Search strategies**: `domain/retrieval/strategy_registry.py` — `@register_strategy(...)` registers a class, its required roles, and its query inputs against the `SearchStrategy` enum.
- **Evaluators**: `EvaluationManager._resolve_evaluator_class` — one explicit branch per `EvaluatorType` → evaluator class (lazy, import-on-use). Not a decorator registry: a new evaluator adds an enum member and a branch.
- **Renderers**: `adapters/renderers/base.py` — `@register_renderer("name")`.
- **Providers**: `adapters/provider_registry.py` — `@register_provider(ProviderRole.X, "kind")` registers the builder that `providers.<role>.kind` selects for each role (LLM, embedding, rerank, token counter, graph and vector indexer and retriever, doc-status registry).

This pattern follows the same philosophy as the existing `ParserFactory._loader_configs` (declarative parser registration).

> Write-path note: `IndexingManager` always writes to **two store roles**, a
> graph store and a vector/lexical store, rather than iterating a list of
> backends: both are constitutive of the framework, and the per-role fan-out
> (entities to both stores, entities-before-edges phasing, the orphan-edge
> cascade) is deliberate domain knowledge rather than arbitrary dispatch. Which
> backend fills each role is open: `providers.graph.kind` / `providers.vector.kind`
> select a registered builder (Neptune, Gremlin Server or the embedded SQLite
> graph; OpenSearch or the embedded SQLite vector store), or a
> `GraphIndexer` / `VectorIndexer` is injected
> (`IndexingManager(vector_indexer=…, graph_indexer=…)`). The read path does the
> same through the `RetrieverRole` → builder map.

### 2.4 Dependency Rule Verification Status

`domain/` imports no `ports`/`adapters`/`application` module at runtime, `ports/` and `shared/` import no other layer, and `adapters/` import no `application`, `evaluation` or `visualization` module; `evaluation/` reaches the application layer only through the chain (`application.retrieval`, `application.bootstrap`). `tests/unit/test_domain_purity.py` enforces these contracts on every import, including `TYPE_CHECKING` ones. Two compile-time-only exceptions exist, and the test checks that both stay under `TYPE_CHECKING`: `domain/retrieval/strategy_registry.py` references `adapters.retrieval.base.BaseSearchStrategy` (the registry stores strategy subclasses), and `domain/ingestion/delta_detector.py` annotates its parameters with `DocStatusPort` (the port module imports the domain models, so a runtime import would be a cycle). Extracting pure strategy/retriever ports would also remove the first type-level reference; it is left in place as a deliberate boundary (see "Deliberate Design Boundaries" at the end of this document).

Code imports from real layer locations (`application.retrieval.rag_chain`, `application.storage.indexing_manager`, `application.ingestion.pipeline`, and the `adapters.*` / `domain.*` modules). `evaluation/` and `visualization/` are real logic packages (see the layout note in §2).

---

## 3. Domain Model

The `domain/models/` package contains pure Pydantic models with no infrastructure dependencies.

- `Entity` (`name`, `description`, `type`, `text_unit_ids`, `community_ids`, `rank`, `confidence`, embedding fields)
- `Relationship` (`source_id`/`target_id`, `description`, `weight`, `text_unit_ids`, `description_embedding`)
- `Community` / `CommunityReport`, `TextUnit`, `Covariate` (claim)
- `DocStatus` (PENDING write-ahead record → PROCESSED|FAILED), `DocStatusRecord` (content hash + the document's text units + suffix), `DocumentDelta` (new/changed/unchanged/deleted), `DocumentLineage` (the text units one document produced)
- `SearchQuery`/`SearchResult`/`RetrievalResult`, `SearchStrategy`/`SearchType`/`RetrieverRole`

**Rank (MS GraphRAG parity)**: an entity's `rank` is its degree (the number of distinct entities it is related to) and a relationship's `rank` the sum of its endpoints' degrees (`domain/ingestion/graph_rank.py`). Graph resolution sets them for a full build; an incremental merge recomputes them for every entity and relationship it writes, over the stored graph's neighbours (`GraphIndexer.read_neighbor_ids`) plus the delta's relationships. A stored entity the delta does not write keeps its rank until the next full build, including when a removed document dropped one of its edges.

**Lineage is the core data.** Entities/relationships record the `text_unit_ids` they appeared in at extraction time, and this lineage, rather than a token-overlap heuristic, decides "is this entity related to this text unit?" accurately and language-independently.

**Index suffix**: every OpenSearch index and Neptune label is named `<prefix>-<suffix>[-<additional_suffix>]`. The suffix is written at ingestion from `processing.document_parsing.index_value` and selected at query time by `RAGInput.suffix` (`--suffix` on `run-rag`/`run-eval`); both default to `default`, and `indexing.additional_suffix` appends the same second segment on both sides. This document calls the resulting value the *index suffix*; one per tenant or corpus version is the intended use.

**Entity names, IDs, and multilingual support**: `Entity.name` (and relationship `source_name`/`target_name`) keeps the *display form* the source used, only trimmed and whitespace-collapsed (`clean_display_name`), so retrieval context and community reports can quote "$1,000 penalty", "Section 4.2", or "C++" verbatim. Identity is a separate key: entity/relationship IDs are hashes of `entity_key(name)` (`shared/utils/common.py`), which applies NFKC + casefold, treats `_`/`-` as spaces, drops quote marks and commas, collapses whitespace, and strips trailing sentence punctuation, but **keeps every other symbol and the letters/digits of all scripts**. Case, whitespace, and quote variants of one name ("ACME  Corp." / "Acme Corp") therefore share an ID, while "C++" and "C#" stay distinct, and Korean, CJK, and accented names get unique IDs. A non-empty name never yields an empty key. Exact-name matching (extraction endpoint lookup, gleaner merges, claim resolution, incremental `merge_entities`) and the fuzzy matcher's shingles use the same key; the punctuation-stripping `normalize_name` is kept only for token-level similarity. **Changing the key changes every ID, so a graph indexed under a different key must be fully re-indexed** (incremental indexing would otherwise add new-ID duplicates next to the old entities).

---

## 4. Ingestion Pipeline

The `DataIngestionPipeline` in `application/ingestion/pipeline.py` runs 12 stages in order (`application/ingestion/pipeline_stages.py`).

![Ingestion pipeline](../assets/ingestion-pipeline.drawio.png)

| # | Stage | Module | Notes |
|---|---|---|---|
| 1 | Document parsing | `parser.py` (`ParserFactory`) | PDF/TXT/CSV/JSON out of the box (+MD/HTML via the optional `unstructured` extra) |
| 2 | Document loading | `loader.py` (`DirectoryLoader`) | MinHash deduplication |
| 3 | Chunking | `chunker.py` (`ChunkerFactory`) | simple / intelligent (LLM semantic) |
| 4 | Translation (optional) | `translator.py` | multilingual → target language |
| 5 | Graph extraction | `graph_extractor.py` | LLM entity/relationship extraction |
| 6 | Gleaning (optional) | `gleaner.py` | iterative refinement: up to `max_rounds` per text unit, re-gleaning only units whose last answer added something new |
| 7 | Graph resolution | `graph_resolver.py` + `description_summarizer.py` | name-variant merge (fuzzy matching opt-in), `text_unit_ids` union, **LLM re-summarization of merged descriptions** |
| 8 | Claim extraction (optional) | `claim_extractor.py` | factual assertions (covariate) |
| 9 | Claim resolution (optional) | `claim_resolver.py` | |
| 10 | Graph analysis | `graph_analyzer.py` | graph build and statistics (the visualization computes centrality with the same module) |
| 11 | Community detection | `community_detector.py` | hierarchical Leiden, community report generation (degree-sort + token-budget pack) |
| 12 | Indexing | `application/storage/indexing_manager.py` | OpenSearch + Neptune; a full build replaces every index type of the run's suffixes, emptying one the run has nothing of |

> The stage order is single-sourced in `DataIngestionPipeline.STAGE_CLASSES` (`application/ingestion/pipeline.py`). Every stage gets its models and stores from the run's provider bundle (`Providers`); **graph resolution (7) calls the LLM too**, because it re-summarizes merged descriptions.

**Per-unit extraction records and the extraction store (stages 5-8)**: Extraction and gleaning write one `TextUnitExtraction` per text unit: the entities and relationships that unit states, each with the unit's own name, type, description, strength, confidence and attributes, plus the entity keys rejected for it (`domain/models/extraction.py`). Gleaning refines one record at a time: its prompt lists the unit's own items, and its answer's additions and corrections edit that record only. A rename gives the entity the id of its new name and moves the unit's edges along; a retype or reversal gives the edge the id of its corrected (source, target, type); items that become equal within the unit merge. Graph resolution starts from the merge of all records (`domain/ingestion/contributions.merge_contributions`: items merged by id, weights kept per text unit, endpoints cited), so the graph a full build merges does not depend on which units shared a batch, and a correction never reaches what another unit says (MS GraphRAG and LightRAG likewise glean each chunk's own extraction before the merge). The claim prompt lists the unit's own entities, so a unit's claims depend on the unit alone. Each record is also stored, without what ties it to its unit, in the vector store's extraction-records index (`indexing.opensearch.extraction_records_index_prefix`, one per suffix), under `<fingerprint>:<SHA-256 of the processed text>`; the fingerprint (`cache_keys.unit_record_fingerprint`) covers the models and LLM knobs, the extraction, gleaning (for a gleaned record) or claim settings, the target language and the prompt templates in effect, built-in ones included. A text unit whose text and settings have a stored record takes it instead of an LLM call (text-unit ids hash the document content and the chunk index, hence the text key), so an edited document re-extracts only its changed chunks, and identical chunks in other documents or a full rebuild reuse theirs. Records are written once a unit's extraction (and gleaning) succeeded; the index is only added to (a full build, `indexing.reset` and document deletion leave it), a store that cannot be read or written is logged and the run re-extracts, and `--force-rebuild` (`PipelineConfig.force_rebuild`) re-extracts every unit and refreshes the stored records. Incremental runs derive the content of what they touch from these records (see *Derived content on incremental runs* in §5).

**Relationship identity (deliberate divergence, with one known limitation)**: a relationship id hashes `relationship:{entity_key(source)}:{entity_key(target)}:{type}`, lowercased (`BaseProcessor._generate_relationship_id`). The free-text, model-written `type` is part of the identity, and the edge is directed: MS GraphRAG and LightRAG key an edge by its entity pair alone, so two types between one pair are one edge there and two here (the community graph still joins them on one undirected edge, see *Community graph* below). Type variants that differ only in case or surrounding whitespace share an id, but "works_for" and "works for" do not. The parts are joined with `:` without escaping, so a name or type containing `:` can collide with another split of the same string (source "A", target "B", type "x:y" and source "A", target "B:x", type "y" get one id). Such a collision needs both a `:` in a name or type and a matching neighbour, and fixing it would re-key every stored relationship, so the format is kept.

**Merged-description re-summarization (stage 7)**: Graph resolution merges descriptions of the same entity/relationship by simple concatenation, so the description of a popular entity that appears in many chunks grows without bound. `DescriptionSummarizer` (run in `GraphResolutionStage`) re-summarizes only descriptions that exceed a token budget into a single coherent description using a cheap LLM (parity with MS GraphRAG `summarize_descriptions` / LightRAG `_handle_entity_relation_summary`, controlled by `DescriptionSummarizationConfig`). The goal is to prevent embedding/prompt bloat. A description longer than one call's input budget (at most `description_summarizer.MAX_INPUT_TOKENS`, 12,000 tokens counted with the summary model's local token estimate; a summary model with a smaller context window gets half of what its window leaves beside `max_summary_tokens`, at least 1,000) is summarized map-reduce, as LightRAG does: its lines are packed into groups within the budget, each group is summarized, and the joined group summaries are summarized again until they fit one call. A description within the budget gets the same single call as before; if any call for a description fails, its concatenation is kept.

**Relationship endpoints cite the relationship's text units (stages 5-7; deliberate divergence)**: In MS GraphRAG and LightRAG an entity cites only the chunks that list it (MS GraphRAG drops a relationship whose endpoint no chunk lists; LightRAG adds such an endpoint citing the relationship's chunks but does not extend an entity that already exists), so an entity's text units depend on which chunks were extracted together. Here every relationship endpoint also cites the text units of each relationship it is an endpoint of, and an endpoint no chunk listed becomes a stub entity citing them (`domain/ingestion/relationship_endpoints.cite_relationship_endpoints`). The merge of the text units' records at the start of graph resolution applies it (so an edge, extracted or gleaned, naming an entity no unit lists adds a stub rather than vanishing), each unit's gleaning prompt lists its endpoints the same way, and the incremental lineage (`build_document_lineage`) attributes endpoints to their relationships' documents on the same rule. The reason is incremental convergence: an entity's text units depend only on the chunks that name it, so an incremental run cites what a full build cites, and removing a document strips an entity to exactly the text units of the surviving documents that name it instead of deleting it under a surviving edge or leaving it with none. The cost is that local search and community lineage also reach chunks that mention an entity only through a relationship. With `entity_grounding.action: penalize`, an ungrounded relationship naming an entity its chunk did not extract is dropped rather than down-weighted, since keeping it would make that entity cite a chunk with no evidence for it. With grounding on, gleaning likewise drops a gleaned relationship naming an entity rejected as ungrounded for its chunk (by extraction, which records its rejections in the unit's record, or by gleaning itself), and one naming an entity its chunk does not list unless the chunk's text contains that name; otherwise the stub step would bring the rejected or unsupported entity back. Gleaning's own additions follow `entity_grounding.action` as extraction's do: under `drop` an addition whose `text_evidence` is not in the chunk is rejected; under `penalize` an ungrounded gleaned entity is kept at its confidence times `penalty_factor` (then subject to `entity_confidence_threshold`), and an ungrounded gleaned relationship at its weight times `penalty_factor` only when the chunk lists both endpoints. An ungrounded correction is ignored under either action.

**Entity merging (stage 7 and incremental merge)**: By default two names merge only when their `compact_name_key` (below) is equal. `processing.fuzzy_entity_resolution` additionally links names whose `resolution_method` similarity reaches `processing.similarity_threshold` (default 0.85). It is off by default because name similarity also joins different organizations that share most of their name ("Northfield University" / "Northfield University Hospital", "Federal Bank" / "Federal Bank Board"), while genuine spelling variants mostly differ in case, spacing, punctuation or legal form, which the compact key already covers. A merged group takes the name cited by the most text units (ties: the smallest name) as its name and id, so the result does not depend on input order; relationships re-pointed at it take that name as their `source_name`/`target_name` too (`relationship_endpoints.sync_relationship_endpoint_names`, shared with gleaning). A fuzzy-matching worker that fails fails the stage unless `processing.ignore_errors` is on (then the name is matched exactly and the failure logged). The incremental merge matches delta names against the stored names with the same matcher and settings (one entity-name projection per delta run), so a full build and an incremental build group the same names.

**Fuzzy-merge guards**: Character-shingle similarity cannot distinguish identifiers that share a long prefix ("purchase order 1001" vs "purchase order 1002" score ~0.96 under MinHash; "vendor a" vs "vendor b" ~0.73). Two names are therefore fuzzy-linked only when their *discriminator tokens* are identical (`base_resolver.discriminator_tokens`: tokens containing a digit, single ASCII letters/digits, and multi-letter Roman numerals), and only when their entity types are compatible (same normalized type, or one side empty/`unknown`). For names ending in Han, Hangul, or Kana, the final character is usually the head noun, so such names are also linked only when their final characters are equal (`base_resolver.head_character`): "가나다연구원" (institute) and "가나다연구소" (lab) share every shingle but the head. Conversely, names whose `compact_name_key` is equal are linked with score 1.0: it strips a short, fixed list of legal-form designators at the start or end of the name (`(주)`, `주식회사`, `(유)`, `유한회사`, `(株)`, `株式会社`, `(有)`, `有限会社`, `有限公司`, and a trailing `Inc`/`Ltd`/`LLC`/`Co`/`Corp`) and drops spaces next to Han, Hangul, or Kana letters, so "(주)가나다", "가나다 상사"/"가나다상사", and "Acme Inc"/"Acme" merge. Designators are also removed before discriminator tokens are computed, so the "주" of "(주)" is not an identifier. These are matching rules only; entity ids still hash `entity_key`. `FuzzyMatcher.find_all_matches` applies the guards, so the full-build resolver and the incremental `merge_entities` path share them. The full-build resolver builds groups with a type-aware union-find (strongest links first, deterministic tie-break) that refuses any union which would place two differently typed names in one group, so an untyped name cannot bridge an `organization` and a `person`. Entities sharing an exact surface name always stay together. Relationship resolution groups by exact `(source_id, target_id, type)` after the entity remap and needs no separate guard.

**Community graph (stages 10-11)**: `GraphBuilder` builds an undirected graph with one edge per entity pair. Relationships of other types or in the other direction between the same pair share that edge: its `weight` is the sum of their weights (so Leiden sees the pair's total strength whatever the input order) and `relationships` lists each one by id. A community's `relationship_ids` and its report context therefore keep every relationship, each written in its own direction.

**Community report context pack (stage 11)**: The report-generation input **sorts entities within a community by graph degree in descending order** (ties broken by stable id sort), caps them at `max_entities_per_report`, and packs them to fit the `max_report_context_tokens` token budget (relationships are sorted/packed identically by the sum of both endpoints' degree, with weight as the tiebreak). The top-degree entity is always included (at least one) even if it alone exceeds the budget, so a report never ends up with empty context (`community_detector._prepare_report_input`). A community whose context left members out and has no sub-community report to stand in for them is logged with the number of entities kept.

**Community hierarchy (deliberate divergence from MS GraphRAG)**: MS GraphRAG runs graspologic's `hierarchical_leiden` top-down on the largest connected component and re-splits every cluster larger than `max_cluster_size`. Here level 0 is one Leiden partition of the whole entity graph, and each coarser level partitions the graph of the previous level's communities (edge weight = summed weight between them), up to `max_levels`, stopping when a level would merge nothing or everything into one community. `resolution` sets the level-0 granularity (higher gives more, smaller communities), and `max_cluster_size` (default 20) bounds it: modularity has a resolution limit, so on a large graph one pass merges well-separated groups into level-0 communities that grow with the corpus (about 680 entities at 50,000 on a synthetic graph), larger than one report's `max_report_context_tokens` can describe. Any level-0 community above `max_cluster_size` is partitioned again on its own subgraph, recursively, until every piece fits or Leiden leaves it whole; the coarser levels then regroup the pieces, so the split adds levels rather than losing structure. The same bound caps a parent's children: a cluster-graph community of more than `max_cluster_size` child communities is split the same way, so a parent's roll-up (child report summaries packed into `max_report_context_tokens`) covers every child (on the synthetic 50,000-entity graph the largest parent had 172 children, 20 after). MS GraphRAG's default is 10 with an 8,000-token report budget; 20 typically keeps a leaf's entities and relationships within the default 4,000 tokens. Level-0 communities below `min_community_size` entities merge into the neighbour community they share the most edge weight with, unless that would take it above `max_cluster_size` (coarser levels are neither split nor merged: their nodes are whole communities); an entity with no relationships has no neighbour, so it stays a one-entity community with its own report, where MS GraphRAG's largest-component step leaves it out of every community. A Leiden failure at any level fails the stage (`GraphError`) instead of truncating the hierarchy.

**Sub-community roll-up (MS GraphRAG parity)**: When a coarse parent community's raw entity/relationship context would overflow `max_report_context_tokens`, or it has more entities than `max_entities_per_report`, rather than simply truncating, the report generator substitutes summaries of that community's already-generated child sub-community reports for the lowest-priority raw context. This requires bottom-up per-level generation: `generate_reports` groups communities by `level` and processes them finest-first (level 0 = leaf), accumulating each level's reports so a parent (`enable_sub_community_rollup`, default on) can fold in its children's summaries (`_generate_reports_with_rollup` / `_build_sub_community_context`, packed biggest-child-first within the same budget). Setting the flag false restores the flat, truncate-on-overflow path (`_generate_reports_flat`). Detection never creates a parent identical to its only child: a cluster Leiden leaves on its own at some level (often an isolated one) keeps its level and joins the first coarser community that merges it, or stays a root, so each distinct community is reported, embedded and retrieved once. Global and DRIFT search retrieve reports by similarity, not by level, so a root below the top level is still found.

**Structured community reports (MS GraphRAG parity)**: The report prompt emits a structured result — an executive `summary`, an importance `rating` (0-10) with a one-sentence `rating_explanation`, and a list of `findings`, each a one-line `summary` plus a multi-sentence `explanation` (`CommunityReport.findings`/`rating`, `CommunityFinding`). `content_length` sets how many findings it asks for (short 5-7, medium 7-10, long 10-15; `prompts.graph_extraction.REPORT_FINDINGS`, which also sizes the output-token floor). The free-text `full_content` used for embeddings, global-search map-reduce, and display is **rendered deterministically** from these structured fields (`CommunityReport.render_full_content`), so the structure adds no second LLM call and the embedding/search path is unchanged. The `rating` is also indexed on the community-reports OpenSearch document for importance-aware ranking.

**Community report lineage (stage 11)**: Each report carries `text_unit_ids` and `document_ids`, indexed as keyword fields on the community-reports document (`CommunityDetector._attach_report_lineage`). These are **community membership provenance**: the union of the member entities' `text_unit_ids`, and the documents those units came from. They are not report-input provenance — `max_entities_per_report` and the token budget may keep a member out of the prompt while its sources stay in the lineage — and they do not establish sentence-level citation support. They exist so a consumer can filter reports by source document or re-summarize a community per reader. Each report also carries its place in the hierarchy (`level`, `depth` from the root, `parent_community_id`, `child_community_ids`, and the derived `max_cut_depth`), attached when the reports are written (`IndexingManager`, `domain/ingestion/community_hierarchy.py`); global search selects reports by it (§6.1.1).

**Pipeline infrastructure**: Stage-checkpoint-based resumption (`shared/pipeline_manager.py`), S3 cache sync (`adapters/aws/s3_cache.py`: the changed files are uploaded after every stage, stage data before the index and run metadata that refer to it; on SIGTERM `run-ingestion` stops the running stage through `DataIngestionPipeline.request_stop` (deferred while a checkpoint is being written; the handler does no I/O, since it may run while the main thread holds a lock the upload needs) and uploads the checkpoint as the run unwinds; S3-synced checkpoints carry no TTL, the bucket lifecycle expires them), an LLM response cache (`adapters/llm/response_cache.py`: a LangChain `BaseCache` that `CachingLLMFactory` attaches to every `ModelPurpose.INGESTION` model, keyed by the rendered messages and LangChain's model/generation-settings string, stored as segment files under the run's `llm-cache/` and synced with the stage cache; each entry of an earlier run is served at most once per run and truncated responses are not kept, so a retry after a bad response asks the model again; a batch abandoned by a stop makes no further retry), a `continue_on_error` toggle, per-stage caching (`shared/cache_manager.py`). The translation stage is skipped at no cost when `TranslationConfig.is_noop` (source == target & no additional languages). LLM output parsing consistently uses a `FixingConfig`-based output-fixing parser. The pipeline releases indexer/client resources via `close()`, which the `run-ingestion` CLI calls in `finally` (§8.6).

**Generalization in practice**:
- The relevance gate (which entities to include in the prompt for claim/gleaning) is decided by `text_unit_ids` lineage membership rather than a token-Jaccard regex heuristic → accurate and language-independent. A gleaning correction (`ENTITY_CORRECTION`/`RELATIONSHIP_CORRECTION`) edits the unit's own record (see *Per-unit extraction records*), so a unit never rewrites what another unit says.
- Gleaning stops per text unit on observed output, not on scores: a unit is re-sent (up to `max_rounds`) only while its previous answer added an entity or relationship its record did not have. An empty answer is the model saying nothing more is missing, the signal MS GraphRAG asks for with a separate Y/N loop prompt, so no extra call is spent on it.

---

## 5. Incremental Indexing

When documents are added/changed/deleted, only the delta is processed instead of a full re-index.

![Incremental indexing](../assets/incremental-indexing.drawio.png)

The names are `IncrementalIndexer` methods (`application/ingestion/incremental.py`) unless noted. A document recorded `FAILED` is classified as changed on the next run until it reaches `indexing.max_document_failures`; final registry records are written only when the delta's writes pass the indexing failure gate, and until then the delta's documents keep their `PENDING` write-ahead records (step 2).

1. **Delta detection** (`domain/ingestion/delta_detector.py`): Builds `{doc_id: content_hash}` from a stable `doc_id` + content SHA-256 hash, and `DocStatusPort.diff(incoming, scope)` classifies them as new/changed/unchanged/deleted. `doc_id` is `<namespace>#<hash>`: the index namespace (index suffix = `index_value` + `indexing.additional_suffix`, tagged with non-default store prefixes) and a hash of the namespace, the corpus source scope and the path relative to the corpus root, so two tenants' identically named files staged in the same directory, and two corpora on one suffix that share a relative path, stay distinct. The namespace is the registry partition (the DynamoDB partition key, a SQLite key column): a run reads only its namespace's records (one projected DynamoDB `Query`, not a table `Scan`), so its cost follows the namespace's size, not the table's. A corpus whose scope changed (a moved local directory) is never adopted from: the registry cannot tell a move from another host's corpus on the same path, so the old scope's records stay until a run lists it in `indexing.retire_source_scopes` (`--retire-source-scope`), which classifies every record of that scope in the run's namespace as deleted (one more projected read of the namespace per retired scope; the run's own scope is rejected). A local default-scope run with new documents warns about the other local source directories of its namespace, from the scopes the diff already returns (`DocumentDelta.stored_scopes`). Deletion is **scoped**: only registry records of the run's scope (index namespace + corpus source, `document_parsing.source_scope` or the resolved source directory) are deletion candidates. Files that failed to parse or load this run are reported as `failed` and excluded from `deleted`. A changed document whose stored `PENDING` record shows that runs started indexing this same content `indexing.max_document_attempts` times in a row (default 3) and never finished it is classified `quarantined` instead (*Crash recovery* below).
2. **Write-ahead** (`IncrementalIndexer.write_ahead`): Before any store write, every new and changed document is recorded `PENDING` under its current key and the run's scope, with the content hash `"pending"` (`PENDING_CONTENT_HASH`, which no document hashes to) and the union of its stored text units and the text units this run is about to write. The records go out in one batched write (`DocStatusPort.put_many`, DynamoDB `BatchWriteItem`), after one batched read of the stored records (`get_many`), which the commit reuses for failure counting. A record over the registry's item limit (`DocStatusPort.record_fits`; DynamoDB 400 KB, roughly 10,000 text units, half that for an edited document, whose `PENDING` record holds its old and new text units) fails the run here with `DataProcessingError`, before anything is written: split the document. The `PENDING` record counts the attempt (`attempt_count`: one more than the stored `PENDING` record's when an earlier run started the same content, `attempted_hash`, and died before committing it, else 1); a commit, `PROCESSED` or `FAILED`, starts it over at 0. See *Crash recovery* below.
3. **Stale cleanup** (`IncrementalIndexer.remove_changed_and_deleted`): Before the delta is written, removes what changed documents (so entities that disappear after re-extraction do not linger in the graph) and deleted documents (step 5) contributed and no *surviving* text unit still states. The registry lists each document's text units only; the stores are asked what cites them (`IndexingManager.remove_text_units`, step 5). The text units of changed and deleted documents are removed in one pass per suffix, so an artifact only a changed and a deleted document state goes too; removing the two sets separately would treat each as a survivor of the other and leave it behind with no text units.
4. **Delta upsert** (`IncrementalIndexer.commit`): With `indexing.cross_run_merge` (default on), `IndexingManager.merge_with_existing_graph` first reads back the existing entities/relationships the delta touches, per index suffix, and merges the delta into them (merge semantics below), so an entity shared with unchanged documents keeps their `text_unit_ids`; the content of what is written is then derived from the records (see *Derived content on incremental runs* below). The stored entities a delta name can merge into are found by name key: without `processing.fuzzy_entity_resolution` names link by their compact name key only, which every entity vertex stores (`name_key`), so `GraphIndexer.find_entity_ids_by_name_key` reads just those entities; with fuzzy matching on, or a graph store that cannot look keys up, every stored name is read (`read_entity_names`). The stored edges of the delta's entities are read too, so a delta edge meets a stored edge of another id between the same entities and of the same type (extracted with another spelling of an endpoint) and merges with it, as a full build does. A delta item that merged into a stored one under another id (a name-variant or fuzzy entity match, or an edge whose endpoint was remapped) is written under the kept id (the stored one, or the id the derivation re-keyed it to), and the delta's community members and claim subjects/objects are pointed at it too. Delta claims are then merged with the stored claims of the same id, read back from the OpenSearch claims index (`merge_with_stored_claims`), together with the stored claims about a re-keyed entity. `IndexingManager.index_delta` then writes the result as is: Neptune uses a Gremlin `coalesce(unfold, addV)` idempotent upsert; OpenSearch upserts by id into the live alias index. The relationship vector index is updated the same way. Once the writes succeeded, the old ids of re-keyed items are deleted (`delete_retired_ids`); after a failed write they are kept.
5. **Deletion propagation** (in step 3's removal, `IndexingManager.remove_text_units`): Finds what cites the removed text units in the stores: the entities and communities in the graph (`GraphIndexer.find_ids_citing`, a `text_unit_ids` lookup), the relationships among those entities' incident edges, and the entities, relationships, claims and community reports in the vector store (`VectorIndexer.find_documents` on the keyword `text_unit_ids` field; a copy one store holds and the other lost to an interruption is found either way). An entity, relationship or claim citing only removed text units is deleted with the text units (`delete_by_id`, both stores); one citing others too is kept, stripped of them, with weight recomputed, in Neptune and OpenSearch alike, and its content derived again from the text units left (`IndexingManager.remove_text_units_from_shared`, which rewrites every shared item it reads back from Neptune, so a retry also repairs an OpenSearch copy an interrupted or failed strip did not reach; OpenSearch updates `text_unit_ids`, `weight` and `rank` in place, `VectorIndexer.update_entity_lineage`/`update_relationship_lineage`, when nothing it embeds changed). The strip writes a re-keyed entity's new vertex first, then the edges, then the other entities: a retry finds the edges through the entities that still cite a removed text unit, so those change last. A community or report is deleted once none of the text units it cites is stored. Text units belong to one registry record each (*Document identity* below), so what another document still states is never removed, whatever its namespace. Changed documents get the same treatment before their re-extraction is merged back in. If any removal fails, the deleted documents' registry records are kept for a retry and the indexing stage fails, so the run (and the `IndexingFailures` alarm) reports it: before the commit when the delta has changed documents (committing would replace their records and lose the text units to remove), after it otherwise.

   **Limitation**: shared communities and their reports keep what a changed or deleted document contributed until a full rebuild (`indexing.reset`; see *Communities on delta runs* below), and keep listing an entity's old id after it was re-keyed. Descriptions, names, ids, types, document `filters`, ranks and claims are derived again (below).
6. **Registry update** (commit): Replaces the `PENDING` records, in one batched write, with `DocStatusRecord`s carrying each document's text units (`DocumentLineage`), suffix, the run's scope and the relative path. A document that translation, graph extraction, gleaning or claim extraction failed on for any of its text units is recorded `FAILED` (with its text units), and `diff` classifies a `FAILED` record as changed even when its hash is unchanged, so the next run prunes and re-extracts it. A document owning an artifact whose write failed (through the text units the artifact cites) is recorded `FAILED` the same way. The record counts consecutive failures of the same content (`failure_count`); after `indexing.max_document_failures` of them `detect_delta` classifies the document unchanged instead, so a deterministic failure is not re-extracted on every run. Once recorded, the extraction records that only removed text units named and no stored text unit names are deleted (`IndexingManager.collect_extraction_records`; after the writes, because a changed document's unchanged chunk names the record its removed copy did). This is best-effort: a run that dies between the removal and the commit leaves those records behind, which costs space only.

`indexing.reset` skips the diff and rebuilds the whole corpus with the full-index path without taking the suffix offline: the stores are not cleared first. Every document is recorded `PENDING` (step 2); the vector indices are written into new timestamped indices whose aliases are swapped once each is built, as every full build does; the graph is rebuilt in place (`GraphIndexer.begin_rebuild`): the full-build writes clear no label and tag every vertex and edge they write with the build (`build_tag`), and `end_rebuild` then drops, in batches, the edges leaving the suffix's entities and the entity and community vertices that lack the tag. Queries keep reading the previous graph while the rebuild writes (a mix of old and new items until the drop). Every document is then recorded again (only when the rebuild's writes pass the same failure gate as a delta commit), and only after that are the namespace's other records, the documents the rebuild replaced, deleted (its partition, `DocStatusPort.list_namespace`; other namespaces sharing the table keep theirs). A failed rebuild drops nothing and keeps those records, so the retry still finds what they wrote. A full build with the registry off rebuilds the graph the same way.

A generation switch (writing the rebuild under new labels and moving a pointer) was rejected: Neptune vertex labels cannot be renamed, so every retriever (Neptune and the embedded SQLite graph) would have to resolve the current generation's label on each query, and the graph would be stored twice during the rebuild. Tag and sweep keeps the read path unchanged and never leaves the graph empty, at the cost of the transient mix above.

**Crash recovery (write-ahead protocol)**: the registry is not updated atomically with the stores, so a run can stop between any two writes (a killed task, a store outage, a failed gate). The protocol keeps one invariant: before the stores hold anything the registry cannot account for, the registry already lists it. Write-ahead records (step 2) precede the removal and the upsert, the deleted documents' rows are deleted only after their artifacts are, and the commit replaces a `PENDING` record only after the document's writes landed. A run that stops anywhere after step 2 leaves each unfinished document `PENDING`, and the next run repairs it:

- `PENDING_CONTENT_HASH` never equals a content hash, so a document that still exists is classified changed, even when its content went back to the version indexed before the interrupted run (without the write-ahead record the registry still held that version's hash, so the document read as unchanged while its artifacts had been pruned).
- A `PENDING` document that is gone is classified deleted (its record keeps the scope), and its text units cover everything the interrupted run may have written (every artifact cites the text units it was stated in), so nothing it added is left unfound.
- The removal takes a document of the current run's text units from the record stored before its own `PENDING` record (after a crash, the crashed run's `PENDING` record), so a normal run removes exactly what the previous run wrote, and a recovery run removes the interrupted run's partial writes before writing again. A new document's `PENDING` record removes nothing.
- A `PENDING` record has `failure_count` 0 and is never `FAILED`, so an interruption is not counted as a failure toward `indexing.max_document_failures`, but it resets the count: the commit of the same run counts a real failure from the record stored before the write-ahead one, while the next run counts from the `PENDING` record, so a document that had failed before is retried until it fails `indexing.max_document_failures` runs again. (A document already at the limit is not re-extracted and gets no `PENDING` record.)
- The old ids of re-keyed items (*Derived content* below) are deleted only after the items are written under their new ids, and kept after a failed write: an interruption leaves at worst both copies. The retry re-keys the old copy again, finds the new id held by the new copy and derives the two together: copies of one item (and stored items a full build merges) become one, and an id another item holds is never written over.
- Documents of the delta that produced no text units stay `PENDING` (logged), and are re-extracted by the next run.
- A document whose indexing kills the run (out of memory, a task timeout) would stop every later run the same way: its `PENDING` record's `attempt_count` grows by one per run that started its content, and once it reaches `indexing.max_document_attempts` the next run classifies it `quarantined` (`DocumentDelta.quarantined`). It is not extracted; the removal pass (step 3) removes what the attempts wrote, and its record becomes `QUARANTINED`: the content hash it was attempted with, no text units, the reason in `error_info`. A `QUARANTINED` record reads unchanged while the content stays, so it is skipped until the file or the extraction settings change (or the limit is raised). The run reports it in the `total_documents_quarantined` metric (the CDK `DocumentsQuarantined` alarm). The culprit is not isolated: every document of a delta that keeps dying counts an attempt and is quarantined with it, and a document that produces no text units (it stays `PENDING`) is quarantined after as many runs.

The cost is one extra batched registry write per run, one batched read in place of one `GetItem` per document at commit, and, for the removal, a few batched store lookups by text unit id per suffix. `tests/unit/test_incremental_write_ahead.py` and `tests/property/test_incremental_interruption_properties.py` interrupt a run at every write step and check that the next run converges to a fresh full build of the same corpus, of the corpus with the changed documents reverted, and of the corpus without the added documents.

**Document identity** (`shared/utils/document_identity.py`): a document version's `document_id` (from which text-unit ids derive) hashes the path relative to the corpus root plus the full text. The parsing and loading stages re-derive it against the root (`delta_detector.assign_document_identity`), so the same corpus yields the same ids wherever it is checked out or synced, and same-named files in different folders never collide. An incremental run then keys it by the registry `doc_id` (step 1) instead of the relative path (`assign_registry_source`), so a text unit belongs to exactly one registry record, even when two corpora of one namespace hold the same file; text-unit ids therefore depend on the source scope too. Changing either rule changes ids and requires a re-index.

**Derived content on incremental runs** (`domain/ingestion/derivation.py`): the merge and the strip keep each touched item's structure (id, text units, per-unit weights); its content is then derived from the records of the text units it cites now (a stored text unit names its records in `attributes.extraction_records`), by the full build's own code: the records are merged and resolved (`merge_contributions`, `GraphResolver`, and for claims `ClaimResolver`) and each touched item takes the name, type, description, confidence and attributes of the resolved item that shares its name keys (resolved entities record the identity keys of the names they were resolved from as `attributes.entity_keys`); a claim takes its type, status, description and source text. An item cites every text unit any of its names is stated in, so its own text units' records hold all of its contributions and the result is what a full build over the current corpus gives it: an edited or deleted document's text leaves shared descriptions and claims, and a name or type follows the current votes. A description over the summarization budget is summarized again only when the description it is derived from changed (`attributes.summarized_from`). Ranks are recomputed over the stored graph, and the stored edges of the touched entities are rewritten when the endpoint, endpoint name or rank they carry now differs. Ids follow the current name: an entity whose derived name has another id than its stored one (a removed document held the spelling that won, or new documents outvote it) takes the full build's id (`Derived.id_remap`); its stored edges are re-pointed at it, a derived edge or claim takes the id the full build gives it (claims that become one fact are merged), the stored claims about it are derived again, and the old ids are deleted from both stores after the new ones are written (kept after a failed write). Stored communities and reports and the text units' attributes keep listing the old entity id until a rebuild, and the old vertex's `MemberOf` edges go with it. An item citing a text unit without a stored record (indexed before records existed, or whose record write failed) keeps its stored content, logged as a WARNING, until a rebuild.

**Merge semantics** (`domain/ingestion/merge/merger.py`, ported from MS GraphRAG `update/*`): Entities merge by id, else by identity key (`entity_key`; description lines unioned without duplicates, union of `text_unit_ids` and `community_ids`, max confidence/rank, attributes unioned with the delta winning except `filters`, whose values are unioned per key, several values as a sorted list (`base_processor.merge_attributes`, the function graph resolution uses, so N items resolved at once and the same items merged two at a time agree; extraction and gleaning union filters the same way), existing ids preserved + a remap for entities that merged under a different id); relationships merge by id, else by (source, target, type) (same description and attribute rules; weight = sum of the extracted strengths per supporting text unit, kept on the edge as `attributes.text_unit_weights` so a delta replaces its text units' entries and the result equals a full build over the union — the same rule the full-build extractor and resolver apply, after MS GraphRAG's summed instance strength); after the relationship merge, an edge's `source_name`/`target_name` are re-derived from the merged entities (`sync_relationship_endpoint_names`), so a delta edge whose endpoint merged into a stored entity under another name names the stored entity, as a full build does after graph resolution; claims merge by id with the full-build `ClaimResolver` rules (`merge_claims`: descriptions and source texts unioned, stored first, sorted union of `text_unit_ids`, the stored status when it has one, which is `ClaimResolver`'s most-common rule for one stored and one delta claim but can differ from a full build over three or more documents, since the stored claim keeps no per-status counts; see [User Guide: Cross-run merge](./user-guide.md#cross-run-merge) for claims indexed before they stored their text units); communities are appended under content-derived ids (see *Communities on delta runs* below). Ids derive from names (an entity's from its identity key, an edge's from its endpoints' keys and its type), gleaning corrections included (see *Per-unit extraction records*), so a delta folds into the stored item of the same key. The merge keeps the stored name when the delta spells the same key otherwise or is a name variant the matcher joined: every edge carries a copy of its endpoints' names, and adopting the delta's would leave the stored edges the delta does not rewrite naming the old one. It keeps the stored entity's type too (a delta's type only fills a missing one), while a full build takes the type of the group's best-supported mention after merging a name's mentions by their most frequent type, and the derivation (above) then sets the name and type a full build gives the entity. Every rule is idempotent: re-applying a delta changes nothing. The cross-run merge runs per index suffix and reads the stored items back through `neptune_codec`, the exact inverse of the Neptune write encoding.

**Communities on delta runs**: a delta run clusters only the delta subgraph (MS GraphRAG's update path also appends rather than re-clustering). Community ids are therefore **content-derived** — a hash of the level and the sorted member entity ids (`community_detector.community_content_id`); the positional `L{level}_C{i}` label survives only as `short_id`/`name`. A delta community gets an id distinct from every corpus community unless its membership is identical (then the upsert rewrites the same community), so the delta's communities and reports (whose ids hash the community id) are appended next to the existing ones instead of overwriting `L0_C0…` of the full corpus. The upsert path therefore needs no read-back or id-offset merge. A corpus community or report is deleted once none of the text units it cites is stored; shared ones keep their pre-change report until a full rebuild, which is also the only path that re-partitions the whole graph (re-clustering only the region a delta touches is not implemented; `indexing.community_drift_threshold` reports when a rebuild is due). Leiden's partition depends on node/edge insertion order even with a fixed seed, so the detector sorts nodes and edges before partitioning (`_canonical_graph`). `graspologic_native` also returns its labels in a Rust `HashMap` whose order changes from process to process, and that order would set the next level's cluster-graph node numbers and the small-community merge order, so each level's communities are renumbered by their smallest member id (`_partition_dict_to_communities`) and merge ties go to the lowest number. The same graph and `random_state` therefore always yield the same communities and ids at every level. Re-detecting over the full graph on every delta run was rejected: it would regenerate every community report (an LLM call each) per delta run, defeating incremental indexing.

**Concurrency**: with the doc-status registry, one run writes an index namespace at a time. A pipeline run takes its namespace's write lease from the registry before its first stage and holds it to its end (`application/ingestion/namespace_lease.py`): one conditional write (`DocStatusPort.try_acquire_lease`; in DynamoDB a put of the item `namespace` / `#lease` with `lease_owner` and `lease_expires_at`, conditioned on no unexpired lease of another owner; in SQLite a `leases` row in a write transaction). A second run fails with `NamespaceLeaseError`, naming the holder and when its lease expires, before it reads or writes anything. A heartbeat thread renews the lease every third of `indexing.lease_seconds` (default 300 s), so a run that dies stops renewing and its lease expires at most that long after (DynamoDB also drops the expired item: `lease_expires_at` is the table's TTL attribute). Every incremental registry and store write (write-ahead, removal, commit, the reset's registry cleanup) first checks the lease, so a run whose renewals failed until it expired, or whose lease another run then took, stops instead of writing over the holder. Without the registry nothing locks: concurrent runs on one suffix race on its OpenSearch aliases (a full reindex deletes the alias's older indices, `_reap_stale_indices`) and Neptune labels, and are not supported. The direct `IncrementalIndexer` API takes no lease. Runs on different namespaces lease apart. The CDK state machine does not block an overlapping execution and the stack starts none on its own (no schedule or trigger); the lease stops the second one. See [Operator Runbook: Concurrent runs](./operations.md#concurrent-runs).

Enable with: `config.aws.dynamodb.enabled = true`.

---

## 6. Retrieval: Two Methodologies

The `GraphRAGChain` (an LCEL Runnable) in `application/retrieval/rag_chain.py` performs memory (a follow-up with conversation history is rewritten as a standalone question, `QueryRewritePrompt`) → strategy resolution → query processing (translation, entity/keyword extraction) → retrieval → (RAG) context build + answer generation. Routing, retrieval and the answer all read the standalone question. The methodology is selected via `RAGInput.search_strategy`.

![Retrieval chain](../assets/retrieval-pipeline.drawio.png)

![Search strategies](../assets/retrieval-strategies.drawio.png)

`ChainMode` decides where the chain stops: `SEARCH` returns the fused, ranked retrieval results without an answer; `RAG` (the default) packs them into the token budget (`TokenManager`, §7), generates the answer, and returns a `RAGOutput` whose sources are exactly the sections the answer model saw. When the budgeted context is empty, the chain answers that it cannot answer without calling the LLM.

Neptune graph expansion is part of `local` and `drift`; for `mix`/`hybrid` it is opt-in (`search.lightrag_search.enable_graph_expansion`). The strategies find the seed entities in OpenSearch and pass their ids to the graph retriever as an `id` filter; `NeptuneRetriever` expands from those ids only and returns nothing for a query without one.

### 6.1 GraphRAG Methodology (`adapters/search_strategies/`)

- **simple**: OpenSearch-only vector/lexical, no graph. If claim extraction is enabled, the claims index is also automatically swept; if disabled, `_apply_claim_gate` explicitly excludes the claims index so a claims-off run never queries that index.
- **local**: Entity-centric — candidate entities → Neptune graph expansion → frequency filter → text-unit combination, enriched with a **community-report section** and a **relationship section** (like MS GraphRAG local search, which assembles entities + their community reports + in-network relationships + text units). `_retrieve_community_reports` and `_retrieve_relationships` query those indices with the entity focus (falling back to the raw query); the relationship section is gated on `build_relationship_vector_index` so a GraphRAG-only deployment that skips the relationship vector index issues no relationship lookup. If claim extraction is enabled, it **injects claims (covariates) into the context** like MS GraphRAG (`_retrieve_claims` queries the claims index separately and adds them as `all_results["claims"]`, folded into the token budget at `SectionType.CLAIM` priority). The default claims-off path performs no additional lookups at all. The community-report, relationship and claim lookups run concurrently with the entity → expansion → text-unit chain. With `search.local_search.include_bridge_relationships` (default on, needs the relationship index), the relationships incident to the expanded entities are fetched as well (`_fetch_incident_relationships`, shared with LightRAG's incident expansion), edges whose both endpoints were retrieved first — MS GraphRAG local's in-network relationships, which carry the hops of a multi-hop chain that the relationship vector query rarely matches.
- **global**: MS GraphRAG global search over a cut of the community hierarchy (see §6.1.1): every report of the `community_depth` cut (default 2 from the root, MS GraphRAG's `community_level`), or of a root-down dynamic selection (`use_dynamic_selection`), is mapped into scored key points; the ranked points are the answer context, followed by the reports they came from. Similarity to the query is used only to cap a cut larger than `max_reports_per_level`. Text units are added only with `text_unit_slots` (default 0).
- **drift**: MS GraphRAG DRIFT search (`drift_search.py`). **Primer** — `DriftHydePrompt` expands the query into a hypothetical answer shaped like the most important community report (MS GraphRAG picks a random one), the `k_followups` reports most similar to that text are split into `primer_folds` contiguous folds, and each fold's `DriftPrimerPrompt` call returns an intermediate answer, a 0-100 score and follow-up queries. **Actions** — for `n_depth` rounds the `k_followups` open follow-up queries whose parent answer scored best (MS GraphRAG shuffles them) each run `LocalSearchStrategy.asearch` (the same local search, sharing the scorer and token counter); the local results, packed into `local_context_tokens`, are answered by `DriftActionPrompt` with a score against the overall question and new follow-up queries, which join the queue (a query asked twice runs once). Actions run `processing.max_concurrency` at a time. **Reduce** — the intermediate answers, best score first and packed into `max_answer_tokens`, are one synthesized `synthesized_drift_answers` section (never a reported source), and the deduplicated local evidence of every action follows it, fused with the local per-section-type quota and reranking text chunks only; the answer model reduces them, as MS GraphRAG's reduce step does. Without community reports, or when the primer yields no follow-up, the question itself runs as the one follow-up. A failed expansion, primer or action call degrades under `processing.ignore_errors` (the expansion falls back to the question, an action keeps its local evidence) and raises otherwise; a fatal error raises either way. The search metadata reports `drift_primer_reports`, `drift_follow_up_queries`, `drift_depth`, `drift_answers`, `drift_open_follow_ups` and `drift_llm_calls`, and `kg_rag.drift.primer` / `kg_rag.drift.actions` spans trace the steps.
- **auto**: LLM routing via `StrategySelectionPrompt` among `search.auto_routable_strategies` (default local, mix, global, drift; simple is left out) minus the strategies the configured pipeline cannot serve: global and drift register `needs_community_reports=True`, so with `graph.community_detection.enabled: false` they are not offered (`routable_strategies`; local when nothing is left); the prompt's strategy guide is built from that list, so it describes exactly the strategies the router may pick. The response is parsed by word token, first routable name wins; unrecognised output, or a failed routing call under `processing.ignore_errors`, falls back to local, or to the first routable strategy when local is not routable.

#### 6.1.1 Global search (`global_search.py`)

**Which reports.** Each indexed community report carries its place in the hierarchy (`domain/ingestion/community_hierarchy.py`, attached by `IndexingManager` when the reports are written, so reports restored from a stage cache get it too): `level` (bottom-up, 0 = finest, as detection builds it), `depth` (from the root, derived from the parent links), `parent_community_id`, `child_community_ids`, and `max_cut_depth` (its own depth for a community with children, a large constant for a leaf). The cut at depth `D` is every community at depth `D` plus the leaves that stop above it, i.e. `depth <= D <= max_cut_depth`: MS GraphRAG's `read_indexer_reports` roll-up, which covers every entity once. MS GraphRAG counts `community_level` from the root, which is why the depth, not the bottom-up level, selects the cut.

- **Fixed cut (default)** — `_cut_reports` browses the cut (no query text, ordered by `rank`) with the caller's filters. If the browse fills `max_reports_per_level` (bounded by `indexing.opensearch.max_query_size`, the most hits one request returns), the cut is too big to map whole and a hybrid search for the query restricted to the same cut picks that many most similar reports; `global_level_capped` in the search metadata says so. This is the only place similarity enters global search.
- **Dynamic selection (`use_dynamic_selection`)** — MS GraphRAG's `DynamicCommunitySelection`: rate the root reports with `community_relevance_model_id` (`CommunityRelevancePrompt`, 1-10, normalized to 0-1), queue the children of each report at or above `relevance_threshold` (default 0.2, MS GraphRAG's 1 on 0-5), drop a parent once one of its children is relevant, and rate the whole next depth when a depth yields no relevant report; the walk stops below `community_depth`. Ratings run `processing.max_concurrency` at a time. A rating that fails is kept (as MS GraphRAG keeps an unparseable one) under `ignore_errors`, and raised otherwise; a fatal error (no model access) is raised either way. No relevant report means no context, so the chain answers that it cannot.
- Reports indexed before these fields existed match no cut; the strategy then raises `ConfigurationError` asking for a re-index (`indexing.reset: true`) instead of answering every query from nothing.

**Map-reduce.** When `enable_map_reduce` is set and at least `map_reduce_min_results` reports were selected, it follows MS GraphRAG's canonical map-reduce. Below `map_reduce_min_results` the reports **pass through unchanged** — no synthesis stage runs at all, because a handful of community reports do not need a map-reduce to be summarized.

1. **MAP** — The selected reports are packed in order into batches of `map_context_tokens` report tokens (default 12000, MS GraphRAG's `data_max_tokens`; a longer report gets a batch of its own), and for each batch `GlobalMapPrompt` asks the map model (fast tier) to extract key points and score query relevance from **0-100**. Batches run `processing.max_concurrency` at a time via `BatchProcessor`, with per-item graceful fallback. The metadata reports `global_mapped_report_count`, `global_map_call_count` and `global_map_report_tokens`, and a `kg_rag.global.map` span carries the same counts.
2. **FILTER+RANK** — Drops points at or below `map_relevance_threshold` and sorts by score in descending order (`_filter_and_rank_points`).
3. **PACK** — Packs the top points up to the `max_map_reduce_tokens` token budget (based on `token_manager.count_tokens`, `_pack_points_within_budget`).
4. **REDUCE** — By default (`reduce_with_llm: false`) the packed points, with their relevance annotations, are prepended to the results as one `synthesized_key_points` `RetrievalResult` and the answer model does the synthesis itself: a separate reduce LLM only added a call and a second rewrite that could drop facts or assert that the summaries lack them. The results after it are the reports of the batches whose points were packed, ordered by their batch's best point (a point is attributed to its batch, not to one report), so the answer budget left after the points goes to the reports behind them and the reported sources name them. With `reduce_with_llm: true`, `MapReduceSummaryPrompt` first synthesizes a summary from the packed points (`_reduce_from_points`), prepended as `synthesized_summary`. Either item is flagged `metadata.synthesized`: the answer model reads it as context, but it is LLM output rather than retrieved evidence, so it is never reported in `RAGOutput.sources`.

Robustness: Even if the map response comes wrapped in code fences or prose, `_parse_map_payload` extracts the JSON, and a batch whose map call failed or returned unparseable output is tracked as *unrated*. `_concat_reduce` is the degradation path when no point passes the threshold but some reports were never rated — over just the unrated reports (all of them when every batch failed), which pass through unchanged unless `reduce_with_llm` summarizes them — so global search still answers instead of hard-failing or declaring no data for reports nobody judged. When the map stage rated *every* batch but every point scored at or below `map_relevance_threshold`, the reports were judged irrelevant: global search returns no results (flagged as `map_reduce_no_relevant_points` in the search metadata), matching MS GraphRAG's no-data answer, and the chain's empty-context guard replies "cannot answer" instead of synthesizing from rejected reports. `MapReduceSummaryPrompt` likewise restricts the reduce step to the provided points and tells it to say so when they do not answer the query.

### 6.2 LightRAG Methodology (`lightrag_search.py`)

Runs dual-level keyword retrieval (hl/ll extraction via `KeywordsExtractionPrompt`) on top of the shared hybrid infrastructure.

Modes (`RAGInput.search_strategy`):
- **naive** — chunk retrieval only, no graph: `chunk_stream_top_k` chunks under the request's `search_type` (`hybrid` BM25 + vector by default; `vector` is upstream's pure-vector baseline).
- **hybrid** — ll → entity index + hl → relationship index + one-hop cross-type expansion (entity hits → incident relationships, relationship hits → endpoint entities) + the chunks those items cite.
- **mix** — hybrid graph search, plus the **source chunks cited by the matched entities/relationships** (`text_unit_ids` lineage), plus a naive vector chunk retrieval, all blended in.

Per-source behavior:
- **Low-level keywords (ll)** → entity index (lexical + semantic, `entities_index_prefix`)
- **High-level keywords (hl)** → **relationship index** (corresponds to LightRAG's `relationships_vdb`; `relationships_index_prefix`, `Relationship.description` embedding)
- **Retrieval rounds** — the independent entity / relationship / (mix) vector-chunk queries run concurrently; the incident-relationship and endpoint-entity expansions then run concurrently; the cited chunks are fetched last (they depend on both). Upstream LightRAG has no multi-hop traversal, so the Neptune neighbourhood expansion (= GRAPH role, `indexing.neptune.max_hops`) is opt-in via `search.lightrag_search.enable_graph_expansion` (default `false`).
- **mix linked chunks** — entities and relationships are indexed with their `text_unit_ids` chunk lineage, so `mix` follows the matched entities/relationships back to the chunks that support them (`_collect_linked_chunk_ids` ranks chunk ids by how many matched items cite each, mirroring LightRAG's `_find_related_text_unit_from_entities`/`_from_relationships`), fetches those chunks by id, and blends them alongside the naive vector chunks. Degrades to the naive-only blend when an index predates the lineage field.
- If keyword extraction yields nothing, short queries fall back to using the raw query as ll keywords (config `search.lightrag_search.raw_query_fallback_max_len`)
- All sources are fused through the shared `HybridScorer`

> The two methodologies share the same ingestion outputs (entities/relationships/communities/chunks + embeddings) and branch only at the retrieval layer — so a single indexed corpus answers both GraphRAG and LightRAG queries with no re-indexing. This is a deliberate extension over the source papers: stock MS GraphRAG does not build a relationship vector index, and stock LightRAG does not do community detection/reports; here the ingestion builds the *union* so either methodology can run.
>
> The trade-off: a full ingestion pays GraphRAG's community-detection + report-generation cost even if you only query with LightRAG. For a LightRAG-only deployment, set `graph.community_detection.enabled: false` to skip the Leiden pass and all community-report LLM calls — `mix`/`hybrid`/`naive` need only entities, relationships, and the relationship vector index. (Leave it on if you want `global`/`drift` available.)

---

## 7. Hybrid Scoring and Token Management

- **HybridScorer** (`adapters/retrieval/hybrid_scorer.py`): Combines per-source results via RRF (`rrf_k`) or weighted fusion, then reranks, then applies MMR diversity filtering (`diversity_lambda`), then cuts to the per-type quota or `top_k`. Reranking comes before diversity, as upstream LightRAG reranks every chunk candidate before truncating: the reranker sees the best `search.reranking.top_k` fused candidates of the rerank-eligible types and MMR works on its scores, so a cut on fused scores cannot drop what the reranker ranks first. Weights/method come from `config.search.fusion`/`hybrid`. Reranking is only active when `search.reranking.enabled`, and each call reranks with a copy of the model whose `top_n` matches the document count. If the rerank model cannot be built (initialization fails, or the provider returns no model), queries keep the fused order and report rerank status `unavailable`, or the scorer raises `RerankModelError` under `search.reranking.required`; status `disabled` is reserved for `search.reranking.enabled: false` and `providers.rerank.kind: none`. Each fusion call records its rerank outcome on the calling query's `RerankTrace` (created per `asearch`, never stored on the shared scorer), which the strategy reports as `SearchResult.metadata["rerank"]`.
  - **IAM caveat**: the Rerank API authorizes `bedrock:Rerank` against a different resource shape than `InvokeModel`, so a statement scoped to foundation-model or inference-profile ARNs denies it. The IaC task role grants `bedrock:Rerank` on `Resource: "*"` in its own statement (`ComputeStack._build_task_role` in `iac/stacks/compute_stack.py`). Without it the scorer logs `Reranking failed: ...` at ERROR and returns the fused results unreranked.
- **TokenManager** (`adapters/retrieval/token_manager.py`): Packs the context into the answer model's budget. Each section type present gets a hard-capped share of the budget (`search.token_manager.type_budgets`, renormalized over the types present); within its share a type's sections are taken in descending score order and the first one that overflows is truncated to fit (dropped if under `min_truncated_section_tokens`). Budget a type leaves unused because all of its candidates fit then goes to the types that still have candidates, which are packed again: to the strategy's `spare_context_budget_first` types first (LightRAG mix/hybrid: chunks, as upstream gives chunks what its KG sections leave of `max_total_tokens`), otherwise in proportion to the receivers' shares, until nothing is left or no type has candidates. Shares and that hand-out, never cross-type score comparisons, decide the mix, and an unfilled share is not wasted while another type is being cut. Synthesized global-search sections (map key points, reduce summary) are seated first. Only when no type can seat anything (a window too small for its shares) is the single highest-scoring section seated, truncated if needed. `SectionType.CLAIM` is the type used to fold query-time claims injection (§6.1) into the token budget. The chain keeps this selection (`OptimizedContext`) in state and builds `RAGOutput.sources` from it, so sources list only what the answer model saw, in retrieval-rank order: sections cut for budget are not reported, a section truncated to fit carries its truncated text and `metadata.truncated: true`, and each source's metadata keeps `truncated` / `source_id` / `document_ids` / `chunk_id` / `section_type` / `score` with `*_embedding` vectors removed. When the answer step short-circuits on an empty context, `sources` is empty.
- **Token counting** (`adapters/aws/token_counter.py`, `adapters/local/token_counter.py`): With the defaults every count is the script-aware estimate (`adapters/llm/token_estimate.py`): CountTokens rejects the default Claude 5.x models and all embedding and rerank models, so their counters skip it (see below). A language model that accepts it is counted with the Bedrock `count_tokens` API, falling back to the estimate on failure. Local language models use the estimate, or a BPE encoding with `providers.llm.tokenizer: tiktoken` (`[local]` extra). A non-transient failure (e.g. `AccessDeniedException`, or a `ValidationException` whose message says the model does not support the operation) marks the model unsupported process-wide so later counts skip the API; throttling, timeouts, and input-level `ValidationException`s do not. Blank or whitespace-only text never calls the API. Embedding and rerank counters are built without a client, so they never call the API (it does not accept those models); language models that CountTokens rejects (Claude 4.7+/5.x, OpenAI GPT) skip it by capability flag (`supports_count_tokens`). Truncation uses a convergence loop that estimates candidates by char ratio and validates them with the counter.

---

## 8. AWS Service Integration

| Service | Module | Purpose |
|---|---|---|
| **Bedrock** | `adapters/aws/bedrock.py`, `adapters/aws/bedrock_models.py` (capability catalog) | LLM/embedding/reranking. Automatic cross-region inference profile resolution, provider-aware request shaping (Anthropic adaptive thinking + `effort` on Claude 4.6+, `budget_tokens` on older Claude; `reasoning.effort` on OpenAI GPT via Converse), 1M context, prompt caching where the model supports explicit cache points, capability table (curated rows, then provider-family defaults by id prefix, then a conservative unknown default; `aws.bedrock.model_overrides` on top) so any Bedrock model id is accepted |
| **Neptune** | `adapters/aws/neptune.py` | Gremlin over `wss://` (`ws://` to a local Gremlin Server with `aws.neptune.use_ssl: false`), SigV4 IAM, batch upsert/delete. Write batches submit concurrently via a thread pool when `indexing.neptune.index_concurrency` > 1 (per-batch independent `IndexingStats` → merged on the main thread, no shared mutation), multiplexed over the Gremlin connection pool (`aws.neptune.pool_size`, raised to at least `index_concurrency`). Default 1 = sequential. Every indexer request, reads and drops included, is retried up to `indexing.neptune.max_attempts` times over a new connection, and repeating one changes nothing: vertices (full build and delta alike) are created or matched by id, a `MemberOf` edge is added only for an entity not yet a member, and a relationship edge only when its id is missing, so a retry after a lost response adds no duplicate |
| **OpenSearch** | `adapters/aws/opensearch.py` | Vector (kNN/HNSW, default engine **lucene**, which supports `cosinesimil` on OpenSearch 2.13 as deployed by `iac/`; faiss rejects `cosinesimil` before 2.19, so use it only with `innerproduct` for >1024-dim models — see the engine note in `config-template.yaml`) + BM25, async SigV4, sync/async clients, alias management, bulk upsert/delete, per-language analyzers (en→english, ko→nori, etc.) |
| **S3** | `adapters/aws/s3_cache.py` | Pipeline cache sync (encryption defaults to `BUCKET_DEFAULT` — the bucket's default encryption, e.g. a CMK; `AES256`/`aws:kms` force per-object SSE) |
| **DynamoDB** | `adapters/doc_status/dynamodb.py` | Incremental-indexing document-status registry |

The AWS session is the bundle's (`Providers(boto_session=...)`, by default created from `config.aws.profile_name` on first use), so a fake/moto session can be injected during testing. Only an AWS client asks for it: a local run, or a store without IAM auth, never resolves the profile.

### 8.5 Retrieval Error Visibility

The retrievers (`opensearch_retriever`/`neptune_retriever`) do not disguise authentication, configuration or store failures as "no results." One policy, `is_fatal_retrieval_error()` (`adapters/store_errors.py`), decides by exception type alone, never by message: fatal errors are re-raised with `exc_info`, and only the rest degrade to `[]`, so an incorrect IAM permission, a wrong model id or a store that is down surfaces instead of being buried as "0 search hits." The store clients translate failures at their boundary (`adapters/aws/opensearch.py`, `adapters/aws/neptune.py`): a store that cannot be reached, closed the connection or still answers 429/502/503/504 after the client's own retries raises `StoreConnectionError` (a `ConnectionError`), one that did not answer within `search.store_timeout_seconds` its subclass `StoreTimeoutError`, and a request refused the same way on every retry raises a `StoreConfigurationError` (a `ConfigurationError`): `StoreAccessDeniedError` for HTTP 401/403 or a Neptune `AccessDeniedException`, `StoreRequestRejectedError` with the server's reason for an HTTP 400 (a filter value of another type than its field, a query vector of another dimension). A botocore error from a model call, such as the query embedding, is fatal unless its code is a transient Bedrock one (`is_transient_bedrock_error`): an invalid or inaccessible model id, missing credentials, or a Bedrock endpoint that could not be reached or timed out after the model wrapper's retries fails the query. A failed query embedding is fatal whatever the cause, a 5xx left after the client's retries included (`EmbeddingModelError`): every vector section of the query shares that one embedding, so nothing is left to answer from. A local model server's error is fatal by the same rule: an OpenAI-compatible or Ollama HTTP 4xx other than 408/429 (a bad key, an unknown model) or a server that could not be reached or did not answer after the client's retries (`openai.APIConnectionError`, `httpx.TransportError`); the OpenAI, Ollama and httpx types are looked up only once their modules are loaded. A prompt longer than the model's context (`langchain_core`'s `ContextOverflowError`, which Bedrock raises as a `ValidationException`) is not fatal: only that prompt hits it, so a global-search map batch that overflows is left unrated and the others still answer. vLLM and llama.cpp report an overflow as a plain HTTP 400 (`openai.BadRequestError`), which LangChain does not map to `ContextOverflowError`, so the map step leaves a batch that got a 400 unrated too, unless every batch got one: a 400 on every map call (a parameter the server does not support, a model id it does not know) fails the query, whichever model the answer call uses. What remains transient is a failure of one request, such as an HTTP 500 for one search. Neptune's per-query `_execute_traversal` re-raises fatal errors too, so they reach that top-level guard, and a lost Neptune connection is dropped (`run_traversal`, `reset`) so the next query, or the indexer's next retry, opens a new one. The query traversal is built in its worker thread from the client's source of that moment (not before waiting for a pool slot), and runs on threads the retriever owns rather than the loop's default executor. Its bound of twice `search.store_timeout_seconds` frees the slot, not the thread; `reset` and `close` wake a request still waiting for a pooled connection of the closed client (gremlinpython's `close()` empties the pool and `pool.get()` has no timeout), which then fails as a lost connection. A request whose write was in flight when the client closed fails the same way: gremlinpython's `close()` shuts down the executor the read would run on, so the closed client's executor hands such a late read a failed future (`_close_connection`) instead of leaving the request unanswered and its thread waiting forever. A request that waits on a server that stopped answering is bounded by the transport's read timeout: on the query side by `search.store_timeout_seconds`, and on the indexer's side by a fixed 180 s, above Neptune's own 120 s default query timeout so its long writes and drops are not cut short; an ingestion request that times out is retried like other transient errors. The search strategies apply the same rule to every sub-retrieval through one helper, `BaseSearchStrategy._safe_aretrieve`, so a section-level handler cannot re-swallow a fatal error the retriever raised; a transient failure still degrades only that section, and local search keeps the seed entities the entities index found when graph expansion degrades.

### 8.6 Client Lifecycle / Resource Release

Each retriever build opens a Neptune WebSocket + thread pool and OpenSearch (a)sync HTTP pools. These resources leak until GC unless explicitly closed. Therefore every layer exposes best-effort `close()`/`aclose()` (never raising):

- **OpenSearchClient**: `close()`/`aclose()` + sync/async context managers (mirroring NeptuneClient). When the event loop changes, the previous `AsyncOpenSearch` is closed: awaited on its own loop while that loop runs, otherwise by running the aiohttp connector's close coroutine to completion without a loop, so per-loop aiohttp pools do not leak. `aclose()` awaits the transport close to prevent the "Unclosed client session" warning.
- **NeptuneClient**: Closes the Gremlin connection pool.
- **Chain wiring**: Retrievers/indexers delegate up to `IndexingManager.close()` / `GraphRAGChain.close()`·`aclose()` (iterating the cached retrievers). Whoever builds a chain must close it when done: a GC finalizer releases a dropped chain's retrievers and loop as a safety net, but only once the garbage collector reaches it (the chain is in a reference cycle, so that can be much later). The `run-rag` and `run-eval` CLIs call `await rag_chain.aclose()` in `finally`, and the `run-ingestion` CLI calls `pipeline.close()` in `finally`, releasing sockets at process exit.
- **Event loops**: `GraphRAGChain` caches retrievers and strategy instances per event loop (tracked by reference, not `id()`). Its sync entry points (`invoke`, `batch`, `stream`) run on one chain-owned loop thread, started lazily and stopped by `close()`/`aclose()`, so repeated sync calls reuse one set of loop-bound clients and also work from a thread that already runs a loop. When the loop does change, the evicted retrievers are closed — awaited on their own loop while it still runs. Blocking backend calls stay off the loop thread: the Neptune connect/close and the fusion + rerank step run via `asyncio.to_thread`, and the rerank `top_n` is applied to a per-call copy of the shared model. LangChain runs `ChatBedrockConverse.ainvoke` in the loop's default executor too, so that executor bounds concurrent Bedrock calls; the CLIs and the chain-owned loop size it to `processing.io_workers` (`shared/utils/event_loop.py`), and an async host does the same with `configure_event_loop`.

### 8.7 Multilingual Processing

- **OpenSearch analyzers**: The language→analyzer mapping is exposed via config (`indexing.opensearch.language_analyzers`, default `{"en": "english", "ko": "nori"}`) so it can be extended without code changes. nori (the Korean morphological analyzer) comes from the `analysis-nori` plugin: the local OpenSearch container installs it, but an Amazon OpenSearch Service domain has it only once its optional Nori package is associated, so the CDK deployment maps `ko` to OpenSearch's built-in `cjk` analyzer (character bigrams) instead (`opensearch_korean_analyzer`, injected as `OPENSEARCH_KOREAN_ANALYZER`). A mapping naming an analyzer the cluster lacks fails index creation with an error that names it. Languages without a mapping fall back to `default_analyzer`.
- **Entity ID normalization**: IDs hash `entity_key` (`shared/utils/common.py`): NFKC + casefold, symbols and letters/digits of all scripts preserved, only quotes/commas/trailing punctuation dropped → Korean, CJK, and accented names get unique IDs and "C++"/"C#" stay distinct, while `Entity.name` keeps the display form (§3). Non-empty input is not collapsed to an empty ID.
- **Translation skip**: When `TranslationConfig.is_noop` (source_language == target_language and no additional target languages), the translation stage is skipped in its entirety at no cost (`TranslationStage._should_skip` in `application/ingestion/pipeline_stages.py`).
- **Encoding auto-detection**: When the text parser hits a `UnicodeDecodeError` on a non-UTF-8 file, it detects the encoding via `charset-normalizer` and retries with an explicit `encoding=` (`FileParser._detect_encoding` in `adapters/ingestion/parser.py`). LangChain's `autodetect_encoding=True` (which pulls in the extra `chardet` dependency) is intentionally not used.

---

## 9. Evaluation Framework

`evaluation/` — `EvaluationManager` dispatches evaluators via `_resolve_evaluator_class` (one lazy-import branch per `EvaluatorType`).

- **LangChain evaluators**: correctness / partial_correctness (LLM-based rubric)
- **RAGAS evaluators**: answer_correctness/relevancy, context_precision/recall, faithfulness
- **Strategy comparison** (`evaluation/comparison.py`, `evaluation/pairwise.py`): `run-eval` with several strategies or `--repeats` runs the same queries per strategy and repeat, averages each metric per query over the repeats and reports per-query deltas against the first strategy with a seeded paired bootstrap CI (comparing pooled means would mix query difficulty and run-to-run noise into the difference); `auto` reports the strategies it routed to. The optional pairwise LLM judge (`PairwiseComparisonPrompt`: comprehensiveness, diversity, empowerment, directness, as in the GraphRAG/LightRAG papers) judges both answer orders and keeps only order-stable verdicts. It is not a `BaseGraphRAGEvaluator`: it scores two results per query, so it runs after the runs instead of per result. `prepare-benchmark` (`evaluation/benchmarks/musique.py`) turns MuSiQue into a corpus + dataset at run time; `evaluation.answer_normalization: official_squad` makes EM/F1 comparable with published numbers.
- **Run manifest**: identifies what was measured. Ingestion writes an `IndexBuildRecord` (pipeline id, full/incremental, build time, documents, extraction fingerprint, package version) into the vector index mappings' `_meta` through `VectorIndexer.record_build` (default no-op for backends without index metadata); `EvaluationManager` reads it back through the chain's DOCUMENT retriever per queried suffix. The config hash covers only retrieval-relevant settings, so it is equal across environments.
- **Retrieval evaluator** (`retrieval_evaluator.py`): `hit_at_k`, `recall_at_k` (k = `evaluation.retrieval_k`) and `mrr` against the dataset's `reference_sources`, over the distinct files of the rank-ordered reported sources (a graph source is attributed to its lineage's files, most-supported first), matched case-insensitively by file name (a reference without an extension also matches the file-name stem; document ids are not used: they are content hashes no dataset names). k counts files, not sources: a community report is attributed to every file of its community, and counting it as one source let random reports score near-perfect hit@k. The same metrics per source section type go to `grouped_statistics.source_type` (the `text` row = only the text units the answer model read). Deterministic, no LLM; skipped for queries without references or source provenance.
- **Answer-match evaluator** (`answer_match_evaluator.py`): `answer_contains` (headline) plus SQuAD-style normalized `exact_match` and `token_f1` (normalization keeps decimals and `C++`-style suffixes, so not leaderboard-comparable) against `answer` plus optional `metadata.answer_aliases` (max over references). Deterministic, no LLM; whitespace tokens, so for scripts without spaces token F1 degrades to exact match.
- **Graph-aware evaluator** (`graph_aware_evaluator.py`): Computes the rate at which the ground truth's `expected_entities`/`expected_relationships` appear in the generated answer (= coverage = recall) as `ENTITY_COVERAGE`/`RELATIONSHIP_COVERAGE`. Deterministic, no LLM required. Precision/F1 are not produced because they would require enumerating the entities in the answer (impossible from free text) — to avoid exaggerating the signal as a duplicate of recall. Latin characters use word-boundary contiguous token matching ("AI" does not match inside "airport"); CJK without whitespace falls back to substring matching. Because answer-text coverage grows with answer length, it also reports `RETRIEVED_ENTITY_COVERAGE`/`RETRIEVED_RELATIONSHIP_COVERAGE` (the expectations matched against the entity names and relationship endpoints of the sources the answer model was given) and `ANSWER_LENGTH`. The manager injects the expectations via `result.metadata`, so the abstract signature is unchanged.

CLI: `run-eval --eval-data-path <json> [--search-strategy ...]`.

---

## 10. Visualization & Analytics

`visualization/` — `BaseRenderer` ABC + `@register_renderer` registry + `RenderContext`.

- `InteractiveRenderer` (pyvis network + community hierarchy), `StaticRenderer` (Bokeh degree/centrality/community-size)
- Layout: embeddings of each node's `name: description` + UMAP dimensionality reduction (spring layout on failure)
- **Standalone execution** (`application/cli/run_visualization.py`): Reads exported graph JSON without ingestion (`export_visualization_data` output format: `nodes`/`edges`/`layout`/`communities.hierarchy`), rehydrates it into typed objects, and renders with the registered renderers. The data-only file ingestion writes when rendering is off (`graph.visualization.export_data`, `domain/ingestion/visualization_data.py`) has an empty `layout` and no centrality; the CLI computes a seeded spring layout and the configured centrality for it.

---

## 11. Prompts and Prompt Tuning

- **Prompts** (`prompts/`): Classes based on `BasePrompt` whose templates and variables are class attributes (never instantiated). System/human templates are version-controlled as `.py`. Every prompt can be overridden from config via `CustomPromptConfig` (e.g., medical/legal/financial domains).
- **Prompt tuning** (`application/prompts/tuner.py`, ported from MS `prompt_tune`): Corpus sample → profile domain/language/persona/entity-types via a Bedrock LLM (`CorpusProfilePrompt`) → generate a domain-adapted `custom_prompts` YAML fragment. CLI: `run-prompt-tuning`. This is an explicit step where the user reviews and applies it to config, not automatic runtime application. The tuned `graph_extraction_system` / `community_report_system` replace only the persona preamble (`system_preamble`): the prompt's built-in `output_rules` (XML schema, verbatim `source_text` grounding) are embedded verbatim, so the parser and grounding guard keep working.

---

## 12. Configuration System

The nested Pydantic tree in `domain/models/config.py` (root `Config`), loaded by `shared/config.py` (`get_config`); the schema example is `config-template.yaml`.

- Sections: `backend`, `providers` (llm/embedding/rerank/graph/vector/doc_status), `aws` (bedrock/neptune/opensearch/s3/dynamodb), `fixing`, `processing` (document_parsing/chunking/translation/graph_extraction/gleaning/claim_extraction), `graph` (analysis/community_detection/visualization), `indexing` (opensearch/neptune), `search` (hybrid/fusion/reranking/global_search/local_search/drift_search/lightrag_search/token_manager), `memory`, `cache` (chunking), `logging`, `observability`, `evaluation`, `custom_prompts`.
- **Config-based generalization**: The language→analyzer mapping (`language_analyzers`), LightRAG fallback length, and eigenvector convergence parameters are all exposed via config.
- Adding a new config section: Define a Pydantic `BaseModel` → attach to its parent via `Field(default_factory=...)` → document in `config-template.yaml`.

---

## 13. Testing Strategy

`tests/{unit,integration,property,fixtures/fakes}/` — **AWS-free by default**.

- **Port-based fake adapters** (`fixtures/fakes/`): In-memory implementations of GraphStore/VectorStore/DocStatus verify domain logic without real AWS (the test-side benefit of hexagonal architecture).
- **moto**: Verifies the DynamoDB/S3 adapters against the boto3 surface.
- Layers: unit (models/registry/merge/dual-keyword/evaluation/token-counter/lineage relevance), property (hypothesis: merge laws and incremental-vs-full-build convergence, interrupted runs, fusion monotonicity, MinHash and token-estimate equivalence, hash length; the `DocStatusPort.diff` partition is checked against every registry in `tests/integration/test_doc_status_conformance.py`), integration (incremental add/change/delete cycles), regression.
- **Store conformance suites** (`tests/integration/test_{graph_store,vector_store,retrieval}_conformance.py`): each store contract is written once and runs on every backend through the `StoreHarness` of `tests/fixtures/store_backends.py`: the fakes (the reference for write semantics), the embedded SQLite stores, the Gremlin Server and OpenSearch containers of `docker/compose.local.yaml` (`LOCAL_STORES=1`) and Neptune and OpenSearch Service (`-m aws`, under a unique `indexing.additional_suffix` that is cleared afterwards). A backend without a capability (the fakes have no retrievers and no index administration) skips the tests that need it. `tests/integration/test_store_parity.py` (containers only) indexes one synthetic corpus into the SQLite engine and the OpenSearch container: lexical search must return the same top 10 with scores within 5 %, hybrid search (the retriever's min-max fusion of a lexical and a kNN search) the same top 10 with fused scores within 0.01, and the container's approximate kNN must find at least 98 % of the exact top 10 (recall@10 ≥ 0.98). The CI `local-stores` job runs both against the containers.
- Markers: `unit`, `integration`, `property`, `aws` (real AWS, excluded in CI), `slow`. `asyncio_mode = "auto"`.

Run: `uv run pytest -m "not aws" --cov=unified_kg_rag`.

---

## 14. CI/CD and Security

- **CI** (`.github/workflows/`): the `quality` workflow gates pull requests and pushes to `main` (lint, types, the test suite with its coverage floor, the image build and the IaC synth with cdk-nag); `licenses` checks dependency licenses when the dependency set changes, and `security` runs a report-only ASH scan on `main`. The workflow files are the source of truth for the job list, and [CONTRIBUTING.md](../CONTRIBUTING.md#checks-to-run-before-pushing) lists the checks to run locally.
- **Dependabot** (`.github/dependabot.yml`): weekly version updates for the `uv` lock (`/`), the IaC `uv` lock (`/iac`, a separate uv project), and the SHA-pinned GitHub Actions. Known-breaking bumps are held back with an `ignore` entry that records the reason.
- **pre-commit** (`.pre-commit-config.yaml`): Mirrors the CI gates. `pre-commit install`.
- **Security hardening**: Content hashes use SHA-256 exclusively (CWE-327-safe). Dependency-scan CVEs are addressed through the Dependabot pull requests above. Tokens are injected via environment/config (no hardcoding in code).
- **Guardrail at the query's edges** (`ports/guardrail.py`, `adapters/guardrail/bedrock.py`): `GraphRAGChain` screens the raw user query with Bedrock `ApplyGuardrail` (source `INPUT`) before memory, rewrite, routing and retrieval, and the final answer (source `OUTPUT`) after generation; a block short-circuits the query, a mask replaces the text, and a guardrail error fails the query closed. The guardrail is not attached to the model calls in between: an attached guardrail evaluates the whole prompt, so on Converse it judges the retrieved corpus and the developer instructions as user input (false prompt-attack blocks, cost proportional to context × calls), and on InvokeModel its prompt-attack filter is inert without input tags. Screening the two user-facing texts evaluates exactly what the user wrote and reads, once each. Ingestion and evaluation models get an attached guardrail only under `apply_to: all`.

---

## 15. Extension Guide

This section is the single reference for extending the framework; the README, user guide and CONTRIBUTING.md link here. Strategies, renderers and parsers extend through registries, evaluators through one branch in their type map, and backends through constructor injection; none needs a change to dispatch code.

- **New search strategy**: Add a `SearchStrategy` enum member (`domain/models/retrieval.py`; strategies are keyed by this closed enum, and the CLI choices follow it) + subclass `BaseSearchStrategy` + `@register_strategy(SearchStrategy.X, required_roles=(...), query_inputs=frozenset({QueryInput.ENTITIES}))` + export from `adapters/search_strategies/__init__.py`. `query_inputs` declares the query-side LLM extractions the strategy reads (`ENTITIES` for `entity_focus`, `DUAL_KEYWORDS` for `hl_keywords`/`ll_keywords`); the chain skips the rest. A strategy that answers from community reports also passes `needs_community_reports=True`, so `auto` skips it when none are built. No edit to `rag_chain` is needed; add the strategy to `search.auto_routable_strategies` if `auto` may pick it.
- **New storage/LLM backend**: Implement the relevant port, then either register a builder for a new `providers.<role>.kind` with `@register_provider(ProviderRole.X, "kind")` (`adapters/provider_registry.py`) and select it in config, or pass the port to the constructor that uses it (see "Custom backends" below). Do not hardcode it into a manager's `__init__`.
- **New evaluator**: Subclass `BaseGraphRAGEvaluator` + add a branch in `EvaluationManager._resolve_evaluator_class` + an `EvaluatorType` enum.
- **New renderer**: Subclass `BaseRenderer` + `@register_renderer("name")`. Registration happens on import: `GraphVisualizationManager` sees any renderer imported in the process, `run-visualization` only those that `adapters/renderers/__init__.py` imports.
- **New parser / file format**: `ParserFactory.register_loader(".ext", MyLangChainLoader, loader_kwargs=..., file_type_name=...)` — any LangChain `BaseLoader` subclass; no edit to the factory, and the extension is then auto-discovered + parseable. Override a built-in by registering its extension.
- **New config section**: Define a Pydantic `BaseModel` → attach it to its parent via `Field(default_factory=...)` → document it in `config-template.yaml` (see §12).

### Custom backends (run without AWS)

Every cross-service dependency is behind a port. Each port has a role in the
provider bundle, the role's adapter is chosen by `providers.<role>.kind` through
the provider registry (below), and the orchestrators also accept any port via
**constructor injection** — so a non-AWS or custom backend is wired in without
subclassing or editing dispatch code. The ports, their default
(Bedrock/Neptune/OpenSearch/DynamoDB) adapters and the local ones
(`adapters/local/` and the SQLite implementations in the role packages, the
`[local]` extra; see the user guide's "Local mode"):

| Port | Contract | Default adapter (local adapters) | Inject via |
|---|---|---|---|
| `LLMFactoryPort` / `EmbeddingFactoryPort` / `RerankFactoryPort` (`ports/model_factory.py`, `Protocol`) | `get_model()` / `get_model_info()` returning a LangChain-compatible model | `BedrockLanguageModelFactory` / `BedrockEmbeddingModelFactory` / `BedrockRerankModelFactory` (`openai_compatible`, `ollama`; `cross_encoder`, `none`) | a `Providers` bundle (see below): `GraphRAGChain(providers=...)`, `DataIngestionPipeline(..., providers=...)`, `EvaluationManager(..., providers=...)`; `GraphRAGChain(model_factory=...)` is shorthand for an LLM-only bundle |
| `TokenCounterPort` (`ports/model_factory.py`, `Protocol`) | `count_tokens()` / `truncate_to_token_limit()` | `BedrockTokenCounter` (`LocalTokenCounter`: estimate or tiktoken; follows the LLM kind) | `Providers(token_counter_factory=...)` |
| `VectorIndexer` / `GraphIndexer` (`ports/indexer.py`, ABC) | `index_*` / `upsert_*` / `delete_by_id`; the incremental lookups `GraphIndexer.find_ids_citing` / `find_entity_ids_by_name_key` and `VectorIndexer.find_documents` / `delete_extraction_records` default to "cannot tell" (`None`: nothing is found through them, and the merge reads every name) and to deleting nothing; `GraphIndexer.begin_rebuild` / `end_rebuild` (an in-place full build, see the reset above) default to "cannot" (`False`: the full-build writes clear each label first); `initialize()` (a no-op unless overridden) / `clear()` return `None` and must raise a typed store error (`StoreConnectionError`, `StoreConfigurationError`, ...) on failure: a returned `False` is ignored | `OpenSearchIndexer` / `NeptuneIndexer` (`gremlin`: the same adapter on Gremlin Server; `sqlite`: `OpenSearchIndexer` on `SQLiteSearchEngine` / `SQLiteGraphIndexer`) | `Providers(vector_indexer_factory=..., graph_indexer_factory=...)`, `DataIngestionPipeline(..., vector_indexer=..., graph_indexer=...)` or `IndexingManager(vector_indexer=..., graph_indexer=...)` |
| retriever (role-keyed builder) | `BaseGraphRAGRetriever.aretrieve` | `OpenSearchRetriever` / `NeptuneRetriever` (`sqlite`: `OpenSearchRetriever` on `SQLiteSearchEngine` / `SQLiteGraphRetriever`) | `Providers(retriever_builders=...)` or `GraphRAGChain(retriever_builders={RetrieverRole.GRAPH: lambda: MyGraphRetriever(...)})` |
| `ContentGuardPort` (`ports/guardrail.py`, `Protocol`) | `async acheck(text, source)` returning a `GuardVerdict` (`none` / `masked` / `blocked` and the text to continue with); raise when it cannot decide | `BedrockGuardrail` (built when `aws.bedrock.guardrail.identifier` is set) | `GraphRAGChain(guard=...)` |
| `DocStatusPort` (`ports/doc_status.py`, `Protocol`) | `get` / `get_many` / `put` / `put_many` / `delete` / `list_namespace` / `list_all` / `diff` (which must take `extraction_fingerprint`), `record_fits` and the namespace write lease (`try_acquire_lease` / `lease_holder` / `release_lease`), all called directly; a record's namespace is the `<namespace>#` prefix of its `doc_id` (`doc_id_namespace`), and `diff` with a scope reads only that scope's namespace. Subclassing `DocStatusPort` inherits defaults for `get_many` / `put_many` / `record_fits` (one record per call, no size limit) | `DynamoDBDocStatusStore` (`SQLiteDocStatusStore`) | `Providers(doc_status=...)` or `DataIngestionPipeline(..., doc_status=...)` (either turns incremental indexing on); conform structurally and pass `tests/integration/test_doc_status_conformance.py` |

**Errors from a custom retriever or backend** are classified by type, never
by message (`adapters/store_errors.is_fatal_retrieval_error`). To fail the
query instead of answering from no context, raise a `ConnectionError` or
`ConfigurationError` subclass: the typed store errors in `shared/exceptions.py`
(`StoreConnectionError`, `StoreTimeoutError`, `StoreConfigurationError`,
`StoreAccessDeniedError`, `StoreRequestRejectedError`) cover the usual cases.
Any other exception, such as `AWSServiceError("access denied ...")`, degrades
only the section that raised it.

Because the model-factory and doc-status ports are `runtime_checkable`
`Protocol`s, a custom class only needs the right **method shapes** — no base
class to import. Ollama and OpenAI-compatible servers are built in
(`providers.llm.kind`); a custom factory covers anything else, for example:

```python
class MyLLMFactory:                       # structurally an LLMFactoryPort
    def get_model(self, model_id, **kwargs): ...   # returns a LangChain chat model
    def get_model_info(self, model_id): ...        # returns a LanguageModelInfo

chain = GraphRAGChain(config=cfg, model_factory=MyLLMFactory())
```

**One bundle, every component.** `Providers` (`adapters/providers.py`) is a
value object holding the boto3 session, the LLM, embedding, rerank and
token-counter providers, and the builders of the graph and vector indexers and
retrievers and the doc-status registry. `application/bootstrap.py`
(`build_providers`) is the framework's composition root: it checks the
configured kinds and builds the bundle; each orchestrator builds one this way
(or receives one) and passes it explicitly to what it constructs, so an
injected or configured provider reaches all of them:

- `GraphRAGChain`: its own prompts, every search strategy (including the
  global map/reduce and DRIFT chains), the hybrid scorer's reranker, both token
  managers, and the default OpenSearch retriever's embeddings. Conversation
  memory (no model of its own: the follow-up rewrite is the chain's prompt) is
  process-wide by default (so history survives a chain built per request) and
  is created from the first chain's config; a later chain with a different
  memory config keeps the shared manager and logs a warning once. Pass
  `GraphRAGChain(memory_manager=MemoryManager(cfg))` to isolate a chain's
  conversations.
- `DataIngestionPipeline`: the chunker, translator, graph/claim extraction,
  gleaning, description summarization, community reports, the default
  OpenSearch indexer's embeddings and the visualization embedder.
- `EvaluationManager`: the LangChain and RAGAS judges (LLM, embeddings, token
  counter). By default it reuses the chain's bundle.

```python
providers = Providers(
    cfg,
    llm_factory=MyLLMFactory(),
    embedding_factory=MyEmbeddingFactory(),
    # Reranking is on by default (search.reranking.enabled) and would otherwise
    # call Bedrock; pass a factory or set search.reranking.enabled: false.
    rerank_factory=MyRerankFactory(),
    token_counter_factory=lambda model_id, **_: MyTokenCounter(model_id),
)
chain = GraphRAGChain(config=cfg, providers=providers)
pipeline = DataIngestionPipeline(cfg, pipeline_config, providers=providers)
```

The stores follow the same bundle: without `retriever_builders` the chain
reads, and without `doc_status`/`vector_indexer`/`graph_indexer` the pipeline
writes to, the stores `providers.graph` / `providers.vector` /
`providers.doc_status` select (see the table above). Those orchestrator
arguments (and `GraphRAGChain(model_factory=...)`) are shorthand for a bundle
built with them: with a `providers=` bundle, set them on the bundle instead
(passing both raises `ValueError`).

Anything not supplied is built lazily by the builder registered for the
configured kind (Bedrock, Neptune, OpenSearch and DynamoDB by default); the
model factories at most once per bundle, so a chain constructs its model
clients once rather than per component or per query; stores and the
doc-status registry are built once per consumer, and strategy instances are
reused per event loop. A component constructed directly, without a
bundle, still builds a default one from the same config, so existing direct
constructors keep working and follow `config.providers`. There is deliberately
no DI container: the bundle covers exactly the providers the framework
consumes.

**Provider registry.** `adapters/provider_registry.py` maps
`(ProviderRole, kind)` to a builder that receives the bundle; the built-in
kinds register at the bottom of `adapters/providers.py`, the local ones import
their adapter only when built. A new kind needs no edit to the bundle, the
config model or any call site:

```python
from unified_kg_rag.adapters.provider_registry import ProviderRole, register_provider

@register_provider(ProviderRole.LLM, "my_server")
def _build(providers):
    return MyServerFactory(providers.config)
```

and `providers.llm.kind: my_server` selects it (a non-Bedrock LLM kind also
needs a token-counter builder for the same kind, or
`Providers(token_counter_factory=...)`). `backend: local` is a config preset
(`domain/models/provider_config.py`), not a code path: it only fills in
`providers`, `aws.neptune` and `aws.opensearch` keys, and the retrieval
context budget (`search.token_manager.max_context_tokens`), that the config
leaves unset.
Model ids are strings; embedding and rerank ids are validated against the
Bedrock catalog only while the Bedrock provider serves that role.

> **Accept and ignore unknown kwargs.** Callers pass framework-specific keyword
> arguments through `get_model(model_id, **kwargs)` — for example
> `model_purpose=ModelPurpose.QUERY | INGESTION | EVALUATION`, which the Bedrock
> factory uses to attach a guardrail to ingestion and evaluation models under
> `apply_to: all`. New ones may be added in later releases. A
> custom factory must keep the `**kwargs` catch-all and ignore keys it does not
> understand (and must not forward them blindly to a model constructor that
> rejects unknown arguments); a strict signature such as
> `get_model(self, model_id, temperature=0.0)` raises `TypeError` on the first
> call that passes `model_purpose`.

The in-memory fakes in `tests/fixtures/fakes/` (e.g. `FakeGraphStore`,
`FakeVectorStore`) are working reference implementations of the indexer ports —
the whole ingestion+indexing pipeline runs against them with no AWS
(`DataIngestionPipeline(cfg, pipeline_config, providers=Providers(cfg,
llm_factory=..., embedding_factory=..., doc_status=FakeDocStatusStore(),
vector_indexer_factory=lambda: vectors, graph_indexer_factory=lambda: graph))`,
or without a bundle `DataIngestionPipeline(cfg, pipeline_config,
doc_status=..., vector_indexer=..., graph_indexer=...)`, exercised in
`tests/integration/test_ingestion_stages.py`). They are
the recommended starting point for a custom store. Besides the AWS adapters the
framework ships the local ones in-tree (`adapters/local/`: OpenAI-compatible
and Ollama models, a cross-encoder reranker, a local token counter and the
embedded SQLite graph and vector stores; `adapters/doc_status/sqlite.py`: the
SQLite registry; all of them what `backend: local` selects). The two stores show the two ways to add a
backend. The vector store, `SQLiteSearchEngine` (`adapters/local/search_engine/`),
implements the `SearchEngineClient` protocol (`adapters/storage/search_engine.py`),
the OpenSearch client calls the indexer and retriever make, so the unchanged
`OpenSearchIndexer` and `OpenSearchRetriever` run on it and both backends
share one copy of the indexing and retrieval logic; a request outside the
query subset it implements is rejected (`StoreRequestRejectedError`). Gremlin
cannot be emulated that way, so the graph store is its own `GraphIndexer`
(`SQLiteGraphIndexer` in `adapters/storage/sqlite_graph_indexer.py`, with
`FakeGraphStore` as the reference for the write semantics) and a `GraphExpansionRetriever` subclass
(`adapters/retrievers/graph_expansion.py`) that implements only the traversal;
the scoring, cutting and content building it shares with `NeptuneRetriever`.

### Deliberate Design Boundaries

The codebase makes three boundary calls worth stating explicitly, so they read as
intentional decisions rather than oversights:

- **`SearchQuery` carries adapter vocabulary (label/index prefixes) by design.**
  The domain query model exposes index/label prefixes that the search strategies
  and both retrievers read and write (~120 references). The second store
  pairing, the embedded SQLite graph and vector stores, kept it: a prefix names
  a logical collection (an index, a vertex label) that each pairing maps to its
  own storage, so the SQLite stores read the same prefixes without a new
  abstraction. The vector store speaks the OpenSearch protocol and the graph
  store keys its rows by the same `graph_label`. The hexagonal boundary that
  matters — the write-side indexer ports and the model-factory ports — *is*
  abstracted and dependency-injected (see §2.1 and the `IndexingManager` /
  `ModelFactoryPort` DI seams). Hoisting the vocabulary behind a fully
  backend-neutral query model would be a large, behavior-risking churn that a
  backend which cannot map named collections would have to justify.

- **One physical OpenSearch index per index suffix per artifact type.**
  Multi-tenant/versioned isolation uses a real index per index suffix
  (`{prefix}-{suffix}`). With a handful of tenants this is fine; with tens of
  thousands of index suffixes it multiplies the cluster's index/shard count and
  the cluster-state overhead. The scale-out fix is a single index per artifact
  type with a `tenant` filter field + routing (delete-by-query instead of index
  drop) — a behavior-affecting change across the index/search/delete paths,
  deferred as a dedicated migration. That design would
  need the tenant filter enforced on every index and Neptune label, unlike
  today's metadata filters: those apply only to the stores that declare a key
  (relationships, claims and community vertices pass unfiltered) and graph
  expansion and community reports cross documents, so they are relevance
  filters, not access control. Isolation comes from separate index suffixes
  (user guide §4, attribute filters).

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
