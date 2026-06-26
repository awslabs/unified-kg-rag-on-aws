# Unified Knowledge Graph RAG on AWS — Technical Documentation

This document is a **design reference for contributors and advanced users**, covering the architecture, algorithms, data model, and operational aspects of the `unified-kg-rag-on-aws` library. For the "what/why" and a quick start, see [README.md](../README.md); for "how to use it," see [docs/user-guide.md](user-guide.md) (English) / [docs/user-guide.ko.md](user-guide.ko.md) (Korean); for contribution/extension conventions, see [CLAUDE.md](../CLAUDE.md). A Korean version of this document is available at [docs/design.md](design.md).

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

---

## 1. Overview and Design Philosophy

`unified-kg-rag-on-aws` is a library that reimplements the Microsoft GraphRAG paper on top of an AWS-native stack (Bedrock + Neptune + OpenSearch + S3 + DynamoDB). The core design principles are as follows.

- **Two methodologies, one infrastructure**: GraphRAG (community-summary) and LightRAG (dual-level keyword) share the same ingestion, indexing, caching, multilingual, and hybrid-search infrastructure, and **only the retrieval algorithm layer is swapped**.
- **Generalization first**: We avoid hardcoding, regex heuristics, and overfitting. Semantic judgments are delegated to the LLM or to authoritative data, token counting uses the Bedrock `count_tokens` API, and thresholds/weights are config-driven.
- **Hexagonal boundaries**: Domain/algorithm code depends on abstract ports, with concrete AWS adapters placed behind them.
- **Registry-based extension**: Search strategies, evaluators, and renderers are registered via decorator registries, so they can be extended without modifying dispatch code.

---

## 2. Hexagonal Architecture (Ports & Adapters)

### 2.0 Dependency Rule and Layer Map

Imports point **inward** (left imports right, never the reverse). `shared/` is a cross-cutting kernel any layer may use. The two RAG methodologies (GraphRAG community-summary, LightRAG dual-level keyword) share one ingestion/indexing/caching/hybrid-search infrastructure and diverge only at the algorithm layer.

![Hexagonal Architecture](../assets/hexagonal-architecture.png)

```
application  ──►  adapters  ──►  ports  ──►  domain
        └──────────────┴────────────┴──────────►  shared (cross-cutting kernel)
```

```
unified_kg_rag/
├─ domain/              # technology-agnostic core (no boto3/LangChain/backend imports)
│  ├─ models/           #   Pydantic domain models
│  ├─ ingestion/        #   pure algorithms: delta_detector, graph_analyzer/
│  │  └─ merge/         #   builder/resolver, claim_resolver, merge/merger
│                       #   (IncrementalIndexer is orchestration, so it lives in application/)
│  ├─ retrieval/        #   strategy_registry, MetricsMixin
│  └─ prompts/          #   version-controlled prompt templates
├─ ports/               # abstract interfaces the domain depends on (DocStatusPort,
│                       #   BaseIndexer/GraphIndexer/VectorIndexer, CachePort,
│                       #   ModelFactoryPort — ports/__init__ is the port catalog)
├─ adapters/            # concrete technology bindings
│  ├─ aws/              #   Bedrock, Neptune, OpenSearch, DynamoDB, S3 clients
│  ├─ storage/          #   Neptune/OpenSearch indexers (write-side port implementations)
│  ├─ retrievers/       #   Neptune/OpenSearch retrievers
│  ├─ search_strategies/#   simple/local/global/drift + lightrag(mix/hybrid/naive)
│  ├─ retrieval/        #   abstract retriever/strategy bases, hybrid scorer, token/memory managers
│  ├─ ingestion/        #   LLM/IO coupled: chunker, *_extractor, loader, parser,
│  │                    #   translator, gleaner, community_detector
│  ├─ renderers/        #   graph visualization renderers
│  └─ evaluators/       #   langchain/ragas evaluators (the pure graph_aware_evaluator is
│                       #   co-located in the evaluation/ facade)
├─ application/         # orchestration + entry points
│  ├─ cli/              #   run-ingestion/rag/eval/visualization/prompt-tuning
│  ├─ ingestion/        #   DataIngestionPipeline + pipeline_stages
│  ├─ storage/          #   IndexingManager (indexer fan-out)
│  ├─ retrieval/        #   rag_chain (GraphRAGChain, RAGInput/Output)
│  └─ prompts/          #   PromptTuner (LLM-based corpus profiling)
├─ shared/              # cross-cutting kernel (config, logging, exceptions, metrics,
│                       #   cache/pipeline manager, utils)
└─ (facades)            # thin re-export shims to keep public import paths stable:
   retrieval/ storage/ ingestion/ evaluation/ visualization/
```

### 2.1 Ports (Abstract Interfaces)

| Port | Location | Adapter | Notes |
|---|---|---|---|
| `DocStatusPort` | `ports/doc_status.py` | `adapters/aws/dynamodb.py` (`DynamoDBDocStatusStore`), `FakeDocStatusStore` for tests | Persists incremental-indexing document status/lineage |
| `CachePort` | `ports/cache.py` (`Protocol`) | `shared/cache_manager.py` (local) + `adapters/aws/s3_cache.py` (S3) | Stage-result persistence boundary |
| `GraphIndexer` (write-side) | `ports/indexer.py` | `adapters/storage/neptune_indexer.py` | Single contract for full + delta (`upsert_*`/`delete_by_id`) |
| `VectorIndexer` (write-side) | `ports/indexer.py` | `adapters/storage/opensearch_indexer.py` | Same |
| `BaseGraphRAGRetriever` (read-side) | `adapters/retrieval/base.py` | `adapters/retrievers/{neptune,opensearch}_retriever.py` | Retrieval adapters |
| LLM/Embedding/Rerank factories | `adapters/aws/bedrock.py` | Bedrock implementation | Future LLMPort extraction target (see §15 below) |

> Design note: Pure ports (`DocStatusPort`, the write-side indexer ABCs) are gathered in `ports/`. The read-side abstract bases (`BaseGraphRAGRetriever`/`BaseSearchStrategy`) are "adapter bases" that construct infrastructure (HybridScorer/TokenManager) in `__init__`, so they live in `adapters/retrieval/base.py` and are re-exported from `ports/__init__` for discovery (no duplicate Protocol definition is kept).

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

- **Search strategies**: `domain/retrieval/strategy_registry.py` — `@register_strategy(...)` registers a class and its required roles against the `SearchStrategy` enum.
- **Evaluators**: `EvaluationManager.EVALUATOR_MAPPING` — `EvaluatorType` → evaluator class.
- **Renderers**: `adapters/renderers/base.py` — `@register_renderer("name")`.

This pattern follows the same philosophy as the existing `ParserFactory._loader_configs` (declarative parser registration).

### 2.4 Dependency Rule Verification Status

Verified with grep: `domain/` does not import `adapters`/`application` at runtime, and neither does `ports/`. One compile-time-only exception remains — `domain/retrieval/strategy_registry.py` references `adapters.retrieval.base.BaseSearchStrategy` under `TYPE_CHECKING` (because the registry stores strategy subclasses). Extracting pure strategy/retriever ports would also remove this type-level reference; it is tracked as future work in §15.

The legacy top-level packages (`retrieval/`, `storage/`, `ingestion/`, `evaluation/`, `visualization/`) are intentionally preserved as thin facade `__init__` modules so that public import paths/APIs remain stable after the layer split.

---

## 3. Domain Model

The `domain/models/` package contains pure Pydantic models with no infrastructure dependencies.

- `Entity` (`name`, `description`, `type`, `text_unit_ids`, `community_ids`, `rank`, `frequency`, `confidence`, embedding fields)
- `Relationship` (`source_id`/`target_id`, `description`, `weight`, `text_unit_ids`, `description_embedding`)
- `Community` / `CommunityReport`, `TextUnit`, `Covariate` (claim)
- `DocStatus` (state machine: PENDING→PARSING→PROCESSING→PROCESSED|FAILED), `DocStatusRecord` (content hash + artifact lineage + suffix), `DocumentDelta` (new/changed/unchanged/deleted), `DocumentLineage` (per-document artifact attribution)
- `SearchQuery`/`SearchResult`/`RetrievalResult`, `SearchStrategy`/`SearchType`/`RetrieverRole`

**Lineage is the core data.** Entities/relationships record the `text_unit_ids` they appeared in at extraction time, and this authoritative data replaces the (old) token-overlap heuristic, judging "is this entity related to this text unit?" accurately and language-independently.

**Entity IDs and multilingual support**: Entity/relationship IDs are hashes of the normalized name. `normalize_name` (`shared/utils/common.py`) applies NFKC + casefold, then **preserves letters/digits of all scripts** (`\w`, `re.UNICODE`) and removes only punctuation. As a result, Korean, CJK, and accented names also get unique IDs (ASCII-only normalization would collapse non-Latin names to empty strings, collapsing the graph). Non-empty input is never collapsed to an empty ID.

---

## 4. Ingestion Pipeline

The `DataIngestionPipeline` in `application/ingestion/pipeline.py` runs 12 stages in order (`application/ingestion/pipeline_stages.py`).

![Ingestion Pipeline](../assets/ingestion_pipeline.png)

| # | Stage | Module | Notes |
|---|---|---|---|
| 1 | Document parsing | `parser.py` (`ParserFactory`) | PDF/TXT/CSV/JSON out of the box (+MD/HTML via the optional `unstructured` extra) |
| 2 | Document loading | `loader.py` (`DirectoryLoader`) | MinHash deduplication |
| 3 | Chunking | `chunker.py` (`ChunkerFactory`) | simple / intelligent (LLM semantic) |
| 4 | Translation (optional) | `translator.py` | multilingual → target language |
| 5 | Graph extraction | `graph_extractor.py` | LLM entity/relationship extraction |
| 6 | Gleaning (optional) | `gleaner.py` | iterative refinement (convergence/quality thresholds are config) |
| 7 | Graph resolution | `graph_resolver.py` + `description_summarizer.py` | fuzzy-matching merge, `text_unit_ids` union, **LLM re-summarization of merged descriptions** |
| 8 | Claim extraction (optional) | `claim_extractor.py` | factual assertions (covariate) |
| 9 | Claim resolution (optional) | `claim_resolver.py` | |
| 10 | Graph analysis | `graph_analyzer.py` | centrality (degree/betweenness/PageRank/eigenvector), statistics |
| 11 | Community detection | `community_detector.py` | hierarchical Leiden, community report generation (degree-sort + token-budget pack) |
| 12 | Indexing | `application/storage/indexing_manager.py` | OpenSearch + Neptune |

> The stage order is single-sourced in `DataIngestionPipeline.STAGE_CLASSES` (`pipeline.py:62`). Stages that require Bedrock are declared in `BOTO_REQUIRED_STAGES`; with the addition of description re-summarization, **graph resolution (7) is now also included in this set**.

**Merged-description re-summarization (stage 7)**: Graph resolution merges descriptions of the same entity/relationship by simple concatenation, so the description of a popular entity that appears in many chunks grows without bound. `DescriptionSummarizer` (run in `GraphResolutionStage`) re-summarizes only descriptions that exceed a token budget into a single coherent description using a cheap LLM (parity with MS GraphRAG `summarize_descriptions` / LightRAG `_handle_entity_relation_summary`, controlled by `DescriptionSummarizationConfig`). The goal is to prevent embedding/prompt bloat.

**Community report context pack (stage 11)**: The report-generation input **sorts entities within a community by graph degree in descending order** (ties broken by stable id sort), caps them at `max_entities_per_report`, and packs them to fit the `max_report_context_tokens` token budget (relationships are sorted/packed identically by the sum of both endpoints' degree, with weight as the tiebreak). The top-degree entity is always included (at least one) even if it alone exceeds the budget, so a report never ends up with empty context (`community_detector.py:651-720`).

**Pipeline infrastructure**: Stage-checkpoint-based resumption (`shared/pipeline_manager.py`), S3 cache sync (`adapters/aws/s3_cache.py`), a `continue_on_error` toggle, per-stage caching (`shared/cache_manager.py`). The translation stage is skipped at no cost when `TranslationConfig.is_noop` (source == target & no additional languages). LLM output parsing consistently uses a `FixingConfig`-based output-fixing parser. The pipeline releases indexer/client resources via `close()`, which the `run-ingestion` CLI calls in `finally` (§8.6).

**Generalization in practice**:
- The relevance gate (which entities to include in the prompt for claim/gleaning) is decided by `text_unit_ids` lineage membership rather than a token-Jaccard regex heuristic → accurate and language-independent.
- The scale constants in the gleaning quality/convergence formula (entities 50, relationships 100, completeness weight 0.6, change scale 20) are all exposed via `GleaningConfig`.

---

## 5. Incremental Indexing

When documents are added/changed/deleted, only the delta is processed instead of a full re-index.

1. **Delta detection** (`domain/ingestion/delta_detector.py`): Builds `{doc_id: content_hash}` from a stable `doc_id` (based on path normalization) + content SHA-256 hash, and `DocStatusPort.diff()` classifies them as new/changed/unchanged/deleted.
2. **Stale cleanup** (`IncrementalIndexer.prune_changed`): For changed documents, first removes existing artifacts that are *not shared* (so entities that disappear after re-extraction do not linger in the graph).
3. **Delta upsert** (`IndexingManager.index_delta`): Neptune uses a Gremlin `coalesce(unfold, addV)` idempotent upsert; OpenSearch upserts by id into the live alias index. The relationship vector index is updated the same way.
4. **Deletion propagation** (`remove_deleted`): Removes only the *exclusive* artifacts of deleted documents via `delete_by_id` (preserving shared entities). Targets the text-unit, entity, and relationship indices alike.
5. **Registry update**: Records processed documents into `DocStatusRecord` as `DocumentLineage` (per-document artifact ids + suffix).

**Merge semantics** (`domain/ingestion/merge/merger.py`, ported from MS GraphRAG `update/*`): Entities merge by normalized name (concatenating descriptions, union of `text_unit_ids`, recomputing `frequency`, preserving existing ids + remap); relationships merge by (source, target) (averaging weight); communities append by id-offset.

Enable with: `config.aws.dynamodb.enabled = true`.

---

## 6. Retrieval: Two Methodologies

The `GraphRAGChain` (an LCEL Runnable) in `application/retrieval/rag_chain.py` performs strategy resolution → query processing (translation, entity/keyword extraction) → memory → retrieval → (RAG) context build + answer generation. The methodology is selected via `RAGInput.search_strategy`.

![Retrieval Pipeline](../assets/retrieval_pipeline.png)

### 6.1 GraphRAG Methodology (`adapters/search_strategies/`)

- **simple**: OpenSearch-only vector/lexical, no graph. If claim extraction is enabled, the claims index is also automatically swept; if disabled, `_apply_claim_gate` explicitly excludes the claims index so a claims-off run never queries that index.
- **local**: Entity-centric — candidate entities → Neptune graph expansion → frequency filter → text-unit combination. If claim extraction is enabled, it **injects claims (covariates) into the context** like MS GraphRAG (`_retrieve_claims` queries the claims index separately and adds them as `all_results["claims"]`, folded into the token budget at `SectionType.CLAIM` priority). The default claims-off path performs no additional lookups at all.
- **global**: Community report search → community node expansion → LLM dynamic relevance selection → **map-reduce synthesis** (see §6.1.1 below).
- **drift**: Iterative query evolution (community seeding → LLM query refinement/keyword expansion → convergence detection).
- **auto**: LLM routing among the above strategies via `StrategySelectionPrompt`.

#### 6.1.1 Global search map-reduce (`global_search.py`)

When `enable_map_reduce` is set and results are at least `map_reduce_min_results`, it follows MS GraphRAG's canonical map-reduce (replacing the earlier simple concat-reduce).

1. **MAP** — Community reports are batched in groups of `map_batch_size`, and for each batch `GlobalMapPrompt` asks the LLM to extract key points and score query relevance from **0-100**. Batches are run concurrently via `BatchProcessor`, with per-item graceful fallback.
2. **FILTER+RANK** — Drops points at or below `map_relevance_threshold` and sorts by score in descending order (`_filter_and_rank_points`).
3. **PACK** — Packs the top points up to the `max_map_reduce_tokens` token budget (based on `token_manager.count_tokens`, `_pack_points_within_budget`).
4. **REDUCE** — Synthesizes the final answer from the packed points (with relevance annotations) via `MapReduceSummaryPrompt` (`_reduce_from_points`). The result is prepended to the results as a `synthesized_summary` `RetrievalResult`.

Robustness: Even if the map response comes wrapped in code fences or prose, `_parse_map_points` extracts the JSON, and a single batch's parse failure is ignored. If the map stage produces no useful points at all or everything is filtered out by the threshold, it gracefully degrades to the legacy `_concat_reduce` so global search does not hard-fail.

### 6.2 LightRAG Methodology (`lightrag_search.py`)

Runs dual-level keyword retrieval (hl/ll extraction via `KeywordsExtractionPrompt`) on top of the shared hybrid infrastructure.

Modes (`RAGInput.search_strategy`):
- **naive** — vector chunk retrieval only, no graph.
- **hybrid** — ll → entity index + hl → relationship index + Neptune graph expansion.
- **mix** — hybrid graph search with naive chunk retrieval additionally blended in.

Per-source behavior:
- **Low-level keywords (ll)** → entity index (lexical + semantic, `entities_index_prefix`)
- **High-level keywords (hl)** → **relationship index** (corresponds to LightRAG's `relationships_vdb`; `relationships_index_prefix`, `Relationship.description` embedding)
- Entity hits are expanded via Neptune (= GRAPH role)
- If keyword extraction yields nothing, short queries fall back to using the raw query as ll keywords (config `search.lightrag_search.raw_query_fallback_max_len`)
- All sources are fused through the shared `HybridScorer`

> The two methodologies share the same ingestion outputs (entities/relationships/communities/chunks + embeddings) and branch only at the retrieval layer.

---

## 7. Hybrid Scoring and Token Management

- **HybridScorer** (`adapters/retrieval/hybrid_scorer.py`): Combines per-source results via RRF (`rrf_k`) or weighted fusion, diversity filtering (`diversity_lambda`), and Bedrock reranking. Weights/method come from `config.search.fusion`/`hybrid`. Reranking is only active when `search.reranking.enabled`, and `compress_documents` temporarily adjusts `top_n` to the document count before restoring it. On initialization failure, the reranker degrades to disabled (`None`).
  - **IAM caveat**: Bedrock Rerank requires the `bedrock:Rerank` permission on **`Resource:*`** (discovered in real-AWS E2E — narrowing it to the model ARN yields AccessDenied). Isolate it as a dedicated statement in IaC.
- **TokenManager** (`adapters/retrieval/token_manager.py`): Optimizes context within model limits. Weights by per-section-type priority multiplier (`PRIORITY_MULTIPLIERS`: TEXT 1.3 / ENTITY 1.2 / RELATIONSHIP 1.1 / CLAIM 1.1 / COMMUNITY 1.0 / GENERAL 0.8) and selects sections within budget in descending priority order. `SectionType.CLAIM` is the type used to fold query-time claims injection (§6.1) into the token budget.
- **Token counting** (`adapters/aws/token_counter.py`): The Bedrock `count_tokens` API is the single source of truth. It degrades to a whitespace word count only on failure (no third-party tokenizer). Truncation uses a convergence loop that estimates candidates by char ratio and validates them via the API.

---

## 8. AWS Service Integration

| Service | Module | Purpose |
|---|---|---|
| **Bedrock** | `adapters/aws/bedrock.py` | LLM/embedding/reranking. Automatic cross-region inference profile resolution, thinking mode, 1M context, prompt caching, capability table |
| **Neptune** | `adapters/aws/neptune.py` | Gremlin over `wss://`, SigV4 IAM, batch upsert/delete. Write batches submit concurrently via a thread pool when `indexing.neptune.index_concurrency` > 1 (per-batch independent `IndexingStats` → merged on the main thread, no shared mutation), with `aws.neptune.pool_size` multiplexing the Gremlin connection pool. Default 1 = sequential |
| **OpenSearch** | `adapters/aws/opensearch.py` | Vector (kNN/HNSW, default engine **faiss** — nmslib is deprecated) + BM25, async SigV4, sync/async clients, hybrid search pipeline, alias management, bulk upsert/delete, per-language analyzers (en→english, ko→nori, etc.) |
| **S3** | `adapters/aws/s3_cache.py` | Pipeline cache sync (AES256/KMS encryption) |
| **DynamoDB** | `adapters/aws/dynamodb.py` | Incremental-indexing document-status registry |

All adapters can be injected with a `boto_session` (by default created from `config.aws.profile_name`), so a fake/moto session can be injected during testing.

### 8.5 Retrieval Error Visibility

The retrievers (`opensearch_retriever`/`neptune_retriever`) no longer silently disguise authentication/configuration/connection failures as "no results." `is_fatal_retrieval_error()` (`adapters/retrieval/base.py:46`) re-raises fatal errors with `exc_info`, and degrades to `[]` only for transient errors. As a result, an incorrect IAM permission or an endpoint typo is not buried as "0 search hits."

### 8.6 Client Lifecycle / Resource Release

Each retriever build opens a Neptune WebSocket + thread pool and OpenSearch (a)sync HTTP pools. These resources leak until GC unless explicitly closed. Therefore every layer exposes best-effort `close()`/`aclose()` (never raising):

- **OpenSearchClient**: `close()`/`aclose()` + sync/async context managers (mirroring NeptuneClient). When the event loop changes, the previous `AsyncOpenSearch` is discarded immediately (best-effort connector close) so per-loop aiohttp pools do not leak. `aclose()` awaits the transport close to prevent the "Unclosed client session" warning.
- **NeptuneClient**: Closes the Gremlin connection pool.
- **Chain wiring**: Retrievers/indexers delegate up to `IndexingManager.close()` / `GraphRAGChain.close()`·`aclose()` (iterating the cached retrievers). The `run-rag` CLI calls `await rag_chain.aclose()` in `finally`, and the `run-ingestion` CLI calls `pipeline.close()` in `finally`, releasing sockets at process exit.

### 8.7 Multilingual Processing

- **OpenSearch analyzers**: The language→analyzer mapping is exposed via config (`indexing.opensearch.language_analyzers`, default `{"en": "english", "ko": "nori"}`) so it can be extended without code changes. nori (the Korean morphological analyzer) is built into OpenSearch Service. Languages without a mapping fall back to `default_analyzer`.
- **Entity ID normalization**: `normalize_name` (`shared/utils/common.py`) applies NFKC + casefold, then preserves letters/digits of all scripts (`\w`, `re.UNICODE`) and removes only punctuation → Korean, CJK, and accented names also get unique IDs (§3). Non-empty input is not collapsed to an empty ID.
- **Translation skip**: When `TranslationConfig.is_noop` (source_language == target_language and no additional target languages), the translation stage is skipped in its entirety at no cost (`pipeline_stages.py:539`).
- **Encoding auto-detection**: When the text parser hits a `UnicodeDecodeError` on a non-UTF-8 file, it detects the encoding via `charset-normalizer` and retries with an explicit `encoding=` (`parser.py:89`). LangChain's `autodetect_encoding=True` (which pulls in the extra `chardet` dependency) is intentionally not used.

---

## 9. Evaluation Framework

`evaluation/` — `EvaluationManager` dispatches evaluators via `EVALUATOR_MAPPING`.

- **LangChain evaluators**: correctness / partial_correctness (LLM-based rubric)
- **RAGAS evaluators**: answer_correctness/relevancy, context_precision/recall, faithfulness
- **Graph-aware evaluator** (`graph_aware_evaluator.py`): Computes the rate at which the ground truth's `expected_entities`/`expected_relationships` appear in the generated answer (= coverage = recall) as `ENTITY_COVERAGE`/`RELATIONSHIP_COVERAGE`. Deterministic, no LLM required. Precision/F1 are not produced because they would require enumerating the entities in the answer (impossible from free text) — to avoid exaggerating the signal as a duplicate of recall. Latin characters use word-boundary contiguous token matching ("AI" does not match inside "airport"); CJK without whitespace falls back to substring matching. The manager injects the expectations via `result.metadata`, so the abstract signature is unchanged.

CLI: `run-eval --eval-data-path <json> [--search-strategy ...]`.

---

## 10. Visualization & Analytics

`visualization/` — `BaseRenderer` ABC + `@register_renderer` registry + `RenderContext`.

- `InteractiveRenderer` (pyvis network + community hierarchy), `StaticRenderer` (Bokeh degree/centrality/community-size)
- Layout: Bedrock Node2Vec embeddings + UMAP dimensionality reduction (spring layout on failure)
- **Standalone execution** (`application/cli/run_visualization.py`): Reads exported graph JSON without ingestion (`export_visualization_data` output format: `nodes`/`edges`/`layout`/`communities.hierarchy`), rehydrates it into typed objects, and renders with the registered renderers.

---

## 11. Prompts and Prompt Tuning

- **Prompts** (`prompts/`): Classes based on `BasePrompt` (frozen dataclass). System/human templates are version-controlled as `.py`. Every prompt can be overridden from config via `CustomPromptConfig` (e.g., medical/legal/financial domains).
- **Prompt tuning** (`application/prompts/tuner.py`, ported from MS `prompt_tune`): Corpus sample → profile domain/language/persona/entity-types via a Bedrock LLM (`CorpusProfilePrompt`) → generate a domain-adapted `custom_prompts` YAML fragment. CLI: `run-prompt-tuning`. This is an explicit step where the user reviews and applies it to config, not automatic runtime application.

---

## 12. Configuration System

The nested Pydantic tree in `domain/models/config.py` (root `Config`), loaded by `shared/config.py` (`get_config`); the schema example is `config-template.yaml`.

- Sections: `aws` (bedrock/neptune/opensearch/s3/dynamodb), `fixing`, `processing` (chunking/translation/graph_extraction/gleaning/claim_extraction), `graph` (analysis/community_detection/visualization), `indexing` (opensearch/neptune), `search` (hybrid/fusion/reranking/global_search/drift_search/lightrag_search/token_manager), `memory`, `cache`, `logging`, `evaluation`, `custom_prompts`.
- **Config-based generalization**: The language→analyzer mapping (`language_analyzers`), OpenSearch clause budget (`max_total_clauses`, etc.), LightRAG fallback length, gleaning scale constants, and eigenvector convergence parameters are all exposed via config.
- Adding a new config section: Define a Pydantic `BaseModel` → attach to its parent via `Field(default_factory=...)` → document in `config-template.yaml`.

---

## 13. Testing Strategy

`tests/{unit,integration,property,fixtures/fakes}/` — **AWS-free by default**.

- **Port-based fake adapters** (`fixtures/fakes/`): In-memory implementations of GraphStore/VectorStore/DocStatus verify domain logic without real AWS (the test-side benefit of hexagonal architecture).
- **moto**: Verifies the DynamoDB/S3 adapters against the boto3 surface.
- Layers: unit (models/registry/merge/dual-keyword/evaluation/token-counter/clause budget/lineage relevance), property (hypothesis: hashing determinism, diff partition completeness, merge laws), integration (incremental add/change/delete cycles), regression.
- Markers: `unit`, `integration`, `property`, `aws` (real AWS, excluded in CI), `slow`. `asyncio_mode = "auto"`.

Run: `uv run pytest -m "not aws" --cov=unified_kg_rag`.

---

## 14. CI/CD and Security

- **CI** (`.github/workflows/`): the `quality` workflow (ruff/black/isort/mypy + pytest+coverage gate, triggered on PR/default branch), the `security` workflow (ASH scan, non-blocking, report-only).
- **pre-commit** (`.pre-commit-config.yaml`): Mirrors the CI gates. `pre-commit install`.
- **Security hardening**: Content hashes use SHA-256 exclusively (MD5 removed, resolving CWE-327). Dependencies are refreshed regularly via `uv lock --upgrade` to address dependency-scan CVEs. Tokens are injected via environment/config (no hardcoding in code).

---

## 15. Extension Guide

Most extensions are possible with registry registration alone and require no changes to dispatch code (details in `CONTRIBUTING.md`/`CLAUDE.md`).

- **New search strategy**: Subclass `BaseSearchStrategy` + `@register_strategy(SearchStrategy.X, required_roles=(...))` + export from `adapters/search_strategies/__init__.py`.
- **New storage/LLM backend**: Implement the relevant port (ABC) and bind it in the registry/builder. Do not hardcode it into a manager's `__init__`.
- **New evaluator**: Subclass `BaseGraphRAGEvaluator` + `EVALUATOR_MAPPING` + add an `EvaluatorType` enum.
- **New renderer**: Subclass `BaseRenderer` + `@register_renderer("name")`.

### Known Future Work (Honest Gaps)

> **Items resolved in this milestone**: cache abstraction (`CachePort` definition + S3/local cache manager wiring), query-time claims injection (local/simple, §6.1), LLM re-summarization of merged descriptions (stage 7, §4), client lifecycle teardown (§8.6). Only the two items below are intentionally on hold.

- **LLM/Embedding/Rerank ports — defined, DI not wired (intentional)**: `ports/model_factory.py` defines the `ModelFactoryPort` (+ `LLMFactoryPort`/`EmbeddingFactoryPort`/`RerankFactoryPort` aliases) Protocol, and the Bedrock factories structurally conform to it (verified by tests). About 20 adapter/application modules construct the concrete `Bedrock*ModelFactory` directly, but they are all in the adapters/application layers, so this is **not a dependency-rule violation** (the domain does not import the factories), and the call sites already receive LangChain-compatible objects, so they are provider-agnostic. Threading DI through 20 construction points before a second provider actually exists would be over-abstraction, so it is **intentionally on hold** — when a non-Bedrock provider is needed, write a conforming factory adapter and inject it at the construction points (the port types are ready).
- **`SearchQuery.ln` (label/index prefix)**: Adapter vocabulary (index/label prefix) remains in the domain query model. Since all search strategies and both retrievers (36 references) read and write it, fully delegating it to a backend-neutral abstraction would entail large churn + behavior-equivalence risk. It is reasonable to refactor this when a backend swap actually materializes.
