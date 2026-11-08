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
6. [Retrieval: Methodologies](#6-retrieval-methodologies)
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
methodologies, with HippoRAG 2's Personalized PageRank and RAPTOR's summary
tree, on an AWS-native stack (Bedrock, Neptune, OpenSearch, S3, DynamoDB).
Its design principles:

- **Several methodologies, one infrastructure.** GraphRAG (community
  summaries), LightRAG (dual-level keywords), HippoRAG 2 (Personalized
  PageRank, §6.4) and RAPTOR (a summary tree, opt-in, §6.5) share
  ingestion, indexing, caching, multilingual processing and hybrid search;
  **only the retrieval algorithm differs**, and an ingestion builds only
  what the selected methodologies read (§4.6).
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
retrieval, several methodologies on one index, incremental indexing, multilingual
ingestion, bring-your-own graphs and schemas, measured defaults) and where it
deliberately departs from them is summarized in
[Overview: Beyond the papers](./concepts/overview.md#beyond-the-papers); the
sections below give the design behind each.

![Architecture at a glance](../assets/architecture-overview.drawio.png)

---

## 2. Hexagonal Architecture (Ports & Adapters)

The package's layers, ports, registries and the tests that hold the dependency rule.
→ [architecture.md §2](./design/architecture.md#2-hexagonal-architecture-ports--adapters)

### 2.1 Dependency Rule and Layer Map

Imports point inward: the layers outermost first, what each package holds and which modules may import vendor bindings.
→ [architecture.md §2.1](./design/architecture.md#21-dependency-rule-and-layer-map)

### 2.2 Ports (Abstract Interfaces)

The ports the application wires, with their locations, adapters and test fakes.
→ [architecture.md §2.2](./design/architecture.md#22-ports-abstract-interfaces)

### 2.3 Role-Based Retriever Injection

Strategies receive retrievers by role (graph, document), so swapping a backend changes no strategy code.
→ [architecture.md §2.3](./design/architecture.md#23-role-based-retriever-injection)

### 2.4 Registries

The registries for search strategies, providers, renderers, parsers and the sources of schemas and seed graphs.
→ [architecture.md §2.4](./design/architecture.md#24-registries)

### 2.5 Dependency Rule Verification and Public API

How `test_layering.py` enforces the layer rules, and the root facade that is the public API.
→ [architecture.md §2.5](./design/architecture.md#25-dependency-rule-verification-and-public-api)

---

## 3. Domain Model

The pure Pydantic models of `domain/models/`: entities, relationships, communities, text units, claims and the rest.
→ [architecture.md §3](./design/architecture.md#3-domain-model)

---

## 4. Ingestion Pipeline

The 13 checkpointed stages of `DataIngestionPipeline`, with their modules, and why extraction works per text unit.
→ [ingestion.md §4](./design/ingestion.md#4-ingestion-pipeline)

### 4.1 Extraction records and the extraction store (stages 5-8)

One extraction record per text unit, gleaning that edits one record at a time, and the store the graph is resolved from.
→ [ingestion.md §4.1](./design/ingestion.md#41-extraction-records-and-the-extraction-store-stages-5-8)

### 4.2 Graph identity and merging (stages 5-7)

Entity and relationship identity (deliberate divergences from the papers) and how statements merge into one graph.
→ [ingestion.md §4.2](./design/ingestion.md#42-graph-identity-and-merging-stages-5-7)

### 4.3 Communities (stages 10-11)

The community graph, the Leiden hierarchy and the community reports.
→ [ingestion.md §4.3](./design/ingestion.md#43-communities-stages-10-11)

### 4.4 Pipeline infrastructure

Resume from checkpoints, S3 sync, reuse across runs and the other machinery every stage shares.
→ [ingestion.md §4.4](./design/ingestion.md#44-pipeline-infrastructure)

### 4.5 Seed graphs as virtual documents

A user-supplied knowledge graph imported as virtual documents of the corpus, so lineage and removal apply unchanged.
→ [ingestion.md §4.5](./design/ingestion.md#45-seed-graphs-as-virtual-documents)

### 4.6 Methodology profiles

One ingestion builds what the selected methodologies read, and a query fails fast on a methodology the namespace lacks.
→ [ingestion.md §4.6](./design/ingestion.md#46-methodology-profiles)

---

## 5. Incremental Indexing

Diffing the corpus against the doc-status registry by content hash, so a run processes only what changed.
→ [incremental.md §5](./design/incremental.md#5-incremental-indexing)

### 5.1 The steps

Delta detection, removal, extraction and merge of an incremental run, step by step.
→ [incremental.md §5.1](./design/incremental.md#51-the-steps)

### 5.2 Full rebuild (`indexing.reset`)

Rebuilding the whole corpus without taking the suffix offline.
→ [incremental.md §5.2](./design/incremental.md#52-full-rebuild-indexingreset)

### 5.3 Crash recovery (write-ahead protocol)

The write-ahead records that let the next run repair an interrupted one.
→ [incremental.md §5.3](./design/incremental.md#53-crash-recovery-write-ahead-protocol)

### 5.4 Identity, derived content and merge rules

Document identity, derived content and the merge rules an incremental run relies on.
→ [incremental.md §5.4](./design/incremental.md#54-identity-derived-content-and-merge-rules)

### 5.5 Communities on delta runs

Clustering only the delta subgraph, with content-derived community ids.
→ [incremental.md §5.5](./design/incremental.md#55-communities-on-delta-runs)

### 5.6 Concurrency

The namespace write lease and the write fence that keep two runs apart.
→ [incremental.md §5.6](./design/incremental.md#56-concurrency)

### 5.7 Document changes from memory

`insert_documents` and `delete_documents`: named changes through the incremental path without a corpus directory.
→ [incremental.md §5.7](./design/incremental.md#57-document-changes-from-memory)

### 5.8 Inspection and export

Reading a namespace's indices and records, and exporting its graph.
→ [incremental.md §5.8](./design/incremental.md#58-inspection-and-export)

---

<a id="6-retrieval-two-methodologies"></a>

## 6. Retrieval: Methodologies

The one chain every strategy runs, and how `RAGInput.search_strategy` selects the methodology.
→ [retrieval-graphrag-lightrag.md §6](./design/retrieval-graphrag-lightrag.md#6-retrieval-methodologies)

### 6.1 GraphRAG Methodology (`adapters/search_strategies/`)

GraphRAG's `simple`, `local`, `global`, `drift` and `auto` strategies.
→ [retrieval-graphrag-lightrag.md §6.1](./design/retrieval-graphrag-lightrag.md#61-graphrag-methodology-adapterssearch_strategies)

#### 6.1.1 Global search (`global_search.py`)

Which community reports global search maps, and how it reduces their answers.
→ [retrieval-graphrag-lightrag.md §6.1.1](./design/retrieval-graphrag-lightrag.md#611-global-search-global_searchpy)

### 6.2 LightRAG Methodology (`lightrag_search.py`)

LightRAG's dual-level keyword retrieval: `naive`, `hybrid` and `mix`.
→ [retrieval-graphrag-lightrag.md §6.2](./design/retrieval-graphrag-lightrag.md#62-lightrag-methodology-lightrag_searchpy)

### 6.3 Iterative bridge follow-up (`iterative_search.py`, opt-in)

The `iterative` wrapper, which follows a bridge entity for multi-hop questions.
→ [retrieval-graphrag-lightrag.md §6.3](./design/retrieval-graphrag-lightrag.md#63-iterative-bridge-follow-up-iterative_searchpy-opt-in)

### 6.4 HippoRAG 2 Personalized PageRank (`ppr`, `local` ranking)

The port of HippoRAG 2's Personalized PageRank, as the `ppr` strategy and as `local`'s text-unit ranking.
→ [retrieval-hipporag-raptor.md §6.4](./design/retrieval-hipporag-raptor.md#64-hipporag-2-personalized-pagerank-ppr-local-ranking)

### 6.5 RAPTOR summary tree (`raptor`, `tree`, opt-in)

The port of RAPTOR's summary tree: the `raptor` methodology builds it and the `tree` strategy searches it.
→ [retrieval-hipporag-raptor.md §6.5](./design/retrieval-hipporag-raptor.md#65-raptor-summary-tree-raptor-tree-opt-in)

### 6.6 Structured graph queries (`structured`, opt-in)

The `structured` strategy: a planned query against the graph store for counting, ranking, listing and path questions.
→ [structured-queries.md §6.6](./design/structured-queries.md#66-structured-graph-queries-structured-opt-in)

---

## 7. Hybrid Scoring and Token Management

Fusion, reranking and diversity in `HybridScorer`, and the token budget of the context.
→ [scoring-and-citations.md §7](./design/scoring-and-citations.md#7-hybrid-scoring-and-token-management)

### 7.1 Document citations

How an answer cites the documents it rests on, per strategy.
→ [scoring-and-citations.md §7.1](./design/scoring-and-citations.md#71-document-citations)

### 7.2 Caller-supplied conversation history

`RAGInput.history`: the conversation a stateless caller keeps itself.
→ [scoring-and-citations.md §7.2](./design/scoring-and-citations.md#72-caller-supplied-conversation-history)

---

## 8. AWS Service Integration

Bedrock, Neptune, OpenSearch, S3 and DynamoDB: the modules that bind each and what they do.
→ [aws-integration.md §8](./design/aws-integration.md#8-aws-service-integration)

### 8.1 Retrieval Error Visibility

Fatal errors fail the query and the rest degrade one section: never disguised as no results.
→ [aws-integration.md §8.1](./design/aws-integration.md#81-retrieval-error-visibility)

### 8.2 Client Lifecycle / Resource Release

The best-effort `close()` / `aclose()` of every layer that holds connections or pools.
→ [aws-integration.md §8.2](./design/aws-integration.md#82-client-lifecycle--resource-release)

### 8.3 Multilingual Processing

Per-language analyzers, translation and the other multilingual pieces.
→ [aws-integration.md §8.3](./design/aws-integration.md#83-multilingual-processing)

### 8.4 Agent access: the read-only MCP server

`run-mcp-server`: one index namespace served to agents as read-only Model Context Protocol tools.
→ [surfaces.md §8.4](./design/surfaces.md#84-agent-access-the-read-only-mcp-server)

### 8.5 OpenSearch engine version

Why the deployment runs OpenSearch 2.19 and how the application stays version-agnostic within 2.x.
→ [aws-integration.md §8.5](./design/aws-integration.md#85-opensearch-engine-version)

---

## 9. Evaluation Framework

How `EvaluationManager` builds and runs the evaluators, and the design choices behind them.
→ [evaluation.md §9](./design/evaluation.md#9-evaluation-framework)

### 9.1 Measured defaults

Where the defaults measured on real models are documented.
→ [evaluation.md §9.1](./design/evaluation.md#91-measured-defaults)

### 9.2 Unanswerable questions and abstention (opt-in)

The unanswerable benchmark split and the decline marker that measure abstention.
→ [evaluation.md §9.2](./design/evaluation.md#92-unanswerable-questions-and-abstention-opt-in)

### 9.3 Length-controlled judging, assertions and the closed-book arm

Four opt-in additions that measure around the pairwise judge's length bias and question-set bias.
→ [evaluation.md §9.3](./design/evaluation.md#93-length-controlled-judging-assertions-and-the-closed-book-arm)

### 9.4 Structural benchmark (the `structured` strategy)

A generated procurement world that tests the `structured` strategy.
→ [evaluation.md §9.4](./design/evaluation.md#94-structural-benchmark-the-structured-strategy)

---

## 10. Visualization & Analytics

`run-visualization` renders the exported graph data through the registered renderers.
→ [surfaces.md §10](./design/surfaces.md#10-visualization--analytics)

### 10.1 The console (`run-console`, opt-in)

The local web console over the same entry points the CLIs use.
→ [surfaces.md §10.1](./design/surfaces.md#101-the-console-run-console-opt-in)

---

## 11. Prompts and Prompt Tuning

Prompts as version-controlled classes, their overrides and prompt tuning.
→ [surfaces.md §11](./design/surfaces.md#11-prompts-and-prompt-tuning)

---

## 12. Configuration System

The one Pydantic configuration tree, how it loads and the references generated from it.
→ [architecture.md §12](./design/architecture.md#12-configuration-system)

---

## 13. Testing Strategy

The AWS-free test suite: port-based fakes, property tests and the integration jobs.
→ [architecture.md §13](./design/architecture.md#13-testing-strategy)

---

## 14. CI/CD and Security

The CI workflows, Dependabot and the security scans.
→ [architecture.md §14](./design/architecture.md#14-cicd-and-security)

---

## 15. Extension Guide

The recipes for plugins (backends, renderers, parsers, schema and seed sources) and in-tree changes (strategies, evaluators).
→ [extend.md §15](./guides/extend.md#15-extension-guide)

### Custom backends (run without AWS)

The ports a custom backend implements and how to inject it.
→ [extend.md](./guides/extend.md#custom-backends-run-without-aws)

### Deliberate Design Boundaries

Two intentional boundaries and why they stay.
→ [extend.md](./guides/extend.md#deliberate-design-boundaries)

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
