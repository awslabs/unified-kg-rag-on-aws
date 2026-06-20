# Architecture — Hexagonal (Ports & Adapters)

`aws-graphrag` is organized in explicit hexagonal layers. The **dependency
rule** is that imports point inward:

```
application  ──►  adapters  ──►  ports  ──►  domain
        └──────────────┴────────────┴──────────►  shared (cross-cutting kernel)
```

A layer may import from layers to its right, never to its left. `shared/` is a
cross-cutting kernel (config, logging, exceptions, metrics, managers, utils) any
layer may use. The two RAG methodologies (GraphRAG community-summary, LightRAG
dual-level keyword) share one ingestion/indexing/caching/hybrid-search
infrastructure and differ only at the algorithm layer.

## Layer map

```
aws_graphrag/
├─ domain/              # technology-agnostic core (no boto3/LangChain/backends)
│  ├─ models/           #   Pydantic domain models
│  ├─ ingestion/        #   pure algorithms: delta_detector, incremental,
│  │  └─ merge/         #   graph_analyzer/builder/resolver, claim_resolver, merge
│  ├─ retrieval/        #   strategy_registry, MetricsMixin
│  └─ prompts/          #   versioned prompt templates
│
├─ ports/               # abstract interfaces the domain depends on
│  ├─ doc_status.py     #   DocStatusPort (Protocol)
│  ├─ indexer.py        #   BaseIndexer / GraphIndexer / VectorIndexer, IndexingStats
│  └─ __init__.py       #   the port catalog (also re-exports adapter-bases)
│
├─ adapters/            # concrete technology bindings
│  ├─ aws/              #   Bedrock, Neptune, OpenSearch, DynamoDB, S3 clients
│  ├─ storage/          #   Neptune / OpenSearch indexers (write-side ports impl)
│  ├─ retrievers/       #   Neptune / OpenSearch retrievers
│  ├─ search_strategies/#   simple/local/global/drift + lightrag (mix/hybrid/naive)
│  ├─ retrieval/        #   abstract retriever/strategy bases, hybrid scorer,
│  │                    #   token & memory managers (construct infra in __init__)
│  ├─ ingestion/        #   LLM/IO-coupled: chunker, *_extractor, loader, parser,
│  │                    #   translator, gleaner, community_detector
│  ├─ renderers/        #   graph visualization renderers
│  └─ evaluators/       #   langchain / ragas evaluator wrappers
│
├─ application/         # orchestration + entry points
│  ├─ cli/              #   run-ingestion / run-rag / run-eval / run-visualization
│  │                    #   / run-prompt-tuning  (pyproject scripts)
│  ├─ ingestion/        #   DataIngestionPipeline + pipeline_stages
│  ├─ storage/          #   IndexingManager (fan-out across indexers)
│  ├─ retrieval/        #   rag_chain (GraphRAGChain, RAGInput/Output)
│  └─ prompts/          #   PromptTuner (LLM-driven corpus profiling)
│
├─ shared/              # cross-cutting kernel
│  ├─ config.py, logging.py, exceptions.py, metrics.py
│  ├─ cache_manager.py, pipeline_manager.py
│  └─ utils/            #   common, display, document_converter, langchain helpers
│
└─ (facades)            # thin re-export shims keeping public import paths stable:
   retrieval/  storage/  ingestion/  evaluation/  visualization/
```

## Ports vs adapter-bases

Two contracts are *abstract adapter bases* rather than pure ports: they
construct infrastructure in `__init__` (`HybridScorer`/`TokenManager`, `tqdm`).
They live beside their adapters (`adapters/retrieval/base.py`,
`evaluation/base.py`) and are re-exported from `ports/__init__` only for
discoverability — `ports/` itself stays free of infra imports.

## Extending (registries over dispatch)

- **New search strategy**: subclass `BaseSearchStrategy`, decorate with
  `@register_strategy(SearchStrategy.X, required_roles=(...))`, export from
  `adapters/search_strategies/__init__.py`. No `rag_chain` edits.
- **New storage / LLM backend**: implement the port; register it in the
  corresponding registry — never hardcode into a manager's `__init__`.
- **New evaluator / renderer**: subclass the base + add the registry entry
  (`EVALUATOR_MAPPING` / `@register_renderer`).

## Dependency-rule status

Verified with grep that `domain/` imports no `adapters`/`application` modules at
runtime, and `ports/` imports neither. One **compile-time-only** exception
remains: `domain/retrieval/strategy_registry.py` references
`adapters.retrieval.base.BaseSearchStrategy` under `TYPE_CHECKING` (the registry
stores strategy subclasses). Extracting a pure strategy/retriever port would
remove even this type-level reference — tracked in `docs/tech-doc.md`.

## Notes / known gaps

The legacy top-level packages `retrieval/`, `storage/`, `ingestion/`,
`evaluation/`, `visualization/` are intentionally retained as thin facade
`__init__` modules so existing import paths and the public API stay stable after
the layer split. See the "known future work" section of `docs/tech-doc.md` for
the remaining boundary cleanups (e.g. LLM/Embedding port extraction).
