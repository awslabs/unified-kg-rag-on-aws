# Overview

> 🇰🇷 한국어판: [docs/concepts/overview.ko.md](./overview.ko.md)

`unified-kg-rag-on-aws` turns large, multilingual document corpora into a
knowledge graph on Amazon Neptune and Amazon OpenSearch Service and answers
questions over it with Amazon Bedrock. It implements two graph RAG
methodologies, Microsoft GraphRAG (community summaries) and
LightRAG (dual-level keywords), on one shared stack, so you choose the
methodology per query rather than per deployment. Every answer returns the
sources that were actually in the model's context.

This page explains the parts and how they fit. To run it, start from the
[README](../../README.md#quickstart) or the [documentation hub](../index.md).

## Architecture at a glance

![Architecture at a glance](../../assets/architecture-overview.drawio.png)

- **Ingestion** (`run-ingestion`) parses a corpus, extracts a knowledge
  graph with an LLM, clusters it into communities with LLM-written reports
  and writes everything to the graph and search stores.
- **Stores.** On AWS: Neptune for the graph, OpenSearch for BM25 and vector
  search, DynamoDB for the doc-status registry that makes ingestion
  incremental, and S3 for the stage cache. In local mode: embedded SQLite
  stores, or a Gremlin Server and OpenSearch in containers.
- **Models.** Bedrock for chat, embedding and rerank models by default;
  Ollama or any OpenAI-compatible server in local mode. Every LLM role runs
  on one of two tiers, so one setting switches the pipeline.
- **Retrieval** (`run-rag`, or the Python chain) runs one of eight
  strategies over the same stores and the same hybrid scorer, then packs the
  context into the answer model's token budget.
- **Evaluation** (`run-eval`) scores answers with LLM-free metrics and LLM
  judges, compares strategies or runs pairwise, and meters each query's
  tokens, cost and latency ([Evaluation](../guides/evaluate.md)).

The code is hexagonal (ports and adapters): the algorithms depend on
interfaces, and the AWS and local backends sit behind them, so a store, a
model provider, a strategy, a parser or a renderer is added through a
registry rather than by editing dispatch code
([Design Doc §2](../design.md#2-hexagonal-architecture-ports--adapters),
[§15 Extension Guide](../design.md#15-extension-guide)).

## Pipeline

Ingestion is a resumable pipeline of 12 checkpointed stages; a re-run
resumes after the last valid one:

1. Parse the documents (PDF, TXT, CSV, JSON; Markdown and HTML with the
   `unstructured` extra) and load the run's corpus.
2. Chunk the text into text units, and optionally translate them.
3. Extract entities and relationships per text unit, glean what the first
   pass missed, and resolve name variants into one entity.
4. Optionally extract and resolve factual claims.
5. Build the graph, detect a hierarchy of Leiden communities and write a
   structured report for each.
6. Index: entities, relationships, text units, claims and reports into
   OpenSearch (BM25 and vectors), the graph into Neptune, and each
   document's state into the registry.

With the doc-status registry on, a run re-indexes only new and changed
documents and removes deleted ones. Stage list and flags:
[Ingestion](../guides/ingest.md#the-12-pipeline-stages); internals:
[Design Doc §4](../design.md#4-ingestion-pipeline) and
[§5](../design.md#5-incremental-indexing).

## Two methodologies

Both methodologies read the same index; only the retrieval algorithm
differs.

| | GraphRAG (community summaries) | LightRAG (dual-level keywords) |
|---|---|---|
| Strategies | `local`, `global`, `drift`, `simple`, and the `auto` router | `mix`, `hybrid`, `naive` |
| Reads | Entities and their graph neighbourhood, community reports, text units, claims | Entities (low-level keywords), relationships (high-level keywords), their source chunks |
| Strong at | Thematic and sensemaking questions over the whole corpus (`global`, `drift`), and entity questions (`local`) | Keyword-driven lookups and a fast, low-cost baseline |

Every strategy goes through the same hybrid scorer: BM25, vector search and
(except `simple` and `naive`) graph traversal, fused by Reciprocal Rank
Fusion, reranked and diversified.
`auto`, the default, lets an LLM router pick `local`, `mix` or `global` per
query. When to use which:
[Retrieval](./retrieval.md#search-strategies--when-to-use-which); how
each one retrieves: [Design Doc §6](../design.md#6-retrieval-two-methodologies).

## Beyond the papers

The framework keeps each methodology's retrieval algorithm and adds what a
production corpus on AWS needs around it. The upstream columns describe the
reference libraries, [microsoft/graphrag](https://github.com/microsoft/graphrag)
3.1.0 and [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) 1.5.4, as read
in their source. A file named in a cell is where to check it, relative to
the library's Python package (`graphrag/`, `lightrag/`) unless it starts at
the repository root (`docs/`, `reproduce/`).

| Area | GraphRAG | LightRAG | This framework | Where documented |
|---|---|---|---|---|
| Triple-hybrid retrieval | Vector search over entity descriptions, text units and report content (`config/embeddings.py`), with graph context read from the indexed tables; no lexical search or reranker in the query code | Vector search over entity, relationship and chunk indexes plus the graph storage, merged round-robin, with an optional rerank model (`operate.py`, `rerank.py`) | BM25 (OpenSearch), vector search (Bedrock embeddings) and Neptune graph traversal fused by RRF, reranked by a Bedrock rerank model and diversified by MMR; one scorer for every strategy of both methodologies (`simple` and `naive` skip the graph) | [Design §7](../design.md#7-hybrid-scoring-and-token-management) |
| Both methodologies on one index | Community reports; no relationship vector index (`config/embeddings.py`) and no keyword-level retrieval | Dual-level keywords over entity and relationship indexes; no community detection or reports | One ingestion builds the union of what both need; the methodology is chosen per query, or by the `auto` router | [Design §6](../design.md#6-retrieval-two-methodologies) |
| Incremental indexing with deletion lineage | `graphrag update` indexes the documents whose title is new; the delta it computes lists deleted documents, but only the new ones reach the update workflows (`index/update/incremental_index.py`, `index/workflows/load_update_documents.py`) | A doc-status store skips an insert whose file name or content hash is already registered (`pipeline.py`); `adelete_by_doc_id` removes a document and rebuilds the entities and relationships other documents still support from cached extractions (`lightrag.py`) | A registry (DynamoDB, or SQLite locally) diffs the corpus by content hash, re-indexes new and changed documents, and removes a deleted one by the text units every item cites, so only what no surviving text unit states goes; write-ahead crash recovery and one writing run per namespace; property tests check that incremental rounds converge with a full build | [Incremental indexing](../guides/incremental.md), [Design §5](../design.md#5-incremental-indexing) |
| Multilingual | `graphrag prompt-tune --language` sets the language of the tuned prompts (`cli/main.py`) | `SUMMARY_LANGUAGE` sets the output language of extraction and summaries (`addon_params.py`) | An optional translation stage at indexing and query time (with extra target languages), per-language OpenSearch analyzers (`nori` for Korean), multilingual keyword extraction and Korean- and CJK-aware answer matching, for both methodologies | [Multilingual corpora](../guides/multilingual.md), [Design §8.3](../design.md#83-multilingual-processing) |
| Bring-your-own ontology and seed graphs | Bring Your Own Graph: supply entity and relationship tables in place of extraction, then run the community workflows (`docs/index/byog.md`) | `insert_custom_kg` inserts given chunks, entities and relationships (`lightrag.py`) | An extraction schema declared or imported from an OWL/RDFS ontology (types, patterns, IRIs); seed graphs (JSONL, CSV, RDF) merged into the extracted graph as virtual documents of the corpus, so incremental runs and removal treat them like files and the documents stay canonical | [Extraction schema](../guides/extraction-schema.md), [ontology](../guides/ontology.md), [seed graphs](../guides/seed-graphs.md) |
| Measured defaults and evaluation tooling | The paper judges answers pairwise (comprehensiveness, diversity, empowerment, directness); the library package has no evaluation command | A RAGAS evaluation script (`evaluation/eval_rag_quality.py`) and the paper's pairwise judge prompt (`reproduce/batch_eval.py`) | `run-eval` with LLM-free evaluators beside LangChain and RAGAS judges, paired comparisons with bootstrap intervals and repeats, an order-swapped pairwise judge with position consistency, metered cost and latency, CI gates and a MuSiQue benchmark; defaults backed by published measurements | [Evaluation](../guides/evaluate.md), [Defaults and evidence](./defaults-and-evidence.md) |

Neither library reads OWL or RDF, and neither has a document-translation
stage in its indexing pipeline, as far as their source shows.

### Deliberate divergences

Where the framework departs from an upstream default or algorithm on
purpose:

- **Relationship identity.** A relationship's type is part of its id and
  the edge is directed, so two types between one pair are two edges. MS
  GraphRAG groups relationships by source and target (`extract_graph.py`),
  and LightRAG keys an edge by its sorted endpoint pair (`operate.py`)
  ([Design §4.2](../design.md#42-graph-identity-and-merging-stages-5-7)).
- **Community hierarchy.** Level 0 is one Leiden partition of the whole
  graph, coarser levels partition the graph of communities, and a community
  over 20 members is split again; only components smaller than
  `min_community_size` get no community. MS GraphRAG runs hierarchical
  Leiden with `max_cluster_size` 10 and, by default (`use_lcc`), clusters
  only the largest connected component (`config/defaults.py`)
  ([Design §4.3](../design.md#43-communities-stages-10-11)).
- **Gleaning.** Three rounds by default against MS GraphRAG's one
  (`max_gleanings`), and a text unit is re-asked only while its last answer
  added something: measured for graph recall
  ([evidence](./defaults-and-evidence.md#gleaning-rounds)).
- **Context shares.** The token budget is split into per-type shares of
  `max_context_tokens` (30,000, LightRAG's total). LightRAG caps entities at
  6,000 and relationships at 8,000 tokens (`constants.py`); here `mix` and
  `hybrid` give text 18,072, entities 7,228 and relationships 4,698
  ([Design §7](../design.md#7-hybrid-scoring-and-token-management)).
- **LightRAG chunk picking.** `kg_chunk_pick_method` defaults to `weight`;
  upstream LightRAG defaults to `VECTOR` (`constants.py`). The two measured
  no different ([evidence](./defaults-and-evidence.md#lightrag-chunk-picking-and-mmr-diversity)).
- **Local search reports.** `local` reads the reports most similar to the
  query entities; MS GraphRAG local reads those of the entities' communities
  (`mixed_context.py`), which is `report_selection: membership` here and
  measured worse ([evidence](./defaults-and-evidence.md#local-report-selection)).
- **Global answers.** The answer model reads the ranked key points and up
  to 4,000 tokens of the reports they cite; MS GraphRAG's reduce step reads
  the key points (`global_search/search.py`), which is
  `evidence_tokens: 0` here
  ([evidence](./defaults-and-evidence.md#global-evidence-cap)).
- **Strategy names.** `local` and `global` are GraphRAG's strategies.
  LightRAG's own `local` and `global` modes (`base.py`) are not offered
  separately; `hybrid` combines both keyword levels, as it does upstream.

## Limitations

The main ones, in short:

- Extraction can hallucinate entities or relationships, and LLM outputs make
  runs non-reproducible; compare configurations over several runs.
- There is no serving layer (no HTTP API, end-user authentication or rate
  limiting), and filters are not access control: separate tenants by index
  suffix.
- Incremental indexing approximates a full build for communities until a
  periodic rebuild, and concurrent runs need the registry's lease.
- A lighter ingestion (no communities, no relationship index) limits the
  strategies that can run.

The full list: [User Guide §12](../user-guide.md#12-limitations).
