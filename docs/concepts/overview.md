# Overview

> 🇰🇷 한국어판: [docs/concepts/overview.ko.md](./overview.ko.md)

`unified-kg-rag-on-aws` turns large, multilingual document corpora into a
knowledge graph on Amazon Neptune and Amazon OpenSearch Service and answers
questions over it with Amazon Bedrock. It implements four graph RAG
methodologies on one shared stack: Microsoft GraphRAG (community
summaries), LightRAG (dual-level keywords), HippoRAG 2 (Personalized
PageRank over entities and passages) and, opt-in, RAPTOR (a summary tree).
One ingestion builds GraphRAG's and LightRAG's indexes (which also serve
HippoRAG 2) and you choose the methodology per query, or it builds only the
methodologies you select and skips the cost of the others. Every answer
returns the sources that were actually in the model's context.

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
- **Retrieval** (`run-rag`, or the Python chain) runs one of eleven
  strategies, or the `auto` router, over the same stores, then packs the
  context into the answer model's token budget.
- **Evaluation** (`run-eval`) scores answers with LLM-free metrics and LLM
  judges, compares strategies or runs pairwise, and meters each query's
  tokens, cost and latency ([Evaluation](../guides/evaluate.md)).

The code is hexagonal (ports and adapters): the algorithms depend on
interfaces, and the AWS and local backends sit behind them, so a store, a
model provider, a strategy, a parser or a renderer is added through a
registry rather than by editing dispatch code
([Design Doc §2](../design/architecture.md#2-hexagonal-architecture-ports--adapters),
[§15 Extension Guide](../guides/extend.md#15-extension-guide)).

## Pipeline

Ingestion is a resumable pipeline of 13 checkpointed stages; a re-run
resumes after the last valid one:

1. Parse the documents (PDF, TXT, CSV, JSON, Markdown, HTML; Word with the
   `docx` extra) and load the run's corpus.
2. Chunk the text into text units, and optionally translate them.
3. Extract entities and relationships per text unit, glean what the first
   pass missed, and resolve name variants into one entity.
4. Optionally extract and resolve factual claims.
5. Build the graph, detect a hierarchy of Leiden communities and write a
   structured report for each.
6. Index: entities, relationships, text units, claims and reports into
   OpenSearch (BM25 and vectors), the graph into Neptune, and each
   document's state into the registry.
7. For the opt-in `raptor` methodology, build a summary tree over the stored
   text units (RAPTOR): clustered, summarized level by level, searched by
   `tree`.

With the doc-status registry on, a run re-indexes only new and changed
documents and removes deleted ones. Stage list and flags:
[Ingestion](../guides/ingest.md#pipeline-stages); internals:
[Design Doc §4](../design/ingestion.md#4-ingestion-pipeline) and
[§5](../design/incremental.md#5-incremental-indexing).

## Methodologies

The methodologies read the same index; only the retrieval algorithm
differs.

| | GraphRAG (community summaries) | LightRAG (dual-level keywords) | HippoRAG 2 (Personalized PageRank) | RAPTOR (summary tree, opt-in) |
|---|---|---|---|---|
| Strategies | `local`, `global`, `drift`, `simple` | `mix`, `hybrid`, `naive` | `ppr` | `tree` |
| Reads | Entities and their graph neighbourhood, community reports, text units, claims | Entities (low-level keywords), relationships (high-level keywords), their source chunks | Relationship hits, passage hits and the graph between their entities and passages: what LightRAG builds | Text units and every level of summaries over them |
| Strong at | Thematic and sensemaking questions over the whole corpus (`global`, `drift`), and entity questions (`local`) | Keyword-driven lookups and a fast, low-cost baseline | Multi-hop questions at the lowest cost: one LLM call a query | A cheap index without graph extraction; on multi-hop questions it answered better than `naive`, below the graph strategies ([evidence](./defaults-and-evidence.md#raptor-summary-tree)) |

`iterative` wraps `mix` or `local` with bounded follow-up rounds on the
bridge entities a multi-hop question depends on, and reads what its base
reads. `local` orders its text units by HippoRAG 2's PageRank by default.

Every strategy but `global` fuses its candidates through the same hybrid
scorer: BM25, vector search and graph signals (`simple` and `naive` read
no graph), fused by Reciprocal Rank Fusion, reranked and diversified.
`global` maps community reports into key points instead (it fuses text
units only with `search.global_search.text_unit_slots`).
`auto`, the default, lets an LLM router pick `local`, `mix`, `global` or
`iterative` (for multi-hop questions) per query. When to use which:
[Retrieval](./retrieval.md#search-strategies--when-to-use-which); how
each one retrieves: [Design Doc §6](../design/retrieval-graphrag-lightrag.md#6-retrieval-methodologies).

### Methodology profiles

`indexing.methodologies` selects what ingestion builds. The default,
`[graphrag, lightrag]`, builds both on one index; select one to pay only for
what its strategies read:

| Profile | Builds | Strategies | Ingestion model calls |
|---|---|---|---|
| `[graphrag, lightrag]` (default) | Text units, entities, relationships and their vectors, the graph, communities and their reports | Every strategy but `tree` | Chunk boundaries, extraction and gleaning per text unit, description summaries, one report per community |
| `[lightrag]` or `[hipporag]` | The same without communities and reports | `mix`, `hybrid`, `naive`, `simple`, `ppr`, `iterative` (over `mix`), `auto` | The same without the Leiden pass and the report calls |
| `[graphrag]` | The default's: the relationship vectors `local` reads also serve `lightrag` and `hipporag` | Every strategy but `tree`; with `indexing.opensearch.build_relationship_vector_index: false`, `local`, `global`, `drift`, `simple`, `naive`, `iterative` (over `local`: `search.iterative_search.base: local`), `auto` | The default's; turning the relationship vector index off skips the relationship embeddings, at the cost of `local`'s relationship section and every `lightrag` and `hipporag` strategy |
| `[raptor]` (opt-in) | Text units and a summary tree over them (RAPTOR) | `tree`, `naive`, `simple`, `auto` (which uses `tree`) | Chunk boundaries and about one summary per five or six text units: no graph extraction ([summary tree](../guides/ingest.md#a-raptor-summary-tree-opt-in)) |

Add `raptor` to any other profile to build the tree beside it.
`hipporag` names what `lightrag` builds, for the strategy that reads it
(`ppr`), and `iterative` reads what its base (`local` or `mix`) reads.
`naive` reads the text units alone and `simple` the indexes the namespace
holds, so every profile serves them.
The index namespace records what it holds: a query whose strategy reads a
methodology it does not hold fails before any model call, and `auto` routes
only among the held ones. What adding or removing a methodology later
rebuilds, reuses and costs:
[Ingestion: Methodology profiles](../guides/ingest.md#methodology-profiles)
(internals: [Design §4.6](../design/ingestion.md#46-methodology-profiles)).

## Compared with Amazon Bedrock Knowledge Bases GraphRAG

[Amazon Bedrock Knowledge Bases GraphRAG](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-build-graphs.html)
is the managed way to run graph RAG on AWS: it builds and maintains the graph
in Amazon Neptune Analytics from documents in Amazon S3, with no graph
infrastructure to run. This framework is code you run and operate in your
account, in exchange for control over how the graph is built and searched.
The managed column restates the Amazon Bedrock documentation (read
2026-11-07); check the linked pages for changes.

| | Amazon Bedrock Knowledge Bases GraphRAG | This framework |
|---|---|---|
| Operation | Fully managed: the knowledge base creates and updates the graph by extracting entities, facts and relationships ([build a knowledge base with Neptune Analytics graphs](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-build-graphs-build.html)) | You run the CLIs or the Python library, on your own Neptune cluster and OpenSearch domain or the bundled CDK stack; the MCP server's HTTP transport and the local console authenticate no caller, so no authenticated serving layer is included ([Limitations](#limitations)) |
| Stores | Neptune Analytics for graph and vector storage; autoscaling is not supported for Neptune Analytics graphs ([considerations and limitations](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-build-graphs.html)) | Neptune (graph) and OpenSearch (BM25 and vectors), DynamoDB for the document-status registry; embedded SQLite stores in local mode |
| Retrieval | One method: a vector search, then the graph nodes linked to the retrieved chunks, expanded by graph traversal ([how GraphRAG works](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-build-graphs.html)) | Four methods (GraphRAG, LightRAG, HippoRAG 2, RAPTOR), `iterative` and the opt-in `structured` graph queries, chosen per query or by the `auto` router ([Retrieval](./retrieval.md)) |
| Keyword search | Hybrid (vector and raw-text) search is supported only for Amazon RDS, OpenSearch Serverless and MongoDB vector stores with a filterable text field; other stores use semantic search ([search type](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-config.html)) | BM25, vector search and graph signals fused by RRF and reranked, for every strategy but `global` |
| Graph build control | Configuration options to customize the graph build are not supported; you choose the graph construction model | Extraction schema or OWL/RDFS ontology, domain-tuned prompts, seed graphs of a graph you already have, and the methodologies to build ([Extraction schema](../guides/extraction-schema.md), [Seed graphs](../guides/seed-graphs.md)) |
| Data sources | Amazon S3 only, up to 1,000 files per data source (a limit increase raises it to 10,000) | Files in a directory (the CDK ingestion pipeline copies its corpus from S3): PDF, TXT, CSV, JSON, Markdown and HTML in a core install (the `unstructured` extra parses Markdown and HTML more richly), and Word (`.docx`) with the `docx` extra |
| Updates | Syncing is incremental: added, modified and deleted documents are re-ingested or removed ([sync your data](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-data-source-sync-ingest.html)) | Incremental indexing by content hash; a deleted document removes only what no surviving text unit states ([Incremental indexing](../guides/incremental.md)) |
| Sources in answers | `RetrieveAndGenerate` returns citations: response segments with their retrieved references ([Citation](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Citation.html)) | Every answer returns the sources that were in the model's context, and cites them with `[n]` markers naming their documents and pages, except `global` and `drift` answers by default ([Citations](../guides/query.md#citations)) |
| Measurement | — | `run-eval` with LLM-free metrics, LLM judges and paired comparisons; the defaults are backed by published measurements ([Defaults and evidence](./defaults-and-evidence.md)) |
| Regions | Seven Regions (US East (N. Virginia), US West (Oregon), Europe (Frankfurt, London, Ireland), Asia Pacific (Tokyo, Singapore)) | Any Region with the Bedrock models you configure, Neptune and OpenSearch, or a Bedrock Region other than the stores' ([Required models](../getting-started/aws.md#required-models)) |

Choose the managed service when an S3 corpus, one retrieval method and no
infrastructure to run fit the need; choose this framework when you need to
pick or compare methods, keyword search beside vectors, control over what
the graph extracts, or a corpus that is not in S3.

## Beyond the papers

The framework keeps each methodology's retrieval algorithm and adds what a
production corpus on AWS needs around it. The upstream columns describe the
reference libraries, [microsoft/graphrag](https://github.com/microsoft/graphrag)
3.1.0, [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) 1.5.4,
[OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) 2.0.0a5
and [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor) (its
main branch, read 2026-11-08), as read in their source. A file named in a
cell is where to check it, relative to the library's Python package
(`graphrag/`, `lightrag/`, `src/hipporag/`, `raptor/`) unless it starts at
the repository root (`docs/`, `reproduce/`).

| Area | GraphRAG | LightRAG | HippoRAG 2 | RAPTOR | This framework | Where documented |
|---|---|---|---|---|---|---|
| Triple-hybrid retrieval | Vector search over entity descriptions, text units and report content (`config/embeddings.py`), with graph context read from the indexed tables; no lexical search or reranker in the query code | Vector search over entity, relationship and chunk indexes plus the graph storage, merged round-robin, with an optional rerank model (`operate.py`, `rerank.py`) | Dense retrieval over passage, entity and fact embeddings, an LLM filter of the facts, then Personalized PageRank over the graph (`HippoRAG.py`, `rerank.py`); no lexical search | Embedding distance over the tree's nodes, every level at once (collapsed tree) or level by level (`tree_retriever.py`); no lexical search or reranker | BM25 (OpenSearch), vector search (Bedrock embeddings) and Neptune graph traversal fused by RRF, reranked by a Bedrock rerank model and diversified by MMR; one scorer for every strategy but `global`, which maps community reports (`simple` and `naive` skip the graph) | [Design §7](../design/scoring-and-citations.md#7-hybrid-scoring-and-token-management) |
| Several methodologies on one index | Community reports; no relationship vector index (`config/embeddings.py`) and no keyword-level retrieval | Dual-level keywords over entity and relationship indexes; no community detection or reports | An open-IE graph of passages, entities and facts with synonym edges; no communities or keyword levels | A summary tree over the chunks; no graph | One ingestion builds the union of what the selected methodologies need (GraphRAG's and LightRAG's by default, which serve HippoRAG 2's `ppr` too); the methodology is chosen per query, or by the `auto` router, and a namespace refuses one it does not hold | [Methodology profiles](#methodology-profiles), [Design §6](../design/retrieval-graphrag-lightrag.md#6-retrieval-methodologies) |
| Incremental indexing with deletion lineage | `graphrag update` indexes the documents whose title is new; the delta it computes lists deleted documents, but only the new ones reach the update workflows (`index/update/incremental_index.py`, `index/workflows/load_update_documents.py`) | A doc-status store skips an insert whose file name or content hash is already registered (`pipeline.py`); `adelete_by_doc_id` removes a document and rebuilds the entities and relationships other documents still support from cached extractions (`lightrag.py`) | `index` embeds new passages by content hash and extracts only those; `delete` removes passages and keeps the triples and entities other passages yield (`HippoRAG.py`) | `add_documents` builds a new tree in place of the existing one (`RetrievalAugmentation.py`) | A registry (DynamoDB, or SQLite locally) diffs the corpus by content hash, re-indexes new and changed documents, and removes a deleted one by the text units every item cites, so only what no surviving text unit states goes; write-ahead crash recovery and one writing run per namespace; property tests check that incremental rounds converge with a full build | [Incremental indexing](../guides/incremental.md), [Design §5](../design/incremental.md#5-incremental-indexing) |
| Multilingual | `graphrag prompt-tune --language` sets the language of the tuned prompts (`cli/main.py`) | `SUMMARY_LANGUAGE` sets the output language of extraction and summaries (`addon_params.py`) | English few-shot extraction prompts (`ner.py`, `triple_extraction.py`) | An English summarization prompt (`SummarizationModels.py`) | An optional translation stage at indexing and query time (with extra target languages), per-language OpenSearch analyzers (`nori` for Korean), multilingual keyword extraction and Korean- and CJK-aware answer matching, for every methodology | [Multilingual corpora](../guides/multilingual.md), [Design §8.3](../design/aws-integration.md#83-multilingual-processing) |
| Bring-your-own ontology and seed graphs | Bring Your Own Graph: supply entity and relationship tables in place of extraction, then run the community workflows (`docs/index/byog.md`) | `insert_custom_kg` inserts given chunks, entities and relationships (`lightrag.py`) | Reuses its own saved open-IE results (`load_existing_openie`); no import of another graph | Builds from text only | An extraction schema declared or imported from an OWL/RDFS ontology (types, patterns, IRIs); seed graphs (JSONL, CSV, RDF) merged into the extracted graph as virtual documents of the corpus, so incremental runs and removal treat them like files and the documents stay canonical | [Extraction schema](../guides/extraction-schema.md), [ontology](../guides/ontology.md), [seed graphs](../guides/seed-graphs.md) |
| Measured defaults and evaluation tooling | The paper judges answers pairwise (comprehensiveness, diversity, empowerment, directness); the library package has no evaluation command | A RAGAS evaluation script (`evaluation/eval_rag_quality.py`) and the paper's pairwise judge prompt (`reproduce/batch_eval.py`) | Exact match, token F1 and retrieval recall metrics (`qa_eval.py`, `retrieval_eval.py`) for the paper's benchmarks | No evaluation code in the package | `run-eval` with LLM-free evaluators beside LangChain and RAGAS judges, paired comparisons with bootstrap intervals and repeats, an order-swapped pairwise judge with position consistency, metered cost and latency, CI gates and a MuSiQue benchmark; defaults backed by published measurements | [Evaluation](../guides/evaluate.md), [Defaults and evidence](./defaults-and-evidence.md) |

None of the four libraries reads OWL or RDF or has a document-translation
stage in its indexing pipeline, as far as their source shows.

### Deliberate divergences

Where the framework departs from an upstream default or algorithm on
purpose:

- **Relationship identity.** A relationship's type is part of its id and
  the edge is directed, so two types between one pair are two edges. MS
  GraphRAG groups relationships by source and target (`extract_graph.py`),
  and LightRAG keys an edge by its sorted endpoint pair (`operate.py`)
  ([Design §4.2](../design/ingestion.md#42-graph-identity-and-merging-stages-5-7)).
- **Community hierarchy.** One Leiden partition of the whole graph, whose
  communities over 20 members are split again and stay the parents of
  their pieces (MS GraphRAG's hierarchical Leiden); coarser levels above it
  partition the graph of communities. Only components smaller than
  `min_community_size` get no community. MS GraphRAG stops at the first
  partition, with `max_cluster_size` 10 and, by default (`use_lcc`), only
  the largest connected component (`config/defaults.py`)
  ([Design §4.3](../design/ingestion.md#43-communities-stages-10-11)).
- **Gleaning.** Three rounds by default against MS GraphRAG's one
  (`max_gleanings`), and a text unit is re-asked only while its last answer
  added something: measured for graph recall
  ([evidence](./defaults-and-evidence.md#gleaning-rounds)).
- **Context shares.** The token budget is split into per-type shares of
  `max_context_tokens` (30,000, LightRAG's total). LightRAG caps entities at
  6,000 and relationships at 8,000 tokens (`constants.py`); here `mix` and
  `hybrid` give text 18,072, entities 7,228 and relationships 4,698
  ([Design §7](../design/scoring-and-citations.md#7-hybrid-scoring-and-token-management)).
- **LightRAG chunk picking.** `kg_chunk_pick_method` defaults to `weight`;
  upstream LightRAG defaults to `VECTOR` (`constants.py`). The two measured
  no different ([evidence](./defaults-and-evidence.md#lightrag-chunk-picking-and-mmr-diversity)).
- **Local search reports.** `local` reads the reports most similar to the
  query entities; MS GraphRAG local reads those of the entities' communities
  (`mixed_context.py`), which `report_selection: membership` follows here
  over the leaf communities rather than upstream's `community_level` cut, and
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
- **No authenticated serving layer**: the MCP server's HTTP transport and the
  local console authenticate no caller and limit no rate; put an
  authenticating proxy or gateway in front of a remote deployment
  ([MCP: HTTP and remote deployment](../guides/mcp.md#http-and-remote-deployment)).
- **Filters are not access control**
  ([Scopes and namespaces](./scopes-and-namespaces.md#not-an-access-control-boundary)):
  separate tenants by index suffix.
- **Shipped backends.** Models: Bedrock (default), Ollama or
  OpenAI-compatible servers; reranking: Bedrock or a sentence-transformers
  cross-encoder ([Local mode](../guides/local-mode.md)). Stores: Neptune (or a
  Gremlin Server) with OpenSearch, or the embedded SQLite stores (one host,
  about 100,000 documents per index). Others need a `register_provider`
  builder or custom adapters ([Design Doc §15](../guides/extend.md#15-extension-guide)).
- **Reranking is regional** ([Required models](../getting-started/aws.md#required-models)); without it
  results keep the fusion order, as `metadata.rerank` shows
  ([Rerank status](../guides/query.md#rerank-status)).
- **A lighter ingestion limits the strategies.** A namespace without
  communities (`indexing.methodologies: [lightrag]`, or
  `graph.community_detection.enabled: false`) holds no `graphrag`: `local`,
  `global` and `drift` are refused there (`simple` sweeps what it holds); without the relationship vector index
  (`indexing.opensearch.build_relationship_vector_index: false`) the
  namespace holds neither `lightrag` nor `hipporag` (`mix`, `hybrid` and
  `ppr` are refused), `local` loses its relationship section, and
  graph expansion under a document filter keeps entities on the
  entities-index check alone, not on the relationships that reach them.
  With `graph.community_detection.report_generation.enabled: false` the
  communities are still detected but no report is written: `local` answers
  without its report section, while `global` and `drift`, which read the
  reports, are refused with a message naming that switch.
- **Markdown and HTML** are read as text in a core install (Markdown as
  written, HTML through the standard library's parser); the `unstructured`
  extra is the richer parser of both, and Word (`.docx`) needs the `docx`
  extra; the container image installs neither by default
  ([Troubleshooting](../troubleshooting.md#which-source-formats-are-read)).
- **Incremental indexing approximates** a full build
  ([Rebuild signals](../guides/incremental.md#rebuild-signals)): delta runs append communities
  instead of re-clustering, and stored communities and reports keep an
  entity's old id after it took its current name's; a full rebuild
  (`indexing.reset: true`) refreshes both. With DynamoDB one document's chunk
  ids must fit one 400 KB item ([*Document size limit*](../guides/incremental.md#document-size-limit)).
- **Concurrent runs need the registry**, whose write lease stops a second run
  on a namespace; without it (or through the direct `IncrementalIndexer` API)
  nothing locks, and two ingestions on one suffix can delete each other's
  index ([Concurrent runs](../operations.md#concurrent-runs)).
- **Isolating a document that kills the task takes runs**
  ([*Quarantined documents*](../guides/incremental.md#quarantined-documents)): about `log2(n) + 1` for one culprit among `n` documents that
  died together, while the deferred ones wait.
- **Changing the embedding model** or dimension needs `indexing.reset: true`;
  an incremental run refuses to write into another model's indices
  ([Workflows](../guides/incremental.md#workflows)).
- **Seed graphs link by name only**
  ([Seed graphs: Precedence](../guides/seed-graphs.md#precedence-documents-decide)):
  a seed node and an unrelated document entity of the same name are one
  entity. IRIs are stored and filterable but are no identity, `owl:sameAs`
  is not followed, and nothing is inferred.
