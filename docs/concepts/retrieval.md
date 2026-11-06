# Retrieval

> 🇰🇷 한국어판: [retrieval.ko.md](./retrieval.ko.md)

A query picks a strategy of one of two methodologies, GraphRAG
(community-summary) or LightRAG (dual-level keyword), or lets `auto` (the
default) route it. Every strategy runs in
one chain over the same indexes: query processing, retrieval, fusion and
rerank, then the context budget and the answer; every answer returns the
sources the answer model actually saw. This page explains the strategies and
the choices that shape an answer; running queries is in
[Querying](../guides/query.md), and
[Design §6](../design.md#6-retrieval-two-methodologies) shows how each
strategy retrieves.

## Search strategies — when to use which

GraphRAG strategies excel at *summarization and thematic synthesis* over a
corpus; LightRAG strategies are faster and lean on *dual-level keyword*
retrieval, good for keyword-driven lookups and as a low-cost baseline. Both
share the hybrid scorer (BM25, vector, graph, RRF, rerank).

Each strategy reads its methodology's index artifacts: on an index namespace
that does not hold that methodology
([methodology profiles](./overview.md#methodology-profiles)) it is refused
before any model call, and `auto` routes only among the strategies of the
methodologies the namespace holds.

**GraphRAG (community-summary):**

| Strategy | Use when | How it works |
|---|---|---|
| `simple` | Fast factual lookups | OpenSearch vector + keyword retrieval, no graph traversal; the claims index too when claims are on. Fastest. |
| `local` | Questions about specific entities | Query entities → graph traversal to neighbors and relationships, with vector/keyword hits; claims when enabled. A query entity seeds the traversal by name, else by its top hit when that scores at least `search.local_search.seed_score_ratio` of the best ([reference](../reference/configuration.md#searchlocal_search)). Community reports: the most similar to the query entities by default ([why not membership](./defaults-and-evidence.md#local-report-selection)). |
| `global` | Broad, thematic questions ("what are the main themes") | Maps the community reports of one cut of the hierarchy (`search.global_search.community_depth`, default 2; a cut over `max_reports_per_level` falls back to the deepest shallower cut that fits, or under a report-key filter to its most similar reports) into scored key points on the fast model, one call per `map_context_tokens` of reports, and answers from the ranked points. Dynamic selection walks down from the roots instead. |
| `drift` | Multi-part questions needing the broad picture and specific facts | Reports similar to a hypothetical answer (HyDE) draft intermediate answers and follow-ups; each follow-up runs a local search and raises more, for `n_depth` rounds; the answer model reduces the results. The most LLM calls (about `primer_folds + k_followups * n_depth`). |
| `auto` | General use (the default) | An LLM router (`models.roles.strategy_selection`) picks among `search.auto_routable_strategies` (default local, mix, global, iterative; drift, at up to 27 LLM calls a query, only when added). It routes questions that chain facts through an entity they do not name to `iterative`, about 10 s slower at the median; leave `iterative` out of the list to keep `auto` off it. The extractions the candidates read run alongside the router, so `auto` takes as long as the strategy it picks, and the one the route does not read is still billed, about one fast-model call a query ([`search.auto_speculative_extraction`](../reference/configuration.md#search)). |

**LightRAG (dual-level keyword):**

| Strategy | Use when | How it works |
|---|---|---|
| `mix` | General LightRAG use | Low-level keywords → entity index, high-level keywords → relationship index, one-hop expansion to incident relationships and endpoint entities (Neptune multi-hop is opt-in: `search.lightrag_search.enable_graph_expansion`), **plus** naive chunk retrieval blended in. |
| `hybrid` | Keyword-driven graph questions | `mix` without the naive chunk blend. |
| `naive` | Fast baseline, comparison eval | Chunk retrieval only (`search.lightrag_search.chunk_stream_top_k`, default 20). The default `--search-type hybrid` adds BM25; `--search-type vector` approximates upstream LightRAG's pure-vector baseline, and MS GraphRAG's `basic` search. |

**HippoRAG 2 (Personalized PageRank, opt-in):**

| Strategy | Use when | How it works |
|---|---|---|
| `ppr` | Multi-hop questions whose facts chain through a few entities; a namespace built for `hipporag` or `lightrag` alone, without community reports (methodology `hipporag`) | The question's relationship hits (the facts, by kNN) and text-unit hits (BM25 + vector) seed a Personalized PageRank over the graph neighbourhood of the facts' entities and the passages they cite (HippoRAG 2, damping 0.5); the ranked passages, the passage hits and the facts are fused and reranked as for every strategy. One LLM call a query: the answer. Not measured on answers yet, so `auto` does not route to it; select it explicitly ([`search.ppr_search`](../reference/configuration.md#searchppr_search), [design](../design.md#64-hipporag-2-personalized-pagerank-opt-in-ppr-local-ranking)). `search.local_search.text_unit_ranking: ppr` uses the same PageRank to order `local`'s text units (on 200 MuSiQue questions it put 11 points more supporting passages in the context; answers not measured yet). |

`mix` and `hybrid` read the relationships index ingestion builds; a short
query that yields no keywords is used as a low-level keyword itself (up to
`search.lightrag_search.raw_query_fallback_max_len`).

Both add the source chunks the matched entities cite and, as a stream of its
own, those the matched relationships cite (each up to
`search.lightrag_search.chunk_stream_top_k`; fusion interleaves them and the
merged chunks are cut to that width after the rerank).
`search.lightrag_search.kg_chunk_pick_method` decides which of an item's
chunks are taken: `weight` (the default) gives each matched item
`related_chunk_number` chunks on a decreasing gradient; `vector` takes the
cited chunks that best match the query, one search a side, as upstream
LightRAG does by default. `weight` stays the default, and MMR's
`search.fusion.diversity_lambda` keeps 0.5: the alternatives measured no
significant difference ([evidence](./defaults-and-evidence.md#lightrag-chunk-picking-and-mmr-diversity)).

**Follow-up rounds:**

| Strategy | Use when | How it works |
|---|---|---|
| `iterative` | Multi-hop questions whose last hop is about an entity the question does not name | Runs `search.iterative_search.base` (`mix` by default, or `local`), then a fast-model controller (`models.roles.query_controller`) reads the packed context: if it does not answer the question, the base search reruns seeded with the bridge entities the controller names and its follow-up queries search the passages, for at most `max_rounds` (2) rounds; the rounds' candidates are fused and reranked, and the answer model answers once. `decompose: true` splits the question into steps instead. About 1-3 fast-model calls more than the base a query. On 200 MuSiQue questions, over `mix`: `answer_contains` +12.5 points and token F1 +5.2 against `mix`, p95 latency 11 s to 27 s ([evidence](./defaults-and-evidence.md#iterative-follow-up-rounds)). `auto` may route multi-hop questions to it; no default selects it ([design](../design.md#63-iterative-bridge-follow-up-iterative_searchpy-opt-in)). |

## Hybrid scoring and rerank

Every strategy fuses its per-source results (BM25, vector, graph) by
reciprocal rank fusion or weights, reranks them, applies MMR diversity and
cuts them to a per-type quota, in that order, so a cut on fused scores cannot
drop what the reranker ranks first
([Design §7](../design.md#7-hybrid-scoring-and-token-management)). What the
rerank did is reported with every query:
[Rerank status](../guides/query.md#rerank-status).

## Answer style

`global` answers with the comprehensive prompt by default, every other
strategy with the concise one. The concise prompt writes a one-sentence
direct answer, then brief support (`custom_prompts.answer_generation_*`);
the comprehensive one a multi-paragraph markdown answer that explains every
key point, modelled on MS GraphRAG's global reduce prompt
(`custom_prompts.comprehensive_answer_*`). `search.answer_prompt_style` is
merged over that default, so naming one strategy leaves the others as they
are:

```yaml
search:
  answer_prompt_style: {drift: comprehensive}   # global stays comprehensive
  query_timeout_seconds: 300                    # see below
# answer_prompt_style: {global: concise}        # concise global answers
```

Comprehensive global answers cover more than concise ones but are less
direct, and pairwise LLM judges favour longer answers: a user who wants a
short answer should keep `{global: concise}`. `drift` stays concise by
default because the comprehensive answer brings its latency close to the
180 s `search.query_timeout_seconds`. With `{drift: comprehensive}`, raise
that deadline (300 s leaves room;
[runbook](../operations.md#query-deadlines-and-model-timeouts)). The
measurements: [Defaults and evidence](./defaults-and-evidence.md#answer-style).

`auto` uses the style of the strategy it routes to. `answer_match` scores
read the first line that is not a heading (`evaluation.answer_span`), so a
comprehensive answer scores lower on exact match whatever its quality.

Global search hands the answer model its ranked key points followed by the
reports they cite. `search.global_search.evidence_tokens` caps that report
text (whole reports, best point first) at 4000 tokens by default, which
cut the cost of a global query with no measured quality difference, and the
answer keeps its sources
([evidence](./defaults-and-evidence.md#global-evidence-cap)). `null` (no cap) keeps as much as the
context budget holds, which on a large cut is most of a global query's input
tokens; `0` (cheaper still) sends the key points only, as MS GraphRAG's reduce step reads
them, and the answer then reports no sources (the key points are model
output), so the `retrieval` evaluator skips its rank metrics for a global
query (`no_sources_by_config`) instead of scoring a miss.

## Context budget

`search.token_manager.max_context_tokens` (`30000`, upstream LightRAG's
budget) is the retrieved context the answer prompt gets, clamped with a
warning to what `models.roles.answer_generation` accepts beside its output
reservation; `null` derives it from that model's window. Each section type
gets a capped share of it, in score order, and a share a type leaves unused
goes to the types that still have candidates
([Design §7](../design.md#7-hybrid-scoring-and-token-management)). Per-type shares and
`local` slot quotas: [reference](../reference/configuration.md#searchtoken_manager).
