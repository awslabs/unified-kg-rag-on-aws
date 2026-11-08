# Design: retrieval, HippoRAG 2 and RAPTOR

Part of the [design reference](../design.md), which lists every section. Code cites
the section numbers: they never change.

### 6.4 HippoRAG 2 Personalized PageRank (`ppr`, `local` ranking)

A port of HippoRAG 2's retrieval (Gutiérrez, Shu, Qi, Zhou, Su, *From RAG to
Memory: Non-Parametric Continual Learning for Large Language Models*, ICML
2025, [arXiv 2502.14802](https://arxiv.org/abs/2502.14802); code
[OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), MIT),
reimplemented on the graph and indexes ingestion already builds: the
algorithm is re-written here, none of its code is vendored or depended on.
HippoRAG 2 seeds a Personalized PageRank (PPR) over a graph of entities and
passages from the facts (triples) most similar to the question and from a
dense passage search, and ranks the passages by their PPR score; the paper
reports it above its dense-retrieval baselines on the multi-hop sets
(MuSiQue among them). It is used twice here: the `ppr` strategy is
opt-in, and the `local` ranking is the default, since a measured
answer-level result (§9.1) showed it better than the earlier order.

- **`ppr` strategy** (`adapters/search_strategies/ppr_search.py`,
  `methodology = Methodology.HIPPORAG`, config `search.ppr_search`): the
  HippoRAG 2 retrieval as a strategy of its own. No query-time LLM call
  besides the answer (no entity or keyword extraction). It reads the
  entities, relationships and their vector index, text units and the graph
  (the `hipporag` profile of §4.6, what `lightrag` builds); no community
  report, so every namespace that holds the relationship vector index
  serves it (the default profile, `[graphrag]` with its relationship
  vectors, `[lightrag]`, `[hipporag]`), and one without it (`[graphrag]`
  with `build_relationship_vector_index: false`, `[raptor]`) refuses it
  before any model call (`MethodologyNotIndexedError`). The `local` ranking
  below stays a `graphrag` strategy. `auto` never routes to `ppr`
  (`search.auto_routable_strategies` refuses it until a measurement shows
  where it wins): its answers were within the noise of `mix` and `local`
  at a fraction of their cost, and weaker on the longest chains, so it is
  the cheap, fast option a caller selects
  ([evidence](../concepts/defaults-and-evidence.md#the-ppr-strategy)).
- **`local` ranking** (`search.local_search.text_unit_ranking: ppr`, the
  default; `support` orders the units by how many expanded entities cite
  each): the same PPR replaces the order of local search's text
  units (`_rank_text_unit_ids`) over local's own expansion; everything else
  of local (seeds, sections, quotas, rerank) is unchanged. DRIFT's follow-up
  local searches keep `support` (`search.drift_search.text_unit_ranking`):
  the ranking was measured for `local` alone. A retrieval-only screen
  chose its seeding (the PageRank adds to the passage hits beyond fusing
  them into `support`'s order, and seeding from the entities alone gains
  less), and answer runs against `support` made it the default
  ([evidence](../concepts/defaults-and-evidence.md#local-ppr-ranking)). Its
  extra searches add CPU work in the querying process; with many concurrent
  queries in one process that work waits on the interpreter lock, which
  the embedded engine's id and exclusion lookups, the dense-script test and
  the query CLIs' one BLAS thread (`launch.one_blas_thread`) keep small.

**Data flow of `ppr`.**

1. *Seeds*, concurrently, each under the caller's filters and suffix: the
   `fact_top_k` (5, HippoRAG's `linking_top_k`) relationships-index hits of
   the question (*facts*) by kNN against the relationship descriptions
   (`fact_search_type`: HippoRAG's query-to-fact similarity; null for the
   request's search), and the `passage_seed_top_k` (20) text-units hits
   (*passage seeds*) under the request's `search_type` (BM25 + kNN fused by
   default), so vector, lexical and graph signals all seed. With
   `entity_seed_top_k` (default 0) the entities-index hits of the question
   seed too; should the relationship vector index be off at query time on a
   namespace that holds it, they take the facts' place (`fact_top_k` of
   them), so the strategy still seeds the graph.
2. *Neighbourhood*: graph expansion (§6, the graph walk or, under a document
   filter, the walk over the relationships index) from the facts' endpoints
   and the entity hits, `max_hops` (2) deep, at most
   `max_neighbourhood_entities` (100) entities besides the seeds, each seed's
   share of the budget in proportion to its fact score.
3. *Edges*: the relationships between two entities of the neighbourhood
   (`BaseSearchStrategy._neighbourhood_edges`, below), at most
   `max_edges_per_vertex` (50) per entity.
4. *Graph* (`domain/retrieval/ppr.py`, `PassageGraph`): one node per entity
   and per passage; one undirected edge of weight 1 per relationship (parallel
   relationships add up; HippoRAG counts the triples between two phrases),
   and one of weight 1 from each entity to each document text unit it cites
   (`text_unit_ids`; HippoRAG's passage-to-phrase *context* edges). An entity
   citing more than `entity_frequency_threshold` (20) document units adds
   no passage edges, a seed neither (it still passes its mass on through its
   relationships), which bounds the graph a hub would blow up. The passage
   seeds are passage nodes too; one no reached entity cites is a node
   without edges.
5. *Reset vector* (HippoRAG 2's seeding, `seed_weights`): the facts' scores
   are read relative to the best fact (HippoRAG min-max normalizes them; a
   BM25 score would otherwise outweigh the passage weight tenfold), the
   entity hits' alike. An entity of the graph scores the mean, over the
   facts that name it (each end counted, as HippoRAG counts a self-loop's
   two), of the fact's score divided by the number of document units the
   entity cites (at least 1), and only the `link_top_k` (5) best entities of
   the graph keep their score (an endpoint the stores did not return takes
   no slot); an entity hit is scored the same way. A passage seed scores
   `passage_node_weight` (0.05) times its hit score min-max normalized over
   the passage seeds (all 1 when they tie), so the lowest-scored seed resets
   with 0 and gains mass only through the graph: HippoRAG's lowest passage
   is the worst of the whole corpus, here it is the last of hits that
   matched (a deliberate difference, below). Normalized to sum 1.
6. *PPR* (`personalized_pagerank`): power iteration of
   `p = (1 - d) r + d (p P + (p · dangling) r)`, damping `d` = `damping`
   (0.5, the probability of following an edge), at most `max_iterations`
   (30) steps or until the L1 change is below `tolerance` (1e-6); the mass of
   a node without edges returns to the reset vector, as in igraph's
   (PRPACK) personalized PageRank HippoRAG calls. Nodes are ordered by id, so
   a run is deterministic; ties rank by id.
7. *Fusion buckets* (`search.fusion.fusion_weights` keys):
   `ppr_passages` (text: the passages by PPR score, the first text quota
   `max(2 * top_k, 20)` of them that pass the caller's filters, fetched by
   id), `ppr_seed_text_units` (text: the hybrid passage hits, in their
   order) and `ppr_facts` (relationship: the facts, in their order). RRF
   fuses them through `HybridScorer` (a passage both rank adds both terms),
   the reranker reorders the text candidates when `rerank` is on (the
   default; off records rerank `skipped`), each type keeps its quota, and
   the token budget packs the context as for every strategy.
8. *Answer*: the standard concise answer prompt
   (`search.answer_prompt_style`); the sources are the sections the model
   saw, with the provenance every strategy reports (§6: document ids,
   chunk ids, `text_unit_ids`, seed-graph provenance, `iri`,
   `source_kinds`), so `run-eval`'s retrieval evaluator scores it as any
   other.

**`local` ranking** is the variant the retrieval-only screen measured
([evidence](../concepts/defaults-and-evidence.md#local-ppr-ranking);
`tests/unit/test_ppr_search.py` pins its ranking to the screen's prototype
on a fixture). It builds the same graph from local's expanded entities
(after its frequency filter, which keeps the seeds as nodes; past
`search.local_search.entity_frequency_threshold` a seed links no passage
either) and their text units, plus the endpoints of the question's
`search.local_search.ppr_seed_relationships` (5) relationships-index hits
(kNN, `fact_search_type`; an endpoint outside the neighbourhood is read from
the entities index and joins the graph with its relationships). Its reset
vector (`summed_seed_weights`): an entity's seed hit score plus the mean
score of the relationship hits naming it, both as the searches return them,
divided by its document units, each counted once as `ppr` counts them (no
`link_top_k` cut), plus the passage seeds
(`search.ppr_search.passage_seed_top_k` (20) text-unit hits of the question
under the request's search, weighted by `passage_node_weight`). The PPR
order over every passage node, passage seeds included, replaces
`_rank_text_unit_ids`: the passage hits enter local as seeds and candidates
of that ranking, not as a fusion bucket of their own (as measured), and are
not fetched again (only the ranked units they are not). Unlike `ppr`, which
scales each kind of hit to its best before seeding (`_relative`), local
seeds with the raw scores, as measured: under the default hybrid search
they are already on a 0-1 scale, but under `search_type: lexical` the BM25
scores of the entity seeds outweigh the facts' kNN scores and the
passages' `passage_node_weight`. Kept as measured; scaling them is a
change to measure first.

**DRIFT** follow-ups are local searches, but rank their text units by
`search.drift_search.text_unit_ranking` (`support` by default, what DRIFT
was measured with): `ppr` was measured for `local` alone and adds a
text-unit search, a relationship search, an edge read and an embedding to
each of up to 20 follow-ups. The
damping, iterations, tolerance and edges per entity are
`search.ppr_search`'s. The search metadata records `text_unit_ranking` and
the run's counts (`ppr`: `entities`, `relationships`, `linked_entities`,
`passages`, `entity_seeds`, `passage_seeds`, `fact_seeds`, `iterations`,
`converged`).

**Neighbourhood edges, one contract per backend.**
`BaseGraphRAGRetriever.aneighbourhood_edges(vertex_ids, query,
per_vertex=n)` returns the relationships whose two endpoints are both in
`vertex_ids`, each vertex contributing at most its first `n` such edges by
relationship id (the edges of either endpoint's first `n` are kept), as
`GraphEdge` (id, endpoints, stored weight, source kinds), sorted by
`(source_id, target_id, id)`. `None` (the base's default) means the backend
cannot tell; the strategy then reads the relationships index.

- Neptune / Gremlin Server (`NeptuneRetriever`):
  `local(bothE().where(otherV().<in the set>).order().by("id").limit(n))`
  from the set's vertices, in one request (custom vertex ids: `g.V(ids)` and
  `hasId`; property ids: `has("id", within(...))`, also while a full build
  switches the label). `indexing.neptune.deterministic_order: false` takes
  the store's first `n`, as the walk does. The traversal reads every
  incident edge of each vertex (a hub of 20,000 edges took 0.4 s on Gremlin
  Server, more than the expansion), so the retriever declares
  `scans_incident_edges` and, where the relationship vector index is
  built, the strategies read the edges from it as below (both endpoints in
  one terms query, no per-vertex scan); the graph serves them without it.
- SQLite (`SQLiteGraphRetriever`): one indexed `IN` read of the edges
  leaving each batch of 400 vertices (an edge within the set leaves one of
  them), the target kept when in the set, the first `n` per vertex by id in
  Python (`cap_edges_per_vertex`, once, in `aneighbourhood_edges`).
- Under a document filter (and from a graph that scans incident edges)
  the graph cannot tell which edges a passing document states, so the
  edges come from the relationships index
  (`_indexed_neighbourhood_edges`: one request per batch of source ids
  with the target pinned to the set, a page of at most `max_query_size`
  edges best first; a batch whose page comes back full is asked again in
  halves down to one vertex, and a source with more than a page is asked
  from its targets' end with the source pinned, so a hub's edges are all
  read; under two or more document keys an edge also needs one passing
  document by its text units), cut per vertex alike. Only a vertex past a
  page at both ends keeps its best-ranked page there. Without the halving
  the whole set got one page: on a real Neptune graph 100 of 178-678
  edges, and the PageRank ran on a rank-biased cut. A `source_kinds` filter applies to the edges of every path; there
  the index also applies the caller's relationship keys (`predicate_iri`,
  `in_schema`, `weight`), which the graph path does not apply to edges, as
  the graph walk crosses every edge.

`tests/integration/test_retrieval_conformance.py` pins one result for
SQLite, Gremlin Server (both vertex-id modes, `LOCAL_STORES=1`) and Neptune
(`-m aws`).

**Why a truncated neighbourhood can be enough.** PPR is the distribution of
where a walk that starts on the reset vector stops, continuing each step
with probability `d`. A node more than `h` hops from every reset node needs
at least `h + 1` steps, so it holds at most `d^(h+1)` of the mass: 12.5% past
2 hops, 6.25% (0.5^4) past 3 at `d = 0.5`. Restricting the graph to the
`h`-hop ball around the reset nodes changes only walks that step off its
boundary, which also take at least `h + 1` steps, so the scores move by at
most `2 d^(h+1)` in L1 (`tests/property/test_ppr_properties.py` checks the
bound on random graphs). The graph built here is not that ball, so the
bound does not cover it: the expansion's budget and per-vertex caps cut the
entity side, and a passage seed links only to the reached entities that
cite it, never to the other entities that do (an index of which entities
cite a passage would be one more read a query), so one no reached entity
cites holds its own reset mass. How much these cuts cost is part of the
measurement.

**Cost.** No LLM call at index time: the graph, the relationships' vectors
and the text units are what ingestion builds anyway. A `ppr` query costs the
question's embedding, three searches (facts, passages, optional entities),
one graph expansion, one edge read, one passage fetch, a PPR off the event
loop (`numpy`: a median of 2 ms for 50 entities and 200 passages, 8 ms for
150 and 600, 31 ms for 300 and 2,000), the rerank and the answer call: one LLM call,
where `local` makes two (entity extraction and answer). The `local` ranking
adds a passage search, a relationship search (`ppr_seed_relationships`)
and an edge read to a local query.

**Failure modes.** No seed at all (nothing matched): no results, so the
answer is "cannot answer" without a model call. A failed seed search,
expansion or edge read that is not fatal degrades that part only (logged;
fatal store errors fail the query, §8.1): without edges the PPR runs over the
entity-passage links, without an expansion over the seeds read from the
entities index; a reset vector without mass leaves the passage hits in their
own order. The `local` ranking falls back to `support` when the PPR has no
passage to rank. Without the relationships index, the edges under a document
filter come from the graph, unchecked against the filter, as the graph walk's
are (§6): an edge only excluded documents state can still carry mass, never
content.

**Fidelity to HippoRAG 2.** `tests/unit/test_ppr_hipporag_parity.py` runs the
original's PPR (python-igraph `personalized_pagerank`, PRPACK, as
`HippoRAG.run_ppr` calls it) and its seeding
(`graph_search_with_fact_entities`, executed from a checkout named by
`HIPPORAG_SRC`) on the same graphs, reset vectors and damping, and compares
the scores and passage order; it is skipped unless both are available (the
original is never a dependency) and runs weekly in the `fidelity` workflow,
at HippoRAG commit `d5c8329`. The deliberate differences:

| HippoRAG 2 | Here | Why |
|---|---|---|
| PPR over the whole graph | over a bounded neighbourhood of the seeds (`max_hops`, the expansion budget, `max_edges_per_vertex`) | a query reads a few store requests, not the corpus; the mass past `h` hops is at most `d^(h+1)` |
| Recognition memory: an LLM filters the top facts | no filter | no query-time LLM call; the facts are fused hybrid hits |
| Facts by dense similarity to triple embeddings, min-max over all facts | the top relationships-index hits by kNN (`fact_search_type`, lexical or hybrid on request), scores relative to the best | the relationship descriptions are the facts this index holds; the top `fact_top_k` alone are searched |
| Passage weights from a dense score of every passage | the top `passage_seed_top_k` hybrid hits, min-max over them (the last hit resets with 0) | a bounded search; hybrid like every strategy here. Kept unfloored: the parity tests compare these weights with the original's on the same scores, and no run measured a floor |
| PPR order alone ranks the passages | the PPR order and the passage hits are two fusion buckets, then the optional rerank | the hybrid fusion every strategy shares; `fusion_weights: {ppr_seed_text_units: 0}` and `rerank: false` come closest to the original |
| Synonymy edges between similar phrases | none | not a merge substitute: entity resolution (§4.2) joins only names with the same compact key, and similar names only with `processing.fuzzy_entity_resolution` (off by default), so near-synonyms stay separate entities linked only through shared passages; synonymy edges (kNN over entity name embeddings) are unmeasured |
| Fact edge weight: the triples between two phrases | 1 per stored relationship (merged over its text units) | one relationship per pair and type after merging |
| Every phrase links to its passages | not past `entity_frequency_threshold` units, seeds included | bounds the graph a hub entity would fill |
| A passage links to every phrase it contains | to the reached entities citing it; a passage hit none cites has no edge | the neighbourhood is read from the seeds outward, not passage by passage |
| (the `local` ranking) | its own seeding, `summed_seed_weights`, on raw search scores | the variant the retrieval screen measured |
| Self-loops kept | dropped | they only hold mass on their node |

### 6.5 RAPTOR summary tree (`raptor`, `tree`, opt-in)

The summary tree reimplements RAPTOR (Sarthi et al., ICLR 2024, arXiv
2401.18059; reference code github.com/parthsarthi03/raptor, MIT, read and
run for comparison only, never vendored): text units are clustered in
embedding space, each cluster is summarized by a model, the summaries are
clustered and summarized again, and a query searches every level at once
(RAPTOR's "collapsed tree"). It is opt-in: `raptor` is not in the default
`indexing.methodologies`, so no default ingestion builds a tree, and
`search.auto_routable_strategies` refuses `tree`: on MuSiQue's multi-hop
questions it answered better than `naive` (+8 points of `answer_contains`,
one repeat on one corpus) but well below `mix` and `local`, and slower than
`naive`
([Defaults and evidence](../concepts/defaults-and-evidence.md#raptor-summary-tree)).
The published evidence is mixed too:
RAPTOR leads on specific QA in one unified benchmark (Zhou et al., arXiv
2503.04338: MuSiQue 24.1 against vanilla RAG's 17.2), but is below vanilla
RAG on GraphRAG-Bench's novel set (arXiv 2506.05690: fact retrieval 49.3
against 58.8). Its case is cheap indexing, not beating `global` on
summaries.

**Methodology (§4.6).** The tree is the `summary_tree` index artifact of the
methodology `raptor` (required: text units and the summary tree), and `tree`
declares `methodology = Methodology.RAPTOR`. The catalogue maps the
artifact to the `summary_tree` stage (`ARTIFACT_STAGES`,
`ARTIFACT_PRODUCERS`), which reads only the indexing stage's stored output
(`STAGE_DEPENDENCIES[SUMMARY_TREE] = {INDEXING, TRANSLATION}`: the stored
text units, translated when translation is on), so the stage runs only
when `raptor` is selected and `indexing.methodologies: [raptor]` alone
runs parsing, loading, chunking, translation (when configured), indexing of
the text units and the tree: no graph extraction, gleaning, resolution,
claims, graph analysis or communities. With the other methodologies the
tree is added beside their artifacts. The summary tree is not a
`CORPUS_WIDE_ARTIFACTS` member: it is built from what the stores hold, so
adding `raptor` to an existing namespace builds only the tree on the next
run (an empty delta included) instead of rebuilding the namespace, and
adding `graphrag` or `lightrag` to a `raptor`-only namespace extracts over
the documents (their rebuild) and leaves the tree as it is: the text units
do not change, so neither does the leaf-set digest. The indexing stage
writes the build record before the tree exists, so the stage adds
`summary_tree` to the record once the tree is current
(`VectorIndexer.add_build_artifacts`). An incremental record keeps only the
artifacts both it and the stored record name; the indexing stage keeps
naming a live tree while `raptor` is selected and its stale nodes were
deleted (*Incremental runs, resets and crash recovery* below), and a failed tree build drops it, so `tree` is
refused only until a run builds the tree again. On a `raptor`-only
namespace `auto` falls back to `tree` (§4.6's fallback), the only strategy of a held
methodology (`naive` and `simple` serve it too when selected).

**Placement and data flow.** A post-indexing stage,
`PipelineStageType.SUMMARY_TREE` (appended after `INDEXING`, so every
earlier stage keeps its cache key), run when `raptor` is selected, `application/ingestion/summary_tree_stage.py`:

1. *Leaves.* The stage reads the namespace's stored text units through
   `VectorIndexer.scan_text_unit_pages` (an id-ordered page walk, the seed
   graphs' units left out as every text search leaves them out; the port's
   default serves a store's `scan_text_units` in pages, so a store that
   implements only that one keeps the tree, holding every unit at once): their
   id, indexed text (the `target_language` translation when one is
   stored, the source text otherwise, as the indexer embeds them), stored
   vector, `n_tokens` (counted again for a translation: the stored count
   is the source text's), file name and document `filters`. Reading the
   committed store, not the run's context, makes the tree a function of
   what is stored: an incremental run and a full build of the same corpus
   build the same tree, and a run with an empty delta can build a tree
   that is missing.
2. *Trigger.* The leaf-set digest (each leaf's id, a hash of the text the
   tree reads, its translation when it holds one, and its document
   `filters`: a reset that re-chunks, re-translates or re-filters text units
   under the same ids changes it) and the tree fingerprint (every tree setting,
   the summary model, prompt and language, the embedding model and
   dimension) are compared with the build record in the summary index's
   `_meta` (`summary_tree`). Equal and complete: nothing is done. A tree
   marked `partial` (a summary failed) is rebuilt; reuse makes that cost
   only the failed clusters.
3. *Clustering and assembly* (`domain/ingestion/summary_tree.py`, pure
   numpy, deterministic): below.
4. *Summaries* (`adapters/ingestion/tree_summarizer.py`,
   `TreeSummaryPrompt`, `ModelRole.SUMMARY_TREE` on the default tier, through
   `BatchProcessor` and the run's LLM response cache), one level at a time,
   bottom-up. A cluster whose input (its children's texts in id order, the
   language and the summary settings: `input_key`) equals a node of the
   live tree takes that node's summary and makes no call. *Length:* the
   prompt asks for the words (characters for Japanese, Chinese and Korean)
   that 70% of `max_summary_tokens` holds at the rate the token counter
   that bounds the summary gives the cluster's own texts (Claude Haiku
   wrote about 1.4 times the hinted length: at 90%, 70% of 603 MuSiQue
   summaries were over the budget and cut), so the hint
   follows the counter (the script-aware estimate scaled for the model's
   tokenizer, or tiktoken) and the language: a fixed words-per-token rate
   asks for too many words where the counter counts about 2 tokens a word,
   and the summaries end cut mid-sentence. A summary still over the
   budget keeps its longest prefix of whole sentences within it (`shared/utils/sentences.py`: `。！？．` anywhere,
   `.!?…` before whitespace and not a `.` before a lowercase word, a line
   break), and is cut at a character only when its first sentence is
   longer than the budget.
5. *Embedding and write.* The summaries are embedded by the vector
   indexer (`embed_summary_texts`, which takes the stored vector of a text
   whose digest the live summary index holds), and the whole tree is
   written into a new timestamped index whose alias is then swapped
   (`write_summary_tree`), the build record written with it. A tree is
   rebuilt, never patched.

The stage runs in its own usage scope (`run-ingestion` reports its calls,
tokens and cost as for any stage) and inside the run's lease and write
fence. It is skipped, with a warning, when the run's indexing stage failed.
It declares `PipelineStage.READS_STORES`: a run of it alone requires no
upstream stage to be backed by the stage cache (`resume._check_upstream_complete`;
after an upgrade the earlier stages' keys change while the stores still
hold their output), and probes only the indices the namespace lacks, as an
incremental run does, rather than every index of a full build.

**Clustering** (each step deterministic: inputs sorted by id, seeded by
`random_state`, coordinates rounded to 1e-6 so a BLAS thread count does not
flip a tie):

- *Reduce.* L2-normalize, center, project onto the top
  `min(reduction_dims (10), n - 1)` principal axes (numpy SVD, each axis's
  largest loading made positive). This replaces UMAP.
- *Partition one level.* `algorithm: kmeans` (default): Lloyd's iterations
  (at most 100), `k = ceil(n / target_children)` (6), hard assignments, an
  empty cluster re-seeded at the farthest point; two steps (about `sqrt(k)`
  coarse clusters, then each split by its share of `k`) when `n * k` passes
  4 million, so a 200,000-leaf level costs about `2 n sqrt(k)` distances.
  The initial centers are the farthest-first `k` of the `2k` nodes of lowest
  seeded id hash, a sample one added or removed node changes by one at
  most: k-means++ samples by cumulative distances, so one inserted node
  changed every later pick and a delta re-clustered, and re-summarized, the
  whole tree. On 2,000 synthetic leaves (40 blobs, 64 dimensions) a 1% delta
  reused 39% of the summaries with this start against 0.8% with k-means++,
  a 10% delta 8% against 0.6% (not yet measured on real embeddings).
  `algorithm: gmm` (RAPTOR's): its two steps, a global Gaussian mixture,
  then a local one inside each global cluster larger than `dims + 1`
  (local PCA of the members' vectors in place of upstream's local UMAP);
  each mixture is EM with full covariances (regularization 1e-6) started
  from k-means, k chosen by BIC over at most 8 values from half to twice
  the expected count, `sqrt(n / target_children)` for the global mixture
  and `members / target_children` for each local one, so the two multiply
  to about `n / target_children` (upstream compares 1..49, its
  `max_clusters` 50, which is slow on large levels;
  `TreeSettings.bic_grid = "upstream"` restores it for the fidelity tests
  only), and a node joins every cluster whose
  posterior exceeds `soft_threshold` (0.1, upstream's `threshold`), its best
  one when none does. Soft membership makes the tree a DAG; lineage is a
  union.
- *Token bound.* A cluster whose children exceed `max_cluster_tokens`
  (8,000, counted by the token counter, so Han, Hangul and Kana are counted
  about a character a token) is clustered again with the same algorithm
  and settings, at least two ways, until it fits (a split that does not
  shrink falls back to id-order packing).
- *Levels.* The new nodes' summary vectors are clustered again, up to
  `max_levels` (5, upstream `num_layers`), until a level holds at most
  `min_nodes_to_cluster` nodes (11, upstream's `reduction_dimension + 1`)
  or would shrink by less than `min_level_shrink` (20%). A cluster of one
  node is not summarized: the node is promoted to the next level. The top
  level may hold several roots.
- *Scope.* `scope: corpus` (default): one tree over the namespace's
  leaves, or one per value tuple of the `partition_by` document filter
  keys. `scope: document`: one tree per document (RAPTOR's own setting),
  documents told apart by their stored file path; one with fewer than
  `min_leaves_per_tree` (3) leaves gets none. MuSiQue's one-paragraph
  documents get no tree under `document`.

**Data model.** `SummaryNode` (`domain/models/summary_node.py`): `id` (a
content id: level, partition and sorted child ids), `level` (1..L; text
units are level 0 and are not copied), `text` (the summary, in the index
language), `child_ids`, `text_unit_ids` (the transitive leaves, sorted,
cut at `max_lineage_text_units`, 10,000: `lineage_complete` false then, and
a query resolves the lineage through the children), `n_leaves`,
`n_tokens`, `partition`, `language`, `input_key`, and `attributes`
(`index`, and the uniform `filters` below). The index is
`indexing.opensearch.summary_nodes_index_prefix`
(`graphrag-summary-nodes`): a kNN field, a `text` field analyzed with the
index language's analyzer (`language_analyzers`: `nori` or `cjk` for
Korean, `cjk` for Japanese and Chinese), keyword arrays for the lineage
and the filters. Neptune and the SQLite graph get nothing, so graph-store
parity is free; OpenSearch and the embedded engine share the indexer code.
Stored text units hold no document id, so a node carries none:
`text_unit_ids`, which the expanded passages and the retrieval evaluator
resolve to files, is its provenance.

**Multilingual.** Leaves are the indexed text, so a translated corpus is
clustered and summarized in its target language and an untranslated one in
its source language: the summaries are written in the index language
(`language_name`, the language the query side extracts in), whatever the
mix of the corpus. Every budget counts with the token counter. Without
translation a mixed-language corpus may cluster by language before topic;
`partition_by` on a language filter key splits the trees.

**Incremental runs, resets and crash recovery.** The rule is rebuild,
never patch: clusters move when the corpus does, so patching lineage up a
corpus tree would not converge with a full build, while a rebuild is the
same function of the same stored leaves (a property test checks it).
Unchanged clusters cost no call (`input_key` reuse) and no embedding
(digest reuse).

1. *Removal, fail-closed.* `IndexingManager.remove_text_units` first
   deletes every summary node citing a removed text unit and every node
   above it (by `child_ids`, so a node whose lineage was cut is found
   too): a summary paraphrases its children, so it goes even where other
   children remain. A full build (reset, or a run without the registry)
   deletes the nodes citing a text unit it did not write. Either deletion
   failing fails the indexing stage like any failed write.
2. *Rebuild.* The `summary_tree` stage then builds the new tree from the
   leaves and swaps it in.
3. *Between the two.* The indexing stage's build record keeps naming the
   summary tree where a live one exists, `raptor` is selected and every
   deletion of step 1 succeeded, so `tree` queries meet the old tree
   without the deleted nodes while the stage runs, and after a crash
   before it finished; new documents are reachable as leaves. The tree
   record still holds the old digest, so the next run rebuilds; an index
   a killed build left unswapped is deleted by the next build's reaper. A
   build that fails (the stage raises) drops the tree from the build
   record: `tree` is refused until a run builds it.

`refresh: on_reset` keeps step 1 on delta runs and rebuilds only on
`indexing.reset` (or a missing tree, or one some of whose summaries
failed), reporting the leaves no level-1 node
covers (`tree_uncovered_leaves`). `scope: document` rebuilds the trees of
every document too, but reuse makes an unchanged document's tree free.

**Filters, scopes and tenants.** One tree per index suffix, never across
suffixes. A node stores a document filter key only when every one of its
leaves has the same value for it (uniform-only, fail-closed); the
retriever's existing rule, "an item without the attribute is excluded",
does the rest: under `attr_k: x` a node passes only if all its documents
have `k = x`. This differs on purpose from reports and entities, which
pass when one document does: a summary sentence can restate an excluded
document directly in the answer context. `partition_by` keeps nodes pure
on its keys. The summary index declares `id`, `text`, `level`,
`text_unit_ids`, `child_ids`, `n_leaves`, `n_tokens`, `lineage_complete`,
`language`, `partition` and the document filters;
`filtered_index_prefixes` of `tree` names the text-units and summary
indexes, so a key neither declares is refused. A key only the text units
declare (`community_ids`, `source_kinds`) would be dropped on the summary
index by the retriever and the summaries returned unfiltered, so `tree`
reads no summary under it (both summary buckets are skipped,
`metadata.summaries_skipped_for_filters` names the keys) and answers from
the text units. Expanded passages are fetched under the caller's filters
too, and the graph bucket's text units are the best-weighted that pass them
(`_passing_ranked_units`), not the best-weighted cut before the check.

**Retrieval: `tree`** (`adapters/search_strategies/tree_search.py`,
`@register_strategy(SearchStrategy.TREE, required_roles=(RetrieverRole.DOCUMENT,))`,
no query-side model call). On a namespace whose build record does not
name the summary tree it is refused before any model call
(`MethodologyNotIndexedError`, §4.6). Its fusion buckets, run concurrently
under the request's `search_type` (BM25 + kNN by default):

| Bucket | What |
|---|---|
| `tree_text_units` | the hybrid text-unit search (`naive`'s) |
| `tree_summaries` | the hybrid search over every level of the summary index (the collapsed tree) |
| `tree_graph_summaries` | the summary nodes citing the text units of the top `graph_seed_entities` (8) entities a hybrid search of the entities index finds with the raw query (LightRAG's raw-query fallback, no model call), each node scored by the seed weights it covers over `sqrt(n_leaves)` so a root does not win by size |
| `tree_graph_text_units` | those entities' text units, ranked the same way |

The graph buckets run when the queried namespace holds the entities index
(`graph_signal: auto`, read from the held artifacts the chain passes in
`SearchQuery.metadata["held_artifacts"]`, the configured methodologies'
without them; `off` turns them off), so a `raptor`-only index answers from the
other two and reports `graph_signal: unavailable`. Text-unit ids are the
join key on purpose: they are never re-keyed by merges or renames, so the
tree needs no update when the graph changes. The graph walk of `local` is
not used: it would need the graph role on every query, which a
`raptor`-only deployment does not have.

Fusion is RRF over the buckets (`fusion_weights` apply), then the
cross-encoder rerank over leaves and summaries together (the one
comparator across the two types), MMR and the type quotas
(`text_top_k` 20, `summary_top_k` 10), then at most `max_nodes_per_level`
(3) summaries of any one level. Each kept summary is expanded to
`expand_children` (2) of its own passages: one search of the text units
within its lineage (`id` filter, cut to the live `max_terms_count`), under
the caller's filters, joined to the text with the summary's score just
below it (`expanded_from`). The context gets the new `summary` section
type, whose share is `search.token_manager.type_budgets.summary` (0.5); the
text units take the spare budget first (`spare_context_budget_first`), so
leaves always ground the answer. A summary reads `Summary (level 2, 14
passages): ...`; its source carries `level`, `child_ids`, `text_unit_ids`,
`n_leaves` and `lineage_complete`, so a user can expand it to the passages,
and the retrieval evaluator attributes it through its lineage (a `summary`
row in `by_source_type`), its files ranked after the passages' as every
lineage-attributed source's are (at its own rank a summary would fill the
top `k` with its lineage). A summary is index content written by a model at
ingestion, as a community report is; it is not `synthesized` (query-time
model output) and is reported as a source. Summaries are not used by
`naive`, `global` or `drift` (deferred until measured: each would change a
measured default path).

**Failure modes.**

| Event | Outcome |
|---|---|
| A summary call fails every attempt | The node is dropped and its children promoted; the record is `partial`, so the next run retries (only the failed clusters cost a call) |
| The vector store cannot page text units or embed (a custom backend: the port defaults return `None`) | The stage skips with a warning; no tree |
| More than `max_leaves_in_memory` (100,000) leaves in one tree | The stage fails before it reads their vectors, naming `partition_by` and `scope: document` |
| Writing the new index fails | The new index is dropped, the old tree (without its stale nodes) stays, the stage fails |
| Indexing failed in the same run | The stage is skipped |
| No summary tree at query time | `tree` is refused when the build record says so (`MethodologyNotIndexedError`); with a record from before methodology profiles it answers from the text units with `metadata["summary_tree"] = "missing"` and a WARNING. `run-ingestion --enabled-stages summary_tree` (with `raptor` selected) builds the tree from the stored text units |

**Cost.** Calls ≈ leaves / (children per node − 1), about 0.18 per leaf.
On the MuSiQue benchmark store (3,318 leaves, 0.37M tokens) the fast tier
(`models.roles.summary_tree: fast`) built 603 nodes with one call each,
reading about three times the corpus tokens; that run's tokens and cost
are in [Defaults and
evidence](../concepts/defaults-and-evidence.md#raptor-summary-tree). Estimated
from those tokens at the [list prices](../concepts/defaults-and-evidence.md#list-prices-used),
the default tier (Sonnet 5.5) costs about 20 times the fast tier, more
with reasoning tokens. A delta pays for the clusters whose children
changed. At query time `tree` costs a `naive` query plus up to three more
searches, a wider rerank, one search per expanded summary and the
summaries' context.

**Fidelity against the original** (`tests/unit/test_raptor_fidelity.py`).
Upstream's outputs on synthetic, pre-reduced points were recorded once by a
script run in upstream's own environment (its code at commit
`7da1d48`, with its model clients stubbed, UMAP replaced by an identity
truncation since it is unseeded upstream, and tiktoken by a word count, so
both sides cluster the same points and count the same tokens). The points
are generated in plain Python from SHA-256 draws
(`tests/fixtures/raptor_fidelity.py`), so the committed fixture
(`tests/fixtures/raptor/upstream.json`) holds only the cases' parameters and
upstream's answers, and the tests run in CI without upstream installed:

- *Tree assembly, exact.* Upstream's cluster assignments are replayed into
  our assembly (the partition function is a parameter of the pure
  builder), with the deviations below set to upstream's behaviour (single
  nodes summarized, no shrink stop): node count per layer, child sets, the
  token-bound splits and the stopping layer must be equal. They are, on all
  12 recorded trees (40-90 leaves of 150-700 words, 3,500-word bound).
- *Collapsed-tree selection, exact.* `collapsed_tree_reference`
  (`tests/fixtures/collapsed_tree.py`, not a product mode: cosine order,
  `top_k` 10, stop at the first node over the 3,500-token budget) must
  select upstream's node ids in upstream's order (it does on all 40 cases); the shipped `tree` strategy
  on the embedded store, vector search without rerank, graph buckets or
  expansion, must hold the reference's top 5 in its top 10.
- *Clustering, by agreement.* The BIC-chosen k equals upstream's
  (sklearn's) on at least 90% of the cases and is within one on all; the
  soft memberships at 0.1 reach an adjusted Rand index of at least 0.9; the
  global/local two-step finds upstream's global count and partition, and
  every local cluster of either side lies inside one global cluster. Exact
  equality is not the bar: our numpy EM is not sklearn's initialization.
  Two findings on the way: with full covariances, BIC is well posed only
  with enough points a cluster (with about 10 points a blob in 10
  dimensions upstream picked near-singleton clusters, 13 for 4 blobs of 57
  points; the recorded cases use 50-90 points a blob in 5 dimensions, where
  both sides pick the true count on all 40), and upstream's local step
  splits a single Gaussian blob of 100-130 points into 6-25 clusters, ours
  into 1-16: a BIC overfit no two EM implementations reproduce, so the
  local counts are not compared. RAPTOR's small clusters (6.7 children a
  parent in the paper) may owe as much to that overfit as to topics.
  Pull requests check the cluster count on one case per count (1 to 5) and
  two two-step cases; the weekly `fidelity` workflow runs every case, and
  checks the recorded answers against upstream's code at that commit.

Deviations from upstream, each deliberate:

| # | Upstream | Here | Why |
|---|---|---|---|
| 1 | UMAP (unseeded) | PCA by numpy SVD, sign-fixed, rounded | determinism, no numba/llvmlite, scale (Zhou et al.: GMM with UMAP did not finish in two days on the largest sets) |
| 2 | GMM, BIC over 1..49 | `kmeans` by default; `gmm` with a bounded grid, `upstream` grid optional | scale, stability on deltas |
| 3 | a one-node cluster is summarized | promoted without a call | no summary that only paraphrases one child |
| 4 | the recursion resets the reduction, threshold and tokenizer to defaults | the recursion keeps the configured ones | an upstream quirk |
| 5 | 3,500 cl100k tokens per cluster | 8,000 by the run's token counter | Claude's window; CJK-correct counts |
| 6 | 100-token chunks | the repository's chunker | one leaf set for every methodology |
| 7 | a generic English prompt | faithfulness rules, conflicts reported, the index language, a length hint | hallucination risk; multilingual requirement |
| 8 | one tree per input text | `corpus` scope by default, `document` available | multi-document corpora |
| 9 | stop at ≤ 11 nodes or 5 layers | the same, plus a stop when a level shrinks less than 20% | no near-identical levels |
| 10 | a node with no posterior above the threshold joins no cluster (it leaves the tree) | it joins its most likely cluster | every leaf stays reachable |
| 10a | GMM and k-means start from scikit-learn's k-means++ | farthest-first from a hash-ranked sample of the node ids | a delta keeps most clusters, so their summaries are reused |
| 11 | collapsed tree: cosine, `top_k` 10, 3,500 tokens | BM25 + kNN + graph signal, RRF, rerank, level quotas, passage expansion | hybrid-retrieval requirement; provenance |
| 12 | an in-memory pickle, no updates | a stored index, rebuilt with reuse, fail-closed removal, filters, tenancy | production indexing |

Where the tree keeps its state: the leaf-set digest is read from the
stored text units, not the registry, so the tree works without the
registry and is a function of the store; a summary is reused from the
live tree's nodes (`input_key`), not from extraction-store records, so the
full build's record retention is unchanged and a summary is reusable while
its cluster is in the last tree; and stale nodes are found by text-unit
lineage and their ancestors, not by document ids, which stored text units
do not hold.
