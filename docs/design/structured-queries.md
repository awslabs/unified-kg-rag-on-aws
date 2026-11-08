# Design: structured graph queries

Part of the [design reference](../design.md), which lists every section. Code cites
the section numbers: they never change.

### 6.6 Structured graph queries (`structured`, opt-in)

Counting, ranking and listing entities by type, relationship and property,
paths between two named entities and facts about curated (seed) properties
are questions a top-k context answers badly: `local` would count what its
context happens to hold. The `structured` strategy answers them by running a
query against the graph store and handing the result, with its evidence, to
the answer model. Neither upstream methodology has such a path (Microsoft
GraphRAG and LightRAG retrieve by similarity and community structure only);
the prior art is LLM-written Cypher (LangChain's `GraphCypherQAChain`,
neo4j-graphrag's `Text2CypherRetriever`, awslabs graphrag-toolkit's
BYOKG-RAG), which this design deliberately does not follow.

It is opt-in: no default selects it, `auto` routes to it only once it is
added to `search.auto_routable_strategies`, and that waits for the
structural benchmark (§9) to show a gain without a MuSiQue loss.

**The contract.** The model writes a small JSON *plan*
(`domain/models/graph_query.py`, `GraphQueryPlan`), never query text: up to
four node patterns joined by edge patterns in one line, each node naming
entity types, conditions on properties and optionally a name mentioned in
the question, plus one return clause (`rows`, `count`, `count_distinct`,
`group_count`, `aggregate`, `exists`, `path`). The plan has no field that
can name a vertex label, an index, a store, an id or a suffix. Code
validates it against a schema profile of the queried graph and compiles it
to Gremlin bytecode (Neptune and Gremlin Server) or parameterised SQL (the
embedded SQLite graph). Raw generated openCypher was rejected: it runs on
Neptune only, its multi-valued property reads lose `text_unit_ids` and
`source_kinds`, and scoping it to one suffix would need a complete
openCypher parser and rewriter.

**Facts this builds on**, rechecked at `1c80d7da` (after the methodology
profiles, the `iterative`, `ppr` and `tree` strategies, the filter-key
checks against held indexes, `query_language`, rerank metering and the
abstention contract):

- An entity vertex has the label `Entity-<suffix>[-<additional_suffix>]`
  (`graph_label`): the label is the only thing that separates suffixes in
  one cluster or SQLite file. The type is the `type` property; the
  relationship type is the edge label (SQLite: `edges.type`); edges join
  vertices of one label. Vertex lists (`text_unit_ids`, `community_ids`,
  `source_kinds`, a list attribute) are set-cardinality; an edge's lists
  are JSON strings; a dict attribute (`attr_seed_properties`, the seed
  properties) is one JSON string, stored, not indexed.
- A strategy declares `methodology` and may declare `required_artifacts`
  (§4.6). `structured` declares `Methodology.GRAPHRAG` and
  `required_artifacts = {text_units, entities, graph}`: any namespace that
  holds the graph serves it (`graphrag`, `lightrag`, `hipporag`), and one
  without (`[raptor]`) refuses it before any model call
  (`MethodologyNotIndexedError`).
- The chain checks filter keys against the indexes the strategy reads
  (`filtered_index_prefixes`), limited to those the namespace holds, and
  passes the query language (`SearchQuery.metadata["query_language"]`) and
  the held artifacts on every query.
- `SearchResult.metadata["context_sufficient"]` is the only abstention
  signal a strategy sets; the answer prompt's decline and hedge
  instructions follow it.

**Layers.**

| Layer | Module | What |
|---|---|---|
| domain | `models/graph_query.py` | the plan grammar, the profile, the validated plan, the store request and result, an in-memory `PropertyGraph` |
| domain | `retrieval/graph_query.py` | validation against the profile, the plan rendered in English, the store operation a plan needs, client-side aggregation, the reference executor, the profile builder |
| domain | `prompts/graph_query.py` | `GraphQueryPlanPrompt`, its few-shot plans and the schema text |
| ports | `graph_query.py` | `GraphQueryPort` (`adescribe`, `aexecute`) |
| adapters | `graph_query/gremlin.py`, `sqlite.py`, `checks.py` | the compilers and the static checkers |
| adapters | `retrievers/neptune_retriever.py`, `sqlite_graph_retriever.py` | the port, on their connections and error policy |
| adapters | `search_strategies/structured_search.py` | the strategy |

No application module imports a compiler; `rag_chain` changes only where
every strategy is wired (the `query_result` section's answer instruction,
the model-written plan text the output guardrail screens).

**Data flow of one query.**

1. *Profile* (`GraphQueryPort.adescribe`, cached per suffix for
   `profile_ttl_seconds`): from `profile_sample_vertices` vertices of the
   label (by id on SQLite, the store's first on Gremlin) and at most
   `profile_edges_per_vertex` out-edges of each, the entity types with their
   stored spellings, the edge labels, the observed `(type, edge, type)`
   patterns and the scalar property keys with their value kinds, including
   `attr.<key>` (extracted attributes) and `seed.<key>` (seed properties,
   decoded from their JSON), with up to `max_example_values` sampled values
   of a low-cardinality string key. No entity names: mentions are linked by
   code. The profile is read without the caller's filters, so a filtered
   query's schema shows no sampled value (it could be an excluded
   document's), and stored names are written on one line without angle
   brackets, which could close the prompt's data tags. A strategy keeps at
   most 256 suffixes' profiles (least recently used dropped). `schema_source:
   both` merges in the declared extraction schema
   (`processing.graph_extraction` inline; a `schema_source` schema reaches
   the planner through what it stored).
2. *Plan*: one call, model role `graph_query` (default tier, ceiling
   `search.output_ceilings.graph_query`), prompt `GraphQueryPlanPrompt`
   (`custom_prompts.graph_query_system`/`_human`) with the schema text
   (capped at `max_schema_tokens` by the configured token counter, least
   frequent entries cut first), the grammar, nine neutral few-shot plans
   (one of them unanswerable) and the question in `<question>` tags marked
   as data. Mentions are written as the question writes them (the query
   language, `query_language`), the string values of conditions as the data
   writes them (the index language), and a text or order comparison is
   allowed only on a property of one value kind (`contains` on strings
   alone: Gremlin fails a substring test of a number). The plan also lists the question's entity names and
   themes (`entities`, `keywords`), which the fallback strategy reads in
   place of the query extraction it would otherwise need.
3. *Validate* (`validate_plan`): names must be ones the profile lists
   (types resolved to every stored spelling by `schema_key`), variables
   unique and referenced, hops within `max_hops`, value kinds those the
   profile observed. A failure is an `InvalidGraphQueryError(field,
   reason)`; with `repair_attempts` (1) the model is called again with the
   error. `unanswerable: true` goes to the fallback.
4. *Link mentions* through the **hybrid entity search** (BM25 with the
   configured analyzers + kNN, the request's `search_type`) on the entities
   index under the caller's filters, restricted to the node's types: the
   hits whose `compact_name_key` equals the mention's (Han, Hangul and Kana
   spacing and legal forms do not matter, as for `local`'s seeds); else
   the top `link_candidates` of a **lexical** (BM25) search of the name,
   above `link_min_score`, whose names cover the mention
   (`domain/retrieval/linking.py` `choose_links`, the rule `local`'s seeds
   use too): every term of the mention's `compact_name_key` (its words,
   legal-form designators left out; a Han, Hangul or Kana word as its
   character bigrams) is a term of the entity's name, or one at least
   `processing.similarity_threshold` alike (a spelling variant). A kNN
   search returns its nearest entities for any text and BM25 every name
   sharing one word, so without a name match or a covering name a mention
   links nothing and the query falls back (`unlinked_mention`): an
   existence question about an entity the data does not name ("Zorblax
   Supplies" beside `Acme Supplies`, 라마바 상사 beside `가나다 상사`) is
   never answered from a lookalike. The linked entities' names are written into the plan's
   description (`named "Acme" = "Acme Supplies"`), so the answer sees what
   was counted.
5. *Compile, check, execute* (`GraphQueryPort.aexecute`), below.
6. *Post-process*: under a document filter, every vertex bound to a node
   (and every edge of a returned path, through the relationships index
   when it is built) is checked through the entities index
   (`_ids_passing`, the multi-key text-unit rule of `_vouch_by_entities_index`);
   seed-property conditions are applied to the decoded properties; the
   result is aggregated in code (`aggregate_bindings`).
7. *Results*: one `query_result` result (section type `query_result`, its
   own context budget share `search.token_manager.type_budgets.query_result`)
   holding the code-rendered plan and the result table, numbered `[n]` like
   any evidence section (§7.1), its citation's lineage (`text_unit_ids`) the
   units its evidence entities cite, the most cited first; one entity result
   per returned vertex (the graph projection, with `source_kinds`, `iri` and
   the entities-index document's filters, so `seed_provenance` applies);
   relationship results for a path's edges (the relationships-index
   documents when built); for an answer that names no entity (a count, an
   aggregate, a yes, groups by a property) the first `rows_in_context`
   matches by rank instead, read by a second bounded `tuples` request (or
   from the scan), so it still cites entities and passages; and the
   supporting passages, fused by RRF from
   two buckets: the text units the returned items cite, by how many cite
   each, and a hybrid text-unit search of the question restricted to them,
   reranked like `local`'s, at most `supporting_chunks`. Buckets:
   `structured_query_result`, `structured_entities`,
   `structured_relationships`, `structured_text_units`,
   `structured_text_hits` (and `structured_fallback` under `augment`).
8. *Answer*: the usual answer prompt (concise), with one instruction added
   when the context holds a query result: its numbers are computed by code
   over every match in the graph (within the request's filters), used as
   stated, never recounted from passages, and "at least" when it is marked
   truncated.

**Semantics (one definition, four executors).** A *match* is a walk
`w_0 … w_m` over the label's vertices whose vertices are pairwise distinct;
edge pattern `i` contributes `h_i ∈ [min, max]` steps, each over an edge
whose label is one of its types (any when none), in its direction (`out`:
source to target, `in`: target to source, `both`: either), and whose
`source_kinds` pass the caller's `source_kinds` filter (an edge without
the property is a document edge). Node `i` binds the vertex at the
boundary after `h_0 + … + h_{i-1}` steps and must have a `type` among its
types' stored spellings, satisfy each condition, be among its linked ids
and pass the caller's graph filters (as graph expansion applies them: an
entity's own `attributes.<key>` as its `attr_<key>` property,
`source_kinds` with the document rule, every key strictly, `id` as a list
of allowed ids). A
*binding* is the tuple of the bound vertices' `id` properties; results are
over distinct bindings, so a vertex stored twice while a label switches
vertex-id mode counts once. A condition holds when any stored value
satisfies it (a missing property fails every operator but `exists:
false`); comparisons are within one value kind. Hops' intermediate
vertices are existential: the caller's filters apply to bound vertices,
and under a document filter to a returned path's vertices and edges.

The store answers one of four operations (`plan_operation`):

| Operation | Returns | Used for |
|---|---|---|
| `count` | distinct bindings (of all or one node), exact | `count`, `count_distinct` |
| `group` | `(key, distinct bindings)` by a node's vertex or a property, ordered by count then key, `limit + 1` | `group_count` |
| `tuples` | distinct bindings of the named nodes, by the first node's `rank` (desc) then the ids, `limit + 1`, with one property value per row when asked | `rows`, `exists`, `aggregate` |
| `paths` | the walks, shortest first, `limit + 1` | `path` |

Every store cuts its page in the same order (groups by count, in the
plan's `order`, then by key; tuples by the first node's `rank`, then the
ids; paths by length), so a page shorter than the matches still holds the
plan's exact top: the group counts and the top rows are exact whenever the
plan's `limit` fits the page (`max_scan_rows`), and the answer says when
more matched. Under a document filter or a seed-property condition (which
the store cannot evaluate: vertices hold no document filters and seed
properties are one JSON string) every plan runs as `tuples` (or `paths`)
of all nodes, at most `max_scan_rows` (5,000), checked and aggregated in
code: past the cap the result is marked `truncated`, and a count reads "at
least N" in the context and in `metadata.structured`. Without either the
store aggregates and counts are exact. The reference executor
(`match_walks` and `reference_execute` in `domain/retrieval/graph_query.py`,
in the package because the benchmark generator computes its gold answers
with it) answers a plan over a `PropertyGraph`; its per-operation form
(`tests/fixtures/graph_query.py`) implements the store operations, the
caller's filters applied as the stores apply them, and the conformance
suite (`tests/integration/test_graph_query_conformance.py`) runs the same
plans on it, SQLite, Gremlin Server (`LOCAL_STORES=1`, both vertex-id
modes) and Neptune (`-m aws`), comparing values and sets.

**Safety.** The question, the history and the stored text can carry
instructions to the planner, so the plan is treated as any JSON a model may
emit. Goals: never write; never read outside the query's label; never
return what the caller's filters exclude, to the degree the other
strategies guarantee; bound each query's time and rows; leak no store
internals. Layers, each independent of the next:

1. *Grammar* (domain): closed Pydantic models (`extra="forbid"`, strict
   scalar types, length and count caps); a plan that does not parse is a
   plan failure, never an exception that escapes.
2. *Validation against the profile* (domain): only listed types, edge
   labels and property keys; never `id`, `name_key`, `build_tag`,
   `text_unit_ids`, `community_ids` or `attr_filters` as a condition or a
   key; caps of 4 nodes, 6 conditions a node, 50 values an `in`, 256
   characters a string, `max_hops` hops.
3. *Compiler* (adapters): read steps only. The label is computed by the
   retriever from `SearchQuery.suffix` and `indexing.additional_suffix`
   (`_get_label`), never taken from the request. Gremlin: every vertex step
   (`V`, `inV`, `outV`, `otherV`) is followed by `hasLabel(L)`, a
   variable-length hop is a `union` of fixed-length chains (each step
   label-checked), linked ids are a `has("id", within(...))` on `L`
   vertices, values travel as bytecode arguments. SQL: every `vertices`
   and `edges` alias carries `label = ?` bound to `L`, each node's and
   edge's constraints are the domain's own predicates run as functions
   registered for that statement (`kg_node`, `kg_edge`, `kg_key`,
   `kg_rank`, `kg_sortable`), so no JSON path or identifier is ever built
   from a plan value, and the SQL text holds no quote at all.
4. *Static checker* (`adapters/graph_query/checks.py`, independent of the
   compiler): Gremlin bytecode, nested traversals included, uses only
   allowlisted steps, starts with `V` followed by `hasLabel(L)` (an `E()`
   start is refused), has `hasLabel(L)` right after every vertex step and
   no other label, takes arguments of the allowed types only (scalars,
   predicates, the ordering tokens, nested traversals: no lambda, no `T`
   token, no `~` key), and its only source instruction is
   `OptionsStrategy` with `evaluationTimeout`. SQL starts with `SELECT` or
   `WITH`, holds no `;`, quote, comment, numbered placeholder, `OR`, comma
   join, recursive CTE or write/administration keyword, references only
   `vertices`, `edges` and its own CTEs, calls only the registered
   functions and `count`, and binds `L` at every alias's `label = ?`.
   Behind it the read-only connection installs a SQLite authorizer for the
   statement: `SELECT`, reads of `vertices` and `edges`, the registered
   functions; everything else (a schema table, recursion, another
   statement) is denied. A failure is a bug: logged at ERROR, raised as
   `GraphQueryCompileError`, never sent.
5. *Execution limits* (adapters): Gremlin `g.with_("evaluationTimeout",
   ms)` with `ms = min(structured.store_timeout_seconds,
   search.store_timeout_seconds)`, so the server stops the traversal when
   the client stops waiting, and `limit(n + 1)`; SQLite a read-only
   connection (`mode=ro`, `PRAGMA query_only`; the shared connection with
   `query_only` for `:memory:`) with a progress handler that aborts past
   the deadline, and `LIMIT n + 1`.
6. *Infrastructure* (outer, not relied on): the deployed query role has
   `neptune-db:ReadDataViaQuery` and no write or delete action. Embedded
   chains and local stores have no such layer, which is why 1 to 5 exist.
7. *Output*: the output guardrail screens the model-written plan text
   returned in `metadata.structured` and in the query-result section (as it
   screens `iterative`'s follow-up queries), and the chain gives the same
   screened text to the copies made before the screening: the answer's
   `metadata.structured` and `metadata.iterative` and the query-result
   source's text and metadata (`_refresh_screened_copies`); errors name the plan field,
   not store internals.

Why a plan cannot leave its suffix: the label is no plan value; every
vertex a traversal produces comes from a step the checker saw followed by
`hasLabel(L)`; no edge is read except as an incident edge of an `L` vertex;
linked ids come from the entities index of the same suffix and are applied
as a filter on `L` vertices; values are arguments or bound parameters.
`tests/unit/test_structured_security.py` sends prompt-injected questions,
stored names that hold instructions and hand-made plans asking for other
labels, writes and suffix escapes, and checks the compiled forms and the
stores.

**Fallback and augment.** `search.structured.mode: fallback` (default):
on a planner failure, a plan still invalid after the repair,
`unanswerable`, an unlinked mention, an empty result (no row, group or
path; a count of 0 and `exists: false` are answers) or a non-fatal
execution error (a malformed query, this query's server timeout), the
query runs `fallback_strategy` (`local`; `null` answers from no context)
and records `metadata.structured = {status: fallback, reason, ...}`. A
fallback the namespace does not serve gives way to `local`, then `mix`.
The fallback reads the plan's `entities` as its entity focus (`local`,
`drift`) or low-level keywords (with the `keywords` as high-level ones:
`mix`, `hybrid`), so no extraction call is added; a planner that failed
outright leaves the fallback the raw question (as `local` without an
entity focus). Fatal store errors still fail the query, as in every
strategy. `mode: augment` runs the fallback beside the graph query once the
plan is in, and fuses both: the query result first, then the fallback's
sections; it costs the fallback's calls on every query. A filter must be
one both paths apply: the chain checks the keys against the indexes
`structured` reads (entities, text units, relationships) and the graph,
and against the fallback that would run on the namespace
(`BaseSearchStrategy.filter_delegates`; with the namespace unknown, the
configured fallback, `local` and `mix`), so a key `naive` or `tree` would
drop is refused, naming the fallback.

**Metadata.** `RAGOutput.metadata["structured"]`: `status` (`answered`,
`fallback`, `failed`), `reason`, `mode`, `fallback_strategy`, `plan` (the
validated plan), `plan_description`, `operation`, `row_count`,
`truncated`, `exact`, `linked` (mention to entity ids), `linked_names`, `repairs`,
`backend`, `schema_cut`, `elapsed_ms` per step and `usage` (the planner's calls); the
compiled query only as a digest (`compiled_digest`), its text at DEBUG.
Spans `kg_rag.graph_query.plan` and `kg_rag.graph_query.execute` carry the
status, operation, row count, truncation and backend, never the question.

**Cost and latency.** Per query: one planner call (about 3k to 4k input
tokens of schema and examples, a few hundred output), at most
`repair_attempts` more, one profile read per suffix per TTL, one linking
search per mention, one graph query, one or two index fetches, one text
search, and the answer. Under `fallback` a failed plan adds the fallback's
cost; under `augment` every query does.

**Failure modes.** The planner's model fails or returns no JSON
(`planner_error`, `invalid_plan`); the profile read fails (the declared
schema alone, or `no_schema`); a checker failure (`compile_error`, logged
at ERROR: a bug); the store refuses or times out the query
(`execution_error`, `timeout`); a custom graph retriever that does not
implement `GraphQueryPort` makes the strategy fail with a
`ConfigurationError` naming `providers.graph.kind`, raised by its first
query before any model call (the chain builds every strategy whose filter
keys it checks, so construction does not refuse it).

**Residual risks.** Plan quality on schemas with many near-synonym types
(the observed profile and the repair mitigate it; the benchmark measures
it). Neptune's `evaluationTimeout` through bytecode and its interaction
with `StrictTimeoutValidation` is verified on Gremlin Server only; without
it a timed-out query runs on in Neptune until `neptune_query_timeout`. A
sampled profile can miss a rare type (a declared one is still allowed).
Counts reflect what extraction and the seeds stored, duplicates entity
resolution did not merge included. Variable-length patterns on hubs are
bounded by `max_hops`, distinct vertices and the server timeout, and fall
back when they time out.

**Not in this release** (decision record, maintainer 2026-11-07): raw
openCypher, as a mode or an escape hatch; the planner as a tool inside
`local`/`mix` or an agentic loop of several plans; seed properties written
as typed vertex properties (`queryable_properties`), to add once the
benchmark shows client-side filtering too slow or truncating.

---
