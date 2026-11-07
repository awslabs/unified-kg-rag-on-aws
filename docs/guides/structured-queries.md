# Structured graph queries

> 🇰🇷 한국어판: [structured-queries.ko.md](./structured-queries.ko.md)

`--search-strategy structured` answers questions that need the structure of
the graph rather than a few passages: how many vendors supply buyers in a
country, which vendor supplies the most buyers, which products the supplier
of a named buyer makes, how two named entities are connected, whether one
supplies the other, or a property of a curated (seed) entity. A model writes
the question as a small JSON *plan* over the graph's types, relationships and
properties; code checks it, compiles it to a graph query scoped to the
query's index suffix and runs it read-only on the graph store. The answer
model then reads the result table, the entities and paths it names and the
passages they cite.

It is opt-in and unmeasured: no default selects it and `auto` does not route
to it until it is added to `search.auto_routable_strategies`. How it works
inside: [Design §6.6](../design.md#66-structured-graph-queries-structured-opt-in).

## Run it

```bash
run-rag --query "How many vendors supply buyers in Norland?" \
  --search-strategy structured --config-path config.yaml
```

Every namespace that holds the graph serves it (built with `graphrag`,
`lightrag` or `hipporag`); a `raptor`-only namespace refuses it before any
model call. It runs on all three graph stores: Neptune, a Gremlin Server
(`providers.graph.kind: gremlin`) and the embedded SQLite graph
(`backend: local`).

## What a plan can ask

| Question shape | Plan |
|---|---|
| How many vendors supply buyers in Norland? | `(v: VENDOR) -[SUPPLIES]-> (b: BUYER, seed.country = "Norland")`, count of distinct `v` |
| Which vendor supplies the most buyers? | the same pattern, distinct `b` per `v`, most first, top 1 |
| Which products does the supplier of Acme Retail make? | `(x named "Acme Retail") <-[SUPPLIES]- (s: VENDOR) -[MAKES]-> (p: PRODUCT)`, the distinct `p` |
| How is Acme Retail connected to Globex Parts? | the paths of 1 to 3 hops between the two names, shortest first |
| Does Globex Parts supply Acme Retail? | whether the pattern matches |
| Which kinds of entities are most common? | entities per `type` |
| What is the largest number of employees of a vendor? | the `max` of `seed.employees` over the vendors |

A plan is one chain of up to four nodes. A node names entity types, a name
the question gives and
conditions on properties: the core ones (`name`, `type`, `rank`,
`confidence`, `in_schema`, `iri`, `type_iri`, `source_kinds`,
`description`), extracted attributes as `attr.<key>` and seed properties as
`seed.<key>`. An edge names relationship types, a direction and 1 to
`max_hops` hops. The planner sees the declared extraction schema and a
profile of what the graph holds (types and relationships with their counts,
the observed patterns, the properties with up to `max_example_values`
sampled values of a low-cardinality string property); it never sees entity
names. A plan naming a type, relationship or property the schema does not
list is refused, and the planner is called once more with the refusal; so
is a substring or order comparison on a property that holds values of
several kinds. Under a filter the schema shows no sampled value.

A name in the plan links to the stored entities whose name it is
(spacing and legal forms aside, so `가나다상사` finds `가나다 상사`) among
the hits of a hybrid entity search, else to the best hits of a lexical
search for it. A name no entity shares a word with links nothing, and the
query falls back (`unlinked_mention`): "Does Zorblax Industries supply
Acme?" is not answered from a lookalike. The context shows which entities
each name linked to.

## When no plan answers

```yaml
search:
  structured:
    mode: fallback            # or augment
    fallback_strategy: local  # null: answer from no context
```

Under `fallback` (the default) a question the plan cannot answer runs
`fallback_strategy`: the planner's call failed or its plan stayed invalid
after the repair (`planner_error`, `invalid_plan`), it judged the question
`unanswerable`, a name linked to nothing (`unlinked_mention`), the result
was empty (`empty_result`; a count of 0 and a "no" are answers), or the store
refused or timed out the query (`execution_error`, `timeout`). The fallback
reads the entity names and themes the planner listed, so it adds no
extraction call. A fallback the namespace does not serve gives way to
`local`, then `mix`. Under `augment` the fallback runs beside the graph query
on every query and both contexts are fused, the query result first: more
recall on questions that mix counts and text, at the fallback's cost on every
query. A filter must be one both apply: the keys are checked against the
fallback that would run too, so a key `naive` or `tree` would drop is
refused before any model call, naming the fallback.

`RAGOutput.metadata["structured"]` says what happened: `status`
(`answered`, `fallback`, `failed`), `reason`, `fallback_strategy`, the
validated `plan`, `plan_description`, `operation`, `row_count`,
`truncated`, `exact`, `linked` and `linked_names` (each name's entities),
`repairs`,
`backend`, `elapsed_ms` per step and `usage` (the planner's calls).

## Exact answers, filters and seed properties

Without a document filter or a seed-property condition the graph store
computes counts and groups over every match: they are exact, and a ranking
or a list keeps the exact top, since every store cuts its page in the
plan's order. A count, an aggregate or a yes still cites the first matches
by rank as entities, with their passages. The result table is numbered
`[n]` like the other evidence sections, and its citation names the
documents those entities come from. Two kinds of
conditions cannot be evaluated inside the graph: document filters
(`attr_<key>`, `attributes.filters.<key>`; graph vertices hold none) and
seed properties (one JSON string per vertex). With either, the store
returns up to `max_scan_rows` (5,000) matches, code applies the seed
conditions, checks every bound entity against the entities index under the
document filters (and a returned path's edges against the relationships
index), and aggregates. Past the cap the answer is marked truncated: a count
reads "at least N" in the context and the answer says so. The caller's
`type`, `source_kinds`, `attr_*` and other entity keys apply to every bound
entity, `source_kinds` also to every edge of the walk
([Filters reference](../reference/filters.md#structured-queries)).

Counts reflect what extraction and the seed graphs stored: two spellings of
one entity that resolution did not merge count twice.

## Settings

```yaml
models:
  roles:
    graph_query: default       # the planner (one call a query; fast is one line away)
search:
  structured:
    max_hops: 3                # longest walk of a plan (1-4)
    rows_in_context: 50        # rows, groups or paths in the context
    max_scan_rows: 5000        # matches a store returns to code at most
    store_timeout_seconds: 10  # server-side limit of one graph query
    repair_attempts: 1         # planner calls after a refused plan
    schema_source: both        # declared, observed or both
    max_schema_tokens: 3000    # the schema text's budget
    profile_sample_vertices: 2000
    profile_ttl_seconds: 600
    max_example_values: 10     # sampled values of a low-cardinality property
    link_candidates: 3         # entities a name links to without an exact match
    supporting_chunks: 5       # passages fetched for the returned entities
  token_manager:
    type_budgets:
      query_result: 0.25       # the result table's share of the context
```

Every key: [`search.structured`](../reference/configuration.md#searchstructured).
The planner's prompt is overridable (`custom_prompts.graph_query_system` /
`_human`, keeping `{query}` and `{schema}`).

## Cost and limits

One planner call (about 3k to 4k input tokens of schema and examples, a few
hundred output tokens), at most `repair_attempts` more, one profile read per
suffix and `profile_ttl_seconds`, one entity search per name, one graph
query, one or two index fetches and one passage search, then the answer.
Under `fallback` a failed plan adds the fallback's cost.

A plan is one chain: no optional parts, unions or subqueries; a path runs
between two nodes. A dense graph can make a three-hop pattern time out (it
then falls back). Raw query text written by a model (openCypher, Gremlin) is
not supported: the plan is the contract.

## Safety

The model writes only the plan: no field can name a vertex label, a suffix,
an index or an id. The label is added by the compiler from the query's
suffix, every vertex step of the compiled Gremlin is followed by that
label, every table of the compiled SQL is bound to it, and a checker reads
the compiled form before it is sent and refuses anything but read steps of
that label. Values are bytecode arguments or bound parameters. The embedded
graph runs the query on a read-only connection, Gremlin with a server-side
time limit; the deployed query role has no Neptune write grant
([Operations](../operations.md#query-access-and-tenant-isolation)). Filters
are not access control: the suffix still has to be set from the
authenticated principal ([Scopes](../concepts/scopes-and-namespaces.md)).
With the output guardrail on, the model-written plan text is screened as an
answer is wherever it is returned: the search and answer metadata, the
result table and the reported query-result source (the console, MCP
`kg_answer` and `run-rag --output-format json` show it).
