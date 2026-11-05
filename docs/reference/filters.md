# Filters

`--filters key:value ...` (or `RAGInput.filters`) narrows what retrieval ranks;
a pair without a key, a `:` or a value, a key given twice and a key no index the
strategy reads declares are usage errors (exit 2).
Filters compile to `term`/`terms`/`range` clauses on OpenSearch and `has`
steps on Neptune. How to use them in a query:
[Querying: Filters](../guides/query.md#filters). Filters are not an
access-control boundary:
[Scopes and namespaces](../concepts/scopes-and-namespaces.md#not-an-access-control-boundary).

## Filterable fields

Each store accepts the fields its indexer writes:

| Store | Filterable fields |
|---|---|
| Text units | `id`, `text`, `translated_text_<language>`, `community_ids`, `n_tokens`, `attr_<key>`, `attributes.filters.<key>` |
| Entities | `id`, `name`, `name.keyword`, `description`, `type`, `rank`, `confidence`, `text_unit_ids`, `in_schema`, `type_iri`, `iri`, `source_kinds`, `attr_<key>`, `attributes.<path>` |
| Relationships | `id`, `source_id`, `target_id`, `source_name`, `target_name`, `description`, `weight`, `rank`, `text_unit_ids`, `in_schema`, `predicate_iri`, `source_kinds`, `attr_<key>`, `attributes.filters.<key>` |
| Claims | `id`, `subject_id`, `object_id`, `subject_name`, `object_name`, `type`, `status`, `start_date`, `end_date`, `description`, `source_text`, `text_unit_ids`, `attr_<key>`, `attributes.filters.<key>` |
| Community reports | `id`, `community_id`, `name`, `summary`, `full_content`, `rank`, `rating`, `text_unit_ids`, `document_ids`, `level`, `depth`, `max_cut_depth`, `parent_community_id`, `child_community_ids`, `attr_<key>`, `attributes.filters.<key>` |
| Neptune entity vertices | `id`, `name`, `type`, `description`, `rank`, `confidence`, `text_unit_ids`, `community_ids`, `in_schema`, `type_iri`, `iri`, `source_kinds`, `attr_<key>` (where present) |

On OpenSearch, `attr_<key>` is entry `<key>` of a document's `filters`
metadata (an item from several documents carries all their values); on
Neptune entity vertices it is the entity's own attribute (`attr_role`).

## Values, case and dates

Entity and claim `type` and claim `status` are stored lowercase and `term`
matches exactly: filter `type:person`, not `type:PERSON`. Relationships
take no `type` filter: the relationships index stores their type (as the
graph labels the edge, `RELATED_TO` when untyped) for the context line
`Relationship: A -[TYPE]-> B`, but a `type` filter is an entity or claim
type and leaves relationships unfiltered. `in_schema`, `type_iri` and `predicate_iri` are the
[extraction schema](../guides/extraction-schema.md#what-is-stored) annotations; `iri` and
`source_kinds` (`document`, `seed_graph`) come from
[seed graphs](../guides/seed-graphs.md#retrieval). Prefer keyword or
numeric fields; a `term` on an analyzed field such as `description` matches
single lowercase tokens. A range filter takes `{"gte": ..., "lte": ...}`
through the Python API.

**Dates.** `attr_<key>`, `attributes.filters.<key>` and the
claim dates are keywords, a date-like value too, so a range on them
compares strings, which orders ISO dates (`YYYY-MM-DD`) correctly. A claim
date that reads as a calendar date (`2024-01-15`, `2024-1-5`,
`January 15th, 2024`, `Sept. 3, 2024`) is stored as `YYYY-MM-DD`, and a
coarser period as its first day in `start_date` and as its last day in
`end_date` (as the claim prompt asks: `YYYY-12-31` for the end of a
year): an ISO week (`2024-W03`) as its Monday or Sunday (an ISO week
with its weekday, `2024-W03-2`, as that day), a quarter (`Q3 2022`) as
2022-07-01 or 2022-09-30, a month (`March 2024`, `2024-03`) as
2024-03-01 or 2024-03-31 and a year (`2024`) as 2024-01-01 or
2024-12-31. So a `start_date` of `{"gte": "2022-07-01", "lte":
"2022-09-30"}` matches `Q3 2022` but `{"gte": "2022-08-01"}` does not,
and an `end_date` of `{"lte": "2024-06-30"}` does not match "until
2024". Any other value (`10/08/2024`, whose day and month order is
unknown, or `early 2024`) is stored as written and compares as text, so
only an exact filter matches it meaningfully: a range can match it
falsely (`10/08/2024` sorts before every ISO date, so `{"lte":
"2000-01-01"}` matches it). An unknown date (`UNKNOWN`, `TBD`,
`present`) is not stored, so it matches no filter, a range included.

**Value typing.** Boolean and numeric fields take their type's values: `in_schema` a
boolean; `rank`, `confidence`, `weight` and `rating` a number; `n_tokens`,
`level`, `depth` and `max_cut_depth` an integer. A string for one of them
(every `--filters` and `set-filter` value is one, and a dataset or
`RAGInput.filters` may hold one) is converted before the search, in a list
and in range bounds too, so every store matches it as it matches the typed
value: `in_schema:false` is `False` (a boolean takes `true`/`false`,
`1`/`0` or `yes`/`no`, in any case) and `rank:2` is `2`. A string its
type cannot take (`in_schema:maybe`, `rank:high`, `level:1.5`) raises
`InvalidFilterError` (a usage error, exit 2, on the command line), and so
does a typed value of another type: a number for a boolean field
(`in_schema: 1`), a boolean for a numeric one (`rank: true`) or a
fractional number for an integer one (`level: 2.5`; `2.0` is `2`). The
values of every other field are matched as given, `attr_<key>` and
`attributes.<path>` included. A number or boolean document filter value
(`attr_<key>`, `attributes.filters.<key>`) is a keyword too, held as its
JSON text, on OpenSearch and the embedded engine alike: `attr_year:2023`
and `{"attr_year": 2023}` match a document's `2023`, a `2.0` matches `2.0`
or `"2.0"` (not `2`), `true` matches `true` or `"true"`, and a range compares
text (`{"gte": "10"}` matches `9`). A document's value of another type than
an earlier one's (`"draft"` after `2023`) is stored either way. On an
OpenSearch index an earlier release built, a field mapped then by its first
number or boolean keeps that type until `indexing.reset: true` (a warning
names it).

## Per-strategy applicability

A filter applies only to the stores that declare its field
(`type:person` leaves text units unfiltered). A key no store the
selected strategy reads declares (e.g. `category`, or `type` with `naive`
or `global`) raises `InvalidFilterError`, listing the filterable keys
(`unified_kg_rag/adapters/storage/filter_schema.py`). `naive` reads text
units; `hybrid` and `mix` entities, relationships and text units (and
Neptune with `search.lightrag_search.enable_graph_expansion`); `local` and
`drift` every index the pipeline builds and Neptune; `global` community
reports (and text units with `search.global_search.text_unit_slots`,
scoped to the selected reports' communities); `simple` every index the
pipeline builds. `global` refuses the keys it scopes its own searches by:
`depth` and `max_cut_depth` (with
`search.global_search.use_dynamic_selection`, `depth` and
`community_id`), and `community_ids` with `text_unit_slots`. `id`,
`source_id` and `target_id` narrow what `local`, `drift`, `mix` and
`hybrid` fetch by id or walk to (only artifacts with a listed value come
back); a range on them is refused with every strategy (`naive` and
`global` too, which do not scope them, so a filter is valid whatever the
strategy), as is an empty list of values on any key. `auto` routes
only among the strategies it may route to that apply every key (with the
default set, `type:person` leaves `local` and `mix`) and refuses the
filters, before routing, only when none does. Neptune filters entity
vertices only.

## OpenSearch strictness

- OpenSearch applies `attr_<key>` and `attributes.<path>` strictly
  (fail-closed): an item without the attribute is excluded. Relationships,
  claims and community reports carry the filter values of their text units'
  documents, so each matches when one of them does. A vector or hybrid search
  skips an index none of whose documents carry the attribute (OpenSearch 2.13
  rejects a kNN filter on a field the index does not map).
- A document's removal clears its filter values from the entities,
  relationships and claims that outlive it: a key no remaining document
  gives is stored as null, which matches no filter.

## Graph expansion

`attr_<key>` and `attributes.filters.<key>` name document attributes. A
graph store holds none, so graph expansion keeps an expanded entity (the
seeds aside) only when its entities-index document, which carries the
filters of every document the entity comes from, passes them; under two or
more keys, one of its text units (one document's) must also pass them all.
It must also be reached from a seed, within `max_hops`, over relationships
that pass the filter (an edge only excluded documents state is not
walked), and its `Path:` follows those relationships: the walk runs over
the relationships index, one hop at a time, so it is cut after these
checks and it passes through entities a vertex filter (`type`) leaves out
of the results. Without that index
(`indexing.opensearch.build_relationship_vector_index: false`, or the
index missing at query time, logged as a WARNING) the graph is walked and
the entities-index check alone decides.

Under a filter on text units, local and mix also fill their chunk quota
with the entity-linked chunks that pass, not the top-ranked ones only.

Other `attributes.<path>` keys name an item's own attributes and apply to the
entities index only; a graph walk's entities, LightRAG's seeds included
(`local`'s and `drift`'s were found under the filters), are checked
against that index for them and for `name.keyword`, which entity
vertices do not store. The check sees what the walk kept after its own
per-hop and `top_k` cuts, so a passing neighbour ranked below them is
not reached. On Neptune, `attr_<key>` is also an entity vertex
property (`attr_role`), applied where present (a vertex without it passes),
but the entities-index check above still applies: `attr_role:buyer` selects
nothing unless documents carry a `role` filter. Every other key is strict
on both stores.
