# Extraction schema

Entity types, relationship types and the patterns allowed between them form
the extraction schema. With the defaults it is the `entity_types` list
alone ([Domain adaptation: Entity types](./domain-adaptation.md#entity-types)),
and extraction works as it always has. Declare relationship types
(and, if you like, patterns and ontology IRIs) when the graph should use
one fixed vocabulary: the same relationship written `works for`,
`Works_For` and `WORKS_FOR` then becomes one edge label, and every stored
item records whether it fits the schema.

The schema can also be read from an OWL or RDFS ontology
([Ontology](./ontology.md)), and a knowledge graph you already have joins
the extracted one under the same schema ([Seed graphs](./seed-graphs.md)). Every key
with its default:
[Configuration Reference](../reference/configuration.md#processinggraph_extraction).

## Which mode

`schema_mode` chooses what extraction does with the schema:

| Mode | Keeps | Use it when |
|---|---|---|
| `off` | everything, as written | the schema should only shape the prompt |
| `align` (default) | everything; declared types written as declared | you want consistent labels and can live with, or want to review, items outside the schema |
| `strict` | only items of declared types and allowed patterns | downstream users of the graph accept only the ontology's types |

`strict` costs recall: whatever the model types outside the schema is lost,
and with an entity go the unit's relationships naming it. Start with
`align`, look at what falls outside the schema (`filters={"in_schema":
False}`, [below](#filters)), widen the schema, and switch to `strict` only when little
falls outside. To measure the cost on your questions, ingest the corpus
under two index suffixes (one per mode, [Index suffix](../concepts/scopes-and-namespaces.md#index-suffix)) and
run the same `run-eval` dataset against each.

## Declaring the schema

Inline, in `processing.graph_extraction`:

```yaml
processing:
  graph_extraction:
    entity_types:
      - "PERSON: People"
      - {label: ORGANIZATION, description: "Companies", iri: "https://schema.org/Organization"}
    relationship_types:
      - "WORKS_FOR: Employment"
      - {label: PARTNERS_WITH, iri: "https://example.org/onto#partnersWith"}
    relationship_patterns:
      - "PERSON -WORKS_FOR-> ORGANIZATION"
      - {source_type: ORGANIZATION, relation: PARTNERS_WITH, target_type: ORGANIZATION}
    schema_mode: align     # off | align (default) | strict
```

- A type is `"LABEL: description"` (the description is optional) or a
  `{label, description, iri}` mapping; `iri` names the ontology class (entity
  type) or property (relationship type) it maps to. A pattern is
  `"SOURCE -RELATION-> TARGET"` or a `{source_type, relation, target_type}`
  mapping, and must name declared types. A misspelled mapping key fails the
  load.
- Labels match ignoring case, surrounding spaces and the choice of space or
  `_` between words (`works for` = `WORKS_FOR`), never by similarity:
  `PERSONS` is not `PERSON`. Two labels of one kind may not match each other.
- From the environment the lists are JSON, e.g.
  `UKR__PROCESSING__GRAPH_EXTRACTION__RELATIONSHIP_TYPES='["WORKS_FOR: Employment"]'`
  and `UKR__PROCESSING__GRAPH_EXTRACTION__SCHEMA_MODE=strict`
  ([Environment variable overrides](./configure.md#environment-variable-overrides)).
- Or keep the schema in a file: `schema_source: {kind: file, path:
  schema.yaml}` reads `relationship_types`, `relationship_patterns` and,
  optionally, `entity_types` from a YAML or JSON file of the same shape (a
  relative path is resolved against the config file). Leave those keys out
  of the config then: a file and a config that both set `entity_types` fail
  the load. The path is a local file (or a `file://` URL) or an S3 object
  (`s3://bucket/schemas/schema.yaml`, at most 16 MiB), read with the run's
  AWS session; an `https://` location fails the load (it needs a source
  registered for it). `kind: owl` reads an OWL or RDFS ontology the same
  way: a local file or directory, an S3 object, or an S3 prefix read as a
  directory (`s3://bucket/ontology/`, the trailing `/` required). Declare
  `@base` (or `xml:base`) in an ontology with relative IRIs: without it they
  resolve against the file's location, so moving the file renames them.
  See [Ontology](./ontology.md). Other formats
  (SHACL, a schema service) plug in as a registered schema source
  ([Design Doc §15](./extend.md#15-extension-guide)).
- **On AWS**, ship the schema file in the image or in S3:
  [Operator Runbook: Schema sources in S3](../operations.md#schema-sources-in-s3)
  and [Ontology schema in the image](../operations.md#ontology-schema-in-the-image).

## What extraction does

- **Prompt.** In every mode the declared relationship types and patterns
  replace the example relationship types of the extraction prompt. With no
  relationship type declared, the prompt keeps its examples.
- **`align`** keeps every entity and relationship. A type that names a
  declared one is written as declared (`works for` becomes `WORKS_FOR`)
  before ids are built, so its spellings give one edge label and one
  relationship id. Off-schema and off-pattern items are counted. With entity
  types alone (no relationship types, as in the default config) labels stay
  as written.
- **`strict`** does the same and drops, per text unit:
  - an entity whose type is empty or undeclared, with the unit's
    relationships naming it. The unit records the rejection, so gleaning
    cannot bring the entity back through a relationship; a statement of the
    same entity with a declared type keeps it.
  - a relationship of an undeclared type.
  - a relationship a pattern does not allow, when the unit lists both
    endpoints with declared types. An endpoint the unit does not list passes,
    and a reversed relationship is dropped, not flipped. A relationship type
    without patterns is not restricted.

  Gleaning follows the same rules: it does not add an off-schema entity or
  relationship, and ignores a correction to an undeclared type or to a
  direction no pattern allows.
- An endpoint's type is the unit's merged type of the entity (its most
  frequent type). Being declared does not make a type win: under `align` an
  entity stated twice as `VENDOR` and once as `ORGANIZATION` stays a
  `vendor`.
- Drops are not failures: a text unit whose items were all dropped is
  extracted (with nothing), counts as a success and is not retried.
- Claims are not checked against the schema, but a claim's subject must be
  an entity its text unit states: under `strict`, a claim about an entity
  dropped as off schema in that unit is dropped with it. An object outside
  the graph is kept as the claim's literal value, as without a schema.
- The graph extraction stage reports `entities_off_schema`,
  `entities_dropped_off_schema`, `relationships_off_schema`,
  `relationships_dropped_off_schema`,
  `relationships_dropped_off_schema_endpoint`, `relationships_off_pattern`
  and `relationships_dropped_off_pattern` in its `extraction_stats`. The
  gleaning stage reports the same counters for what gleaning proposes in
  its `gleaning_stats` (kept and counted under `align`, dropped under
  `strict`), with `corrections_ignored_off_schema`. Both are in the stage
  metrics of `<cache directory>/<pipeline_id>/pipeline_metadata.json`, and
  each stage logs one summary line; the off-schema labels themselves are
  logged at DEBUG only.
- A relationship of an undeclared type is counted in
  `relationships_off_schema` whatever its endpoints, so the count is the
  same under `align` and `strict`. Each strict drop is counted once: an
  undeclared type first, then an endpoint dropped as off schema
  (`relationships_dropped_off_schema_endpoint`), then a pattern. The counters
  count statements per text unit; an `in_schema: false` listing shows the
  merged items, one per entity or relationship.

## What is stored

Indexing writes these fields on entities and
relationships, in the graph and the vector stores:

| Field | On | Value | Written |
|---|---|---|---|
| `in_schema` | entities, relationships | whether the stored type is a declared type of that kind (an empty type is not); `true` for every item of a kind with no declared type (nothing of it is off schema) | under `strict`, and under `align` once relationship types are declared, on both kinds |
| `type_iri` | entities | the `iri` of the declared entity type the stored type names | whenever such a type has an `iri` and the mode is not `off`, with entity types alone too |
| `predicate_iri` | relationships | the `iri` of the declared relationship type | as `type_iri` |

- The values follow the stored (merged) type, so a full build and an
  incremental run store the same ones. Whether a relationship fits a
  pattern is not stored: it depends on the endpoints' types, which other
  documents can change.
- An entity named only as a relationship endpoint has no type and is off
  schema when entity types are declared, under `strict` too.
- A field that does not apply is not written at all, so a config without
  relationship types or IRIs (the default included) stores its items
  without these fields. Ids never change because of them.

## Filters

The fields are filter keys on the entities and relationships
indices and on Neptune entity vertices (graph expansion), like any other
field ([Filters reference](../reference/filters.md)):

```python
RAGInput(query="...", search_strategy="local", filters={"in_schema": False})
RAGInput(query="...", search_strategy="mix",
         filters={"type_iri": "https://schema.org/Organization"})
```

The first returns only off-schema entities (and, with `mix`, `hybrid`,
`local` or `drift`, off-schema relationships): the review list of what
`align` kept outside the schema. An item stored without the field matches
no value of it, and so does an index that does not map the field yet (the
next ingestion run maps it; no query fails for it).

A filter on a field the configured schema writes on no item is refused
before any model call (`InvalidFilterError`), as it could only empty the
results: `in_schema` under `off` or under `align` with entity types alone,
`type_iri` when no entity type declares an `iri` (or the mode is `off`),
`predicate_iri` likewise for relationship types. The check reads the
query's config, its schema source included; when the source cannot be
read, the key is not checked and a warning is logged once.

## Changing the schema

What the next incremental run does depends on
whether the model sees the change
([Incremental indexing: Workflows](./incremental.md#workflows)):

- **Re-extracts its scope**: a change the prompt or the schema check
  reads. That is a type's label or description, a pattern, `strict` on or
  off, and `off` against `align` once relationship types are declared.
  Items of other index scopes keep what their own runs extracted (labels as
  the model wrote them, off-schema items `strict` would drop) until those
  scopes run again or `indexing.reset: true` rebuilds the corpus.
- **Re-annotates only**: an IRI added, changed or removed, `off` against
  `align` with entity types alone, or another form of the same item
  (`"GENE: A gene"` or `{label: GENE, description: A gene}`; a pattern as a
  string or a mapping). Nothing is extracted or embedded again. The run
  rewrites `in_schema`, `type_iri` and `predicate_iri` of every stored
  entity and relationship of the index suffix, of every scope, from their
  stored types, and logs one INFO line with the counts; the stores then
  hold what a full build writes. The registry records which annotations
  the stores of each index namespace hold, once for all its scopes: the
  rewrite runs once per namespace, whichever scope's run finds the
  schema changed (a schema reverted on one scope after another scope's run
  is rewritten too), and a new scope of an annotated namespace reads
  nothing. A namespace without this record is read once.
  The stores are read in bounded pages. A rewrite that does not finish (a
  killed run, a failed write, a store that cannot be read) warns, leaves
  the run going, and is made again by the next run, under any schema. A
  custom store adapter that cannot list its items is skipped with a
  warning; `indexing.reset: true` rewrites it.
- **Changes nothing**: moving a schema file without editing it (the content
  is keyed, not the path). Upgrading with the defaults, or with entity
  types written as strings and no IRI, changes no cache key, extraction
  record or stored item; an incremental run maps the new fields on its live
  OpenSearch indices before its first write.

## Drafting a schema with prompt tuning

`run-prompt-tuning --schema-out
schema.yaml` also writes a schema file draft (see
[Domain adaptation: Prompt tuning](./domain-adaptation.md#prompt-tuning-run-prompt-tuning)):

1. Run it over a representative sample of the corpus.
2. Review the draft: it lists the profile's entity types (the configured
   ones when the config declares them) and the relationship types the
   tuning examples state most often, with a count
   each, and a pattern only where the examples support it. Merge synonyms,
   drop noise, add descriptions and IRIs. To refine a schema the config
   already loads, write the draft to another file (the command refuses to
   overwrite the loaded one): the draft keeps every declared type and
   pattern with its description and IRI, so you merge only what it adds.
3. Load it with `schema_source: {kind: file, path: schema.yaml}` and
   `schema_mode: align` (the draft comes from a few chunks, so `strict`
   would drop what they did not show), and remove `entity_types` from the
   config if the draft sets them.
4. Ingest, review the off-schema items with `filters={"in_schema": False}`
   and the stage counters, extend the file, re-run.

Over a `kind: owl` config the draft adds to the ontology's types instead:
[Ontology: Freezing the mapping](./ontology.md#freezing-the-mapping).

## Local backend

`backend: local` behaves the same: the SQLite graph
store and the embedded vector store hold the same fields, the same filters
apply (graph expansion included), and live embedded indices gain the
fields the same way.
