# Seed graphs (bring your own knowledge graph)

A seed graph is a knowledge graph you already have (a supplier master, a
product catalog, the instances of an ontology) merged into the graph that
extraction builds from the documents. Each seed is split into virtual
documents of the run's own corpus (`seed_graphs/<name>/<k>-of-<n>.jsonl`),
so incremental runs, removal, resume and the stores treat it like any
file. No model reads a seed: its statements are parsed, never chunked by a
model, translated, extracted, gleaned or mined for claims.

Use one when the documents name entities a curated source already
describes and answers should use its descriptions, properties and IRIs,
or when the graph needs edges the text does not state (an org chart, who
supplies whom) for local search and graph expansion to walk. It does not
override the documents ([*Precedence*](#precedence-documents-decide), below), and it brings instances, not
types: the vocabulary of types is the
[extraction schema](./extraction-schema.md). The file formats:
[Seed graph formats](../reference/seed-graph-formats.md). How the import
works inside:
[Design Doc §4.5](../design.md#45-seed-graphs-as-virtual-documents).

## Configure

Seeds are declared under `processing.seed_graphs`, a list:

```yaml
processing:
  graph_extraction:
    entity_types:
      - {label: VENDOR, description: "Companies that sell goods", iri: "https://example.org/procure#Vendor"}
      - {label: BUYER, description: "Companies that buy goods", iri: "https://example.org/procure#Buyer"}
    relationship_types:
      - {label: SUPPLIES, iri: "https://example.org/procure#supplies"}
    relationship_patterns:
      - "VENDOR -SUPPLIES-> BUYER"
    schema_mode: align
  seed_graphs:
    - name: procure
      kind: jsonl
      path: seed/procure.jsonl         # relative to this file
      filters: {department: purchasing}
      description_properties: [country]
```

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Names the shard documents and the `seed_graph` document filter of everything the seed states. Up to 64 lowercase letters, digits, `-` and `_`, starting with a letter or digit; unique. Renaming it imports the seed again |
| `kind` | required | `jsonl`, `csv`, `rdf` (with the `ontology` extra), or a kind a deployment registers ([Design Doc §15](../design.md#15-extension-guide)) |
| `path` | required | A local file or directory, relative to the config file (also when set from the environment; a `file://` URL works too), an S3 object (`s3://bucket/key`) or an S3 prefix read as a directory (`s3://bucket/seed/`, the trailing `/` required), read with the run's AWS session. Other URLs fail |
| `options` | `{}` | Settings of the kind ([Seed graph formats](../reference/seed-graph-formats.md)). An unknown option fails the import |
| `secret_arn` | `null` | A Secrets Manager secret a registered service source reads; `jsonl`, `csv` and `rdf` refuse one |
| `filters` | `{}` | Document filters of the seed's items, as a document's metadata filters (string, number or boolean values). `seed_graph: <name>` is always added, so the key `seed_graph` is reserved |
| `on_name_collision` | `error` | Two nodes of the seed that are one entity by name: `error`, `merge` or `qualify` ([*Precedence*](#precedence-documents-decide)) |
| `description_properties` | `[]` | Property keys written as `key: value` lines after the description, so answers can use them |
| `confidence` | `1.0` | Confidence of the seed's statements; a node's own `confidence` wins |
| `shards` | `16` (1 to 1,024) | Virtual documents the seed is split into, by a stable hash of each node's id |
| `max_items_per_unit` | `50` (1 to 500) | Node and edge statements per text unit, the unit of lineage |
| `max_nodes` | `100000` | Fail above this many nodes |
| `max_edges` | `500000` | Fail above this many edges |
| `max_bytes` | `268435456` (256 MiB); `33554432` (32 MiB) for `rdf` | Fail when the files hold more bytes together (checked before they are read, and while reading). An RDF parser holds the whole graph in memory before `max_nodes` is checked, many times the file size (32 MiB: up to about 5 GB, [measured](../concepts/defaults-and-evidence.md#ontology-and-rdf-parse-cost)): size the task before raising it for `rdf` |
| `max_property_keys` | `64` (0 to 512) | Fail when the seed uses more distinct property keys (each is a key of `attributes.seed_properties`) |
| `max_description_chars` | `1000` (at least 50) | Cut descriptions here, at a word boundary with "…", so an item only the seed states never needs the description summarizer |

- The caps fail the import; nothing is truncated, as a cut would change the
  graph arbitrarily. A shard over 2,000 text units fails too, naming its
  largest node: raise `shards`, or, for a hub (a node and its outgoing
  edges stay in one shard, so more shards do not split it),
  `max_items_per_unit`. So does a JSONL line or CSV row over 1 MiB.
- Every name, IRI and type is bounded so that a store can index it as a
  keyword (at most 32,766 UTF-8 bytes): a node `name` (a qualified one
  included) holds at most 1,000 characters, an `iri`, `type_iri` or
  `predicate_iri` 4,000, a `type` 256. Text that is not valid Unicode (a
  JSON `\ud800` escape: a lone surrogate) fails in any field, naming the
  file and line, or the RDF subject.
- Every kind takes two options: `dangling` (an edge naming an unknown
  node: `error`, the default, or `skip`, which leaves it out and counts
  it) and `qualifier_property` (for `on_name_collision: qualify`). `rdf`
  ignores `dangling`: an object that is not an instance is always skipped
  and counted ([RDF](../reference/seed-graph-formats.md#rdf)).
- From the environment the list is JSON:
  `UKR__PROCESSING__SEED_GRAPHS='[{"name": "procure", "kind": "jsonl", "path": "seed/procure.jsonl"}]'`.
- `run-ingestion` loads every seed once at start-up, after the extraction
  schema and before any model is built: a seed that cannot be imported
  exits 2, naming the seed and, for a bad record, the file and line, as
  does a configured kind whose extra is missing (`rdf`). The other CLIs
  do not load seeds (`run-prompt-tuning` profiles the documents only) and
  need no seed kind's extra: a query image answers over an index an `rdf`
  seed was imported into without the `ontology` extra.
- The corpus may be empty when seeds are configured: an empty
  `--source-directory` is a seed-only run ([*Lifecycle*](#lifecycle)) for a source scope
  the registry holds no documents of (a new one, or a seed-only scope).
  For a scope with documents (under any spelling of it: `s3://bkt/corpus`
  and `s3://bkt/corpus/` are one scope) an empty corpus is far likelier a
  volume not mounted yet or an S3 sync that copied nothing, so the run
  fails at `document_loading` before it writes (`No source document
  reached this run, only its seed graphs, but the doc-status registry
  holds 2 documents of source scope ...`, exit 1) rather than delete them.
  With the registry off, see [*Lifecycle*](#lifecycle).
- **On AWS**, ship the seed files in the image or in S3, outside the
  ingestion source prefix:
  [Operator Runbook: Seed graphs in the image or S3](../operations.md#seed-graphs-in-the-image-or-s3).

## How seeds meet the schema

Each seed is converted against the run's
resolved [extraction schema](./extraction-schema.md), the whole seed at once, so a pattern check
always sees both endpoint types. Under `on_name_collision: qualify`, nodes
sharing a name are checked as the distinct entities they become, each by
its own type; under `merge`, as the one entity, as a full build sees it:

- A `type_iri` (or `predicate_iri`) names the declared type with that
  `iri`; one no type declares is counted (`nodes_unmapped_type_iri`) and
  the `type` label is used instead. A `type` label is written as
  extraction writes it (`vendor` becomes `VENDOR` once relationship types
  are declared).
- `off` checks nothing. `align` keeps everything and counts what falls
  outside the schema. `strict` drops off-schema nodes with their edges,
  edges of undeclared types and edges a pattern does not allow. Adding a
  `DISTRIBUTOR` node with a `SUPPLIES` edge, and a `BUYER -SUPPLIES->
  VENDOR` edge, to the [JSONL example](../reference/seed-graph-formats.md#jsonl) counts `entities_off_schema=1` and
  `relationships_off_pattern=1` under `align`, and under `strict` also
  drops the node and both edges.
- The counts are in the INFO line `Seed graph 'procure': ...`, with the
  counter names extraction uses. Seed items get `in_schema`, `type_iri`
  and `predicate_iri` like extracted ones; `iri` is the node's own IRI.

## Precedence: documents decide

A seed node and a document mention whose
names are one entity (equal after normalization, or a legal-form variant,
as extraction merges names) become one entity:

| What | Rule |
|---|---|
| Type, name, confidence | The documents' statements decide whenever any states the entity. A seed's fill only a gap: an entity no document states, or one documents name only as a relationship endpoint, with no type |
| Description | Unioned, document lines first, the seed's lines after them |
| Properties | Kept under `attributes.seed_properties`, never over an extracted attribute. Stored, not indexed: any key and any string, number or boolean value can be written, and no filter or search matches them |
| IRI | The smallest one stated; an entity stated with several is counted (`entities_with_conflicting_iris`, INFO) |
| Edges | Added. A seed edge and a document edge with the same endpoints and type are one relationship, their weights added |
| Ranks | Count document-backed edges only (an edge is seed-only when every text unit it cites is a seed's): a seed never makes an entity more important, and an entity only seed edges join has rank 0. Leiden still clusters every edge at its full weight. `indexing.neptune.entity_importance_source: degree` counts the edges at query time, seed edges included |
| Communities | A community whose members only seeds state is stored without a report (`communities_seed_only` in the stage metrics); a report on a mixed community lists the document-ranked entities first. On an incremental run a community counts as mixed when a stored member is document-backed, though the run's own graph states it only from a seed (a seed added beside unchanged documents) |

- **Homonyms.** Identity is the name: an unrelated seed node "Mercury" (a
  planet) and the documents' "Mercury" (an element) are one entity, which
  gains the seed's description line and IRI but keeps the documents'
  type. `on_name_collision` covers names within one seed only. Curate the
  names, or qualify them in the seed.
- **Spelling variants.** The rules hold per name. Names that resolve to
  one entity without being equal ("Acme" in a document, "Acme Inc" in the
  seed) still let the seed take part: where two document spellings tie,
  the seed's units can decide the name, and the variants' description
  lines interleave. With `processing.fuzzy_entity_resolution` on, a seed
  spelling can also join two document spellings.
- **`on_name_collision`**, for two nodes of one seed whose names are one
  entity but that share neither an id nor an IRI: `error` (the default)
  fails the import listing them (`1 names are shared by distinct nodes
  (ids v1, v2); set on_name_collision to merge or qualify`); `merge` makes
  them one entity (the smallest IRI kept); `qualify` appends
  ` (<qualifier>)` to every name but the one of the smallest id: the
  `qualifier_property` value, else the IRI's local name, else the id
  (`ACME supplies (v2)`). A qualified name no longer links to a document
  mention.

## Retrieval

Seed facts reach answers through the entities and
relationships, in every strategy that reads them (vector and keyword
search, graph expansion over seed edges), never as text chunks: the seed's
text units are stored for their lineage, every text search leaves them
out, and `local` and `mix` spend no chunk slot on them. `global` and the
`drift` primer read community reports: a community whose members only
seeds state has none, but the report of a mixed community describes its
seed members too (their descriptions are in its prompt), so seed facts
reach those strategies through the communities documents also state.
`naive` reads text units only.

- **Citations.** A source a seed states carries `provenance` in its
  metadata, with no chunk (`chunk_id` is `None`):

  ```python
  {"kind": "seed_graph",          # or "document+seed_graph"
   "seed_graphs": ["procure"],
   "iri": "https://example.org/procure#acme"}
  ```

  A source no seed states has no `provenance` key.
  `local` (and `drift` and `auto`, which run it) reports the graph's copy
  of an entity the entities index matched, which takes that hit's document
  filters and so names its seed graphs; an entity reached only by a graph
  walk has `"seed_graphs": []`, as graph vertices hold no document
  filters. Its metadata `source_kinds` is a list in every store.
- **Filters** ([Filters reference](../reference/filters.md)):
  - `source_kinds` (entities, relationships, Neptune entity vertices):
    `seed_graph` keeps what a seed states, `document` what a document
    states (also an entity both state, with its seed description lines);
    a list matches either. Any case. A community report carries no source
    kind and summarizes every member of its community, seed-only ones
    included, so under a filter that keeps one kind `local`, `drift` and
    `simple` read no report (DRIFT then runs one local search on the
    question itself; `simple` sweeps every other index; `auto` routes to one
    of these) and `global`, which reads reports only, refuses the key.
    `mix` and `hybrid` read no report, and `naive`, which reads text chunks
    only, refuses the key too. The key filters entities and relationships
    only: text units declare no `source_kinds`, and a filter applies only
    to the stores that declare its field, so `local`, `mix` and `hybrid`
    still return the documents' text chunks under `seed_graph` (a seed's
    own units are never text chunks).
  - `iri` (entities, Neptune entity vertices): the exact IRI.
  - `attr_seed_graph: procure` keeps what that seed states, through the
    document filters every index carries. A tenant filter such as
    `attr_department: legal` excludes what only a seed states unless the
    seed's `filters` declare the key, as for documents.
  - Seed property keys are not filterable: the vector store keeps
    `attributes.seed_properties` unmapped, so a property of any type,
    length or key name never fails a write. To filter on a value, put it in
    the seed's `filters` (one value for the whole seed), or list the key in
    `description_properties` so text and vector search see it.

```python
RAGInput(query="...", search_strategy="local", filters={"source_kinds": "document"})
RAGInput(query="...", search_strategy="mix", filters={"attr_seed_graph": "procure"})
```

## What re-indexes

The converted statements are keyed, not the files:

| Change | Next incremental run |
|---|---|
| Edit a node or edge | Imports again only the shards whose statements changed (a removal and an add, no model call). No document is extracted again |
| Change `shards` | Imports the whole seed again: every shard path changes |
| Rename the seed, change `filters` | Imports the whole seed again: every shard names them |
| Change `description_properties`, `confidence`, `max_description_chars`, `max_items_per_unit` or `on_name_collision` | Imports again the shards whose statements change |
| A schema change that changes the converted statements (a label, `schema_mode`, an IRI a `type_iri`, `predicate_iri` or RDF class maps through) | Imports again the affected shards (and extracts the documents again, as for any schema change) |
| A model or extraction setting | Extracts the documents again; the seed's shards stay unchanged, as no setting changes what they state |
| Move or re-serialise the files, change a cap the seed fits | Nothing |

A resumed run whose seed changed since the crash recomputes its stages:
the seeds' content is part of every stage's cache key.

## Lifecycle

Seeds need the doc-status registry
(`providers.doc_status.enabled`, [Incremental indexing](./incremental.md)) for these; without it every run is a full
build of the corpus and the configured seeds, and nothing tells an empty
corpus from a missing one. So with the registry off, an empty corpus
builds from the seeds alone without asking only while the index holds no
entity, relationship or text unit yet (a first seed-only build: nothing
to lose).
Once it holds some, the run asks first (`... rebuild the index 'default'
from the seed graphs alone, replacing what it holds; continue?`), as a
reset does below: `--confirm-reset`, `UKR_CONFIRM_RESET=1` or `--yes`
answers yes, and a no, or no terminal, stops the run before it writes
(exit 1). A store that cannot tell, or cannot be read, counts as holding
items.

- **Removal.** Drop the seed from the config: the next incremental run
  deletes its shards, so what only it stated goes, and entities documents
  state lose its description lines, properties, IRIs and edges.
- **Reset.** `indexing.reset: true` rebuilds the namespace from the run's
  corpus, its configured seeds included. The run asks first when the
  reset would drop what is easy to lose unawares: another source scope's
  seed documents (a seed-only scope, below), or, from a corpus of seeds
  alone, another scope's documents (a reset of a seed-only scope rebuilds
  the whole namespace from its seeds). `run-ingestion` asks on the
  terminal; `--confirm-reset` (or `UKR_CONFIRM_RESET=1`) answers yes to
  this question only, `--yes` to every question. A no, or no terminal,
  stops the run before it writes (`Pipeline stopped before it wrote
  anything: ...`, exit 1). Two resets from an empty corpus fail, whatever
  the answer: of a scope that holds documents (above), and of a scope that
  holds no record at all while other scopes hold documents (`... holds no
  record of source scope ...`): a new, moved or mistyped prefix, not a
  seed-only scope, whose earlier runs recorded its seeds. To rebuild a
  namespace from seeds alone, run the seed-only scope once without
  `indexing.reset`, then reset it. From Python, pass
  `DataIngestionPipeline(..., confirm=callable)`; without it, or on a no,
  `run` raises `ConfirmationDeclinedError` (exported by `unified_kg_rag`).
- **A seed-only scope.** A run with seeds, an empty `--source-directory`
  and a fixed `processing.document_parsing.source_scope` (say
  `seeds-procure`) keeps a seed in a scope of its own, with no model call.
  Other scopes' runs on the namespace leave it alone, except a reset.
- **One seed, two corpora.** A seed belongs to the scope of the run that
  configures it. Configured on two scopes of one namespace, it is imported
  twice: the same entities, but its edge weights count twice. Configure a
  shared seed on one scope, or give it a seed-only scope. A scope's first
  run warns when another scope already holds a seed of the same name
  (`Seed graph 'procure' is already imported by source scope(s) ...`): a
  moved or mistyped source directory looks the same.

## Cost and limits

- No extraction call (extraction, gleaning, claims, translation) reads
  a seed's text units, but seed descriptions do reach a model in three
  places: the description summarizer, for an entity documents also state
  whose merged description outgrows its threshold; the report of a
  community documents also state, which describes its seed members; and
  the answer prompt, whose context holds the retrieved entities and
  relationships. Write seed descriptions as data a model may read. A
  community only seeds state gets no report, and an item only a seed
  states is cut to `max_description_chars`, so it never needs the
  summarizer.
- Its items are embedded: two vectors per node (name and description),
  one per edge (with
  `indexing.opensearch.build_relationship_vector_index`, the default) and
  one per text unit, about 2 × nodes + edges + (nodes + edges) / 50 at the
  defaults. The 100,000-node, 500,000-edge caps are about 712,000
  embeddings. The whole seed is held in memory and in the stage cache.
- **Delta runs cluster the delta**
  ([Rebuild signals](./incremental.md#rebuild-signals)): a seed added by an incremental run
  is clustered on its own shards' entities until a reset re-clusters the
  graph. Those communities are reported when a stored member is
  document-backed, from the stored members' descriptions followed by the
  seed's lines, as the run indexes them: an edited seed's earlier lines
  are left out, and a description over
  `description_summarization.force_summary_threshold_tokens` is
  summarized first (once: the indexing reuses that summary). A full build
  summarizes all of a member's lines at once, so the incremental text is
  an equivalent summary rather than the same one; with summarization off
  both read the lines whole.
- **Translation.** Seed statements are never translated, while the
  documents' entities are extracted from the translated text: write seed
  names in `processing.translation.target_language` for them to link
  ([Multilingual corpora](./multilingual.md)).
- **Local backend.** The SQLite graph and the embedded vector store hold
  the same fields and apply the same filters.

## Preview

What the seeds convert to, with no model call and no store:

```python
from unified_kg_rag import get_config
from unified_kg_rag.application.bootstrap import load_seed_graphs, resolve_extraction_schema

config = resolve_extraction_schema(get_config("config.yaml"))
for conversion in load_seed_graphs(config):
    print(conversion.name, conversion.stats.model_dump(exclude_defaults=True))
```

For the [JSONL example](../reference/seed-graph-formats.md#jsonl) this prints `procure {'nodes_read': 2, 'edges_read':
1, 'nodes': 2, 'edges': 1, 'units': 2, 'documents': 2}`.
