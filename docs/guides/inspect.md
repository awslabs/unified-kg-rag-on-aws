# Inspect and export a namespace

> 🇰🇷 한국어 문서 안내: [docs/index.ko.md](../index.ko.md)
> (this page is in English only)

`run-describe` reports what an index namespace holds and, with
`--output-path`, writes its graph to a file. It reads the stores only: no
model is called. From Python the same are `describe_namespace` and
`export_graph`.

```bash
# What the default namespace (or the config's index_value) holds
run-describe --config-path config.yaml

# Another suffix, as one JSON document
run-describe --suffix tenant_a --output-format json

# Export its graph as GraphML (Gephi, Cytoscape, networkx) or JSON Lines
run-describe --suffix tenant_a --output-path graph.graphml --export-format graphml
run-describe --suffix tenant_a --output-path graph.jsonl --max-items 10000
```

`--documents failed,quarantined` (any of `failed`, `quarantined`,
`pending`, `processed`) also lists the doc-status registry's documents of
those statuses: file, status, `error_info`, failure, retry and attempt
counts, and the pipeline id and time of the run that recorded them (needs
`providers.doc_status.enabled`).

An existing `--output-path` file is refused unless `--force`, and the
`--config-path` file and the schema file the config loads are refused
always; `export_graph` replaces an existing file only with
`overwrite=True`.

Every flag: [CLI Reference](../reference/cli.md#run-describe).

## What a description holds

- **Methodologies and strategies.** The methodologies the namespace holds,
  read from the build record the last ingestion run wrote, the search
  strategies they serve and those `auto` may route to: what
  [methodology profiles](../concepts/overview.md#methodology-profiles)
  check before a query. A namespace written before build records named
  their artifacts, or without a record (`build_record: missing`: a store
  rejected it; the next ingestion run writes it), reads as what a run of
  the config builds (`artifacts_from: config`).
- **Indices.** Each index (text units, entities, relationships, claims,
  community reports, summary nodes): its alias, the live index behind it,
  how many items it holds and the embedding model and dimension its
  vectors were made with.
- **Build record.** The last run that wrote the namespace: pipeline id,
  `full` or `incremental`, when, how many documents, the extraction
  fingerprint and settings, and the community counters.
- **Registry.** With the doc-status registry on, its records of the
  namespace by status (`processed`, `failed`, `quarantined`, ...). A
  registry the caller cannot read (no table, or no access to it, as the
  deployed query task has none) is reported unavailable, with the reason
  (`registry_unavailable`), and the rest is still described.

A namespace that holds no index exits 2 with the error on stderr (a
mistyped `--suffix`, say), which lists the namespaces the store holds, as
`run-rag` does, unless the registry holds records of it: a
first build that failed is still described (`exists: false`), so
`--documents failed,pending` lists how far it got; exporting it exits 2.
Describing creates no store file where nothing was ingested.

## Exported graph

The export streams the vector store's items a page at a time; `--max-items`
caps each kind. Its memory grows with the namespace only by the map of text
unit ids to documents it keeps to name each item's documents, and the graph
census below holds one page of the graph store (plus, on Neptune, the list
of ids it pages through).
The export's `counts`, `truncated` and `missing` (in `run-describe
--output-format json`'s `export`) name the kinds as its records do (`text`,
`entity`, ...); `indices[].kind` names an index (`text_units`, `entities`,
`community_reports`, `summary_nodes`, ...).

- **JSON Lines** (`jsonl`, the default): one object per line, `kind`
  `text` (a text unit), `entity`, `relationship`, `community` (its report:
  level, parent and child communities, title, summary), `claim` or
  `summary` (a RAPTOR summary node: level, children, the text units it
  summarizes), with every stored field but the vectors, plus `documents`:
  the documents of the text units the item cites. `--include-text` adds the
  text units' text.
- **GraphML** (`graphml`): the entities as nodes and the relationships as
  edges, the communities as nodes (`kind: community`) linked to their
  parent by `parent_of` edges; type, description, rank, weight, IRIs, text
  unit ids and documents as attributes (lists as JSON strings).

Both read the graph store too: the export's `graph` (in the
`--output-format json` document, under `export`) is a census of its entity
vertices and the relationship edges between them, each with its `count`,
`by_type` (an entity's type, an edge's label) and SHA-256 digests of the
ids in id order with their types (`digest`), their stored ranks
(`rank_digest`), the text units they cite (`provenance_digest`), their
descriptions (`description_digest`) and, for the edges, their source,
target and weight (`topology_digest`; `null` for the vertices). A JSON
Lines export ends with it, as two records of kind `graph` (ids `vertices`
and `edges`). It is read through the graph store's port, which returns
only those fields (the same for the embedded SQLite store and Neptune).
Two stores of the same graph give the same census; a lost, extra or
duplicate vertex or edge (every stored copy is counted), or a changed
type, rank, provenance, description, endpoint or weight, does not, though
the vector store's copies are unchanged. Communities are compared through
their reports.

A sample export (`--max-items`) takes no census (`graph` is `null`): the
census reads the whole graph store. `graph` is also `null` for a graph
backend that cannot list its items. When the graph store cannot be read
(unreachable, access denied), the export is still written, from the vector
store, with `graph: null` and why in `graph_unavailable` (logged as a
WARNING), as a description is without the registry; comparing it with an
export that holds the census does not converge.

Provenance is what the stores hold: a text unit names its document
(`file_path`: a document given from memory by its id, a file by the path
it was ingested from), and every other item the text units it was
extracted from. Relationships come from the relationships index: with
`indexing.opensearch.build_relationship_vector_index: false` there is
none, and the export holds no relationships (logged as a WARNING).
