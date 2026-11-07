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

Every flag: [CLI Reference](../reference/cli.md#run-describe).

## What a description holds

- **Methodologies and strategies.** The methodologies the namespace holds,
  read from the build record the last ingestion run wrote, the search
  strategies they serve and those `auto` may route to: what
  [methodology profiles](../concepts/overview.md#methodology-profiles)
  check before a query. A namespace written before build records named
  their artifacts reads as what a run of the config builds
  (`artifacts_from: config`).
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
mistyped `--suffix`, say); describing creates no store file where
nothing was ingested.

## Exported graph

The export streams the vector store's items a page at a time, so its
memory does not grow with the namespace; `--max-items` caps each kind.

- **JSON Lines** (`jsonl`, the default): one object per line, `kind`
  `text_unit`, `entity`, `relationship`, `community` (its report: level,
  parent and child communities, title, summary) or `claim`, with every
  stored field but the vectors, plus `documents`: the documents of the
  text units the item cites. `--include-text` adds the text units' text.
- **GraphML** (`graphml`): the entities as nodes and the relationships as
  edges, the communities as nodes (`kind: community`) linked to their
  parent by `parent_of` edges; type, description, rank, weight, IRIs, text
  unit ids and documents as attributes (lists as JSON strings).

Provenance is what the stores hold: a text unit names its document
(`file_path`: a document given from memory by its id, a file by the path
it was ingested from), and every other item the text units it was
extracted from. Relationships come from the relationships index: with
`indexing.opensearch.build_relationship_vector_index: false` there is
none, and the export holds no relationships (logged as a WARNING).
