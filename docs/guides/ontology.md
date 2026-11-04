# Ontology

`schema_source: {kind: owl}` fills the [extraction schema](./extraction-schema.md)
from an OWL or RDFS ontology:
its classes become entity types, its object properties relationship types
(each with its IRI, so `type_iri` and `predicate_iri` name the ontology's
terms) and its domains and ranges the patterns. The mapping translates, it
does not reason: it applies a fixed set of RDFS-style rules once at
start-up, with no model call and no OWL reasoner. Every option, what is
read, the mapping rules and a worked example:
[Ontology mapping reference](../reference/ontology-mapping.md).

## Install

The source needs the `ontology` extra (rdflib):

```bash
uv sync --extra ontology    # or: pip install -e '.[ontology]'
```

Without it, every CLI run with such a config (`run-ingestion`, `run-rag`,
`run-eval`, `run-prompt-tuning`) exits 2 before any model call, naming the
extra; `run-rag` needs it too, as queries read the schema to check filter
keys. On AWS, build the image with the extra and ship the ontology in it or
in S3:
[Operator Runbook: Ontology schema in the image](../operations.md#ontology-schema-in-the-image)
and [Schema sources in S3](../operations.md#schema-sources-in-s3).

## Configure

```yaml
processing:
  graph_extraction:
    schema_source:
      kind: owl
      path: ontology/procure.ttl   # a file, or a directory of ontology files
    schema_mode: align
```

- Leave `entity_types` out of the config: the ontology sets them.
- `path` is a local file or directory, relative to the config file (a
  `file://` URL works too), or an `s3://` object or `s3://bucket/prefix/`
  read as a directory ([Schema sources in S3](../operations.md#schema-sources-in-s3)).
  A directory is read recursively as one ontology (files sorted, names
  starting with `.` skipped, an OASIS XML catalog such as Protégé's
  `catalog-v001.xml` skipped). Any other URL, and a `secret_arn`, fail the
  load.
- The format follows the suffix: `.ttl` Turtle, `.rdf` / `.owl` / `.xml`
  RDF/XML, `.jsonld` / `.json-ld` JSON-LD, `.nt` N-Triples, `.nq` N-Quads,
  `.trig` TriG; other files of a directory are skipped. `options.format`
  reads every file in one format instead, whatever its suffix (in a
  directory too, so keep only ontology files there).
- Options go under `schema_source.options`
  ([Options](../reference/ontology-mapping.md#options)): which namespaces,
  classes and properties to map, how labels and descriptions are read, and
  the caps. A [worked example](../reference/ontology-mapping.md#worked-example)
  shows what a selection maps to; a selection over a cap fails the load
  and says what to narrow
  ([Prompt budget and caps](../reference/ontology-mapping.md#prompt-budget-and-caps)).

## Korean labels

For an ontology labelled in Korean:

```yaml
    schema_source:
      kind: owl
      path: ontology/procure.ttl
      options:
        label_source: label        # rdfs:label only
        languages: [ko]            # Korean first; then untagged, then any other tag
        label_case: preserve       # labels as written
        labels:
          "ex:suppliedBy": 공급받는다  # no Korean label in the ontology
```

maps the [worked example](../reference/ontology-mapping.md#worked-example)
to `사람`, `조직`, `공급업체`, `구매자` and `거래한다`,
`공급한다`, `고용한다`, `공급받는다`, `근무한다`, with the same 17 patterns.

- A term without a Korean label takes an untagged one, then another
  language, then its local name (percent-decoded, so a Hangul IRI gives a
  Hangul label). Under `preserve` that local name stays as written
  (`suppliedBy`; `upper_snake` would give `SUPPLIED_BY`), so give such
  terms a `labels` entry.
- Hangul has no case, so for a Korean label the two cases differ only in
  the separator (`공급 업체` → `공급_업체` or `공급 업체`), which label
  matching ignores; they differ for Latin labels and local names.
- Descriptions follow `languages` too (here the English comment, the only
  one). The hierarchy hints are always English: "Kind of 조직."

## Declare a base IRI

Write absolute IRIs, or declare a base: `@base`
in Turtle and TriG, `xml:base` in RDF/XML, `@base` in a JSON-LD context. A
relative IRI (`<#Vendor>`) resolves against the file's own location
(`file:///app/ontology/procure.ttl#Vendor`, or its `s3://` URL), so moving
the file, between paths or between a path and S3, or reading
it from another path renames every such term: their `type_iri` and
`predicate_iri` change and the next run re-annotates every stored item.
Two files of a directory that both write `<#Vendor>` name two classes. The
load warns `N terms have IRIs relative to their file`.

## Load cost

The files are parsed into memory once per process, at
start-up (`run-rag`: at the first filtered query), and the mapping then
runs in time and memory about linear in the ontology's size, however deep
its hierarchy. Parsing dominates, about linear in the file size
([measured](../concepts/defaults-and-evidence.md#ontology-and-rdf-parse-cost)):
at the default `max_bytes` (64 MiB) a load can take about
5 GB, within the default 8 GiB task but not alongside a large ingestion,
so lower `max_bytes` to what you load, give the task memory to match, or
map a module of the ontology (its own files) instead of all of it.

## What a change re-runs

The mapped schema is keyed, not the files, so
an incremental run reacts to what the mapping yields
([Changing the schema](./extraction-schema.md#changing-the-schema)):

- **Re-extracts**: a different label, description or pattern, or a new
  order of the types. That is an edit to a selected term's label or
  comment, a class or property entering or leaving the selection, and an
  option change that alters any of these (`hierarchy_hints` included). Settle
  the options before the first full ingest; if the ontology's editors
  relabel often, use `label_source: local_name` or pin `labels`.
- **Re-annotates only**: an IRI that changes while the labels stay
  (`ex:Vendor` renamed `ex:Supplier`, its label kept), including an
  equivalent class with a smaller IRI, which then names the type
  (`ex:Supplier owl:equivalentClass ex:Vendor` makes `type_iri`
  `ex:Supplier`). Only while the label stays, though: a merged type takes
  the best label of all its members, or under `label_source: local_name`
  (or with no label) the local name of its smallest IRI, so a new
  equivalent class with a label of its own, or with a smaller IRI under
  `local_name`, can relabel the type and re-extract. Pin such a type in
  `labels`.
- **Nothing**: re-serialising (Turtle to RDF/XML), moving or renaming the
  files (with absolute IRIs, [above](#declare-a-base-iri)), `owl:versionInfo`, `rdfs:seeAlso`,
  a datatype property's label or comment, edits to terms outside the
  selection that the closure does not use, and adding statements the
  closure already infers. With `on_collision: qualify`, bind every
  colliding namespace in `options.prefixes`: otherwise the suffix may come
  from the prefixes the files declare, which re-serialising can rename or
  drop (turning `BANK_EX` into a hash), and a relative IRI's namespace,
  and so its hash, follows the file's location.

## Preview

What extraction will use, with no model call:

```python
import yaml

from unified_kg_rag import get_config
from unified_kg_rag.application.bootstrap import resolve_extraction_schema

extraction = resolve_extraction_schema(get_config("config.yaml")).processing.graph_extraction
print(yaml.safe_dump(
    extraction.model_dump(
        mode="json",
        include={"entity_types", "relationship_types", "relationship_patterns"},
        exclude_none=True,
    ),
    allow_unicode=True,
    sort_keys=False,
))
```

Each load also logs one INFO line, `Ontology '<path>' (N files): ... entity
types, ... relationship types and ... patterns from ... classes and ...
properties; ... properties dropped, ... ends lifted, ...`; DEBUG names each
dropped, lifted, merged, skipped and qualified term. A property the
reader cannot type (only a domain, range or `owl:FunctionalProperty`)
gets its own INFO line, `Ontology: N properties not read ...`, naming the
first five.

## Freezing the mapping

Two ways to stop ontology edits from reaching
the graph:

- Save the preview's output as `schema.yaml` and load it with
  `schema_source: {kind: file, path: schema.yaml}`. It is the same schema,
  so nothing re-extracts, and the extra is no longer needed.
- `run-prompt-tuning --schema-out draft.yaml` over the `kind: owl` config
  ([Drafting a schema](./extraction-schema.md#drafting-a-schema-with-prompt-tuning)) writes the ontology's types,
  patterns and IRIs plus the relationship types the sample states that
  the ontology lacks, and its header names them and counts the patterns
  it adds for them. A type the ontology declares keeps exactly the
  ontology's patterns (none under `patterns: none`), so with nothing added
  the draft loads as the same schema and nothing re-extracts. Add the new
  types to the ontology (keep `kind: owl`), or review the draft and load
  it with `kind: file` (the added types change the schema, so that
  re-extracts).
