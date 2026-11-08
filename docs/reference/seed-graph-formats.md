# Seed graph formats

The file formats of the built-in seed graph kinds (`jsonl`, `csv`, `rdf`)
and their `options`. Configuring a seed, how it meets the schema and the
documents, and its lifecycle: [Seed graphs](../guides/seed-graphs.md).
Every kind also takes `dangling` and `qualifier_property`
([Configure](../guides/seed-graphs.md#configure)); another format plugs in
as a registered seed source
([Design Doc §15](../guides/extend.md#15-extension-guide)).

## JSONL

`kind: jsonl`: one file (any name), or a directory whose
`*.jsonl` files are read recursively in sorted order (names starting with
`.` skipped). Each non-blank line is a JSON object whose `"kind"` is
`node` or `edge`:

```json
{"kind": "node", "id": "v1", "name": "Acme Supplies", "type": "VENDOR", "iri": "https://example.org/procure#acme", "description": "Industrial valve vendor.", "properties": {"country": "DE"}}
{"kind": "node", "id": "b1", "name": "Globex Buyer", "type_iri": "https://example.org/procure#Buyer"}
{"kind": "edge", "source": "v1", "target": "b1", "type": "SUPPLIES", "description": "Acme supplies Globex with valves.", "weight": 2}
```

- A node takes `id` (required, unique), `name` (default: the id), `type`
  (a label), `type_iri` (a class IRI the schema maps), `description`,
  `iri` (the node's own IRI), `confidence` and `properties`.
- An edge takes `source` and `target` (node ids), `type` or a
  `predicate_iri` a relationship type declares (one of them is required),
  `description`, `weight` (default 1) and `properties`.
- Property values are strings, numbers or booleans; `null` is dropped. An
  unknown key, a repeated node id, a non-scalar property, `NaN`, a line
  that is not a JSON object and one nested too deeply to decode fail the
  import, naming the file and line.
- A self-loop is dropped (a full build draws none). Edges with the same
  endpoints and type merge: weights add, descriptions are unioned.

## CSV

`kind: csv`: `path` is a directory (or an S3 prefix ending in
`/`) holding `nodes.csv` and `edges.csv`. The first row names the columns;
the node and edge keys above are columns, and any other column is a
string property:

```text
id,name,type,type_iri,description,iri,confidence,country
v1,Acme Supplies,VENDOR,,Industrial valve vendor.,https://example.org/procure#acme,,DE
b1,Globex Buyer,,https://example.org/procure#Buyer,,,,
```

```text
source,target,type,predicate_iri,description,weight
v1,b1,SUPPLIES,,Acme supplies Globex with valves.,2
```

```yaml
  seed_graphs:
    - name: procure
      kind: csv
      path: seed/procure               # a directory holding nodes.csv and edges.csv
      filters: {department: purchasing}
      description_properties: [country]
      options:
        property_columns: [country]
```

These files state the same graph as the JSONL example. An empty cell is an
unset value. `options.nodes` and `options.edges` name other files in the
directory (`edges: null` reads nodes only) and `options.property_columns`
keeps only the listed extra columns. UTF-8 with an optional byte-order
mark; a row with more cells than columns, a repeated column, a missing
`id` (`source`, `target`) column and a field over Python's CSV limit
(131,072 characters) fail the import.

## RDF

`kind: rdf`: the instances of an ontology (an A-Box) in Turtle,
N-Triples, N-Quads, TriG, RDF/XML or JSON-LD. It needs the `ontology` extra
(the image too: `--build-arg UV_EXTRAS="--extra ontology"`), and reads the
files exactly as the `owl` schema source does (a file, a directory, an
`s3://` object or prefix; `options.format`; nothing fetched, `owl:imports`
not followed, the same refusals: [Not followed, not read](./ontology-mapping.md#not-followed-not-read)),
within the seed's `max_bytes` (32 MiB unless set: the parsed graph is held
in memory, up to about 5 GB at that size,
[measured](../concepts/defaults-and-evidence.md#ontology-and-rdf-parse-cost)). With the procure ontology
([Worked example](./ontology-mapping.md#worked-example)) as the
schema:

```yaml
processing:
  graph_extraction:
    schema_source: {kind: owl, path: ontology/procure.ttl}
    schema_mode: strict
  seed_graphs:
    - name: procure
      kind: rdf
      path: seed/procure-abox.ttl
      options:
        languages: [en, ""]            # the default; [ko, en, ""] names Acme in Korean
```

```turtle
@prefix ex: <https://example.org/procure#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:acme a ex:Organization, ex:Vendor ;
    rdfs:label "Acme Supplies"@en, "애크미 서플라이"@ko ;
    skos:definition "Industrial valve vendor."@en ;
    ex:country "DE" ;
    ex:rating "4"^^xsd:integer ;
    ex:supplies ex:globex ;
    ex:headquarters ex:berlin .

ex:globex a ex:Buyer ;
    rdfs:label "Globex Buyer" .

ex:berlin rdfs:label "Berlin"@en .
```

gives two nodes and one edge: `Acme Supplies` (`VENDOR`, the most specific
of its two classes, with the properties `ex:country` = `DE` and
`ex:rating` = 4), `Globex Buyer` (`BUYER`) and `Acme Supplies -SUPPLIES->
Globex Buyer` with `predicate_iri` `ex:supplies`. `ex:berlin` has no type,
so it is no instance and the `ex:headquarters` edge is skipped (counted as
`objects_not_instances`).

- **Instances** are named subjects with an `rdf:type` outside the RDF,
  RDFS, OWL, XSD and SKOS vocabularies that are not classes or properties
  themselves. A node's `id` and `iri` are its IRI.
- **Names** come from `name_properties` (default `skos:prefLabel`,
  `rdfs:label`, `schema:name`) in the order of `languages` (default `[en,
  ""]`, `""` = untagged; any other tag after them), else the IRI's
  percent-decoded local name as written. **Descriptions** likewise from
  `description_properties` (default `skos:definition`, `rdfs:comment`,
  `schema:description`).
- **Types.** Each `rdf:type` is looked up among the schema's entity type
  IRIs; when several map, the one the schema declares last wins. An `owl`
  schema lists a class before its subclasses, so that is the most specific
  class (`VENDOR` above, not `ORGANIZATION`). A class no type declares
  keeps its IRI as `type_iri` with the label `labels` gives it, else its
  local name in `UPPER_SNAKE`: `strict` then drops the node, `align` and
  `off` keep it.
- **Edges** are triples whose object is an instance, typed by the
  relationship type whose IRI is the predicate, else `labels`, else the
  predicate's local name in `UPPER_SNAKE`. An object that is not an
  instance is skipped and counted (`dangling` does not apply: `rdf`
  ignores it).
- **Properties** are the other literals, keyed by CURIE through
  `options.prefixes`, the well-known prefixes, then the files' own
  bindings, and by the full IRI when none binds the namespace (N-Triples
  binds none, so its keys are full IRIs unless `options.prefixes` binds
  them). Typed numbers and booleans become numbers and booleans. A
  predicate with several values keeps the preferred string by language,
  else the smallest value. `properties` keeps only the listed predicates,
  `exclude_properties` drops some.
- **Not read**: blank nodes, reified statements, RDF-star and
  `owl:sameAs` are skipped and counted, never followed. Nothing is
  inferred: no inverse, symmetric or sub-property edge, no type a
  subclass statement implies. One INFO line per seed gives the counts.
- **IRIs relative to their file** (no `@base`) are warned about: moving
  the file renames the nodes, which then re-import.

A term of these options is a full IRI or a CURIE whose prefix comes from
`prefixes`, else the well-known `rdf`, `rdfs`, `owl`, `xsd`, `skos`,
`schema` and `dcterms`, else the files' own bindings, as for the [`owl`
schema source](ontology-mapping.md): a prefix the files bind to another
namespace than the well-known one (`schema:` to `http://schema.org/`)
fails the import until `prefixes` says which. The defaults are full IRIs,
so such a file does not affect them.

| Option | Default | Meaning |
|---|---|---|
| `prefixes` | `{}` | Prefix → namespace for the CURIEs of these options and of the property keys |
| `name_properties` | `skos:prefLabel`, `rdfs:label`, `schema:name` (`https` and `http`) | Literals a name is read from |
| `description_properties` | `skos:definition`, `rdfs:comment`, `schema:description` (`https` and `http`) | Literals a description is read from |
| `languages` | `[en, ""]` | Preferred language tags of names, descriptions and string properties |
| `labels` | `{}` | Class or property (IRI or CURIE) no schema type declares → the type label to use |
| `properties` | `null`: every other literal | The literal predicates kept as properties |
| `exclude_properties` | `[]` | Literal predicates never kept |
| `format` | `auto` | `auto` (by suffix), `turtle`, `xml`, `json-ld`, `nt`, `nquads` or `trig` |
