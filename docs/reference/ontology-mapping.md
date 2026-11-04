# Ontology mapping

How `schema_source: {kind: owl}` reads an OWL or RDFS ontology and maps it
to the extraction schema: the options, what is read, the mapping rules, and
a worked example. Installing, configuring, previewing and freezing the
mapping: [Ontology](../guides/ontology.md). What the schema does at
extraction: [Extraction schema](../guides/extraction-schema.md).

## Options

Options go under `schema_source.options`; an unknown option or a bad
value fails the load naming `schema_source.options.<name>`. A term is a full
IRI or a CURIE (`ex:Vendor`) whose prefix comes from `prefixes`, then the
well-known `rdf`, `rdfs`, `owl`, `xsd`, `skos`, `schema` and `dcterms`,
then the ontology's own prefix bindings (a prefix bound to two namespaces
must be set in `prefixes`). A term that names nothing in the ontology fails
the load, so a typo does not silently select nothing.

| Option | Default | Meaning |
|---|---|---|
| `namespaces` | `[]`: every namespace but RDF, RDFS, OWL, XSD, SKOS and Dublin Core | IRI prefixes of the classes and properties to map |
| `classes` | `[]`: every class | Root classes, each mapped with its subclasses |
| `exclude_classes` | `[]` | Classes left out, with their subclasses |
| `max_depth` | `null`: every level | Subclass levels below the roots (`0`: the roots only; without `classes`, the roots are the top-level classes) |
| `properties` | `[]`: every object property | Root properties, each mapped with its sub-properties |
| `exclude_properties` | `[]` | Properties left out, with their sub-properties |
| `include_deprecated` | `false` | Map terms marked `owl:deprecated true` too |
| `prefixes` | `{}` | Prefix → namespace IRI for the CURIEs of these options |
| `label_source` | `auto` | `auto`: `skos:prefLabel`, then `rdfs:label`; `label`: `rdfs:label` only; `local_name`: the IRI's local name only. A term without a label gets its local name |
| `label_case` | `upper_snake` | `upper_snake` (`supplyChain` → `SUPPLY_CHAIN`, `공급 업체` → `공급_업체`) or `preserve` (as written, punctuation made spaces) |
| `languages` | `[en, ""]` | Preferred language tags of labels and descriptions, in order (`""`: untagged; `en` also matches `en-US`, after an exact `en`); any other tag comes after them |
| `labels` | `{}` | IRI or CURIE → the label to use, as written |
| `on_collision` | `error` | Two terms of one kind with one label: fail, or `qualify`: suffix each with its namespace's prefix (`BANK_EX`, `BANK_GEO`): one `prefixes` binds, else a well-known one, else one the files bind, else a short hash of the namespace. Pin the colliding namespaces in `prefixes` (see [*What a change re-runs*](../guides/ontology.md#what-a-change-re-runs)) |
| `description_properties` | `[skos:definition, rdfs:comment]` | Annotation properties a description is read from, the first in the preferred language. Another property named here must be a property of the ontology or a literal annotation of one of its terms, or the load fails |
| `max_description_chars` | `300` | Longer descriptions are cut at a word boundary with "…" (`0`: no description from the ontology) |
| `hierarchy_hints` | `true` | Append "Kind of X.", "Narrower than P." and "Inverse of Q." to the descriptions |
| `patterns` | `closure` | `closure`: patterns from the domains and ranges; `none`: no patterns |
| `max_entity_types` | `64` | Fail when more classes are selected |
| `max_relationship_types` | `64` | Fail when more properties are mapped |
| `max_patterns` | `256` | Fail when more patterns result |
| `format` | `auto` | `auto` (by suffix), `turtle`, `xml`, `json-ld`, `nt`, `nquads` or `trig` |
| `max_bytes` | `67108864` (64 MiB) | Fail when the files hold more bytes together (see [*Load cost*](../guides/ontology.md#load-cost)) |

## What is read

Only named terms of the files' statements:

- Classes: what is typed `owl:Class` or `rdfs:Class`, the ends of
  `rdfs:subClassOf` and `owl:equivalentClass`, and the classes named in a
  relationship type's domain or range. A class named only in a datatype
  property's domain is none.
- Parents: a class's named `rdfs:subClassOf` objects, and the named
  members of an `owl:intersectionOf` it is a subclass of or equivalent to
  (`ex:Vendor owl:equivalentClass [owl:intersectionOf (ex:Organization
  [a owl:Restriction ...])]` puts `ex:Vendor` below `ex:Organization`). A
  class equivalent to an `owl:unionOf` covers it: the union's named
  members are below it.
- Relationship types: object properties (`owl:ObjectProperty` or one of
  its characteristics such as `owl:SymmetricProperty`, an end of
  `owl:inverseOf`, or a sub-, super- or equivalent property of an object
  property), an `rdf:Property` with a range that is not a literal type,
  and a sub-, super- or equivalent property of such an `rdf:Property`.
  Literal types are the XSD types, `rdfs:Literal`, `rdf:langString` and
  the other RDF literal types, an `rdfs:Datatype`, and the schema.org data
  types (`schema:Text`, `URL`, `Number`, `Integer`, `Float`, `Boolean`,
  `Date`, `DateTime`, `Time`, `CssSelectorType`, `PronounceableText`,
  `XPathType`, and anything typed `schema:DataType`, over `http` or
  `https`); they are never entity types.
- Left out: datatype properties (`owl:DatatypeProperty`, or an
  `rdf:Property` whose ranges are all literal types) and annotation
  properties (reported as skipped), a property typed only
  `owl:FunctionalProperty` or with only a domain or range (an INFO line
  names them: `N properties not read ...`), individuals, the RDF, RDFS,
  OWL and XSD vocabularies, and anonymous classes other than the class
  expressions above. Labels and descriptions are string literals.

## Mapping rules

- **Closure.** `rdfs:subClassOf` is transitive over the whole ontology,
  unselected classes included. A `subClassOf` cycle or an
  `owl:equivalentClass` makes one type, whose `iri` is its smallest
  selected IRI (properties likewise, with `owl:equivalentProperty`). A
  property's domain is what its own `rdfs:domain` statements, its
  super-properties' and its inverses' ranges all allow; the range mirrors
  it. Each statement is a class, an `owl:unionOf` of classes or all of the
  property's `schema:domainIncludes` together, and allows any of its
  classes. `owl:Thing` and `rdfs:Resource` allow anything. A symmetric
  property, or one that is its own inverse, allows its domain and range at
  both ends; with statements on one end only it takes them for the other,
  and its sub-properties and inverses inherit both (decided on the whole
  closure, so a range an inverse supplies is not overridden).
  Transitivity, functionality, cardinality and restrictions are ignored.
- **Selection.** Namespaces and deprecation first, then the `classes`
  subtrees down to `max_depth`, minus the `exclude_classes` subtrees; then
  properties the same way. Selecting a class does not select its
  ancestors; the closure still goes through unselected ones.
- **Flat hierarchy, push-down and lift.** Every selected class is a type of
  its own; no type inherits another's patterns at extraction time.
  Instead the patterns are written out per class: a property's sources are
  the selected classes that are, or are below, a class every domain
  statement allows (pushed down). When no selected class is, they are the
  nearest selected ancestors of the allowed classes (lifted). When neither
  exists, the property is dropped and the log says why. Targets follow the
  range the same way, and the patterns are every source with every target.
  A property with neither a domain nor a range gets no pattern, so even
  `strict` does not restrict it.
- **Inverses and symmetry.** An inverse pair is two relationship types,
  each with patterns in its own direction; nothing is flipped. Leave one
  out with `exclude_properties` if the graph should hold one direction. A
  symmetric property between n classes gives n × n patterns.
- **Order.** Types follow a depth-first walk of the selected hierarchy,
  siblings by label then IRI; patterns sort by relationship type, source,
  target. Only the statements decide it, never the file or its
  serialisation.

## Hierarchy and the stored annotations

Because the types are flat:

- `in_schema` is true for an entity whose stored type is a selected class.
  With `classes: ["ex:Organization"]` and `max_depth: 0`, an entity the
  model types `VENDOR` is off schema (and `strict` drops it).
- `type_iri` is the IRI of the class the stored type names, the one the
  model wrote, not its ancestors: an entity stored as `VENDOR` has
  `type_iri` `https://example.org/procure#Vendor`, not `#Organization`.
  To fetch a subtree, filter on the list of its IRIs:

  ```python
  EX = "https://example.org/procure#"
  RAGInput(query="...", search_strategy="local",
           filters={"type_iri": [EX + "Organization", EX + "Vendor", EX + "Buyer"]})
  ```

  The command line cannot say this: `run-rag --filters` takes each key
  once and `set-filter` one value, and a comma is read as part of the
  value. Use `RAGInput.filters` or the `filters` of a `run-eval` dataset
  item for a list.

- The pattern check is not hierarchy-aware: under `strict`, an
  `ORGANIZATION` pattern does not admit a `VENDOR` endpoint, which is why
  the mapping writes the subclass patterns out. An entity's stored type is
  still its most frequent one: stated twice as `VENDOR` and once as
  `ORGANIZATION`, it stays a vendor.

## Worked example

`tests/fixtures/ontologies/procure.ttl` (`ex:` is
`https://example.org/procure#`) declares `ex:Organization` with the
subclasses `ex:Vendor` and `ex:Buyer`, and `ex:Person`; `ex:tradesWith`,
symmetric, Organization → Organization; `ex:supplies`, a sub-property of
`tradesWith`, Vendor → Buyer; `ex:suppliedBy`, the inverse of `supplies`;
`ex:employs`, Organization → Person; `ex:worksFor`, the inverse of
`employs`; and `ex:name`, a datatype property. With the default options it
maps to:

| Kind | Types (description) |
|---|---|
| Entity types | `ORGANIZATION`; `BUYER` ("Kind of ORGANIZATION."); `VENDOR` ("An organization that sells goods. Kind of ORGANIZATION."); `PERSON` |
| Relationship types | `EMPLOYS` ("Inverse of WORKS_FOR."); `SUPPLIED_BY` ("Inverse of SUPPLIES."); `TRADES_WITH`; `SUPPLIES` ("Narrower than TRADES_WITH. Inverse of SUPPLIED_BY."); `WORKS_FOR` ("Inverse of EMPLOYS.") |

and 17 patterns:

- `EMPLOYS`: `ORGANIZATION`, `BUYER` and `VENDOR` → `PERSON` (3, pushed
  down from Organization).
- `SUPPLIED_BY`: `BUYER` → `VENDOR` (1, from the inverse's domain and
  range).
- `TRADES_WITH`: every pair of `ORGANIZATION`, `BUYER` and `VENDOR` (9,
  symmetric).
- `SUPPLIES`: `VENDOR` → `BUYER` (1).
- `WORKS_FOR`: `PERSON` → `ORGANIZATION`, `BUYER` and `VENDOR` (3).

`ex:name` is skipped. Under `strict`, `VENDOR -SUPPLIES-> BUYER` passes and
`BUYER -SUPPLIES-> VENDOR` is dropped. Narrowing the selection:

```yaml
    schema_source:
      kind: owl
      path: ontology/procure.ttl
      options:
        classes: ["ex:Organization"]   # ORGANIZATION, BUYER, VENDOR
        max_depth: 0                   # ORGANIZATION only
```

- `classes: ["ex:Organization"]` alone maps `ORGANIZATION`, `BUYER` and
  `VENDOR`; `EMPLOYS` and `WORKS_FOR` are dropped (`PERSON` is not
  selected), leaving 11 patterns.
- With `max_depth: 0` only `ORGANIZATION` is left, and `SUPPLIES` and
  `SUPPLIED_BY` are lifted to `ORGANIZATION -> ORGANIZATION` (3 patterns),
  which under `strict` admits either direction between two organizations.

The same ontology labelled in Korean:
[Ontology: Korean labels](../guides/ontology.md#korean-labels).

## Prompt budget and caps

The types and patterns are in the prompt of
every graph extraction call: about 5 tokens per type plus its description
(a 300-character one is about 75) and about 13 per pattern. At the default
caps that is up to about 3,300 tokens of patterns, plus about 10,000 for
128 types with 300-character descriptions. Patterns multiply: a property between m
source and n target classes gives m × n, so a broad symmetric property
dominates. A selection over a cap fails the load (exit 2) instead of being
cut, since cutting would change the prompt arbitrarily, and the message
says what to narrow; for patterns it names the five properties that yield
the most:

```text
schema_source.options: the selection yields 17 patterns, more than max_patterns (10); the most come from https://example.org/procure#tradesWith (9), ... Narrow classes or properties, exclude some, set patterns: none, or raise max_patterns
```

Narrow with `namespaces`, `classes`, `max_depth` and the `exclude_*`
options, set `patterns: none` (types only; `strict` then checks types but
no pattern), or raise the cap knowing the prompt grows.

## Not followed, not read

- `owl:imports` is not followed. Put the imported ontology's files in the
  same directory (a directory is one ontology); an import that no file
  declares is logged once as a warning.
- Nothing is fetched beyond `path` itself. A JSON-LD `@context` or
  `@import` naming a document
  is refused (inline the context, or convert the file to Turtle); XML
  external entities are not resolved (internal ones are expanded).
- OWL/XML (`.owx`, or an RDF/XML suffix whose root element is
  `owl:Ontology`), OWL functional (`.ofn`) and Manchester (`.omn`) syntax,
  OBO (`.obo`) and Notation3 (`.n3`) are refused. Convert them first:
  `robot convert --input onto.owx --output onto.ttl` (ROBOT), or Protégé's
  *File > Save as > Turtle*.
- Shapes the parsers take quadratic time over are refused before parsing,
  naming the limit: RDF/XML whose entities expand past the XML parser's
  amplification limit, an RDF/XML element of more than 10,000 pieces of
  text (`MAX_XML_TEXT_CHUNKS`: each line break and entity reference is
  one), more than 1,000 elements inside `rdf:parseType="Literal"` values
  of a file (`MAX_XML_LITERAL_ELEMENTS`), and an N-Triples or N-Quads line
  over 256 KiB (`MAX_LINE_BYTES`). Convert such a file to Turtle. An IRI
  with whitespace, which RDF/XML lets through, and a cyclic `rdf:List`
  fail the load too. A parse error names the line and what is wrong,
  never the file's text.

The `rdf` seed source reads its files the same way
([Seed graph formats: RDF](./seed-graph-formats.md#rdf)).
