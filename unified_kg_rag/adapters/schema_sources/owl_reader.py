# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""RDF statements as an :class:`.ontology_model.OntologyModel` (rdflib).

The ``owl`` source (:mod:`.owl`) checks each file and hands its bytes to
:func:`.source_files.rdf_parse.parse`; :func:`ontology_model` then reads,
from every statement of the files together, only named nodes:

- Classes: what is typed ``owl:Class`` or ``rdfs:Class``, the ends of
  ``rdfs:subClassOf`` and ``owl:equivalentClass``, the named members of the
  class expressions below, and the classes of the object and untyped
  properties' domains and ranges; never an RDF, RDFS, OWL or XSD term or a
  literal type (an ``rdfs:Datatype``, or a schema.org ``DataType`` such as
  ``schema:Text``). Individuals are not read.
- A class's parents are its named ``rdfs:subClassOf`` objects and the named
  members of an ``owl:intersectionOf`` it is a subclass of or equivalent
  to; a class equivalent to an ``owl:unionOf`` (a covering axiom) is a
  parent of the union's named members.
- Properties, by how they are declared: ``object`` (``owl:ObjectProperty``
  or one of its characteristics, an end of ``owl:inverseOf``, or a
  sub-, super- or equivalent property of an object property), ``datatype``
  (``owl:DatatypeProperty``, or an ``rdf:Property`` whose ranges are all
  literal types), ``annotation`` (``owl:AnnotationProperty``) and
  ``untyped`` (any other ``rdf:Property``, and a sub-, super- or equivalent
  property of one). ``owl:FunctionalProperty`` alone says neither, so such
  a property is not read (:func:`unread_properties` names them).
- A domain or range statement is a named class or an ``owl:unionOf`` of
  named classes (anything else restricts nothing); ``schema:domainIncludes``
  (``http`` or ``https`` schema.org) together are one statement.
- The parsers let an IRI with whitespace through (RDF/XML, JSON-LD); a
  model term with one is refused, and a prefix bound to one is left out.
- Annotations are the string literals on a term; ``owl:deprecated true``
  marks it deprecated.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import ValidationError
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.term import Node

from unified_kg_rag.adapters.source_files.rdf_terms import OWL, RDF, RDFS, XSD, Text
from unified_kg_rag.shared.exceptions import SchemaSourceError

from .ontology_model import (
    OntologyClass,
    OntologyModel,
    OntologyProperty,
    PropertyKind,
)

SCHEMA_HTTP = "http://schema.org/"
SCHEMA_HTTPS = "https://schema.org/"
# The literal value spaces: a range among them makes a datatype property.
_LITERAL_RANGES = frozenset(
    {
        RDFS + "Literal",
        RDF + "langString",
        RDF + "PlainLiteral",
        RDF + "XMLLiteral",
        RDF + "HTML",
        RDF + "JSON",
    }
)
# The schema.org data types (http and https): literal value spaces, never
# classes. Anything typed schema:DataType is one too.
_SCHEMA_DATA_TYPES = frozenset(
    namespace + name
    for namespace in (SCHEMA_HTTP, SCHEMA_HTTPS)
    for name in (
        "DataType",
        "Text",
        "URL",
        "Number",
        "Integer",
        "Float",
        "Boolean",
        "Date",
        "DateTime",
        "Time",
        "CssSelectorType",
        "PronounceableText",
        "XPathType",
    )
)
# rdf:type objects that make a property an object property.
_OBJECT_PROPERTY_TYPES = tuple(
    OWL + name
    for name in (
        "ObjectProperty",
        "SymmetricProperty",
        "TransitiveProperty",
        "InverseFunctionalProperty",
        "AsymmetricProperty",
        "ReflexiveProperty",
        "IrreflexiveProperty",
    )
)
# The ontology languages' own namespaces: never a class or a property.
_BUILT_IN = (RDF, RDFS, OWL, XSD)
_TYPE = URIRef(RDF + "type")
_FIRST, _REST, _NIL = URIRef(RDF + "first"), URIRef(RDF + "rest"), URIRef(RDF + "nil")
# Statements that make their subject a property, though they say neither
# object nor datatype (for unread_properties).
_PROPERTY_HINTS = (
    RDFS + "domain",
    RDFS + "range",
    SCHEMA_HTTP + "domainIncludes",
    SCHEMA_HTTPS + "domainIncludes",
    SCHEMA_HTTP + "rangeIncludes",
    SCHEMA_HTTPS + "rangeIncludes",
)
_STRING_TYPES = (None, URIRef(XSD + "string"), URIRef(RDF + "langString"))


def unfollowed_imports(graph: Graph) -> list[str]:
    """The ``owl:imports`` no ontology of ``graph`` is (by its IRI or
    ``owl:versionIRI``), sorted."""
    known = {str(n) for n in graph.subjects(_TYPE, URIRef(OWL + "Ontology"))}
    known |= {str(n) for n in graph.objects(None, URIRef(OWL + "versionIRI"))}
    return sorted(
        {
            str(node)
            for node in graph.objects(None, URIRef(OWL + "imports"))
            if isinstance(node, URIRef) and str(node) not in known
        }
    )


def ontology_model(graph: Graph, prefixes: dict[str, str]) -> OntologyModel:
    """The named classes and properties ``graph`` states, with what it
    states about them (the rules in the module docstring).

    Raises:
        SchemaSourceError: An ``rdf:List`` the reader follows is cyclic, or
            a term's IRI has whitespace.
    """
    try:
        return _ModelReader(graph).read(prefixes)
    except ValidationError as e:
        # The parsers let an IRI with whitespace through (RDF/XML
        # rdf:about, a JSON-LD @id); the model refuses it.
        bad = sorted(
            {
                repr(str(detail["input"]))[:120]
                for detail in e.errors(include_url=False)
                if detail["type"] == "string_pattern_mismatch"
            }
        )
        if not bad:
            raise
        raise SchemaSourceError(
            f"Ontology IRIs with whitespace: {', '.join(bad[:5])}; an IRI has "
            "none (write %20, or fix the rdf:about / @id)"
        ) from e


def unread_properties(graph: Graph) -> list[str]:
    """The named nodes ``graph`` gives a domain or range, or types
    ``owl:FunctionalProperty``, that the reader reads as no property (no
    statement says object, datatype or annotation; none relates them to a
    read property), sorted."""
    named: set[str] = set()
    for predicate in _PROPERTY_HINTS:
        named |= _named(graph.subjects(URIRef(predicate), None))
    named |= _named(graph.subjects(_TYPE, URIRef(OWL + "FunctionalProperty")))
    read = set(_ModelReader(graph).kinds())
    return sorted(iri for iri in named - read if not iri.startswith(_BUILT_IN))


def _named(nodes: Iterable[Node]) -> set[str]:
    return {str(node) for node in nodes if isinstance(node, URIRef)}


class _ModelReader:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.datatypes = self._typed(
            RDFS + "Datatype", SCHEMA_HTTP + "DataType", SCHEMA_HTTPS + "DataType"
        )
        self.symmetric = self._typed(OWL + "SymmetricProperty")

    def _typed(self, *types: str) -> set[str]:
        return {
            iri
            for name in types
            for iri in _named(self.graph.subjects(_TYPE, URIRef(name)))
        }

    def _ends(self, predicate: str) -> set[str]:
        """The named subjects and objects of ``predicate``."""
        pairs = list(self.graph.subject_objects(URIRef(predicate)))
        return _named(s for s, _ in pairs) | _named(o for _, o in pairs)

    def _objects(self, subject: str, *predicates: str) -> set[str]:
        return {
            iri
            for predicate in predicates
            for iri in _named(self.graph.objects(URIRef(subject), URIRef(predicate)))
        }

    def _either_way(self, iri: str, predicate: str) -> frozenset[str]:
        node, link = URIRef(iri), URIRef(predicate)
        return frozenset(
            _named(self.graph.objects(node, link))
            | _named(self.graph.subjects(link, node))
        ) - {iri}

    def _list(self, head: Node) -> list[Node] | None:
        """The items of the ``rdf:List`` at ``head``; None when it is not
        well formed (an item with no or two ``rdf:first`` or ``rdf:rest``).

        Raises:
            SchemaSourceError: The list is cyclic.
        """
        items: list[Node] = []
        seen: set[Node] = set()
        node = head
        while node != _NIL:
            if node in seen:
                raise SchemaSourceError(
                    "Ontology rdf:List is cyclic (its rdf:rest leads back to "
                    "itself); end the list with rdf:nil"
                )
            seen.add(node)
            firsts = list(self.graph.objects(node, _FIRST))
            rests = list(self.graph.objects(node, _REST))
            if len(firsts) != 1 or len(rests) != 1:
                return None
            items.append(firsts[0])
            node = rests[0]
        return items

    def _members(self, node: Node, operator: str) -> list[Node] | None:
        """The members of the one ``owl:<operator>`` list of ``node``, or
        None."""
        lists = list(self.graph.objects(node, URIRef(OWL + operator)))
        if len(lists) != 1:
            return None
        return self._list(lists[0])

    def _union(self, node: Node) -> frozenset[str] | None:
        """A domain or range statement's classes: the named class, or the
        named members of an ``owl:unionOf``; None for anything else."""
        if isinstance(node, URIRef):
            return frozenset({str(node)})
        items = self._members(node, "unionOf")
        if not items or not all(isinstance(item, URIRef) for item in items):
            return None
        return frozenset(str(item) for item in items)

    def _told(self) -> dict[str, set[str]]:
        """Class IRI -> the parents its class expressions state: the named
        members of an ``owl:intersectionOf`` it is a subclass of or
        equivalent to; and, for a named member of an ``owl:unionOf``
        equivalent to a class, that class."""
        parents: dict[str, set[str]] = {}
        pairs: list[tuple[Node, Node, bool]] = [
            (s, o, False)
            for s, o in self.graph.subject_objects(URIRef(RDFS + "subClassOf"))
        ]
        for s, o in self.graph.subject_objects(URIRef(OWL + "equivalentClass")):
            pairs += [(s, o, True), (o, s, True)]
        for named, expression, equivalent in pairs:
            if not isinstance(named, URIRef) or not isinstance(expression, BNode):
                continue
            for member in _named(self._members(expression, "intersectionOf") or ()):
                parents.setdefault(str(named), set()).add(member)
            if equivalent:
                for member in _named(self._members(expression, "unionOf") or ()):
                    parents.setdefault(member, set()).add(str(named))
        return parents

    def _statements(
        self, iri: str, predicate: str, includes: str
    ) -> frozenset[frozenset[str]]:
        unions = {
            union
            for node in self.graph.objects(URIRef(iri), URIRef(predicate))
            if (union := self._union(node)) is not None
        }
        allowed = self._objects(iri, SCHEMA_HTTP + includes, SCHEMA_HTTPS + includes)
        if allowed:
            unions.add(frozenset(allowed))
        return frozenset(unions)

    def _annotations(self, iri: str) -> dict[str, tuple[Text, ...]]:
        values: dict[str, set[Text]] = {}
        for predicate, obj in self.graph.predicate_objects(URIRef(iri)):
            if isinstance(obj, Literal) and obj.datatype in _STRING_TYPES:
                values.setdefault(str(predicate), set()).add(
                    Text(value=str(obj), language=obj.language or "")
                )
        return {
            predicate: tuple(sorted(texts, key=lambda t: (t.language, t.value)))
            for predicate, texts in sorted(values.items())
        }

    def _deprecated(self, iri: str) -> bool:
        return any(
            isinstance(value, Literal) and str(value).strip().lower() in ("true", "1")
            for value in self.graph.objects(URIRef(iri), URIRef(OWL + "deprecated"))
        )

    def _is_literal(self, iri: str) -> bool:
        return (
            iri.startswith(XSD)
            or iri in _LITERAL_RANGES
            or iri in _SCHEMA_DATA_TYPES
            or iri in self.datatypes
        )

    def _rdf_property_kind(self, iri: str) -> PropertyKind:
        """``datatype`` when every range of ``iri`` is a literal type,
        else ``untyped``."""
        ranges = self._objects(
            iri,
            RDFS + "range",
            SCHEMA_HTTP + "rangeIncludes",
            SCHEMA_HTTPS + "rangeIncludes",
        )
        if ranges and all(self._is_literal(r) for r in ranges):
            return "datatype"
        return "untyped"

    def kinds(self) -> dict[str, PropertyKind]:
        """Property IRI -> how it is declared; a property typed two ways
        is read as the first of object, datatype, annotation."""
        kinds: dict[str, PropertyKind] = {}
        for iri in sorted(
            self._typed(*_OBJECT_PROPERTY_TYPES) | self._ends(OWL + "inverseOf")
        ):
            kinds[iri] = "object"
        for iri in sorted(self._typed(OWL + "DatatypeProperty")):
            kinds.setdefault(iri, "datatype")
        for iri in sorted(self._typed(OWL + "AnnotationProperty")):
            kinds.setdefault(iri, "annotation")
        for iri in sorted(self._typed(RDF + "Property")):
            if iri not in kinds:
                kinds[iri] = self._rdf_property_kind(iri)
        # Below, above or equivalent to an object property: one too.
        links: dict[str, set[str]] = {}
        for predicate in (RDFS + "subPropertyOf", OWL + "equivalentProperty"):
            for s, o in self.graph.subject_objects(URIRef(predicate)):
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    links.setdefault(str(s), set()).add(str(o))
                    links.setdefault(str(o), set()).add(str(s))
        self._spread(kinds, links, "object")
        # Below, above or equivalent to an rdf:Property: one too (read by
        # its ranges, as it is).
        self._spread(kinds, links, "untyped")
        return {
            iri: kind for iri, kind in kinds.items() if not iri.startswith(_BUILT_IN)
        }

    def _spread(
        self,
        kinds: dict[str, PropertyKind],
        links: dict[str, set[str]],
        kind: PropertyKind,
    ) -> None:
        """Give every property ``links`` connects to one of ``kind`` a kind
        too: ``object`` as it is; from an ``untyped`` one, the kind an
        ``rdf:Property`` with its ranges has."""
        queue = sorted(iri for iri, k in kinds.items() if k == kind)
        while queue:
            for other in sorted(links.get(queue.pop(), ())):
                if other not in kinds:
                    kinds[other] = (
                        "object" if kind == "object" else self._rdf_property_kind(other)
                    )
                    queue.append(other)

    def read(self, prefixes: dict[str, str]) -> OntologyModel:
        properties = []
        # The classes the domains and ranges name.
        ends: set[str] = set()
        for iri, kind in sorted(self.kinds().items()):
            domains: frozenset[frozenset[str]] = frozenset()
            ranges: frozenset[frozenset[str]] = frozenset()
            # Only a relationship type's domain and range are mapped (and
            # make classes): a class only a datatype property's domain
            # names types no relationship end.
            if kind in ("object", "untyped"):
                domains = self._statements(iri, RDFS + "domain", "domainIncludes")
                ranges = self._statements(iri, RDFS + "range", "rangeIncludes")
            ends.update(*domains, *ranges)
            properties.append(
                OntologyProperty(
                    iri=iri,
                    kind=kind,
                    annotations=self._annotations(iri),
                    parents=frozenset(self._objects(iri, RDFS + "subPropertyOf"))
                    - {iri},
                    equivalents=self._either_way(iri, OWL + "equivalentProperty"),
                    deprecated=self._deprecated(iri),
                    domains=domains,
                    ranges=ranges,
                    inverses=self._either_way(iri, OWL + "inverseOf"),
                    symmetric=iri in self.symmetric,
                )
            )
        told = self._told()
        class_iris = (
            self._typed(OWL + "Class", RDFS + "Class")
            | self._ends(RDFS + "subClassOf")
            | self._ends(OWL + "equivalentClass")
            | ends
            | set(told)
            | {parent for parents in told.values() for parent in parents}
        )
        classes = [
            OntologyClass(
                iri=iri,
                annotations=self._annotations(iri),
                parents=frozenset(
                    self._objects(iri, RDFS + "subClassOf") | told.get(iri, set())
                )
                - {iri},
                equivalents=self._either_way(iri, OWL + "equivalentClass"),
                deprecated=self._deprecated(iri),
            )
            for iri in sorted(class_iris)
            if not iri.startswith(_BUILT_IN) and not self._is_literal(iri)
        ]
        return OntologyModel(
            classes=tuple(classes), properties=tuple(properties), prefixes=prefixes
        )
