# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""RDF statements as an :class:`.ontology_model.OntologyModel` (rdflib).

The ``owl`` source (:mod:`.owl`) checks each file and hands its bytes to
:func:`parse`; :func:`ontology_model` then reads, from every statement of
the files together, only named nodes:

- Classes: what is typed ``owl:Class`` or ``rdfs:Class``, the ends of
  ``rdfs:subClassOf`` and ``owl:equivalentClass``, and the classes of the
  properties' domains and ranges; never an RDF, RDFS, OWL or XSD term or a
  datatype (``rdfs:Datatype``). Individuals are not read.
- Properties, by how they are declared: ``object`` (``owl:ObjectProperty``
  or one of its characteristics, an end of ``owl:inverseOf``, or a
  sub-, super- or equivalent property of an object property), ``datatype``
  (``owl:DatatypeProperty``, or an ``rdf:Property`` with a literal range),
  ``annotation`` (``owl:AnnotationProperty``) and ``untyped`` (any other
  ``rdf:Property``). ``owl:FunctionalProperty`` alone says neither, so such
  a property is not read.
- A domain or range statement is a named class or an ``owl:unionOf`` of
  named classes (anything else restricts nothing); ``schema:domainIncludes``
  (``http`` or ``https`` schema.org) together are one statement.
- Annotations are the string literals on a term; ``owl:deprecated true``
  marks it deprecated.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from typing import Any

from rdflib import Dataset, Graph, Literal, URIRef
from rdflib.collection import Collection
from rdflib.namespace import NamespaceManager
from rdflib.term import Node

from unified_kg_rag.shared.exceptions import SchemaSourceError

from .ontology_model import (
    OWL,
    RDF,
    RDFS,
    XSD,
    OntologyClass,
    OntologyModel,
    OntologyProperty,
    PropertyKind,
    Text,
)

logger = logging.getLogger(__name__)

SCHEMA_HTTP = "http://schema.org/"
SCHEMA_HTTPS = "https://schema.org/"
# Formats parsed into a dataset: their statements may sit in named graphs.
_QUAD_FORMATS = ("nquads", "trig", "json-ld")
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
_STRING_TYPES = (None, URIRef(XSD + "string"), URIRef(RDF + "langString"))

_TRIG_PREFIX = re.compile(
    r"(?:@prefix|\bPREFIX)\s+([A-Za-z][\w.-]*)?:\s*<([^>\s]*)>", re.IGNORECASE
)
# A JSON-LD term whose IRI ends so is a prefix (JSON-LD 1.1, 4.1.5).
_PREFIX_END = (":", "/", "?", "#", "[", "]", "@")


class Statements:
    """What the files state together: their triples and prefix bindings."""

    def __init__(self) -> None:
        self.graph = Graph(bind_namespaces="none")
        self._bindings: dict[str, set[str]] = {}

    def bind(self, prefixes: Iterable[tuple[str, str]]) -> None:
        for prefix, namespace in prefixes:
            self._bindings.setdefault(prefix, set()).add(namespace)

    def prefixes(self) -> dict[str, str]:
        """The prefixes the files bind alike (one bound two ways is left
        out: the options must bind it)."""
        ambiguous = sorted(p for p, ns in self._bindings.items() if len(ns) > 1)
        if ambiguous:
            logger.debug(
                "Ontology prefixes bound to several namespaces, left out: %s",
                ", ".join(ambiguous),
            )
        return {
            prefix: next(iter(namespaces))
            for prefix, namespaces in sorted(self._bindings.items())
            if len(namespaces) == 1
        }


def parse(
    statements: Statements, name: str, base: str, data: bytes, format_: str
) -> None:
    """Add what ``data``, the bytes of the file ``name`` in ``format_``,
    states.

    ``data`` is parsed as given, never re-read from its location: ``base``
    (the file's ``file://`` URI or ``s3://`` URL) is only the base of its
    relative IRIs.

    Raises:
        SchemaSourceError: ``data`` does not parse.
    """
    public_id = base
    try:
        if format_ in _QUAD_FORMATS:
            dataset = Dataset()
            dataset.namespace_manager = NamespaceManager(dataset, "none")
            dataset.parse(data=data, format=format_, publicID=public_id)
            for subject, predicate, obj, _ in dataset.quads():
                statements.graph.add((subject, predicate, obj))
        else:
            graph = Graph(bind_namespaces="none")
            graph.parse(data=data, format=format_, publicID=public_id)
            statements.graph += graph
            statements.bind((p, str(ns)) for p, ns in graph.namespaces())
    # The parsers raise many types (BadSyntax, SAXParseException, ValueError,
    # ...): each is a file that does not parse.
    except Exception as e:  # noqa: BLE001
        raise SchemaSourceError(
            f"Ontology file '{name}' does not parse as {format_}: {_first_line(e)}"
        ) from e
    # The dataset parsers bind the library's own prefixes too: read the
    # file's.
    if format_ == "trig":
        text = data.decode("utf-8", errors="replace")
        statements.bind((p or "", ns) for p, ns in _TRIG_PREFIX.findall(text))
    elif format_ == "json-ld":
        statements.bind(_json_ld_prefixes(json.loads(data)).items())


def _first_line(error: BaseException) -> str:
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return lines[0] if lines else type(error).__name__


def _json_ld_prefixes(document: Any) -> dict[str, str]:
    """The prefixes the top-level contexts of ``document`` define."""
    bindings: dict[str, str] = {}
    for top in document if isinstance(document, list) else [document]:
        if not isinstance(top, dict):
            continue
        contexts = top.get("@context")
        for context in contexts if isinstance(contexts, list) else [contexts]:
            if not isinstance(context, dict):
                continue
            for term, definition in context.items():
                iri = (
                    definition.get("@id")
                    if isinstance(definition, dict)
                    else definition
                )
                if (
                    isinstance(iri, str)
                    and not term.startswith("@")
                    and ":" not in term
                    and "://" in iri
                    and iri.endswith(_PREFIX_END)
                ):
                    bindings[term] = iri
    return bindings


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
    states about them (the rules in the module docstring)."""
    return _ModelReader(graph).read(prefixes)


def _named(nodes: Iterable[Node]) -> set[str]:
    return {str(node) for node in nodes if isinstance(node, URIRef)}


class _ModelReader:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.datatypes = self._typed(RDFS + "Datatype")
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

    def _union(self, node: Node) -> frozenset[str] | None:
        """A domain or range statement's classes: the named class, or the
        named members of an ``owl:unionOf``; None for anything else."""
        if isinstance(node, URIRef):
            return frozenset({str(node)})
        lists = list(self.graph.objects(node, URIRef(OWL + "unionOf")))
        if len(lists) != 1:
            return None
        items = list(Collection(self.graph, lists[0]))
        if not items or not all(isinstance(item, URIRef) for item in items):
            return None
        return frozenset(str(item) for item in items)

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
        return iri.startswith(XSD) or iri in _LITERAL_RANGES or iri in self.datatypes

    def _kinds(self) -> dict[str, PropertyKind]:
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
                ranges = self._objects(
                    iri,
                    RDFS + "range",
                    SCHEMA_HTTP + "rangeIncludes",
                    SCHEMA_HTTPS + "rangeIncludes",
                )
                literal = any(self._is_literal(r) for r in ranges)
                kinds[iri] = "datatype" if literal else "untyped"
        # Below, above or equivalent to an object property: one too.
        links: dict[str, set[str]] = {}
        for predicate in (RDFS + "subPropertyOf", OWL + "equivalentProperty"):
            for s, o in self.graph.subject_objects(URIRef(predicate)):
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    links.setdefault(str(s), set()).add(str(o))
                    links.setdefault(str(o), set()).add(str(s))
        queue = sorted(iri for iri, kind in kinds.items() if kind == "object")
        while queue:
            for other in sorted(links.get(queue.pop(), ())):
                if other not in kinds:
                    kinds[other] = "object"
                    queue.append(other)
        return {
            iri: kind for iri, kind in kinds.items() if not iri.startswith(_BUILT_IN)
        }

    def read(self, prefixes: dict[str, str]) -> OntologyModel:
        properties = []
        # The classes the domains and ranges name.
        ends: set[str] = set()
        for iri, kind in sorted(self._kinds().items()):
            domains = self._statements(iri, RDFS + "domain", "domainIncludes")
            ranges: frozenset[frozenset[str]] = frozenset()
            if kind in ("object", "untyped"):
                ranges = self._statements(iri, RDFS + "range", "rangeIncludes")
            else:
                # Only a relationship type's domain and range are mapped;
                # the domain of a datatype property is still a class.
                ends.update(*domains)
                domains = frozenset()
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
        class_iris = (
            self._typed(OWL + "Class", RDFS + "Class")
            | self._ends(RDFS + "subClassOf")
            | self._ends(OWL + "equivalentClass")
            | ends
        )
        classes = [
            OntologyClass(
                iri=iri,
                annotations=self._annotations(iri),
                parents=frozenset(self._objects(iri, RDFS + "subClassOf")) - {iri},
                equivalents=self._either_way(iri, OWL + "equivalentClass"),
                deprecated=self._deprecated(iri),
            )
            for iri in sorted(class_iris)
            if not iri.startswith(_BUILT_IN) and iri not in self.datatypes
        ]
        return OntologyModel(
            classes=tuple(classes), properties=tuple(properties), prefixes=prefixes
        )
