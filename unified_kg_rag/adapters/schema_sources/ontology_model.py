# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The terminology of an ontology, as the schema mapping reads it.

A reader (an RDF parser, an ontology service client) builds an
:class:`OntologyModel` of the named classes and properties it found, with
what it stated about them; :func:`.ontology_mapping.map_ontology` turns the
model into a ``SchemaDocument``. The model holds statements, not
inferences: the mapping applies its own fixed closure rules. It holds no
blank nodes: an ``owl:unionOf`` of named classes in a domain or range is
one :attr:`OntologyProperty.domains` set, and anything else anonymous is
left out by the reader.

Nothing here depends on an RDF library, so a source that is not RDF can
build the same model and get the same mapping.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from unified_kg_rag.adapters.source_files.rdf_terms import RDFS, SKOS, Text

RDFS_LABEL = RDFS + "label"
RDFS_COMMENT = RDFS + "comment"
SKOS_PREF_LABEL = SKOS + "prefLabel"
SKOS_DEFINITION = SKOS + "definition"

# An IRI has no whitespace (RFC 3987); its form is otherwise not checked.
Iri = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S+$")]
# How a property is declared: object (owl:ObjectProperty and its
# characteristics), datatype (owl:DatatypeProperty, or an rdf:Property with a
# literal range), annotation (owl:AnnotationProperty), untyped (an
# rdf:Property with no literal range). Only object and untyped properties
# can become relationship types.
PropertyKind = Literal["object", "datatype", "annotation", "untyped"]


# Annotation property IRI -> its values on a term (rdfs:label, skos:prefLabel,
# skos:definition, rdfs:comment, ...).
Annotations = dict[Iri, tuple[Text, ...]]


class OntologyTerm(BaseModel):
    """What a class and a property share."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    iri: Iri
    annotations: Annotations = Field(default_factory=dict)
    # rdfs:subClassOf / rdfs:subPropertyOf objects that are named.
    parents: frozenset[Iri] = frozenset()
    # owl:equivalentClass / owl:equivalentProperty, either direction.
    equivalents: frozenset[Iri] = frozenset()
    # owl:deprecated true.
    deprecated: bool = False


class OntologyClass(OntologyTerm):
    """A named class (``owl:Class``, ``rdfs:Class``, or a node used as one)."""


class OntologyProperty(OntologyTerm):
    """A named property.

    ``domains`` and ``ranges`` are conjunctions of unions: each set is one
    ``rdfs:domain`` statement (a class, or the classes of an
    ``owl:unionOf``) or all of a property's ``schema:domainIncludes``
    together, and the domain is what every set allows.
    """

    kind: PropertyKind = "object"
    domains: frozenset[frozenset[Iri]] = frozenset()
    ranges: frozenset[frozenset[Iri]] = frozenset()
    # owl:inverseOf, either direction.
    inverses: frozenset[Iri] = frozenset()
    # owl:SymmetricProperty.
    symmetric: bool = False


class OntologyModel(BaseModel):
    """The classes and properties of an ontology, and its prefix bindings.

    The order of ``classes`` and ``properties`` does not matter: the mapping
    sorts what it reads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    classes: tuple[OntologyClass, ...] = ()
    properties: tuple[OntologyProperty, ...] = ()
    # Prefix -> namespace IRI, as the source binds them (used to read the
    # CURIEs of the mapping options and to qualify colliding labels).
    prefixes: dict[str, Iri] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique(self) -> OntologyModel:
        # One IRI may name a class and a property (OWL 2 punning), not two
        # of either.
        for kind, terms in (("class", self.classes), ("property", self.properties)):
            seen: set[str] = set()
            for term in terms:
                if term.iri in seen:
                    raise ValueError(f"{kind} {term.iri} is declared twice")
                seen.add(term.iri)
        return self
