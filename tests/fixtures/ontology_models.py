# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A synthetic procurement ontology as an ``OntologyModel``, and the schema
it maps to by default (the worked example of the ontology mapping).

``ex:`` is ``https://example.org/procure#``: ``Organization`` with the
subclasses ``Vendor`` and ``Buyer``, and ``Person``; ``tradesWith``
(symmetric, Organization to Organization), ``supplies`` (narrower than
``tradesWith``, Vendor to Buyer), ``suppliedBy`` (its inverse), ``employs``
(Organization to Person), ``worksFor`` (its inverse) and the datatype
property ``name``. English and Korean labels, except ``suppliedBy``.
"""

from __future__ import annotations

from unified_kg_rag.adapters.schema_sources.ontology_model import (
    RDFS_COMMENT,
    RDFS_LABEL,
    OntologyClass,
    OntologyModel,
    OntologyProperty,
    Text,
)
from unified_kg_rag.domain.models.schema import (
    EntityTypeSpec,
    RelationshipPattern,
    RelationshipTypeSpec,
    SchemaDocument,
)

EX = "https://example.org/procure#"


def ends(*classes: str) -> frozenset[frozenset[str]]:
    """One domain or range statement of the ``ex:`` classes given (several:
    a union)."""
    return frozenset({frozenset(EX + name for name in classes)})


def labels(english: str, korean: str | None = None) -> dict[str, tuple[Text, ...]]:
    texts = [Text(value=english, language="en")]
    if korean is not None:
        texts.append(Text(value=korean, language="ko"))
    return {RDFS_LABEL: tuple(texts)}


def procure_model() -> OntologyModel:
    return OntologyModel(
        classes=(
            OntologyClass(
                iri=EX + "Organization", annotations=labels("Organization", "조직")
            ),
            OntologyClass(
                iri=EX + "Vendor",
                parents=frozenset({EX + "Organization"}),
                annotations={
                    **labels("Vendor", "공급업체"),
                    RDFS_COMMENT: (
                        Text(value="An organization that sells goods.", language="en"),
                    ),
                },
            ),
            OntologyClass(
                iri=EX + "Buyer",
                parents=frozenset({EX + "Organization"}),
                annotations=labels("Buyer", "구매자"),
            ),
            OntologyClass(iri=EX + "Person", annotations=labels("Person", "사람")),
        ),
        properties=(
            OntologyProperty(
                iri=EX + "tradesWith",
                symmetric=True,
                domains=ends("Organization"),
                ranges=ends("Organization"),
                annotations=labels("trades with", "거래한다"),
            ),
            OntologyProperty(
                iri=EX + "supplies",
                parents=frozenset({EX + "tradesWith"}),
                domains=ends("Vendor"),
                ranges=ends("Buyer"),
                annotations=labels("supplies", "공급한다"),
            ),
            OntologyProperty(
                iri=EX + "suppliedBy", inverses=frozenset({EX + "supplies"})
            ),
            OntologyProperty(
                iri=EX + "employs",
                domains=ends("Organization"),
                ranges=ends("Person"),
                annotations=labels("employs", "고용한다"),
            ),
            OntologyProperty(
                iri=EX + "worksFor",
                inverses=frozenset({EX + "employs"}),
                annotations=labels("works for", "근무한다"),
            ),
            OntologyProperty(iri=EX + "name", kind="datatype"),
        ),
        prefixes={"ex": EX},
    )


def _patterns(*items: str) -> list[RelationshipPattern]:
    return [RelationshipPattern.parse(item) for item in items]


PROCURE_GOLDEN = SchemaDocument(
    entity_types=[
        EntityTypeSpec(label="ORGANIZATION", iri=EX + "Organization"),
        EntityTypeSpec(
            label="BUYER", description="Kind of ORGANIZATION.", iri=EX + "Buyer"
        ),
        EntityTypeSpec(
            label="VENDOR",
            description="An organization that sells goods. Kind of ORGANIZATION.",
            iri=EX + "Vendor",
        ),
        EntityTypeSpec(label="PERSON", iri=EX + "Person"),
    ],
    relationship_types=[
        RelationshipTypeSpec(
            label="EMPLOYS", description="Inverse of WORKS_FOR.", iri=EX + "employs"
        ),
        RelationshipTypeSpec(
            label="SUPPLIED_BY",
            description="Inverse of SUPPLIES.",
            iri=EX + "suppliedBy",
        ),
        RelationshipTypeSpec(label="TRADES_WITH", iri=EX + "tradesWith"),
        RelationshipTypeSpec(
            label="SUPPLIES",
            description="Narrower than TRADES_WITH. Inverse of SUPPLIED_BY.",
            iri=EX + "supplies",
        ),
        RelationshipTypeSpec(
            label="WORKS_FOR", description="Inverse of EMPLOYS.", iri=EX + "worksFor"
        ),
    ],
    relationship_patterns=_patterns(
        "ORGANIZATION -EMPLOYS-> PERSON",
        "BUYER -EMPLOYS-> PERSON",
        "VENDOR -EMPLOYS-> PERSON",
        "BUYER -SUPPLIED_BY-> VENDOR",
        "ORGANIZATION -TRADES_WITH-> ORGANIZATION",
        "ORGANIZATION -TRADES_WITH-> BUYER",
        "ORGANIZATION -TRADES_WITH-> VENDOR",
        "BUYER -TRADES_WITH-> ORGANIZATION",
        "BUYER -TRADES_WITH-> BUYER",
        "BUYER -TRADES_WITH-> VENDOR",
        "VENDOR -TRADES_WITH-> ORGANIZATION",
        "VENDOR -TRADES_WITH-> BUYER",
        "VENDOR -TRADES_WITH-> VENDOR",
        "VENDOR -SUPPLIES-> BUYER",
        "PERSON -WORKS_FOR-> ORGANIZATION",
        "PERSON -WORKS_FOR-> BUYER",
        "PERSON -WORKS_FOR-> VENDOR",
    ),
)
