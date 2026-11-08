# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A synthetic procurement seed graph (a vendor supplying a buyer), as JSONL
records, as CSV files, and as the ``SeedGraph`` both state; and the shard
documents of a seed with the text units and records the pipeline makes of
them (for the incremental properties)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from hypothesis import strategies as st

from unified_kg_rag.domain.ingestion.contributions import (
    RECORD_KEYS_ATTRIBUTE,
    record_key,
)
from unified_kg_rag.domain.ingestion.seed_graph import (
    SEED_RECORD_FINGERPRINT,
    SeedConversion,
    SeedConversionSettings,
    SeedEdgeStatement,
    SeedNodeStatement,
    convert_seed,
    parse_seed_unit,
    render_statement,
    seed_text_units,
)
from unified_kg_rag.domain.models import (
    Document,
    ExtractionSchema,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.domain.models.config import SeedNameCollision
from unified_kg_rag.domain.models.seed_graph import SeedEdge, SeedGraph, SeedNode

EX = "https://example.org/procure#"

NODES: list[dict[str, Any]] = [
    {
        "kind": "node",
        "id": "v1",
        "name": "Acme Supplies",
        "type": "VENDOR",
        "iri": EX + "acme",
        "description": "Industrial valve vendor.",
        "properties": {"country": "DE", "rating": None},
    },
    {"kind": "node", "id": "b1", "name": "Globex Buyer", "type_iri": EX + "Buyer"},
]
EDGES: list[dict[str, Any]] = [
    {
        "kind": "edge",
        "source": "v1",
        "target": "b1",
        "type": "SUPPLIES",
        "description": "Acme supplies Globex with valves.",
        "weight": 2,
    }
]
NODES_CSV = (
    "id,name,type,type_iri,description,iri,confidence,country\n"
    f"v1,Acme Supplies,VENDOR,,Industrial valve vendor.,{EX}acme,,DE\n"
    f"b1,Globex Buyer,,{EX}Buyer,,,,\n"
)
EDGES_CSV = (
    "source,target,type,predicate_iri,description,weight\n"
    "v1,b1,SUPPLIES,,Acme supplies Globex with valves.,2\n"
)
# What both files above state.
PROCURE = SeedGraph(
    nodes=(
        SeedNode(
            id="v1",
            name="Acme Supplies",
            type="VENDOR",
            iri=EX + "acme",
            description="Industrial valve vendor.",
            properties={"country": "DE"},
        ),
        SeedNode(id="b1", name="Globex Buyer", type_iri=EX + "Buyer"),
    ),
    edges=(
        SeedEdge(
            source="v1",
            target="b1",
            type="SUPPLIES",
            description="Acme supplies Globex with valves.",
            weight=2,
        ),
    ),
)


def lines(*records: dict[str, Any]) -> str:
    return "".join(json.dumps(record) + "\n" for record in records)


def seed_shards(
    name: str,
    nodes: Sequence[SeedNode],
    edges: Sequence[SeedEdge],
    schema: ExtractionSchema,
    *,
    shards: int = 1,
    max_items_per_unit: int = 2,
    on_name_collision: SeedNameCollision = "merge",
) -> dict[str, Document]:
    """The shard documents of a seed by relative path, converted against
    ``schema`` as bootstrap converts a configured one.

    Raises:
        SeedGraphError: The seed does not convert (names collide under
            ``on_name_collision: error``, say): its run fails at start-up.
    """
    conversion = convert_seed(
        SeedGraph(nodes=tuple(nodes), edges=tuple(edges)),
        SeedConversionSettings(
            name=name,
            shards=shards,
            max_items_per_unit=max_items_per_unit,
            on_name_collision=on_name_collision,
        ),
        schema,
    )
    return {document.file_path: document for document in conversion.documents()}


def seed_shard_extraction(
    document: Document,
) -> tuple[list[TextUnit], list[TextUnitExtraction]]:
    """A shard document's text units, keyed under the seed fingerprint, and
    their records, as the chunking and extraction stages make them."""
    units = [
        unit.model_copy(
            update={
                "attributes": {
                    **(unit.attributes or {}),
                    RECORD_KEYS_ATTRIBUTE: {
                        "graph": record_key(SEED_RECORD_FINGERPRINT, unit.text)
                    },
                }
            }
        )
        for unit in seed_text_units(document)
    ]
    return units, [parse_seed_unit(unit) for unit in units]


class Seed(NamedTuple):
    """A seed for the incremental properties (a plain ``(shards, nodes,
    edges)`` tuple is one whose colliding names merge)."""

    shards: int
    nodes: tuple[SeedNode, ...]
    edges: tuple[SeedEdge, ...]
    on_name_collision: SeedNameCollision = "merge"


def _node_id(name: str) -> str:
    # Stable across rounds: an edited node stays in its shard.
    return name.replace(" ", "-")


@st.composite
def random_seeds(
    draw: st.DrawFn,
    names: Sequence[str],
    entity_types: Sequence[str | None],
    relationship_types: Sequence[str],
    collisions: Sequence[SeedNameCollision] = (),
) -> Seed:
    """A random seed over ``names``: 1-3 shards, nodes with a drawn type,
    IRI (two nodes may share one), description and confidence, and edges
    between them.

    With ``collisions``, up to two drawn names are also spelled upper- or
    lower-case by a node of their own (an in-seed name collision, unless the
    nodes share an IRI), and the seed's ``on_name_collision`` is drawn from
    them.
    """
    chosen = draw(st.lists(st.sampled_from(names), min_size=1, max_size=4, unique=True))
    if collisions:
        variants = sorted(
            {v for n in chosen for v in (n.upper(), n.lower())} - {*chosen}
        )
        # A caseless script (Hangul, kana, Han) has no other spelling.
        if variants:
            chosen += draw(st.lists(st.sampled_from(variants), max_size=2, unique=True))
    nodes = tuple(
        SeedNode(
            id=_node_id(name),
            name=name,
            type=draw(st.sampled_from(entity_types)),
            iri=draw(
                st.sampled_from([None, EX + _node_id(name).lower(), EX + "shared"])
            ),
            description=draw(
                st.sampled_from([None, f"{name} seed fact", f"{name} seed fact 2"])
            ),
            confidence=draw(st.sampled_from([None, 0.3, 0.9])),
        )
        for name in chosen
    )
    ids = [node.id for node in nodes]
    edges = tuple(
        SeedEdge(
            source=source,
            target=target,
            type=type_,
            description=f"seed {source} {type_} {target}",
            weight=weight,
        )
        for source, target, type_, weight in draw(
            st.lists(
                st.tuples(
                    st.sampled_from(ids),
                    st.sampled_from(ids),
                    st.sampled_from(relationship_types),
                    st.sampled_from([1.0, 2.5]),
                ),
                max_size=4,
                unique_by=lambda e: e[:3],
            )
        )
        if source != target
    )
    policy = draw(st.sampled_from(collisions)) if collisions else "merge"
    return Seed(draw(st.integers(1, 3)), nodes, edges, policy)


def seeded_corpus(
    documents: Mapping[str, Any],
    seeds: Mapping[str, Seed | None],
    schema: ExtractionSchema,
) -> dict[str, Any]:
    """A run's corpus: the present documents (content not None) and the
    shard documents of every configured seed (not None), converted against
    ``schema``."""
    corpus = {path: content for path, content in documents.items() if content}
    for name, seed in seeds.items():
        if seed is not None:
            shards, nodes, edges, policy = Seed(*seed)
            corpus.update(
                seed_shards(
                    name,
                    nodes,
                    edges,
                    schema,
                    shards=shards,
                    on_name_collision=policy,
                )
            )
    return corpus


def seed_statements(conversion: SeedConversion) -> list[str]:
    """Every statement line of a conversion, in document order."""
    return [
        line
        for document in conversion.documents()
        for unit in seed_text_units(document)
        for line in unit.text.split("\n")
        if line
    ]


def render_unit(statements: Iterable[SeedNodeStatement | SeedEdgeStatement]) -> str:
    """The text of a seed unit holding ``statements``: one line each."""
    return "\n".join(render_statement(statement) for statement in statements)
