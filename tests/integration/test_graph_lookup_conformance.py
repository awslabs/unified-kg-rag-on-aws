# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MCP graph tools' store reads (``application/retrieval/graph_lookup``)
on every backend with retrievers: the embedded SQLite stores, Gremlin Server
and OpenSearch (``LOCAL_STORES=1``) and Neptune with OpenSearch Service
(``-m aws``); see ``tests/fixtures/store_backends.py``.

An entity by id, IRI and exact name (Latin, Hangul and Kanji), the name
search fallback, the one- and two-hop neighbourhood with the relationships
among it (from the relationships index, and from the graph without it) and
the text units' file names. No model is called: the language model fails if
asked.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.tokens import WordTokenCounter
from tests.fixtures.store_backends import (
    RETRIEVERS,
    StoreHarness,
    backends,
    open_harness,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.retrieval import graph_lookup, methodology_check
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    Entity,
    IndexArtifact,
    Relationship,
    TextUnit,
)

pytestmark = pytest.mark.integration

_T = TypeVar("_T")
SUFFIX = "default"


@pytest.fixture(params=backends(RETRIEVERS, graph=True))
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StoreHarness]:
    yield from open_harness(request.param, tmp_path)


def _no_model(system: str, human: str) -> str:
    raise AssertionError("the graph lookups call no model")


def _units() -> list[TextUnit]:
    return [
        TextUnit(
            id="t1",
            text="Vendor supplies widgets to Buyer.",
            attributes={
                "file_name": "contract.txt",
                "file_path": "/srv/in/contract.txt",
            },
        ),
        TextUnit(
            id="t2",
            text="Depot stores widgets for Vendor.",
            attributes={"file_name": "logistics.txt"},
        ),
        TextUnit(
            id="t3",
            text="한빛물류 moves widgets for Depot; 東京倉庫 keeps spares.",
            attributes={"file_path": "/srv/in/asia/korea.txt"},
        ),
    ]


def _entity(id_: str, name: str, units: list[str], **fields: Any) -> Entity:
    return Entity(
        id=id_,
        name=name,
        type="ORGANIZATION",
        description=f"{name} in the supply chain",
        text_unit_ids=units,
        rank=len(units),
        **fields,
    )


def _entities() -> list[Entity]:
    return [
        _entity("e-vendor", "Vendor", ["t1", "t2"], iri="urn:example:vendor"),
        _entity("e-buyer", "Buyer", ["t1"]),
        _entity("e-depot", "Depot", ["t2", "t3"]),
        _entity("e-hanbit", "한빛물류", ["t3"]),
        _entity("e-tokyo", "東京倉庫", ["t3"]),
    ]


def _relationship(id_: str, source: Entity, target: Entity, unit: str) -> Relationship:
    return Relationship(
        id=id_,
        source_id=source.id,
        target_id=target.id,
        source_name=source.name,
        target_name=target.name,
        type="WORKS_WITH",
        description=f"{source.name} works with {target.name}",
        weight=1.0,
        text_unit_ids=[unit],
    )


def _relationships() -> list[Relationship]:
    vendor, buyer, depot, hanbit, tokyo = _entities()
    return [
        _relationship("r-vb", vendor, buyer, "t1"),
        _relationship("r-dv", depot, vendor, "t2"),
        _relationship("r-hd", hanbit, depot, "t3"),
        _relationship("r-th", tokyo, hanbit, "t3"),
    ]


def _ingest(h: StoreHarness) -> None:
    h.initialize()
    entities, relationships = _entities(), _relationships()
    for stats in (
        h.vector_indexer.index_text_units(_units()),
        h.vector_indexer.index_entities(entities),
        h.vector_indexer.index_relationships(relationships),
        h.graph_indexer.index_entities(entities),
        h.graph_indexer.index_relationships(relationships),
    ):
        assert stats.failed_items == 0, stats.errors


def _run(h: StoreHarness, use: Callable[[GraphRAGChain], Awaitable[_T]]) -> _T:
    config = h.config.model_copy(deep=True)
    config.providers.rerank.kind = "none"
    providers = Providers(
        config,
        llm_factory=ScriptedLLMFactory(_no_model),
        embedding_factory=h.embedding_factory,
        token_counter_factory=lambda *_, **__: WordTokenCounter(),
    )

    async def run() -> _T:
        chain = GraphRAGChain(config, providers=providers)
        try:
            return await use(chain)
        finally:
            await chain.aclose()

    return asyncio.run(run())


def test_entities_by_id_iri_and_exact_name_in_any_script(harness: StoreHarness) -> None:
    _ingest(harness)

    async def lookups(chain: GraphRAGChain) -> list[tuple[str, list[str]]]:
        found = []
        for key, value in (
            ("entity_id", "e-depot"),
            ("iri", "urn:example:vendor"),
            ("name", "Buyer"),
            ("name", "한빛물류"),
            ("name", "東京倉庫"),
            ("name", "Nobody At All"),
        ):
            match, hits = await graph_lookup.find_entities(
                chain, SUFFIX, limit=3, **{key: value}
            )
            found.append((match, [str(h.metadata.get("name")) for h in hits]))
        return found

    assert _run(harness, lookups) == [
        ("id", ["Depot"]),
        ("iri", ["Vendor"]),
        ("name", ["Buyer"]),
        ("name", ["한빛물류"]),
        ("name", ["東京倉庫"]),
        ("none", []),
    ]


def test_a_name_without_an_exact_match_falls_back_to_a_search(
    harness: StoreHarness,
) -> None:
    _ingest(harness)

    async def lookup(chain: GraphRAGChain) -> tuple[str, list[str]]:
        match, hits = await graph_lookup.find_entities(
            chain, SUFFIX, name="the Depot warehouse", limit=3
        )
        return match, [str(h.metadata.get("name")) for h in hits]

    match, names = _run(harness, lookup)
    assert match == "search" and "Depot" in names


def test_the_neighbourhood_and_its_relationships(harness: StoreHarness) -> None:
    _ingest(harness)

    async def walks(chain: GraphRAGChain) -> tuple[Any, Any]:
        one = await graph_lookup.neighbourhood(
            chain, SUFFIX, "e-depot", max_hops=1, limit=10
        )
        two = await graph_lookup.neighbourhood(
            chain, SUFFIX, "e-depot", max_hops=2, limit=10
        )
        return one, two

    one, two = _run(harness, walks)
    assert set(one.hops) == {"e-vendor", "e-hanbit"}
    assert set(one.hops.values()) == {1}
    # The stored relationships among the start and its neighbours, with
    # their text units.
    assert {str(r.metadata["id"]) for r in one.relationships} == {"r-dv", "r-hd"}
    assert all(r.metadata.get("text_unit_ids") for r in one.relationships)
    assert two.hops == {"e-vendor": 1, "e-hanbit": 1, "e-buyer": 2, "e-tokyo": 2}
    assert {str(r.metadata["id"]) for r in two.relationships} == {
        "r-vb",
        "r-dv",
        "r-hd",
        "r-th",
    }


def test_without_relationship_vectors_the_edges_come_from_the_graph(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ingest(harness)
    held = frozenset(IndexArtifact) - {IndexArtifact.RELATIONSHIP_VECTORS}

    async def artifacts(*_args: Any, **_kwargs: Any) -> frozenset[IndexArtifact]:
        return held

    monkeypatch.setattr(methodology_check, "namespace_artifacts", artifacts)

    async def walk(chain: GraphRAGChain) -> Any:
        return await graph_lookup.neighbourhood(
            chain, SUFFIX, "e-depot", max_hops=1, limit=10
        )

    found = _run(harness, walk)
    assert found.relationships == []
    assert {(e.source_id, e.target_id) for e in found.edges} == {
        ("e-depot", "e-vendor"),
        ("e-hanbit", "e-depot"),
    }


def test_text_unit_file_names_never_carry_a_path(harness: StoreHarness) -> None:
    _ingest(harness)

    async def names(chain: GraphRAGChain) -> dict[str, str]:
        return await graph_lookup.text_unit_file_names(
            chain, SUFFIX, ["t1", "t2", "t3", "t-missing"]
        )

    assert _run(harness, names) == {
        "t1": "contract.txt",
        "t2": "logistics.txt",
        "t3": "korea.txt",
    }
