# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``indexing.neptune.vertex_ids: custom`` on Gremlin Server.

The graph contracts run in both modes in the conformance suites
(``container-custom-ids``); this module keeps what only a real server shows
for custom ids: incremental runs (interrupted ones included) converge with a
full build, every vertex they leave has its custom vertex id, and a graph of
the other mode is refused until one full build rewrites it. Skipped unless
``LOCAL_STORES=1`` (see ``tests/integration/test_local_stores.py``).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from gremlin_python.process.graph_traversal import __

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeVectorStore
from tests.fixtures.incremental_runs import Corpus, IncrementalRun, expected_state
from tests.fixtures.store_backends import (
    CUSTOM_IDS,
    StoreHarness,
    local_store_config,
    open_harness,
)
from tests.property.test_incremental_interruption_properties import (
    _FIRING_CASES,
    _follow_up,
)
from unified_kg_rag.adapters.aws.neptune import VERTEX_ID_SWITCH_LABEL, vertex_id
from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.models import (
    Community,
    Config,
    Entity,
    Relationship,
    RetrieverRole,
    SearchQuery,
)
from unified_kg_rag.shared.exceptions import StoreConfigurationError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("LOCAL_STORES") != "1",
        reason="set LOCAL_STORES=1 with docker/compose.local.yaml running",
    ),
]

_SUFFIX = "default"
_GRAPH_KEYS = ("graph_entities", "graph_edges", "entity_content", "edge_content")


def _custom_config() -> Config:
    config = local_store_config()
    config.indexing.additional_suffix = f"tid{uuid.uuid4().hex[:8]}"
    config.indexing.neptune.vertex_ids = "custom"
    return config


def _vertex_ids_by_id(indexer: NeptuneIndexer, prefix: str) -> dict[str, str]:
    """``{id property: vertex id}`` of the default suffix's ``prefix`` label."""
    label = indexer._get_label(prefix, _SUFFIX)
    rows = (
        indexer.neptune_client.g.V()
        .hasLabel(label)
        .project("id", "vertex_id")
        .by("id")
        .by(__.id_())
        .toList()
    )
    return {str(row["id"]): str(row["vertex_id"]) for row in rows}


class _GremlinGraph(NeptuneIndexer):
    """The indexer, with the ids the fake graph store lists (``ids``)."""

    def ids(self, kind: str) -> set[str]:
        prefix = (
            self.neptune_config.community_label_prefix
            if kind == "communities"
            else self.neptune_config.entity_label_prefix
        )
        label = self._get_label(prefix, _SUFFIX)
        g = self.neptune_client.g
        traversal = (
            g.V().hasLabel(label).outE().values("id")
            if kind == "relationships"
            else g.V().hasLabel(label).values("id")
        )
        return {str(item) for item in traversal.toList()}


class _GremlinRun(IncrementalRun):
    """``IncrementalRun`` on Gremlin Server in custom mode (fake vector store
    and registry)."""

    def __init__(self) -> None:
        super().__init__(FakeDocStatusStore())
        self.config = _custom_config()
        self.graph = _GremlinGraph(self.config)  # type: ignore[assignment]
        self.vector = FakeVectorStore(opensearch_config=self.config.indexing.opensearch)
        self.manager = self._manager()

    def close(self) -> None:
        try:
            self.graph.clear([_SUFFIX])
        finally:
            self.graph.close()

    def graph_state(self) -> dict[str, Any]:
        state = self.state()
        return {key: state[key] for key in _GRAPH_KEYS}

    def assert_custom_ids(self) -> None:
        graph: Any = self.graph
        neptune = self.config.indexing.neptune
        for prefix in (neptune.entity_label_prefix, neptune.community_label_prefix):
            label = graph._get_label(prefix, _SUFFIX)
            stored = _vertex_ids_by_id(graph, prefix)
            assert stored == {i: vertex_id(label, i) for i in stored}, stored


def _graph_expected(corpus: Corpus) -> dict[str, Any]:
    expected = expected_state(corpus)
    return {key: expected[key] for key in _GRAPH_KEYS}


@pytest.fixture
def run() -> Iterator[_GremlinRun]:
    harness = _GremlinRun()
    try:
        yield harness
    finally:
        harness.close()


@pytest.mark.parametrize(
    "case",
    _FIRING_CASES,
    ids=[f"{c['point']}-k{c['k']}" for c in _FIRING_CASES],
)
def test_an_interrupted_incremental_run_converges_with_a_full_build(
    run: _GremlinRun, case: dict
) -> None:
    # The property suite's interruption cases (each reaches its point), on
    # the server: a follow-up run leaves what a full build of the corpus
    # stores, under custom vertex ids only.
    before, interrupted = case["before"], case["interrupted"]
    corpus = _follow_up(case["follow"], before, interrupted)
    run.run(before)
    assert run.graph_state() == _graph_expected(before)
    assert run.run(interrupted, interrupt=case["point"], k=case["k"])
    run.run(corpus)
    assert run.graph_state() == _graph_expected(corpus)
    assert run.registry_problems(corpus) == []
    run.assert_custom_ids()
    run.run(corpus)
    assert run.extracted == []


def test_incremental_rounds_converge_with_a_full_build(run: _GremlinRun) -> None:
    v, d, c, b = "Vendor", "Depot", "Carrier", "Buyer"
    rounds: list[Corpus] = [
        {"/corpus/a.txt": [[(v, d, 1)]], "/corpus/b.txt": [[(c, b, 2)]]},
        # a changed (two units), c added.
        {
            "/corpus/a.txt": [[(v, d, 3), (d, c, 1)], [(v, b, 1)]],
            "/corpus/b.txt": [[(c, b, 2)]],
            "/corpus/c.txt": [[(b, d, 2)]],
        },
        # b removed, a reverted.
        {"/corpus/a.txt": [[(v, d, 1)]], "/corpus/c.txt": [[(b, d, 2)]]},
        {},
        {"/corpus/b.txt": [[(c, b, 2)]]},
    ]
    for corpus in rounds:
        run.run(corpus)
        assert run.graph_state() == _graph_expected(corpus), corpus
        run.assert_custom_ids()


# --- switching modes -----------------------------------------------------------


def _graph() -> tuple[list[Entity], list[Relationship], list[Community]]:
    entities = [
        Entity(id=i, name=i.title(), type="ORG", rank=1, text_unit_ids=["t1"])
        for i in ("vendor", "buyer", "depot")
    ]
    relationships = [
        Relationship(
            id=f"r-{a}-{z}",
            source_id=a,
            target_id=z,
            source_name=a,
            target_name=z,
            text_unit_ids=["t1"],
        )
        for a, z in (("vendor", "buyer"), ("vendor", "depot"))
    ]
    communities = [
        Community(
            id="c1",
            name="c1",
            level="0",
            parent="-1",
            children=[],
            entity_ids=["vendor", "buyer", "depot"],
            text_unit_ids=["t1"],
        )
    ]
    return entities, relationships, communities


def _full_build(harness: StoreHarness) -> dict[str, Any]:
    entities, relationships, communities = _graph()
    manager = IndexingManager(
        config=harness.config,
        vector_indexer=FakeVectorStore(harness.config.indexing.opensearch),
        graph_indexer=harness.graph_indexer,
    )
    return manager.index_all_data(
        entities=entities,
        relationships=relationships,
        communities=communities,
    )


def _expand(harness: StoreHarness) -> list[str]:
    (hits,) = harness.retrieve(
        RetrieverRole.GRAPH,
        SearchQuery(query="", suffix=_SUFFIX, filters={"id": ["vendor"]}),
    )
    return sorted(hit.source for hit in hits)


def _switches(harness: StoreHarness) -> list[str]:
    g = harness.graph_indexer.neptune_client.g
    labels = {
        harness.graph_indexer._get_label(prefix, _SUFFIX)
        for prefix in (
            harness.config.indexing.neptune.entity_label_prefix,
            harness.config.indexing.neptune.community_label_prefix,
        )
    }
    return sorted(
        str(label)
        for label in g.V().hasLabel(VERTEX_ID_SWITCH_LABEL).values("id").toList()
        if label in labels
    )


@pytest.fixture
def custom(tmp_path: Path) -> Iterator[StoreHarness]:
    yield from open_harness(f"container+{CUSTOM_IDS}", tmp_path)


def _in(mode: str) -> Any:
    return lambda config: setattr(config.indexing.neptune, "vertex_ids", mode)


@pytest.mark.parametrize("built_in", ["property", "custom"])
def test_a_graph_of_the_other_mode_is_refused_until_a_full_build(
    custom: StoreHarness, built_in: str
) -> None:
    other = "custom" if built_in == "property" else "property"
    old = custom.derive(_in(built_in))
    entities, relationships, communities = _graph()
    assert old.graph_indexer.upsert_entities(entities).failed_items == 0
    assert old.graph_indexer.upsert_relationships(relationships).failed_items == 0
    assert old.graph_indexer.upsert_communities(communities).failed_items == 0
    assert _expand(old) == ["buyer", "depot", "vendor"]

    new = custom.derive(_in(other))
    for refused in (
        lambda: new.graph_indexer.upsert_entities(entities[:1]),
        lambda: new.graph_indexer.read_entities(["vendor"]),
        lambda: new.graph_indexer.upsert_communities(communities),
    ):
        with pytest.raises(StoreConfigurationError, match="indexing.reset: true"):
            refused()
    # Queries in the new mode before the build: read by the id property.
    assert _expand(new) == ["buyer", "depot", "vendor"]
    assert new.stored_ids("graph.entities") == {"vendor", "buyer", "depot"}

    # One full build rewrites it: the old vertices are swept, the switch ends.
    results = _full_build(new)
    assert not any(stats.failed_items for stats in results.values()), results
    assert _switches(new) == []
    assert new.stored_ids("graph.entities") == {"vendor", "buyer", "depot"}
    assert new.stored_ids("graph.communities") == {"c1"}
    assert new.stored_ids("graph.relationships") == {
        "r-vendor-buyer",
        "r-vendor-depot",
    }
    assert _expand(new) == ["buyer", "depot", "vendor"]
    entity_label = new.graph_indexer._get_label("Entity", _SUFFIX)
    vids = _vertex_ids_by_id(new.graph_indexer, "Entity")
    custom_ids = {i: vertex_id(entity_label, i) for i in vids}
    assert (vids == custom_ids) is (other == "custom"), vids
    # And now the old mode is refused.
    again = custom.derive(_in(built_in))
    with pytest.raises(StoreConfigurationError):
        again.graph_indexer.read_entities(["vendor"])


def test_an_unswept_switch_is_refused_until_a_full_build_finishes_it(
    custom: StoreHarness,
) -> None:
    old = custom.derive(_in("property"))
    entities, relationships, _ = _graph()
    assert old.graph_indexer.upsert_entities(entities).failed_items == 0
    assert old.graph_indexer.upsert_relationships(relationships).failed_items == 0

    # A full build to custom ids that failed a write: no sweep, both modes
    # stay in the label.
    indexer = custom.graph_indexer
    indexer.begin_rebuild([_SUFFIX])
    assert indexer.index_entities(entities, suffixes=[_SUFFIX]).failed_items == 0
    indexer.end_rebuild([_SUFFIX], sweep=False)
    assert len(_switches(custom)) == 1
    # Both copies of each entity are stored.
    assert (
        len(
            indexer.neptune_client.g.V()
            .hasLabel(indexer._get_label("Entity", _SUFFIX))
            .toList()
        )
        == 6
    )

    for mode in ("custom", "property"):
        later = custom.derive(_in(mode))
        with pytest.raises(StoreConfigurationError, match="did not finish"):
            later.graph_indexer.upsert_entities(entities[:1])

    rerun = custom.derive(_in("custom"))
    results = _full_build(rerun)
    assert not any(stats.failed_items for stats in results.values()), results
    assert _switches(rerun) == []
    assert (
        len(
            rerun.graph_indexer.neptune_client.g.V()
            .hasLabel(rerun.graph_indexer._get_label("Entity", _SUFFIX))
            .toList()
        )
        == 3
    )
    assert _expand(rerun) == ["buyer", "depot", "vendor"]


@pytest.mark.parametrize("target", ["property", "custom"])
def test_queries_in_either_mode_read_a_switch_in_progress_once_per_entity(
    custom: StoreHarness, target: str
) -> None:
    # A full build switching to `target` wrote some of the graph and has not
    # swept yet: the label holds the rewritten vertices next to their old
    # copies. A query in either mode (the query service switched before or
    # after the build started) reads every entity once, the rewritten copy
    # where there is one.
    built_in = "custom" if target == "property" else "property"
    old = custom.derive(_in(built_in))
    entities, relationships, _ = _graph()
    assert old.graph_indexer.upsert_entities(entities).failed_items == 0
    assert old.graph_indexer.upsert_relationships(relationships).failed_items == 0

    switching = custom.derive(_in(target))
    indexer = switching.graph_indexer
    indexer.begin_rebuild([_SUFFIX])
    rewritten = [
        entity.model_copy(update={"description": "rewritten"})
        for entity in entities
        if entity.id in ("vendor", "buyer")
    ]
    assert indexer.index_entities(rewritten, suffixes=[_SUFFIX]).failed_items == 0
    assert indexer.index_relationships(relationships[:1]).failed_items == 0
    assert _switches(switching) == [indexer._get_label("Entity", _SUFFIX)]
    try:
        for mode in ("property", "custom"):
            query = custom.derive(_in(mode))
            (hits,) = query.retrieve(
                RetrieverRole.GRAPH,
                SearchQuery(query="", suffix=_SUFFIX, filters={"id": ["vendor"]}),
            )
            by_id = {hit.source: hit for hit in hits}
            assert sorted(hit.source for hit in hits) == ["buyer", "depot", "vendor"]
            assert "rewritten" in by_id["vendor"].content
            assert "rewritten" in by_id["buyer"].content
            # Not rewritten yet: its old copy.
            assert "rewritten" not in by_id["depot"].content
    finally:
        indexer.end_rebuild([_SUFFIX])
    assert _switches(switching) == []


@pytest.mark.parametrize("built_in", ["custom", "property"])
def test_a_long_lived_retriever_follows_a_switch_like_a_fresh_one(
    custom: StoreHarness, built_in: str
) -> None:
    # The query service keeps one retriever (in the label's old mode) while
    # a full build switches the label: before, during and after the build
    # it answers what a retriever opened at that moment answers. It cached
    # the label as settled, so before it followed neither the switch (seeds
    # only) nor its end (nothing until a restart, custom -> property).
    target = "property" if built_in == "custom" else "custom"
    old = custom.derive(_in(built_in))
    entities, relationships, _ = _graph()
    assert old.graph_indexer.upsert_entities(entities).failed_items == 0
    assert old.graph_indexer.upsert_relationships(relationships).failed_items == 0
    query = SearchQuery(query="", suffix=_SUFFIX, filters={"id": ["vendor"]})

    def hits(results: list[Any]) -> list[tuple[str, float | None, str]]:
        return sorted((hit.source, hit.score, hit.content) for hit in results)

    def fresh() -> list[tuple[str, float | None, str]]:
        (results,) = old.retrieve(RetrieverRole.GRAPH, query)
        return hits(results)

    async def follow() -> dict[str, tuple[Any, Any]]:
        indexer = custom.derive(_in(target)).graph_indexer
        rewritten = [
            e.model_copy(update={"description": "rewritten"}) for e in entities
        ]
        assert old.graph_retriever is not None
        retriever = old.graph_retriever()
        phases: dict[str, tuple[Any, Any]] = {}
        try:
            phases["before"] = (
                hits(await retriever.aretrieve(query)),
                await asyncio.to_thread(fresh),
            )
            await asyncio.to_thread(indexer.begin_rebuild, [_SUFFIX])
            await asyncio.to_thread(
                indexer.index_entities, rewritten, suffixes=[_SUFFIX]
            )
            await asyncio.to_thread(indexer.index_relationships, relationships)
            phases["during"] = (
                hits(await retriever.aretrieve(query)),
                await asyncio.to_thread(fresh),
            )
            await asyncio.to_thread(indexer.end_rebuild, [_SUFFIX], True)
            phases["after"] = (
                hits(await retriever.aretrieve(query)),
                await asyncio.to_thread(fresh),
            )
        finally:
            await retriever.aclose()
        return phases

    phases = asyncio.run(follow())
    assert _switches(old) == []
    for phase, (same, opened) in phases.items():
        assert same == opened, phase
        assert [source for source, _, _ in same] == ["buyer", "depot", "vendor"]
    assert all("rewritten" in content for _, _, content in phases["after"][0])


def test_a_switch_swept_between_two_queries_is_followed(
    custom: StoreHarness,
) -> None:
    # A full build switches a custom label to property ids and sweeps
    # between two queries of one retriever: no switch vertex is left to see,
    # so the custom-id lookup finds no seed, and the label is sampled again
    # and the expansion re-run by the id property.
    old = custom.derive(_in("custom"))
    entities, relationships, _ = _graph()
    assert old.graph_indexer.upsert_entities(entities).failed_items == 0
    assert old.graph_indexer.upsert_relationships(relationships).failed_items == 0
    query = SearchQuery(query="", suffix=_SUFFIX, filters={"id": ["vendor"]})

    async def follow() -> tuple[list[str], list[str]]:
        indexer = custom.derive(_in("property")).graph_indexer
        assert old.graph_retriever is not None
        retriever = old.graph_retriever()
        try:
            before = sorted(h.source for h in await retriever.aretrieve(query))
            await asyncio.to_thread(indexer.begin_rebuild, [_SUFFIX])
            await asyncio.to_thread(
                indexer.index_entities, entities, suffixes=[_SUFFIX]
            )
            await asyncio.to_thread(indexer.index_relationships, relationships)
            await asyncio.to_thread(indexer.end_rebuild, [_SUFFIX], True)
            after = sorted(h.source for h in await retriever.aretrieve(query))
        finally:
            await retriever.aclose()
        return before, after

    before, after = asyncio.run(follow())
    assert _switches(old) == []
    assert before == after == ["buyer", "depot", "vendor"]
