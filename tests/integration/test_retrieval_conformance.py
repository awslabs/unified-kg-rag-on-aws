# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One retrieval contract for every backend with retrievers.

The OpenSearch retriever over the embedded SQLite engine, the OpenSearch
container (``LOCAL_STORES=1``) and OpenSearch Service (``-m aws``), and graph
expansion over the SQLite graph, Gremlin Server and Neptune; see
``tests/fixtures/store_backends.py``. The fake backend has no retrievers.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.store_backends import (
    RETRIEVERS,
    StoreHarness,
    backends,
    open_harness,
)
from unified_kg_rag.domain.models import (
    CommunityReport,
    Entity,
    Relationship,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchType,
    TextUnit,
)
from unified_kg_rag.domain.models.config import LanguageCode
from unified_kg_rag.shared import ConfigurationError, IndexNotFoundError

pytestmark = pytest.mark.integration

_SUFFIX = "default"


@pytest.fixture(params=backends(RETRIEVERS))
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StoreHarness]:
    yield from open_harness(request.param, tmp_path)


def _text_units() -> list[TextUnit]:
    return [
        TextUnit(
            id="t1",
            text="Vendor supplies widgets to Buyer under a purchase order.",
            attributes={"filters": {"category": "contract"}},
        ),
        TextUnit(
            id="t2",
            text="Vendor ships widgets from the Depot every week.",
            attributes={"filters": {"category": "logistics"}},
        ),
        TextUnit(id="t3", text="Courier delivers gears to the Depot at night."),
    ]


def _entities() -> list[Entity]:
    return [
        Entity(
            id="e-vendor",
            name="Vendor",
            type="ORG",
            description="Vendor supplies widgets",
            rank=3,
        ),
        Entity(
            id="e-buyer",
            name="Buyer",
            type="ORG",
            description="Buyer orders widgets",
            rank=2,
        ),
        Entity(
            id="e-depot",
            name="Depot",
            type="LOCATION",
            description="Depot stores widgets",
            rank=1,
        ),
    ]


def _relationships() -> list[Relationship]:
    return [
        Relationship(
            id="r-supplies",
            source_id="e-vendor",
            target_id="e-buyer",
            source_name="Vendor",
            target_name="Buyer",
            type="SUPPLIES",
            description="Vendor supplies widgets to Buyer",
        ),
        Relationship(
            id="r-ships",
            source_id="e-vendor",
            target_id="e-depot",
            source_name="Vendor",
            target_name="Depot",
            description="Vendor ships widgets from Depot",
        ),
    ]


def _reports() -> list[CommunityReport]:
    return [
        CommunityReport(
            id="cr1",
            community_id="c1",
            name="Widget supply",
            summary="Vendor supplies widgets to Buyer through the Depot.",
            full_content="Widget supply\n\nVendor, Buyer and Depot move widgets.",
        )
    ]


def _ingest(h: StoreHarness) -> None:
    h.initialize()
    indexer = h.vector_indexer
    for stats in (
        indexer.index_text_units(_text_units()),
        indexer.index_entities(_entities()),
        indexer.index_relationships(_relationships()),
        indexer.index_community_reports(_reports()),
    ):
        assert stats.failed_items == 0, stats.errors


def _search(h: StoreHarness, *queries: SearchQuery) -> list[list[RetrievalResult]]:
    return h.retrieve(RetrieverRole.DOCUMENT, *queries)


def _query(text: str, prefix: str | None = None, **fields: Any) -> SearchQuery:
    return SearchQuery(query=text, suffix=_SUFFIX, index_prefixes=prefix, **fields)


def _prefixes(h: StoreHarness) -> Any:
    return h.config.indexing.opensearch


def test_fetch_by_id_returns_every_id_above_max_query_size(
    harness: StoreHarness,
) -> None:
    h = harness.derive(lambda c: setattr(c.indexing.opensearch, "max_query_size", 3))
    entities = [
        Entity(id=f"e{i}", name=f"Entity {i}", type="ORG", description="widgets")
        for i in range(8)
    ]
    h.initialize()
    assert h.vector_indexer.index_entities(entities).failed_items == 0
    ids = [f"e{i}" for i in range(1, 7)]
    prefix = _prefixes(h).entities_index_prefix
    fetched, capped = _search(
        h,
        _query("", prefix, top_k=len(ids), retrieval_multiplier=1, filters={"id": ids}),
        _query("widgets", prefix, top_k=6, retrieval_multiplier=1),
    )
    assert sorted(r.source for r in fetched) == ids
    # A search not naming ids stays capped at max_query_size.
    assert len(capped) == 3


def test_term_terms_range_and_attribute_filters(harness: StoreHarness) -> None:
    _ingest(harness)
    o = _prefixes(harness)
    results = _search(
        harness,
        _query("widgets", o.entities_index_prefix, filters={"type": "ORG"}),
        _query("", o.entities_index_prefix, filters={"id": ["e-buyer", "e-depot"]}),
        _query("widgets", o.entities_index_prefix, filters={"rank": {"gte": 2}}),
        _query("", o.entities_index_prefix, filters={"name.keyword": "Depot"}),
        _query(
            "widgets", o.text_units_index_prefix, filters={"attr_category": "logistics"}
        ),
        _query(
            "widgets",
            o.text_units_index_prefix,
            filters={"attr_category": ["contract", "logistics"]},
        ),
        _query("widgets", o.entities_index_prefix, filters={"type": "VEHICLE"}),
    )
    by_query = [sorted(r.source for r in hits) for hits in results]
    assert by_query == [
        ["e-buyer", "e-vendor"],
        ["e-buyer", "e-depot"],
        ["e-buyer", "e-vendor"],
        ["e-depot"],
        ["t2"],
        ["t1", "t2"],
        [],
    ]
    # A filter-only query matches without scoring.
    assert all(r.score == 0.0 for r in results[1])


def test_untranslated_korean_corpus_has_lexical_recall(harness: StoreHarness) -> None:
    # A Korean corpus with a Korean target is never translated, so only the
    # ``text`` field carries it, with the source-language analyzer (nori, or
    # the embedded engine's bigram stand-in): "홍길동" must match inside
    # "홍길동으로부터" and "보증 기간" inside "보증기간은".
    def korean(config: Any) -> None:
        config.processing.translation.source_language = LanguageCode.KO
        config.processing.translation.target_language = LanguageCode.KO

    h = harness.derive(korean)
    units = [
        TextUnit(id="k1", text="가나다상사는 홍길동으로부터 공급계약서를 받았다."),
        TextUnit(id="k2", text="보증기간은 납품일로부터 2년이다."),
        TextUnit(id="k3", text="김철수는 가나다상사의 직원이다."),
    ]
    assert h.vector_indexer.index_text_units(units).failed_items == 0
    prefix = _prefixes(h).text_units_index_prefix
    person, warranty, contract, employee, other = _search(
        h,
        *(
            _query(text, prefix, search_type=SearchType.LEXICAL)
            for text in ("홍길동", "보증 기간", "공급 계약", "김철수", "김철민")
        ),
    )
    assert "k1" in {r.source for r in person}
    assert "k2" in {r.source for r in warranty}
    assert "k1" in {r.source for r in contract}
    assert employee[0].source == "k3"
    if harness.kind == "sqlite":
        # The bigram analyzer shares "김철" between 김철민 and 김철수: a weaker
        # match, never a full one.
        assert {r.source for r in other} <= {"k3"}
        assert all((r.score or 0) < (employee[0].score or 0) for r in other)
    else:
        # nori keeps 김철민 one token: no fuzzy edit on Hangul, no hit.
        assert other == []


def test_additional_target_language_is_searchable(harness: StoreHarness) -> None:
    # Each translation is indexed with its own analyzer and searched lexically.
    def languages(config: Any) -> None:
        config.processing.translation.target_language = LanguageCode.EN
        config.processing.translation.additional_target_languages = [LanguageCode.KO]

    h = harness.derive(languages)
    unit = TextUnit(
        id="a1",
        text="The warranty period is two years.",
        translated_texts={
            "en": "The warranty period is two years.",
            "ko": "보증기간은 2년이다.",
        },
    )
    assert h.vector_indexer.index_text_units([unit]).failed_items == 0
    (hits,) = _search(
        h,
        _query(
            "보증 기간",
            _prefixes(h).text_units_index_prefix,
            search_type=SearchType.LEXICAL,
        ),
    )
    assert [r.source for r in hits] == ["a1"]


def test_hybrid_scores_are_normalized_and_follow_the_weights(
    harness: StoreHarness,
) -> None:
    _ingest(harness)
    prefix = _prefixes(harness).text_units_index_prefix
    text = "Which widgets does Vendor ship?"

    def weighted(lexical: float, vector: float) -> StoreHarness:
        def change(config: Any) -> None:
            config.search.hybrid.lexical_weight = lexical
            config.search.hybrid.vector_weight = vector
            config.indexing.opensearch.hybrid_search_pipeline_name = (
                f"{harness.tag}-{int(lexical * 10)}-pipeline"
            )

        h = harness.derive(change)
        h.initialize()
        return h

    (hybrid,) = _search(harness, _query(text, prefix, search_type=SearchType.HYBRID))
    assert hybrid
    assert all(0.0 <= (r.score or 0.0) <= 1.0 for r in hybrid)
    assert max(r.score or 0.0 for r in hybrid) > 0.5

    lexical, vector = _search(
        harness,
        _query(text, prefix, search_type=SearchType.LEXICAL),
        _query(text, prefix, search_type=SearchType.VECTOR),
    )
    (lexical_only,) = _search(
        weighted(1.0, 0.0), _query(text, prefix, search_type=SearchType.HYBRID)
    )
    (vector_only,) = _search(
        weighted(0.0, 1.0), _query(text, prefix, search_type=SearchType.HYBRID)
    )
    # One sub-query weighted out: the fused order is the other's (min-max
    # keeps the order) and its best hit scores 1.
    assert [r.source for r in lexical_only][: len(lexical)] == [
        r.source for r in lexical
    ]
    assert [r.source for r in vector_only][: len(vector)] == [r.source for r in vector]
    assert lexical_only[0].score == pytest.approx(1.0)
    assert vector_only[0].score == pytest.approx(1.0)


def test_hits_carry_their_section_type_and_relationship_endpoints(
    harness: StoreHarness,
) -> None:
    _ingest(harness)
    (hits,) = _search(
        harness, _query("widgets Vendor", search_type=SearchType.LEXICAL, top_k=20)
    )
    by_source = {r.source: r for r in hits}
    assert {
        source: by_source[source].retriever_type
        for source in ("t1", "e-vendor", "r-supplies", "cr1")
    } == {
        "t1": "text",
        "e-vendor": "entity",
        "r-supplies": "relationship",
        "cr1": "community",
    }
    # The relationship index stores no edge type: a plain arrow.
    assert "Relationship: Vendor -> Buyer" in by_source["r-supplies"].content
    assert "Relationship: Vendor -> Depot" in by_source["r-ships"].content
    assert all(not any(k.endswith("_embedding") for k in r.metadata) for r in hits)


def test_query_with_another_embedding_model_is_refused(harness: StoreHarness) -> None:
    _ingest(harness)
    prefix = _prefixes(harness).text_units_index_prefix
    query = _query("Vendor widgets", prefix)
    (hits,) = _search(harness, query)
    assert hits
    other_model = harness.derive(
        lambda c: setattr(c.indexing.opensearch, "embedding_model_id", "another-model")
    )
    with pytest.raises(ConfigurationError, match="'another-model'"):
        _search(other_model, query)
    other_dimension = harness.derive(embedding_factory=HashingEmbeddingFactory(32))
    with pytest.raises(ConfigurationError, match=r"dimension 64.*dimension 32"):
        _search(other_dimension, query)

    # An index that recorded no model: the store's own refusal of the vector.
    def skip_check(retriever: Any) -> None:
        retriever._embedding_checked_aliases.add(retriever._get_name(prefix, _SUFFIX))

    with pytest.raises(ConfigurationError, match="invalid dimension: 32"):
        other_dimension.retrieve(RetrieverRole.DOCUMENT, query, prepare=skip_check)


def test_query_of_a_never_ingested_suffix_raises_index_not_found(
    harness: StoreHarness,
) -> None:
    _ingest(harness)
    never = harness.suffix("never")
    with pytest.raises(
        IndexNotFoundError, match=f"No indices found for suffix '{never}'"
    ):
        _search(
            harness,
            SearchQuery(
                query="widgets",
                suffix=never,
                index_prefixes=[_prefixes(harness).entities_index_prefix],
            ),
        )


def test_search_results_are_deterministic(harness: StoreHarness) -> None:
    _ingest(harness)
    queries = [
        _query("Which widgets does Vendor supply?", search_type=search_type)
        for search_type in SearchType
    ]
    first = _search(harness, *queries)
    second = _search(harness, *queries)

    def ranked(results: list[list[RetrievalResult]]) -> list[list[tuple[str, float]]]:
        return [
            [(r.source, round(r.score or 0.0, 6)) for r in hits] for hits in results
        ]

    assert ranked(first) == ranked(second)
    for search_type, hits in zip(SearchType, first, strict=True):
        assert {"t1", "e-vendor"} <= {r.source for r in hits}, search_type


# --- graph expansion ---------------------------------------------------------


def _graph_retrieve(
    h: StoreHarness, *queries: SearchQuery
) -> list[list[RetrievalResult]]:
    return h.retrieve(RetrieverRole.GRAPH, *queries)


def test_graph_expansion_returns_seeds_first_then_distance_bands(
    harness: StoreHarness,
) -> None:
    # Each seed has its own neighbours; one traversal-wide limit used to let
    # the first seeds' neighbourhoods consume it, and the seeds themselves
    # were never emitted.
    h = harness.derive(
        lambda c: setattr(c.indexing.neptune, "traversal_fetch_multiplier", 1)
    )
    seeds = [f"seed{i}" for i in range(4)]
    near = {seed: [f"{seed}-n{j}" for j in range(4)] for seed in seeds}
    far = {seed: f"{seed}-far" for seed in seeds}
    ids = seeds + [n for group in near.values() for n in group] + list(far.values())
    entities = [
        Entity(id=i, name=i, type="ORG", description=f"{i} description", rank=1)
        for i in ids
    ]
    edges = [(seed, n) for seed, group in near.items() for n in group] + [
        (group[0], far[seed]) for seed, group in near.items()
    ]
    relationships = [
        Relationship(
            id=f"r-{a}-{b}",
            source_id=a,
            target_id=b,
            source_name=a,
            target_name=b,
            description=f"{a} works with {b}",
        )
        for a, b in edges
    ]
    assert h.graph_indexer.index_entities(entities).failed_items == 0
    assert h.graph_indexer.index_relationships(relationships).failed_items == 0

    budgeted, wide = _graph_retrieve(
        h,
        SearchQuery(query="", suffix=_SUFFIX, top_k=8, filters={"id": seeds}),
        SearchQuery(query="", suffix=_SUFFIX, top_k=100, filters={"id": seeds[:1]}),
    )
    scores = {r.source: r.score or 0.0 for r in budgeted}
    # Every seed comes back at proximity 1.0 (equal ranks -> importance 1.0).
    assert all(scores.get(seed) == 1.0 for seed in seeds), scores
    # 8 new entities split over 4 seeds: 2 neighbours each, none starved.
    for seed, group in near.items():
        found = [n for n in group if n in scores]
        assert len(found) == 2, (seed, sorted(scores))
        assert all(0 < scores[n] < 1.0 for n in found)

    # From one seed: the seed, then its four neighbours, then the one entity
    # two hops out, each distance in its own score band.
    ranked = [r.source for r in wide]
    assert ranked[0] == "seed0"
    assert set(ranked[1:5]) == set(near["seed0"])
    assert ranked[5] == far["seed0"]
    bands = {r.source: r.score or 0.0 for r in wide}
    assert min(bands[n] for n in near["seed0"]) > bands[far["seed0"]] > 0
    assert len(wide) == 6


def test_graph_expansion_filters_and_is_deterministic(harness: StoreHarness) -> None:
    entities = [
        Entity(id="a", name="A", type="ORG", rank=2, attributes={"role": "vendor"}),
        Entity(id="b", name="B", type="ORG", rank=1, attributes={"role": "buyer"}),
        Entity(id="c", name="C", type="LOCATION", rank=1),
    ]
    relationships = [
        Relationship(id="ab", source_id="a", target_id="b", description="a-b"),
        Relationship(id="ac", source_id="a", target_id="c", description="a-c"),
    ]
    assert harness.graph_indexer.index_entities(entities).failed_items == 0
    assert harness.graph_indexer.index_relationships(relationships).failed_items == 0
    queries = [
        SearchQuery(query="", suffix=_SUFFIX, filters={"id": ["a"]}),
        SearchQuery(query="", suffix=_SUFFIX, filters={"id": ["a"], "type": "ORG"}),
        # attr_* filters apply where the vertex has the property.
        SearchQuery(
            query="", suffix=_SUFFIX, filters={"id": ["a"], "attr_role": "vendor"}
        ),
    ]
    first = _graph_retrieve(harness, *queries)
    assert [sorted(r.source for r in hits) for hits in first] == [
        ["a", "b", "c"],
        ["a", "b"],
        ["a", "c"],
    ]
    second = _graph_retrieve(harness, *queries)
    assert [[(r.source, r.score) for r in hits] for hits in first] == [
        [(r.source, r.score) for r in hits] for hits in second
    ]
