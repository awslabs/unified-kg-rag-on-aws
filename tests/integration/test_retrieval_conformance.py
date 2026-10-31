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
    Claim,
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


# Graph expansion also runs with custom Neptune vertex ids.
@pytest.fixture(params=backends(RETRIEVERS, graph=True))
def graph_harness(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[StoreHarness]:
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


def test_text_less_search_ranks_by_rank_then_weight(harness: StoreHarness) -> None:
    # A filter browse (the incident edges of an entity) has no relevance: a
    # capped page must keep the top-ranked edges, not the first indexed ones.
    h = harness.derive(lambda c: setattr(c.indexing.opensearch, "max_query_size", 3))
    edges = [
        Relationship(
            id=f"r{i}",
            source_id="e-hub",
            target_id=f"e{i}",
            source_name="Hub",
            target_name=f"Node {i}",
            description=f"Hub relates to node {i}",
            rank=min(i, 6),
            weight=float(i),
        )
        for i in range(8)
    ]
    h.initialize()
    assert h.vector_indexer.index_relationships(edges).failed_items == 0
    (hits,) = _search(
        h,
        _query(
            "",
            _prefixes(h).relationships_index_prefix,
            search_type=SearchType.LEXICAL,
            top_k=3,
            retrieval_multiplier=1,
            filters={"source_id": ["e-hub"]},
        ),
    )
    # r6 and r7 tie on rank; weight orders them.
    assert [r.source for r in hits] == ["r7", "r6", "r5"]


@pytest.mark.parametrize("text", ["", "node"], ids=["browse", "text"])
@pytest.mark.parametrize("search_type", list(SearchType), ids=lambda t: t.value)
@pytest.mark.parametrize("attribute", [False, True], ids=["ids", "ids+attribute"])
def test_exclude_leaves_its_matches_out_before_the_page_is_cut(
    harness: StoreHarness, text: str, search_type: SearchType, attribute: bool
) -> None:
    # The filtered walk leaves the vertices it reached out of its edge
    # pages (SearchQuery.exclude), so edges back to them cannot fill a page.
    h = harness.derive(lambda c: setattr(c.indexing.opensearch, "max_query_size", 3))
    edges = [
        Relationship(
            id=f"r{i}",
            source_id="e-hub",
            target_id=f"e{i}",
            source_name="Hub",
            target_name=f"Node {i}",
            description=f"Hub relates to node {i}",
            rank=i,
            weight=float(i),
            attributes={"filters": {"grp": "a" if i % 2 else "b"}},
        )
        for i in range(8)
    ]
    h.initialize()
    assert h.vector_indexer.index_relationships(edges).failed_items == 0
    filters: dict[str, Any] = {"source_id": ["e-hub"]}
    if attribute:
        filters["attr_grp"] = "a"
    (hits,) = _search(
        h,
        _query(
            text,
            _prefixes(h).relationships_index_prefix,
            search_type=search_type,
            top_k=3,
            retrieval_multiplier=1,
            filters=filters,
            exclude={"target_id": ["e7", "e6", "e5"]},
        ),
    )
    kept = ["r3", "r1"] if attribute else ["r4", "r3", "r2", "r1", "r0"]
    got = [r.source for r in hits]
    if text or search_type != SearchType.LEXICAL:
        # By relevance, or a browse in index order: kept edges, as many as fit.
        assert set(got) <= set(kept) and len(got) == min(3, len(kept))
    else:  # the walk's page: by rank, the best of the kept edges
        assert got == kept[:3]


@pytest.mark.parametrize("search_type", list(SearchType), ids=lambda t: t.value)
def test_index_sweep_keeps_each_index_best_hits(
    harness: StoreHarness, search_type: SearchType
) -> None:
    # Raw scores of two indices do not compare (a vector search sums the two
    # entity embedding fields' scores), so a sweep takes each index in turn.
    harness.initialize()
    units = [
        TextUnit(
            id=f"t{i}",
            text="widget price " * (5 - i) + "about the Buyer contract " * (i + 1),
        )
        for i in range(5)
    ]
    entities = [
        Entity(id="e-courier", name="Courier", type="ORG", description="price once"),
        Entity(id="e-depot", name="Depot", type="LOCATION", description="price list"),
    ]
    assert harness.vector_indexer.index_text_units(units).failed_items == 0
    assert harness.vector_indexer.index_entities(entities).failed_items == 0
    o = _prefixes(harness)
    (hits,) = _search(
        harness,
        _query(
            "widget price",
            [o.text_units_index_prefix, o.entities_index_prefix],
            search_type=search_type,
            top_k=4,
            retrieval_multiplier=1,
        ),
    )
    assert [r.retriever_type for r in hits] == ["text", "entity"] * 2
    assert [r.source for r in hits][::2] == ["t0", "t1"]


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


def test_an_empty_scope_list_matches_nothing(harness: StoreHarness) -> None:
    # A strategy's own scope may leave nothing (``_ask_incident`` intersects
    # a caller source_id with the frontier): the retriever takes the empty
    # list (a caller's is refused before any search) and matches no hit,
    # with query text too (OpenSearch refused that beside a kNN query).
    _ingest(harness)
    o = _prefixes(harness)
    empty = {"source_id": []}
    results = _search(
        harness,
        _query("", o.relationships_index_prefix, filters=empty),
        *(
            _query("widgets", o.relationships_index_prefix, filters=empty).model_copy(
                update={"search_type": search_type}
            )
            for search_type in SearchType
        ),
    )
    assert results == [[]] * (1 + len(SearchType))


@pytest.mark.parametrize("search_type", list(SearchType), ids=lambda t: t.value)
def test_attribute_filter_skips_an_index_without_the_attribute(
    harness: StoreHarness, search_type: SearchType
) -> None:
    # Community reports and these entities carry no document attributes, so
    # OpenSearch never maps attr_category on them; 2.13 rejected a kNN filter
    # on it (400 "Rewrite first"), failing every vector and hybrid search.
    _ingest(harness)
    o = _prefixes(harness)
    every_index = [
        o.text_units_index_prefix,
        o.entities_index_prefix,
        o.community_reports_index_prefix,
    ]
    results = _search(
        harness,
        *(
            _query("widgets", prefixes, search_type=search_type, filters=filters)
            for prefixes in (o.community_reports_index_prefix, every_index)
            for filters in (
                {"attr_category": "logistics"},
                {"attributes.filters.category": "logistics"},
            )
        ),
    )
    assert [sorted(r.source for r in hits) for hits in results] == [
        [],
        [],
        ["t2"],
        ["t2"],
    ]


@pytest.mark.parametrize("search_type", list(SearchType), ids=lambda t: t.value)
def test_relationships_and_claims_are_filtered_by_their_documents(
    harness: StoreHarness, search_type: SearchType
) -> None:
    # The relationship and claim indexes carry their documents' filters, so
    # the store applies an attribute filter to them (they were searched
    # unfiltered, and only the strategy's text-unit check excluded them).
    h = harness
    h.initialize()
    edges = [
        r.model_copy(update={"attributes": {"filters": {"category": category}}})
        for r, category in zip(_relationships(), ("contract", "logistics"), strict=True)
    ]
    claims = [
        Claim(
            id=f"c-{category}",
            subject_id="e-vendor",
            subject_name="Vendor",
            object_name="",
            type="obligation",
            description=f"Vendor ships widgets ({category})",
            attributes={"filters": {"category": category}},
        )
        for category in ("contract", "logistics")
    ]
    assert h.vector_indexer.index_relationships(edges).failed_items == 0
    assert h.vector_indexer.index_claims(claims).failed_items == 0
    o = _prefixes(h)
    results = _search(
        h,
        *(
            _query("widgets", prefix, search_type=search_type, filters=filters)
            for prefix in (o.relationships_index_prefix, o.claims_index_prefix)
            for filters in (
                {"attr_category": "logistics"},
                {"attributes.filters.category": "logistics"},
                {"attr_category": "nomatch"},
            )
        ),
    )
    assert [sorted(r.source for r in hits) for hits in results] == [
        ["r-ships"],
        ["r-ships"],
        [],
        ["c-logistics"],
        ["c-logistics"],
        [],
    ]


_SECRET = {"category": "logistics", "clearance": "secret"}


def _filtered_items(filters: dict[str, Any]) -> tuple[Entity, Relationship, Claim]:
    attributes = {"filters": dict(filters)} if filters else None
    return (
        _entities()[0].model_copy(update={"attributes": attributes}),
        _relationships()[0].model_copy(update={"attributes": attributes}),
        Claim(
            id="c-supplies",
            subject_id="e-vendor",
            subject_name="Vendor",
            object_name="",
            type="obligation",
            description="Vendor supplies widgets",
            attributes=attributes,
        ),
    )


@pytest.mark.parametrize(
    "kept",
    [{"category": "contract"}, {}],
    ids=["other-filters", "no-filters"],
)
def test_a_lineage_update_clears_the_filter_keys_an_item_lost(
    harness: StoreHarness, kept: dict[str, Any]
) -> None:
    # Regression: a lineage update is a partial update, which keeps every
    # field it does not name, so the filters of a removed document stayed
    # on the entity, relationship and claim and still matched.
    h = harness
    h.initialize()
    indexer = h.vector_indexer
    entity, edge, claim = _filtered_items(_SECRET)
    for stats in (
        indexer.index_entities([entity]),
        indexer.index_relationships([edge]),
        indexer.index_claims([claim]),
    ):
        assert stats.failed_items == 0, stats.errors
    entity, edge, claim = _filtered_items(kept)
    for stats in (
        indexer.update_entity_lineage([entity]),
        indexer.update_relationship_lineage([edge]),
        indexer.update_claim_lineage([claim]),
    ):
        assert stats.failed_items == 0, stats.errors
    o = _prefixes(h)
    prefixes = (
        o.entities_index_prefix,
        o.relationships_index_prefix,
        o.claims_index_prefix,
    )
    filters = (
        {"attr_clearance": "secret"},
        {"attributes.filters.clearance": "secret"},
        {"attr_category": "logistics"},
        {"attr_category": "contract"},
    )
    results = _search(
        h,
        *(
            _query("widgets", prefix, search_type=SearchType.LEXICAL, filters=f)
            for prefix in prefixes
            for f in filters
        ),
    )
    found = [[r.source for r in hits] for hits in results]
    # A claim reads back with the document filters it stores, so a claim a
    # later run rewrites keeps them.
    (read,) = indexer.read_claims(["c-supplies"], suffix=_SUFFIX)
    assert (read.attributes or {}).get("filters") == (kept or None)
    assert found == [
        hit
        for item in ("e-vendor", "r-supplies", "c-supplies")
        for hit in [[], [], [], [item] if kept else []]
    ]


def test_pruning_communities_keeps_the_text_units_filters(
    harness: StoreHarness,
) -> None:
    # Regression: the prune of deleted communities' ids wrote the units it
    # built with no attributes through the filter-clearing update, nulling
    # every stored filter key: an unchanged document's units dropped out of
    # filtered retrieval for good, and later derivations lost the filters.
    h = harness
    h.initialize()
    units = [
        u.model_copy(update={"community_ids": ["c-gone", "c-kept"]})
        for u in _text_units()
    ]
    assert h.vector_indexer.index_text_units(units).failed_items == 0
    stats = h.vector_indexer.prune_text_unit_communities(["c-gone"], suffix=_SUFFIX)
    assert stats.failed_items == 0, stats.errors
    o = _prefixes(h)
    results = _search(
        h,
        *(
            _query(
                "", o.text_units_index_prefix, search_type=SearchType.LEXICAL, filters=f
            )
            for f in (
                {"attr_category": "logistics"},
                {"attributes.filters.category": "contract"},
                {"community_ids": "c-gone"},
                {"community_ids": "c-kept"},
            )
        ),
    )
    assert [sorted(r.source for r in hits) for hits in results] == [
        ["t2"],
        ["t1"],
        [],
        ["t1", "t2", "t3"],
    ]


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
    # Lexical and vector search rank this corpus differently, so fusing with
    # the weights swapped puts the other search's order first. BM25 favours
    # "rare", the one document with the query's rare terms; the bag-of-words
    # cosine favours the short documents of the common words.
    harness.initialize()
    units = [TextUnit(id="rare", text="Vendor ship zebra yak xylophone walrus")]
    units += [
        TextUnit(id=f"common{i}", text=f"which does which does filler{i}")
        for i in range(6)
    ]
    assert harness.vector_indexer.index_text_units(units).failed_items == 0
    prefix = _prefixes(harness).text_units_index_prefix
    text = "Which widgets does Vendor ship?"

    def weighted(lexical: float, vector: float) -> StoreHarness:
        def change(config: Any) -> None:
            config.search.hybrid.lexical_weight = lexical
            config.search.hybrid.vector_weight = vector

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
    assert lexical[0].source == "rare"
    assert vector[0].source != "rare"
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
        lambda c: setattr(c.models.embedding, "model_id", "another-model")
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
    graph_harness: StoreHarness,
) -> None:
    # Each seed has its own neighbours; one traversal-wide limit used to let
    # the first seeds' neighbourhoods consume it, and the seeds themselves
    # were never emitted.
    h = graph_harness.derive(
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


def test_graph_expansion_budget_caps_each_seed_breadth_first(
    graph_harness: StoreHarness,
) -> None:
    # Neptune applied the per-seed budget (a limit() inside repeat()) to each
    # traverser of each hop: on this tree of fan-out 8 a budget of 5 fetched
    # 5 + 25 entities. The budget caps the seed's total, hop 1 first.
    def change(config: Any) -> None:
        config.indexing.neptune.traversal_fetch_multiplier = 1  # no cut
        config.indexing.neptune.max_hops = 2

    h = graph_harness.derive(change)
    near = [f"n{i}" for i in range(8)]
    far = [f"{n}-f{j}" for n in near for j in range(8)]
    entities = [
        Entity(id=i, name=i, type="ORG", description=f"{i} description", rank=1)
        for i in ["root", *near, *far]
    ]
    edges = [("root", n) for n in near] + [(f.split("-")[0], f) for f in far]
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

    within_hop_one, across_hops = _graph_retrieve(
        h,
        *(
            SearchQuery(
                query="",
                suffix=_SUFFIX,
                top_k=top_k,
                retrieval_multiplier=1,
                filters={"id": ["root"]},
            )
            for top_k in (5, 12)
        ),
    )
    first = [r.source for r in within_hop_one]
    assert first[0] == "root"
    assert len(first) == 6 and set(first[1:]) <= set(near), first
    wide = [r.source for r in across_hops]
    assert len(wide) == 13, wide
    assert set(wide[1:9]) == set(near) and set(wide[9:]) <= set(far), wide


def test_graph_expansion_filters_and_is_deterministic(
    graph_harness: StoreHarness,
) -> None:
    harness = graph_harness
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
