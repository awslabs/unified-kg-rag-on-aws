# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenSearch indexer and retriever end to end on the embedded SQLite store.

``providers.vector.kind: sqlite`` builds the unchanged ``OpenSearchIndexer``
and ``OpenSearchRetriever`` on a ``SQLiteSearchEngine``. This runs a full
build, the incremental writes (upserts, lineage updates, deletions) of every
artifact type and the three search types through the provider registry, with
a hashing embedding provider: no server, no AWS call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrievers import OpenSearchRetriever
from unified_kg_rag.adapters.storage import OpenSearchIndexer
from unified_kg_rag.domain.models import (
    Claim,
    CommunityReport,
    Config,
    Entity,
    Relationship,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchType,
    TextUnit,
)
from unified_kg_rag.domain.models.config import LanguageCode
from unified_kg_rag.shared import IndexNotFoundError

pytestmark = pytest.mark.integration

_SUFFIX = "default"


def _config(tmp_path: Path) -> Config:
    config = Config()
    config.providers.vector.kind = "sqlite"
    config.providers.vector.sqlite_path = str(tmp_path / "vector.sqlite3")
    return config


def _providers(config: Config) -> Providers:
    return Providers(config, embedding_factory=HashingEmbeddingFactory())


def _retrieve(
    providers: Providers, *queries: SearchQuery
) -> list[list[RetrievalResult]]:
    async def run() -> list[list[RetrievalResult]]:
        retriever = providers.build_retriever(RetrieverRole.DOCUMENT)
        try:
            return [await retriever.aretrieve(query) for query in queries]
        finally:
            await retriever.aclose()

    return asyncio.run(run())


def _entities() -> list[Entity]:
    return [
        Entity(
            id="e-vendor",
            name="Vendor",
            type="ORG",
            description="Vendor supplies widgets",
            text_unit_ids=["t1", "t2"],
            rank=3,
        ),
        Entity(
            id="e-buyer",
            name="Buyer",
            type="ORG",
            description="Buyer orders widgets",
            text_unit_ids=["t1"],
            rank=2,
        ),
        Entity(
            id="e-depot",
            name="Depot",
            type="LOCATION",
            description="Depot stores widgets",
            text_unit_ids=["t2"],
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
            description="Vendor supplies widgets to Buyer",
            weight=1.0,
            text_unit_ids=["t1"],
        ),
        Relationship(
            id="r-ships",
            source_id="e-vendor",
            target_id="e-depot",
            source_name="Vendor",
            target_name="Depot",
            description="Vendor ships widgets from Depot",
            weight=0.5,
            text_unit_ids=["t2"],
        ),
    ]


def _claims() -> list[Claim]:
    return [
        Claim(
            id="cl1",
            subject_id="e-vendor",
            subject_name="Vendor",
            object_name="Buyer",
            type="OBLIGATION",
            description="Vendor must deliver widgets every week",
            source_text="Vendor supplies widgets",
            text_unit_ids=["t1"],
        )
    ]


def _reports() -> list[CommunityReport]:
    return [
        CommunityReport(
            id="cr1",
            community_id="c1",
            name="Widget supply",
            summary="Vendor supplies widgets to Buyer through the Depot.",
            full_content="Widget supply\n\nVendor, Buyer and Depot form the supply.",
            rank=2.0,
            text_unit_ids=["t1", "t2"],
        )
    ]


def _text_units() -> list[TextUnit]:
    return [
        TextUnit(
            id="t1",
            text="Vendor supplies widgets to Buyer under a purchase order.",
        ),
        TextUnit(id="t2", text="Vendor ships widgets from the Depot every week."),
    ]


def _ids(engine: SQLiteSearchEngine, alias: str) -> set[str]:
    response = engine.search(
        index=alias, body={"size": 100, "query": {"match_all": {}}}
    )
    return {hit["_id"] for hit in response["hits"]["hits"]}


def _source(engine: SQLiteSearchEngine, alias: str, doc_id: str) -> dict:
    (hit,) = engine.search(index=alias, body={"query": {"ids": {"values": [doc_id]}}})[
        "hits"
    ]["hits"]
    return dict(hit["_source"])


def test_every_artifact_type_round_trips(tmp_path: Path) -> None:
    config = _config(tmp_path)
    providers = _providers(config)
    indexer = providers.build_vector_indexer()
    assert isinstance(indexer, OpenSearchIndexer)
    engine = indexer.opensearch_client
    assert isinstance(engine, SQLiteSearchEngine)
    o = config.indexing.opensearch
    alias = {
        "text_units": indexer._get_name(o.text_units_index_prefix, _SUFFIX),
        "entities": indexer._get_name(o.entities_index_prefix, _SUFFIX),
        "relationships": indexer._get_name(o.relationships_index_prefix, _SUFFIX),
        "claims": indexer._get_name(o.claims_index_prefix, _SUFFIX),
        "reports": indexer._get_name(o.community_reports_index_prefix, _SUFFIX),
    }
    try:
        # Full build.
        indexer.initialize()
        for stats in (
            indexer.index_text_units(_text_units()),
            indexer.index_entities(_entities()),
            indexer.index_relationships(_relationships()),
            indexer.index_claims(_claims()),
            indexer.index_community_reports(_reports()),
        ):
            assert stats.failed_items == 0, stats.errors
        assert _ids(engine, alias["entities"]) == {"e-vendor", "e-buyer", "e-depot"}
        assert _ids(engine, alias["relationships"]) == {"r-supplies", "r-ships"}

        # Incremental: upsert a new and a changed item of every type.
        new_unit = TextUnit(id="t3", text="Buyer returns faulty widgets to Vendor.")
        changed = _entities()[0].model_copy(
            update={"description": "Vendor sells gears"}
        )
        new_entity = Entity(id="e-courier", name="Courier", type="ORG", rank=1)
        for stats in (
            indexer.upsert_text_units([new_unit]),
            indexer.upsert_entities([changed, new_entity]),
            indexer.upsert_relationships(
                [_relationships()[0].model_copy(update={"weight": 2.0})]
            ),
            indexer.upsert_claims([_claims()[0].model_copy(update={"status": "TRUE"})]),
            indexer.upsert_community_reports(
                [_reports()[0].model_copy(update={"summary": "Gears and widgets."})]
            ),
        ):
            assert stats.failed_items == 0, stats.errors
        assert _ids(engine, alias["text_units"]) == {"t1", "t2", "t3"}
        assert _source(engine, alias["entities"], "e-vendor")["description"] == (
            "Vendor sells gears"
        )
        assert _source(engine, alias["relationships"], "r-supplies")["weight"] == 2.0
        assert _source(engine, alias["claims"], "cl1")["status"] == "TRUE"
        assert _source(engine, alias["reports"], "cr1")["summary"] == (
            "Gears and widgets."
        )
        assert indexer.read_entity_descriptions(["e-vendor"]) == {
            "e-vendor": "Vendor sells gears"
        }

        # Lineage updates rewrite one field and keep the vectors.
        for stats in (
            indexer.update_entity_lineage(
                [changed.model_copy(update={"text_unit_ids": ["t1"]})]
            ),
            indexer.update_relationship_lineage(
                [_relationships()[1].model_copy(update={"text_unit_ids": ["t3"]})]
            ),
            indexer.update_claim_lineage(
                [_claims()[0].model_copy(update={"text_unit_ids": ["t3"]})]
            ),
        ):
            assert stats.failed_items == 0, stats.errors
        assert _source(engine, alias["entities"], "e-vendor")["text_unit_ids"] == ["t1"]
        (claim,) = indexer.read_claims(["cl1"], _SUFFIX)
        assert claim.text_unit_ids == ["t3"]
        knn = {
            "knn": {
                "description_embedding": {
                    "vector": HashingEmbeddingFactory()
                    .get_model("m")
                    .embed_query("Vendor sells gears"),
                    "k": 1,
                }
            }
        }
        (top,) = engine.search(index=alias["entities"], body={"query": knn})["hits"][
            "hits"
        ]
        assert top["_id"] == "e-vendor"

        # A document deletion removes its artifacts from every index.
        results = indexer.delete_document_artifacts(
            ["t2", "e-depot", "cr1"], _SUFFIX, extra_relationship_ids=["r-ships"]
        )
        assert all(stats.failed_items == 0 for stats in results.values())
        assert _ids(engine, alias["text_units"]) == {"t1", "t3"}
        assert _ids(engine, alias["entities"]) == {"e-vendor", "e-buyer", "e-courier"}
        assert _ids(engine, alias["relationships"]) == {"r-supplies"}
        assert _ids(engine, alias["reports"]) == set()
        assert (
            indexer.delete_by_id(["cl1"], o.claims_index_prefix, _SUFFIX).failed_items
            == 0
        )
        assert _ids(engine, alias["claims"]) == set()

        # The settings check of the next incremental run passes.
        indexer.check_index_settings(_SUFFIX)
    finally:
        indexer.close()


def test_retriever_search_types_through_the_registry(tmp_path: Path) -> None:
    config = _config(tmp_path)
    providers = _providers(config)
    indexer = providers.build_vector_indexer()
    try:
        indexer.initialize()
        assert indexer.index_text_units(_text_units()).failed_items == 0
        assert indexer.index_entities(_entities()).failed_items == 0
        assert indexer.index_relationships(_relationships()).failed_items == 0
        assert indexer.index_community_reports(_reports()).failed_items == 0
    finally:
        indexer.close()

    # A second engine on the file, as a query process has.
    retriever = providers.build_retriever(RetrieverRole.DOCUMENT)
    assert isinstance(retriever, OpenSearchRetriever)
    retriever.close()
    queries = [
        SearchQuery(
            query="Which widgets does Vendor supply?",
            search_type=search_type,
            suffix=_SUFFIX,
        )
        for search_type in SearchType
    ]
    by_id = SearchQuery(
        query="",
        suffix=_SUFFIX,
        top_k=2,
        retrieval_multiplier=1,
        filters={"id": ["e-buyer", "e-depot"]},
        index_prefixes=config.indexing.opensearch.entities_index_prefix,
    )
    *hit_lists, fetched = _retrieve(providers, *queries, by_id)
    for search_type, hits in zip(SearchType, hit_lists, strict=True):
        assert {"t1", "e-vendor"} <= {r.source for r in hits}, search_type
        assert all(not any(k.endswith("_embedding") for k in r.metadata) for r in hits)
    hybrid = hit_lists[list(SearchType).index(SearchType.HYBRID)]
    assert all(0.0 <= (r.score or 0.0) <= 1.0 for r in hybrid)
    assert sorted(r.source for r in fetched) == ["e-buyer", "e-depot"]
    # A suffix never ingested is reported, not answered from nothing.
    with pytest.raises(IndexNotFoundError, match="No indices found for suffix 'other'"):
        _retrieve(
            providers,
            SearchQuery(
                query="widgets", search_type=SearchType.LEXICAL, suffix="other"
            ),
        )


def test_untranslated_korean_corpus_has_lexical_recall(tmp_path: Path) -> None:
    # A Korean corpus with a Korean target is never translated, so only the
    # ``text`` field carries it, analyzed with the bigram stand-in for nori:
    # "홍길동" matches inside "홍길동으로부터" and "보증 기간" inside "보증기간은".
    # (Unlike nori, a name sharing one bigram also matches, e.g. "김철민" and
    # "김철수"; it ranks below a full match.)
    config = _config(tmp_path)
    config.processing.translation.source_language = LanguageCode.KO
    config.processing.translation.target_language = LanguageCode.KO
    providers = _providers(config)
    indexer = providers.build_vector_indexer()
    try:
        units = [
            TextUnit(id="k1", text="가나다상사는 홍길동으로부터 공급계약서를 받았다."),
            TextUnit(id="k2", text="보증기간은 납품일로부터 2년이다."),
            TextUnit(id="k3", text="김철수는 가나다상사의 직원이다."),
        ]
        assert indexer.index_text_units(units).failed_items == 0
    finally:
        indexer.close()
    prefix = config.indexing.opensearch.text_units_index_prefix
    person, warranty, contract, company = (
        [r.source for r in hits]
        for hits in _retrieve(
            providers,
            *(
                SearchQuery(
                    query=text,
                    search_type=SearchType.LEXICAL,
                    suffix=_SUFFIX,
                    index_prefixes=prefix,
                )
                for text in ("홍길동", "보증 기간", "공급 계약", "가나다상사")
            ),
        )
    )
    assert person == ["k1"]
    assert warranty == ["k2"]
    assert contract == ["k1"]
    assert sorted(company) == ["k1", "k3"]
