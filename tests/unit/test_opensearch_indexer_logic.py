# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for OpenSearchIndexer pure document/mapping logic.

Complements ``test_relationship_indexing.py`` (relationship doc + mapping) and
``test_embedding_cache.py`` (``_batch_embed`` cache/dedup). Here we cover the
remaining ``_prepare_*_doc`` shapes, the ``_get_*_mapping`` builders, the
shared ``_prepare_common_doc_properties`` / ``_prepare_documents`` helpers, and
``delete_by_id`` stats accounting — all without constructing any AWS client.

The indexer is built via ``__new__`` so the AWS-client ``__init__`` never runs;
only the attributes the methods under test touch are set.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import (
    Claim,
    CommunityReport,
    Config,
    Entity,
    TextUnit,
)
from unified_kg_rag.ports.indexer import IndexingStats

pytestmark = pytest.mark.unit


@pytest.fixture
def indexer(config: Config) -> OpenSearchIndexer:
    inst = OpenSearchIndexer.__new__(OpenSearchIndexer)
    inst.config = config
    inst.opensearch_config = config.indexing.opensearch
    inst.analyzer = "standard"
    inst.target_language = config.processing.translation.target_language.value
    inst._embedding_dimension = 1024
    return inst


# --------------------------------------------------------------------------- #
# _prepare_entity_doc
# --------------------------------------------------------------------------- #


def test_prepare_entity_doc_full(indexer) -> None:
    entity = Entity(
        id="e1",
        name="Alice",
        description="A researcher",
        type="PERSON",
        rank=5,
        confidence=0.9,
    )
    doc = indexer._prepare_entity_doc(entity, ([0.1], [0.2]))
    assert doc["id"] == "e1"
    assert doc["name"] == "Alice"
    assert doc["name_embedding"] == [0.1]
    assert doc["description"] == "A researcher"
    assert doc["description_embedding"] == [0.2]
    assert doc["type"] == "PERSON"
    assert doc["rank"] == 5
    assert doc["confidence"] == 0.9


def test_prepare_entity_doc_defaults_when_none(indexer) -> None:
    entity = Entity(id="e1", name="Alice", confidence=None)
    doc = indexer._prepare_entity_doc(entity, ([0.0], [0.0]))
    assert doc["description"] == ""
    assert doc["type"] == ""
    assert doc["rank"] == 1.0
    assert doc["confidence"] == 1.0  # None -> 1.0 default


def test_prepare_entity_doc_confidence_zero_preserved(indexer) -> None:
    # confidence 0.0 is a real value, not "missing" — must not coerce to 1.0.
    entity = Entity(id="e1", name="Alice", confidence=0.0)
    doc = indexer._prepare_entity_doc(entity, ([0.0], [0.0]))
    assert doc["confidence"] == 0.0


# --------------------------------------------------------------------------- #
# _prepare_text_unit_doc
# --------------------------------------------------------------------------- #


def test_prepare_text_unit_doc_basic(indexer) -> None:
    unit = TextUnit(id="t1", text="hello world", n_tokens=3)
    doc = indexer._prepare_text_unit_doc(unit, ([0.5],))
    assert doc["id"] == "t1"
    assert doc["text"] == "hello world"
    assert doc["text_embedding"] == [0.5]
    assert doc["n_tokens"] == 3
    # No community_ids / translation keys present.
    assert "community_ids" not in doc


def test_prepare_text_unit_doc_includes_community_ids(indexer) -> None:
    unit = TextUnit(id="t1", text="x", community_ids=["c1", "c2"])
    doc = indexer._prepare_text_unit_doc(unit, ([0.0],))
    assert doc["community_ids"] == ["c1", "c2"]


def test_prepare_text_unit_doc_adds_translation_key(indexer) -> None:
    lang = indexer.target_language
    unit = TextUnit(id="t1", text="hola", translated_texts={lang: "hello"})
    doc = indexer._prepare_text_unit_doc(unit, ([0.0],))
    assert doc[f"translated_text_{lang}"] == "hello"


def test_text_unit_embedding_text_prefers_translation(indexer) -> None:
    lang = indexer.target_language
    unit = TextUnit(id="t1", text="hola", translated_texts={lang: "hello"})
    assert indexer._text_unit_embedding_text(unit) == "hello"


def test_text_unit_embedding_text_falls_back_to_text(indexer) -> None:
    unit = TextUnit(id="t1", text="hola")
    assert indexer._text_unit_embedding_text(unit) == "hola"


# --------------------------------------------------------------------------- #
# community report + claim doc shape
# --------------------------------------------------------------------------- #


def test_index_community_reports_prepare_doc_via_closure(indexer, mocker) -> None:
    # index_community_reports defines prepare_doc inline; capture it through
    # _index_item_type to test the doc shape it produces.
    captured = {}

    def fake_index_item_type(**kwargs):
        captured["prepare"] = kwargs["prepare_doc_func"]
        return IndexingStats()

    mocker.patch.object(indexer, "_index_item_type", side_effect=fake_index_item_type)
    report = CommunityReport(
        id="cr1",
        community_id="c1",
        name="Cluster",
        summary="A summary",
        full_content="Full body",
        rank=4,
    )
    indexer.index_community_reports([report])
    doc = captured["prepare"](report, ([0.1], [0.2], [0.3]))
    assert doc["community_id"] == "c1"
    assert doc["name"] == "Cluster"
    assert doc["name_embedding"] == [0.1]
    assert doc["summary"] == "A summary"
    assert doc["summary_embedding"] == [0.2]
    assert doc["full_content"] == "Full body"
    assert doc["full_content_embedding"] == [0.3]
    assert doc["rank"] == 4


def test_prepare_claim_doc_shape(indexer) -> None:
    claim = Claim(
        id="cl1",
        subject_id="e1",
        subject_name="Alice",
        object_id="e2",
        object_name="Acme",
        type="employment",
        status="TRUE",
        description="works at",
        source_text="Alice works at Acme.",
    )
    doc = OpenSearchIndexer._prepare_claim_doc(claim, ([0.9],))
    assert doc["id"] == "cl1"
    assert doc["subject_id"] == "e1"
    assert doc["subject_name"] == "Alice"
    assert doc["object_id"] == "e2"
    assert doc["object_name"] == "Acme"
    assert doc["type"] == "employment"
    assert doc["status"] == "TRUE"
    assert doc["description"] == "works at"
    assert doc["description_embedding"] == [0.9]
    assert doc["source_text"] == "Alice works at Acme."


def test_prepare_claim_doc_degrades_none(indexer) -> None:
    claim = Claim(
        id="cl1",
        subject_id="e1",
        subject_name="Alice",
        object_name="literal",
        type="t",
    )
    doc = OpenSearchIndexer._prepare_claim_doc(claim, ([0.0],))
    assert doc["status"] == ""
    assert doc["description"] == ""
    assert doc["source_text"] == ""


# --------------------------------------------------------------------------- #
# _prepare_common_doc_properties
# --------------------------------------------------------------------------- #


def test_common_doc_properties_id_only(indexer) -> None:
    entity = Entity(id="e1", name="Alice")
    doc = OpenSearchIndexer._prepare_common_doc_properties(entity)
    assert doc == {"id": "e1"}


def test_common_doc_properties_includes_attributes_and_filters(indexer) -> None:
    entity = Entity(
        id="e1",
        name="Alice",
        attributes={"sector": "tech", "filters": {"region": "us", "tier": None}},
    )
    doc = OpenSearchIndexer._prepare_common_doc_properties(entity)
    assert doc["id"] == "e1"
    assert doc["attributes"]["sector"] == "tech"
    # Each non-None filter is flattened with the attr_ prefix.
    assert doc["attr_region"] == "us"
    assert "attr_tier" not in doc


# --------------------------------------------------------------------------- #
# _prepare_documents (skips failed embeddings, propagates ids)
# --------------------------------------------------------------------------- #


def test_prepare_documents_skips_none_embeddings(indexer) -> None:
    items = [Entity(id="e1", name="A"), Entity(id="e2", name="B")]
    embeddings = [([0.1], [0.2]), ([0.3], None)]  # e2 has a failed embedding
    docs, failed = OpenSearchIndexer._prepare_documents(
        items, embeddings, indexer._prepare_entity_doc
    )
    assert [d["id"] for d in docs] == ["e1"]
    assert failed == ["e2"]


def test_prepare_documents_all_ok(indexer) -> None:
    items = [Entity(id="e1", name="A")]
    docs, failed = OpenSearchIndexer._prepare_documents(
        items, [([0.1], [0.2])], indexer._prepare_entity_doc
    )
    assert len(docs) == 1
    assert failed == []


# --------------------------------------------------------------------------- #
# mapping builders
# --------------------------------------------------------------------------- #


def _props(mapping: dict) -> dict:
    return mapping["mappings"]["properties"]


def test_base_mapping_settings(indexer) -> None:
    mapping = indexer._get_base_mapping({"id": {"type": "keyword"}})
    settings = mapping["settings"]
    assert settings["index.knn"] is True
    assert "number_of_shards" in settings
    # dynamic_templates map strings to keyword.
    templates = mapping["mappings"]["dynamic_templates"]
    assert templates[0]["strings_as_keywords"]["mapping"]["type"] == "keyword"


def test_knn_vector_mapping_uses_dimension(indexer) -> None:
    m = indexer._get_knn_vector_mapping()
    assert m["type"] == "knn_vector"
    assert m["dimension"] == 1024
    assert "method" in m


def test_entities_mapping(indexer) -> None:
    props = _props(indexer._get_entities_mapping())
    assert props["name"]["type"] == "text"
    assert props["name"]["analyzer"] == "standard"
    assert props["name"]["fields"]["keyword"]["type"] == "keyword"
    assert props["name_embedding"]["type"] == "knn_vector"
    assert props["description_embedding"]["type"] == "knn_vector"
    assert props["type"]["type"] == "keyword"


def test_text_units_mapping(indexer) -> None:
    lang = indexer.target_language
    props = _props(indexer._get_text_units_mapping())
    assert props["text"]["type"] == "text"
    assert props["text_embedding"]["type"] == "knn_vector"
    assert props[f"translated_text_{lang}"]["analyzer"] == "standard"
    assert props["n_tokens"]["type"] == "integer"


def test_claims_mapping(indexer) -> None:
    props = _props(indexer._get_claims_mapping())
    assert props["subject_id"]["type"] == "keyword"
    assert props["object_id"]["type"] == "keyword"
    assert props["description_embedding"]["type"] == "knn_vector"
    assert props["status"]["type"] == "keyword"


def test_community_reports_mapping(indexer) -> None:
    props = _props(indexer._get_community_reports_mapping())
    assert props["community_id"]["type"] == "keyword"
    assert props["summary_embedding"]["type"] == "knn_vector"
    assert props["full_content_embedding"]["type"] == "knn_vector"
    assert props["rank"]["type"] == "double"


# --------------------------------------------------------------------------- #
# delete_by_id stats accounting
# --------------------------------------------------------------------------- #


def test_delete_by_id_empty_ids(indexer) -> None:
    stats = indexer.delete_by_id([], "graphrag-entities", "default")
    assert stats.total_items == 0
    assert stats.successful_items == 0


def test_delete_by_id_no_live_index(indexer, mocker) -> None:
    client = mocker.MagicMock()
    client.get_index_name_by_alias.return_value = None
    indexer.opensearch_client = client
    stats = indexer.delete_by_id(["a", "b"], "graphrag-entities", "default")
    # No index -> nothing deleted, but total reflects the requested ids.
    assert stats.total_items == 2
    assert stats.successful_items == 0
    client.bulk_delete.assert_not_called()


def test_delete_by_id_counts_success_minus_errors(indexer, mocker) -> None:
    client = mocker.MagicMock()
    client.get_index_name_by_alias.return_value = "graphrag-entities-default-1"
    # bulk_delete "items" carries ONLY error results; 1 of 3 failed.
    client.bulk_delete.return_value = {"errors": True, "items": [{"x": "err"}]}
    indexer.opensearch_client = client

    stats = indexer.delete_by_id(["a", "b", "c"], "graphrag-entities", "default")
    assert stats.total_items == 3
    assert stats.successful_items == 2
    assert stats.failed_items == 1


def test_delete_by_id_all_success_when_no_errors(indexer, mocker) -> None:
    client = mocker.MagicMock()
    client.get_index_name_by_alias.return_value = "graphrag-entities-default-1"
    client.bulk_delete.return_value = {"errors": False, "items": []}
    indexer.opensearch_client = client
    stats = indexer.delete_by_id(["a", "b"], "graphrag-entities", "default")
    assert stats.successful_items == 2
    assert stats.failed_items == 0


def test_delete_by_id_exception_counts_all_failed(indexer, mocker) -> None:
    client = mocker.MagicMock()
    client.get_index_name_by_alias.return_value = "graphrag-entities-default-1"
    client.bulk_delete.side_effect = RuntimeError("boom")
    indexer.opensearch_client = client
    stats = indexer.delete_by_id(["a", "b"], "graphrag-entities", "default")
    assert stats.failed_items == 2
    assert stats.successful_items == 0
