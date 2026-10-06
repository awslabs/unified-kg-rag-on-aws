# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Items without a description must still reach the OpenSearch vector indices.

Stub entities created from relationship endpoints carry no description; an
empty embedding text used to yield no vector, so the item was skipped and
counted as an indexing failure (able to trip ``max_failure_rate``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import Claim, Config, Entity, Relationship

pytestmark = pytest.mark.unit


@pytest.fixture
def indexer(mocker) -> OpenSearchIndexer:
    mocker.patch("unified_kg_rag.adapters.storage.opensearch_indexer.OpenSearchClient")
    factory = mocker.patch(
        "unified_kg_rag.adapters.storage.opensearch_indexer.BedrockEmbeddingModelFactory"
    )
    factory.return_value.get_model_info.return_value = MagicMock(dimensions=4)
    factory.return_value.get_model.return_value = MagicMock()
    ix = OpenSearchIndexer(config=Config())
    embedded: list[str] = []

    def _embed(texts: list[str]) -> list[list[float]]:
        embedded.extend(texts)
        return [[1.0] * 4 for _ in texts]

    ix.embedding_model.embed_documents = _embed
    ix._embedded_log = embedded  # type: ignore[attr-defined]
    ix.opensearch_client.get_index_name_by_alias.return_value = "live-index"
    ix.opensearch_client.bulk_index.return_value = {"errors": False, "items": []}
    return ix


def test_entity_fallback_text_uses_name_and_type() -> None:
    assert (
        OpenSearchIndexer._entity_description_text(
            Entity(id="e1", name="Vendor", type="ORGANIZATION")
        )
        == "Vendor (ORGANIZATION)"
    )
    assert (
        OpenSearchIndexer._entity_description_text(Entity(id="e1", name="Vendor"))
        == "Vendor"
    )
    assert (
        OpenSearchIndexer._entity_description_text(
            Entity(id="e1", name="Vendor", description="  ")
        )
        == "Vendor"
    )
    assert (
        OpenSearchIndexer._entity_description_text(
            Entity(id="e1", name="Vendor", description="Supplies parts")
        )
        == "Supplies parts"
    )


def test_relationship_fallback_text_uses_endpoints() -> None:
    rel = Relationship(
        id="r1",
        source_id="e1",
        target_id="e2",
        source_name="Vendor",
        target_name="Buyer",
    )
    assert OpenSearchIndexer._relationship_description_text(rel) == (
        "Vendor related to Buyer"
    )
    rel_typed = rel.model_copy(update={"type": "SUPPLIES"})
    assert OpenSearchIndexer._relationship_description_text(rel_typed) == (
        "Vendor SUPPLIES Buyer"
    )


def test_claim_fallback_text_uses_subject_type_object() -> None:
    claim = Claim(
        id="c1",
        subject_id="e1",
        subject_name="Vendor",
        object_name="Buyer",
        type="PAYMENT OBLIGATION",
    )
    assert OpenSearchIndexer._claim_description_text(claim) == (
        "Vendor PAYMENT OBLIGATION Buyer"
    )


def test_upsert_stub_entity_without_description_is_indexed(indexer) -> None:
    stats = indexer.upsert_entities(
        [
            Entity(id="e1", name="Vendor", type="ORGANIZATION"),
            Entity(id="e2", name="Buyer", description="Purchases parts"),
        ]
    )
    assert stats.failed_items == 0
    assert stats.successful_items == 2
    docs = indexer.opensearch_client.bulk_index.call_args.args[1]
    stub = next(d for d in docs if d["id"] == "e1")
    # The stored description stays empty; only the vector uses the surrogate.
    assert stub["description"] == ""
    assert stub["description_embedding"] is not None
    assert "Vendor (ORGANIZATION)" in indexer._embedded_log


def test_upsert_relationship_without_description_is_indexed(indexer) -> None:
    stats = indexer.upsert_relationships(
        [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                source_name="Vendor",
                target_name="Buyer",
            )
        ]
    )
    assert stats.failed_items == 0
    assert stats.successful_items == 1


def test_upsert_claim_without_description_is_indexed(indexer) -> None:
    stats = indexer.upsert_claims(
        [
            Claim(
                id="c1",
                subject_id="e1",
                subject_name="Vendor",
                object_name="Buyer",
                type="DELIVERY",
            )
        ]
    )
    assert stats.failed_items == 0
    assert stats.successful_items == 1
