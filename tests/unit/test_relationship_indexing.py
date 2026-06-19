# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for relationship vector indexing (M3, AWS-free).

Exercises the pure doc-preparation and mapping logic that backs the LightRAG
high-level (relationship) retrieval path, without constructing AWS clients.
"""
from __future__ import annotations

import pytest

from aws_graphrag.models import Relationship
from aws_graphrag.storage.opensearch_indexer import OpenSearchIndexer

pytestmark = pytest.mark.unit


def test_prepare_relationship_doc_embeds_description() -> None:
    rel = Relationship(
        id="r1",
        source_id="e1",
        target_id="e2",
        source_name="Alice",
        target_name="Acme",
        description="Alice works at Acme",
        weight=0.7,
        rank=3,
    )
    doc = OpenSearchIndexer._prepare_relationship_doc(rel, ([0.1, 0.2, 0.3],))

    assert doc["id"] == "r1"
    assert doc["source_id"] == "e1"
    assert doc["target_id"] == "e2"
    assert doc["source_name"] == "Alice"
    assert doc["target_name"] == "Acme"
    assert doc["description"] == "Alice works at Acme"
    assert doc["description_embedding"] == [0.1, 0.2, 0.3]
    assert doc["weight"] == 0.7
    assert doc["rank"] == 3


def test_prepare_relationship_doc_degrades_none_to_empty() -> None:
    rel = Relationship(id="r1", source_id="e1", target_id="e2")
    doc = OpenSearchIndexer._prepare_relationship_doc(rel, ([0.0],))
    assert doc["description"] == ""
    assert doc["source_name"] == ""
    assert doc["target_name"] == ""
    assert doc["weight"] == 1.0  # default when None


def test_relationships_mapping_has_knn_vector(config) -> None:
    # Build the indexer without touching AWS by stubbing the heavy __init__ deps.
    indexer = OpenSearchIndexer.__new__(OpenSearchIndexer)
    indexer.opensearch_config = config.indexing.opensearch
    indexer.analyzer = "standard"
    indexer._embedding_dimension = 1024
    mapping = indexer._get_relationships_mapping()
    props = mapping["mappings"]["properties"]
    assert props["description_embedding"]["type"] == "knn_vector"
    assert props["description"]["type"] == "text"
    assert props["source_id"]["type"] == "keyword"
