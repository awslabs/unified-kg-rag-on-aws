# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for claim (covariate) vector indexing — connecting the former
extract-only dead-end to retrieval (AWS-free)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aws_graphrag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from aws_graphrag.domain.models import Claim

pytestmark = pytest.mark.unit


@pytest.fixture
def indexer(mocker):
    mocker.patch("aws_graphrag.adapters.storage.opensearch_indexer.OpenSearchClient")
    factory = mocker.patch(
        "aws_graphrag.adapters.storage.opensearch_indexer.BedrockEmbeddingModelFactory"
    )
    factory.return_value.get_model_info.return_value = MagicMock(dimensions=1024)
    factory.return_value.get_model.return_value = MagicMock()
    from aws_graphrag.domain.models import Config

    return OpenSearchIndexer(config=Config())


def _claim() -> Claim:
    return Claim(
        id="c1",
        subject_id="e1",
        subject_name="Alice",
        object_id="e2",
        object_name="Acme",
        type="EMPLOYMENT",
        status="TRUE",
        description="Alice is employed by Acme",
        source_text="...",
    )


def test_prepare_claim_doc_embeds_description(indexer) -> None:
    doc = indexer._prepare_claim_doc(_claim(), ([0.1, 0.2],))
    assert doc["id"] == "c1"
    assert doc["subject_name"] == "Alice"
    assert doc["object_name"] == "Acme"
    assert doc["type"] == "EMPLOYMENT"
    assert doc["status"] == "TRUE"
    assert doc["description"] == "Alice is employed by Acme"
    assert doc["description_embedding"] == [0.1, 0.2]


def test_claims_mapping_has_knn_vector(indexer) -> None:
    props = indexer._get_claims_mapping()["mappings"]["properties"]
    assert props["description_embedding"]["type"] == "knn_vector"
    assert props["subject_id"]["type"] == "keyword"
    assert props["description"]["type"] == "text"


def test_claims_index_is_retrievable(indexer) -> None:
    # The claims index prefix is wired into the retriever field mappings.
    from aws_graphrag.adapters.retrievers.opensearch_retriever import (
        OpenSearchRetriever,
    )
    from aws_graphrag.domain.models import Config

    retriever = OpenSearchRetriever.__new__(OpenSearchRetriever)
    retriever._opensearch_config = Config().indexing.opensearch
    retriever._config = Config()
    mappings = retriever._initialize_field_mappings()
    claims_prefix = Config().indexing.opensearch.claims_index_prefix
    assert claims_prefix in mappings
    assert "description_embedding" in mappings[claims_prefix]["vector"]
