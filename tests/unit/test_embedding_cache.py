# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for the content-hash embedding cache (perf/cost hardening)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aws_graphrag.adapters.storage.opensearch_indexer import OpenSearchIndexer

pytestmark = pytest.mark.unit


@pytest.fixture
def indexer(mocker):
    mocker.patch("aws_graphrag.adapters.storage.opensearch_indexer.OpenSearchClient")
    factory = mocker.patch(
        "aws_graphrag.adapters.storage.opensearch_indexer.BedrockEmbeddingModelFactory"
    )
    factory.return_value.get_model_info.return_value = MagicMock(dimensions=1024)
    factory.return_value.get_model.return_value = MagicMock()
    from aws_graphrag.models import Config

    ix = OpenSearchIndexer(config=Config())
    embedded: list[str] = []

    def _embed(texts):
        embedded.extend(texts)
        return [[float(len(t))] * 4 for t in texts]

    ix.embedding_model.embed_documents = _embed
    ix._embedded_log = embedded  # type: ignore[attr-defined]
    return ix


def test_intra_call_dedup(indexer) -> None:
    result = indexer._batch_embed(["a", "b", "a"])
    # "a" embedded once, fanned to both indices.
    assert indexer._embedded_log == ["a", "b"]
    assert result[0] == result[2]


def test_empty_text_skipped(indexer) -> None:
    result = indexer._batch_embed(["", "  ", "x"])
    assert result[0] is None and result[1] is None
    assert result[2] is not None
    assert indexer._embedded_log == ["x"]


def test_cross_call_cache_reuse(indexer) -> None:
    indexer._batch_embed(["a", "b"])
    indexer._embedded_log.clear()
    result = indexer._batch_embed(["a", "c"])
    # "a" served from cache; only "c" hits the model.
    assert indexer._embedded_log == ["c"]
    assert result[0] is not None


def test_cache_persists_in_dict(indexer) -> None:
    indexer._batch_embed(["alpha", "beta"])
    assert len(indexer._embedding_cache) == 2
