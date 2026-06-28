# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the content-hash embedding cache (perf/cost hardening)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer

pytestmark = pytest.mark.unit


@pytest.fixture
def indexer(mocker):
    mocker.patch("unified_kg_rag.adapters.storage.opensearch_indexer.OpenSearchClient")
    factory = mocker.patch(
        "unified_kg_rag.adapters.storage.opensearch_indexer.BedrockEmbeddingModelFactory"
    )
    factory.return_value.get_model_info.return_value = MagicMock(dimensions=1024)
    factory.return_value.get_model.return_value = MagicMock()
    from unified_kg_rag.domain.models import Config

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


def test_cache_hit_rate_tracked(indexer) -> None:
    indexer._batch_embed(["a", "b"])  # 2 misses
    assert indexer.embedding_cache_hit_rate == 0.0
    indexer._batch_embed(["a", "b"])  # 2 hits
    # 2 hits / 4 total lookups = 0.5
    assert indexer.embedding_cache_hit_rate == 0.5


def test_batch_embed_does_not_flush_s3_per_call(indexer) -> None:
    # The S3 cache must NOT be flushed inside _batch_embed (it is called once per
    # extractor per item-type; flushing there read-merge-overwrites the whole
    # growing S3 object many times per run). Flushing is deferred to
    # _flush_embedding_cache at the item-type boundary.
    s3 = MagicMock()
    indexer._s3_embedding_cache = s3
    indexer._batch_embed(["a", "b"])
    indexer._batch_embed(["c", "d"])
    s3.flush.assert_not_called()  # decoupled from per-batch embedding
    s3.load.assert_called()  # load stays (idempotent, guarded by _loaded)


def test_flush_embedding_cache_persists_once(indexer) -> None:
    s3 = MagicMock()
    indexer._s3_embedding_cache = s3
    indexer._flush_embedding_cache()
    s3.flush.assert_called_once()


def test_flush_embedding_cache_noop_without_s3(indexer) -> None:
    indexer._s3_embedding_cache = None
    # Should not raise when the S3 tier is disabled.
    indexer._flush_embedding_cache()
