# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the content-hash embedding cache (perf/cost hardening)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer

pytestmark = pytest.mark.unit


@pytest.fixture
def indexer(mocker):
    mocker.patch("unified_kg_rag.adapters.storage.opensearch_indexer.OpenSearchClient")
    factory = mocker.patch(
        "unified_kg_rag.adapters.aws.bedrock.BedrockEmbeddingModelFactory"
    )
    factory.return_value.get_model_info.return_value = MagicMock(dimensions=1024)
    factory.return_value.get_model.return_value = MagicMock()
    from unified_kg_rag.domain.models import Config

    ix = OpenSearchIndexer(config=Config(), embedding_factory=factory.return_value)
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


def test_without_the_s3_cache_no_vector_is_kept_across_calls(indexer) -> None:
    # An in-process tier held every vector of the run for no measured hit:
    # texts repeat within a call (deduplicated above), rarely across calls.
    assert not hasattr(indexer, "_embedding_cache")
    indexer._batch_embed(["a", "b"])
    indexer._embedded_log.clear()
    assert indexer._batch_embed(["a"]) == [[1.0] * 4]
    assert indexer._embedded_log == ["a"]


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


def test_final_embedding_failures_summarized_at_warning(indexer, caplog) -> None:
    # Batch call fails, then the per-item fallback fails only for "bad": the
    # failure is reported once as a WARNING summary, not as per-item ERRORs.
    def _embed(texts):
        if len(texts) > 1 or texts[0] == "bad":
            raise RuntimeError("synthetic model error")
        return [[1.0] * 4]

    indexer.embedding_model.embed_documents = _embed
    with caplog.at_level("DEBUG"):
        result = indexer._batch_embed(["good", "bad"])
    assert result[0] is not None and result[1] is None
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
    summaries = [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and "unique texts after retries" in r.getMessage()
    ]
    assert len(summaries) == 1
    assert "1 of 2" in summaries[0].getMessage()


def test_s3_cache_serves_a_later_call(indexer) -> None:
    from unified_kg_rag.adapters.aws.embedding_cache import S3EmbeddingCache

    s3 = S3EmbeddingCache("bucket", "key", "titan", 4)
    s3._loaded = True  # nothing to load; no S3 call
    indexer._s3_embedding_cache = s3
    indexer._batch_embed(["alpha", "beta"])
    indexer._embedded_log.clear()

    assert indexer._batch_embed(["alpha"]) == [[5.0] * 4]
    assert indexer._embedded_log == []


def test_embedding_requests_run_sixteen_at_once(indexer) -> None:
    # 8 requests in flight left Bedrock headroom unused: 16 halved the
    # embedding time in the real-AWS E2E with no throttling.
    lock = threading.Lock()
    in_flight = peak = 0
    full = threading.Event()

    def _embed(texts):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
            if in_flight >= 16:
                full.set()
        full.wait(timeout=2)
        with lock:
            in_flight -= 1
        return [[1.0] * 4 for _ in texts]

    indexer.embedding_model.embed_documents = _embed
    result = indexer._batch_embed([f"text {i}" for i in range(50 * 32)])
    assert all(r is not None for r in result)
    assert peak == 16
