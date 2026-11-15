# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The chunking stage warns when its chunk sizes do not fit the corpus or the
models that read the chunks.

Chunk sizes may count characters (``processing.chunking.length_unit: chars``):
a dense-script character (Han, Hangul, Kana) is about one token, so a
Korean or Japanese chunk holds some four times the tokens of an English one
of the same setting. And a chunk longer than the embedding model's input or
the reranker's document limit is read only in part. Both are warnings: no
default changes.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.domain.models import Config, Document, TextUnit
from unified_kg_rag.ports.model_factory import EmbeddingModelInfo, RerankModelInfo

pytestmark = pytest.mark.unit

_KOREAN = "공급자는 매월 구매자의 창고로 부품을 납품하고 대금은 납품 후 지급한다. " * 20
_JAPANESE = "東邦商事は倉庫を借りて物流を委託し、代金は納品後に支払う。" * 20
_ENGLISH = "The Vendor ships parts to the Buyer's warehouse every month. " * 20


def _stage(
    *,
    embedding: int | None = None,
    rerank: int | None = None,
    length_unit: str = "chars",
) -> ps.TextChunkingStage:
    config = Config(processing={"chunking": {"length_unit": length_unit}})
    providers = Providers(config)
    providers.embedding_model_info = MagicMock(  # type: ignore[method-assign]
        return_value=EmbeddingModelInfo(max_sequence_tokens=embedding)
    )
    providers.rerank_model_info = MagicMock(  # type: ignore[method-assign]
        return_value=(
            None if rerank is None else RerankModelInfo(max_document_tokens=rerank)
        )
    )
    return ps.TextChunkingStage(config, providers=providers)


def _documents(text: str) -> list[Document]:
    return [
        Document(
            page_content=text, document_id="d", file_name="d.txt", file_path="d.txt"
        )
    ]


def _units(*texts: str) -> list[TextUnit]:
    return [
        TextUnit(id=f"u{i}", text=text, document_ids=["d"])
        for i, text in enumerate(texts)
    ]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.getMessage().startswith("Chunk size")
    ]


@pytest.mark.parametrize("text", [_KOREAN, _JAPANESE])
def test_a_dense_script_corpus_with_character_sizes_is_warned_of(
    text: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        _stage()._warn_of_chunk_sizes(_documents(text), [])
    (warning,) = _warnings(caplog)
    assert "length_unit: tokens" in warning


@pytest.mark.parametrize(
    ("text", "length_unit"),
    [(_ENGLISH, "chars"), (_KOREAN, "tokens")],
)
def test_no_dense_warning_for_latin_text_or_token_sizes(
    text: str, length_unit: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        _stage(length_unit=length_unit)._warn_of_chunk_sizes(_documents(text), [])
    assert _warnings(caplog) == []


def test_chunks_over_a_model_limit_are_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    short, long_ = "word " * 10, _ENGLISH  # about 10 and 300 estimated tokens
    with caplog.at_level(logging.WARNING):
        _stage(embedding=100, rerank=200)._warn_of_chunk_sizes(
            _documents(_ENGLISH), _units(short, long_, long_)
        )
    embedding, rerank = _warnings(caplog)
    assert embedding.startswith(
        "Chunk size: 2 of 3 text units hold more than the embedding model's "
        "100-token input"
    )
    assert "reranker's 200-token input" in rerank
    assert (
        f"max_chunk_size ({Config().processing.chunking.max_chunk_size} chars)"
        in rerank
    )


def test_chunks_within_the_limits_are_not_warned_of(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        _stage(embedding=8192, rerank=None)._warn_of_chunk_sizes(
            _documents(_ENGLISH), _units(_ENGLISH)
        )
    assert _warnings(caplog) == []


def test_unreadable_limits_skip_the_check(caplog: pytest.LogCaptureFixture) -> None:
    stage = _stage()
    stage.providers.embedding_model_info.side_effect = (  # type: ignore[attr-defined]
        RuntimeError("no catalog")
    )
    with caplog.at_level(logging.WARNING):
        stage._warn_of_chunk_sizes(_documents(_ENGLISH), _units(_ENGLISH))
    assert _warnings(caplog) == []


def test_bedrock_limits_come_from_the_catalog_without_a_client() -> None:
    providers = Providers(Config())
    info = providers.embedding_model_info()
    assert info is not None and info.max_sequence_tokens == 8192  # Titan V2
    rerank = providers.rerank_model_info()
    assert rerank is not None and rerank.max_document_tokens == 4096  # Cohere 3.5
    assert providers._embedding_factory is None
    assert providers._rerank_factory is None


def test_no_reranker_has_no_limit() -> None:
    providers = Providers(Config(providers={"rerank": {"kind": "none"}}))
    assert providers.rerank_model_info() is None
