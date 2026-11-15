# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The chunking stage warns when its chunk sizes do not fit the corpus or the
model that embeds the chunks.

Chunk sizes may count characters (``processing.chunking.length_unit: chars``):
a Han, Hangul or Kana character is about one token, a Thai one about two
thirds, so such a chunk holds several times the tokens of an English one of
the same setting. And a chunk longer than the embedding model's input is
embedded only in part. Both are warnings: no default changes, and no
backend's defaults warn.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.domain.models import Backend, Config, Document, TextUnit
from unified_kg_rag.ports.model_factory import EmbeddingModelInfo

pytestmark = pytest.mark.unit

_KOREAN = "공급자는 매월 구매자의 창고로 부품을 납품하고 대금은 납품 후 지급한다. " * 20
_JAPANESE = "東邦商事は倉庫を借りて物流を委託し、代金は納品後に支払う。" * 20
_THAI = "ผู้ขายส่งชิ้นส่วนไปยังคลังสินค้าของผู้ซื้อทุกเดือนและชำระเงินหลังส่งมอบ " * 20
_RUSSIAN = "Поставщик ежемесячно отгружает детали на склад покупателя. " * 20
_ENGLISH = "The Vendor ships parts to the Buyer's warehouse every month. " * 20


def _stage(
    *,
    embedding: int | None = None,
    length_unit: str = "chars",
    chunker_type: str = "intelligent",
) -> ps.TextChunkingStage:
    config = Config(
        processing={
            "chunking": {"length_unit": length_unit, "chunker_type": chunker_type}
        }
    )
    providers = Providers(config)
    providers.embedding_model_info = MagicMock(  # type: ignore[method-assign]
        return_value=EmbeddingModelInfo(max_sequence_tokens=embedding)
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


@pytest.mark.parametrize("text", [_KOREAN, _JAPANESE, _THAI])
def test_a_corpus_of_costly_characters_with_character_sizes_is_warned_of(
    text: str, caplog: pytest.LogCaptureFixture
) -> None:
    # Keyed on the estimated tokens a character, not on a script class: Thai
    # is spaceless, not dense, and costs about 2.7 times Latin text.
    with caplog.at_level(logging.WARNING):
        _stage()._warn_of_chunk_sizes(_documents(text), [])
    (warning,) = _warnings(caplog)
    assert "length_unit: tokens" in warning
    assert "max_chunk_size" in warning


@pytest.mark.parametrize(
    ("text", "length_unit"),
    [(_ENGLISH, "chars"), (_RUSSIAN, "chars"), (_KOREAN, "tokens")],
)
def test_no_character_warning_for_cheap_text_or_token_sizes(
    text: str, length_unit: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        _stage(length_unit=length_unit)._warn_of_chunk_sizes(_documents(text), [])
    assert _warnings(caplog) == []


def test_the_warning_names_the_size_that_made_the_units(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The simple chunker cuts at fallback_chunk_size; max_chunk_size only
    # caps the intelligent chunker's chunks.
    with caplog.at_level(logging.WARNING):
        _stage(chunker_type="simple", embedding=100)._warn_of_chunk_sizes(
            _documents(_KOREAN), _units(_ENGLISH)
        )
    warnings = _warnings(caplog)
    assert len(warnings) == 2
    assert all("fallback_chunk_size" in w for w in warnings)
    assert not any("max_chunk_size" in w for w in warnings)


def test_chunks_over_the_embedding_input_are_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    short, long_ = "word " * 10, _ENGLISH  # about 10 and 300 estimated tokens
    with caplog.at_level(logging.WARNING):
        _stage(embedding=100, length_unit="tokens")._warn_of_chunk_sizes(
            _documents(_ENGLISH), _units(short, long_, long_)
        )
    (embedding,) = _warnings(caplog)
    assert embedding.startswith(
        "Chunk size: 2 of 3 text units hold more than the embedding model's "
        "100-token input"
    )
    max_size = Config().processing.chunking.max_chunk_size
    assert f"processing.chunking.max_chunk_size ({max_size} tokens)" in embedding


def test_chunks_within_the_limit_are_not_warned_of(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        _stage(embedding=8192)._warn_of_chunk_sizes(
            _documents(_ENGLISH), _units(_ENGLISH)
        )
    assert _warnings(caplog) == []


def test_an_unreadable_limit_skips_the_check(caplog: pytest.LogCaptureFixture) -> None:
    stage = _stage()
    stage.providers.embedding_model_info.side_effect = (  # type: ignore[attr-defined]
        RuntimeError("no catalog")
    )
    with caplog.at_level(logging.WARNING):
        stage._warn_of_chunk_sizes(_documents(_ENGLISH), _units(_ENGLISH))
    assert _warnings(caplog) == []


@pytest.mark.parametrize("backend", list(Backend))
@pytest.mark.parametrize("chunker_type", ["intelligent", "simple"])
@pytest.mark.parametrize("text", [_ENGLISH, _KOREAN])
def test_no_backend_warns_at_its_defaults(
    backend: Backend,
    chunker_type: str,
    text: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Units as large as the defaults make them (the cap, in tokens), read
    # against the preset's own embedding model's limit.
    config = Config(
        backend=backend, processing={"chunking": {"chunker_type": chunker_type}}
    )
    stage = ps.TextChunkingStage(config, providers=Providers(config))
    largest = config.processing.chunking.max_chunk_size
    unit = " ".join(["word"] * largest)  # one estimated token a word
    with caplog.at_level(logging.WARNING):
        stage._warn_of_chunk_sizes(_documents(text * 10), _units(unit))
    assert _warnings(caplog) == []


def test_bedrock_limits_come_from_the_catalog_without_a_client() -> None:
    providers = Providers(Config())
    info = providers.embedding_model_info()
    assert info is not None and info.max_sequence_tokens == 8192  # Titan V2
    assert providers._embedding_factory is None


def test_a_rerank_profile_or_arn_finds_its_models_record() -> None:
    from unified_kg_rag.adapters.aws.bedrock_models import get_rerank_model_info

    bare = get_rerank_model_info("cohere.rerank-v3-5:0")
    assert bare is not None
    assert get_rerank_model_info("us.cohere.rerank-v3-5:0") == bare
    arn = (
        "arn:aws:bedrock:us-west-2:111122223333:inference-profile/"
        "us.cohere.rerank-v3-5:0"
    )
    assert get_rerank_model_info(arn) == bare
