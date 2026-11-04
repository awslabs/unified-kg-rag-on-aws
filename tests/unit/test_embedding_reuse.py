# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A write embeds only the texts it changed (AWS-free, embedded engine).

Each vector field stores the digest of the text it embeds, with the
embedding settings hashed in. An upsert of touched ids, and a rebuild of the
live index, read the stored vectors and digests of their items (one read per
write slice) and embed only the texts whose digest differs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.embeddings import HashingEmbeddings
from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
from unified_kg_rag.adapters.storage import opensearch_indexer as module
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import Config, Entity, TextUnit

pytestmark = pytest.mark.unit

DIM = 8


class _Counting(HashingEmbeddings):
    """Hashing embeddings that record every text they embed."""

    def __init__(self, dimensions: int) -> None:
        super().__init__(dimensions)
        self.texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return super().embed_documents(texts)


class _Factory:
    def __init__(self, dimensions: int = DIM) -> None:
        self.dimensions = dimensions
        self.model = _Counting(dimensions)

    def get_model(self, model_id: Any, **kwargs: Any) -> _Counting:
        return self.model

    def get_model_info(self, model_id: Any) -> Any:
        return type("Info", (), {"dimensions": self.dimensions})()


def _config(model: str | None = None, dimension: int = DIM) -> Config:
    config = Config(
        providers={"embedding": {"kind": "ollama"}},
        models={"embedding": {"model_id": model or "embed-a", "dimension": dimension}},
    )
    return config


def _indexer(
    path: Path, config: Config | None = None, dimension: int = DIM
) -> tuple[OpenSearchIndexer, _Counting]:
    factory = _Factory(dimension)
    indexer = OpenSearchIndexer(
        config or _config(dimension=dimension),
        embedding_factory=factory,
        opensearch_client=SQLiteSearchEngine(path),
    )
    return indexer, factory.model


def _entities(**descriptions: str) -> list[Entity]:
    return [
        Entity(id=name, name=name, description=description)
        for name, description in descriptions.items()
    ]


def _stored(indexer: OpenSearchIndexer, prefix: str, ids: list[str]) -> dict:
    return {
        source["id"]: source
        for source in indexer._read_sources(
            prefix, ids, None, fields=["description_embedding", "text_embedding"]
        )
    }


def test_upsert_embeds_only_the_texts_that_changed(tmp_path: Path) -> None:
    indexer, model = _indexer(tmp_path / "v.sqlite3")
    indexer.index_entities(_entities(Vendor="sells", Buyer="buys", Depot="stores"))
    assert len(model.texts) == 6  # name + description each
    model.texts.clear()

    stats = indexer.upsert_entities(
        _entities(Vendor="sells", Buyer="buys gadgets", Depot="stores")
    )

    assert stats.successful_items == 3
    assert model.texts == ["buys gadgets"]
    # The reused vectors are the stored ones; the new one embeds the new text.
    prefix = indexer.opensearch_config.entities_index_prefix
    stored = _stored(indexer, prefix, ["Buyer", "Vendor"])
    assert stored["Buyer"]["description_embedding"] == pytest.approx(
        HashingEmbeddings(DIM).embed_query("buys gadgets")
    )
    assert stored["Vendor"]["description_embedding"] == pytest.approx(
        HashingEmbeddings(DIM).embed_query("sells")
    )


def test_a_rebuild_reuses_the_vectors_of_the_index_it_replaces(
    tmp_path: Path,
) -> None:
    indexer, model = _indexer(tmp_path / "v.sqlite3")
    units = [TextUnit(id=f"t{i}", text=f"text number {i}") for i in range(5)]
    indexer.index_text_units(units)
    model.texts.clear()

    indexer.index_text_units([*units, TextUnit(id="t9", text="a new unit")])

    assert model.texts == ["a new unit"]
    prefix = indexer.opensearch_config.text_units_index_prefix
    assert set(_stored(indexer, prefix, [f"t{i}" for i in (0, 9)])) == {"t0", "t9"}


@pytest.mark.parametrize(
    "other",
    [
        pytest.param(lambda: _config(model="embed-b"), id="model"),
        pytest.param(
            lambda: Config(
                providers={"embedding": {"kind": "openai_compatible"}},
                models={"embedding": {"model_id": "embed-a", "dimension": DIM}},
            ),
            id="provider",
        ),
        pytest.param(
            lambda: Config(
                providers={"embedding": {"kind": "ollama", "max_input_tokens": 99}},
                models={"embedding": {"model_id": "embed-a", "dimension": DIM}},
            ),
            id="input-cap",
        ),
    ],
)
def test_other_embedding_settings_never_reuse_a_vector(
    tmp_path: Path, other: Any
) -> None:
    path = tmp_path / "v.sqlite3"
    first, _ = _indexer(path)
    first.index_text_units([TextUnit(id="t0", text="same text")])

    second, model = _indexer(path, other())
    second.index_text_units([TextUnit(id="t0", text="same text")])

    assert first._embedding_digest("same text") != second._embedding_digest("same text")
    assert model.texts == ["same text"]


def test_another_dimension_never_reuses_a_vector(tmp_path: Path) -> None:
    path = tmp_path / "v.sqlite3"
    first, _ = _indexer(path)
    first.index_text_units([TextUnit(id="t0", text="same text")])

    second, model = _indexer(path, _config(dimension=4), dimension=4)
    second.index_text_units([TextUnit(id="t0", text="same text")])

    assert model.texts == ["same text"]


def test_reuse_off_embeds_every_text(tmp_path: Path) -> None:
    indexer, model = _indexer(tmp_path / "v.sqlite3")
    indexer.index_entities(_entities(Vendor="sells"))
    model.texts.clear()

    indexer.set_embedding_reuse(False)
    indexer.upsert_entities(_entities(Vendor="sells"))

    assert sorted(model.texts) == ["Vendor", "sells"]


def test_an_index_without_digests_is_embedded_once_after_its_upgrade(
    tmp_path: Path,
) -> None:
    # An index an earlier release built: no digest field, documents without
    # digests. The incremental check maps the field; the next upsert embeds
    # its texts once, the one after reuses them.
    path = tmp_path / "v.sqlite3"
    indexer, model = _indexer(path)
    engine = indexer.opensearch_client
    prefix = indexer.opensearch_config.entities_index_prefix
    alias = indexer._get_name(prefix, None)
    mapping = indexer._get_entities_mapping()
    del mapping["mappings"]["properties"][module._DIGESTS_FIELD]
    engine.create_index(f"{alias}-20260101000000000", mapping)
    engine.update_alias(alias, f"{alias}-20260101000000000")
    vector = HashingEmbeddings(DIM).embed_query("sells")
    engine.bulk_index(
        f"{alias}-20260101000000000",
        [
            {
                "id": "Vendor",
                "name": "Vendor",
                "description": "sells",
                "name_embedding": vector,
                "description_embedding": vector,
            }
        ],
    )

    indexer.check_index_settings()
    assert module._DIGESTS_FIELD in (
        engine.get_index_mappings(f"{alias}-20260101000000000")["properties"]
    )
    indexer.upsert_entities(_entities(Vendor="sells"))
    assert sorted(model.texts) == ["Vendor", "sells"]
    model.texts.clear()

    indexer.upsert_entities(_entities(Vendor="sells"))
    assert model.texts == []


def test_a_rewrite_reuses_the_vector_an_earlier_write_replaced(
    tmp_path: Path,
) -> None:
    # A removal rewrites an item the commit after it writes back.
    indexer, model = _indexer(tmp_path / "v.sqlite3")
    indexer.index_entities(_entities(Vendor="sells widgets"))
    indexer.upsert_entities(_entities(Vendor="sells"))
    model.texts.clear()

    indexer.upsert_entities(_entities(Vendor="sells widgets"))

    assert model.texts == []


def _counting_reads(
    indexer: OpenSearchIndexer, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[int, tuple[str, ...]]]:
    """The (ids, fields) of each ``_read_sources`` call from now on."""
    reads: list[tuple[int, tuple[str, ...]]] = []
    read = indexer._read_sources

    def counting(prefix: str, ids: list[str], suffix: Any, fields: Any = None):
        reads.append((len(ids), tuple(fields or ())))
        return read(prefix, ids, suffix, fields=fields)

    monkeypatch.setattr(indexer, "_read_sources", counting)
    return reads


def test_one_read_of_stored_digests_per_write_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer, model = _indexer(tmp_path / "v.sqlite3")
    count = module._WRITE_SLICE_SIZE + 5
    units = [TextUnit(id=f"t{i:05d}", text=f"unit {i}") for i in range(count)]
    indexer.index_text_units(units)
    model.texts.clear()
    reads = _counting_reads(indexer, monkeypatch)

    indexer.upsert_text_units(units)

    assert model.texts == []
    assert module._WRITE_SLICE_SIZE <= module._READ_BATCH_SIZE
    digests = (module._DIGESTS_FIELD,)
    assert [n for n, fields in reads if fields == digests] == [
        module._WRITE_SLICE_SIZE,
        5,
    ]
    vectors = [n for n, fields in reads if fields != digests]
    assert sum(vectors) == count
    assert max(vectors) == module._VECTOR_READ_BATCH_SIZE


def test_a_rebuild_reads_only_the_vectors_it_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Read whole, the stored vectors of a slice whose texts changed took
    # gigabytes of responses and floats for vectors no write took.
    indexer, model = _indexer(tmp_path / "v.sqlite3")
    indexer.index_entities(_entities(Vendor="sells", Buyer="buys", Depot="stores"))
    model.texts.clear()
    reads = _counting_reads(indexer, monkeypatch)

    indexer.index_entities(_entities(Vendor="sells", Buyer="buys gadgets", Depot="x"))

    assert sorted(model.texts) == ["buys gadgets", "x"]
    digests = (module._DIGESTS_FIELD,)
    assert sorted(fields for _, fields in reads if fields != digests) == sorted(
        [
            ("name_embedding", digests[0]),
            ("name_embedding", "description_embedding", digests[0]),
        ]
    )
    assert indexer._replaced == {}
