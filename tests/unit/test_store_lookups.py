# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run reads each alias and index mapping once (``shared.store_lookups``).

The vector indexers of a run share the cache; an alias swap, an index
deletion or a mapping update through any of them drops what it changes.
Outside a run nothing is cached.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import Config, TextUnit
from unified_kg_rag.shared.store_lookups import (
    StoreLookupCache,
    install_lookup_cache,
    lookup_cache,
    remove_lookup_cache,
)

pytestmark = pytest.mark.unit


class _CountingEngine(SQLiteSearchEngine):
    lookups = 0
    mapping_reads = 0

    def get_index_name_by_alias(self, alias_name: str) -> str | None:
        type(self).lookups += 1
        return super().get_index_name_by_alias(alias_name)

    def get_index_mappings(self, index_name: str) -> dict:
        type(self).mapping_reads += 1
        return super().get_index_mappings(index_name)


@pytest.fixture
def engine_class() -> Iterator[type[_CountingEngine]]:
    class Engine(_CountingEngine):
        pass

    yield Engine


@pytest.fixture
def run_cache() -> Iterator[StoreLookupCache]:
    token = install_lookup_cache()
    cache = lookup_cache()
    assert cache is not None
    try:
        yield cache
    finally:
        remove_lookup_cache(token)


def _indexer(path: Path, engine_class: type[SQLiteSearchEngine]) -> OpenSearchIndexer:
    return OpenSearchIndexer(
        Config(),
        embedding_factory=HashingEmbeddingFactory(dimensions=8),
        opensearch_client=engine_class(path),
    )


def test_a_run_reads_an_alias_once(
    tmp_path: Path, engine_class: type[_CountingEngine], run_cache
) -> None:
    indexer = _indexer(tmp_path / "v.sqlite3", engine_class)
    indexer.index_text_units([TextUnit(id="t0", text="a unit")])
    engine_class.lookups = 0

    for _ in range(3):
        indexer.read_text_unit_ids(["t0"])
        indexer.upsert_text_units([TextUnit(id="t0", text="a unit")])

    assert engine_class.lookups == 1


def test_without_a_run_every_lookup_reads(
    tmp_path: Path, engine_class: type[_CountingEngine]
) -> None:
    assert lookup_cache() is None
    indexer = _indexer(tmp_path / "v.sqlite3", engine_class)
    indexer.index_text_units([TextUnit(id="t0", text="a unit")])
    engine_class.lookups = 0

    indexer.read_text_unit_ids(["t0"])
    indexer.read_text_unit_ids(["t0"])

    assert engine_class.lookups == 2


def test_a_swap_by_another_indexer_of_the_store_is_seen(
    tmp_path: Path, engine_class: type[_CountingEngine], run_cache
) -> None:
    path = tmp_path / "v.sqlite3"
    reader, writer = _indexer(path, engine_class), _indexer(path, engine_class)
    reader.index_text_units([TextUnit(id="t0", text="first build")])
    assert reader.read_text_unit_ids(["t0", "t1"]) == ["t0"]

    # A rebuild through another client of the same file swaps the alias.
    writer.index_text_units([TextUnit(id="t1", text="second build")])

    assert reader.read_text_unit_ids(["t0", "t1"]) == ["t1"]


def test_a_mapping_update_is_read_again(
    tmp_path: Path, engine_class: type[_CountingEngine], run_cache
) -> None:
    indexer = _indexer(tmp_path / "v.sqlite3", engine_class)
    indexer.index_text_units([TextUnit(id="t0", text="a unit")])
    engine_class.mapping_reads = 0

    indexer.check_index_settings()
    indexer.check_index_settings()
    reads = engine_class.mapping_reads
    index = indexer._live_index(
        indexer._get_name(indexer.opensearch_config.text_units_index_prefix, None)
    )
    assert index is not None
    indexer._put_index_meta(index, {"note": 1})

    assert indexer._index_mappings(index)["_meta"] == {"note": 1}
    assert engine_class.mapping_reads == reads + 1


def test_a_lookup_racing_a_forget_is_not_kept() -> None:
    cache = StoreLookupCache()

    def read() -> str:
        # The swap lands while the lookup is in flight.
        cache.forget_aliases("store")
        return "old-index"

    assert cache.alias("store", "alias", read) == "old-index"
    assert cache.alias("store", "alias", lambda: "new-index") == "new-index"


def test_stores_do_not_share_entries() -> None:
    cache = StoreLookupCache()
    assert cache.alias("a", "alias", lambda: "index-a") == "index-a"
    assert cache.alias("b", "alias", lambda: "index-b") == "index-b"
    cache.forget_aliases("b")
    assert cache.alias("a", "alias", lambda: "other") == "index-a"
