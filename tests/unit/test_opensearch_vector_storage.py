# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The opt-in vector storage of the OpenSearch indices (AWS-free).

``indexing.opensearch.vector_mode`` / ``compression_level`` add the
``mode`` / ``compression_level`` keys of OpenSearch 2.17 to every knn_vector
mapping. The default sends neither (OpenSearch before 2.17 refuses the keys).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
from unified_kg_rag.adapters.storage import opensearch_indexer
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import ConfigurationError
from unified_kg_rag.shared.exceptions import StoreRequestRejectedError

pytestmark = pytest.mark.unit

# The knn_vector field every vector index mapped before the option existed,
# serialized: the default must stay byte-identical.
_DEFAULT_KNN_FIELD = (
    '{"type": "knn_vector", "dimension": 8, "method": {"name": "hnsw", '
    '"space_type": "cosinesimil", "engine": "lucene", "parameters": '
    '{"ef_construction": 128, "m": 24}}}'
)
# How OpenSearch 2.13 (the CDK domain) refuses the keys.
_REFUSED_ON_2_13 = (
    "mapper_parsing_exception: Failed to parse mapping [_doc]: unknown "
    "parameter [mode] on mapper [text_embedding] of type [knn_vector]"
)


class _RemoteEngine(SQLiteSearchEngine):
    """The embedded engine standing in for an OpenSearch cluster: not in
    process, so the indexer sends it the storage keys (it keeps them in the
    mapping it returns, as OpenSearch does)."""

    in_process: ClassVar[bool] = False


def _config(**storage: Any) -> Config:
    config = Config()
    opensearch = config.indexing.opensearch
    if "engine" in storage:
        opensearch.vector_search = {
            **opensearch.vector_search,
            "engine": storage.pop("engine"),
        }
    for key in ("compression_level", "vector_mode"):
        if key in storage:
            setattr(opensearch, key, storage.pop(key))
    assert not storage
    return config


def _indexer(config: Config, engine: SQLiteSearchEngine) -> OpenSearchIndexer:
    return OpenSearchIndexer(
        config,
        embedding_factory=HashingEmbeddingFactory(dimensions=8),
        opensearch_client=engine,
    )


def _knn_fields(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        field
        for field in mapping["mappings"]["properties"].values()
        if field.get("type") == "knn_vector"
    ]


class TestMapping:
    def test_the_default_mapping_is_unchanged(self, tmp_path: Path) -> None:
        indexer = _indexer(Config(), _RemoteEngine(tmp_path / "v.sqlite3"))
        fields = [f for spec in indexer._specs() for f in _knn_fields(spec.mapping())]
        assert len(fields) == 8
        assert {json.dumps(field) for field in fields} == {_DEFAULT_KNN_FIELD}

    @pytest.mark.parametrize(
        ("storage", "keys"),
        [
            ({"engine": "faiss", "vector_mode": "on_disk"}, {"mode": "on_disk"}),
            (
                {
                    "engine": "faiss",
                    "vector_mode": "on_disk",
                    "compression_level": "16x",
                },
                {"mode": "on_disk", "compression_level": "16x"},
            ),
            (
                {"engine": "faiss", "compression_level": "2x"},
                {"compression_level": "2x"},
            ),
            (
                {"vector_mode": "on_disk", "compression_level": "4x"},
                {"mode": "on_disk", "compression_level": "4x"},
            ),
            ({"compression_level": "4x"}, {"compression_level": "4x"}),
            # in_memory and 1x are OpenSearch's defaults: only what is set.
            ({"vector_mode": "in_memory"}, {}),
            ({"compression_level": "1x"}, {"compression_level": "1x"}),
        ],
    )
    def test_every_vector_field_gets_the_storage_keys(
        self, tmp_path: Path, storage: dict[str, Any], keys: dict[str, str]
    ) -> None:
        config = _config(**storage)
        indexer = _indexer(config, _RemoteEngine(tmp_path / "v.sqlite3"))
        for spec in indexer._specs():
            for field in _knn_fields(spec.mapping()):
                assert {
                    k: field[k] for k in ("mode", "compression_level") if k in field
                } == keys
                # The method is the configured one either way.
                assert field["method"]["engine"] == (
                    config.indexing.opensearch.vector_search["engine"]
                )

    def test_the_embedded_engine_ignores_them_and_warns_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch
    ) -> None:
        monkeypatch.setattr(opensearch_indexer, "_storage_ignored_warned", False)
        config = _config(engine="faiss", vector_mode="on_disk")
        indexer = _indexer(config, SQLiteSearchEngine(tmp_path / "v.sqlite3"))
        with caplog.at_level(logging.WARNING):
            fields = [
                json.dumps(field)
                for spec in indexer._specs()
                for field in _knn_fields(spec.mapping())
            ]
            indexer._get_entities_mapping()
        assert all('"mode"' not in field for field in fields)
        warnings = [r for r in caplog.records if "embedded search engine" in r.message]
        assert len(warnings) == 1


class TestConfig:
    @pytest.mark.parametrize(
        ("storage", "match"),
        [
            ({"engine": "faiss", "compression_level": "4x"}, "4x needs the lucene"),
            ({"compression_level": "2x"}, "2x needs the faiss"),
            ({"vector_mode": "on_disk"}, "lucene engine needs compression_level"),
            (
                {
                    "engine": "faiss",
                    "compression_level": "1x",
                    "vector_mode": "on_disk",
                },
                "1x is refused with vector_mode on_disk",
            ),
            ({"engine": "nmslib", "compression_level": "1x"}, "lucene or faiss"),
        ],
    )
    def test_combinations_no_version_takes_are_refused(
        self, storage: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValidationError, match=match):
            _config(**storage)

    def test_the_space_type_and_method_are_checked(self) -> None:
        config = Config()
        opensearch = config.indexing.opensearch
        opensearch.compression_level = "4x"
        for key, value, match in [
            ("space_type", "l1", "space type"),
            ("name", "ivf", "hnsw method"),
        ]:
            with pytest.raises(ValidationError, match=match):
                opensearch.vector_search = {**opensearch.vector_search, key: value}

    def test_the_default_sends_nothing(self) -> None:
        assert Config().indexing.opensearch.vector_storage() == {}


class TestPreflight:
    def test_a_cluster_before_2_17_is_refused_with_the_version(
        self, tmp_path: Path
    ) -> None:
        engine = _RemoteEngine(tmp_path / "v.sqlite3")
        create = engine.create_index

        def as_2_13(index: str, body: dict[str, Any], *, exist_ok: bool = True):
            if any("mode" in field for field in _knn_fields(body)):
                raise StoreRequestRejectedError(
                    _REFUSED_ON_2_13, reason="mapper_parsing_exception"
                )
            return create(index, body, exist_ok=exist_ok)

        engine.create_index = as_2_13  # type: ignore[method-assign]
        indexer = _indexer(_config(engine="faiss", vector_mode="on_disk"), engine)
        with pytest.raises(ConfigurationError, match="OpenSearch 2.17 or later"):
            indexer.check_index_creation("x")
        assert not engine.get_aliases_by_index("*settings-probe*")

        # The default creates its probes on the same cluster.
        _indexer(Config(), engine).check_index_creation("x")

    def test_another_refusal_names_the_storage_settings(self) -> None:
        error = opensearch_indexer._creation_refused(
            StoreRequestRejectedError(
                'Validation Failed: 1: "lucene" does not support "32x" compression;',
                reason="mapper_parsing_exception",
            )
        )
        assert "2.17" not in str(error)
        assert "compression_level" in str(error)


class TestLiveIndex:
    def _built(self, tmp_path: Path, config: Config) -> _RemoteEngine:
        engine = _RemoteEngine(tmp_path / "v.sqlite3")
        indexer = _indexer(config, engine)
        spec = indexer._specs()[0]
        alias = indexer._get_name(spec.alias_prefix, "x")
        engine.create_index(f"{alias}-1", spec.mapping())
        engine.update_alias(alias, f"{alias}-1")
        return engine

    @pytest.mark.parametrize(
        ("built", "run"),
        [
            ({}, {"engine": "faiss", "vector_mode": "on_disk"}),
            ({"engine": "faiss", "vector_mode": "on_disk"}, {"engine": "faiss"}),
            (
                {"engine": "faiss", "vector_mode": "on_disk"},
                {
                    "engine": "faiss",
                    "vector_mode": "on_disk",
                    "compression_level": "8x",
                },
            ),
            ({"compression_level": "4x"}, {}),
        ],
    )
    def test_an_incremental_run_refuses_another_storage(
        self, tmp_path: Path, built: dict[str, Any], run: dict[str, Any]
    ) -> None:
        engine = self._built(tmp_path, _config(**built))
        with pytest.raises(ConfigurationError, match="indexing.reset: true"):
            _indexer(_config(**run), engine).check_index_settings("x")

    @pytest.mark.parametrize(
        ("built", "run"),
        [
            ({}, {}),
            ({}, {"compression_level": "1x"}),
            ({}, {"vector_mode": "in_memory"}),
            # The on_disk default spelled out.
            (
                {"engine": "faiss", "vector_mode": "on_disk"},
                {
                    "engine": "faiss",
                    "vector_mode": "on_disk",
                    "compression_level": "32x",
                },
            ),
        ],
    )
    def test_the_same_storage_passes(
        self, tmp_path: Path, built: dict[str, Any], run: dict[str, Any]
    ) -> None:
        engine = self._built(tmp_path, _config(**built))
        _indexer(_config(**run), engine).check_index_settings("x")
