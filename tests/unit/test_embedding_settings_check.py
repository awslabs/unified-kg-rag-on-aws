# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An incremental run must not write vectors of another embedding model.

The unchanged documents of an incremental run are not re-embedded, so after a
change of the embedding model (or dimension) their old vectors would sit next
to the new ones in the live indices. The OpenSearch indexer records the
embedding settings in each index's ``_meta`` and
``check_embedding_settings`` refuses a mismatch. AWS-free: the indexer runs
against an in-memory stand-in for the OpenSearch client.
"""

from __future__ import annotations

import copy
import fnmatch
from typing import Any

import pytest

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from unified_kg_rag.adapters.storage import opensearch_indexer as module
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import Config, EmbeddingModelId, TextUnit
from unified_kg_rag.shared import ConfigurationError

pytestmark = pytest.mark.unit

_TITAN = EmbeddingModelId.TITAN_EMBED_V2.value
_COHERE = EmbeddingModelId.EMBED_MULTILINGUAL_V3.value


class _FakeOpenSearchClient:
    """Indices with their mappings, aliases and the documents written."""

    def __init__(self) -> None:
        self.mappings: dict[str, dict[str, Any]] = {}
        self.aliases: dict[str, str] = {}
        self.documents: dict[str, list[dict[str, Any]]] = {}

    def create_index(
        self, index: str, body: dict[str, Any], *, exist_ok: bool = True
    ) -> None:
        self.mappings.setdefault(index, copy.deepcopy(body["mappings"]))

    def get_index_mappings(self, index_name: str) -> dict[str, Any]:
        return copy.deepcopy(self.mappings[index_name])

    def update_alias(
        self, alias_name: str, new_index_name: str, remove_pattern: str | None = None
    ) -> None:
        self.aliases[alias_name] = new_index_name

    def get_index_name_by_alias(self, alias_name: str) -> str | None:
        return self.aliases.get(alias_name)

    def get_indices_by_alias(self, alias_name: str) -> list[str]:
        index = self.aliases.get(alias_name)
        return [index] if index else []

    def get_aliases_by_index(self, index_pattern: str) -> dict[str, list[str]]:
        return {
            name: sorted(a for a, i in self.aliases.items() if i == name)
            for name in self.mappings
            if fnmatch.fnmatch(name, index_pattern)
        }

    def bulk_index(
        self, index: str, documents: list[dict[str, Any]], refresh: bool = False
    ) -> dict[str, Any]:
        self.documents.setdefault(index, []).extend(documents)
        return {"errors": False, "items": []}

    def close(self) -> None:
        return None


def _indexer(
    mocker, client: _FakeOpenSearchClient, model_id: str, dimension: int
) -> OpenSearchIndexer:
    mocker.patch.object(module, "OpenSearchClient", return_value=client)
    config = Config(indexing={"opensearch": {"embedding_model_id": model_id}})
    return OpenSearchIndexer(
        config, embedding_factory=HashingEmbeddingFactory(dimensions=dimension)
    )


def _unit(unit_id: str = "t-1", suffix: str | None = None) -> TextUnit:
    return TextUnit(
        id=unit_id,
        text="Vendor ships to Buyer.",
        document_ids=["doc-1"],
        attributes={"index": suffix} if suffix else None,
    )


def test_index_records_the_embedding_model_and_dimension(mocker) -> None:
    client = _FakeOpenSearchClient()
    _indexer(mocker, client, _TITAN, 8).upsert_text_units([_unit()])

    (mappings,) = client.mappings.values()
    assert mappings["_meta"] == {
        "embedding_model_id": _TITAN,
        "embedding_dimension": 8,
    }


def test_same_embedding_settings_pass(mocker) -> None:
    client = _FakeOpenSearchClient()
    _indexer(mocker, client, _TITAN, 8).upsert_text_units([_unit()])

    _indexer(mocker, client, _TITAN, 8).check_embedding_settings("default")


@pytest.mark.parametrize(
    ("model_id", "dimension", "message"),
    [
        (_COHERE, 8, f"model '{_TITAN}' \\(dimension 8\\), but this run embeds "),
        (_TITAN, 16, "dimension 8\\), but .*\\(dimension 16\\)"),
    ],
)
def test_other_embedding_settings_fail_with_the_reset_remedy(
    mocker, model_id: str, dimension: int, message: str
) -> None:
    client = _FakeOpenSearchClient()
    _indexer(mocker, client, _TITAN, 8).upsert_text_units([_unit()])

    changed = _indexer(mocker, client, model_id, dimension)
    with pytest.raises(ConfigurationError, match=message) as raised:
        changed.check_embedding_settings("default")
    assert "indexing.reset: true" in str(raised.value)


def test_index_without_recorded_settings_is_checked_by_dimension(mocker) -> None:
    # Indices built before the settings were recorded have no _meta: the
    # model cannot be told, the dimension still can.
    client = _FakeOpenSearchClient()
    _indexer(mocker, client, _TITAN, 8).upsert_text_units([_unit()])
    for mappings in client.mappings.values():
        del mappings["_meta"]

    _indexer(mocker, client, _COHERE, 8).check_embedding_settings("default")
    with pytest.raises(ConfigurationError, match="an unrecorded model"):
        _indexer(mocker, client, _COHERE, 16).check_embedding_settings("default")


def test_check_is_scoped_to_the_suffix(mocker) -> None:
    client = _FakeOpenSearchClient()
    _indexer(mocker, client, _TITAN, 8).upsert_text_units([_unit(suffix="old")])

    changed = _indexer(mocker, client, _COHERE, 8)
    changed.check_embedding_settings("new")  # nothing indexed there yet
    with pytest.raises(ConfigurationError):
        changed.check_embedding_settings("old")
