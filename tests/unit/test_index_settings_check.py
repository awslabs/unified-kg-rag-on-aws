# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An incremental run must not write into indices built for other settings.

The unchanged documents of an incremental run are not rewritten, so after a
change of the embedding model (or dimension) their old vectors would sit next
to the new ones in the live indices; after a change of the languages the live
text fields keep their analyzers (a newly translated language is mapped as a
keyword); and an index a run with the registry off built holds documents the
registry does not describe. The OpenSearch indexer records its settings in
each index's ``_meta`` and ``check_index_settings`` refuses a mismatch.
AWS-free: the indexer runs against an in-memory stand-in for the OpenSearch
client.
"""

from __future__ import annotations

import copy
import fnmatch
from datetime import datetime
from typing import Any

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeGraphStore
from unified_kg_rag.adapters.storage import opensearch_indexer as module
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.ingestion.delta_detector import document_doc_id
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    Document,
    DocumentDelta,
    EmbeddingModelId,
    Entity,
    PipelineContext,
    PipelineStageStatus,
    TextUnit,
)
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

    def put_index_meta(self, index_name: str, meta: dict[str, Any]) -> None:
        self.mappings[index_name]["_meta"] = copy.deepcopy(meta)

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
    mocker,
    client: _FakeOpenSearchClient,
    model_id: str = _TITAN,
    dimension: int = 8,
    translation: dict[str, Any] | None = None,
) -> OpenSearchIndexer:
    mocker.patch.object(module, "OpenSearchClient", return_value=client)
    config = Config(
        models={"embedding": {"model_id": model_id}},
        processing={"translation": translation or {}},
    )
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
    # Only an incremental run upserts, so the registry tracks the index.
    assert mappings["_meta"] == {
        "embedding_model_id": _TITAN,
        "embedding_dimension": 8,
        "doc_status_registry": True,
    }


@pytest.mark.parametrize("tracked", [True, False, None])
def test_full_build_records_whether_the_registry_tracks_it(
    mocker, tracked: bool | None
) -> None:
    client = _FakeOpenSearchClient()
    indexer = _indexer(mocker, client)
    if tracked is not None:
        indexer.set_doc_status_tracking(tracked)
    indexer.index_text_units([_unit()])

    (mappings,) = client.mappings.values()
    assert mappings["_meta"].get("doc_status_registry") is tracked


def test_same_embedding_settings_pass(mocker) -> None:
    client = _FakeOpenSearchClient()
    _indexer(mocker, client, _TITAN, 8).upsert_text_units([_unit()])

    _indexer(mocker, client, _TITAN, 8).check_index_settings("default")


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
        changed.check_index_settings("default")
    assert "indexing.reset: true" in str(raised.value)


def _drop_meta(client: _FakeOpenSearchClient) -> None:
    for mappings in client.mappings.values():
        del mappings["_meta"]


def test_an_index_without_recorded_settings_is_refused(mocker) -> None:
    # An earlier release built it (no _meta): this release does not read it.
    client = _FakeOpenSearchClient()
    _indexer(mocker, client).upsert_text_units([_unit()])
    _drop_meta(client)

    with pytest.raises(ConfigurationError, match="earlier release.*indexing.reset"):
        _indexer(mocker, client).check_index_settings("default")


_ENGLISH = {"enabled": True, "source_language": "en", "target_language": "en"}


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        # A newly translated language: its field would be mapped as keyword.
        (
            _ENGLISH,
            {**_ENGLISH, "additional_target_languages": ["ja"]},
            "'translated_text_ja' is not mapped",
        ),
        # Another corpus language: every analyzed field keeps 'english'.
        (
            {"enabled": False, "source_language": "en", "target_language": "en"},
            {"enabled": False, "source_language": "ko", "target_language": "ko"},
            "'text' uses 'english' \\(needs 'nori'\\)",
        ),
    ],
)
def test_other_language_settings_fail_with_the_reset_remedy(
    mocker, before: dict[str, Any], after: dict[str, Any], message: str
) -> None:
    client = _FakeOpenSearchClient()
    _indexer(mocker, client, translation=before).upsert_text_units([_unit()])

    _indexer(mocker, client, translation=before).check_index_settings("default")
    changed = _indexer(mocker, client, translation=after)
    with pytest.raises(ConfigurationError, match=message) as raised:
        changed.check_index_settings("default")
    assert "indexing.reset: true" in str(raised.value)


def test_dropped_language_still_passes(mocker) -> None:
    # The unchanged documents keep the field; nothing new is mapped.
    client = _FakeOpenSearchClient()
    with_ja = {**_ENGLISH, "additional_target_languages": ["ja"]}
    _indexer(mocker, client, translation=with_ja).upsert_text_units([_unit()])

    _indexer(mocker, client, translation=_ENGLISH).check_index_settings("default")


def test_index_built_without_the_registry_fails_with_the_reset_remedy(
    mocker,
) -> None:
    client = _FakeOpenSearchClient()
    untracked = _indexer(mocker, client)
    untracked.set_doc_status_tracking(False)
    untracked.index_text_units([_unit()])

    with pytest.raises(ConfigurationError, match="registry off") as raised:
        _indexer(mocker, client).check_index_settings("default")
    assert "indexing.reset: true" in str(raised.value)

    # A reset with the registry on rebuilds it tracked.
    tracked = _indexer(mocker, client)
    tracked.set_doc_status_tracking(True)
    tracked.index_text_units([_unit()])
    _indexer(mocker, client).check_index_settings("default")


def test_check_is_scoped_to_the_suffix(mocker) -> None:
    client = _FakeOpenSearchClient()
    _indexer(mocker, client, _TITAN, 8).upsert_text_units([_unit(suffix="old")])

    changed = _indexer(mocker, client, _COHERE, 8)
    changed.check_index_settings("new")  # nothing indexed there yet
    with pytest.raises(ConfigurationError):
        changed.check_index_settings("old")


def _stage_context(documents: list[Document], **fields: Any) -> PipelineContext:
    return PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory="/corpus",
        documents=documents,
        **fields,
    )


def _document(path: str) -> Document:
    return Document(
        page_content=f"synthetic text of {path}",
        document_id=f"run-{path}",
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
    )


def test_incremental_run_after_a_registry_off_build_stops_before_any_write(
    mocker,
) -> None:
    # A run with the registry off replaces the indices without touching the
    # registry; the next incremental run trusted the registry's stale records
    # ("1 new, 1 unchanged") and the graph check passed on the shared
    # entities, so the documents of the registry-off build were lost.
    from unified_kg_rag.adapters.providers import Providers
    from unified_kg_rag.application.ingestion import pipeline_stages as ps

    client = _FakeOpenSearchClient()
    graph = FakeGraphStore()
    config = Config()
    indexer = _indexer(mocker, client)

    def providers(**kwargs: Any) -> Providers:
        return Providers(
            config,
            vector_indexer_factory=lambda: indexer,
            graph_indexer_factory=lambda: graph,
            **kwargs,
        )

    old, new = _document("/old.txt"), _document("/new.txt")
    store = FakeDocStatusStore()
    store.put(
        DocStatusRecord(
            doc_id=document_doc_id(old),
            content_hash="h-old",
            status=DocStatus.PROCESSED,
            entity_ids=["e-vendor"],
            text_unit_ids=["t-old"],
        )
    )
    vendor = Entity(id="e-vendor", name="Vendor", type="ORG", text_unit_ids=["t-1"])

    # Registry off: a full build of another corpus over the same suffix.
    ps.IndexingStage(config, providers=providers())._execute_core(
        _stage_context([old], text_units=[_unit("t-1")], resolved_entities=[vendor])
    )
    assert graph.ids("entities") == {"e-vendor"}
    before = store.list_all()

    incremental = _stage_context(
        [new],
        text_units=[_unit("t-2")],
        resolved_entities=[vendor.model_copy(update={"text_unit_ids": ["t-2"]})],
    )
    incremental.incremental_delta = DocumentDelta(
        new=[document_doc_id(new)], unchanged=[document_doc_id(old)]
    )
    stage = ps.IndexingStage(
        config, doc_status=store, providers=providers(doc_status=store)
    )
    written = len(client.documents)
    with pytest.raises(ConfigurationError, match="registry off"):
        stage._execute_core(incremental)

    assert store.list_all() == before
    assert len(client.documents) == written


def test_record_build_failure_is_logged_not_raised(mocker, caplog) -> None:
    from unified_kg_rag.domain.models import IndexBuildRecord

    client = _FakeOpenSearchClient()
    indexer = _indexer(mocker, client)
    indexer.upsert_text_units([_unit()])
    mocker.patch.object(client, "put_index_meta", side_effect=RuntimeError("denied"))
    record = IndexBuildRecord(
        pipeline_id="run-1",
        mode="incremental",
        built_at=datetime(2026, 1, 1),
        documents=1,
        extraction_fingerprint="abc",
    )
    with caplog.at_level("WARNING"):
        indexer.record_build(record, "default")
    assert "Could not record the build of pipeline 'run-1'" in caplog.text
