# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One ``VectorIndexer`` contract for every vector backend.

Runs on the fake (the reference), the embedded SQLite engine, the OpenSearch
container (``LOCAL_STORES=1``) and OpenSearch Service (``-m aws``); see
``tests/fixtures/store_backends.py``. Index administration (aliases, mappings,
``_meta``) is checked on the OpenSearch-protocol backends only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory, HashingEmbeddings
from tests.fixtures.store_backends import (
    BACKENDS,
    INDEX_ADMIN,
    SUFFIX_SCOPED_VECTOR_IDS,
    StoreHarness,
    backends,
    open_harness,
)
from unified_kg_rag.domain.models import (
    Claim,
    CommunityReport,
    Config,
    Entity,
    Relationship,
    TextUnit,
)
from unified_kg_rag.domain.models.config import LanguageCode
from unified_kg_rag.shared import ConfigurationError

pytestmark = pytest.mark.integration

_KINDS = ("text_units", "entities", "relationships", "claims", "community_reports")


@pytest.fixture(params=BACKENDS)
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StoreHarness]:
    yield from open_harness(request.param, tmp_path)


def _tagged(items: list[Any], suffix: str | None) -> list[Any]:
    if suffix is None:
        return items
    return [
        item.model_copy(
            update={"attributes": {**(item.attributes or {}), "index": suffix}}
        )
        for item in items
    ]


def _text_units(suffix: str | None = None) -> list[TextUnit]:
    return _tagged(
        [
            TextUnit(
                id="t1", text="Vendor supplies widgets to Buyer under a purchase order."
            ),
            TextUnit(id="t2", text="Vendor ships widgets from the Depot every week."),
        ],
        suffix,
    )


def _entities(suffix: str | None = None) -> list[Entity]:
    return _tagged(
        [
            Entity(
                id="e-vendor",
                name="Vendor",
                type="ORG",
                description="Vendor supplies widgets",
                text_unit_ids=["t1", "t2"],
                rank=3,
            ),
            Entity(
                id="e-buyer",
                name="Buyer",
                type="ORG",
                description="Buyer orders widgets",
                text_unit_ids=["t1"],
                rank=2,
            ),
            Entity(
                id="e-depot",
                name="Depot",
                type="LOCATION",
                description="Depot stores widgets",
                text_unit_ids=["t2"],
                rank=1,
            ),
        ],
        suffix,
    )


def _relationships(suffix: str | None = None) -> list[Relationship]:
    return _tagged(
        [
            Relationship(
                id="r-supplies",
                source_id="e-vendor",
                target_id="e-buyer",
                source_name="Vendor",
                target_name="Buyer",
                type="SUPPLIES",
                description="Vendor supplies widgets to Buyer",
                weight=1.0,
                text_unit_ids=["t1"],
            ),
            Relationship(
                id="r-ships",
                source_id="e-vendor",
                target_id="e-depot",
                source_name="Vendor",
                target_name="Depot",
                description="Vendor ships widgets from Depot",
                weight=0.5,
                text_unit_ids=["t2"],
            ),
        ],
        suffix,
    )


def _claims(suffix: str | None = None) -> list[Claim]:
    return _tagged(
        [
            Claim(
                id="cl1",
                subject_id="e-vendor",
                subject_name="Vendor",
                object_id="e-buyer",
                object_name="Buyer",
                type="OBLIGATION",
                status="TRUE",
                description="Vendor must deliver widgets every week",
                source_text="Vendor supplies widgets",
                text_unit_ids=["t1"],
            )
        ],
        suffix,
    )


def _reports(suffix: str | None = None) -> list[CommunityReport]:
    return _tagged(
        [
            CommunityReport(
                id="cr1",
                community_id="c1",
                name="Widget supply",
                summary="Vendor supplies widgets to Buyer through the Depot.",
                full_content="Widget supply\n\nVendor, Buyer and Depot supply.",
                rank=2.0,
                text_unit_ids=["t1", "t2"],
            )
        ],
        suffix,
    )


_ITEMS: dict[str, Callable[[str | None], list[Any]]] = {
    "text_units": _text_units,
    "entities": _entities,
    "relationships": _relationships,
    "claims": _claims,
    "community_reports": _reports,
}
_PREFIX_ATTRIBUTES = {
    "text_units": "text_units_index_prefix",
    "entities": "entities_index_prefix",
    "relationships": "relationships_index_prefix",
    "claims": "claims_index_prefix",
    "community_reports": "community_reports_index_prefix",
}


def _write(h: StoreHarness, kind: str, mode: str, items: list[Any], **kw: Any):
    stats = getattr(h.vector_indexer, f"{mode}_{kind}")(items, **kw)
    assert stats.failed_items == 0, stats.errors
    return stats


@pytest.mark.parametrize("kind", _KINDS)
def test_index_upsert_and_delete_each_artifact_type(
    harness: StoreHarness, kind: str
) -> None:
    items = _ITEMS[kind](None)
    _write(harness, kind, "index", items)
    assert harness.stored_ids(kind) == {item.id for item in items}

    new = items[0].model_copy(update={"id": f"{items[0].id}-new"})
    _write(harness, kind, "upsert", [items[0], new])
    assert harness.stored_ids(kind) == {item.id for item in items} | {new.id}

    prefix = getattr(harness.config.indexing.opensearch, _PREFIX_ATTRIBUTES[kind])
    stats = harness.vector_indexer.delete_by_id(
        [items[0].id, "never-written"], prefix, "default"
    )
    assert stats.failed_items == 0
    assert harness.stored_ids(kind) == {item.id for item in items[1:]} | {new.id}

    # A full build replaces the suffix's documents.
    _write(harness, kind, "index", [new])
    assert harness.stored_ids(kind) == {new.id}


def test_a_full_build_with_suffixes_empties_them(harness: StoreHarness) -> None:
    tenant = harness.suffix("s1")
    _write(harness, "entities", "index", _entities(tenant))
    _write(harness, "entities", "index", _entities())
    _write(harness, "entities", "index", [], suffixes=[tenant])
    assert harness.stored_ids("entities", tenant) == set()
    assert harness.stored_ids("entities") == {"e-vendor", "e-buyer", "e-depot"}


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_full_reindex_keeps_one_index_per_alias(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each full run builds a new timestamped index and swaps the alias onto
    # it; the previous ones must be deleted, or every run leaks an index.
    start = datetime(2026, 1, 1)
    ticks = iter(range(10_000))

    class _Clock:
        @staticmethod
        def now() -> datetime:
            return start + timedelta(seconds=next(ticks))

    monkeypatch.setattr("unified_kg_rag.shared.utils.store_names.datetime", _Clock)
    indexer = harness.vector_indexer
    client = indexer.opensearch_client
    alias = indexer._get_name(
        harness.config.indexing.opensearch.entities_index_prefix, None
    )
    # Another store's index whose name also matches ``<alias>-*``.
    sibling = f"{alias}-2-20250101000000"
    client.create_index(sibling, {})
    try:
        for _ in range(3):
            _write(harness, "entities", "index", _entities())
        live = client.get_indices_by_alias(alias)
        assert len(live) == 1
        assert client.get_aliases_by_index(f"{alias}-*") == {
            live[0]: [alias],
            sibling: [],
        }
        assert harness.stored_ids("entities") == {"e-vendor", "e-buyer", "e-depot"}
        indexer.clear(["default"])
        assert client.get_aliases_by_index(f"{alias}-*") == {sibling: []}
    finally:
        client.delete_indices([sibling])


def test_clear_keeps_sibling_namespaces(harness: StoreHarness) -> None:
    tag = harness.suffix("clr")
    extended = harness.suffix("clr-eu")
    sibling = harness.derive(
        lambda c: setattr(
            c.indexing, "additional_suffix", f"{c.indexing.additional_suffix or ''}x"
        )
    )
    for store, suffix in ((harness, tag), (harness, extended), (sibling, tag)):
        _write(store, "entities", "index", _entities(suffix))
    harness.vector_indexer.clear([tag])
    assert harness.stored_ids("entities", tag) == set()
    assert len(harness.stored_ids("entities", extended)) == 3
    assert len(sibling.stored_ids("entities", tag)) == 3
    # A namespace never indexed clears too.
    harness.vector_indexer.clear([harness.suffix("never")])


class _CountingFactory(HashingEmbeddingFactory):
    """Counts the texts embedded through its models."""

    def __init__(self) -> None:
        super().__init__()
        self.embedded = 0

    def get_model(self, model_id: Any, **kwargs: Any) -> HashingEmbeddings:
        factory = self

        class _Counting(HashingEmbeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                factory.embedded += len(texts)
                return super().embed_documents(texts)

            def embed_query(self, text: str) -> list[float]:
                factory.embedded += 1
                return super().embed_query(text)

        return _Counting(self.dimensions)


def test_lineage_updates_rewrite_fields_without_reembedding(
    harness: StoreHarness,
) -> None:
    counting = _CountingFactory()
    h = harness.derive(embedding_factory=counting)
    indexer = h.vector_indexer
    _write(h, "entities", "index", _entities())
    _write(h, "relationships", "index", _relationships())
    _write(h, "claims", "index", _claims())
    # The fake embeds nothing; a real store embeds each written document.
    written = counting.embedded

    vendor, *_ = _entities()
    ships = _relationships()[1]
    (claim,) = _claims()
    for stats in (
        indexer.update_entity_lineage(
            [vendor.model_copy(update={"text_unit_ids": ["t9"]})]
        ),
        indexer.update_relationship_lineage(
            [ships.model_copy(update={"text_unit_ids": ["t9"], "weight": 4.0})]
        ),
        indexer.update_claim_lineage(
            [claim.model_copy(update={"text_unit_ids": ["t9"]})]
        ),
        # An id without a document has nothing to update.
        indexer.update_entity_lineage([Entity(id="e-ghost", name="Ghost")]),
    ):
        assert stats.failed_items == 0, stats.errors
    assert counting.embedded == written
    (read,) = indexer.read_claims(["cl1"])
    assert read.text_unit_ids == ["t9"]
    assert h.stored_ids("entities") == {"e-vendor", "e-buyer", "e-depot"}
    assert indexer.read_entity_descriptions(["e-vendor"]) == {
        "e-vendor": "Vendor supplies widgets"
    }


def test_claims_read_from_a_suffix_are_tagged_and_written_back_there(
    harness: StoreHarness,
) -> None:
    indexer = harness.vector_indexer
    tenant = harness.suffix("tenant")
    (claim,) = _claims(tenant)
    _write(
        harness,
        "claims",
        "upsert",
        [claim.model_copy(update={"text_unit_ids": ["t1", "t2"]})],
    )
    scoped = SUFFIX_SCOPED_VECTOR_IDS in harness.capabilities
    if scoped:
        # The same id in the default suffix is another document.
        _write(harness, "claims", "upsert", _claims())

    (read,) = indexer.read_claims(["cl1", "ghost"], suffix=tenant)
    assert read.text_unit_ids == ["t1", "t2"]
    assert read.attributes == {"index": tenant}
    assert (read.subject_id, read.object_name, read.status) == (
        "e-vendor",
        "Buyer",
        "TRUE",
    )
    stats = indexer.update_claim_lineage(
        [read.model_copy(update={"text_unit_ids": ["t2"]})]
    )
    assert (stats.successful_items, stats.failed_items) == (1, 0)
    (after,) = indexer.read_claims(["cl1"], suffix=tenant)
    assert after.text_unit_ids == ["t2"]
    if scoped:
        (default,) = indexer.read_claims(["cl1"])
        assert default.text_unit_ids == ["t1"]
        assert default.attributes in (None, {})


def test_description_reads_are_scoped_to_a_suffix(harness: StoreHarness) -> None:
    indexer = harness.vector_indexer
    tenant = harness.suffix("tenant")
    _write(harness, "relationships", "upsert", _relationships(tenant))
    _write(harness, "entities", "upsert", _entities(tenant))
    assert indexer.read_relationship_descriptions(
        ["r-supplies", "r-missing"], suffix=tenant
    ) == {"r-supplies": "Vendor supplies widgets to Buyer"}
    assert indexer.read_entity_descriptions(["e-depot"], suffix=tenant) == {
        "e-depot": "Depot stores widgets"
    }
    # The default suffix holds none of them.
    assert indexer.read_relationship_descriptions(["r-supplies"]) == {}
    assert indexer.read_entity_descriptions(["e-depot"]) == {}


def test_text_unit_id_reads_are_scoped_to_a_suffix(harness: StoreHarness) -> None:
    # The incremental probe that the vector store still holds the registry's
    # documents: [] without a live index, the stored ids with one.
    indexer = harness.vector_indexer
    tenant = harness.suffix("tenant")
    assert indexer.read_text_unit_ids(["t1"], suffix=tenant) == []
    _write(harness, "text_units", "upsert", _text_units(tenant))
    found = indexer.read_text_unit_ids(["t1", "t-missing", "t2"], suffix=tenant)
    assert sorted(found) == ["t1", "t2"]
    assert indexer.read_text_unit_ids(["t1"]) == []


def test_delete_document_artifacts_reaches_every_index(harness: StoreHarness) -> None:
    for kind in _KINDS:
        _write(harness, kind, "index", _ITEMS[kind](None))
    results = harness.vector_indexer.delete_document_artifacts(
        ["t2", "e-depot", "cr1", "cl1"],
        "default",
        extra_relationship_ids=["r-ships"],
    )
    assert results
    assert all(stats.failed_items == 0 for stats in results.values())
    assert harness.stored_ids("text_units") == {"t1"}
    assert harness.stored_ids("entities") == {"e-vendor", "e-buyer"}
    # The orphaned edge goes from the relationship index only.
    assert harness.stored_ids("relationships") == {"r-supplies"}
    assert harness.stored_ids("claims") == set()
    assert harness.stored_ids("community_reports") == set()
    assert harness.vector_indexer.delete_document_artifacts([], "default") == {}


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_incremental_write_refuses_another_embedding_model(
    harness: StoreHarness,
) -> None:
    # The index records the model and dimension in its mapping's _meta; a
    # run embedding with another one must not write into it.
    _write(harness, "entities", "upsert", _entities())
    harness.vector_indexer.check_index_settings("default")
    other_model = harness.derive(
        lambda c: setattr(c.models.embedding, "model_id", "other-embed")
    )
    other_dimension = harness.derive(embedding_factory=HashingEmbeddingFactory(32))
    for other in (other_model, other_dimension):
        with pytest.raises(ConfigurationError, match="indexing.reset: true"):
            other.vector_indexer.check_index_settings("default")


def _set_languages(**languages: Any) -> Callable[[Config], None]:
    def change(config: Config) -> None:
        for key, value in languages.items():
            setattr(config.processing.translation, key, value)

    return change


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_incremental_write_refuses_other_languages_or_an_untracked_build(
    harness: StoreHarness,
) -> None:
    # A text field's analyzer cannot change on a live index, and a field it
    # does not map yet would be mapped as a keyword on first write.
    indexer = harness.vector_indexer
    _write(harness, "text_units", "upsert", _text_units())
    indexer.check_index_settings("default")
    for change, message in (
        (
            _set_languages(additional_target_languages=[LanguageCode.JA]),
            "'translated_text_ja' is not mapped",
        ),
        (_set_languages(source_language=LanguageCode.KO), "'text' uses 'english'"),
    ):
        with pytest.raises(ConfigurationError, match=message):
            harness.derive(change).vector_indexer.check_index_settings("default")

    # A full build by a run with the registry off is not updated
    # incrementally, nor is an index without _meta (an earlier release's).
    indexer.set_doc_status_tracking(False)
    _write(harness, "text_units", "index", _text_units())
    with pytest.raises(ConfigurationError, match="registry off"):
        indexer.check_index_settings("default")
    client = indexer.opensearch_client
    index_name = client.get_index_name_by_alias(
        indexer._get_name(
            harness.config.indexing.opensearch.text_units_index_prefix, None
        )
    )
    assert index_name
    client.put_index_meta(index_name, {})
    with pytest.raises(ConfigurationError, match="earlier release"):
        indexer.check_index_settings("default")
