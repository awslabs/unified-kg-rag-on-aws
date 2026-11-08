# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One ``VectorIndexer`` contract for every vector backend.

Runs on the fake (the reference), the embedded SQLite engine, the OpenSearch
container (``LOCAL_STORES=1``) and OpenSearch Service (``-m aws``); see
``tests/fixtures/store_backends.py``. Index administration (aliases, mappings,
``_meta``) is checked on the OpenSearch-protocol backends only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory, HashingEmbeddings
from tests.fixtures.store_backends import (
    BACKENDS,
    INDEX_ADMIN,
    RETRIEVERS,
    SUFFIX_SCOPED_VECTOR_IDS,
    StoreHarness,
    backends,
    open_harness,
)
from unified_kg_rag.domain.ingestion.report_references import (
    REPORT_FINDINGS_FIELD,
    report_cited_ids,
)
from unified_kg_rag.domain.models import (
    Claim,
    CommunityFinding,
    CommunityReport,
    Config,
    Entity,
    IndexArtifact,
    IndexBuildRecord,
    Methodology,
    Relationship,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchType,
    TextUnit,
)
from unified_kg_rag.domain.models.config import LanguageCode
from unified_kg_rag.ports.indexer import STORED_ITEM_KINDS, StoredAnnotation
from unified_kg_rag.shared import ConfigurationError, NamespaceLeaseError
from unified_kg_rag.shared.exceptions import GraphRAGException

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

    monkeypatch.setattr("unified_kg_rag.domain.retrieval.store_names.datetime", _Clock)
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
        "true",  # stored lowercase
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


def test_a_claims_dates_are_read_back(harness: StoreHarness) -> None:
    # The dates are part of a claim's identity (resolved_claim_id): read back
    # without them, a stored claim re-keyed to follow a renamed entity took a
    # date-free id that a full build never gives it.
    (claim,) = _claims()
    dated = claim.model_copy(update={"start_date": "2024-01-01"})
    _write(harness, "claims", "upsert", [dated])
    (read,) = harness.vector_indexer.read_claims(["cl1"])
    assert (read.start_date, read.end_date) == ("2024-01-01", None)


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


def test_vector_copy_reads_hold_the_compared_fields(harness: StoreHarness) -> None:
    # What an incremental write compares with the derived items (see
    # VectorIndexer.read_vector_copies), scoped to a suffix.
    indexer = harness.vector_indexer
    tenant = harness.suffix("tenant")
    _write(harness, "relationships", "upsert", _relationships(tenant))
    _write(harness, "entities", "upsert", _entities(tenant))

    assert indexer.read_vector_copies(
        "entities", ["e-depot", "e-missing"], suffix=tenant
    ) == {
        "e-depot": {
            "id": "e-depot",
            "name": "Depot",
            "type": "LOCATION",
            "confidence": 1.0,
            "description": "Depot stores widgets",
        }
    }
    assert indexer.read_vector_copies("relationships", ["r-ships"], suffix=tenant) == {
        "r-ships": {
            "id": "r-ships",
            "source_id": "e-vendor",
            "target_id": "e-depot",
            "source_name": "Vendor",
            "target_name": "Depot",
            "description": "Vendor ships widgets from Depot",
        }
    }
    assert indexer.read_vector_copies("entities", ["e-depot"]) == {}


def test_vector_copy_reads_hold_the_schema_annotations(harness: StoreHarness) -> None:
    indexer = harness.vector_indexer
    iri = "https://example.org/onto#Location"
    entities = _entities()
    entities[2] = entities[2].model_copy(update={"in_schema": True, "type_iri": iri})
    entities[1] = entities[1].model_copy(update={"in_schema": False})
    relationships = _relationships()
    relationships[1] = relationships[1].model_copy(update={"in_schema": False})
    _write(harness, "entities", "upsert", entities)
    _write(harness, "relationships", "upsert", relationships)
    copies = indexer.read_vector_copies("entities", ["e-depot", "e-buyer", "e-vendor"])
    annotations = {
        i: {k: c[k] for k in ("in_schema", "type_iri") if k in c}
        for i, c in copies.items()
    }
    # Written only when set.
    assert annotations == {
        "e-depot": {"in_schema": True, "type_iri": iri},
        "e-buyer": {"in_schema": False},
        "e-vendor": {},
    }
    copies = indexer.read_vector_copies("relationships", ["r-ships", "r-supplies"])
    assert copies["r-ships"]["in_schema"] is False
    assert "in_schema" not in copies["r-supplies"]
    assert "predicate_iri" not in copies["r-ships"]


def test_vector_copy_reads_hold_the_seed_graph_fields(harness: StoreHarness) -> None:
    indexer = harness.vector_indexer
    iri = "https://example.org/procure#depot"
    entities = _entities()
    entities[2] = entities[2].model_copy(
        update={"iri": iri, "source_kinds": ["document", "seed_graph"]}
    )
    entities[1] = entities[1].model_copy(update={"source_kinds": ["seed_graph"]})
    relationships = _relationships()
    relationships[1] = relationships[1].model_copy(
        update={"source_kinds": ["seed_graph"]}
    )
    _write(harness, "entities", "upsert", entities)
    _write(harness, "relationships", "upsert", relationships)
    copies = indexer.read_vector_copies("entities", ["e-depot", "e-buyer", "e-vendor"])
    fields = {
        i: {k: c[k] for k in ("iri", "source_kinds") if k in c}
        for i, c in copies.items()
    }
    # Written only when set: an item no seed graph states is stored as before.
    assert fields == {
        "e-depot": {"iri": iri, "source_kinds": ["document", "seed_graph"]},
        "e-buyer": {"source_kinds": ["seed_graph"]},
        "e-vendor": {},
    }
    copies = indexer.read_vector_copies("relationships", ["r-ships", "r-supplies"])
    assert copies["r-ships"]["source_kinds"] == ["seed_graph"]
    assert "source_kinds" not in copies["r-supplies"]


@pytest.mark.parametrize("mode", ["index", "upsert"])
def test_a_seed_graph_unit_is_stored_with_its_source_kind(
    harness: StoreHarness, mode: str
) -> None:
    # The top-level keyword a text search leaves seed units out by; a
    # document's unit is written without it, as before seed graphs.
    seed = TextUnit(
        id="sg-00000000-0000-0000-0000-000000000001",
        text='{"id":"v1","kind":"node","name":"Vendor"}',
    )
    _write(harness, "text_units", mode, [*_text_units(), seed])
    found = harness.vector_indexer.find_documents(
        harness.config.indexing.opensearch.text_units_index_prefix,
        "source_kind",
        ["seed_graph"],
        "default",
    )
    assert set(found) == {seed.id}
    assert harness.stored_ids("text_units") == {"t1", "t2", seed.id}


def test_schema_annotations_are_listed_and_rewritten_in_place(
    harness: StoreHarness,
) -> None:
    indexer = harness.vector_indexer
    tenant = harness.suffix("ann")
    iri = "https://example.org/onto#Location"
    entities = _entities()
    entities[2] = entities[2].model_copy(update={"in_schema": True, "type_iri": iri})
    relationships = _relationships()
    relationships[1] = relationships[1].model_copy(update={"in_schema": False})
    _write(harness, "entities", "upsert", entities)
    _write(harness, "relationships", "upsert", relationships)
    elsewhere = _entities(tenant)[2].model_copy(
        update={"id": "e-depot-2", "in_schema": True, "type_iri": iri}
    )
    _write(harness, "entities", "upsert", [elsewhere])
    assert indexer.read_schema_annotations("entities") == {
        "e-vendor": StoredAnnotation(type="ORG"),
        "e-buyer": StoredAnnotation(type="ORG"),
        "e-depot": StoredAnnotation(type="LOCATION", in_schema=True, iri=iri),
    }
    # The relationships index does not store the type.
    assert indexer.read_schema_annotations("relationships") == {
        "r-supplies": StoredAnnotation(),
        "r-ships": StoredAnnotation(in_schema=False),
    }

    place = "https://example.org/onto#Place"
    stats = indexer.write_schema_annotations(
        "entities",
        {"e-depot": (None, place), "e-vendor": (True, None), "gone": (False, None)},
    )
    assert stats.failed_items == 0
    stats = indexer.write_schema_annotations(
        "relationships", {"r-ships": (None, None), "r-supplies": (True, "urn:x:s")}
    )
    assert stats.failed_items == 0
    assert indexer.read_schema_annotations("entities") == {
        "e-vendor": StoredAnnotation(type="ORG", in_schema=True),
        "e-buyer": StoredAnnotation(type="ORG"),
        "e-depot": StoredAnnotation(type="LOCATION", iri=place),
    }
    assert indexer.read_schema_annotations("relationships") == {
        "r-supplies": StoredAnnotation(in_schema=True, iri="urn:x:s"),
        "r-ships": StoredAnnotation(),
    }
    # In place: the rest of the copy, its vectors included, is kept.
    copies = indexer.read_vector_copies("entities", ["e-depot"])
    assert copies["e-depot"]["description"] == "Depot stores widgets"
    assert harness.stored_ids("entities") >= {"e-vendor", "e-buyer", "e-depot"}
    assert indexer.read_schema_annotations("entities", suffix=tenant) == {
        "e-depot-2": StoredAnnotation(type="LOCATION", in_schema=True, iri=iri)
    }


def test_an_in_schema_false_is_written_as_false(harness: StoreHarness) -> None:
    # Not as null: in_schema false is a filter value of its own.
    indexer = harness.vector_indexer
    _write(harness, "entities", "upsert", _entities())
    indexer.write_schema_annotations("entities", {"e-buyer": (False, None)})
    assert indexer.read_schema_annotations("entities")["e-buyer"] == StoredAnnotation(
        type="ORG", in_schema=False
    )


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_a_cleared_annotation_filters_as_one_never_written(
    harness: StoreHarness,
) -> None:
    # Re-annotation clears an annotation with null (an in-place update that
    # left the field out would keep the old value): a filter treats the
    # cleared copy as one written without the field, as a full build writes.
    import asyncio

    from unified_kg_rag.domain.models import SearchQuery, SearchType

    indexer = harness.vector_indexer
    entities = _entities()
    entities[1] = entities[1].model_copy(update={"in_schema": True})
    entities[2] = entities[2].model_copy(update={"in_schema": False})
    _write(harness, "entities", "upsert", entities)
    indexer.write_schema_annotations(
        "entities", {"e-buyer": (None, None), "e-depot": (None, None)}
    )
    assert harness.vector_retriever is not None
    prefix = harness.config.indexing.opensearch.entities_index_prefix

    async def matching(filters: dict[str, Any]) -> set[str]:
        retriever = harness.vector_retriever()
        try:
            hits = await retriever.aretrieve(
                SearchQuery(
                    query="",
                    search_type=SearchType.LEXICAL,
                    top_k=10,
                    retrieval_multiplier=1,
                    index_prefixes=[prefix],
                    filters=filters,
                    source_fields=[],
                )
            )
        finally:
            await retriever.aclose()
        return {str(hit.source) for hit in hits}

    assert asyncio.run(matching({"in_schema": True})) == set()
    assert asyncio.run(matching({"in_schema": False})) == set()
    assert asyncio.run(matching({"type": ["ORG", "LOCATION"]})) == {
        "e-vendor",
        "e-buyer",
        "e-depot",
    }
    copies = indexer.read_vector_copies("entities", ["e-vendor", "e-buyer"])
    assert copies["e-buyer"].get("in_schema") is None
    assert {k: v for k, v in copies["e-buyer"].items() if v is not None}.keys() == {
        k for k, v in copies["e-vendor"].items() if v is not None
    }


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_schema_annotations_are_read_past_one_search_window(
    harness: StoreHarness,
) -> None:
    harness = _windowed(harness, 2)
    _write(harness, "entities", "upsert", _entities())
    assert set(harness.vector_indexer.read_schema_annotations("entities") or {}) == {
        "e-vendor",
        "e-buyer",
        "e-depot",
    }


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_schema_annotation_write_failures_are_counted(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer = harness.vector_indexer
    _write(harness, "entities", "upsert", _entities())
    client = indexer.opensearch_client
    monkeypatch.setattr(
        client,
        "bulk_update",
        lambda index, docs, refresh=False: {
            "items": [{"update": {"_id": "e-buyer", "error": {"type": "x"}}}]
        },
    )
    stats = indexer.write_schema_annotations(
        "entities", {"e-buyer": (True, None), "e-vendor": (True, None)}
    )
    assert (stats.successful_items, stats.failed_items) == (1, 1)
    assert stats.failed_ids == ["e-buyer"]

    def refused(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("bulk refused")

    monkeypatch.setattr(client, "bulk_update", refused)
    stats = indexer.write_schema_annotations(
        "entities", {"e-buyer": (True, None), "e-vendor": (True, None)}
    )
    assert (stats.successful_items, stats.failed_items) == (0, 2)

    def lost(*args: Any, **kwargs: Any) -> Any:
        raise NamespaceLeaseError("lease lost")

    monkeypatch.setattr(client, "bulk_update", lost)
    with pytest.raises(NamespaceLeaseError):
        indexer.write_schema_annotations("entities", {"e-buyer": (True, None)})


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
    with pytest.raises(ConfigurationError, match="registry off") as refused:
        indexer.check_index_settings("default")
    # Named for the engine the harness runs (not "OpenSearch" on SQLite).
    from unified_kg_rag.adapters.storage.search_engine import engine_name

    assert str(refused.value).startswith(
        f"The {engine_name(indexer.opensearch_client)} index "
    )
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


def _dated_claim(claim_id: str, date: str | None, **filters: str) -> Claim:
    return _claims()[0].model_copy(
        update={
            "id": claim_id,
            "start_date": date,
            "end_date": date,
            "attributes": {"filters": filters} if filters else None,
        }
    )


def _earlier_claims_index(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, str]:
    """A claims index as built before the claim dates were mapped (and date
    detection turned off): its search engine client and index name."""
    indexer = harness.vector_indexer
    current = indexer._get_claims_mapping

    def earlier() -> dict[str, Any]:
        mapping = current()
        mapping["mappings"].pop("date_detection", None)
        for name in ("start_date", "end_date"):
            del mapping["mappings"]["properties"][name]
        return mapping

    def earlier_doc(claim: Claim, embeddings: tuple[list[float], ...]) -> Any:
        doc = prepare(claim, embeddings)
        doc.pop("start_date", None), doc.pop("end_date", None)
        return doc

    prepare = indexer._prepare_claim_doc
    monkeypatch.setattr(indexer, "_get_claims_mapping", earlier)
    monkeypatch.setattr(indexer, "_prepare_claim_doc", earlier_doc)
    _write(harness, "claims", "index", [_dated_claim("cl0", None)])
    monkeypatch.undo()
    client = indexer.opensearch_client
    index_name = client.get_index_name_by_alias(
        indexer._get_name(harness.config.indexing.opensearch.claims_index_prefix, None)
    )
    assert index_name
    return client, index_name


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_incremental_run_maps_the_claim_dates_of_an_earlier_claims_index(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unmapped, the claim dates would be mapped as dates on the first dated
    # claim, and every claim with an unknown date would then fail to write.
    client, index_name = _earlier_claims_index(harness, monkeypatch)
    harness.vector_indexer.check_index_settings("default")  # the run's preflight
    _write(harness, "claims", "upsert", [_dated_claim("cl1", "2024-01-15")])
    _write(harness, "claims", "upsert", [_dated_claim("cl2", None)])
    assert harness.stored_ids("claims") == {"cl0", "cl1", "cl2"}
    mappings = client.get_index_mappings(index_name)
    assert mappings["date_detection"] is False
    assert mappings["properties"]["start_date"] == {"type": "keyword"}
    harness.vector_indexer.check_index_settings("default")  # nothing left to add


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_incremental_run_refuses_claim_dates_mapped_as_dates(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As a run without the preflight's mapping update left them: a live
    # index's field types cannot change, so only a reset recovers.
    client, index_name = _earlier_claims_index(harness, monkeypatch)
    client.put_index_mapping(
        index_name, {"properties": {"start_date": {"type": "date"}}}
    )
    with pytest.raises(ConfigurationError, match="'start_date' is mapped as date"):
        harness.vector_indexer.check_index_settings("default")


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_incremental_run_turns_date_detection_off_on_an_index_lacking_no_field(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every field mapped, only date detection still on: the mapping update
    # is sent anyway, or a new date-like filter value typed its field.
    indexer = harness.vector_indexer
    current = indexer._get_claims_mapping

    def earlier() -> dict[str, Any]:
        mapping = current()
        mapping["mappings"].pop("date_detection", None)
        return mapping

    monkeypatch.setattr(indexer, "_get_claims_mapping", earlier)
    _write(harness, "claims", "index", [_dated_claim("cl0", None)])
    monkeypatch.undo()
    client = indexer.opensearch_client
    name = client.get_index_name_by_alias(
        indexer._get_name(harness.config.indexing.opensearch.claims_index_prefix, None)
    )
    assert client.get_index_mappings(name).get("date_detection") is not False
    indexer.check_index_settings("default")
    assert client.get_index_mappings(name)["date_detection"] is False
    _write(
        harness, "claims", "upsert", [_dated_claim("cl1", None, signed="2024-01-15")]
    )
    _write(harness, "claims", "upsert", [_dated_claim("cl2", None, signed="pending")])
    assert harness.stored_ids("claims") >= {"cl1", "cl2"}


def _earlier_entities_index(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, str]:
    """An entities index as built before the schema annotations were mapped:
    its search engine client and index name."""
    indexer = harness.vector_indexer
    current = indexer._get_entities_mapping

    def earlier() -> dict[str, Any]:
        mapping = current()
        for name in ("in_schema", "type_iri", "iri", "source_kinds"):
            del mapping["mappings"]["properties"][name]
        return mapping

    monkeypatch.setattr(indexer, "_get_entities_mapping", earlier)
    _write(harness, "entities", "index", _entities()[:1])
    monkeypatch.undo()
    client = indexer.opensearch_client
    index_name = client.get_index_name_by_alias(
        indexer._get_name(
            harness.config.indexing.opensearch.entities_index_prefix, None
        )
    )
    assert index_name
    assert "in_schema" not in client.get_index_mappings(index_name)["properties"]
    return client, index_name


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_incremental_run_maps_the_schema_annotations_of_an_earlier_index(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, index_name = _earlier_entities_index(harness, monkeypatch)
    harness.vector_indexer.check_index_settings("default")  # the run's preflight
    properties = client.get_index_mappings(index_name)["properties"]
    assert properties["in_schema"] == {"type": "boolean"}
    assert properties["type_iri"] == {"type": "keyword"}
    # And the seed-graph fields, added to an index built before them.
    assert properties["iri"] == properties["source_kinds"] == {"type": "keyword"}
    annotated = [
        e.model_copy(update={"in_schema": i == 1}) for i, e in enumerate(_entities())
    ]
    _write(harness, "entities", "upsert", annotated[1:])
    copies = harness.vector_indexer.read_vector_copies(
        "entities", ["e-vendor", "e-buyer", "e-depot"]
    )
    assert {i: c.get("in_schema") for i, c in copies.items()} == {
        "e-vendor": None,
        "e-buyer": True,
        "e-depot": False,
    }
    harness.vector_indexer.check_index_settings("default")  # nothing left to add


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_incremental_run_maps_the_seed_graph_fields_of_earlier_indices(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The relationships and text-unit indices of an earlier release gain
    # source_kinds / source_kind in the run's preflight, before any write.
    indexer = harness.vector_indexer
    client = indexer.opensearch_client
    o = harness.config.indexing.opensearch
    for kind, field, items, prefix in (
        (
            "relationships",
            "source_kinds",
            _relationships(),
            o.relationships_index_prefix,
        ),
        ("text_units", "source_kind", _text_units(), o.text_units_index_prefix),
    ):
        current = getattr(indexer, f"_get_{kind}_mapping")

        def earlier(current: Any = current, field: str = field) -> dict[str, Any]:
            mapping = current()
            del mapping["mappings"]["properties"][field]
            return mapping

        monkeypatch.setattr(indexer, f"_get_{kind}_mapping", earlier)
        _write(harness, kind, "index", items)
        monkeypatch.undo()
        name = client.get_index_name_by_alias(indexer._get_name(prefix, None))
        assert name and field not in client.get_index_mappings(name)["properties"]
        indexer.check_index_settings("default")
        properties = client.get_index_mappings(name)["properties"]
        assert properties[field] == {"type": "keyword"}, kind
    seeded = _relationships()[0].model_copy(update={"source_kinds": ["seed_graph"]})
    _write(harness, "relationships", "upsert", [seeded])
    copies = indexer.read_vector_copies("relationships", [seeded.id])
    assert copies[seeded.id]["source_kinds"] == ["seed_graph"]


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_a_relationship_hit_names_its_type_on_an_earlier_index(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A relationship without a description (a seed edge) read as its
    # endpoints alone: the vector copy did not store its type.
    import asyncio

    from unified_kg_rag.domain.models import SearchQuery, SearchType

    indexer = harness.vector_indexer
    client = indexer.opensearch_client
    prefix = harness.config.indexing.opensearch.relationships_index_prefix
    undescribed = [r.model_copy(update={"description": ""}) for r in _relationships()]
    current_mapping = indexer._get_relationships_mapping
    current_doc = indexer._prepare_relationship_doc

    def earlier_mapping() -> dict[str, Any]:
        mapping = current_mapping()
        del mapping["mappings"]["properties"]["type"]
        return mapping

    def earlier_doc(rel: Relationship, embeddings: Any) -> dict[str, Any]:
        doc = current_doc(rel, embeddings)
        doc.pop("type", None)
        return doc

    monkeypatch.setattr(indexer, "_get_relationships_mapping", earlier_mapping)
    monkeypatch.setattr(indexer, "_prepare_relationship_doc", earlier_doc)
    _write(harness, "relationships", "index", undescribed)
    monkeypatch.undo()
    name = client.get_index_name_by_alias(indexer._get_name(prefix, None))
    assert name and "type" not in client.get_index_mappings(name)["properties"]
    indexer.check_index_settings("default")  # the run's preflight maps it
    assert client.get_index_mappings(name)["properties"]["type"] == {"type": "keyword"}

    def contents() -> dict[str, str]:
        assert harness.vector_retriever is not None
        retriever = harness.vector_retriever()

        async def search() -> list[Any]:
            try:
                return await retriever.aretrieve(
                    SearchQuery(
                        query="Vendor",
                        search_type=SearchType.LEXICAL,
                        top_k=10,
                        retrieval_multiplier=1,
                        index_prefixes=[prefix],
                    )
                )
            finally:
                await retriever.aclose()

        return {str(hit.source): hit.content for hit in asyncio.run(search())}

    # An earlier copy still reads, without its type.
    assert contents() == {
        "r-supplies": "Relationship: Vendor -> Buyer",
        "r-ships": "Relationship: Vendor -> Depot",
    }
    # A copy written now names it; an untyped one names the graph's label.
    _write(harness, "relationships", "upsert", undescribed)
    assert contents() == {
        "r-supplies": "Relationship: Vendor -[SUPPLIES]-> Buyer",
        "r-ships": "Relationship: Vendor -[RELATED_TO]-> Depot",
    }


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_incremental_run_refuses_an_annotation_mapped_as_another_type(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, index_name = _earlier_entities_index(harness, monkeypatch)
    client.put_index_mapping(
        index_name, {"properties": {"in_schema": {"type": "keyword"}}}
    )
    with pytest.raises(
        ConfigurationError, match="'in_schema' is mapped as keyword \\(needs boolean\\)"
    ):
        harness.vector_indexer.check_index_settings("default")


# Seed properties a dynamic mapping would refuse: a key typed by its first
# value, a string over the 32 KB keyword limit, an integer over 2^63, keys
# OpenSearch cannot map as fields and more keys than its field limit.
_ODD_SEED_PROPERTIES: list[dict[str, Any]] = [
    {"code": 5},
    {"code": "X1"},
    {"flag": True},
    {"flag": "yes"},
    {"note": "a" * 40_000},
    {"n": 2**64},
    {"": "x"},
    {".": "x"},
    {"a.b": "x"},
    {"a": "y"},
    {f"a{k}": "v" for k in range(512)},
    {f"b{k}": "v" for k in range(512)},
]


def _with_seed_properties(properties: list[dict[str, Any]]) -> list[Entity]:
    return [
        Entity(
            id=f"e-seed{i}",
            name=f"Seed {i}",
            type="ORG",
            description="A seed graph states it",
            text_unit_ids=["sg-t1"],
            attributes={"seed_properties": props, "filters": {"region": "eu"}},
        )
        for i, props in enumerate(properties)
    ]


@pytest.mark.parametrize("mode", ["index", "upsert"])
def test_seed_properties_of_any_form_are_written_and_read_back(
    harness: StoreHarness, mode: str
) -> None:
    entities = _with_seed_properties(_ODD_SEED_PROPERTIES)
    _write(harness, "entities", mode, entities)
    copies = harness.vector_indexer.read_vector_copies(
        "entities", [e.id for e in entities]
    )
    assert {i: c.get("seed_properties") for i, c in copies.items()} == {
        e.id: (e.attributes or {})["seed_properties"] for e in entities
    }


def test_a_lineage_update_sets_the_filters_and_keeps_other_attributes(
    harness: StoreHarness,
) -> None:
    # As the incremental strip relies on: a lineage update rewrites the
    # document filters (a key the item lost stops matching) and leaves every
    # other attribute as written, so a copy whose seed properties changed
    # has to be written in full (indexing_manager._vector_content).
    indexer = harness.vector_indexer
    prefix = harness.config.indexing.opensearch.entities_index_prefix
    (vendor,) = _with_seed_properties([{"tier": "gold"}])
    stored = vendor.model_copy(
        update={
            "attributes": {
                **(vendor.attributes or {}),
                "filters": {"region": "eu", "year": "2024"},
            }
        }
    )
    _write(harness, "entities", "upsert", [stored])
    stats = indexer.update_entity_lineage(
        [
            vendor.model_copy(
                update={
                    "text_unit_ids": ["t9"],
                    "attributes": {"filters": {"region": "eu"}},
                }
            )
        ]
    )
    assert stats.failed_items == 0, stats.errors
    copies = indexer.read_vector_copies("entities", [vendor.id])
    assert copies[vendor.id]["seed_properties"] == {"tier": "gold"}
    assert indexer.find_documents(
        prefix, "attributes.filters.region", ["eu"], "default"
    ) == {vendor.id: ["t9"]}
    assert (
        indexer.find_documents(prefix, "attributes.filters.year", ["2024"], "default")
        == {}
    )
    # An item that lost its last filtered document: no key matches.
    stats = indexer.update_entity_lineage(
        [vendor.model_copy(update={"text_unit_ids": ["t8"], "attributes": None})]
    )
    assert stats.failed_items == 0, stats.errors
    assert (
        indexer.find_documents(prefix, "attributes.filters.region", ["eu"], "default")
        == {}
    )
    copies = indexer.read_vector_copies("entities", [vendor.id])
    assert copies[vendor.id]["seed_properties"] == {"tier": "gold"}


def _entities_index_without_seed_properties(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, str]:
    """An entities index built before seed properties were left unmapped:
    its search engine client and index name."""
    indexer = harness.vector_indexer
    current = indexer._get_entities_mapping

    def earlier() -> dict[str, Any]:
        mapping = current()
        mapping["mappings"]["properties"]["attributes"].pop("properties")
        return mapping

    monkeypatch.setattr(indexer, "_get_entities_mapping", earlier)
    _write(harness, "entities", "index", _entities()[:1])
    monkeypatch.undo()
    client = indexer.opensearch_client
    index_name = client.get_index_name_by_alias(
        indexer._get_name(
            harness.config.indexing.opensearch.entities_index_prefix, None
        )
    )
    assert index_name
    attributes = client.get_index_mappings(index_name)["properties"]["attributes"]
    assert "seed_properties" not in (attributes.get("properties") or {})
    return client, index_name


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_incremental_run_leaves_the_seed_properties_of_an_earlier_index_unmapped(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, index_name = _entities_index_without_seed_properties(harness, monkeypatch)
    harness.vector_indexer.check_index_settings("default")  # the run's preflight
    attributes = client.get_index_mappings(index_name)["properties"]["attributes"]
    assert attributes["properties"]["seed_properties"] == {
        "type": "object",
        "enabled": False,
    }
    _write(harness, "entities", "upsert", _with_seed_properties(_ODD_SEED_PROPERTIES))
    harness.vector_indexer.check_index_settings("default")  # nothing left to add


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_incremental_run_refuses_seed_properties_mapped_key_by_key(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What OpenSearch's dynamic mapping made of an earlier build's seed
    # properties; enabled cannot change on a live object.
    client, index_name = _entities_index_without_seed_properties(harness, monkeypatch)
    client.put_index_mapping(
        index_name,
        {
            "properties": {
                "attributes": {
                    "properties": {
                        "seed_properties": {"properties": {"code": {"type": "long"}}}
                    }
                }
            }
        },
    )
    with pytest.raises(
        ConfigurationError,
        match="'attributes.seed_properties' field by field.*indexing.reset: true",
    ):
        harness.vector_indexer.check_index_settings("default")


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_a_live_objects_own_parameters_cannot_change(harness: StoreHarness) -> None:
    # Why an index whose seed properties a dynamic mapping made is refused,
    # not remapped: a sub-field may be added, enabled may not change.
    _write(harness, "entities", "index", _entities()[:1])
    client = harness.vector_indexer.opensearch_client
    index_name = client.get_index_name_by_alias(
        harness.vector_indexer._get_name(
            harness.config.indexing.opensearch.entities_index_prefix, None
        )
    )
    assert index_name
    # OpenSearch answers a 500 mapper_exception, the embedded engine a 400.
    with pytest.raises(GraphRAGException, match="enabled"):
        client.put_index_mapping(
            index_name,
            {
                "properties": {
                    "attributes": {
                        "properties": {
                            "seed_properties": {
                                "type": "object",
                                "enabled": True,
                                "properties": {"code": {"type": "long"}},
                            }
                        }
                    }
                }
            },
        )
    client.put_index_mapping(
        index_name,
        {"properties": {"attributes": {"properties": {"note": {"type": "keyword"}}}}},
    )
    attributes = client.get_index_mappings(index_name)["properties"]["attributes"]
    assert attributes["properties"]["note"] == {"type": "keyword"}
    assert attributes["properties"]["seed_properties"]["enabled"] is False


def test_a_date_like_filter_value_does_not_type_its_field(
    harness: StoreHarness,
) -> None:
    # Date detection would map attr_signed as a date on the first value, and
    # the next value of another form would fail to write.
    _write(
        harness, "claims", "upsert", [_dated_claim("cl1", None, signed="2024-01-15")]
    )
    _write(harness, "claims", "upsert", [_dated_claim("cl2", None, signed="pending")])
    assert harness.stored_ids("claims") == {"cl1", "cl2"}


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_claim_filters_match_as_the_user_guide_says(harness: StoreHarness) -> None:
    # Regression: type and status were stored as the model wrote them, so
    # status:true matched nothing; an unknown date was stored as "", which
    # every "lte" range matched; a written-out date compared as a string.
    import asyncio

    from unified_kg_rag.domain.models import SearchQuery, SearchType

    (claim,) = _claims(None)
    dated = {
        "cl-iso": "2024-01-15",
        "cl-written": "January 20, 2024",
        "cl-unknown": "UNKNOWN",
        "cl-none": None,
        "cl-quarter": "Q1 2024",
        "cl-text": "early 2024",
    }
    _write(
        harness,
        "claims",
        "upsert",
        [
            claim.model_copy(update={"id": i, "status": "TRUE", "end_date": d})
            for i, d in dated.items()
        ],
    )
    assert harness.vector_retriever is not None
    prefix = harness.config.indexing.opensearch.claims_index_prefix

    async def matching(filters: dict[str, Any]) -> set[str]:
        retriever = harness.vector_retriever()
        try:
            hits = await retriever.aretrieve(
                SearchQuery(
                    query="",
                    search_type=SearchType.LEXICAL,
                    top_k=10,
                    retrieval_multiplier=1,
                    index_prefixes=[prefix],
                    filters=filters,
                    source_fields=[],
                )
            )
        finally:
            await retriever.aclose()
        return {str(hit.source) for hit in hits}

    assert asyncio.run(matching({"status": "true"})) == set(dated)
    assert asyncio.run(matching({"type": "obligation"})) == set(dated)
    # An end date in a quarter is stored as its last day: the claim lasts
    # until 2024-03-31, so it ends after January.
    assert asyncio.run(matching({"end_date": {"lte": "2024-01-31"}})) == {
        "cl-iso",
        "cl-written",
    }
    assert asyncio.run(
        matching({"end_date": {"gte": "2024-02-01", "lte": "2024-12-31"}})
    ) == {"cl-quarter"}
    assert asyncio.run(matching({"end_date": "early 2024"})) == {"cl-text"}


def test_find_documents_matches_a_keyword_field_in_a_suffix(
    harness: StoreHarness,
) -> None:
    indexer = harness.vector_indexer
    tenant = harness.suffix("tenant")
    prefixes = harness.config.indexing.opensearch
    _write(harness, "claims", "upsert", _claims(None))
    _write(harness, "community_reports", "upsert", _reports(tenant))

    found = indexer.find_documents(
        prefixes.claims_index_prefix, "text_unit_ids", ["t2", "t-none"], "default"
    )
    assert {i: sorted(units) for i, units in found.items()} == {
        c.id: sorted(c.text_unit_ids or [])
        for c in _claims(None)
        if "t2" in (c.text_unit_ids or [])
    }
    reports = indexer.find_documents(
        prefixes.community_reports_index_prefix, "text_unit_ids", ["t1"], tenant
    )
    assert {i: sorted(units) for i, units in reports.items()} == {"cr1": ["t1", "t2"]}
    assert (
        indexer.find_documents(
            prefixes.community_reports_index_prefix, "text_unit_ids", ["t1"], "default"
        )
        == {}
    )


def _windowed(harness: StoreHarness, window: int) -> StoreHarness:
    def set_window(config: Config) -> None:
        config.indexing.opensearch.index_settings = {
            **config.indexing.opensearch.index_settings,
            "max_result_window": window,
        }

    return harness.derive(set_window)


def test_find_documents_reads_more_matches_than_one_search_returns(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A hub's edges outnumber one search's window (an index created with
    # max_result_window 2): they are read in pages (the lookup used to raise,
    # failing the run for good). The stores count the matches only up to
    # the window, as OpenSearch counts only 10,000 by default (``relation:
    # gte``): paging must not wait for a total past the window, which never
    # comes.
    harness = _windowed(harness, 2)
    indexer = harness.vector_indexer
    if harness.kind != "fake":
        search = indexer.opensearch_client.search

        def counting_to_the_window(index: str, body: dict[str, Any]) -> Any:
            return search(index=index, body={**body, "track_total_hits": 2})

        monkeypatch.setattr(indexer.opensearch_client, "search", counting_to_the_window)
    prefixes = harness.config.indexing.opensearch
    # Written in reverse id order: the pages follow the id order only if the
    # search sorts by id (they continue past the last id of a page).
    claims = [
        claim.model_copy(update={"id": f"cl-hub-{n}"})
        for n in reversed(range(5))
        for claim in _claims(None)[:1]
    ]
    for claim in claims:
        _write(harness, "claims", "upsert", [claim])
    wanted = sorted({u for c in claims for u in c.text_unit_ids or []})

    found = indexer.find_documents(
        prefixes.claims_index_prefix, "text_unit_ids", wanted, "default"
    )

    assert {i: sorted(units) for i, units in found.items()} == {
        c.id: sorted(c.text_unit_ids or []) for c in claims
    }


def test_find_documents_reads_past_ten_thousand_matches(harness: StoreHarness) -> None:
    # OpenSearch counts a search's matches up to 10,000 unless asked for
    # more: a lookup that paged only on a larger total stopped there.
    if harness.kind == "fake":
        pytest.skip("the fake counts no hits")
    indexer = harness.vector_indexer
    (claim,) = _claims(None)
    claims = [claim.model_copy(update={"id": f"cl-hub-{n:05d}"}) for n in range(10_001)]
    _write(harness, "claims", "upsert", claims)

    found = indexer.find_documents(
        harness.config.indexing.opensearch.claims_index_prefix,
        "subject_id",
        ["e-vendor"],
        "default",
    )

    assert len(found) == 10_001


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
@pytest.mark.parametrize("later_window", [3, 10_000, 20_000])
def test_reads_follow_the_window_of_the_live_index(
    harness: StoreHarness, later_window: int
) -> None:
    # Indices created with max_result_window 3, then read under the same
    # window, the default one (the setting reverted) and a raised one (not
    # reindexed): the reads by id and by field sized their searches by the
    # config, or by a fixed 1,000 ids, so every one asked for more hits than
    # the live index returns (a 400), and an incremental run failed for good.
    import asyncio

    from unified_kg_rag.domain.models import SearchQuery, SearchType

    built = _windowed(harness, 3)
    tenant = built.suffix("window")
    ids = [f"t{n}" for n in range(7)]
    units = _tagged(
        [TextUnit(id=i, text=f"Vendor ships widget lot {i}.") for i in ids], tenant
    )
    claims = [
        claim.model_copy(update={"id": f"cl-{n}"})
        for n in range(7)
        for claim in _claims(tenant)
    ]
    _write(built, "text_units", "index", units, suffixes=[tenant])
    _write(built, "claims", "index", claims, suffixes=[tenant])
    later = _windowed(harness, later_window)
    prefixes = later.config.indexing.opensearch

    assert set(later.vector_indexer.read_text_unit_attributes(ids, tenant)) == set(ids)
    found = later.vector_indexer.find_documents(
        prefixes.claims_index_prefix, "subject_id", ["e-vendor"], tenant
    )
    assert len(found or {}) == 7

    assert later.vector_retriever is not None

    async def fetch_by_id() -> tuple[int, int]:
        retriever = later.vector_retriever()
        try:
            limits = await retriever.aindex_limits(
                prefixes.text_units_index_prefix, tenant
            )
            hits = await retriever.aretrieve(
                SearchQuery(
                    query="",
                    search_type=SearchType.LEXICAL,
                    top_k=len(ids),
                    retrieval_multiplier=1,
                    index_prefixes=[prefixes.text_units_index_prefix],
                    suffix=tenant,
                    filters={"id": ids},
                    source_fields=[],
                )
            )
        finally:
            await retriever.aclose()
        return limits.id_batch_size, len(hits)

    # A fetch by id of more ids than one search returns goes in batches of
    # IndexLimits.id_batch_size and returns every one, rather than failing
    # the query or keeping a window's worth.
    assert asyncio.run(fetch_by_id()) == (3, 7)


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_a_retriever_follows_an_index_rebuilt_with_a_smaller_window(
    harness: StoreHarness,
) -> None:
    # A retriever read each index's limits once for its life: after a reset
    # rebuilt the index with a smaller max_result_window, its searches asked
    # for the old window's worth and failed (a 400) until the process
    # restarted. The rejection now drops the limits read and searches again.
    import asyncio

    from unified_kg_rag.domain.models import SearchQuery, SearchType

    harness = harness.derive(
        lambda config: setattr(config.indexing.opensearch, "max_query_size", 5)
    )
    tenant = harness.suffix("rebuilt")
    _write(harness, "entities", "index", _entities(tenant))
    small = _windowed(harness, 1)
    query = SearchQuery(
        query="",
        search_type=SearchType.LEXICAL,
        top_k=5,
        retrieval_multiplier=1,
        index_prefixes=[harness.config.indexing.opensearch.entities_index_prefix],
        suffix=tenant,
    )

    async def search_before_and_after() -> tuple[int, int]:
        retriever = harness.vector_retriever()
        try:
            before = len(await retriever.aretrieve(query))
            await asyncio.to_thread(small.vector_indexer.clear, [tenant])
            await asyncio.to_thread(
                _write, small, "entities", "index", _entities(tenant)
            )
            return before, len(await retriever.aretrieve(query))
        finally:
            await retriever.aclose()

    before, after = asyncio.run(search_before_and_after())
    assert (before, after) == (len(_entities(tenant)), 1)


def _with_settings(harness: StoreHarness, **settings: int) -> StoreHarness:
    def apply(config: Config) -> None:
        config.indexing.opensearch.index_settings = {
            **config.indexing.opensearch.index_settings,
            **settings,
        }

    return harness.derive(apply)


def _terms_lists(body: Any) -> list[int]:
    """The length of every ``terms`` and ``ids`` list in a search body."""
    if isinstance(body, dict):
        found = [
            len(v)
            for k, clause in body.items()
            if k in ("terms", "ids") and isinstance(clause, dict)
            for v in clause.values()
            if isinstance(v, list)
        ]
        return found + [n for v in body.values() for n in _terms_lists(v)]
    if isinstance(body, list):
        return [n for v in body for n in _terms_lists(v)]
    return []


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_reads_and_searches_follow_the_live_max_terms_count(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Text units and claims indexed with max_terms_count 2, entities with
    # the default; read under the default config (the setting reverted, no
    # reindex). OpenSearch refuses a terms clause past the live count (a
    # 400): the reads by field batch under it, a fetch by id goes in batches
    # of that many ids, and a search filter listing more fails up front,
    # also when another index searched beside it takes more.
    import asyncio

    from unified_kg_rag.domain.models import SearchQuery, SearchType
    from unified_kg_rag.shared.exceptions import InvalidFilterError

    built = _with_settings(harness, max_terms_count=2)
    tenant = built.suffix("terms")
    ids = [f"t{n}" for n in range(5)]
    units = _tagged(
        [TextUnit(id=i, text=f"Vendor ships widget lot {i}.") for i in ids], tenant
    )
    claims = [
        claim.model_copy(update={"id": f"cl-{n}", "subject_id": f"s{n}"})
        for n in range(5)
        for claim in _claims(tenant)
    ]
    _write(built, "text_units", "index", units, suffixes=[tenant])
    _write(built, "claims", "index", claims, suffixes=[tenant])
    _write(harness, "entities", "index", _entities(tenant), suffixes=[tenant])
    prefixes = harness.config.indexing.opensearch
    indexer = harness.vector_indexer
    asked: list[int] = []
    search = indexer.opensearch_client.search

    def recorded(index: str, body: dict[str, Any]) -> Any:
        asked.extend(_terms_lists(body))
        return search(index=index, body=body)

    monkeypatch.setattr(indexer.opensearch_client, "search", recorded)

    found = indexer.find_documents(
        prefixes.claims_index_prefix, "subject_id", [f"s{n}" for n in range(5)], tenant
    )
    assert len(found or {}) == 5
    assert asked and max(asked) <= 2

    async def fetch_by_id() -> tuple[int, int, int]:
        retriever = harness.vector_retriever()
        try:
            limits = await retriever.aindex_limits(
                prefixes.text_units_index_prefix, tenant
            )
            hits = await retriever.aretrieve(
                SearchQuery(
                    query="",
                    search_type=SearchType.LEXICAL,
                    top_k=len(ids),
                    retrieval_multiplier=1,
                    index_prefixes=[prefixes.text_units_index_prefix],
                    suffix=tenant,
                    filters={"id": ids},
                    source_fields=[],
                )
            )
            with pytest.raises(InvalidFilterError, match="max_terms_count"):
                await retriever.aretrieve(
                    SearchQuery(
                        query="widget",
                        search_type=SearchType.LEXICAL,
                        top_k=3,
                        retrieval_multiplier=1,
                        index_prefixes=[
                            prefixes.entities_index_prefix,
                            prefixes.text_units_index_prefix,
                        ],
                        suffix=tenant,
                        filters={"type": ["a", "b", "c"]},
                    )
                )
        finally:
            await retriever.aclose()
        return limits.max_terms_count, limits.id_batch_size, len(hits)

    assert asyncio.run(fetch_by_id()) == (2, 2, 5)


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_a_hybrid_search_stays_inside_the_live_window(harness: StoreHarness) -> None:
    # A hybrid search asks each leg for the normalization window (50 hits)
    # at least: under an index created with max_result_window 3 and searched
    # under the default config, past the live window it failed (a 400).
    import asyncio

    from unified_kg_rag.domain.models import SearchQuery, SearchType

    built = _with_settings(harness, max_result_window=3)
    tenant = built.suffix("hybrid")
    units = _tagged(
        [TextUnit(id=f"t{n}", text=f"Vendor ships widget lot {n}.") for n in range(6)],
        tenant,
    )
    _write(built, "text_units", "index", units, suffixes=[tenant])
    prefix = harness.config.indexing.opensearch.text_units_index_prefix

    async def search() -> int:
        retriever = harness.vector_retriever()
        try:
            hits = await retriever.aretrieve(
                SearchQuery(
                    query="Vendor widget",
                    search_type=SearchType.HYBRID,
                    top_k=2,
                    retrieval_multiplier=1,
                    index_prefixes=[prefix],
                    suffix=tenant,
                )
            )
        finally:
            await retriever.aclose()
        return len(hits)

    assert asyncio.run(search()) == 2


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_a_configured_limit_below_the_live_one_wins(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The limits in force are the smaller of the configured and the live
    # ones: an index created with the defaults, read and searched under a
    # smaller configured window and terms count.
    import asyncio

    tenant = harness.suffix("smaller")
    ids = [f"t{n}" for n in range(7)]
    units = _tagged(
        [TextUnit(id=i, text=f"Vendor ships widget lot {i}.") for i in ids], tenant
    )
    _write(harness, "text_units", "index", units, suffixes=[tenant])
    later = _with_settings(harness, max_result_window=4, max_terms_count=3)
    prefix = later.config.indexing.opensearch.text_units_index_prefix
    indexer = later.vector_indexer
    asked: list[int] = []
    search = indexer.opensearch_client.search

    def recorded(index: str, body: dict[str, Any]) -> Any:
        asked.extend(_terms_lists(body))
        return search(index=index, body=body)

    monkeypatch.setattr(indexer.opensearch_client, "search", recorded)
    assert set(indexer.read_text_unit_attributes(ids, tenant)) == set(ids)
    assert asked and max(asked) <= 3

    async def limits() -> tuple[int, int]:
        retriever = later.vector_retriever()
        try:
            found = await retriever.aindex_limits(prefix, tenant)
        finally:
            await retriever.aclose()
        return found.max_result_window, found.max_terms_count

    assert asyncio.run(limits()) == (4, 3)


def _misspelled_setting(config: Config) -> None:
    config.indexing.opensearch.index_settings = {"max_result_windw": 20_000}


def _lucene_ivf(config: Config) -> None:
    # lucene has no ivf method on any OpenSearch version (faiss with
    # cosinesimil, refused before 2.19, is checked per version in
    # test_opensearch_vector_storage.py).
    config.indexing.opensearch.vector_search = {
        **config.indexing.opensearch.vector_search,
        "engine": "lucene",
        "name": "ivf",
    }


def _missing_analyzer(config: Config) -> None:
    config.indexing.opensearch.language_analyzers = {"en": "no_such_analyzer"}


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
@pytest.mark.parametrize(
    ("change", "named"),
    [
        (_misspelled_setting, "max_result_windw"),
        # The probe index had no fields: these passed it and failed the
        # first index creation after extraction.
        (_lucene_ivf, "ivf"),
        (_missing_analyzer, "no_such_analyzer"),
    ],
    ids=["setting", "vector-method", "analyzer"],
)
def test_index_settings_the_engine_refuses_fail_before_extraction(
    harness: StoreHarness, change, named: str
) -> None:
    # A misspelled setting failed the first index creation, the extraction
    # records' write after every unit was extracted (so none was saved);
    # the run now checks index creation before its first model call.
    typo = harness.derive(change)
    tenant = typo.suffix("typo")
    # The namespace exists (a reset or an incremental run): its records
    # index is not created again, and the settings are checked all the same.
    harness.vector_indexer.prepare_extraction_records(tenant)

    if harness.kind == "sqlite":
        # The embedded engine applies max_result_window only: nothing to refuse.
        typo.vector_indexer.check_index_creation(tenant)
        return
    with pytest.raises(ConfigurationError, match=named):
        typo.vector_indexer.check_index_creation(tenant)


def test_index_settings_reach_the_index(harness: StoreHarness) -> None:
    # Retrieval sizes a fetch by id by index_settings.max_result_window: the
    # index must be created with it. Only three keys reached the index, so a
    # raised window made every such search rejected.
    if harness.kind == "fake":
        pytest.skip("the fake keeps no index settings")
    window = 20_000

    def raise_window(config: Config) -> None:
        config.indexing.opensearch.index_settings = {
            **config.indexing.opensearch.index_settings,
            "max_result_window": window,
        }

    raised = harness.derive(raise_window)
    suffix = harness.suffix("window")
    _write(raised, "entities", "index", _entities(suffix))
    indexer = raised.vector_indexer
    alias = indexer._get_name(
        raised.config.indexing.opensearch.entities_index_prefix, suffix
    )

    response = indexer.opensearch_client.search(
        index=alias, body={"size": window, "query": {"match_all": {}}}
    )

    assert len(response["hits"]["hits"]) == 3


def test_text_unit_attributes_round_trip_in_a_suffix(harness: StoreHarness) -> None:
    indexer = harness.vector_indexer
    tenant = harness.suffix("tenant")
    keys = {"extraction_records": {"graph": "fp:hash-1", "claims": "fp2:hash-1"}}
    units = _tagged(
        [TextUnit(id="t1", text="Vendor supplies widgets.", attributes=keys)], tenant
    )
    _write(harness, "text_units", "upsert", units)

    read = indexer.read_text_unit_attributes(["t1", "t-none"], suffix=tenant)
    assert set(read) == {"t1"}
    assert read["t1"]["extraction_records"] == keys["extraction_records"]
    assert indexer.read_text_unit_attributes(["t1"], suffix="default") == {}


def test_extraction_records_are_written_read_and_deleted_per_suffix(
    harness: StoreHarness,
) -> None:
    indexer = harness.vector_indexer
    tenant = harness.suffix("tenant")
    stats = indexer.write_extraction_records(
        {"k1": {"entities": [1]}, "k2": {}}, suffix=tenant
    )
    assert (stats.total_items, stats.successful_items, stats.failed_items) == (2, 2, 0)
    indexer.write_extraction_records({"k1": {"entities": [2]}}, suffix=tenant)

    assert indexer.read_extraction_records(["k1", "k2", "k3"], suffix=tenant) == {
        "k1": {"entities": [2]},
        "k2": {},
    }
    assert indexer.read_extraction_records(["k1"], suffix="default") == {}
    deleted = indexer.delete_extraction_records(["k1", "k9"], suffix=tenant)
    assert deleted.failed_items == 0
    assert indexer.read_extraction_records(["k1", "k2"], suffix=tenant) == {"k2": {}}


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_the_build_record_is_read_back_by_the_retriever(
    harness: StoreHarness,
) -> None:
    import asyncio

    from unified_kg_rag.domain.ingestion.cache_keys import extraction_settings
    from unified_kg_rag.domain.models import IndexBuildRecord

    _write(harness, "text_units", "index", _text_units(None), suffixes=["default"])
    record = IndexBuildRecord(
        pipeline_id="run-1",
        mode="full",
        built_at=datetime(2026, 1, 1, tzinfo=UTC),
        documents=2,
        extraction_fingerprint="fp",
        extraction_settings=extraction_settings(harness.config),
        package_version="0.0.0",
    )
    harness.vector_indexer.record_build(record, "default")

    assert harness.vector_retriever is not None
    read = asyncio.run(harness.vector_retriever().aread_build_record("default"))
    assert read is not None
    assert (read["pipeline_id"], read["documents"], read["mode"]) == (
        "run-1",
        2,
        "full",
    )


def test_deleted_communities_leave_the_text_units_naming_them(
    harness: StoreHarness,
) -> None:
    indexer = harness.vector_indexer
    tenant = harness.suffix("s1")
    t1, t2 = _text_units()
    _write(
        harness,
        "text_units",
        "index",
        [
            t1.model_copy(update={"community_ids": ["c1", "c2"]}),
            t2.model_copy(update={"community_ids": ["c1"]}),
        ],
    )
    _write(
        harness,
        "text_units",
        "upsert",
        _tagged([t1.model_copy(update={"id": "t9", "community_ids": ["c1"]})], tenant),
    )
    prefix = indexer.opensearch_config.text_units_index_prefix

    stats = indexer.prune_text_unit_communities(["c1"], "default")

    assert stats.failed_items == 0, stats.errors
    assert not indexer.find_documents(prefix, "community_ids", ["c1"], "default")
    assert set(indexer.find_documents(prefix, "community_ids", ["c2"], "default")) == {
        "t1"
    }
    assert set(indexer.find_documents(prefix, "community_ids", ["c1"], tenant)) == {
        "t9"
    }


@pytest.mark.parametrize("kind", ["entities", "relationships"])
def test_holds_any_item_tells_a_suffix_with_items(
    harness: StoreHarness, kind: str
) -> None:
    indexer = harness.vector_indexer
    tenant, other = harness.suffix("held"), harness.suffix("none")
    assert indexer.holds_any_item(suffix=tenant) is False
    items = _entities(tenant) if kind == "entities" else _relationships(tenant)
    _write(harness, kind, "upsert", items)
    assert indexer.holds_any_item(suffix=tenant) is True
    assert indexer.holds_any_item(suffix=other) is False
    assert indexer.holds_any_text_unit(suffix=tenant) is False


def test_holds_any_text_unit_tells_a_suffix_with_text_units(
    harness: StoreHarness,
) -> None:
    # Text units alone (a corpus that stated no item): holds_any_item does
    # not see them, holds_any_text_unit does.
    indexer = harness.vector_indexer
    tenant, other = harness.suffix("units"), harness.suffix("none")
    assert indexer.holds_any_text_unit(suffix=tenant) is False
    _write(harness, "text_units", "upsert", _text_units(tenant))
    assert indexer.holds_any_text_unit(suffix=tenant) is True
    assert indexer.holds_any_item(suffix=tenant) is False
    assert indexer.holds_any_text_unit(suffix=other) is False


@pytest.mark.parametrize("kind", ["text_units", "entities"])
def test_an_emptied_index_holds_nothing(harness: StoreHarness, kind: str) -> None:
    # A live index that a full build emptied exists but holds no document:
    # neither probe reads it as holding one.
    indexer = harness.vector_indexer
    tenant = harness.suffix("emptied")
    items = _text_units(tenant) if kind == "text_units" else _entities(tenant)
    _write(harness, kind, "upsert", items)
    _write(harness, kind, "index", [], suffixes=[tenant])
    assert harness.stored_ids(kind, tenant) == set()
    assert indexer.holds_any_text_unit(suffix=tenant) is False
    assert indexer.holds_any_item(suffix=tenant) is False


def test_a_build_record_reads_back_what_the_namespace_holds(
    harness: StoreHarness,
) -> None:
    # Ingestion reads the record before a run (an artifact it lacks makes an
    # incremental run rebuild) and the chain before a query (a strategy whose
    # methodology it does not hold is refused): both read what record_build
    # wrote, on every backend (docs/design.md#46-methodology-profiles).
    light = (
        IndexArtifact.TEXT_UNITS,
        IndexArtifact.ENTITIES,
        IndexArtifact.RELATIONSHIPS,
        IndexArtifact.RELATIONSHIP_VECTORS,
        IndexArtifact.GRAPH,
    )
    full = (*light, IndexArtifact.COMMUNITIES, IndexArtifact.COMMUNITY_REPORTS)
    indexer = harness.vector_indexer
    assert indexer.read_build_record("default") is None
    _write(harness, "text_units", "index", _text_units())

    def record(mode: str, artifacts: tuple[IndexArtifact, ...]) -> IndexBuildRecord:
        return IndexBuildRecord(
            pipeline_id=f"run-{mode}",
            mode=mode,
            built_at=datetime.now(UTC),
            documents=2,
            extraction_fingerprint="fp",
        ).with_artifacts(artifacts)

    written = indexer.record_build(record("full", full), "default")
    assert written is not None and written.methodologies == list(Methodology)[:3]
    stored = indexer.read_build_record("default")
    assert stored is not None
    assert stored["methodologies"] == ["graphrag", "lightrag", "hipporag"]
    # An incremental run that stopped building communities keeps what both
    # name: graphrag leaves the record.
    indexer.record_build(record("incremental", light), "default")
    stored = indexer.read_build_record("default")
    assert stored is not None
    assert stored["artifacts"] == [a.value for a in light]
    assert stored["methodologies"] == ["lightrag", "hipporag"]
    if harness.vector_retriever is not None:
        retriever = harness.vector_retriever()

        async def read() -> dict[str, Any] | None:
            try:
                return await retriever.aread_build_record("default")
            finally:
                await retriever.aclose()

        assert asyncio.run(read()) == stored


def test_scan_items_streams_each_kind_in_id_order_without_vectors(
    harness: StoreHarness,
) -> None:
    _write(harness, "text_units", "index", _text_units())
    _write(harness, "entities", "index", _entities())
    indexer = harness.vector_indexer

    pages = indexer.scan_items("entities", "default", page_size=2)
    assert pages is not None
    first = next(pages)
    # One page at a time: the second is read only when asked for.
    assert [item["id"] for item in first] == ["e-buyer", "e-depot"]
    assert [item["id"] for page in pages for item in page] == ["e-vendor"]
    vendor = next(
        item
        for page in indexer.scan_items("entities", "default")
        for item in page
        if item["id"] == "e-vendor"
    )
    assert vendor["name"] == "Vendor"
    assert vendor["text_unit_ids"] == ["t1", "t2"]
    assert not [key for key in vendor if key.endswith("_embedding")]
    units = [i for p in indexer.scan_items("text_units", "default") for i in p]
    assert [u["id"] for u in units] == ["t1", "t2"]
    assert units[0]["text"].startswith("Vendor supplies widgets")
    # A kind without an index yields nothing.
    assert list(indexer.scan_items("claims", "default")) == []
    assert list(indexer.scan_items("entities", "no_such_suffix")) == []


def test_describe_indices_counts_each_live_index(harness: StoreHarness) -> None:
    _write(harness, "text_units", "index", _text_units())
    _write(harness, "entities", "index", _entities())

    described = {
        entry["kind"]: entry
        for entry in harness.vector_indexer.describe_indices("default")
    }

    assert set(described) == set(STORED_ITEM_KINDS)
    assert described["text_units"]["documents"] == 2
    assert described["entities"]["documents"] == 3
    assert described["claims"]["index"] is None
    if INDEX_ADMIN in harness.capabilities:
        # The embedding settings the index was built with.
        meta = described["entities"]["meta"]
        assert isinstance(meta["embedding_dimension"], int)
        assert "embedding_model_id" in meta


def _report_with_findings() -> CommunityReport:
    return _reports()[0].model_copy(
        update={
            "findings": [
                CommunityFinding(
                    summary="Vendor supplies Buyer",
                    explanation="Vendor supplies widgets to Buyer.",
                    entity_ids=["e-vendor", "e-buyer"],
                    relationship_ids=["r-supplies"],
                ),
                CommunityFinding(
                    summary="Depot stores widgets",
                    explanation="Vendor ships widgets from the Depot.",
                    entity_ids=["e-depot", "e-vendor"],
                    relationship_ids=["r-ships"],
                ),
            ]
        }
    )


def _report_hits(harness: StoreHarness) -> list[RetrievalResult]:
    prefix = harness.config.indexing.opensearch.community_reports_index_prefix
    (hits,) = harness.retrieve(
        RetrieverRole.DOCUMENT,
        SearchQuery(
            query="Vendor widgets",
            search_type=SearchType.LEXICAL,
            top_k=5,
            retrieval_multiplier=1,
            index_prefixes=[prefix],
        ),
    )
    return hits


@pytest.mark.parametrize("harness", backends(RETRIEVERS), indirect=True)
@pytest.mark.parametrize("mode", ["index", "upsert"])
def test_a_report_hit_names_the_records_its_findings_cite(
    harness: StoreHarness, mode: str
) -> None:
    # The findings' [Data: ...] references were resolved to record ids and
    # then never stored: a retrieved report could not say what it rests on.
    _write(harness, "community_reports", mode, [_report_with_findings()])
    (hit,) = _report_hits(harness)
    assert report_cited_ids(hit.metadata) == (
        ["e-vendor", "e-buyer", "e-depot"],
        ["r-supplies", "r-ships"],
    )
    assert [f["summary"] for f in hit.metadata[REPORT_FINDINGS_FIELD]] == [
        "Vendor supplies Buyer",
        "Depot stores widgets",
    ]


@pytest.mark.parametrize("harness", backends(RETRIEVERS), indirect=True)
def test_a_re_key_repoints_the_stored_reports_findings(harness: StoreHarness) -> None:
    # An incremental run re-keys "e-vendor" (its name changed) and deletes
    # the old id: the stored reports citing it cite the new id instead.
    _write(harness, "community_reports", "upsert", [_report_with_findings()])

    stats = harness.vector_indexer.repoint_report_findings(
        {"e-vendor": "e-vendor-inc", "r-ships": "r-ships-2", "e-absent": "e-x"},
        "default",
    )

    assert (stats.successful_items, stats.failed_items) == (1, 0)
    (hit,) = _report_hits(harness)
    assert report_cited_ids(hit.metadata) == (
        ["e-vendor-inc", "e-buyer", "e-depot"],
        ["r-supplies", "r-ships-2"],
    )
    # Nothing left to repoint, and another suffix's reports are not read.
    again = harness.vector_indexer.repoint_report_findings(
        {"e-vendor": "e-vendor-inc"}, "default"
    )
    assert (again.total_items, again.failed_items) == (0, 0)
    other = harness.vector_indexer.repoint_report_findings(
        {"e-vendor-inc": "e-y"}, "other"
    )
    assert (other.total_items, other.failed_items) == (0, 0)


@pytest.mark.parametrize("harness", backends(INDEX_ADMIN), indirect=True)
def test_incremental_run_maps_the_findings_of_an_earlier_reports_index(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer = harness.vector_indexer
    client = indexer.opensearch_client
    prefix = harness.config.indexing.opensearch.community_reports_index_prefix
    current_mapping = indexer._get_community_reports_mapping
    current_doc = indexer._prepare_community_report_doc

    def earlier_mapping() -> dict[str, Any]:
        mapping = current_mapping()
        del mapping["mappings"]["properties"][REPORT_FINDINGS_FIELD]
        return mapping

    def earlier_doc(report: CommunityReport, embeddings: Any) -> dict[str, Any]:
        doc = current_doc(report, embeddings)
        doc.pop(REPORT_FINDINGS_FIELD)
        return doc

    monkeypatch.setattr(indexer, "_get_community_reports_mapping", earlier_mapping)
    monkeypatch.setattr(indexer, "_prepare_community_report_doc", earlier_doc)
    _write(harness, "community_reports", "index", _reports())
    monkeypatch.undo()
    # A report stored before cites nothing.
    (hit,) = _report_hits(harness)
    assert report_cited_ids(hit.metadata) == ([], [])

    name = client.get_index_name_by_alias(indexer._get_name(prefix, None))
    indexer.check_index_settings("default")  # the run's preflight maps it
    assert client.get_index_mappings(name)["properties"][REPORT_FINDINGS_FIELD] == {
        "type": "object",
        "enabled": False,
    }
    _write(harness, "community_reports", "upsert", [_report_with_findings()])
    (hit,) = _report_hits(harness)
    assert report_cited_ids(hit.metadata)[1] == ["r-supplies", "r-ships"]
