# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The same claim in two documents, built incrementally (AWS-free).

A claim's id derives from its subject, object, type and dates, not from its
document, so a second document stating it yields the stored claim's id. The
incremental commit must merge the two as the full build's ``ClaimResolver``
does, and removing one document must strip its text units from the claim.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
)
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.claim_resolver import ClaimResolver
from unified_kg_rag.domain.ingestion.delta_detector import (
    detect_delta,
    document_doc_id,
)
from unified_kg_rag.domain.ingestion.merge import (
    merge_claims,
    remove_claim_text_units,
)
from unified_kg_rag.domain.models import Claim, Config, Document, Entity, TextUnit

pytestmark = pytest.mark.unit

_CLAIM_ID = "claim-vendor-supplies-buyer"


def _doc(path: str) -> Document:
    return Document(
        page_content=f"content {path}",
        document_id=path,
        file_name=path,
        file_path=path,
        file_type="txt",
        total_pages=1,
    )


def _claim(path: str) -> Claim:
    return Claim(
        id=_CLAIM_ID,
        subject_id="e-vendor",
        subject_name="Vendor",
        object_id="e-buyer",
        object_name="Buyer",
        type="SUPPLY",
        status="TRUE",
        description=f"Vendor supplies Buyer per {path}.",
        source_text=f"source {path}",
        text_unit_ids=[f"tu-{path}"],
    )


def _extract(paths: list[str]):
    units = [
        TextUnit(id=f"tu-{p}", text=f"Vendor supplies Buyer ({p})", document_ids=[p])
        for p in paths
    ]
    entities = [
        Entity(id=f"e-{n.lower()}", name=n, text_unit_ids=[u.id for u in units])
        for n in ("Vendor", "Buyer")
    ]
    return units, entities, [_claim(p) for p in paths]


def _run(store, manager, corpus: list[str]) -> None:
    # A fresh indexer per run, as each ingestion run builds one.
    inc = IncrementalIndexer(store, manager)
    docs = [_doc(p) for p in corpus]
    delta, fingerprints = detect_delta(docs, store)
    inc.write_ahead(delta, [])
    assert inc.remove_changed_and_deleted(delta)
    todo = [d for d in docs if document_doc_id(d) in set(delta.to_process)]
    if not todo:
        return
    units, entities, claims = _extract([d.file_path for d in todo])
    lineages = build_document_lineage(todo, units, entities, [], [], claims)
    inc.write_ahead(delta, lineages)
    inc.commit(
        lineages=lineages,
        fingerprints=fingerprints,
        text_units=units,
        entities=entities,
        relationships=[],
        claims=claims,
    )


@pytest.fixture
def stores():
    config = Config()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(
        config=config, vector_indexer=vector, graph_indexer=FakeGraphStore()
    )
    return FakeDocStatusStore(), manager, vector


def test_adding_a_document_merges_its_claim_into_the_stored_one(stores) -> None:
    store, manager, vector = stores
    _run(store, manager, ["a.txt"])
    _run(store, manager, ["a.txt", "b.txt"])

    (stored,) = vector.data["claims"].values()
    _, entities, claims = _extract(["a.txt", "b.txt"])
    (full,), _ = ClaimResolver(
        Config(), use_process_pool=False, show_progress=False
    ).resolve(claims, entities)
    assert stored.text_unit_ids == full.text_unit_ids == ["tu-a.txt", "tu-b.txt"]
    assert stored.description == full.description
    assert stored.source_text == full.source_text


def test_deleting_a_document_strips_its_text_units_from_a_shared_claim(
    stores,
) -> None:
    store, manager, vector = stores
    _run(store, manager, ["a.txt"])
    _run(store, manager, ["a.txt", "b.txt"])
    _run(store, manager, ["a.txt"])

    (stored,) = vector.data["claims"].values()
    assert stored.text_unit_ids == ["tu-a.txt"]


def test_cross_run_merge_off_overwrites_the_stored_claim(stores) -> None:
    store, manager, vector = stores
    manager.config.indexing.cross_run_merge = False
    _run(store, manager, ["a.txt"])
    _run(store, manager, ["a.txt", "b.txt"])

    (stored,) = vector.data["claims"].values()
    assert stored.text_unit_ids == ["tu-b.txt"]


def test_merge_claims_is_idempotent_and_keeps_the_stored_status() -> None:
    stored = _claim("a.txt").model_copy(update={"status": "SUSPECTED"})
    delta = _claim("b.txt")

    (once,) = merge_claims([stored], [delta])
    (twice,) = merge_claims([once], [delta])

    assert once == twice
    assert once.status == "SUSPECTED"
    assert once.description == (
        "Vendor supplies Buyer per a.txt.\nVendor supplies Buyer per b.txt."
    )
    # A delta claim with nothing stored is returned as it is.
    assert merge_claims([], [delta]) == [delta]


def test_remove_claim_text_units_returns_only_the_claims_it_changed() -> None:
    shared = _claim("a.txt").model_copy(update={"text_unit_ids": ["tu-a", "tu-b"]})
    other = _claim("c.txt").model_copy(update={"id": "other"})

    (stripped,) = remove_claim_text_units([shared, other], ["tu-b"])

    assert stripped.id == _CLAIM_ID
    assert stripped.text_unit_ids == ["tu-a"]
    assert shared.text_unit_ids == ["tu-a", "tu-b"]
