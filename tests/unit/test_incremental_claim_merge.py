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
    _remap_references,
    build_document_lineage,
)
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.claim_resolver import (
    ClaimResolver,
    resolved_claim_id,
)
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
    )


def _claim(path: str, suffix: str = "default") -> Claim:
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
        attributes={"index": suffix},
    )


def _extract(paths: list[str], suffix: str = "default"):
    attributes = {"index": suffix}
    units = [
        TextUnit(
            id=f"tu-{p}",
            text=f"Vendor supplies Buyer ({p})",
            document_ids=[p],
            attributes=attributes,
        )
        for p in paths
    ]
    entities = [
        Entity(
            id=f"e-{n.lower()}",
            name=n,
            text_unit_ids=[u.id for u in units],
            attributes=attributes,
        )
        for n in ("Vendor", "Buyer")
    ]
    return units, entities, [_claim(p, suffix) for p in paths]


def _run(store, manager, corpus: list[str], suffix: str = "default") -> None:
    # A fresh indexer per run, as each ingestion run builds one.
    inc = IncrementalIndexer(store, manager, suffix=suffix)
    docs = [_doc(p) for p in corpus]
    delta, fingerprints = detect_delta(docs, store)
    inc.write_ahead(delta, [])
    assert inc.remove_changed_and_deleted(delta)
    todo = [d for d in docs if document_doc_id(d) in set(delta.to_process)]
    if not todo:
        return
    units, entities, claims = _extract([d.file_path for d in todo], suffix)
    lineages = build_document_lineage(todo, units, suffix=suffix)
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


@pytest.mark.parametrize("suffix", ["default", "acme"])
def test_deleting_a_document_strips_its_text_units_from_a_shared_claim(
    stores, suffix: str
) -> None:
    # Under a non-default suffix (document_parsing.index_value) the claim
    # read back must be written back to that suffix's index, not the
    # default one.
    store, manager, vector = stores
    _run(store, manager, ["a.txt"], suffix)
    _run(store, manager, ["a.txt", "b.txt"], suffix)
    _run(store, manager, ["a.txt"], suffix)

    (stored,) = vector.data["claims"].values()
    assert stored.text_unit_ids == ["tu-a.txt"]
    assert stored.attributes == {"index": suffix}


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


def test_merge_claims_status_matches_the_resolver_for_two_claims() -> None:
    # The docstring's claim: for one stored and one delta claim the stored
    # status is ClaimResolver's most-common rule (the first on a tie).
    stored = _claim("a.txt").model_copy(update={"status": "SUSPECTED"})
    delta = _claim("b.txt")
    resolver = ClaimResolver(Config(), use_process_pool=False, show_progress=False)

    (merged,) = merge_claims([stored], [delta])

    assert merged.status == resolver._merge_claims([stored, delta]).status


def test_merge_claims_unions_document_filters() -> None:
    stored = _claim("a.txt").model_copy(
        update={"attributes": {"index": "default", "filters": {"tenant": "a"}}}
    )
    delta = _claim("b.txt").model_copy(
        update={"attributes": {"index": "default", "filters": {"tenant": "b"}}}
    )

    (merged,) = merge_claims([stored], [delta])

    assert merged.attributes == {"index": "default", "filters": {"tenant": ["a", "b"]}}


def test_remapped_claims_are_rekeyed_and_merged() -> None:
    # Two delta claims whose subjects ("Acme" and "Acme Inc", separate
    # entities in this run) both merged into the stored entity e-acme: they
    # become the one claim the full build would have, under its id.
    def delta(subject_id: str, unit: str) -> Claim:
        claim = _claim(unit).model_copy(update={"subject_id": subject_id})
        claim_id = resolved_claim_id(claim)
        return claim.model_copy(update={"id": claim_id, "short_id": claim_id[:8]})

    first, second = delta("e-acme-2", "a.txt"), delta("e-acme-3", "b.txt")
    remap = {"default": {"e-acme-2": "e-acme", "e-acme-3": "e-acme"}}

    _, (claim,) = _remap_references(None, [first, second], remap)

    expected = resolved_claim_id(
        _claim("x").model_copy(update={"subject_id": "e-acme"})
    )
    assert claim.id == expected
    assert claim.subject_id == "e-acme"
    assert claim.text_unit_ids == ["tu-a.txt", "tu-b.txt"]
