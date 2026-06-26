# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial scenario tests for incremental indexing (AWS-free).

Incremental indexing (ADD / MODIFY / DELETE and their messy combinations) is a
headline feature; these tests construct deltas that would plausibly cause data
loss, orphan artifacts, double-counting, or non-idempotency, and pin down how
the pure delta/merge logic plus the :class:`IncrementalIndexer` orchestrator
actually behave under them.

Everything here runs against the in-memory ``FakeDocStatusStore`` + a recording
fake indexing manager and the pure ``merge_*`` / ``detect_delta`` functions —
no AWS clients. Cases that document a SUSPECTED BUG are flagged in-line with a
``SUSPECTED BUG`` comment and assert the *current* behaviour (source is not
modified).
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
)
from unified_kg_rag.domain.ingestion.delta_detector import compute_doc_id
from unified_kg_rag.domain.ingestion.merge import (
    merge_communities,
    merge_community_reports,
    merge_entities,
    merge_relationships,
)
from unified_kg_rag.domain.models import (
    Community,
    CommunityReport,
    Document,
    DocumentLineage,
    Entity,
    Relationship,
    TextUnit,
)
from unified_kg_rag.ports.indexer import IndexingStats

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Test doubles / helpers
# --------------------------------------------------------------------------- #
class FakeIndexingManager:
    """Records index_delta / delete_documents calls instead of touching stores."""

    def __init__(self) -> None:
        self.delta_calls: list[dict] = []
        self.delete_calls: list[dict] = []

    def index_delta(self, **kwargs) -> dict[str, IndexingStats]:
        self.delta_calls.append(kwargs)
        return {}

    def delete_documents(self, ids_by_suffix) -> dict[str, IndexingStats]:
        self.delete_calls.append(ids_by_suffix)
        return {}


def _doc(path: str, text: str, *, metadata: dict | None = None) -> Document:
    return Document(
        page_content=text,
        document_id="x",
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
        file_type="txt",
        total_pages=1,
        metadata=metadata or {},
    )


def _lineage(path: str, **artifacts) -> DocumentLineage:
    return DocumentLineage(doc_id=compute_doc_id(path), **artifacts)


def _entity(id_: str, name: str, **kw) -> Entity:
    return Entity(id=id_, name=name, **kw)


def _all_deleted_ids(manager: FakeIndexingManager) -> list[str]:
    return [
        artifact_id
        for call in manager.delete_calls
        for ids in call.values()
        for artifact_id in ids
    ]


@pytest.fixture
def rig() -> tuple[IncrementalIndexer, FakeDocStatusStore, FakeIndexingManager]:
    store = FakeDocStatusStore()
    manager = FakeIndexingManager()
    return IncrementalIndexer(store, manager), store, manager  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 1. ADD scenarios
# --------------------------------------------------------------------------- #
class TestAdd:
    def test_doc_id_stable_across_runs_for_same_path(self) -> None:
        # doc_id is derived from the path, not the per-run document_id, so the
        # same file is recognised on every run regardless of runtime ids.
        d1 = _doc("/corpus/a.txt", "hello")
        d2 = _doc("/corpus/a.txt", "hello")  # same path, fresh object
        assert compute_doc_id(d1.file_path) == compute_doc_id(d2.file_path)

    def test_path_normalization_collapses_equivalent_paths(self) -> None:
        # ./a.txt and a.txt and backslash variants must map to one doc_id, or a
        # single file would be indexed twice on different OSes / invocations.
        assert compute_doc_id("./corpus/a.txt") == compute_doc_id("corpus/a.txt")
        assert compute_doc_id("corpus\\a.txt") == compute_doc_id("corpus/a.txt")

    def test_new_docs_only_all_classified_new(self, rig) -> None:
        inc, _store, _manager = rig
        docs = [_doc("/a.txt", "A"), _doc("/b.txt", "B"), _doc("/c.txt", "C")]
        delta, fps = inc.plan(docs)
        assert set(delta.new) == set(fps)
        assert not delta.changed and not delta.deleted and not delta.unchanged

    def test_new_entity_merges_with_existing_by_natural_key(self) -> None:
        # ADD a second doc that mentions an entity already in the graph (by
        # name). It must merge into the existing id, not create a duplicate.
        old = [_entity("e1", "Acme Corp", text_unit_ids=["t1"])]
        delta = [_entity("dX", "acme corp", text_unit_ids=["t2"])]
        merged, remap = merge_entities(old, delta)
        assert len(merged) == 1
        assert merged[0].id == "e1"
        assert remap == {"dX": "e1"}
        assert set(merged[0].text_unit_ids) == {"t1", "t2"}

    def test_new_entity_frequency_recomputed_on_merge(self) -> None:
        old = [_entity("e1", "Acme", text_unit_ids=["t1"])]
        delta = [_entity("dX", "Acme", text_unit_ids=["t2", "t3"])]
        merged, _ = merge_entities(old, delta)
        # frequency tracks the count of supporting (deduped) text units.
        assert merged[0].frequency == 3

    def test_brand_new_entity_frequency_not_auto_set(self) -> None:
        # SUSPECTED WEAK SPOT: a brand-new (non-merged) delta entity keeps
        # whatever frequency it arrived with. merge_entities only recomputes
        # frequency for entities that merged by name (merger.py:87). A delta
        # entity with frequency=None and one text unit stays frequency=None,
        # so "frequency == len(text_unit_ids)" is NOT an invariant for new
        # entities. Documenting current behaviour.
        old: list[Entity] = []
        delta = [_entity("dX", "Brand New", text_unit_ids=["t1"])]
        merged, _ = merge_entities(old, delta)
        assert merged[0].frequency is None  # not coerced to len(text_unit_ids)


# --------------------------------------------------------------------------- #
# 2. MODIFY scenarios
# --------------------------------------------------------------------------- #
class TestModify:
    def test_content_change_classified_changed(self, rig) -> None:
        inc, _store, _manager = rig
        docs = [_doc("/a.txt", "original")]
        _, fps = inc.plan(docs)
        inc.commit(
            [_lineage("/a.txt", entity_ids=["e1"])],
            fps,
            entities=[_entity("e1", "A")],
        )

        edited = [_doc("/a.txt", "EDITED CONTENT")]
        delta, _ = inc.plan(edited)
        assert delta.changed == [compute_doc_id("/a.txt")]
        assert not delta.new and not delta.deleted

    def test_modify_prunes_old_exclusive_keeps_shared(self, rig) -> None:
        inc, _store, manager = rig
        a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
        _, fps = inc.plan([a, b])
        inc.commit(
            [_lineage("/a.txt", entity_ids=["shared", "only_a"])],
            fps,
            entities=[_entity("shared", "S"), _entity("only_a", "A")],
        )
        inc.commit(
            [_lineage("/b.txt", entity_ids=["shared"])],
            fps,
            entities=[_entity("shared", "S")],
        )

        edited = [_doc("/a.txt", "EDITED"), b]
        delta, _ = inc.plan(edited)
        inc.prune_changed(delta)

        pruned = _all_deleted_ids(manager)
        assert "only_a" in pruned  # exclusive to changed doc -> pruned
        assert "shared" not in pruned  # survivor /b.txt still references it -> kept

    def test_modify_drops_an_entity_it_used_to_mention(self, rig) -> None:
        # A doc that previously produced e1+e2 is edited to mention only e1.
        # prune_changed removes BOTH old artifacts; commit then re-adds only e1,
        # so e2 does not leak as an orphan after re-extraction.
        inc, _store, manager = rig
        _, fps = inc.plan([_doc("/a.txt", "v1 mentions Bob and Carol")])
        inc.commit(
            [_lineage("/a.txt", entity_ids=["e1", "e2"])],
            fps,
            entities=[_entity("e1", "Bob"), _entity("e2", "Carol")],
        )

        edited = [_doc("/a.txt", "v2 mentions only Bob")]
        delta, fps2 = inc.plan(edited)
        inc.prune_changed(delta)
        pruned = _all_deleted_ids(manager)
        assert "e1" in pruned and "e2" in pruned

        # Re-extract: only e1 survives in the new lineage.
        inc.commit(
            [_lineage("/a.txt", entity_ids=["e1"])],
            fps2,
            entities=[_entity("e1", "Bob")],
        )
        rec = _store.get(compute_doc_id("/a.txt"))
        assert rec is not None and rec.entity_ids == ["e1"]  # e2 no longer tracked

    def test_metadata_only_change_is_noop_delta(self, rig) -> None:
        # content_hash is computed over document TEXT only (delta_detector.py:38).
        # Changing only metadata leaves the hash identical -> no-op delta. This
        # is correct (don't reindex on a metadata touch), pinned here so a future
        # hash change that folds metadata in doesn't silently regress it.
        inc, _store, _manager = rig
        original = _doc("/a.txt", "same text", metadata={"author": "alice"})
        _, fps = inc.plan([original])
        inc.commit(
            [_lineage("/a.txt", entity_ids=["e1"])],
            fps,
            entities=[_entity("e1", "A")],
        )

        touched = _doc("/a.txt", "same text", metadata={"author": "BOB", "tag": "x"})
        delta, _ = inc.plan([touched])
        assert delta.is_empty
        assert delta.unchanged == [compute_doc_id("/a.txt")]

    def test_modified_to_match_another_doc_dedup_collision(self, rig) -> None:
        # Two distinct files; one is edited until its TEXT is identical to the
        # other. They share a content_hash but are DIFFERENT documents (different
        # paths -> different doc_ids), so both remain registered. A content-hash
        # collision must not collapse two files into one registry entry.
        inc, _store, _manager = rig
        a, b = _doc("/a.txt", "unique A"), _doc("/b.txt", "unique B")
        _, fps = inc.plan([a, b])
        inc.commit(
            [
                _lineage("/a.txt", entity_ids=["ea"]),
                _lineage("/b.txt", entity_ids=["eb"]),
            ],
            fps,
            entities=[_entity("ea", "A"), _entity("eb", "B")],
        )

        # Edit b to be textually identical to a.
        b2 = _doc("/b.txt", "unique A")
        delta, fps2 = inc.plan([a, b2])
        assert compute_doc_id("/a.txt") in delta.unchanged
        assert compute_doc_id("/b.txt") in delta.changed
        # Same content hash for the two distinct docs, but two registry entries.
        assert fps2[compute_doc_id("/a.txt")] == fps2[compute_doc_id("/b.txt")]
        assert len({compute_doc_id("/a.txt"), compute_doc_id("/b.txt")}) == 2


# --------------------------------------------------------------------------- #
# 3. DELETE scenarios
# --------------------------------------------------------------------------- #
class TestDelete:
    def test_delete_removes_exclusive_entities_relationships_claims(self, rig) -> None:
        inc, store, manager = rig
        a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
        _, fps = inc.plan([a, b])
        inc.commit(
            [
                _lineage(
                    "/a.txt",
                    entity_ids=["ea"],
                    relationship_ids=["ra"],
                    claim_ids=["cla"],
                )
            ],
            fps,
            entities=[_entity("ea", "A")],
        )
        inc.commit(
            [
                _lineage(
                    "/b.txt",
                    entity_ids=["eb"],
                    relationship_ids=["rb"],
                    claim_ids=["clb"],
                    community_report_ids=["crb"],
                )
            ],
            fps,
            entities=[_entity("eb", "B")],
        )

        delta, _ = inc.plan([a])  # b deleted
        assert delta.deleted == [compute_doc_id("/b.txt")]
        inc.remove_deleted(delta)

        deleted = _all_deleted_ids(manager)
        for exclusive in ("eb", "rb", "clb", "crb"):
            assert exclusive in deleted
        for survivor in ("ea", "ra", "cla"):
            assert survivor not in deleted
        assert {r.doc_id for r in store.list_all()} == {compute_doc_id("/a.txt")}

    def test_delete_preserves_shared_artifacts(self, rig) -> None:
        inc, store, manager = rig
        a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
        _, fps = inc.plan([a, b])
        inc.commit(
            [_lineage("/a.txt", entity_ids=["shared"])],
            fps,
            entities=[_entity("shared", "S")],
        )
        inc.commit(
            [_lineage("/b.txt", entity_ids=["shared"])],
            fps,
            entities=[_entity("shared", "S")],
        )

        delta, _ = inc.plan([a])  # delete b
        inc.remove_deleted(delta)
        assert "shared" not in _all_deleted_ids(manager)
        assert {r.doc_id for r in store.list_all()} == {compute_doc_id("/a.txt")}

    def test_delete_last_referencing_doc_removes_the_entity(self, rig) -> None:
        # Two docs both reference 'shared'. Delete one -> kept (still shared).
        # Delete the second -> now exclusive to it -> removed. Lineage shrinks to
        # one referrer between the two deletes.
        inc, store, manager = rig
        a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
        _, fps = inc.plan([a, b])
        inc.commit(
            [_lineage("/a.txt", entity_ids=["shared"])],
            fps,
            entities=[_entity("shared", "S")],
        )
        inc.commit(
            [_lineage("/b.txt", entity_ids=["shared"])],
            fps,
            entities=[_entity("shared", "S")],
        )

        # Delete a first: 'shared' still referenced by b -> preserved.
        d1, _ = inc.plan([b])
        inc.remove_deleted(d1)
        assert "shared" not in _all_deleted_ids(manager)

        # Delete b (the last referrer): now 'shared' is exclusive -> removed.
        d2, _ = inc.plan([])
        inc.remove_deleted(d2)
        assert "shared" in _all_deleted_ids(manager)
        assert store.list_all() == []

    def test_delete_empty_corpus_removes_everything(self, rig) -> None:
        inc, store, manager = rig
        _, fps = inc.plan([_doc("/a.txt", "A")])
        inc.commit(
            [_lineage("/a.txt", entity_ids=["ea"])],
            fps,
            entities=[_entity("ea", "A")],
        )
        delta, _ = inc.plan([])  # entire corpus gone
        assert delta.deleted == [compute_doc_id("/a.txt")]
        inc.remove_deleted(delta)
        assert store.list_all() == []
        assert "ea" in _all_deleted_ids(manager)


# --------------------------------------------------------------------------- #
# 4. COMBINATIONS / adversarial
# --------------------------------------------------------------------------- #
class TestCombinations:
    def test_add_modify_delete_in_one_delta(self, rig) -> None:
        inc, store, manager = rig
        a, b, c = _doc("/a.txt", "A"), _doc("/b.txt", "B"), _doc("/c.txt", "C")
        _, fps = inc.plan([a, b, c])
        for path, eid in (("/a.txt", "ea"), ("/b.txt", "eb"), ("/c.txt", "ec")):
            inc.commit(
                [_lineage(path, entity_ids=[eid])], fps, entities=[_entity(eid, eid)]
            )

        # Next corpus: a unchanged, b modified, c deleted, d added.
        a2 = _doc("/a.txt", "A")  # unchanged
        b2 = _doc("/b.txt", "B EDITED")  # modified
        d = _doc("/d.txt", "D")  # new
        delta, fps2 = inc.plan([a2, b2, d])

        assert delta.unchanged == [compute_doc_id("/a.txt")]
        assert delta.changed == [compute_doc_id("/b.txt")]
        assert delta.new == [compute_doc_id("/d.txt")]
        assert delta.deleted == [compute_doc_id("/c.txt")]

        # Apply: prune changed, remove deleted, commit new+changed re-extraction.
        inc.prune_changed(delta)
        inc.remove_deleted(delta)
        deleted = _all_deleted_ids(manager)
        assert "eb" in deleted  # b's stale artifact pruned
        assert "ec" in deleted  # c deleted
        assert "ea" not in deleted  # a untouched

        inc.commit(
            [
                _lineage("/b.txt", entity_ids=["eb2"]),
                _lineage("/d.txt", entity_ids=["ed"]),
            ],
            fps2,
            entities=[_entity("eb2", "B2"), _entity("ed", "D")],
        )
        live = {r.doc_id for r in store.list_all()}
        assert live == {compute_doc_id(p) for p in ("/a.txt", "/b.txt", "/d.txt")}

    def test_delete_then_readd_same_doc_id(self, rig) -> None:
        inc, store, manager = rig
        _, fps = inc.plan([_doc("/a.txt", "v1")])
        inc.commit(
            [_lineage("/a.txt", entity_ids=["e_old"])],
            fps,
            entities=[_entity("e_old", "Old")],
        )

        # Delete it.
        del_delta, _ = inc.plan([])
        inc.remove_deleted(del_delta)
        assert store.get(compute_doc_id("/a.txt")) is None

        # Re-add the same path with new content -> classified new again (clean
        # slate), and its registry record reflects only the new artifacts.
        readd, fps2 = inc.plan([_doc("/a.txt", "v2")])
        assert readd.new == [compute_doc_id("/a.txt")]
        inc.commit(
            [_lineage("/a.txt", entity_ids=["e_new"])],
            fps2,
            entities=[_entity("e_new", "New")],
        )
        rec = store.get(compute_doc_id("/a.txt"))
        assert rec is not None and rec.entity_ids == ["e_new"]

    def test_entity_spanning_deleted_and_surviving_doc_survives(self, rig) -> None:
        # An entity whose text_unit_ids span a deleted doc AND a surviving doc
        # must survive; only the deleted doc's registry record/lineage goes away.
        inc, store, manager = rig
        docs = [_doc("/a.txt", "Alice and Bob"), _doc("/b.txt", "Alice again")]
        d1, d2 = docs
        d1.document_id = "rd1"
        d2.document_id = "rd2"
        text_units = [
            TextUnit(id="t1", text="...", document_ids=["rd1"]),
            TextUnit(id="t2", text="...", document_ids=["rd2"]),
        ]
        # 'alice' spans both docs (t1 from a, t2 from b); 'bob' is exclusive to a.
        alice = _entity("e_alice", "Alice", text_unit_ids=["t1", "t2"])
        bob = _entity("e_bob", "Bob", text_unit_ids=["t1"])
        lineages = build_document_lineage(docs, text_units, [alice, bob], [], [], [])
        _, fps = inc.plan(docs)
        inc.commit(lineages, fps, entities=[alice, bob])

        # Both docs recorded 'alice' in their lineage.
        for path in ("/a.txt", "/b.txt"):
            rec = store.get(compute_doc_id(path))
            assert rec is not None and "e_alice" in rec.entity_ids

        # Delete /a.txt -> bob exclusive (removed), alice shared (kept).
        delta, _ = inc.plan([d2])
        inc.remove_deleted(delta)
        deleted = _all_deleted_ids(manager)
        assert "e_bob" in deleted
        assert "e_alice" not in deleted
        # Surviving record's lineage still references alice.
        surviving = store.get(compute_doc_id("/b.txt"))
        assert surviving is not None and "e_alice" in surviving.entity_ids

    def test_relationship_endpoint_entity_deleted_orphan_edge(self, rig) -> None:
        # SUSPECTED WEAK SPOT (orphan handling): a relationship whose endpoint
        # entity gets deleted. Lineage attribution is purely id-set based
        # (incremental.py:_collect_exclusive_artifact_ids) and tracks artifact
        # ids per doc; it does NOT inspect relationship endpoints. If the
        # surviving doc's lineage records the EDGE but the deleted doc owned the
        # endpoint ENTITY exclusively, the entity is removed while the edge
        # survives -> a dangling endpoint. We construct that and assert the edge
        # is NOT removed (current behaviour: orphan edge can occur).
        inc, store, manager = rig
        a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
        _, fps = inc.plan([a, b])
        # Doc a owns entity 'e_target' exclusively. Doc b owns edge 'r1' that
        # points AT e_target. (Plausible when an extraction in b references an
        # entity first seen in a.)
        inc.commit(
            [_lineage("/a.txt", entity_ids=["e_target"])],
            fps,
            entities=[_entity("e_target", "Target")],
        )
        inc.commit(
            [_lineage("/b.txt", entity_ids=["e_src"], relationship_ids=["r1"])],
            fps,
            entities=[_entity("e_src", "Src")],
            relationships=[
                Relationship(id="r1", source_id="e_src", target_id="e_target")
            ],
        )

        delta, _ = inc.plan([b])  # delete a
        inc.remove_deleted(delta)
        deleted = _all_deleted_ids(manager)
        # e_target is removed (exclusive to deleted doc a)...
        assert "e_target" in deleted
        # ...and r1 is NOT in the lineage-derived exclusive id-set (it is owned
        # by surviving doc b). The orphan-edge cascade does not happen at THIS
        # (lineage) layer — it is handled downstream in
        # IndexingManager.delete_documents, which queries Neptune for edges
        # incident to the deleted entities and folds them into the relationship-
        # index deletion (see test_indexing_manager_orphan_cleanup). So at the
        # incremental layer r1 stays out of the exclusive set by design.
        assert "r1" not in deleted

    def test_empty_delta_is_clean_noop(self, rig) -> None:
        inc, _store, manager = rig
        docs = [_doc("/a.txt", "A")]
        _, fps = inc.plan(docs)
        inc.commit(
            [_lineage("/a.txt", entity_ids=["e1"])],
            fps,
            entities=[_entity("e1", "A")],
        )
        before = len(manager.delete_calls)

        delta, _ = inc.plan(docs)  # identical corpus
        assert delta.is_empty
        inc.prune_changed(delta)
        inc.remove_deleted(delta)
        # No prune/remove side effects when nothing changed.
        assert len(manager.delete_calls) == before

    def test_rerunning_same_delta_twice_is_idempotent(self, rig) -> None:
        # Re-running an identical add commit twice must not duplicate entities in
        # the merged graph or grow the registry lineage. Pure merge is the unit
        # of idempotency here.
        inc, store, _manager = rig
        docs = [_doc("/a.txt", "A")]
        _, fps = inc.plan(docs)
        inc.commit(
            [_lineage("/a.txt", entity_ids=["e1"])],
            fps,
            entities=[_entity("e1", "A")],
        )
        inc.commit(
            [_lineage("/a.txt", entity_ids=["e1"])],
            fps,
            entities=[_entity("e1", "A")],
        )
        rec = store.get(compute_doc_id("/a.txt"))
        assert rec is not None and rec.entity_ids == ["e1"]
        assert len(store.list_all()) == 1


# --------------------------------------------------------------------------- #
# 5. Merge-law properties (extending test_merge.py with adversarial cases)
# --------------------------------------------------------------------------- #
class TestMergeLaws:
    def test_entity_merge_idempotent_on_reapplied_delta(self) -> None:
        old = [_entity("e1", "Alice", description="A", text_unit_ids=["t1"])]
        delta = [_entity("dX", "Alice", description="B", text_unit_ids=["t2"])]
        once, _ = merge_entities(old, delta)
        twice, _ = merge_entities(once, delta)
        # Re-applying the same delta does not grow the entity set...
        assert len(once) == len(twice) == 1
        # ...nor double-accumulate text units (dedupe by id).
        assert set(twice[0].text_unit_ids) == {"t1", "t2"}

    def test_relationship_weight_double_counts_on_reapply(self) -> None:
        # SUSPECTED WEAK SPOT (non-idempotent weight): merge_relationships SUMS
        # weights (merger.py:154). Re-applying the SAME delta therefore adds the
        # weight again — merge is NOT idempotent on relationship weight. With
        # text_unit dedupe the text_unit_ids stay stable, but weight drifts. This
        # matters if a pipeline retries a commit. Documenting current behaviour.
        old = [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                weight=1.0,
                text_unit_ids=["t1"],
            )
        ]
        delta = [
            Relationship(
                id="rX",
                source_id="e1",
                target_id="e2",
                weight=2.0,
                text_unit_ids=["t2"],
            )
        ]
        once = merge_relationships(old, delta)
        twice = merge_relationships(once, delta)
        assert once[0].weight == 3.0
        assert twice[0].weight == 5.0  # NOT 3.0 -> re-apply double-counts weight
        assert set(twice[0].text_unit_ids) == {"t1", "t2"}  # tu set stays stable

    def test_merge_entities_order_independent_on_names(self) -> None:
        # Merge is commutative on the resulting NAME set (associativity of the
        # natural-key union), regardless of delta ordering.
        old = [_entity("e1", "Alice", text_unit_ids=["t1"])]
        d_ab = [
            _entity("a", "Bob", text_unit_ids=["t2"]),
            _entity("b", "Carol", text_unit_ids=["t3"]),
        ]
        d_ba = [
            _entity("b", "Carol", text_unit_ids=["t3"]),
            _entity("a", "Bob", text_unit_ids=["t2"]),
        ]
        m1, _ = merge_entities(old, d_ab)
        m2, _ = merge_entities(old, d_ba)
        assert {e.name for e in m1} == {e.name for e in m2} == {"Alice", "Bob", "Carol"}

    def test_communities_reapply_idempotent_no_double_suffix(self) -> None:
        old = [Community(id="c1", name="A", level="0", parent="", children=[])]
        delta = [Community(id="c1", name="B", level="0", parent="", children=[])]
        once = merge_communities(old, delta)
        twice = merge_communities(once, delta)
        thrice = merge_communities(twice, delta)
        # c1-delta must never grow into c1-delta-delta on repeated runs.
        assert [c.id for c in thrice] == ["c1", "c1-delta"]

    def test_community_collision_disambiguated_not_dropped(self) -> None:
        # Regression for the silent-drop bug: a genuine collision must be
        # disambiguated with a counter (-delta, -delta-2, ...), never dropped,
        # even when an earlier delta community was literally named 'c1-delta'.
        existing = [Community(id="c1", name="A", level="0", parent="", children=[])]
        # A delta community literally named 'c1-delta' (not a collision on c1).
        d1 = [
            Community(id="c1-delta", name="Verbatim", level="0", parent="", children=[])
        ]
        merged = merge_communities(existing, d1)
        assert [c.id for c in merged] == ["c1", "c1-delta"]

        # A genuine collision on base 'c1': 'c1-delta' is taken, so it must
        # fall through to 'c1-delta-2' — NOT be silently skipped.
        d2 = [
            Community(id="c1", name="RealCollision", level="0", parent="", children=[])
        ]
        merged2 = merge_communities(merged, d2)
        names = {c.name for c in merged2}
        assert "RealCollision" in names  # no longer dropped
        by_id = {c.id: c.name for c in merged2}
        assert by_id["c1-delta"] == "Verbatim"  # the verbatim one is untouched
        assert by_id["c1-delta-2"] == "RealCollision"  # collision disambiguated

    def test_community_reports_reapply_idempotent(self) -> None:
        old = [CommunityReport(id="cr1", community_id="c1", name="R1")]
        delta = [CommunityReport(id="cr1", community_id="c1", name="R2")]
        once = merge_community_reports(old, delta)
        twice = merge_community_reports(once, delta)
        assert [r.id for r in once] == [r.id for r in twice] == ["cr1", "cr1-delta"]

    def test_lineage_union_then_trim_on_partial_delete(self, rig) -> None:
        # Lineage union (shared artifact attributed to both docs) followed by a
        # trim on partial delete: deleting one referrer must leave the shared id
        # referenced by the survivor (so it is preserved), and the survivor's
        # exclusive-id computation correctly excludes it from removal.
        inc, _store, manager = rig
        docs = [_doc("/a.txt", "A"), _doc("/b.txt", "B")]
        a, b = docs
        a.document_id, b.document_id = "ra", "rb"
        tus = [
            TextUnit(id="t1", text="x", document_ids=["ra"]),
            TextUnit(id="t2", text="x", document_ids=["rb"]),
        ]
        shared = _entity("shared", "S", text_unit_ids=["t1", "t2"])
        lineages = build_document_lineage(docs, tus, [shared], [], [], [])
        # union: both lineages carry 'shared'
        assert all("shared" in ln.entity_ids for ln in lineages)

        _, fps = inc.plan(docs)
        inc.commit(lineages, fps, entities=[shared])
        delta, _ = inc.plan([b])  # delete a; trim shared from removal set
        inc.remove_deleted(delta)
        assert "shared" not in _all_deleted_ids(manager)
