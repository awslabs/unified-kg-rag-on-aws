# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial scenario tests for incremental indexing (AWS-free).

Incremental indexing (ADD / MODIFY / DELETE and their messy combinations) is a
headline feature; these tests construct deltas that would plausibly cause data
loss, orphan artifacts, double-counting, or non-idempotency, and pin down how
the pure delta/merge logic plus the :class:`IncrementalIndexer` orchestrator
actually behave under them.

Everything here runs against the in-memory ``FakeDocStatusStore`` and a real
:class:`IndexingManager` over the in-memory graph and vector stores, plus the
pure ``merge_*`` / ``detect_delta`` functions: no AWS clients. A document's
registry record lists its text units; a removal finds what cites them in the
stores, deletes what only removed text units state and strips the rest.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
)
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.delta_detector import compute_doc_id, detect_delta
from unified_kg_rag.domain.ingestion.merge import (
    merge_entities,
    merge_relationships,
)
from unified_kg_rag.domain.models import (
    Config,
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
def _doc(path: str, text: str, *, metadata: dict | None = None) -> Document:
    return Document(
        page_content=text,
        document_id="x",
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
        metadata=metadata or {},
    )


def _entity(id_: str, name: str, **kw) -> Entity:
    return Entity(id=id_, name=name, **kw)


def _unit_id(path: str, version: str = "") -> str:
    return f"tu-{path.strip('/')}{version}"


class _Rig:
    """Registry + manager over in-memory graph and vector stores."""

    def __init__(self) -> None:
        config = Config()
        self.graph = FakeGraphStore()
        self.vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
        self.manager = IndexingManager(
            config=config, vector_indexer=self.vector, graph_indexer=self.graph
        )
        self.store = FakeDocStatusStore()
        self.inc = IncrementalIndexer(self.store, self.manager)

    def commit(
        self,
        fps: dict[str, str],
        path: str,
        names: dict[str, str],
        *,
        version: str = "",
        relationships: list[Relationship] = (),  # type: ignore[assignment]
    ) -> str:
        """Commit ``path`` as one text unit stating the ``{id: name}``
        entities (and ``relationships``); returns the text unit id."""
        unit_id = _unit_id(path, version)
        self.inc.commit(
            lineages=[
                DocumentLineage(doc_id=compute_doc_id(path), text_unit_ids=[unit_id])
            ],
            fingerprints=fps,
            text_units=[TextUnit(id=unit_id, text="...", document_ids=["x"])],
            entities=[_entity(i, n, text_unit_ids=[unit_id]) for i, n in names.items()],
            relationships=[
                r.model_copy(update={"text_unit_ids": [unit_id]}) for r in relationships
            ],
        )
        return unit_id

    def entity_units(self) -> dict[str, list[str]]:
        entities = self.graph.read_entities(sorted(self.graph.ids("entities")))
        return {e.id: sorted(e.text_unit_ids or []) for e in entities}

    def stored_ids(self) -> set[str]:
        return self.graph.ids("entities") | self.vector.ids("entities")


@pytest.fixture
def rig() -> _Rig:
    return _Rig()


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
        docs = [_doc("/a.txt", "A"), _doc("/b.txt", "B"), _doc("/c.txt", "C")]
        delta, fps = detect_delta(docs, rig.inc.doc_status)
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

    def test_new_entity_text_units_unioned_on_merge(self) -> None:
        old = [_entity("e1", "Acme", text_unit_ids=["t1"])]
        delta = [_entity("dX", "Acme", text_unit_ids=["t2", "t3"])]
        merged, _ = merge_entities(old, delta)
        assert merged[0].text_unit_ids == ["t1", "t2", "t3"]


# --------------------------------------------------------------------------- #
# 2. MODIFY scenarios
# --------------------------------------------------------------------------- #
class TestModify:
    def test_content_change_classified_changed(self, rig) -> None:
        _, fps = detect_delta([_doc("/a.txt", "original")], rig.inc.doc_status)
        rig.commit(fps, "/a.txt", {"e1": "A"})

        edited = [_doc("/a.txt", "EDITED CONTENT")]
        delta, _ = detect_delta(edited, rig.inc.doc_status)
        assert delta.changed == [compute_doc_id("/a.txt")]
        assert not delta.new and not delta.deleted

    def test_modify_prunes_old_exclusive_keeps_shared(self, rig) -> None:
        a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
        _, fps = detect_delta([a, b], rig.inc.doc_status)
        rig.commit(fps, "/a.txt", {"shared": "S", "only_a": "A"})
        rig.commit(fps, "/b.txt", {"shared": "S"})

        edited = [_doc("/a.txt", "EDITED"), b]
        delta, _ = detect_delta(edited, rig.inc.doc_status)
        assert rig.inc.remove_changed_and_deleted(delta)

        # Exclusive to the changed doc -> pruned; survivor /b.txt still states
        # 'shared' -> kept, citing only b's text unit.
        assert rig.entity_units() == {"shared": [_unit_id("/b.txt")]}
        assert rig.vector.ids("entities") == {"shared"}

    def test_modify_drops_an_entity_it_used_to_mention(self, rig) -> None:
        # A doc that previously produced e1+e2 is edited to mention only e1.
        # The removal drops BOTH old artifacts; commit then re-adds only e1,
        # so e2 does not leak as an orphan after re-extraction.
        _, fps = detect_delta(
            [_doc("/a.txt", "v1 mentions Bob and Carol")], rig.inc.doc_status
        )
        rig.commit(fps, "/a.txt", {"e1": "Bob", "e2": "Carol"})

        edited = [_doc("/a.txt", "v2 mentions only Bob")]
        delta, fps2 = detect_delta(edited, rig.inc.doc_status)
        assert rig.inc.remove_changed_and_deleted(delta)
        assert rig.stored_ids() == set()

        # Re-extract: only e1 is stated by the new version.
        new_unit = rig.commit(fps2, "/a.txt", {"e1": "Bob"}, version="-v2")
        assert rig.entity_units() == {"e1": [new_unit]}
        rec = rig.store.get(compute_doc_id("/a.txt"))
        assert rec is not None and rec.text_unit_ids == [new_unit]

    def test_metadata_only_change_is_noop_delta(self, rig) -> None:
        # content_hash is computed over document TEXT only (delta_detector.py).
        # Changing only metadata leaves the hash identical -> no-op delta. This
        # is correct (don't reindex on a metadata touch), pinned here so a future
        # hash change that folds metadata in doesn't silently regress it.
        original = _doc("/a.txt", "same text", metadata={"author": "alice"})
        _, fps = detect_delta([original], rig.inc.doc_status)
        rig.commit(fps, "/a.txt", {"e1": "A"})

        touched = _doc("/a.txt", "same text", metadata={"author": "BOB", "tag": "x"})
        delta, _ = detect_delta([touched], rig.inc.doc_status)
        assert delta.is_empty
        assert delta.unchanged == [compute_doc_id("/a.txt")]

    def test_modified_to_match_another_doc_dedup_collision(self, rig) -> None:
        # Two distinct files; one is edited until its TEXT is identical to the
        # other. They share a content_hash but are DIFFERENT documents (different
        # paths -> different doc_ids), so both remain registered. A content-hash
        # collision must not collapse two files into one registry entry.
        a, b = _doc("/a.txt", "unique A"), _doc("/b.txt", "unique B")
        _, fps = detect_delta([a, b], rig.inc.doc_status)
        rig.commit(fps, "/a.txt", {"ea": "A"})
        rig.commit(fps, "/b.txt", {"eb": "B"})

        b2 = _doc("/b.txt", "unique A")
        delta, fps2 = detect_delta([a, b2], rig.inc.doc_status)
        assert compute_doc_id("/a.txt") in delta.unchanged
        assert compute_doc_id("/b.txt") in delta.changed
        # Same content hash for the two distinct docs, but two registry entries.
        assert fps2[compute_doc_id("/a.txt")] == fps2[compute_doc_id("/b.txt")]
        assert len(rig.store.list_all()) == 2


# --------------------------------------------------------------------------- #
# 3. DELETE scenarios
# --------------------------------------------------------------------------- #
class TestDelete:
    def test_delete_removes_exclusive_entities_relationships(self, rig) -> None:
        a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
        _, fps = detect_delta([a, b], rig.inc.doc_status)
        rig.commit(
            fps,
            "/a.txt",
            {"ea": "A", "ea2": "A2"},
            relationships=[Relationship(id="ra", source_id="ea", target_id="ea2")],
        )
        rig.commit(
            fps,
            "/b.txt",
            {"eb": "B", "eb2": "B2"},
            relationships=[Relationship(id="rb", source_id="eb", target_id="eb2")],
        )

        delta, _ = detect_delta([a], rig.inc.doc_status)  # b deleted
        assert delta.deleted == [compute_doc_id("/b.txt")]
        assert rig.inc.remove_changed_and_deleted(delta)

        assert rig.stored_ids() == {"ea", "ea2"}
        assert rig.graph.ids("relationships") == {"ra"}
        assert rig.vector.ids("relationships") == {"ra"}
        assert rig.vector.ids("text_units") == {_unit_id("/a.txt")}
        assert {r.doc_id for r in rig.store.list_all()} == {compute_doc_id("/a.txt")}

    def test_delete_preserves_shared_artifacts(self, rig) -> None:
        a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
        _, fps = detect_delta([a, b], rig.inc.doc_status)
        rig.commit(fps, "/a.txt", {"shared": "S"})
        rig.commit(fps, "/b.txt", {"shared": "S"})

        delta, _ = detect_delta([a], rig.inc.doc_status)  # delete b
        assert rig.inc.remove_changed_and_deleted(delta)
        assert rig.entity_units() == {"shared": [_unit_id("/a.txt")]}
        assert {r.doc_id for r in rig.store.list_all()} == {compute_doc_id("/a.txt")}

    def test_delete_last_referencing_doc_removes_the_entity(self, rig) -> None:
        # Two docs both state 'shared'. Delete one -> kept (still stated).
        # Delete the second -> nothing states it any more -> removed.
        a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
        _, fps = detect_delta([a, b], rig.inc.doc_status)
        rig.commit(fps, "/a.txt", {"shared": "S"})
        rig.commit(fps, "/b.txt", {"shared": "S"})

        d1, _ = detect_delta([b], rig.inc.doc_status)
        assert rig.inc.remove_changed_and_deleted(d1)
        assert rig.entity_units() == {"shared": [_unit_id("/b.txt")]}

        d2, _ = detect_delta([], rig.inc.doc_status)
        assert rig.inc.remove_changed_and_deleted(d2)
        assert rig.stored_ids() == set()
        assert rig.store.list_all() == []

    def test_delete_empty_corpus_removes_everything(self, rig) -> None:
        _, fps = detect_delta([_doc("/a.txt", "A")], rig.inc.doc_status)
        rig.commit(fps, "/a.txt", {"ea": "A"})
        delta, _ = detect_delta([], rig.inc.doc_status)  # entire corpus gone
        assert delta.deleted == [compute_doc_id("/a.txt")]
        assert rig.inc.remove_changed_and_deleted(delta)
        assert rig.store.list_all() == []
        assert rig.stored_ids() == set()
        assert rig.vector.ids("text_units") == set()


# --------------------------------------------------------------------------- #
# 4. COMBINATIONS / adversarial
# --------------------------------------------------------------------------- #
class TestCombinations:
    def test_add_modify_delete_in_one_delta(self, rig) -> None:
        a, b, c = _doc("/a.txt", "A"), _doc("/b.txt", "B"), _doc("/c.txt", "C")
        _, fps = detect_delta([a, b, c], rig.inc.doc_status)
        for path, eid in (("/a.txt", "ea"), ("/b.txt", "eb"), ("/c.txt", "ec")):
            rig.commit(fps, path, {eid: eid})

        # Next corpus: a unchanged, b modified, c deleted, d added.
        a2 = _doc("/a.txt", "A")
        b2 = _doc("/b.txt", "B EDITED")
        d = _doc("/d.txt", "D")
        delta, fps2 = detect_delta([a2, b2, d], rig.inc.doc_status)

        assert delta.unchanged == [compute_doc_id("/a.txt")]
        assert delta.changed == [compute_doc_id("/b.txt")]
        assert delta.new == [compute_doc_id("/d.txt")]
        assert delta.deleted == [compute_doc_id("/c.txt")]

        # Apply: prune changed, remove deleted, commit new+changed re-extraction.
        assert rig.inc.remove_changed_and_deleted(delta)
        assert rig.stored_ids() == {"ea"}

        rig.commit(fps2, "/b.txt", {"eb2": "B2"}, version="-v2")
        rig.commit(fps2, "/d.txt", {"ed": "D"})
        assert rig.stored_ids() == {"ea", "eb2", "ed"}
        live = {r.doc_id for r in rig.store.list_all()}
        assert live == {compute_doc_id(p) for p in ("/a.txt", "/b.txt", "/d.txt")}

    def test_delete_then_readd_same_doc_id(self, rig) -> None:
        _, fps = detect_delta([_doc("/a.txt", "v1")], rig.inc.doc_status)
        rig.commit(fps, "/a.txt", {"e_old": "Old"})

        del_delta, _ = detect_delta([], rig.inc.doc_status)
        assert rig.inc.remove_changed_and_deleted(del_delta)
        assert rig.store.get(compute_doc_id("/a.txt")) is None

        # Re-add the same path with new content -> classified new again (clean
        # slate), and its registry record lists only the new text units.
        readd, fps2 = detect_delta([_doc("/a.txt", "v2")], rig.inc.doc_status)
        assert readd.new == [compute_doc_id("/a.txt")]
        unit = rig.commit(fps2, "/a.txt", {"e_new": "New"}, version="-v2")
        rec = rig.store.get(compute_doc_id("/a.txt"))
        assert rec is not None and rec.text_unit_ids == [unit]
        assert rig.stored_ids() == {"e_new"}

    def test_entity_spanning_deleted_and_surviving_doc_survives(self, rig) -> None:
        # An entity whose text_unit_ids span a deleted doc AND a surviving doc
        # must survive, stripped of the deleted doc's text unit.
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
        lineages = build_document_lineage(docs, text_units)
        _, fps = detect_delta(docs, rig.inc.doc_status)
        rig.inc.commit(lineages, fps, text_units=text_units, entities=[alice, bob])

        delta, _ = detect_delta([d2], rig.inc.doc_status)  # delete /a.txt
        assert rig.inc.remove_changed_and_deleted(delta)
        assert rig.entity_units() == {"e_alice": ["t2"]}
        assert rig.vector.ids("entities") == {"e_alice"}
        assert rig.vector.ids("text_units") == {"t2"}

    def test_relationship_to_a_deleted_entity_is_removed(self, rig) -> None:
        # A relationship whose endpoint entity gets deleted: doc a states
        # 'e_target' exclusively, doc b states an edge pointing AT it. Removing
        # a deletes e_target, and with it the incident edge, from both stores
        # (no dangling endpoint), although the edge cites b's text unit.
        a, b = _doc("/a.txt", "A"), _doc("/b.txt", "B")
        _, fps = detect_delta([a, b], rig.inc.doc_status)
        rig.commit(fps, "/a.txt", {"e_target": "Target"})
        rig.commit(
            fps,
            "/b.txt",
            {"e_src": "Src"},
            relationships=[
                Relationship(id="r1", source_id="e_src", target_id="e_target")
            ],
        )
        assert rig.graph.ids("relationships") == {"r1"}

        delta, _ = detect_delta([b], rig.inc.doc_status)  # delete a
        assert rig.inc.remove_changed_and_deleted(delta)
        assert rig.stored_ids() == {"e_src"}
        assert rig.graph.ids("relationships") == set()
        assert rig.vector.ids("relationships") == set()

    def test_empty_delta_is_clean_noop(self, rig, mocker) -> None:
        docs = [_doc("/a.txt", "A")]
        _, fps = detect_delta(docs, rig.inc.doc_status)
        rig.commit(fps, "/a.txt", {"e1": "A"})
        remove = mocker.spy(rig.manager, "remove_text_units")

        delta, _ = detect_delta(docs, rig.inc.doc_status)  # identical corpus
        assert delta.is_empty
        assert rig.inc.remove_changed_and_deleted(delta)
        # No prune/remove side effects when nothing changed.
        remove.assert_not_called()
        assert rig.stored_ids() == {"e1"}

    def test_rerunning_same_delta_twice_is_idempotent(self, rig) -> None:
        # Re-running an identical add commit twice must not duplicate entities
        # in the merged graph or grow the registry record.
        docs = [_doc("/a.txt", "A")]
        _, fps = detect_delta(docs, rig.inc.doc_status)
        rig.commit(fps, "/a.txt", {"e1": "A"})
        rig.commit(fps, "/a.txt", {"e1": "A"})
        rec = rig.store.get(compute_doc_id("/a.txt"))
        assert rec is not None and rec.text_unit_ids == [_unit_id("/a.txt")]
        assert len(rig.store.list_all()) == 1
        assert rig.entity_units() == {"e1": [_unit_id("/a.txt")]}


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

    def test_relationship_weight_idempotent_on_reapply(self) -> None:
        # Regression: relationship weight is the sum of the per-text-unit
        # strengths and a re-applied delta replaces its text units' entries, so
        # re-applying the same delta (e.g. a retried commit) leaves both the
        # weight and the text-unit set stable instead of double-counting.
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
        assert once[0].weight == 3.0  # t1 (1.0) + t2 (2.0)
        assert twice[0].weight == 3.0  # re-apply does NOT inflate the weight
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

    def test_union_then_trim_on_partial_delete(self, rig) -> None:
        # A shared entity cites both docs' text units; deleting one referrer
        # keeps it for the survivor, stripped of the deleted doc's unit.
        docs = [_doc("/a.txt", "A"), _doc("/b.txt", "B")]
        a, b = docs
        a.document_id, b.document_id = "ra", "rb"
        tus = [
            TextUnit(id="t1", text="x", document_ids=["ra"]),
            TextUnit(id="t2", text="x", document_ids=["rb"]),
        ]
        shared = _entity("shared", "S", text_unit_ids=["t1", "t2"])
        lineages = build_document_lineage(docs, tus)
        assert [ln.text_unit_ids for ln in lineages] == [["t1"], ["t2"]]

        _, fps = detect_delta(docs, rig.inc.doc_status)
        rig.inc.commit(lineages, fps, text_units=tus, entities=[shared])
        delta, _ = detect_delta([b], rig.inc.doc_status)  # delete a
        assert rig.inc.remove_changed_and_deleted(delta)
        assert rig.entity_units() == {"shared": ["t2"]}


class TestDeltaWriteGate:
    """The gate that decides whether a delta is durably marked PROCESSED.

    Recording a delta as PROCESSED when its writes actually failed is permanent
    silent data loss: on the next run the docs classify as ``unchanged`` and
    their never-indexed artifacts are never retried. The gate must mirror the
    SAME failure criteria as IndexingStage._validate_backend_success (complete
    failure OR failure rate above max_failure_rate), because that gate runs
    after commit() has already written the registry.
    """

    @staticmethod
    def _inc(max_failure_rate: float = 0.2) -> IncrementalIndexer:
        config = Config()
        config.indexing.max_failure_rate = max_failure_rate
        return IncrementalIndexer(
            FakeDocStatusStore(),
            SimpleNamespace(config=config),  # type: ignore[arg-type]
        )

    def test_complete_success_records_processed(self) -> None:
        results = {
            "opensearch_entities": IndexingStats(total_items=5, successful_items=5),
        }
        assert self._inc()._delta_writes_succeeded(results) is True

    def test_partial_success_under_threshold_records(self) -> None:
        # 2/5 = 40% failed > 20% default -> blocked; raise tolerance to confirm a
        # below-threshold partial failure still records.
        results = {
            "opensearch_entities": IndexingStats(
                total_items=5, successful_items=3, failed_items=2
            ),
        }
        assert self._inc(max_failure_rate=0.5)._delta_writes_succeeded(results) is True

    def test_partial_failure_over_threshold_blocks(self) -> None:
        # Regression: partial failure above the tolerated rate must NOT record
        # PROCESSED, else _validate_backend_success later raises (failing the
        # pipeline) while the registry write is already committed -> the failed
        # artifacts are classified `unchanged` next run and never retried.
        results = {
            "neptune_relationships": IndexingStats(
                total_items=100, successful_items=22, failed_items=78
            ),
        }
        assert self._inc()._delta_writes_succeeded(results) is False

    def test_zero_success_with_seeded_total_blocks(self) -> None:
        results = {
            "opensearch_entities": IndexingStats(total_items=5, successful_items=0),
        }
        assert self._inc()._delta_writes_succeeded(results) is False

    def test_exception_before_seeding_blocks(self) -> None:
        # Regression: an index type that raises before _perform_indexing seeds
        # total_items surfaces as total_items=0, failed_items=N. Gating on
        # total_items alone let this complete failure pass as success.
        stats = IndexingStats()
        stats.add_error("Bedrock embedding endpoint unreachable", 500)
        assert stats.total_items == 0 and stats.failed_items == 500
        assert self._inc()._delta_writes_succeeded({"os_entities": stats}) is False

    def test_no_work_is_success(self) -> None:
        # total_items==0 AND failed_items==0 means nothing had to be written.
        results = {"os_entities": IndexingStats()}
        assert self._inc()._delta_writes_succeeded(results) is True
