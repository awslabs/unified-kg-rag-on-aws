# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Removal by text unit, ids that follow the current name, record GC (AWS-free).

An incremental run finds what a removed document contributed in the stores,
by the text units every artifact cites (``IndexingManager.remove_text_units``);
an entity whose derived name has another id takes it, with its edges and
claims; and the extraction records only removed text units named are deleted
after the commit.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from tests.fixtures.incremental_runs import Corpus, IncrementalRun, expected_state
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.delta_detector import (
    assign_document_identity,
    assign_registry_source,
)
from unified_kg_rag.domain.models import (
    Community,
    CommunityReport,
    Config,
    Document,
    Entity,
    TextUnit,
)

pytestmark = pytest.mark.unit


def _manager(
    fuzzy: bool = False,
) -> tuple[IndexingManager, FakeGraphStore, FakeVectorStore]:
    config = Config()
    config.processing.fuzzy_entity_resolution = fuzzy
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    return (
        IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph),
        graph,
        vector,
    )


def test_a_community_and_its_report_go_with_their_last_stored_text_unit() -> None:
    manager, graph, vector = _manager()
    vector.upsert_text_units(
        [TextUnit(id=unit, text="...", document_ids=["d"]) for unit in ("t1", "t2")]
    )
    graph.upsert_communities(
        [
            Community(
                id="c-only",
                name="only",
                level="0",
                parent="",
                children=[],
                text_unit_ids=["t1"],
            ),
            Community(
                id="c-both",
                name="both",
                level="0",
                parent="",
                children=[],
                text_unit_ids=["t1", "t2"],
            ),
        ]
    )
    vector.upsert_community_reports(
        [
            CommunityReport(
                id="r-only", name="r", community_id="c-only", text_unit_ids=["t1"]
            ),
            CommunityReport(
                id="r-both", name="r", community_id="c-both", text_unit_ids=["t1", "t2"]
            ),
        ]
    )

    manager.remove_text_units("default", ["t1"])

    assert graph.ids("communities") == {"c-both"}
    assert vector.ids("community_reports") == {"r-both"}

    manager.remove_text_units("default", ["t2"])

    assert graph.ids("communities") == set()
    assert vector.ids("community_reports") == set()


def test_the_name_key_lookup_reads_no_full_name_list(mocker) -> None:
    manager, graph, _ = _manager()
    acme = Entity(id="e-acme-inc", name="Acme Inc", text_unit_ids=["t1"])
    graph.upsert_entities([acme, Entity(id="e-x", name="Other", text_unit_ids=["t1"])])
    every_name = mocker.spy(graph, "read_entity_names")

    matcher, matched = manager._match_stored_names(
        "default", [Entity(id="e-acme", name="ACME", text_unit_ids=["t2"])]
    )

    assert matched == {"e-acme-inc"}
    assert matcher is not None and matcher.candidates == ["Acme Inc"]
    every_name.assert_not_called()


def test_fuzzy_matching_still_reads_every_name(mocker) -> None:
    manager, graph, _ = _manager(fuzzy=True)
    graph.upsert_entities(
        [Entity(id="e-acme", name="Acme Corporation", text_unit_ids=["t1"])]
    )
    every_name = mocker.spy(graph, "read_entity_names")

    _, matched = manager._match_stored_names(
        "default", [Entity(id="e-new", name="Acme Corporatio", text_unit_ids=["t2"])]
    )

    assert matched == {"e-acme"}
    every_name.assert_called_once()


_V, _VI, _D, _B = "Vendor", "Vendor Inc", "Depot", "Buyer"
# "Vendor" was indexed first; two later documents call it "Vendor Inc", the
# spelling a full build picks (more text units), so its id is that name's.
# (Odd strengths: the units type every entity ORG, so the spellings merge.)
_FIRST: Corpus = {"/c/a.txt": [[(_V, _D, 1), (_V, _B, 3)]]}
_FLIPPED: Corpus = {
    **_FIRST,
    "/c/d.txt": [[(_VI, _D, 1)]],
    "/c/e.txt": [[(_VI, _B, 3)]],
}


def test_an_entity_takes_the_id_of_the_name_its_text_units_support() -> None:
    harness = IncrementalRun()
    harness.run(_FIRST)
    old_id = BaseProcessor._generate_entity_id(_V)

    harness.run(_FLIPPED)

    assert harness.state() == expected_state(_FLIPPED)
    new_id = BaseProcessor._generate_entity_id(_VI)
    assert new_id in harness.graph.ids("entities")
    # The old id is gone from both stores, and nothing points at it.
    assert old_id not in harness.graph.ids("entities")
    assert old_id not in harness.vector.ids("entities")
    edges = harness.graph.read_relationships(sorted(harness.graph.ids("relationships")))
    assert old_id not in {i for r in edges for i in (r.source_id, r.target_id)}
    assert old_id not in {
        c.subject_id for c in harness.vector.data.get("claims", {}).values()
    }
    assert harness.registry_problems(_FLIPPED) == []


def test_the_id_follows_the_name_back_when_its_support_is_deleted() -> None:
    harness = IncrementalRun()
    harness.run(_FIRST)
    harness.run(_FLIPPED)
    only_first = {"/c/a.txt": _FIRST["/c/a.txt"], "/c/d.txt": _FLIPPED["/c/d.txt"]}

    harness.run(only_first)
    harness.run(_FIRST)

    assert harness.state() == expected_state(_FIRST)
    assert BaseProcessor._generate_entity_id(_VI) not in harness.graph.ids("entities")


def test_extraction_records_only_removed_text_units_named_are_deleted() -> None:
    before: Corpus = {
        "/c/a.txt": [[(_V, _D, 1)], [(_D, _B, 2)]],
        "/c/b.txt": [[(_B, _V, 1)]],
    }
    # b.txt is deleted; a.txt keeps its first unit and replaces its second.
    after: Corpus = {"/c/a.txt": [[(_V, _D, 1)], [(_D, _B, 3)]]}
    harness = IncrementalRun()
    harness.run(before)
    assert harness.unused_extraction_records() == set()

    harness.run(after)

    assert harness.state() == expected_state(after)
    assert harness.unused_extraction_records() == set()
    # The records the stored text units name are all still there.
    assert len(harness.vector.extraction_records) == 2 * len(
        harness.vector.ids("text_units")
    )


def test_a_text_unit_belongs_to_one_registry_record() -> None:
    # The same file with the same text in two corpora written to one
    # namespace: two registry records, so their text units must differ, or
    # removing one corpus's copy would remove the other's text units.
    def staged(source_scope: str) -> Document:
        document = Document(
            page_content="Vendor ships goods.",
            document_id="x",
            file_name="a.txt",
            file_path="/staging/a.txt",
        )
        assign_document_identity(document, "/staging")
        assign_registry_source(document, source_scope)
        return document

    first, second = staged("s3://bucket/one/"), staged("s3://bucket/two/")

    assert first.document_id != second.document_id
    assert staged("s3://bucket/one/").document_id == first.document_id


def test_an_edge_of_another_spelling_merges_into_the_stored_one() -> None:
    # c.txt's "Depot -> Vendor Inc" is stored under an id of its own while
    # Vendor Inc is the entity's name; when d.txt reverts to "Vendor", the
    # delta's "Depot -> Vendor" edge is the same edge in a full build: it
    # must merge with the stored one, not overwrite it under the shared id.
    before: Corpus = {"/c/d.txt": [[(_D, _V, 1)], [(_V, _D, 1)]]}
    flipped: Corpus = {
        "/c/b.txt": [[(_B, _V, 3)]],
        "/c/d.txt": [[(_B, _VI, 3)]],
        "/c/c.txt": [[(_D, _B, 1)], [(_D, _VI, 1)]],
    }
    reverted = {**flipped, "/c/d.txt": before["/c/d.txt"]}
    harness = IncrementalRun()
    for corpus in (before, flipped, reverted):
        harness.run(corpus)

    assert harness.state() == expected_state(reverted)
    assert harness.registry_problems(reverted) == []


def test_a_rekey_interrupted_mid_commit_converges() -> None:
    corpus: Corpus = {
        "/c/c.txt": [[(_V, _D, 1)]],
        "/c/d.txt": [[(_VI, _D, 1), (_V, _D, 1)]],
    }
    harness = IncrementalRun()
    assert harness.run(corpus, interrupt="mid_commit", k=2)

    harness.run(corpus)

    assert harness.state() == expected_state(corpus)
    assert harness.registry_problems(corpus) == []


def test_a_rekey_interrupted_before_the_old_id_is_deleted_converges() -> None:
    # Deleting d.txt re-keys the shared "Vendor -> Depot" edge to the id a.txt
    # alone gives it; the run dies after writing the new copy, before deleting
    # the old one. The retry finds the old copy by d.txt's text unit and the
    # new one by the id it re-keys to, and keeps one edge.
    before: Corpus = {
        "/c/a.txt": [
            [(_V, _B, 1), (_B, _V, 1)],
            [(_B, _V, 3), (_VI, _D, 1), (_D, _VI, 1)],
        ],
        "/c/d.txt": [[(_V, _D, 1)]],
    }
    after = {"/c/a.txt": before["/c/a.txt"]}
    harness = IncrementalRun()
    harness.run(before)
    assert harness.run(after, interrupt="mid_removal", k=3)

    harness.run(after)

    assert harness.state() == expected_state(after)
    assert harness.registry_problems(after) == []


def test_the_merged_copies_of_an_entity_rank_over_all_their_edges() -> None:
    # Interrupted before the old id was deleted, the entity's edges are split
    # between its old and new copy: the retry merges the copies and ranks the
    # entity over the edges of both.
    before: Corpus = {
        "/c/b.txt": [[(_VI, _D, 1)]],
        "/c/c.txt": [[(_D, _VI, 3), (_V, _D, 1)]],
    }
    after = {"/c/c.txt": before["/c/c.txt"]}
    harness = IncrementalRun()
    harness.run(before)
    assert harness.run(after, interrupt="mid_removal", k=3)

    harness.run(after)

    assert harness.state() == expected_state(after)


def test_only_a_resumed_removal_looks_edges_up_by_text_unit(mocker) -> None:
    # The lookup scans a suffix's edges on Neptune. A first removal finds the
    # edges among those of the entities citing the removed units; only one
    # whose units are already gone from the vector store (an earlier attempt
    # got past deleting them and died) needs it.
    manager, graph, vector = _manager()
    vector.upsert_text_units(
        [TextUnit(id=unit, text="...", document_ids=["d"]) for unit in ("t1", "t2")]
    )
    graph.upsert_entities([Entity(id="e", name="E", text_unit_ids=["t1", "t2"])])
    lookup = mocker.spy(graph, "find_relationship_ids_citing")

    manager.remove_text_units("default", ["t1"])
    assert lookup.call_count == 0

    manager.remove_text_units("default", ["t1"])
    assert lookup.call_count == 1


def test_nothing_is_stripped_until_the_removed_units_are_deleted(mocker) -> None:
    # A retry tells that a removal got past deleting the removed units by
    # their missing copies, so no shared item is stripped (and no edge
    # re-keyed) while that deletion failed: the retry starts over.
    from unified_kg_rag.ports.indexer import IndexingStats

    manager, graph, vector = _manager()
    vector.upsert_text_units(
        [TextUnit(id=unit, text="...", document_ids=["d"]) for unit in ("t1", "t2")]
    )
    graph.upsert_entities([Entity(id="e", name="E", text_unit_ids=["t1", "t2"])])
    failed = IndexingStats(total_items=1)
    failed.add_error("store down", count=1)
    mocker.patch.object(manager, "delete_documents", return_value={"x": failed})
    strip = mocker.spy(manager, "remove_text_units_from_shared")

    results = manager.remove_text_units("default", ["t1"])

    assert results == {"x": failed}
    assert strip.call_count == 0


def test_a_claim_left_on_the_other_spelling_marks_a_leftover_split() -> None:
    # A split a killed run left half-written leaves the claims it did not
    # rewrite on the other spelling, citing the split-off part's units. Only
    # such a spelling is rederived; one with neither such an edge nor such a
    # claim (a full build keeps the two apart) is left as stored.
    from unified_kg_rag.domain.models import Claim

    manager, _, vector = _manager()
    part = Entity(id="e-part", name="Acme Inc", text_unit_ids=["t-part"])
    vector.upsert_claims(
        [
            Claim(
                id="c-left",
                subject_id="e-whole",
                subject_name="Acme",
                object_name="100",
                type="PAYS",
                text_unit_ids=["t-part"],
            ),
            Claim(
                id="c-own",
                subject_id="e-apart",
                subject_name="Acme",
                object_name="200",
                type="PAYS",
                text_unit_ids=["t-apart"],
            ),
        ]
    )

    left = manager._leftover_spellings(
        "default", [part], {"e-part": "e-whole"}, edges=[]
    )
    kept_apart = manager._leftover_spellings(
        "default", [part], {"e-part": "e-apart"}, edges=[]
    )

    assert left == {"e-part": "e-whole"}
    assert kept_apart == {}


@pytest.mark.parametrize("held_by", ["edge", "claim"])
def test_a_third_spellings_statement_on_a_whole_marks_a_leftover_split(
    held_by: str,
) -> None:
    # Two parts of one name ("Acme Inc", "Acme Ltd") split off two stored
    # spellings. The whole of "Acme Inc" holds a statement of "Acme Ltd"
    # (an edge or a claim citing its unit): a killed run left that split
    # half-written too, though the statement cites no unit of "Acme Inc".
    from unified_kg_rag.domain.models import Claim, Relationship

    manager, _, vector = _manager()
    parts = [
        Entity(id="e-inc", name="Acme Inc", text_unit_ids=["t-inc"]),
        Entity(id="e-ltd", name="Acme Ltd", text_unit_ids=["t-ltd"]),
    ]
    edges = []
    if held_by == "edge":
        edges = [
            Relationship(
                id="r-left",
                source_id="e-whole",
                target_id="e-depot",
                source_name="Acme",
                target_name="Depot",
                text_unit_ids=["t-ltd"],
            )
        ]
    else:
        vector.upsert_claims(
            [
                Claim(
                    id="c-left",
                    subject_id="e-whole",
                    subject_name="Acme",
                    object_name="100",
                    type="PAYS",
                    text_unit_ids=["t-ltd"],
                )
            ]
        )

    left = manager._leftover_spellings(
        "default", parts, {"e-inc": "e-whole", "e-ltd": "e-other"}, edges=edges
    )

    assert left == {"e-inc": "e-whole"}


def test_a_strip_note_that_cannot_be_written_stops_the_strip(mocker) -> None:
    # Without its note a retry could not finish the merge: the strip fails
    # (its removal is retried) rather than write the merged items.
    from unified_kg_rag.ports.indexer import IndexingStats
    from unified_kg_rag.shared import DataProcessingError

    manager, _, vector = _manager()
    failed = IndexingStats(total_items=1)
    failed.add_error("store down", count=1)
    mocker.patch.object(vector, "write_extraction_records", return_value=failed)

    with pytest.raises(DataProcessingError):
        manager._note_pending_strip("default", ["t1"], ["e"], [])


def test_a_strip_note_left_behind_only_logs(mocker, caplog) -> None:
    # Deleting the note is best-effort: one left makes a retry strip the
    # noted items again, which changes nothing.
    manager, _, vector = _manager()
    mocker.patch.object(
        vector, "delete_extraction_records", side_effect=RuntimeError("store down")
    )

    manager._clear_pending_strip("default", ["t1"])

    assert "Deleting the strip notes" in caplog.text


def test_only_a_deleted_communitys_id_leaves_a_stripped_entity() -> None:
    # Removing t1 deletes "c-only" (it cites no other unit) and keeps
    # "c-both": the stripped entity drops the one id, not both.
    manager, graph, vector = _manager()
    vector.upsert_text_units(
        [TextUnit(id=unit, text="...", document_ids=["d"]) for unit in ("t1", "t2")]
    )
    graph.upsert_communities(
        [
            Community(
                id=cid, name=cid, level="0", parent="", children=[], text_unit_ids=units
            )
            for cid, units in (("c-only", ["t1"]), ("c-both", ["t1", "t2"]))
        ]
    )
    graph.upsert_entities(
        [
            Entity(
                id="e",
                name="E",
                text_unit_ids=["t1", "t2"],
                community_ids=["c-only", "c-both"],
            )
        ]
    )

    manager.remove_text_units("default", ["t1"])

    (entity,) = graph.read_entities(["e"])
    assert entity.community_ids == ["c-both"]


def test_a_claim_the_derivation_drops_is_retired(mocker) -> None:
    # A stored claim no claim of its units corresponds to is deleted with the
    # retired ids, once the run's writes succeeded.
    from unified_kg_rag.domain.ingestion.derivation import Derived
    from unified_kg_rag.domain.models import Claim

    manager, _, _ = _manager()
    stale = Claim(
        id="stale",
        subject_id="e",
        subject_name="E",
        object_name="x",
        type="PAYS",
        text_unit_ids=["t1"],
    )
    mocker.patch.object(manager, "derive", return_value=Derived(dropped=["stale"]))

    assert manager._derive_claims("default", [stale]) == []
    assert "stale" in manager._dropped["default"]


def test_a_deleted_community_leaves_a_kept_text_unit_naming_it() -> None:
    # A later delta wrote "c" anew under its id without t2, which still names
    # it; once "c" goes with t1, t2 names it no more.
    manager, graph, vector = _manager()
    vector.upsert_text_units(
        [
            TextUnit(id="t1", text="...", document_ids=["d"], community_ids=["c"]),
            TextUnit(id="t2", text="...", document_ids=["e"], community_ids=["c"]),
        ]
    )
    graph.upsert_communities(
        [
            Community(
                id="c",
                name="c",
                level="0",
                parent="",
                children=[],
                text_unit_ids=["t1"],
            )
        ]
    )

    manager.remove_text_units("default", ["t1"])

    assert graph.ids("communities") == set()
    assert vector.data["text_units"]["t2"].community_ids == []
