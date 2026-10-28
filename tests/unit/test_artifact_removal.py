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
    Relationship,
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
        manager._note_pending_strip("default", ["e"], [])


def test_a_strip_note_left_behind_only_logs(mocker, caplog) -> None:
    # Deleting the note is best-effort: one left makes a retry strip the
    # noted items again, which changes nothing.
    manager, _, vector = _manager()
    mocker.patch.object(
        vector, "delete_extraction_records", side_effect=RuntimeError("store down")
    )

    manager._clear_pending_strip("default")

    assert "Deleting the note" in caplog.text


def test_a_removed_vector_only_entity_takes_its_vector_edges() -> None:
    # A killed run wrote an entity and its edge to the vector store only.
    # Removing the entity's text unit deletes it; the graph knows no edge at
    # it, so the edge (citing a kept unit too) is found in the vector store.
    from unified_kg_rag.domain.models import Relationship

    manager, graph, vector = _manager()
    vector.upsert_text_units(
        [TextUnit(id=unit, text="...", document_ids=["d"]) for unit in ("t1", "t2")]
    )
    graph.upsert_entities([Entity(id="kept", name="Kept", text_unit_ids=["t2"])])
    vector.upsert_entities(
        [
            Entity(id="gone", name="Gone", text_unit_ids=["t1"]),
            Entity(id="kept", name="Kept", text_unit_ids=["t2"]),
        ]
    )
    vector.upsert_relationships(
        [
            Relationship(
                id="r",
                source_id="kept",
                target_id="gone",
                source_name="Kept",
                target_name="Gone",
                text_unit_ids=["t2"],
            )
        ]
    )

    manager.remove_text_units("default", ["t1"])

    assert "r" not in vector.data.get("relationships", {})


def test_a_graph_store_without_read_back_is_not_taken_for_missing_copies() -> None:
    # The port's read_entities reads nothing back: every shared id would
    # look like a copy only the vector store holds, and be deleted. Also
    # through the view the pipeline hands its stages of an injected graph
    # indexer (asked of its type, the view hid it, and a removal deleted
    # the shared vector entities).
    from unified_kg_rag.application.ingestion.pipeline import _Borrowed
    from unified_kg_rag.application.storage.indexing_manager import _reads_back
    from unified_kg_rag.ports.indexer import GraphIndexer

    class NoReadBack(GraphIndexer):
        pass

    NoReadBack.__abstractmethods__ = frozenset()
    no_read_back = NoReadBack.__new__(NoReadBack)

    assert _reads_back(FakeGraphStore())
    assert _reads_back(_Borrowed(FakeGraphStore()))
    assert not _reads_back(no_read_back)
    assert not _reads_back(_Borrowed(no_read_back))


@pytest.mark.parametrize(
    ("stored_units", "retired"),
    [
        pytest.param(["t1"], True, id="citing-nothing-more"),
        pytest.param(["t1", "t9"], False, id="citing-another-unit"),
    ],
)
def test_a_stored_edge_of_a_written_edges_key_is_retired_only_if_it_adds_nothing(
    stored_units: list[str], retired: bool
) -> None:
    # A stored edge between the same entities with the same type under
    # another id is the written edge under an earlier id, unless it cites a
    # unit the written one does not (another statement a full build keeps).
    manager, graph, _ = _manager()
    entities = [
        Entity(id="a", name="A", type="ORGANIZATION", text_unit_ids=["t1", "t9"]),
        Entity(id="b", name="B", type="ORGANIZATION", text_unit_ids=["t1", "t9"]),
    ]
    graph.upsert_entities(entities)
    graph.upsert_relationships(
        [
            Relationship(
                id="r-stored",
                source_id="a",
                target_id="b",
                source_name="A",
                target_name="B",
                type="SUPPLIES",
                text_unit_ids=stored_units,
            )
        ]
    )
    written = Relationship(
        id="r-written",
        source_id="a",
        target_id="b",
        source_name="A",
        target_name="B",
        type="supplies",
        text_unit_ids=["t1"],
    )

    manager._finish("default", entities, [written])

    assert ("r-stored" in manager._dropped.get("default", set())) is retired


def test_a_retired_vector_only_entity_takes_its_vector_edges() -> None:
    # A killed run wrote an entity and its edge to the vector store only. A
    # removal finds the entity by a removed unit, but it cites a kept one
    # too: the strip retires it (the graph holds no copy to strip), and the
    # edge at it (citing the kept unit) goes with it, found in the vector
    # store.
    from unified_kg_rag.domain.models import Relationship

    manager, graph, vector = _manager()
    vector.upsert_text_units(
        [TextUnit(id=unit, text="...", document_ids=["d"]) for unit in ("t1", "t2")]
    )
    graph.upsert_entities([Entity(id="kept", name="Kept", text_unit_ids=["t2"])])
    vector.upsert_entities(
        [
            Entity(id="partly", name="Partly", text_unit_ids=["t1", "t2"]),
            Entity(id="kept", name="Kept", text_unit_ids=["t2"]),
        ]
    )
    vector.upsert_relationships(
        [
            Relationship(
                id="r",
                source_id="kept",
                target_id="partly",
                source_name="Kept",
                target_name="Partly",
                text_unit_ids=["t2"],
            )
        ]
    )

    manager.remove_text_units("default", ["t1"])

    assert "partly" not in vector.data.get("entities", {})
    assert "r" not in vector.data.get("relationships", {})


def test_a_stripped_entity_that_gains_units_is_written_before_its_edges(
    mocker,
) -> None:
    # Another spelling's statement merged into a stripped entity: the
    # re-pointed edges hang on it, so it is written with the re-keyed ones,
    # before them (written last, a run killed in between left the edges on
    # a vertex citing only removed units).
    from unified_kg_rag.domain.ingestion.derivation import Derived

    manager, graph, vector = _manager()
    vector.upsert_text_units(
        [TextUnit(id=unit, text="...", document_ids=["d"]) for unit in ("t0", "t1")]
    )
    graph.upsert_entities([Entity(id="e", name="E", text_unit_ids=["t0", "t1"])])
    gained = Entity(id="e", name="E", text_unit_ids=["t1", "t2"])
    mocker.patch.object(manager, "derive", return_value=Derived(entities=[gained]))
    phases = mocker.spy(manager, "_run_indexing_phase")

    manager.remove_text_units_from_shared(
        "default", entity_ids=["e"], relationship_ids=[], text_unit_ids=["t0"]
    )

    (first,) = phases.call_args_list[0].args[0]
    assert first.key == "neptune_rekeyed_entities"
    assert [e.id for e in first.args[0]] == ["e"]


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


def test_a_strip_that_notes_again_keeps_the_earlier_notes_claims(mocker) -> None:
    # A retry's strip re-keys "e-old" and replaces the earlier note: the
    # claims the earlier one named (a split's, which nothing else finds)
    # stay in it.
    from unified_kg_rag.domain.ingestion.derivation import Derived

    manager, graph, vector = _manager()
    stored = Entity(id="e-old", name="Acme", text_unit_ids=["t1", "t2"])
    graph.upsert_entities([stored])
    vector.upsert_entities([stored])
    mocker.patch.object(
        manager,
        "derive",
        return_value=Derived(
            entities=[Entity(id="e-new", name="ACME", text_unit_ids=["t1"])],
            id_remap={"e-old": "e-new"},
        ),
    )
    note = mocker.spy(manager, "_note_pending_strip")

    manager.remove_text_units_from_shared(
        "default",
        entity_ids=["e-old"],
        relationship_ids=[],
        text_unit_ids=["t2"],
        earlier_note={"entity_ids": [], "claim_ids": ["c-split"]},
    )

    _, entity_ids, _, claim_ids = note.call_args.args
    assert entity_ids == ["e-new", "e-old"]
    assert claim_ids == ["c-split"]


def test_a_removal_clears_a_note_whose_items_are_gone(mocker) -> None:
    # An earlier strip's note names an item this removal deletes outright
    # (it cites only the removed unit), so nothing is left to strip: the
    # note is deleted here, or it would stay, and be re-read, for good.
    manager, graph, vector = _manager()
    vector.upsert_text_units([TextUnit(id="t1", text="...", document_ids=["d"])])
    entity = Entity(id="e-x", name="X", text_unit_ids=["t1"])
    graph.upsert_entities([entity])
    vector.upsert_entities([entity])
    note = {"entity_ids": ["e-x"], "relationship_ids": []}
    vector.write_extraction_records({"pending-strip": note}, suffix="default")
    read = mocker.spy(vector, "read_extraction_records")

    manager.remove_text_units("default", ["t1"])

    # Every removal reads the note, a first one as well as a retry.
    assert any(call.args[0] == ["pending-strip"] for call in read.call_args_list)
    assert vector.read_extraction_records(["pending-strip"], suffix="default") == {}
    assert "e-x" not in graph.ids("entities")


@pytest.mark.parametrize(
    ("rewired", "split_from", "vectors", "copy", "rewritten"),
    [
        pytest.param({"a"}, None, True, False, True, id="rewired-missing-copy"),
        pytest.param(
            set(), {"a": "z"}, True, False, True, id="split-part-missing-copy"
        ),
        pytest.param({"a"}, None, True, True, False, id="rewired-current-copy"),
        pytest.param(set(), None, True, False, False, id="not-re-pointed"),
        pytest.param({"a"}, None, False, False, False, id="no-relationship-vectors"),
    ],
)
def test_a_retry_writes_the_missing_vector_copy_of_a_re_pointed_edge(
    rewired: set[str],
    split_from: dict[str, str] | None,
    vectors: bool,
    copy: bool,
    rewritten: bool,
) -> None:
    # A run killed between a derivation's graph edges and their vector
    # copies left an edge at a re-pointed entity (rewired, or a split's
    # part) current in the graph and missing from the vector store: its
    # retry finds nothing else to rewrite it for, so it is rewritten because
    # its copy is missing, unless the run builds no relationship vectors.
    from unified_kg_rag.domain.models import Relationship

    manager, graph, vector = _manager()
    manager.config.indexing.opensearch.build_relationship_vector_index = vectors
    entities = [
        Entity(id="a", name="A", type="ORGANIZATION", text_unit_ids=["t1"]),
        Entity(id="b", name="B", type="ORGANIZATION", text_unit_ids=["t1"]),
    ]
    graph.upsert_entities(entities)
    vector.upsert_entities(entities)
    edge = Relationship(
        id="r1",
        source_id="a",
        target_id="b",
        source_name="A",
        target_name="B",
        type="SUPPLIES",
        text_unit_ids=["t1"],
        rank=2,
    )
    graph.upsert_relationships([edge])
    if copy:
        vector.upsert_relationships([edge])

    _, relationships = manager._finish(
        "default",
        entities,
        [],
        stored_entities=entities,
        rewired=rewired,
        split_from=split_from,
    )

    assert ([r.id for r in relationships] == ["r1"]) is rewritten


def test_a_strip_derives_claims_with_the_stored_spellings_of_derived_names() -> None:
    # A strip re-derives a claim whose object is an entity this run derived
    # (not stored yet): the stored entities sharing its name key are read
    # too (a split's part may be among them), once each, besides those the
    # claim names.
    from unified_kg_rag.domain.ingestion.base_resolver import compact_name_key
    from unified_kg_rag.domain.models import Claim

    manager, graph, _ = _manager()
    graph.upsert_entities(
        [
            Entity(id="a", name="Acme", type="ORGANIZATION", text_unit_ids=["t1"]),
            Entity(id="g", name="Globex", type="ORGANIZATION", text_unit_ids=["t1"]),
        ]
    )
    derived = Entity(
        id="x", name="Globex Ltd", type="ORGANIZATION", text_unit_ids=["t1"]
    )
    manager._derived_entities["default"] = {"x": derived}
    keys: list[list[str]] = []
    reads: list[list[str]] = []
    by_key, read = graph.find_entity_ids_by_name_key, graph.read_entities

    def find_by_key(name_keys, suffix=None):
        keys.append(sorted(name_keys))
        return by_key(name_keys, suffix)

    def read_entities(ids, suffix=None):
        reads.append(list(ids))
        return read(ids, suffix)

    graph.find_entity_ids_by_name_key = find_by_key  # type: ignore[method-assign]
    graph.read_entities = read_entities  # type: ignore[method-assign]
    claim = Claim(
        id="c1",
        subject_id="a",
        subject_name="Acme",
        object_id="x",
        object_name="Globex Ltd",
        type="OWNS",
        description="Acme owns Globex Ltd",
        text_unit_ids=["t1"],
    )

    manager._derive_claims("default", [claim], removed=["t9"])

    assert keys == [sorted({compact_name_key("Acme"), compact_name_key("Globex Ltd")})]
    assert reads == [["a"], ["g"]]


def test_a_dropped_edge_still_stored_does_not_rank_its_endpoints(mocker) -> None:
    # A strip killed before the retired ids were deleted leaves the edge its
    # derivation dropped in the graph store: the resumed strip reads it back
    # among the stored links, and ranked the entity over it too.
    manager, graph, _ = _manager()
    mocker.patch.object(
        graph, "read_neighbor_ids", return_value={"a": ["b", "c"], "c": ["a"]}
    )
    mocker.patch.object(graph, "find_incident_relationship_ids", return_value=[])
    dropped = Relationship(
        id="r-ab", source_id="a", target_id="b", description="x", text_unit_ids=[]
    )
    entities, _ = manager._finish(
        "default",
        [Entity(id="a", name="A", type="ORG", description="A")],
        [],
        stored_relationships=[dropped],
        dropped=[dropped.id],
    )
    assert [e.rank for e in entities] == [1]
