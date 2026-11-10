# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed artifact write is written again, not re-extracted (AWS-free).

A store that rejects an entity, relationship or claim write (a hub entity
whose lineage the graph store cannot take, a throttled bulk item) used to
record every document owning the item FAILED: the next run extracted them
all again, merged the hub again, failed again, and so on every run. The
commit now notes the rejected items in the extraction store
(``pending-writes``); the documents are recorded PROCESSED, and the next
commit writes the noted items again with its own delta. A document is
recorded FAILED only for a write of its own text units.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from typing import Any

import pytest

from tests.fixtures import incremental_runs
from tests.fixtures.doc_status_records import all_records
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.summarizer import AtomSetSummarizer
from tests.fixtures.incremental_runs import (
    POINTS,
    Corpus,
    IncrementalRun,
    document,
    expected_state,
    extract,
)
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - a fixture
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.models import Config, DocStatus, PipelineStageStatus
from unified_kg_rag.ports.indexer import IndexingStats

pytestmark = pytest.mark.integration

NOTE = "pending-writes"

_V, _D, _C, _B = "Vendor", "Depot", "Carrier", "Buyer"

# Vendor is a hub: every document states it.
_CORPUS: Corpus = {
    "/corpus/a.txt": [[(_V, _D, 1)]],
    "/corpus/b.txt": [[(_V, _C, 3)]],
    "/corpus/c.txt": [[(_V, _B, 2)], [(_C, _B, 3)]],
}


def _harness() -> IncrementalRun:
    # A hub's held back edges are most of this small corpus's: the failure
    # rate a run tolerates is raised so its documents are recorded.
    config = Config()
    config.indexing.max_failure_rate = 0.9
    return IncrementalRun(FakeDocStatusStore(), config)


def _hub_edges() -> list[str]:
    hub = _id(_V)
    return [
        r.id
        for r in extract(_CORPUS).relationships
        if hub in (r.source_id, r.target_id)
    ]


def _id(name: str) -> str:
    (entity,) = [e for e in extract(_CORPUS).entities if e.name == name]
    return entity.id


def _rejecting(where: str, *ids: str) -> Callable[[str, str], bool]:
    return lambda at, item_id: at.startswith(where) and item_id in ids


def _statuses(harness: IncrementalRun) -> dict[str, DocStatus]:
    return {r.file_path or r.doc_id: r.status for r in all_records(harness.store)}


def _note(harness: IncrementalRun) -> dict | None:
    return harness.vector.read_extraction_records([NOTE]).get(NOTE)


@pytest.mark.parametrize(
    "where",
    ["graph.upsert_entities", "vector.upsert_entities", "graph.upsert_relationships"],
)
def test_a_rejected_hub_write_is_written_next_run_without_extraction(
    where: str,
) -> None:
    harness = _harness()
    hub = _id(_V)
    rejected = _hub_edges() if "relationships" in where else [hub]

    harness.run(_CORPUS, fail=_rejecting(where, *rejected))

    # The documents are indexed: only the store failed, and its items are
    # noted for the next commit.
    assert set(_statuses(harness).values()) == {DocStatus.PROCESSED}
    note = _note(harness)
    assert note is not None
    noted = {i["id"] for kind in note.values() for i in kind}
    # With the edges held back for an endpoint the graph rejected.
    held = set(_hub_edges()) if where.startswith("graph.") else set()
    assert noted == set(rejected) | held

    harness.run(_CORPUS)

    assert harness.extracted == []
    assert harness.state() == expected_state(_CORPUS, harness.config)
    assert harness.registry_problems(_CORPUS) == []
    assert _note(harness) is None


def test_a_write_that_keeps_failing_stays_noted_and_extracts_nothing() -> None:
    harness = _harness()
    always = _rejecting("graph.upsert_entities", _id(_V))

    for _ in range(3):
        harness.run(_CORPUS, fail=always)
        assert harness.extracted == [] or len(harness.extracted) == len(_CORPUS)
        assert set(_statuses(harness).values()) == {DocStatus.PROCESSED}
        assert _note(harness) is not None

    harness.run(_CORPUS)

    assert harness.extracted == []
    assert harness.state() == expected_state(_CORPUS, harness.config)


def test_a_rejected_claim_is_written_next_run() -> None:
    harness = _harness()
    claim, _ = extract(_CORPUS).claims

    harness.run(_CORPUS, fail=_rejecting("vector.upsert_claims", claim.id))
    assert set(_statuses(harness).values()) == {DocStatus.PROCESSED}

    harness.run(_CORPUS)

    assert harness.extracted == []
    assert harness.state() == expected_state(_CORPUS, harness.config)


def test_a_rejected_text_unit_fails_its_own_document_only() -> None:
    harness = _harness()
    unit = next(
        u.id
        for u in extract(_CORPUS).units
        if u.document_ids
        == [document("/corpus/a.txt", _CORPUS["/corpus/a.txt"]).document_id]
    )

    harness.run(_CORPUS, fail=_rejecting("vector.upsert_text_units", unit))

    assert _statuses(harness) == {
        "/corpus/a.txt": DocStatus.FAILED,
        "/corpus/b.txt": DocStatus.PROCESSED,
        "/corpus/c.txt": DocStatus.PROCESSED,
    }
    harness.run(_CORPUS)
    assert harness.extracted == ["/corpus/a.txt"]
    assert harness.state() == expected_state(_CORPUS, harness.config)


def test_a_note_that_cannot_be_written_fails_the_owners(monkeypatch) -> None:
    # Without the note nothing would write the item again: its owners are
    # recorded FAILED, as before notes, so the next run rewrites them.
    harness = _harness()
    write = harness.vector.write_extraction_records

    def no_note(records, suffix=None):  # noqa: ANN001, ANN202
        if NOTE not in records:
            return write(records, suffix=suffix)
        stats = IndexingStats(total_items=1)
        stats.add_error("store down", ids=[NOTE])
        return stats

    monkeypatch.setattr(harness.vector, "write_extraction_records", no_note)
    harness.run(_CORPUS, fail=_rejecting("vector.upsert_entities", _id(_D)))

    assert _statuses(harness) == {
        "/corpus/a.txt": DocStatus.FAILED,
        "/corpus/b.txt": DocStatus.PROCESSED,
        "/corpus/c.txt": DocStatus.PROCESSED,
    }
    monkeypatch.undo()
    harness.run(_CORPUS)
    assert harness.extracted == ["/corpus/a.txt"]
    assert harness.state() == expected_state(_CORPUS, harness.config)


@pytest.mark.parametrize(
    "follow",
    [
        # A document stating the noted hub is removed, or changed: the noted
        # copy cites its units no more.
        {k: v for k, v in _CORPUS.items() if k != "/corpus/b.txt"},
        {**_CORPUS, "/corpus/a.txt": [[(_D, _C, 2)]]},
        # A new document states the hub too.
        {**_CORPUS, "/corpus/d.txt": [[(_V, _C, 1)]]},
    ],
    ids=["removed", "changed", "added"],
)
@pytest.mark.parametrize(
    "where", ["graph.upsert_entities", "vector.upsert_entities", "graph.upsert"]
)
def test_the_noted_items_meet_the_next_delta(follow: Corpus, where: str) -> None:
    harness = _harness()
    rejected = [_id(_V), *_hub_edges()]
    harness.run(_CORPUS, fail=_rejecting(where, *rejected))

    harness.run(follow)

    assert harness.state() == expected_state(follow, harness.config)
    assert harness.registry_problems(follow) == []
    assert _note(harness) is None


@pytest.mark.parametrize("point", POINTS)
def test_an_interrupted_replay_converges(point: str) -> None:
    harness = _harness()
    harness.run(_CORPUS, fail=_rejecting("graph.upsert", _id(_V)))
    follow = {**_CORPUS, "/corpus/a.txt": [[(_V, _C, 2)]]}

    harness.run(follow, interrupt=point)
    harness.run(follow)

    assert harness.state() == expected_state(follow, harness.config)
    assert harness.registry_problems(follow) == []
    harness.run(follow)
    assert harness.extracted == []


# Two spellings of one entity: the second document's edges turn the
# spelling the first stored them under, so the commit re-keys their ids.
_SPELLINGS: tuple[Corpus, Corpus] = (
    {
        "/corpus/d.txt": [
            [("Acme Corp", _D, 2), ("Acme Corp", _B, 2), (_B, _D, 1)],
            [(_B, _D, 1), ("Acme Inc", _D, 2)],
        ]
    },
    {
        # Its units sort before d.txt's (ids derive from the document's path
        # and content), so the edge to Buyer is re-keyed, not dropped.
        "/corpus/a.txt": [[(_D, "Acme Corp", 1), ("Acme Inc", _B, 1)]],
        "/corpus/d.txt": [
            [("Acme Corp", _D, 2), ("Acme Corp", _B, 2), (_B, _D, 1)],
            [(_B, _D, 1), ("Acme Inc", _D, 2)],
        ],
    },
)


def _depot_ids(config: Config) -> list[str]:
    return sorted(
        {e.id for c in _SPELLINGS for e in extract(c, config).entities if e.name == _D}
    )


def test_a_rejected_write_does_not_keep_a_re_keyed_items_old_copy() -> None:
    # The commit keeps the old ids when a write fails (deleting them could
    # lose an item); its note names them, so the next commit deletes them.
    harness = _harness()
    before, after = _SPELLINGS
    harness.run(before)

    harness.run(
        after, fail=_rejecting("graph.upsert_entities", *_depot_ids(harness.config))
    )
    harness.run(after)

    assert harness.state() == expected_state(after, harness.config)
    assert harness.vector.read_extraction_records(["pending-commit"]) == {}
    harness.run(after)
    assert harness.extracted == []


@pytest.mark.parametrize("point", POINTS)
def test_an_interrupted_re_key_converges(point: str) -> None:
    harness = _harness()
    before, after = _SPELLINGS
    harness.run(before)

    harness.run(after, interrupt=point)
    harness.run(after)

    assert harness.state() == expected_state(after, harness.config)
    assert harness.registry_problems(after) == []


class _Summarizing(IncrementalRun):
    """A harness whose runs summarize every merged description (with the
    model-free :class:`AtomSetSummarizer`), recording what they summarize."""

    def __init__(self) -> None:
        self.summarized: list[tuple[str, str]] = []
        config = Config()
        config.indexing.max_failure_rate = 0.9
        summarization = config.processing.graph_extraction.description_summarization
        summarization.force_summary_threshold_tokens = 1
        super().__init__(FakeDocStatusStore(), config)

    def _manager(self) -> IndexingManager:
        manager = super()._manager()
        summarizer = AtomSetSummarizer(self.config)
        summarize = summarizer._summarize_many

        def recording(names: list[str], descriptions: list[str]) -> list[str | None]:
            self.summarized += list(zip(names, descriptions, strict=True))
            return summarize(names, descriptions)

        summarizer._summarize_many = recording  # type: ignore[method-assign]
        manager._description_summarizer = summarizer
        return manager


def test_a_noted_summary_is_not_summarized_again() -> None:
    # The noted hub's description is a summary of its lines; extending its
    # stored copy's with it would summarize a summary, where a full build
    # summarizes the lines its records state.
    harness = _Summarizing()
    harness.run(_CORPUS)
    grown = {**_CORPUS, "/corpus/d.txt": [[(_V, _D, 2), (_C, _D, 1)]]}
    harness.run(grown, fail=_rejecting("graph.upsert_entities", _id(_V)))
    (noted,) = [e for e in _note(harness)["entities"] if e["name"] == _V]
    assert noted["description"].startswith("SUM{")
    harness.summarized.clear()

    harness.run({**grown, "/corpus/e.txt": [[(_V, _B, 1)]]})

    (hub,) = [d for name, d in harness.summarized if name == _V]
    assert "SUM{" not in hub, hub


# --- the pipeline, on the embedded stores -------------------------------------


def _rejecting_entity(name: str):  # noqa: ANN202
    """A wrapper of an ``OpenSearchIndexer`` entity write that rejects the
    entity ``name`` (reported failed by id) and writes the others."""

    def wrap(write):  # noqa: ANN001, ANN202
        def rejecting(self, entities, **kwargs):  # noqa: ANN001, ANN202
            rejected = [e.id for e in entities if e.name == name]
            stats = write(self, [e for e in entities if e.name != name], **kwargs)
            stats.total_items = len(entities)
            if rejected:
                stats.add_error("write rejected", len(rejected), ids=rejected)
            return stats

        return rejecting

    return wrap


@pytest.mark.usefixtures("no_aws")
@pytest.mark.parametrize("reset", [False, True], ids=["incremental", "reset"])
def test_the_pipeline_writes_a_rejected_entity_next_run_without_extraction(
    tmp_path, mocker, reset: bool
) -> None:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.indexing.max_failure_rate = 0.9
    config.indexing.reset = reset
    providers = e2e._providers(
        config, FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    wrap = _rejecting_entity("Vendor")
    for method in ("index_entities", "upsert_entities"):
        mocker.patch.object(
            OpenSearchIndexer, method, wrap(getattr(OpenSearchIndexer, method))
        )

    context = e2e._ingest(config, providers, source, tmp_path / "run-1")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert set(e2e._registry(providers).values()) == {DocStatus.PROCESSED}
    prefix = config.indexing.opensearch.entities_index_prefix
    names = {hit["_source"]["name"] for hit in e2e._stored(config, providers, prefix)}
    assert _V not in names

    mocker.stopall()
    config.indexing.reset = False
    context = e2e._ingest(config, providers, source, tmp_path / "run-2")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert context.incremental_delta is not None
    assert context.incremental_delta.to_process == []
    names = {hit["_source"]["name"] for hit in e2e._stored(config, providers, prefix)}
    assert _V in names
    assert set(e2e._registry(providers).values()) == {DocStatus.PROCESSED}


def test_a_failed_report_repoint_keeps_the_old_ids_for_a_retry(monkeypatch) -> None:
    # Deleting the old ids while the stored reports still cite them leaves
    # the reports' findings pointing at nothing: they stay, and the commit
    # note, kept, has the next run point the reports and delete them.
    harness = _harness()
    before, after = _SPELLINGS
    harness.run(before)
    repoint = type(harness.vector).repoint_report_findings
    rekeyed: list[dict[str, str]] = []

    def failing(self, id_remap, suffix=None):  # noqa: ANN001, ANN202
        rekeyed.append(dict(id_remap))
        stats = IndexingStats(total_items=len(id_remap))
        stats.add_error("mapping update denied", count=len(id_remap))
        return stats

    monkeypatch.setattr(type(harness.vector), "repoint_report_findings", failing)
    harness.run(after)
    (old,) = {old for remap in rekeyed for old in remap}
    assert old in harness.state()["graph_edges"]
    assert harness.vector.read_extraction_records(["pending-commit"]) != {}
    monkeypatch.setattr(type(harness.vector), "repoint_report_findings", repoint)

    harness.run(after)

    assert harness.state() == expected_state(after, harness.config)
    assert harness.vector.read_extraction_records(["pending-commit"]) == {}


def test_a_large_note_is_split_into_parts(monkeypatch) -> None:
    # One record holding every rejected item of a hub can exceed a store's
    # request limit (10 MiB on small OpenSearch instances).
    from unified_kg_rag.application.storage import pending_writes

    monkeypatch.setattr(pending_writes, "_PART_BYTES", 600)
    harness = _harness()
    rejected = [_id(_V), *_hub_edges()]

    harness.run(_CORPUS, fail=_rejecting("graph.upsert", *rejected))

    note = _note(harness)
    assert note is not None and len(note["parts"]) > 1
    parts = harness.vector.read_extraction_records(note["parts"])
    assert set(parts) == set(note["parts"])
    noted = {i["id"] for part in parts.values() for kind in part.values() for i in kind}
    assert set(rejected) <= noted

    harness.run(_CORPUS)

    assert harness.extracted == []
    assert harness.state() == expected_state(_CORPUS, harness.config)
    assert _note(harness) is None
    assert harness.vector.read_extraction_records(note["parts"]) == {}


def test_a_write_that_keeps_failing_holds_back_only_the_re_keys_it_touches() -> None:
    # A hub whose write keeps failing kept every other re-keyed item's old
    # copy too, for as long as it failed.
    harness = _harness()
    before, after = _SPELLINGS
    after = {**after, "/corpus/v.txt": [[(_V, _C, 1)]]}
    vendor = _id(_V)
    harness.run(before)

    harness.run(after, fail=_rejecting("graph.upsert_entities", vendor))

    want = expected_state(after, harness.config)
    stale = set(harness.state()["graph_edges"]) - set(want["graph_edges"])
    assert stale == set()
    harness.run(after)
    assert harness.state() == want


def test_a_noted_edge_the_delta_states_again_takes_its_endpoints(
    monkeypatch,
) -> None:
    # The noted edge cites a unit the delta writes again under its id (a
    # unit id kept while its document changed: content-keyed, as the
    # fixture once was), and the delta states the edge again from it. The
    # delta resolves its subject to the other spelling now; the noted
    # spelling's entity went with the document's removal. Joined with the
    # noted endpoints, the edge pointed at nothing in every later commit.
    def content_keyed(path: str, index: int, unit: Any) -> str:
        digest = hashlib.sha256(f"{path}|{index}|{sorted(unit)}".encode()).hexdigest()
        return str(uuid.UUID(digest[:32]))

    units: dict[tuple[str, int], Any] = {}
    records = incremental_runs._unit_records

    def keyed_records(
        path, document_id, index, unit, conformance
    ):  # noqa: ANN001, ANN202
        units[(document_id, index)] = (path, unit)
        return records(path, document_id, index, unit, conformance)

    def unit_id(document_id: str, index: int) -> str:
        path, unit = units[(document_id, index)]
        return content_keyed(path, index, unit)

    monkeypatch.setattr(incremental_runs, "_unit_id", unit_id)
    monkeypatch.setattr(incremental_runs, "_unit_records", keyed_records)
    kept = [("주식회사 한빛", _D, 1), ("주식회사 한빛", "Acme", 1), ("(주)한빛", _B, 1)]
    before: Corpus = {"/corpus/a.txt": [kept]}
    after: Corpus = {
        "/corpus/a.txt": [kept, [(_D, _B, 1)]],
        "/corpus/d.txt": [[(_D, "주식회사 한빛", 1)]],
    }
    rejected = [r.id for r in extract(before).relationships if r.target_name == "Acme"]
    harness = _harness()
    harness.run(before, fail=_rejecting("graph.upsert_relationships", *rejected))
    assert _note(harness) is not None

    harness.run(after)

    assert harness.state() == expected_state(after, harness.config)
    assert _note(harness) is None


@pytest.mark.parametrize(
    "spellings",
    [("Acme", "Acme Inc", "Acme Corp"), ("(주)한빛", "주식회사 한빛", "한빛")],
    ids=["en", "ko"],
)
def test_a_claim_derived_by_a_removal_resolves_against_the_noted_entities(
    spellings: tuple[str, str, str],
) -> None:
    # The removal of b.txt derives the claim of c.txt about the third
    # spelling. The first one, stated only in a.txt, is a noted write (its
    # graph write was rejected): resolved without it, the claim's object
    # was the third spelling, an entity no store holds, and no later run
    # touched the claim again.
    first, second, third = spellings
    mid: Corpus = {
        "/corpus/a.txt": [[(_D, first, 1)]],
        "/corpus/b.txt": [[(_D, _B, 1), (_B, second, 3)]],
        "/corpus/c.txt": [[(_B, third, 3)]],
    }
    after = {path: content for path, content in mid.items() if path != "/corpus/b.txt"}
    (noted,) = [e.id for e in extract(mid).entities if e.name == first]
    harness = _harness()
    harness.run(mid, fail=_rejecting("graph.upsert_entities", noted))
    assert _note(harness) is not None

    harness.run(after)
    harness.run(after)

    assert harness.state() == expected_state(after, harness.config)
    entities = set(harness.graph.ids("entities"))
    claims = harness.vector.data.get("claims", {}).values()
    assert claims and all({c.subject_id, c.object_id} <= entities for c in claims)


@pytest.mark.parametrize(
    "spellings",
    [("Acme Inc", "Acme"), ("주식회사 한빛", "한빛"), ("株式会社ひかり", "ひかり")],
    ids=["en", "ko", "ja"],
)
def test_a_stored_claim_follows_a_noted_entity_the_commit_merges(
    spellings: tuple[str, str],
) -> None:
    # The two spellings are kept apart while a.txt types the first one
    # otherwise; the second one's entity write is rejected, its claim's
    # is not. Removing a.txt merges them: the commit folded the noted
    # entity into the first one, and the stored claim kept its id.
    first, second = spellings
    mid: Corpus = {
        "/corpus/a.txt": [[(_B, first, 2)]],
        "/corpus/c.txt": [[(_D, first, 1), (_D, second, 3)], [(_D, first, 2)]],
    }
    after = {path: content for path, content in mid.items() if path != "/corpus/a.txt"}
    (noted,) = [e.id for e in extract(mid).entities if e.name == second]
    assert len(extract(after).entities) == 2
    harness = _harness()
    harness.run(mid, fail=_rejecting("graph.upsert_entities", noted))
    assert _note(harness) is not None

    harness.run(after)

    assert harness.state() == expected_state(after, harness.config)
    entities = set(harness.graph.ids("entities"))
    claims = harness.vector.data.get("claims", {}).values()
    assert claims and all({c.subject_id, c.object_id} <= entities for c in claims)


@pytest.mark.parametrize("hub", ["Acme", "한빛", "ひかり"], ids=["en", "ko", "ja"])
def test_the_claims_about_a_noted_entity_that_keeps_its_id_are_not_rewritten(
    hub: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A hub whose entity write the vector store keeps rejecting: each run
    # writes it again under its id, so the claims about it stay as they
    # are. Every run read and rewrote each of them.
    corpus: Corpus = {f"/corpus/d{i}.txt": [[(f"V{i}", hub, 3)]] for i in range(12)}
    (hub_id,) = [e.id for e in extract(corpus).entities if e.name == hub]
    harness = _harness()
    harness.run(corpus)
    claims_read: list[str] = []
    claims_written: set[str] = set()
    read, upsert = harness.vector.read_claims, harness.vector.upsert_claims

    def counted_read(ids, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        claims_read.extend(ids)
        return read(ids, *args, **kwargs)

    def counted_upsert(items, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        claims_written.update(c.id for c in items)
        return upsert(items, *args, **kwargs)

    monkeypatch.setattr(harness.vector, "read_claims", counted_read)
    monkeypatch.setattr(harness.vector, "upsert_claims", counted_upsert)
    for run in range(4):
        corpus = {**corpus, f"/corpus/new{run}.txt": [[(f"C{run}", hub, 1)]]}
        rejected = run < 3
        harness.run(
            corpus,
            fail=_rejecting("vector.upsert_entities", hub_id) if rejected else None,
        )
        assert (_note(harness) is not None) is rejected

    # No stored claim is read back or written again (the new documents'
    # claims are written with the delta's index write).
    assert claims_read == []
    assert claims_written == set()
    assert harness.state() == expected_state(corpus, harness.config)


@pytest.mark.parametrize(
    "spellings",
    [
        ("Acme Inc", "Acme", "Globex Corp", "Globex"),
        ("주식회사 한빛", "한빛", "주식회사 누리", "누리"),
        ("株式会社ひかり", "ひかり", "株式会社みどり", "みどり"),
        ("주식회사 한빛", "한빛", "株式会社ひかり", "ひかり"),
        ("株式会社ひかり", "ひかり", "주식회사 한빛", "한빛"),
    ],
    ids=["en", "ko", "ja", "mix", "mix-reversed"],
)
@pytest.mark.parametrize("direction", ["object", "subject"])
def test_a_rejected_claim_re_derived_alone_keeps_its_own_entity(
    spellings: tuple[str, str, str, str], direction: str
) -> None:
    # The claim about the second company's first spelling is rejected; the
    # removal of b.txt re-derives it from d.txt alone, where the third
    # spelling merges both of the second company's. No resolved claim has
    # its id, so the statement of d.txt about the same names is its claim:
    # it took the statement winning a tie-break by id instead, a claim
    # about the first company's entity, which no store holds under that
    # spelling, and no later run touched it again. The companies are the
    # claims' objects, or their subjects (the same statements reversed):
    # both of a claim's names tell its statement apart.
    first, first_short, second, second_short = spellings
    mid: Corpus = {
        "/corpus/b.txt": [[(_C, second, 1)]],
        "/corpus/c.txt": [[(_C, first_short, 1)], [(_B, first_short, 1)]],
        "/corpus/d.txt": [[(_C, first, 3), (_C, second, 1), (_C, second_short, 3)]],
    }
    if direction == "subject":
        mid = {
            path: [
                [(target, source, k) for source, target, k in unit] for unit in units
            ]
            for path, units in mid.items()
        }
    after = {path: content for path, content in mid.items() if path != "/corpus/b.txt"}
    rejected = [
        c.id for c in extract(mid).claims if getattr(c, f"{direction}_name") == second
    ]
    assert rejected
    harness = _harness()
    harness.config.indexing.max_failure_rate = 0.95
    harness.run(mid, fail=_rejecting("vector.upsert_claims", *rejected))
    assert _note(harness) is not None

    harness.run(after)
    harness.run(after)

    assert harness.state() == expected_state(after, harness.config)
    entities = set(harness.graph.ids("entities"))
    claims = harness.vector.data.get("claims", {}).values()
    assert claims and all({c.subject_id, c.object_id} <= entities for c in claims)


@pytest.mark.parametrize(
    "hub", ["Acme", "주식회사 한빛", "株式会社ひかり"], ids=["en", "ko", "ja"]
)
def test_a_hub_over_the_graph_property_cap_converges(hub: str) -> None:
    # With the summaries off, a hub's description over many documents is
    # longer than the graph's property cap (indexing.neptune
    # .property_max_length): the graph copy is the full build's cut there,
    # the vector copy the whole of it, through additions that sort first
    # and last, a rejected hub write, a removal killed in its writes, and
    # clean runs.
    config = Config()
    config.indexing.max_failure_rate = 0.95
    config.processing.graph_extraction.description_summarization.enabled = False
    corpus: Corpus = {
        f"/corpus/d{i:03d}.txt": [[(f"Vendor{i}", hub, 3)]] for i in range(160)
    }
    (hub_id,) = [e.id for e in extract(corpus, config).entities if e.name == hub]
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(corpus)
    (stored,) = [e for e in harness.vector.data["entities"].values() if e.id == hub_id]
    assert len(stored.description or "") > config.indexing.neptune.property_max_length
    assert harness.state() == expected_state(corpus, config)

    corpus = {"/corpus/a.txt": [[("Carrier", hub, 1)]], **corpus}
    harness.run(corpus, fail=_rejecting("vector.upsert_entities", hub_id))
    assert _note(harness) is not None
    corpus = {k: v for k, v in corpus.items() if k != "/corpus/d005.txt"}
    corpus["/corpus/z.txt"] = [[("Buyer", hub, 2)]]
    assert harness.run(corpus, interrupt="mid_index_delta", k=2)
    harness.run(corpus)
    harness.run(corpus)

    assert harness.state() == expected_state(corpus, config)
    assert harness.registry_problems(corpus) == []
    assert _note(harness) is None


@pytest.mark.parametrize(
    "hub", ["Acme", "주식회사 한빛", "株式会社ひかり"], ids=["en", "ko", "ja"]
)
def test_an_edge_over_the_graph_property_cap_keeps_its_whole_vector_copy(
    hub: str,
) -> None:
    # One edge stated in many documents: its description is over the
    # graph's property cap, so only the vector copy holds the whole of it
    # (state() compares that copy too); through additions, a rejected edge
    # write and a removal killed in its writes it stays the full build's.
    config = Config()
    config.indexing.max_failure_rate = 0.95
    config.processing.graph_extraction.description_summarization.enabled = False
    corpus: Corpus = {
        f"/corpus/d{i:03d}.txt": [[("Vendor", hub, 2)]] for i in range(170)
    }
    (edge,) = extract(corpus, config).relationships
    assert len(edge.description or "") > config.indexing.neptune.property_max_length
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(corpus)
    assert harness.state() == expected_state(corpus, config)

    corpus = {"/corpus/a.txt": [[("Vendor", hub, 1), ("Carrier", hub, 3)]], **corpus}
    harness.run(corpus, fail=_rejecting("vector.upsert_relationships", edge.id))
    assert _note(harness) is not None
    corpus = {k: v for k, v in corpus.items() if k != "/corpus/d005.txt"}
    corpus["/corpus/z.txt"] = [[("Vendor", hub, 2)]]
    assert harness.run(corpus, interrupt="mid_removal", k=1)
    harness.run(corpus)
    harness.run(corpus)

    assert harness.state() == expected_state(corpus, config)
    assert _note(harness) is None


@pytest.mark.parametrize(
    "spellings",
    [
        ("Acme Inc", "Acme", _D),
        ("주식회사 한빛", "한빛", _D),
        ("株式会社ひかり", "ひかり", _D),
        ("주식회사 한빛", "한빛", "株式会社ひかり"),
    ],
    ids=["en", "ko", "ja", "mix"],
)
def test_a_kept_edge_whose_write_failed_over_the_rate_survives_a_removal(
    spellings: tuple[str, str, str],
) -> None:
    # a.txt and b.txt state one edge at the long spelling, c.txt the short
    # one with the same type, so the build folds them. a.txt changes to type
    # the short one otherwise: its removal folds the stored entities into
    # the short spelling and re-points b.txt's edge at it, the commit splits
    # them again, and every graph edge write is rejected (over the rate, so
    # a.txt is not recorded). b.txt's edge is left at the short spelling,
    # which only a.txt and c.txt state: removing them deleted it with that
    # entity, and nothing noted it for the commit, so it was lost.
    long, short, other = spellings
    before: Corpus = {
        "/corpus/a.txt": [[(other, long, 1)]],
        "/corpus/b.txt": [[(other, long, 1)]],
        "/corpus/c.txt": [[(short, _B, 1)], [(short, _B, 2)]],
    }
    mid = {**before, "/corpus/a.txt": [[(short, _B, 2)]]}
    after = {"/corpus/b.txt": before["/corpus/b.txt"]}
    (kept,) = extract(after).relationships
    harness = _harness()
    harness.config.indexing.max_failure_rate = 0.95
    harness.run(before)

    harness.run(mid, fail=lambda at, _: at == "graph.upsert_relationships")

    assert _statuses(harness)["/corpus/a.txt"] is DocStatus.PENDING
    note = _note(harness)
    assert note is not None
    assert kept.id in {r["id"] for r in note.get("relationships", [])}

    harness.run(after)
    harness.run(after)

    assert kept.id in harness.graph.ids("relationships")
    assert kept.id in harness.vector.ids("relationships")
    assert harness.state() == expected_state(after, harness.config)
    assert harness.registry_problems(after) == []
    assert _note(harness) is None


@pytest.mark.parametrize(
    "spellings",
    [
        ("Acme", "Acme Corp", "Acme Inc", "ACME"),
        ("주식회사 한빛", "한빛", "한빛 주식회사", "(주)한빛"),
        ("株式会社ひかり", "ひかり", "ひかり株式会社", "(株)ひかり"),
        ("株式会社ひかり", "주식회사 한빛", "한빛", "ひかり"),
    ],
    ids=["en", "ko", "ja", "mix"],
)
def test_a_noted_edge_a_retried_document_states_again_is_written_once(
    spellings: tuple[str, str, str, str],
) -> None:
    # Every vector edge write of the run adding c.txt is rejected (over the
    # rate: c.txt is not recorded, and the edges are noted). a.txt's
    # Buyer -> second spelling now cites c.txt's second unit too, which
    # states Buyer and the third spelling (one key). The next run removes
    # a.txt and extracts c.txt again: the noted edge kept c.txt's unit, so
    # it was written beside the delta's Buyer -> third spelling at the
    # second one's entity, which a.txt's removal deleted, and its graph
    # write failed on every later run. The delta states that unit again:
    # the noted items are cut to the units it does not write.
    first, second, third, fourth = spellings
    before: Corpus = {
        "/corpus/d.txt": [[(_V, fourth, 3)]],
        "/corpus/a.txt": [[(_B, second, 2), (_D, first, 3)]],
    }
    mid = {
        **before,
        "/corpus/c.txt": [[(_V, fourth, 3), (first, _B, 2)], [(_B, third, 2)]],
    }
    after = {path: content for path, content in mid.items() if path != "/corpus/a.txt"}
    harness = _harness()
    harness.config.indexing.max_failure_rate = 0.2
    harness.run(before)

    harness.run(mid, fail=lambda at, _: at == "vector.upsert_relationships")

    assert _statuses(harness)["/corpus/c.txt"] is DocStatus.PENDING
    assert _note(harness) is not None

    harness.run(after)

    assert harness.extracted == ["/corpus/c.txt"]
    assert set(_statuses(harness).values()) == {DocStatus.PROCESSED}
    assert harness.state() == expected_state(after, harness.config)
    assert harness.registry_problems(after) == []
    assert _note(harness) is None
