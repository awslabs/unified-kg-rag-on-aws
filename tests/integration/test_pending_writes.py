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

from collections.abc import Callable

import pytest

from tests.fixtures.doc_status_records import all_records
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.summarizer import AtomSetSummarizer
from tests.fixtures.incremental_runs import (
    POINTS,
    Corpus,
    IncrementalRun,
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
        u.id for u in extract(_CORPUS).units if u.document_ids == ["/corpus/a.txt"]
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
