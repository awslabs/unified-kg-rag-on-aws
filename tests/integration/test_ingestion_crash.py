# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A killed ingestion process, and the run after it (embedded stores).

The crash tests elsewhere raise in-process; here an incremental run in a
child process is killed with SIGKILL halfway through the indexing stage
(nothing unwinds: no checkpoint upload, no lease release, no attempt
closed). Its lease, held as by another host's run (the SQLite registry
hands this host's killed run's on at once), blocks the next run until it
expires; then a rerun of the same pipeline id takes the lease over,
resumes, and leaves the stores
and the registry as a fresh build of the same corpus does.
"""

from __future__ import annotations

import os
import signal
import subprocess  # nosec B404 - runs the test's own driver module
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.rerank import FakeCrossEncoder
from tests.fixtures.killable_ingest import LEASE_SECONDS, crash_config
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.domain.models import PipelineStageStatus
from unified_kg_rag.shared import NamespaceLeaseError

pytestmark = [pytest.mark.integration, pytest.mark.slow]

ROOT = Path(__file__).parents[2]
no_aws = e2e.no_aws


def _write(source: Path, files: dict[str, str | None]) -> None:
    source.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        if text is None:
            (source / name).unlink()
        else:
            (source / name).write_text(text, encoding="utf-8")


_FIRST = {
    "contract.txt": "Vendor supplies widgets to Buyer.",
    "logistics.txt": "Depot stores widgets for Vendor.",
}
_SECOND = {
    "freight.txt": "Carrier moves gadgets for Depot.",
    "logistics.txt": e2e.ANSWER,
    "contract.txt": None,
}


def _providers(config: Any) -> Any:
    return e2e._providers(
        config, FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM), FakeCrossEncoder("r")
    )


def _state(config: Any, providers: Any) -> tuple[Any, ...]:
    return (
        e2e._entity_names(providers),
        e2e._stored_texts(config, providers),
        e2e._registry(providers),
    )


@pytest.mark.timeout(240)
def test_a_killed_run_is_taken_over_once_its_lease_expires_and_converges(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    stores, source = tmp_path / "stores", tmp_path / "corpus"
    config = crash_config(stores)
    providers = _providers(config)
    _write(source, _FIRST)
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)

    # The incremental run, killed with the indexing half done.
    _write(source, _SECOND)
    cache, marker = tmp_path / "run-2", tmp_path / "indexing.marker"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "AWS_EC2_METADATA_DISABLED": "true"}
    child = subprocess.Popen(  # nosec B603 - the test's own module
        [
            sys.executable,
            "-m",
            "tests.fixtures.killable_ingest",
            str(stores),
            str(source),
            str(cache),
            str(marker),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 120
        while not marker.exists():
            assert child.poll() is None, child.communicate()[0].decode()[-3000:]
            assert time.monotonic() < deadline, "the run never reached indexing"
            time.sleep(0.1)
    finally:
        child.send_signal(signal.SIGKILL)
        child.communicate()
    killed_at = time.time()
    assert child.returncode == -signal.SIGKILL

    # It died before it committed: the new and changed documents are not
    # recorded as processed.
    registry = providers.build_doc_status()
    assert e2e._registry(providers) != {
        "freight.txt": e2e.DocStatus.PROCESSED,
        "logistics.txt": e2e.DocStatus.PROCESSED,
    }
    holder = registry.lease_holder("default")
    assert holder is not None and holder[1] > time.time()
    # The SQLite registry hands a killed run's lease of this host on at once
    # (tests/unit/test_processes.py); a holder on another host it cannot
    # judge, whose lease runs to its expiry: as such a holder's.
    assert registry.try_acquire_lease(
        "default", "another-host:1:2:3:killed", holder[1] - time.time()
    )
    holder = registry.lease_holder("default")
    assert holder is not None and holder[1] > time.time()
    # Its lease stops the next run before it reads or writes anything.
    with pytest.raises(NamespaceLeaseError):
        e2e._ingest(config, _providers(config), source, cache)

    # Once it expires (no renewal since the kill), the rerun of the same
    # pipeline id takes the lease over and finishes the run.
    time.sleep(max(0.0, holder[1] - time.time()) + 0.5)
    assert time.time() - killed_at <= LEASE_SECONDS + 5
    context = e2e._ingest(config, _providers(config), source, cache)
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert registry.lease_holder("default") is None

    # As a fresh build of the same corpus.
    fresh = crash_config(tmp_path / "fresh")
    fresh_providers = _providers(fresh)
    context = e2e._ingest(fresh, fresh_providers, source, tmp_path / "fresh-run")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert _state(config, _providers(config)) == _state(fresh, fresh_providers)
    assert e2e._entity_names(fresh_providers) == {"Vendor", "Depot", "Carrier"}


def _reports_state(providers: Any) -> dict[str, tuple[Any, ...]]:
    """Each stored community report's summary, outdated flag and units."""
    from unified_kg_rag.domain.ingestion.report_references import (
        REPORT_OUTDATED_FIELD,
    )

    vectors = providers.build_vector_indexer()
    try:
        pages = vectors.scan_items("community_reports", "default")
        assert pages is not None
        return {
            item["id"]: (
                item["summary"],
                bool(item.get(REPORT_OUTDATED_FIELD)),
                tuple(sorted(item["text_unit_ids"])),
            )
            for page in pages
            for item in page
        }
    finally:
        vectors.close()


@pytest.mark.timeout(300)
def test_a_run_killed_in_the_report_refresh_resumes_to_the_unkilled_run(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # RV3-I1: a document is deleted and a report it fed is kept for the other
    # document's units, flagged outdated. Its id lived in memory only, and
    # the deleted document's registry record went before the commit, so a
    # run killed in the refresh's report call resumed with nothing to
    # refresh: the report kept the deleted text, flagged, listed nowhere.
    from tests.integration.test_outdated_reports import _DELETED, _KEPT, _Model

    def build(stores: Path) -> tuple[Any, Any]:
        config = crash_config(stores)
        server = FakeOllama(_Model(), e2e.ANSWER, e2e.DIM)
        return config, e2e._providers(config, server, FakeCrossEncoder("r"))

    # One corpus folder: a text unit's id derives from its document's path.
    source = tmp_path / "corpus"

    def first_run(name: str) -> tuple[Any, Any, Path]:
        config, providers = build(tmp_path / name / "stores")
        _write(source, {"a.txt": _KEPT, "b.txt": _DELETED})
        context = e2e._ingest(config, providers, source, tmp_path / name / "run-1")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        _write(source, {"b.txt": None})
        return config, providers, source

    # The unkilled run.
    config, providers, source = first_run("unkilled")
    context = e2e._ingest(config, providers, source, tmp_path / "unkilled" / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    unkilled = _reports_state(providers)
    assert not any(outdated for _, outdated, _ in unkilled.values())

    # The same run, killed in the refresh's report call.
    config, providers, source = first_run("killed")
    cache = tmp_path / "killed" / "run-2"
    _kill_in_refresh(tmp_path / "killed" / "stores", source, cache)
    # Killed after the removal flagged the report and the deleted
    # document's record went.
    assert any(outdated for _, outdated, _ in _reports_state(providers).values())
    assert "b.txt" not in e2e._registry(providers)

    # The resume (the same pipeline id; the SQLite registry hands this
    # host's killed run's lease on at once) writes the report again.
    context = e2e._ingest(config, _providers_of(config), source, cache)
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert _reports_state(providers) == unkilled
    vectors = providers.build_vector_indexer()
    try:
        record = vectors.read_build_record("default")
        assert record is not None and record["outdated_reports"] == []
        # The note that carried the reports across the kill is gone.
        from unified_kg_rag.application.storage.stored_items import (
            OUTDATED_REPORTS,
        )

        assert OUTDATED_REPORTS not in vectors.read_extraction_records(
            [OUTDATED_REPORTS], suffix="default"
        )
    finally:
        vectors.close()


def _kill_in_refresh(stores: Path, source: Path, cache: Path) -> None:
    """Run an ingestion of ``source`` on ``stores`` in a child process and
    SIGKILL it in the refresh's first report call (after the removal)."""
    marker = cache.parent / "report.marker"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "AWS_EC2_METADATA_DISABLED": "true"}
    child = subprocess.Popen(  # nosec B603 - the test's own module
        [
            sys.executable,
            "-m",
            "tests.fixtures.killable_ingest",
            str(stores),
            str(source),
            str(cache),
            str(marker),
            "report",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 120
        while not marker.exists():
            assert child.poll() is None, child.communicate()[0].decode()[-3000:]
            assert time.monotonic() < deadline, "the run never reached the refresh"
            time.sleep(0.1)
    finally:
        child.send_signal(signal.SIGKILL)
        child.communicate()
    assert child.returncode == -signal.SIGKILL
    marker.unlink()


def _outdated_note(providers: Any) -> dict[str, Any] | None:
    from unified_kg_rag.application.storage.stored_items import OUTDATED_REPORTS

    vectors = providers.build_vector_indexer()
    try:
        return vectors.read_extraction_records(
            [OUTDATED_REPORTS], suffix="default"
        ).get(OUTDATED_REPORTS)
    finally:
        vectors.close()


def _outdated_record(providers: Any) -> list[str]:
    vectors = providers.build_vector_indexer()
    try:
        record = vectors.read_build_record("default")
        assert record is not None
        return list(record["outdated_reports"])
    finally:
        vectors.close()


# Two communities of the outdated-report suite's names, each from two
# documents: a deletion in one leaves the other's report as it was.
_TWO_GROUPS = {
    "a.txt": "Vendor supplies widgets to Buyer. Depot stores widgets for Vendor.",
    "b.txt": "Carrier moves widgets for Vendor and Buyer every week.",
    "c.txt": "Alpha supplies gears to Beta. Gamma stores gears for Alpha.",
    "d.txt": "Omega moves gears for Alpha and Beta every week.",
}


@pytest.mark.timeout(300)
def test_a_crash_note_outlives_a_later_commit_that_writes_no_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_aws  # noqa: F811
) -> None:
    # R2A-2: a run killed in the refresh leaves its outdated report in its
    # note only. A later run whose commit tolerated one failed write wrote
    # no report again: its build record listed only its own, and it deleted
    # the note, so the killed run's report kept the deleted text, flagged,
    # listed nowhere, for good.
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    stores, source = tmp_path / "stores", tmp_path / "corpus"
    config = crash_config(stores)
    providers = _providers_of(config)
    _write(source, _TWO_GROUPS)
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    _write(source, {"b.txt": None})
    _kill_in_refresh(stores, source, tmp_path / "run-2")
    note = _outdated_note(providers)
    assert note is not None and len(note["report_ids"]) == 1
    (killed,) = note["report_ids"]

    # Another run deletes d.txt; one write of its commit fails, which the
    # write gate tolerates (one in 31 text units), so no report is written.
    _write(source, {"d.txt": None})
    _write(source, {f"e{i}.txt": f"Gamma stores gears in bin {i}." for i in range(30)})
    index_delta = IndexingManager.index_delta

    def one_failed_write(self: Any, **items: Any) -> Any:
        results = index_delta(self, **items)
        stats = next(s for s in results.values() if s and s.total_items >= 20)
        stats.failed_items += 1
        return results

    monkeypatch.setattr(IndexingManager, "index_delta", one_failed_write)
    context = e2e._ingest(config, _providers_of(config), source, tmp_path / "run-3")
    monkeypatch.undo()
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    # The record lists the killed run's report (a rebuild reason names it),
    # and the note stays for the run that writes it again.
    assert killed in _outdated_record(providers)
    kept = _outdated_note(providers)
    assert kept is not None and killed in kept["report_ids"]
    assert set(note["candidates"]) <= set(kept["candidates"])

    context = e2e._ingest(config, _providers_of(config), source, tmp_path / "run-4")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    reports = _reports_state(providers)
    assert not any(outdated for _, outdated, _ in reports.values())
    assert not any("Carrier" in summary for summary, _, _ in reports.values())
    assert _outdated_record(providers) == []
    assert _outdated_note(providers) is None


@pytest.mark.timeout(300)
def test_a_resume_reads_the_members_its_killed_removal_stripped(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # R2A-7: Carrier's only unit the report cites goes with b.txt; c.txt,
    # added after the report was written, keeps Carrier. The report's
    # refresh finds Carrier only among the entities the removal stripped
    # (the member candidates), which a resume reads off the killed run's
    # note.
    corpus = {
        "a.txt": "Vendor supplies widgets to Buyer. Depot stores widgets for Vendor.",
        "b.txt": "Carrier moves widgets for Vendor and Buyer every week.",
    }

    def setup(name: str) -> tuple[Any, Any, Path]:
        config = crash_config(tmp_path / name / "stores")
        providers = _providers_of(config)
        source = tmp_path / "corpus"
        if source.exists():
            for path in source.iterdir():
                path.unlink()
        _write(source, corpus)
        for run, files in (("run-1", {}), ("run-2", {"c.txt": "Carrier owns trucks."})):
            _write(source, files)
            context = e2e._ingest(config, providers, source, tmp_path / name / run)
            assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        _write(source, {"b.txt": None})
        return config, providers, source

    config, providers, source = setup("unkilled")
    context = e2e._ingest(config, providers, source, tmp_path / "unkilled" / "run-3")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    unkilled = _reports_state(providers)
    assert any("Carrier" in summary for summary, _, _ in unkilled.values())

    config, providers, source = setup("killed")
    cache = tmp_path / "killed" / "run-3"
    _kill_in_refresh(tmp_path / "killed" / "stores", source, cache)
    context = e2e._ingest(config, _providers_of(config), source, cache)
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert _reports_state(providers) == unkilled


def _providers_of(config: Any) -> Any:
    from tests.integration.test_outdated_reports import _Model

    return e2e._providers(
        config, FakeOllama(_Model(), e2e.ANSWER, e2e.DIM), FakeCrossEncoder("r")
    )
