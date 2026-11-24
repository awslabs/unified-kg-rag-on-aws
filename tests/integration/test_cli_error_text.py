# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What ``run-ingestion`` prints of an error whose message is the package's
own: a held namespace lease, a resume that cannot go on, a corrupt metadata
file, an unknown pipeline id under ``--verify-metadata``. The operator reads
the message that says what to do (the holder, the stage to run, the file),
not the generic "unexpected error", on the console and in the JSON status.

The CLI runs in-process over the fake stores (``_Harness``), AWS-free.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.doc_status_records import all_records
from tests.integration.test_pipeline_resume_flows import _Harness
from unified_kg_rag.shared.exceptions import INTERNAL_ERROR_MESSAGE

pytestmark = pytest.mark.integration


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    return _Harness(tmp_path, monkeypatch)


def _failed(
    harness: _Harness, capsys: pytest.CaptureFixture[str], *flags: str
) -> tuple[str, dict[str, Any]]:
    """Run the CLI with ``flags``, then again with ``--output-format json``:
    the first run's text output (whitespace folded) and the second's JSON
    status; both must exit 1."""
    capsys.readouterr()
    assert harness.run(*flags) == 1
    captured = capsys.readouterr()
    text = " ".join((captured.out + captured.err).split())
    assert harness.run(*flags, "--output-format", "json") == 1
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "failed"
    return text, status


def test_a_held_lease_names_its_holder_and_no_rebuild(
    harness: _Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    # Another run is writing the namespace: the run stops before it writes,
    # and says who holds it and what to do (wait), not "a store or model
    # error"; a rebuild is no remedy for a run that is still writing.
    harness.registry.try_acquire_lease("default", "other-host:4242:run", 600)
    text, status = _failed(harness, capsys, "--pipeline-id", "p1")

    assert "is being written by other-host:4242:run" in text
    assert "Wait for that run to finish" in text
    assert INTERNAL_ERROR_MESSAGE not in text
    assert "--force-rebuild" not in text
    assert "Recovery suggestions" not in text
    assert status["error_type"] == "NamespaceLeaseError"
    assert "is being written by other-host:4242:run" in status["error"]
    assert all_records(harness.registry) == []


def test_a_resume_with_an_upstream_stage_missing_says_which_to_run(
    harness: _Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    # A finished run whose graph_resolution output is gone: a later window
    # cannot run on it, and the message names the stage to run first.
    assert harness.run("--pipeline-id", "p1") == 0
    shutil.rmtree(harness.tmp_path / "cache" / "p1" / "graph_resolution")
    flags = (
        "--pipeline-id",
        "p1",
        "--enabled-stages",
        "community_detection,indexing",
        "--yes",
    )
    text, status = _failed(harness, capsys, *flags)

    expected = "Stage 'graph_resolution', upstream of this run's enabled stages"
    assert expected in text
    assert "--enabled-stages including 'graph_resolution'" in text
    assert INTERNAL_ERROR_MESSAGE not in text
    assert expected in status["error"]
    assert status["error"].startswith("Pipeline resume failed: ")


def test_a_corrupt_metadata_file_names_the_file(
    harness: _Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    assert harness.run("--pipeline-id", "p1") == 0
    path = harness.tmp_path / "cache" / "p1" / "pipeline_metadata.json"
    path.write_text('{"pipeline_id": "p1", "status": ', encoding="utf-8")
    text, status = _failed(harness, capsys, "--pipeline-id", "p1", "--yes")

    assert "Corrupted metadata file, invalid JSON" in text
    assert str(path) in text
    assert INTERNAL_ERROR_MESSAGE not in text
    assert "Corrupted metadata file, invalid JSON" in status["error"]


def test_verify_metadata_of_an_unknown_pipeline_says_not_found(
    harness: _Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    text, status = _failed(
        harness, capsys, "--pipeline-id", "nope", "--verify-metadata"
    )

    assert "Pipeline 'nope' not found" in text
    assert INTERNAL_ERROR_MESSAGE not in text
    assert status["error"] == "Pipeline 'nope' not found"


def test_verify_metadata_of_a_corrupt_file_says_so(
    harness: _Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    assert harness.run("--pipeline-id", "p1") == 0
    path = harness.tmp_path / "cache" / "p1" / "pipeline_metadata.json"
    path.write_text("[]", encoding="utf-8")
    _, status = _failed(harness, capsys, "--pipeline-id", "p1", "--verify-metadata")

    assert status["error"] == "Metadata for pipeline 'p1' is corrupted or invalid"
