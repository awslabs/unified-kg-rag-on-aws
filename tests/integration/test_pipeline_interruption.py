# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stopped or late-resumed ``run-ingestion --s3-sync`` keeps its finished stages.

ECS stops a task with SIGTERM (StopTask, a Step Functions timeout) and kills it
once the stop timeout has passed; the next task starts on an empty disk and
reads the checkpoint from S3. The CLI runs in-process over the synthetic
corpus of ``test_pipeline_resume_flows`` with S3 in moto (AWS-free).
"""

from __future__ import annotations

import json
import os
import signal
from datetime import datetime, timedelta
from typing import Any

import boto3
import pytest
from moto import mock_aws

from tests.integration import test_ingestion_stages as stages_test
from tests.integration.test_pipeline_resume_flows import _Harness
from unified_kg_rag.application.ingestion import pipeline as pipeline_module
from unified_kg_rag.application.ingestion import pipeline_stages as ps

pytestmark = pytest.mark.integration

_BUCKET = "pipeline-cache"
_PREFIX = "pipeline-runs"


@pytest.fixture
def s3_harness(tmp_path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    with mock_aws():
        session = boto3.Session(region_name="us-east-1")
        session.client("s3").create_bucket(Bucket=_BUCKET)
        harness = _Harness(tmp_path, monkeypatch)
        fake_providers = pipeline_module.build_providers

        def build_providers(config, boto_session=None, **overrides):  # type: ignore[no-untyped-def]
            providers = fake_providers(config, **overrides)
            providers._boto_session = session
            return providers

        monkeypatch.setattr(pipeline_module, "build_providers", build_providers)
        yield harness, session.client("s3")


def _run(harness: _Harness, cache: str) -> int:
    return harness.run(
        "--pipeline-id",
        "p1",
        "--s3-sync",
        "--s3-bucket-name",
        _BUCKET,
        "--s3-prefix",
        _PREFIX,
        cache=cache,
    )


def _s3_json(s3: Any, name: str) -> dict[str, Any]:
    body = s3.get_object(Bucket=_BUCKET, Key=f"{_PREFIX}/p1/{name}")["Body"]
    return json.loads(body.read())


def _s3_statuses(s3: Any) -> dict[str, str]:
    metadata = _s3_json(s3, "pipeline_metadata.json")
    return {r["stage_name"]: r["status"] for r in metadata["stage_results"]}


def _count_executions(monkeypatch: pytest.MonkeyPatch, stage_class: type) -> list[int]:
    calls: list[int] = []
    original = stage_class.execute

    def execute(stage, context):  # type: ignore[no-untyped-def]
        calls.append(1)
        return original(stage, context)

    monkeypatch.setattr(stage_class, "execute", execute)
    return calls


def test_sigterm_mid_stage_keeps_every_finished_stage(
    s3_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness, s3 = s3_harness
    extractions = _count_executions(monkeypatch, ps.GraphExtractionStage)
    seen: dict[str, dict[str, str]] = {}
    stop = {"on": True}
    original = ps.GraphResolutionStage.execute

    def execute(stage, context):  # type: ignore[no-untyped-def]
        if not stop["on"]:
            return original(stage, context)
        # Already in S3 before any stop: uploaded when gleaning finished.
        seen["before_stop"] = _s3_statuses(s3)
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), "run-ingestion installs no SIGTERM handler"
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM did not stop the stage")

    monkeypatch.setattr(ps.GraphResolutionStage, "execute", execute)
    assert _run(harness, "task-1") == 128 + signal.SIGTERM
    assert seen["before_stop"]["gleaning"] == "completed"
    assert extractions == [1]
    statuses = _s3_statuses(s3)
    assert statuses["graph_extraction"] == statuses["gleaning"] == "completed"
    assert statuses["graph_resolution"] != "completed"
    # The CLI restored the handler it replaced.
    assert not callable(signal.getsignal(signal.SIGTERM))

    # The next task (empty disk) resumes at the stage that was running.
    stop["on"] = False
    calls = harness.llm_calls
    assert _run(harness, "task-2") == 0
    assert extractions == [1]
    assert set(_s3_statuses(s3).values()) == {"completed"}
    assert harness.graph.ids("entities")
    # Only community reports call the model after resolution.
    assert harness.llm_calls > calls


def test_a_resume_after_a_day_reuses_the_s3_checkpoint(
    s3_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness, s3 = s3_harness
    # A full (non-incremental) run, so recomputing a stage calls the model
    # again instead of skipping the unchanged documents.
    harness.incremental = False
    extractions = _count_executions(monkeypatch, ps.GraphExtractionStage)
    assert _run(harness, "task-1") == 0
    calls = harness.llm_calls

    # 25 hours later: entries stamped with the old 24 h checkpoint TTL have
    # expired, while the bucket (30-day lifecycle) still holds them.
    index = _s3_json(s3, "cache_index.json")
    for entry in index["entries"].values():
        entry["expires_at"] = (datetime.now() - timedelta(hours=1)).isoformat()
    s3.put_object(
        Bucket=_BUCKET,
        Key=f"{_PREFIX}/p1/cache_index.json",
        Body=json.dumps(index).encode(),
    )

    # Every stage is still backed by the checkpoint: only indexing re-runs.
    assert _run(harness, "task-2") == 0
    assert extractions == [1]
    assert harness.llm_calls == calls


def test_a_stopped_stage_does_not_repeat_its_answered_model_calls(
    s3_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The stop lands while graph extraction is running: the responses it
    # already had reach S3 with the checkpoint, and the next task (empty
    # disk) answers those prompts from them instead of asking the model.
    harness, s3 = s3_harness
    harness.incremental = False
    extraction_prompts: list[str] = []
    respond = stages_test._respond

    def stop_during_extraction(system: str, human: str) -> str:
        answer = respond(system, human)
        if "knowledge graph extraction expert" in system:
            extraction_prompts.append(human)
            if len(extraction_prompts) == 1:
                os.kill(os.getpid(), signal.SIGTERM)
        return answer

    monkeypatch.setattr(stages_test, "_respond", stop_during_extraction)
    assert _run(harness, "task-1") == 128 + signal.SIGTERM
    assert extraction_prompts
    assert _s3_statuses(s3).get("graph_extraction") != "completed"

    assert _run(harness, "task-2") == 0
    assert _s3_statuses(s3)["graph_extraction"] == "completed"
    # Every text unit was extracted, none twice.
    assert len(extraction_prompts) == len(set(extraction_prompts)) == 2
