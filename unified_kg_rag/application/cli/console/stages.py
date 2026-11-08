# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An ingestion job's progress: the stages its run will execute, and how far
the run got, read from the ``pipeline_metadata.json`` the pipeline writes
after every stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.ingestion.methodologies import selected_stages
from unified_kg_rag.domain.models import Config, PipelineStageType

METADATA_FILENAME = "pipeline_metadata.json"


def planned_stages(config: Config) -> list[str]:
    """The stages a run of ``config`` with every stage enabled executes, in
    order: those the selected methodologies need, less those a config
    section switches off (``DataIngestionPipeline._stage_enabled`` without an
    explicit stage list, which the console never passes)."""
    needed = selected_stages(config)
    return [
        stage.value
        for stage in DataIngestionPipeline.STAGE_CLASSES
        if stage in needed
        and DataIngestionPipeline.CONFIG_ENABLED.get(stage, lambda _: True)(config)
    ]


def read_metadata(cache_directory: Path, pipeline_id: str) -> dict[str, Any] | None:
    """The run's metadata, or ``None`` before its first write (or while one
    is half written: the next poll reads it)."""
    path = cache_directory / pipeline_id / METADATA_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def stage_progress(
    plan: list[str],
    metadata: dict[str, Any] | None,
    *,
    running: bool,
    started_after: str | None = None,
) -> list[dict[str, Any]]:
    """Each planned stage's state: ``completed`` / ``failed`` / ``skipped``
    from the metadata; ``running`` for the stage the metadata says started
    (else the first one without a result) while the job runs, ``pending``
    after it; ``stopped`` for that stage once the job ended, ``not_run``
    after it.

    A resumed run keeps the results of the stages an earlier run finished;
    those started before ``started_after`` (ISO time) are ``restored``.
    """
    results = {
        str(r.get("stage_name")): r
        for r in (metadata or {}).get("stage_results") or []
        if isinstance(r, dict)
    }
    stages: list[dict[str, Any]] = []
    current_found = False
    names = list(plan) + [n for n in results if n not in plan]
    order = [s.value for s in PipelineStageType]
    names.sort(key=lambda n: order.index(n) if n in order else len(order))
    for name in names:
        result = results.get(name)
        if result is not None:
            status = str(result.get("status") or "completed")
            if status == "running":
                # Saved as it started: the stage the run is in, or was in
                # when it stopped (the next one did not run).
                current_found = True
                status = "running" if running else "stopped"
            start = str(result.get("start_time") or "")
            if (
                started_after
                and status == "completed"
                and start
                and start < started_after
            ):
                status = "restored"
            metrics = result.get("metrics") or {}
            stages.append(
                {
                    "name": name,
                    "status": status,
                    "duration_seconds": result.get("duration_seconds"),
                    "input_count": result.get("input_count"),
                    "output_count": result.get("output_count"),
                    "error": result.get("error_message"),
                    "failed_items": _failed_items(metrics),
                    "llm_calls": (metrics.get("llm_usage") or {}).get("llm_calls"),
                }
            )
            continue
        if not current_found:
            current_found = True
            status = "running" if running else "stopped"
        else:
            status = "pending" if running else "not_run"
        stages.append({"name": name, "status": status})
    return stages


def _failed_items(metrics: dict[str, Any]) -> list[str]:
    """The files or units a stage reports failed, as short strings."""
    items: list[str] = []
    for key in ("failed_files", "failed_units", "failed_documents"):
        value = metrics.get(key)
        if isinstance(value, list):
            for entry in value[:20]:
                if isinstance(entry, dict):
                    name = (
                        entry.get("file") or entry.get("file_name") or entry.get("id")
                    )
                    error = entry.get("error") or entry.get("reason")
                    items.append(f"{name}: {error}" if error else str(name))
                else:
                    items.append(str(entry))
    return items
