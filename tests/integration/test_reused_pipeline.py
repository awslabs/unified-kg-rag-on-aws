# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A pipeline run again after a failed commit (library use) ends where a new
pipeline would (embedded stores).

The indexing manager holds a commit's re-key, split and drop decisions in
memory until the old ids are deleted. A commit that failed before that kept
them, and the next run of the same pipeline deleted the old ids although
their replacements were never written (a claim was lost).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_name_variant_convergence as nv
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import (
    PipelineConfig,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.ports.indexer import IndexingStats

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

no_aws = e2e.no_aws

RUNS = nv.RUNS_COMMIT_SPLIT_CLAIM


def _write(source: Path, corpus: dict[str, str]) -> None:
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True)
    for name, text in corpus.items():
        (source / name).write_text(text)


class _Stores:
    """A config, providers and corpus directory under ``tmp``."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.config = e2e._config(tmp / "stores", tmp / "viz")
        self.config.processing.claim_extraction.enabled = True
        self.config.processing.gleaning.max_rounds = 1
        self.config.processing.document_parsing.source_scope = "fuzz-corpus"
        self.providers = e2e._providers(
            self.config,
            FakeOllama(nv._respond, "x", e2e.DIM),
            e2e.FakeCrossEncoder("r"),
        )
        self.source = tmp / "corpus"

    def pipeline(self) -> DataIngestionPipeline:
        pipeline = DataIngestionPipeline(
            self.config,
            PipelineConfig(
                stages_enabled=dict.fromkeys(PipelineStageType, True),
                local_directory=self.tmp / "cache",
            ),
            source_directory=self.source,
            providers=self.providers,
        )
        for stage in pipeline.stages:
            if isinstance(stage, ps.GraphResolutionStage):
                stage.resolver.entity_resolver.use_process_pool = False
                stage.resolver.relationship_resolver.use_process_pool = False
        return pipeline

    def state(self) -> dict[str, list[tuple]]:
        graph = self.providers.build_graph_indexer()
        try:
            ids = [i for i, _ in graph.read_entity_names()]
            entities = graph.read_entities(ids)
            edges = graph.read_relationships(graph.find_incident_relationship_ids(ids))
        finally:
            graph.close()
        prefixes = self.config.indexing.opensearch
        claims = e2e._stored(self.config, self.providers, prefixes.claims_index_prefix)
        return {
            "entities": sorted(
                (e.id, e.name, len(e.text_unit_ids or [])) for e in entities
            ),
            "edges": sorted(
                (r.id, r.source_name, r.target_name, r.weight) for r in edges
            ),
            "claims": sorted(
                (
                    hit["_id"],
                    hit["_source"]["subject_name"],
                    hit["_source"]["object_name"],
                    tuple(sorted(hit["_source"].get("text_unit_ids") or [])),
                )
                for hit in claims
            ),
        }


_write_records = OpenSearchIndexer.write_extraction_records


def _commit_note_fails(self: Any, records: Any, *args: Any, **kwargs: Any) -> Any:
    if "pending-commit" in records:
        stats = IndexingStats(total_items=1)
        stats.add_error("store down", count=1)
        return stats
    return _write_records(self, records, *args, **kwargs)


def _sequence(tmp: Path, reuse: bool, follow: dict[str, str]) -> dict[str, list]:
    """``RUNS`` then ``follow``; the commit of the last of ``RUNS`` fails
    (its note cannot be written). ``reuse`` runs them all with one pipeline,
    else each with a new one."""
    stores = _Stores(tmp)
    pipeline = stores.pipeline()
    try:
        for n, corpus in enumerate([*RUNS, follow]):
            _write(stores.source, corpus)
            if not reuse and n:
                pipeline.close()
                pipeline = stores.pipeline()
            if n == len(RUNS) - 1:
                with patch.object(
                    OpenSearchIndexer, "write_extraction_records", _commit_note_fails
                ):
                    context = pipeline.run(stores.source, pipeline_id=f"run-{n}")
                assert context.status is PipelineStageStatus.FAILED
            else:
                context = pipeline.run(stores.source, pipeline_id=f"run-{n}")
                assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(
                    context
                )
    finally:
        pipeline.close()
    return stores.state()


@pytest.mark.parametrize(
    "follow",
    [
        pytest.param(RUNS[0], id="back-to-the-first-corpus"),
        pytest.param(RUNS[1], id="back-to-the-second-corpus"),
        pytest.param({**RUNS[1], "x.txt": "Depot trade cash 1."}, id="an-edited-one"),
    ],
)
def test_a_pipeline_run_again_after_a_failed_commit_matches_a_new_one(
    tmp_path: Path, follow: dict[str, str]
) -> None:
    fresh = _sequence(tmp_path / "fresh", reuse=False, follow=follow)
    reused = _sequence(tmp_path / "reused", reuse=True, follow=follow)

    assert reused == fresh


def test_a_reused_pipeline_finishes_a_killed_commit_from_its_note(
    tmp_path: Path,
) -> None:
    # Killed after its note and before its claims, the last commit's split
    # is finished by the next run of the same pipeline from the stored note,
    # as a new pipeline would: the in-memory state goes, the note stays.
    stores = _Stores(tmp_path / "inc")
    pipeline = stores.pipeline()
    try:
        for n, corpus in enumerate(RUNS):
            _write(stores.source, corpus)
            if n == len(RUNS) - 1:
                with (
                    patch.object(
                        OpenSearchIndexer,
                        "upsert_claims",
                        nv._dying("upsert_claims", 1, False),
                    ),
                    pytest.raises(nv._Died),
                ):
                    pipeline.run(stores.source, pipeline_id=f"run-{n}")
            context = pipeline.run(stores.source, pipeline_id=f"run-{n}")
            assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    finally:
        pipeline.close()
    full = _Stores(tmp_path / "full")
    _write(full.source, RUNS[-1])
    pipeline = full.pipeline()
    try:
        context = pipeline.run(full.source, pipeline_id="full")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    finally:
        pipeline.close()

    assert stores.state() == full.state()
