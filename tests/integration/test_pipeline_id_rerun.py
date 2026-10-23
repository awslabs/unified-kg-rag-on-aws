# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A pipeline id run again on the embedded stores (``backend: local``).

A finished pipeline id run again resumes: it restores its delta and its
stages' cached outputs. When another pipeline id had written a document of
the delta since (left it PENDING or FAILED under other settings, or removed
and retried it), the restored delta still read as applying, and the rerun
indexed the document over that run's artifacts. The registry records which
pipeline id wrote each record (``DocStatusRecord.written_by``), and a record
of the delta another pipeline id wrote makes the resume compute the delta
again.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_ingestion_stages import _source_text
from tests.integration.test_name_variant_convergence import _Died, _dying, _respond
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.integration

_EXTRACTION = "extraction expert"


def _setup(
    tmp: Path, fails: Callable[[str], bool]
) -> tuple[Config, Any, dict[str, bool]]:
    """Embedded config + providers whose extraction fails on the units
    ``fails`` matches while ``failing["on"]``."""
    config = e2e._config(tmp / "stores", tmp / "viz")
    config.processing.claim_extraction.enabled = True
    config.processing.document_parsing.source_scope = "rerun-corpus"
    failing = {"on": False}

    def respond(system: str, human: str) -> str:
        if failing["on"] and _EXTRACTION in system and fails(_source_text(human)):
            raise RuntimeError("model down")
        return _respond(system, human)

    providers = e2e._providers(
        config, FakeOllama(respond, "x", e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    return config, providers, failing


def _write_corpus(source: Path, files: dict[str, str]) -> None:
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True)
    for name, text in files.items():
        (source / name).write_text(text)


# --- another pipeline id wrote a document of the delta -----------------------

_D6 = "Acme Inc and Buyer and Depot trade goods 1."


def _relationship_descriptions(
    tmp: Path, text: str, steps: list[tuple[bool, str, str]]
) -> list[str]:
    """Run (gleaning on, pipeline id, how: "" / "kill" / "fail") steps over
    one document; the stored relationship descriptions."""
    config, providers, failing = _setup(
        tmp, lambda unit: "today" in unit and "Acme" not in unit
    )
    config.processing.gleaning.max_rounds = 1
    source = tmp / "corpus"
    _write_corpus(source, {"d6.txt": text})
    for gleaning, pipeline_id, how in steps:
        config.processing.gleaning.enabled = gleaning
        failing["on"] = how == "fail"
        if how == "kill":
            dying = _dying("upsert_relationships", 1, True, SQLiteGraphIndexer)
            with (
                patch.object(SQLiteGraphIndexer, "upsert_relationships", dying),
                pytest.raises(_Died),
            ):
                e2e._ingest(config, providers, source, tmp / pipeline_id)
        else:
            e2e._ingest(config, providers, source, tmp / pipeline_id)
    failing["on"] = False
    stored = e2e._stored(
        config, providers, config.indexing.opensearch.relationships_index_prefix
    )
    return sorted(r["_source"].get("description", "") for r in stored)


@pytest.mark.parametrize("how", ["kill", "fail"])
def test_a_rerun_pipeline_id_recomputes_a_delta_another_id_left_half_written(
    tmp_path: Path, how: str
) -> None:
    # P1 indexes d6; C (gleaning on) leaves it PENDING (killed while writing)
    # or FAILED (a unit failed); P1 run again must not index its old delta
    # over C's gleaned edge.
    text = _D6 + " Nobody trades cash 2 today." if how == "fail" else _D6
    full = _relationship_descriptions(tmp_path / "full", text, [(False, "F", "")])
    rerun = _relationship_descriptions(
        tmp_path / "inc",
        text,
        [(False, "P1", ""), (True, "C", how), (False, "P1", "")],
    )
    assert rerun == full


_D1 = "Acme LLC trade goods 2. Acme Inc and Acme LLC and Buyer trade cash 2."
_D5 = (
    "Acme LLC and Acme and Acme Inc trade goods 3. "
    "Acme and Depot and Buyer trade parts 2."
)


def _entities(tmp: Path, steps: list[tuple[dict[str, str], str, str]]) -> list[tuple]:
    """Run (corpus, pipeline id, how: "" / "fail" / "die") steps; the
    stored entities."""
    config, providers, failing = _setup(tmp, lambda unit: "Acme Inc" in unit)
    source = tmp / "corpus"
    run_stage = DataIngestionPipeline._run_stage

    def dying(self: Any, stage: Any, context: Any) -> Any:
        if stage.name == "graph_extraction":
            raise _Died
        return run_stage(self, stage, context)

    for corpus, pipeline_id, how in steps:
        _write_corpus(source, corpus)
        failing["on"] = how == "fail"
        if how == "die":
            with (
                patch.object(DataIngestionPipeline, "_run_stage", dying),
                pytest.raises(_Died),
            ):
                e2e._ingest(config, providers, source, tmp / pipeline_id)
        else:
            e2e._ingest(config, providers, source, tmp / pipeline_id)
        failing["on"] = False
    graph = providers.build_graph_indexer()
    try:
        ids = [entity_id for entity_id, _ in graph.read_entity_names()]
        return sorted(
            (e.id, e.name, e.type, e.description) for e in graph.read_entities(ids)
        )
    finally:
        graph.close()


@pytest.mark.parametrize("how", ["fail", "die"])
def test_a_rerun_pipeline_id_recomputes_a_delta_another_id_removed_and_retried(
    tmp_path: Path, how: str
) -> None:
    # P1 deletes d1 (re-keying the shared entity), P2 attempts d1 again
    # without committing it; P0 run again must not index d1 from outputs
    # resolved against its first run's store.
    both = {"d1.txt": _D1, "d5.txt": _D5}
    full = _entities(tmp_path / "full", [(both, "F", "")])
    rerun = _entities(
        tmp_path / "inc",
        [
            (both, "P0", ""),
            ({"d5.txt": _D5}, "P1", ""),
            (both, "P2", how),
            (both, "P0", ""),
        ],
    )
    assert rerun == full
