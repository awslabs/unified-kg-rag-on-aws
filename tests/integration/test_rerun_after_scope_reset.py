# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A pipeline id of one source scope run again after another scope's run.

A reset of scope A killed mid-rebuild moves scope B's records into A
(PENDING, so the next run removes what the rebuild dropped). The next run
of A deletes them; B is then indexed again. Run again, the A pipeline id
restored its delta and deleted B's documents once more.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_name_variant_convergence import _Died, _dying
from tests.integration.test_pipeline_id_rerun import _setup, _write_corpus
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer

pytestmark = pytest.mark.integration


def test_a_rerun_does_not_delete_documents_another_scope_indexed_again(
    tmp_path: Path,
) -> None:
    config, providers, _ = _setup(tmp_path, lambda unit: False)
    sources = {"A": tmp_path / "a", "B": tmp_path / "b"}
    _write_corpus(sources["A"], {"a1.txt": "Acme Inc and Buyer trade goods 1."})
    _write_corpus(sources["B"], {"b1.txt": "Depot and Globex trade cash 2."})

    def run(scope: str, pipeline_id: str, *, reset: bool = False) -> None:
        config.processing.document_parsing.source_scope = scope
        config.indexing.reset = reset
        e2e._ingest(config, providers, sources[scope], tmp_path / pipeline_id)
        config.indexing.reset = False

    run("A", "P0")
    run("B", "P1")
    dying = _dying("upsert_relationships", 1, True, SQLiteGraphIndexer)
    with (
        patch.object(SQLiteGraphIndexer, "upsert_relationships", dying),
        pytest.raises(_Died),
    ):
        run("A", "P1", reset=True)
    run("A", "P0")  # removes what the killed rebuild dropped (B)
    run("B", "P1")  # B indexed again
    run("A", "P0")  # rerun: restores its delta
    scopes = sorted(
        (r.scope or "").rsplit("|", 1)[-1]
        for r in providers.build_doc_status().list_all()
    )
    assert scopes == ["A", "B"]
