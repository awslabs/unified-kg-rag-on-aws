# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental runs store what a full build of the final corpus stores.

Whole-pipeline runs on the embedded stores (``backend: local``, a fake model
server): after an edit or a deletion the shared entities' and edges'
descriptions, ranks and names equal a fresh full build's, so a removed
document's text and edges leave no trace.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.test_extraction_records import _Model, _Store

pytestmark = pytest.mark.integration


def _entities(store: _Store) -> dict[str, tuple]:
    graph = store.providers.build_graph_indexer()
    try:
        ids = [i for i, _ in graph.read_entity_names()]
        return {
            e.name: (
                e.rank,
                len(e.text_unit_ids or []),
                frozenset((e.description or "").split("\n")),
            )
            for e in graph.read_entities(ids)
        }
    finally:
        graph.close()


def _converges(tmp_path: Path, runs: list[dict[str, str]]) -> None:
    full = _Store(tmp_path / "full", _Model())
    full.ingest(runs[-1])
    incremental = _Store(tmp_path / "incremental", _Model())
    for corpus in runs:
        incremental.ingest(corpus)
    assert _entities(incremental) == _entities(full)
    assert incremental.edges() == full.edges()


def test_an_edited_documents_old_text_leaves_the_shared_entity(tmp_path) -> None:
    _converges(
        tmp_path,
        [
            {"a.txt": "Acme supplies Buyer for 100.", "b.txt": "Acme supplies Depot."},
            {"a.txt": "Acme supplies Buyer for 120.", "b.txt": "Acme supplies Depot."},
        ],
    )


def test_a_deleted_documents_edge_leaves_the_ranks(tmp_path) -> None:
    _converges(
        tmp_path,
        [
            {
                "a.txt": "Acme supplies Buyer.",
                "b.txt": "Acme supplies Depot.",
                "c.txt": "Depot and Buyer trade goods.",
            },
            {"a.txt": "Acme supplies Buyer.", "b.txt": "Acme supplies Depot."},
        ],
    )
