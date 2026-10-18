# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A delta's small components that touch the stored graph get a community.

An incremental run clusters only its delta. A component of fewer than
``min_community_size`` entities gets no community (#347), so a new entity
whose one relationship is to a stored entity, a pair in the delta graph,
got none, where a full build clusters it with its stored neighbour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
from unified_kg_rag.domain.models import PipelineStageStatus

pytestmark = pytest.mark.integration


def _communities(providers, name: str) -> list[str]:
    graph = providers.build_graph_indexer()
    try:
        assert isinstance(graph, SQLiteGraphIndexer)
        ids = [i for i, n in graph.read_entity_names() if n == name]
        return [c for e in graph.read_entities(ids) for c in e.community_ids or []]
    finally:
        graph.close()


def test_a_new_entity_linked_only_to_a_stored_one_gets_a_community(
    tmp_path: Path, no_aws
) -> None:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    server = FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(
        config, server, e2e.FakeCrossEncoder("synthetic-reranker")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert _communities(providers, "Vendor")

    # The delta graph is the pair Carrier - Vendor; Vendor is stored.
    (source / "freight.txt").write_text("Carrier moves widgets for Vendor.")
    context = e2e._ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert len(context.incremental_delta.new) == 1
    assert _communities(providers, "Carrier")


no_aws = e2e.no_aws
