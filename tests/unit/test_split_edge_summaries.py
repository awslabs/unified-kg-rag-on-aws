# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stored edge summary next to an entity split off by an incremental run.

A delta that resolves the Korean legal form "(주)벤더" into "벤더" where the
stored graph keeps them apart splits "(주)벤더" off again (``split_from``).
The delta's Vendor -> 벤더 edge then holds the line of the Vendor -> (주)벤더
edge the split moved it to, while still citing every text unit its derived
edge cites: extending the stored Vendor -> 벤더 summary with it would state
that line twice. The edge is summarized from its derived lines instead, as a
full build does.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from tests.fixtures.fakes.summarizer import AtomSetSummarizer
from tests.property.test_incremental_rebuild_properties import (
    _KOREAN_LEGAL_FORM_B,
    _KOREAN_LEGAL_FORM_C,
    _PATHS,
    _SUMMARIZING,
    _expected_state,
    _graph_state,
    _run_changes,
    _summarized_extract,
)
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.application.storage.indexing_manager import IndexingManager

pytestmark = pytest.mark.unit


def test_an_edge_of_a_split_entity_is_summarized_from_its_derived_lines() -> None:
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=_SUMMARIZING.indexing.opensearch)
    manager = IndexingManager(
        config=_SUMMARIZING, vector_indexer=vector, graph_indexer=graph
    )
    manager._description_summarizer = AtomSetSummarizer(_SUMMARIZING)
    inc = IncrementalIndexer(FakeDocStatusStore(), manager)
    stored: dict[str, Any] = {}

    _run_changes(inc, stored, {_PATHS[2]: _KOREAN_LEGAL_FORM_C}, _summarized_extract)
    _run_changes(inc, stored, {_PATHS[1]: _KOREAN_LEGAL_FORM_B}, _summarized_extract)

    edges = {
        (r.source_name, r.target_name): r.description
        for r in graph.read_relationships(sorted(graph.ids("relationships")))
    }
    assert "(주)벤더" not in edges[("Vendor", "벤더")]
    assert "(주)벤더" in edges[("Vendor", "(주)벤더")]
    assert _graph_state(graph) == _expected_state(stored, _summarized_extract)
