# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``indexing.summary_tree`` options end to end on the embedded stores
(docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in): which text
units each tree is built over (``scope``, ``partition_by``,
``min_leaves_per_tree``), when an incremental run rebuilds it (``refresh``),
the memory bound (``max_leaves_in_memory``) and a failed tree write.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_raptor_e2e as raptor
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - fixture
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import (
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.ports.indexer import IndexingStats

pytestmark = pytest.mark.integration

# Two documents of six text units and one of five: more than
# min_nodes_to_cluster (4, raptor.setup) each.
DOCUMENTS = {
    "alpha.txt": [f"Alpha depot {i} stores widgets for Vendor." for i in range(6)],
    "beta.txt": [f"Beta carrier {i} moves gadgets for Depot." for i in range(6)],
    "gamma.txt": [f"Gamma bank {i} lends money to Buyer." for i in range(5)],
}


def _documents(source: Path, documents: dict[str, list[str]] = DOCUMENTS) -> Path:
    source.mkdir(parents=True, exist_ok=True)
    for name, passages in documents.items():
        (source / name).write_text("\n\n".join(passages))
    return source


# The stages of a run over pre-parsed documents (Document JSON, which may
# carry filters): no parsing stage.
PARSED = set(PipelineStageType) - {PipelineStageType.DOCUMENT_PARSING}


def _parsed(directory: Path, filters: Callable[[str], dict[str, str]]) -> Path:
    """``DOCUMENTS`` as pre-parsed documents, each with ``filters(name)``."""
    directory.mkdir(parents=True)
    for name, passages in DOCUMENTS.items():
        (directory / f"{Path(name).stem}.json").write_text(
            json.dumps(
                {
                    "document_id": Path(name).stem,
                    "file_name": name,
                    "file_path": name,
                    "page_content": "\n\n".join(passages),
                    "metadata": {"filters": filters(name)},
                }
            )
        )
    return directory


def _unit_files(providers: Providers) -> dict[str, str]:
    """Each stored text unit's file name."""
    indexer = providers.build_vector_indexer()
    try:
        rows = indexer.scan_text_units(None, fields=["attributes"]) or []
    finally:
        indexer.close()
    return {str(r["id"]): (r.get("attributes") or {})["file_name"] for r in rows}


def _nodes(providers: Providers) -> list[dict[str, Any]]:
    indexer = providers.build_vector_indexer()
    try:
        return (
            indexer.read_summary_nodes(
                None, fields=["level", "text", "text_unit_ids", "partition"]
            )
            or []
        )
    finally:
        indexer.close()


def _tree_metrics(context: PipelineContext) -> dict[str, Any]:
    (result,) = [r for r in context.stage_results if r.stage_name == "summary_tree"]
    return result.metrics


def test_a_document_scope_builds_one_tree_per_large_enough_document(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    config, providers, _, _ = raptor.setup(tmp_path)
    tree = config.indexing.summary_tree
    tree.scope = "document"
    tree.min_leaves_per_tree = 6
    source = _documents(tmp_path / "corpus")

    context = raptor.ingest(config, providers, source, tmp_path / "run-1")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    files = _unit_files(providers)
    assert Counter(files.values()) == {"alpha.txt": 6, "beta.txt": 6, "gamma.txt": 5}
    nodes = _nodes(providers)
    trees: dict[str, set[str]] = {}
    for node in nodes:
        (name,) = {files[unit] for unit in node["text_unit_ids"]}
        assert node["partition"].endswith(name)
        trees.setdefault(name, set()).update(node["text_unit_ids"])
    # gamma has fewer than min_leaves_per_tree text units: no tree, though
    # it has enough to cluster.
    assert set(trees) == {"alpha.txt", "beta.txt"}


def test_partition_by_builds_one_tree_per_filter_value(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    config, providers, _, _ = raptor.setup(tmp_path)
    config.indexing.summary_tree.partition_by = ["region"]
    regions = {"alpha.txt": "north", "beta.txt": "south", "gamma.txt": "north"}
    parsed = _parsed(tmp_path / "parsed", lambda name: {"region": regions[name]})

    context = raptor.ingest(config, providers, parsed, tmp_path / "run-1", PARSED)

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    files = _unit_files(providers)
    nodes = _nodes(providers)
    assert {n["partition"] for n in nodes} == {'["north"]', '["south"]'}
    for node in nodes:
        (region,) = {regions[files[unit]] for unit in node["text_unit_ids"]}
        assert node["partition"] == json.dumps([region])


def test_refresh_on_reset_keeps_the_tree_of_an_incremental_run(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    config, providers, _, calls = raptor.setup(tmp_path)
    config.indexing.summary_tree.refresh = "on_reset"
    source = raptor.corpus(tmp_path)
    context = raptor.ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    built = raptor.shape(raptor.tree_nodes(providers))
    assert built

    (source / "lab.txt").write_text("Lab tests widgets for Buyer.\n\nLab has a roof.")
    before = Counter(calls.kinds)
    context = raptor.ingest(config, providers, source, tmp_path / "run-2")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert calls.since(before)["summary"] == 0
    assert raptor.shape(raptor.tree_nodes(providers)) == built
    # The new text units are reported, not summarized: what no level-1
    # summary covers.
    covered = {
        unit
        for node in raptor.tree_nodes(providers)
        if node["level"] == 1
        for unit in node["text_unit_ids"]
    }
    files = _unit_files(providers)
    uncovered = {unit for unit in files if unit not in covered}
    assert {u for u, name in files.items() if name == "lab.txt"} <= uncovered
    assert _tree_metrics(context)["tree_uncovered_leaves"] == len(uncovered)
    record = raptor.tree_record(providers)
    assert record is not None and record["pipeline_id"] == "run-1"

    # every_run (the default) rebuilds it on the next run.
    config.indexing.summary_tree.refresh = "every_run"
    context = raptor.ingest(config, providers, source, tmp_path / "run-3")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    record = raptor.tree_record(providers)
    assert record is not None and record["pipeline_id"] == "run-3"


def test_a_tree_over_max_leaves_in_memory_fails_and_names_the_ways_out(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    config, providers, _, calls = raptor.setup(tmp_path)
    config.indexing.summary_tree.max_leaves_in_memory = 10
    source = _documents(tmp_path / "corpus")

    context = raptor.ingest(config, providers, source, tmp_path / "run-1")

    assert context.status is PipelineStageStatus.FAILED
    (failed,) = [r for r in context.stage_results if r.stage_name == "summary_tree"]
    assert failed.status is PipelineStageStatus.FAILED
    message = failed.error_message or ""
    assert "would hold 17 text units" in message
    assert "partition_by" in message and "scope: document" in message
    assert calls.kinds["summary"] == 0
    assert raptor.tree_nodes(providers) == []

    # One tree per document stays within the bound.
    config.indexing.summary_tree.scope = "document"
    context = raptor.ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert raptor.tree_nodes(providers)


def test_a_failed_tree_write_fails_the_stage_and_keeps_the_old_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_aws,  # noqa: F811 - the fixture
) -> None:
    from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer

    config, providers, _, _ = raptor.setup(tmp_path)
    source = raptor.corpus(tmp_path)
    context = raptor.ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    built = raptor.shape(raptor.tree_nodes(providers))
    record = raptor.tree_record(providers)

    def failing(self: Any, nodes: list[Any], *args: Any, **kwargs: Any) -> Any:
        return IndexingStats(
            total_items=len(nodes), failed_items=1, errors=["node rejected"]
        )

    monkeypatch.setattr(OpenSearchIndexer, "write_summary_tree", failing)
    (source / "lab.txt").write_text("Lab tests widgets for Buyer.\n\nLab has a roof.")
    context = raptor.ingest(config, providers, source, tmp_path / "run-2")

    (result,) = [r for r in context.stage_results if r.stage_name == "summary_tree"]
    assert result.status is PipelineStageStatus.FAILED
    assert "node rejected" in (result.error_message or "")
    assert "the previous tree" in (result.error_message or "")
    assert raptor.shape(raptor.tree_nodes(providers)) == built
    assert raptor.tree_record(providers) == record
