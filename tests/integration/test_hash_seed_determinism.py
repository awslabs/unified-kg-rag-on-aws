# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ingestion and retrieval give the same output under any ``PYTHONHASHSEED``.

``str`` hashes are salted per process, so a ``set`` (or ``frozenset``) of ids
or names iterates in a different order in every interpreter. Wherever that
order reaches an id, a merge winner, a tie-break, a batch, a prompt or a
written file, two runs of identical code on an identical corpus differ. The
test runs a full build, an incremental run (a document added, one changed,
one deleted) and every search strategy in two subprocesses with different
hash seeds, on the embedded stores with a fake model whose answers depend
only on the prompt, and compares everything they produced: the stored
graph and vector documents, the exported visualization data, every prompt
the model received and every strategy's answer and ranked results.

Run as a module (``python -m tests.integration.test_hash_seed_determinism
WORK OUT``), it is that subprocess.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tests.integration.test_hash_seed_determinism"
NAMES = (
    "Vendor",
    "Buyer",
    "Depot",
    "Carrier",
    "Broker",
    "Insurer",
    "Auditor",
    "Lender",
    "Courier",
    "Warehouse",
)
CORPUS = {
    "contract.txt": "Vendor supplies widgets to Buyer. Broker arranged the deal.",
    "logistics.txt": "Depot stores widgets for Vendor. Carrier moves them.",
    "freight.txt": "Carrier moves gadgets for Depot and Courier.",
    "finance.txt": "Lender funds Buyer. Insurer covers Lender and Broker.",
    "audit.txt": "Auditor reviews Vendor, Buyer and Insurer every year.",
    "storage.txt": "Warehouse rents space to Depot and Courier.",
}
# Wall-clock fields, and the build stamp in physical index names.
VOLATILE = re.compile(r"(_at|time|timestamp|build_tag|_search_index)$", re.IGNORECASE)
INDEX_STAMP = re.compile(r"-\d{14,}$")


def _names(text: str) -> list[str]:
    return [name for name in NAMES if name in text]


def _refinement(text: str) -> str:
    # One gleaned edge per unit naming three or more parties: the first and
    # the last, which extraction (consecutive names only) did not link.
    names = _names(text)
    if len(names) < 3:
        return (
            "<refinement_plan><identified_issues></identified_issues></refinement_plan>"
        )
    a, b = names[0], names[-1]
    return (
        "<refinement_plan><identified_issues><issue>"
        "<issue_type>MISSING_RELATIONSHIP</issue_type><details>"
        f"<source>{a}</source><target>{b}</target><type>DEALS_WITH</type>"
        f"<description>{a} deals with {b}</description></details>"
        f"<text_evidence>{text.strip()}</text_evidence>"
        "</issue></identified_issues></refinement_plan>"
    )


def _claims(text: str) -> str:
    names = _names(text)
    return (
        "<claims>"
        + "".join(
            f"<claim><subject>{a}</subject><object>{b}</object>"
            "<claim_type>DEALING</claim_type><claim_status>TRUE</claim_status>"
            "<start_date>UNKNOWN</start_date><end_date>UNKNOWN</end_date>"
            f"<description>{a} deals with {b}.</description>"
            f"<source_text>{text.strip()}</source_text></claim>"
            for a, b in zip(names, names[1:], strict=False)
        )
        + "</claims>"
    )


def _scrub(value: Any) -> Any:
    """``value`` without its wall-clock fields."""
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if not VOLATILE.search(k)}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _stores(stores: Path) -> dict[str, Any]:
    """Every stored vertex, edge and vector-store document."""
    graph = sqlite3.connect(stores / "graph.sqlite3")
    vector = sqlite3.connect(stores / "vector.sqlite3")
    try:
        return {
            "vertices": [
                [label, id_, _scrub(json.loads(props))]
                for label, id_, props in graph.execute(
                    "SELECT label, id, props FROM vertices ORDER BY label, id"
                )
            ],
            "edges": [
                [label, id_, source, target, _scrub(json.loads(props))]
                for label, id_, source, target, props in graph.execute(
                    "SELECT label, id, source_id, target_id, props FROM edges "
                    "ORDER BY label, id"
                )
            ],
            "documents": [
                [INDEX_STAMP.sub("", index), id_, _scrub(json.loads(source))]
                for index, id_, source in vector.execute(
                    "SELECT index_name, id, source FROM docs ORDER BY id, index_name"
                )
            ],
        }
    finally:
        graph.close()
        vector.close()


def run(work: Path) -> dict[str, Any]:
    """Build, update and query the corpus under ``work``; what it produced."""
    from tests.fixtures.fakes.model_servers import FakeOllama
    from tests.integration import test_embedded_e2e as e2e
    from tests.integration import test_ingestion_stages as stages
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
    from unified_kg_rag.domain.models import PipelineStageStatus, SearchStrategy

    prompts: list[list[str]] = []

    def respond(system: str, human: str) -> str:
        prompts.append([system, human])
        if "knowledge graph extraction expert" in system:
            return stages._extraction(stages._source_text(human), NAMES)
        if "graph refinement specialist" in system:
            return _refinement(stages._source_text(human))
        if "claim extraction specialist" in system:
            return _claims(stages._source_text(human))
        return e2e._respond(system, human)

    stores, viz, source = work / "stores", work / "viz", work / "corpus"
    source.mkdir(parents=True)
    for name, text in CORPUS.items():
        (source / name).write_text(text, encoding="utf-8")
    config = e2e._config(stores, viz)
    processing = config.processing
    processing.gleaning.enabled = True
    processing.gleaning.max_rounds = 1
    processing.claim_extraction.enabled = True
    processing.fuzzy_entity_resolution = True
    summarization = processing.graph_extraction.description_summarization
    summarization.force_summary_threshold_tokens = 10
    community_detection = config.graph.community_detection
    community_detection.min_community_size = 2
    community_detection.max_cluster_size = 4
    server = FakeOllama(respond, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(
        config, server, e2e.FakeCrossEncoder("synthetic-reranker")
    )

    out: dict[str, Any] = {}
    context = e2e._ingest(config, providers, source, work / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    out["full"] = _stores(stores)
    exported = json.loads((viz / "visualization_data.json").read_text("utf-8"))
    out["visualization"] = _scrub(exported)

    (source / "freight.txt").write_text(
        "Carrier moves gadgets for Depot, Courier and Lender.", encoding="utf-8"
    )
    (source / "audit.txt").unlink()
    (source / "claims.txt").write_text(
        "Insurer pays Buyer. Auditor checks Insurer and Warehouse.", encoding="utf-8"
    )
    context = e2e._ingest(config, providers, source, work / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    out["incremental"] = _stores(stores)
    out["ingestion_prompts"] = sorted(prompts)
    prompts.clear()

    chain = GraphRAGChain(config, providers=providers)
    try:
        answers = {}
        for strategy in SearchStrategy:
            result = chain.invoke(
                RAGInput(
                    query="Who stores the gadgets for Vendor and Carrier?",
                    search_strategy=strategy,
                )
            )
            answers[strategy.value] = {
                "answer": result.answer,
                "results": [
                    [r.source, r.retriever_type, r.content, round(r.score or 0.0, 9)]
                    for r in result.search_results.results
                ],
                "sources": _scrub(result.sources),
            }
    finally:
        chain.close()
    out["answers"] = answers
    out["query_prompts"] = sorted(prompts)
    return out


@pytest.mark.timeout(600)
def test_outputs_do_not_depend_on_the_hash_seed(tmp_path: Path) -> None:
    # One work directory for both runs: document ids derive from file paths.
    work = tmp_path / "work"
    outputs = []
    for seed in ("1", "2"):
        shutil.rmtree(work, ignore_errors=True)
        out = tmp_path / f"seed-{seed}.json"
        subprocess.run(
            [sys.executable, "-m", MODULE, str(work), str(out)],
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
            timeout=500,
        )
        outputs.append(json.loads(out.read_text("utf-8")))
    first, second = outputs
    assert first["full"]["edges"], "the build stored no relationships"
    assert any(
        len(entry["nodes"]) > 2
        for entry in first["visualization"]["communities"]["hierarchy"]
    ), "no community large enough to show a member order"
    for key in first:
        assert first[key] == second[key], key


if __name__ == "__main__":
    import logging

    logging.disable(logging.WARNING)
    result = run(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result, sort_keys=True), "utf-8")
