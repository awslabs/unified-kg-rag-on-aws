# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every search strategy reports a failed rerank (#83, AWS-free).

Regression: a strategy whose fusion call did not pass its query's
``rerank_trace`` reported ``skipped`` while the rerank actually failed, and
no test noticed. Each strategy here runs through ``GraphRAGChain`` on the
embedded stores (the ``test_embedded_e2e`` harness: fake Ollama server,
SQLite stores) with a cross-encoder that always raises, so a strategy that
drops the trace fails its case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import Config, SearchStrategy

pytestmark = pytest.mark.integration


class _FailingCrossEncoder(e2e.FakeCrossEncoder):
    def predict(self, pairs: list[tuple[str, str]], **_: Any) -> list[float]:
        self.calls += 1
        raise RuntimeError("rerank backend down")


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory):
    tmp: Path = tmp_path_factory.mktemp("rerank-status")
    config: Config = e2e._config(tmp / "stores", tmp / "viz")
    # Global search reranks only the text units it is asked to add.
    config.search.global_search.text_unit_slots = 5
    server = FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM)
    # Ingest with a working encoder, then query with one that always fails.
    providers = e2e._providers(config, server, e2e.FakeCrossEncoder("ok"))
    source = tmp / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    context = e2e._ingest(config, providers, source, tmp / "run-1")
    assert context.status.value == "completed", e2e._errors(context)

    encoder = _FailingCrossEncoder("synthetic-reranker")
    rag = GraphRAGChain(config, providers=e2e._providers(config, server, encoder))
    try:
        yield rag, encoder
    finally:
        rag.close()


# Every enum member, AUTO included (routed to ``local`` by the fake model).
@pytest.mark.parametrize("strategy", list(SearchStrategy), ids=lambda s: s.value)
def test_every_strategy_reports_a_failed_rerank(chain, strategy) -> None:
    rag, encoder = chain
    calls = encoder.calls
    out = e2e._ask(rag, strategy)
    assert out.answer == e2e.ANSWER  # the fused order still answers
    assert encoder.calls > calls, strategy  # the rerank was attempted
    rerank = out.metadata["rerank"]
    assert rerank["status"] == "failed", rerank
    assert rerank["reason"] == "rerank call failed: RuntimeError"
    assert all(stage["status"] == "failed" for stage in rerank["stages"]), rerank
