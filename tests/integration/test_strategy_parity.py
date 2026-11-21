# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy-level parity of the embedded stores with the containers.

One synthetic corpus is ingested, by the same fake models, into the embedded
SQLite stores and into Gremlin Server and OpenSearch of
``docker/compose.local.yaml``; every search strategy then answers the same
question on each. Each strategy must cite the same set of stored records on
both backends. The corpus is small enough that no strategy's result cap
cuts it, so the set a strategy cites depends on what it reaches, not on
the ranking.

Only the set is compared, not the order: the lexical scores of the two
engines differ (OpenSearch applies ``fuzziness: AUTO`` to lexical matches and
its own analyzer chains, the embedded engine neither), so records the
strategies fetch alike may still be fused in another order (and, over a
corpus larger than a cap, cut at another record).
``tests/integration/test_store_parity.py`` pins the ranking itself on corpora
where the engines agree. Skipped unless ``LOCAL_STORES=1``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.rerank import FakeCrossEncoder
from tests.fixtures.store_backends import StoreHarness, open_harness
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - fixture
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import (
    Methodology,
    PipelineStageStatus,
    SearchStrategy,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.local_stores,
]

_CORPUS = {
    "contract.txt": "Vendor supplies widgets to Buyer.",
    "logistics.txt": "Depot stores widgets for Vendor.",
    "freight.txt": "Carrier moves gadgets for Depot.",
    "storage.txt": e2e.ANSWER,
}


def _cited_per_strategy(
    tmp_path: Path, source: Path, harness: StoreHarness, suffix: str
) -> dict[SearchStrategy, set[str]]:
    """Ingest ``source`` into the stores of ``harness`` under the namespace
    ``suffix`` and ask every strategy: the records each one cites. Record
    ids derive from the namespace and the source files' paths, so both
    backends get the same of each."""
    config = e2e._config(tmp_path / "stores", tmp_path / "viz-data")
    config.indexing.methodologies = [
        *config.indexing.methodologies,
        Methodology.RAPTOR,
    ]
    e2e._on_harness_stores(config, harness)
    config.indexing.additional_suffix = suffix
    providers = e2e._providers(
        config,
        FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM),
        FakeCrossEncoder("reranker"),
    )
    context = e2e._ingest(config, providers, source, tmp_path / "run")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    chain = GraphRAGChain(config, providers=providers)
    try:
        return {
            strategy: {
                str(cited["source"]) for cited in e2e._ask(chain, strategy).sources
            }
            for strategy in SearchStrategy
        }
    finally:
        chain.close()


@pytest.mark.timeout(600)
@pytest.mark.usefixtures("no_aws")
def test_every_strategy_cites_the_same_records_on_both_backends(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    for name, text in _CORPUS.items():
        (source / name).write_text(text, encoding="utf-8")
    # The containers' namespace (which their teardown clears) on both.
    shared = open_harness("container", tmp_path / "container")
    container = next(shared)
    embedded_harness = open_harness("sqlite", tmp_path / "sqlite")
    sqlite = next(embedded_harness)
    try:
        suffix = container.config.indexing.additional_suffix
        assert suffix
        embedded = _cited_per_strategy(tmp_path / "embedded", source, sqlite, suffix)
        containers = _cited_per_strategy(tmp_path / "shared", source, container, suffix)
    finally:
        next(embedded_harness, None)
        next(shared, None)
    for strategy in SearchStrategy:
        assert embedded[strategy], strategy
    differ = {
        strategy.value: {
            "sqlite only": sorted(embedded[strategy] - containers[strategy]),
            "container only": sorted(containers[strategy] - embedded[strategy]),
        }
        for strategy in SearchStrategy
        if embedded[strategy] != containers[strategy]
    }
    assert not differ, differ
