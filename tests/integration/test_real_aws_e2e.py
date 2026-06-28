# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-AWS end-to-end smoke tests (opt-in, never run in CI).

Marked ``aws`` so they are excluded by the default ``-m "not aws"`` selector.
They exercise the live Bedrock + Neptune + OpenSearch (+ optional DynamoDB)
stack against a real config, and are skipped unless that config is provided via
``GRAPHRAG_TEST_CONFIG`` (path to a config.yaml with real endpoints).

Run explicitly, after provisioning resources and credentials::

    GRAPHRAG_TEST_CONFIG=./config.yaml uv run pytest -m aws -v

These are smoke tests (connectivity + a tiny ingest→search round trip), not a
substitute for the AWS-free integration suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.aws, pytest.mark.slow]

_CONFIG_ENV = "GRAPHRAG_TEST_CONFIG"


def _require_config():
    path = os.getenv(_CONFIG_ENV)
    if not path or not Path(path).is_file():
        pytest.skip(
            f"set {_CONFIG_ENV} to a config.yaml with real AWS endpoints to run "
            "the real-AWS E2E smoke tests"
        )
    from unified_kg_rag.shared import get_config

    return get_config(path)


@pytest.fixture(scope="module")
def live_config():
    return _require_config()


def test_bedrock_embedding_roundtrip(live_config) -> None:
    """A real embedding call returns a non-empty vector of the configured dim."""
    from unified_kg_rag.adapters.aws import BedrockEmbeddingModelFactory

    factory = BedrockEmbeddingModelFactory(
        config=live_config,
        region_name=live_config.aws.bedrock.region_name,
    )
    model = factory.get_model(
        model_id=live_config.indexing.opensearch.embedding_model_id
    )
    vector = model.embed_query("knowledge graph retrieval augmented generation")
    assert isinstance(vector, list) and len(vector) > 0


def test_opensearch_connectivity(live_config) -> None:
    """The OpenSearch client can reach the cluster and report stats."""
    from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer

    indexer = OpenSearchIndexer(config=live_config)
    assert indexer.initialize() is True


def test_neptune_connectivity(live_config) -> None:
    """A real Gremlin round-trip against the live Neptune endpoint.

    NeptuneIndexer.initialize() is a no-op (returns True without touching the
    cluster), so issue an actual traversal to verify connectivity.
    """
    from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer

    indexer = NeptuneIndexer(config=live_config)
    # A trivial count traversal exercises the wss:// connection + SigV4 auth.
    count = indexer.neptune_client.g.V().limit(1).count().next()
    assert isinstance(count, int)


def test_neptune_relationship_read_roundtrips(live_config) -> None:
    """Regression for cross-run merge: read_relationships must reconstruct
    source_id/target_id from edge topology (not edge properties), so the
    relationship half of cross_run_merge actually accumulates."""
    from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
    from unified_kg_rag.domain.models import Entity, Relationship

    indexer = NeptuneIndexer(config=live_config)
    a = Entity(id="_t_e_a", name="_T_A")
    b = Entity(id="_t_e_b", name="_T_B")
    rel = Relationship(id="_t_r_ab", source_id="_t_e_a", target_id="_t_e_b")
    indexer.upsert_entities([a, b])
    indexer.upsert_relationships([rel])
    try:
        read = indexer.read_relationships(["_t_r_ab"])
        assert len(read) == 1
        assert read[0].source_id == "_t_e_a"
        assert read[0].target_id == "_t_e_b"
    finally:
        indexer.delete_by_id(["_t_e_a", "_t_e_b", "_t_r_ab"])


@pytest.mark.skipif(
    not os.getenv("GRAPHRAG_TEST_RUN_INGEST"),
    reason="set GRAPHRAG_TEST_RUN_INGEST=1 to run the (slow, cost-incurring) "
    "ingest->search round trip",
)
def test_ingest_then_search_round_trip(live_config, tmp_path) -> None:
    """Tiny corpus ingest, then a query returns at least one result.

    Guarded behind an extra env flag because it creates indices/vertices and
    incurs Bedrock cost. Intended for manual pre-release verification.
    """
    import asyncio

    from unified_kg_rag.application.retrieval.rag_chain import create_rag_chain
    from unified_kg_rag.domain.models import RAGInput

    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "doc.txt").write_text(
        "Alice is a research scientist at Acme Corp. Acme builds knowledge graphs.",
        encoding="utf-8",
    )

    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline

    pipeline = DataIngestionPipeline(config=live_config, source_directory=corpus)
    pipeline.run()

    chain = create_rag_chain(live_config)
    result = asyncio.run(chain.ainvoke(RAGInput(query="Where does Alice work?")))
    assert result is not None
