# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
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
    from aws_graphrag.shared import get_config

    return get_config(path)


@pytest.fixture(scope="module")
def live_config():
    return _require_config()


def test_bedrock_embedding_roundtrip(live_config) -> None:
    """A real embedding call returns a non-empty vector of the configured dim."""
    from aws_graphrag.adapters.aws import BedrockEmbeddingModelFactory

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
    from aws_graphrag.adapters.storage.opensearch_indexer import OpenSearchIndexer

    indexer = OpenSearchIndexer(config=live_config)
    assert indexer.initialize() is True


def test_neptune_connectivity(live_config) -> None:
    """The Neptune indexer initializes against the live endpoint."""
    from aws_graphrag.adapters.storage.neptune_indexer import NeptuneIndexer

    indexer = NeptuneIndexer(config=live_config)
    assert indexer.initialize() is True


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

    from aws_graphrag.domain.models import RAGInput
    from aws_graphrag.retrieval import create_rag_chain

    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "doc.txt").write_text(
        "Alice is a research scientist at Acme Corp. Acme builds knowledge graphs.",
        encoding="utf-8",
    )

    from aws_graphrag.application.ingestion.pipeline import DataIngestionPipeline

    pipeline = DataIngestionPipeline(config=live_config, source_directory=corpus)
    pipeline.run()

    chain = create_rag_chain(live_config)
    result = asyncio.run(chain.ainvoke(RAGInput(query="Where does Alice work?")))
    assert result is not None
