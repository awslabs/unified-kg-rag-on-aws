# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``indexing.opensearch.vector_mode`` / ``compression_level`` on a real
OpenSearch (``LOCAL_STORES=1``).

On the ``docker/compose.local.yaml`` cluster (2.13, as the CDK domain) the
creation probe refuses the keys and names the 2.17 requirement. On 2.17 or
later (an OpenSearch published on ``LOCAL_STORES_OPENSEARCH_PORT``, e.g.
``docker run -p 127.0.0.1:19200:9200 -e discovery.type=single-node -e
DISABLE_SECURITY_PLUGIN=true opensearchproject/opensearch:2.19.3``) the
probe passes, and a synthetic set of ``VECTOR_STORAGE_VECTORS`` vectors
(default 50,000, about 6 minutes) measures each storage's recall@10 against
exact search and its k-NN native memory; ``-s`` prints them.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from opensearchpy import helpers

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.store_backends import local_store_config
from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient
from unified_kg_rag.adapters.retrievers.opensearch_retriever import (
    OpenSearchRetriever,
)
from unified_kg_rag.adapters.storage import OpenSearchIndexer
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import ConfigurationError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("LOCAL_STORES") != "1",
        reason="set LOCAL_STORES=1 with an OpenSearch running",
    ),
]

_DIMENSION = 256
_QUERIES = 200


@pytest.fixture(scope="module")
def client() -> Iterator[OpenSearchClient]:
    opensearch = OpenSearchClient(local_store_config())
    yield opensearch
    opensearch.close()


def _version(client: OpenSearchClient) -> tuple[int, ...]:
    number = client.client.info()["version"]["number"]
    return tuple(int(part) for part in number.split("-")[0].split("."))


def _config(engine: str = "lucene", **storage: Any) -> Config:
    config = local_store_config()
    config.indexing.additional_suffix = f"vs{uuid.uuid4().hex[:8]}"
    opensearch = config.indexing.opensearch
    opensearch.vector_search = {**opensearch.vector_search, "engine": engine}
    for key in ("compression_level", "vector_mode"):
        if key in storage:
            setattr(opensearch, key, storage[key])
    return config


def _indexer(config: Config, client: OpenSearchClient) -> OpenSearchIndexer:
    return OpenSearchIndexer(
        config,
        embedding_factory=HashingEmbeddingFactory(dimensions=_DIMENSION),
        opensearch_client=client,
    )


def test_the_creation_probe_follows_the_cluster_version(
    client: OpenSearchClient,
) -> None:
    indexer = _indexer(_config(vector_mode="on_disk", compression_level="4x"), client)
    if _version(client) < (2, 17):
        with pytest.raises(ConfigurationError, match="OpenSearch 2.17 or later"):
            indexer.check_index_creation("default")
    else:
        indexer.check_index_creation("default")
    assert not client.get_aliases_by_index(
        f"{indexer._get_name(indexer.opensearch_config.extraction_records_index_prefix, 'default')}-settings-probe-*"
    )
    # The default takes every version.
    _indexer(_config(), client).check_index_creation("default")


def _vectors(count: int) -> tuple[np.ndarray, np.ndarray]:
    """``count`` unit vectors around 200 centres, and ``_QUERIES`` queries."""
    rng = np.random.default_rng(7)
    centres = rng.normal(size=(200, _DIMENSION))

    def around(n: int) -> np.ndarray:
        points = centres[rng.integers(0, 200, n)] + 0.6 * rng.normal(
            size=(n, _DIMENSION)
        )
        return points / np.linalg.norm(points, axis=1, keepdims=True)

    return around(count), around(_QUERIES)


def _measure(
    client: OpenSearchClient,
    config: Config,
    data: np.ndarray,
    queries: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, int]:
    """The recall@10 of the indexer's knn_vector mapping, queried as the
    retriever queries, and the k-NN native memory (KB) it took."""
    indexer = _indexer(config, client)
    name = f"{config.indexing.additional_suffix}-vectors"
    body = indexer._get_base_mapping({"v": indexer._get_knn_vector_mapping()})
    body["settings"]["refresh_interval"] = "-1"
    client.create_index(name, body, exist_ok=False)
    try:
        helpers.bulk(
            client.client,
            (
                {"_index": name, "_id": str(i), "v": row.tolist()}
                for i, row in enumerate(data)
            ),
            chunk_size=1000,
            request_timeout=300,
        )
        client.client.indices.refresh(index=name)
        client.client.indices.forcemerge(
            index=name, max_num_segments=1, request_timeout=600
        )
        retriever = OpenSearchRetriever.__new__(OpenSearchRetriever)
        object.__setattr__(retriever, "_opensearch_config", config.indexing.opensearch)
        found = 0
        for query, expected in zip(queries, truth, strict=True):
            knn = retriever._build_vector_query(
                query.tolist(), retriever._knn_k(10), ["v"]
            )
            hits = client.search(
                index=name, body={"size": 10, "query": knn, "_source": False}
            )["hits"]["hits"]
            found += len({int(hit["_id"]) for hit in hits} & set(expected.tolist()))
        stats = client.client.transport.perform_request("GET", "/_plugins/_knn/stats")
        memory = sum(node["graph_memory_usage"] for node in stats["nodes"].values())
        return found / truth.size, memory
    finally:
        client.client.transport.perform_request(
            "POST", f"/_plugins/_knn/clear_cache/{name}"
        )
        client.delete_indices([name])


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_on_disk_storage_keeps_recall_and_cuts_memory(
    client: OpenSearchClient,
) -> None:
    version = _version(client)
    if version < (2, 17):
        pytest.skip("vector_mode / compression_level need OpenSearch 2.17+")
    if version < (2, 19):
        pytest.skip("faiss with cosinesimil needs OpenSearch 2.19")
    data, queries = _vectors(int(os.getenv("VECTOR_STORAGE_VECTORS", "50000")))
    truth = np.argsort(-(queries @ data.T), axis=1)[:, :10]
    results = {
        label: _measure(client, _config(engine, **storage), data, queries, truth)
        for label, engine, storage in [
            ("lucene in_memory", "lucene", {}),
            (
                "lucene on_disk 4x",
                "lucene",
                {"vector_mode": "on_disk", "compression_level": "4x"},
            ),
            ("faiss in_memory", "faiss", {}),
            ("faiss on_disk 32x", "faiss", {"vector_mode": "on_disk"}),
            (
                "faiss on_disk 8x",
                "faiss",
                {"vector_mode": "on_disk", "compression_level": "8x"},
            ),
        ]
    }
    for label, (recall, memory) in results.items():
        print(f"{label}: recall@10 {recall:.3f}, k-NN native memory {memory} KB")
    # Unquantized HNSW finds all but a few of the exact top 10 here.
    assert results["lucene in_memory"][0] >= 0.95
    assert results["faiss in_memory"][0] >= 0.95
    # Measured on 2.19.3 at 50,000 vectors: lucene 4x 0.93, faiss 8x 0.95,
    # faiss 32x 0.78 (binary quantization loses most on few dimensions).
    assert results["lucene on_disk 4x"][0] >= 0.85
    assert results["faiss on_disk 8x"][0] >= 0.85
    assert results["faiss on_disk 32x"][0] >= 0.6
    # lucene's vectors live in the JVM heap and page cache, not in the k-NN
    # native memory; faiss's quantized index takes a fraction of it.
    faiss_memory = results["faiss in_memory"][1]
    for label in ("faiss on_disk 32x", "faiss on_disk 8x"):
        assert 0 < results[label][1] * 3 < faiss_memory, label
