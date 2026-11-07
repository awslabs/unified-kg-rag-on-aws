# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An embedded-store ingestion a test can kill mid-stage (SIGKILL).

``python -m tests.fixtures.killable_ingest <stores> <source> <cache> <marker>``
runs one ingestion of ``source`` over the SQLite stores in ``stores``
(the embedded end-to-end suite's config, see :func:`crash_config`), its
models a fake Ollama in-process. Once the indexing stage has written
(its second embedding request), it creates ``marker`` and blocks, so the
test kills a process that holds the namespace lease with the indexing
half done. No AWS, no network.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import httpx

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.domain.models import Config

# The shortest lease the config takes: the killed run's expires this long
# after its last renewal.
LEASE_SECONDS = 10.0


def crash_config(stores: Path) -> Config:
    """The embedded suite's config on ``stores``, with the shortest lease and
    a short embedded write (what a takeover waits for)."""
    config = e2e._config(stores, stores / "viz")
    config.indexing.lease_seconds = LEASE_SECONDS
    config.search.store_timeout_seconds = 1.0
    return config


class _BlockingServer:
    """``FakeOllama``, blocking for good at the indexing stage's second
    embedding request once ``marker`` is created."""

    def __init__(self, marker: Path) -> None:
        self.server = FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM)
        self.marker = marker
        self.indexing = False
        self.embeds = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.indexing and request.url.path == "/api/embed":
            self.embeds += 1
            if self.embeds == 2:
                self.marker.write_text("indexing")
                time.sleep(3600)
        return self.server(request)


def main(argv: list[str]) -> int:
    stores, source, cache, marker = (Path(arg) for arg in argv)
    config = crash_config(stores)
    blocking = _BlockingServer(marker)
    execute = ps.IndexingStage.execute

    def indexing(stage: Any, context: Any) -> Any:
        blocking.indexing = True
        return execute(stage, context)

    ps.IndexingStage.execute = indexing  # type: ignore[method-assign]
    from unified_kg_rag.adapters.local.embeddings import OllamaEmbeddingFactory
    from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
    from unified_kg_rag.adapters.local.rerank import CrossEncoderRerankFactory
    from unified_kg_rag.application.bootstrap import build_providers

    transport = httpx.MockTransport(blocking)
    providers = build_providers(
        config,
        llm_factory=OllamaLanguageModelFactory(config, transport=transport),
        embedding_factory=OllamaEmbeddingFactory(config, transport=transport),
        rerank_factory=CrossEncoderRerankFactory(
            config, loader=lambda *_, **__: e2e.FakeCrossEncoder("r")
        ),
    )
    context = e2e._ingest(config, providers, source, cache)
    print(context.status.value)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
