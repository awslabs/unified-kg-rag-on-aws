# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The store preflight, before any model call (``backend: local``)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.shared.exceptions import (
    PipelineExecutionError,
    StoreConnectionError,
)

pytestmark = pytest.mark.integration

_FILES = {"d0.txt": "Vendor and Buyer trade goods."}


def test_an_unreachable_vector_store_fails_the_run_before_any_model_call(
    tmp_path: Path,
) -> None:
    # The preflight only logged it: the run extracted the whole corpus and
    # then failed in the indexing stage.
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    server = FakeOllama(e2e._respond, "x", e2e.DIM)
    providers = e2e._providers(config, server, e2e.FakeCrossEncoder("r"))
    source = tmp_path / "corpus"
    source.mkdir()
    for name, text in _FILES.items():
        (source / name).write_text(text)

    def unreachable(self, suffix=None, *, rebuild=True):
        raise StoreConnectionError("Failed to connect to OpenSearch.")

    with (
        patch.object(OpenSearchIndexer, "check_index_creation", unreachable),
        pytest.raises(PipelineExecutionError) as raised,
    ):
        e2e._ingest(config, providers, source, tmp_path / "run")
    assert isinstance(raised.value.__cause__, StoreConnectionError)
    assert server.chat_models == [] and server.embed_models == []


@pytest.mark.parametrize("delete", [False, True])
def test_a_run_without_documents_to_extract_is_refused_by_the_live_indices(
    tmp_path: Path, delete: bool
) -> None:
    # An incremental run with nothing to extract (no change, or only a
    # deletion) skipped the preflight: its indexing stage failed on the
    # live index instead of the run being refused as a usage error.
    from unified_kg_rag.shared import ConfigurationError

    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    server = FakeOllama(e2e._respond, "x", e2e.DIM)
    providers = e2e._providers(config, server, e2e.FakeCrossEncoder("r"))
    source = tmp_path / "corpus"
    source.mkdir()
    for name, text in {**_FILES, "d1.txt": "Depot and Carrier trade."}.items():
        (source / name).write_text(text)
    e2e._ingest(config, providers, source, tmp_path / "seed")
    if delete:
        (source / "d1.txt").unlink()
    config.models.embedding.model_id = "another-embed"
    providers = e2e._providers(config, server, e2e.FakeCrossEncoder("r"))
    with pytest.raises(ConfigurationError, match="holds embeddings of"):
        e2e._ingest(config, providers, source, tmp_path / "run")
