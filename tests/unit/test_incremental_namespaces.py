# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry namespace the loading stage assigns when parsing did not run.

Without a parsing stage the loading stage reads pre-parsed ``Document`` JSON
and assigns each document's identity itself. The namespace it records must
follow ``indexing.additional_suffix`` and the store prefixes exactly as the
parsing stage does, or an incremental run keys its registry records (and plans
removals) in another namespace than the stores it writes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from unified_kg_rag.application.ingestion.pipeline_stages import (
    DocumentLoadingStage,
)
from unified_kg_rag.domain.ingestion.document_identity import (
    REGISTRY_NAMESPACE_KEY,
    store_namespace_tag,
)
from unified_kg_rag.domain.models import (
    Config,
    PipelineContext,
    PipelineStageStatus,
)

pytestmark = pytest.mark.unit


def _write_document(directory: Path) -> None:
    (directory / "contract.json").write_text(
        json.dumps(
            {
                "document_id": "pre-parsed",
                "file_name": "contract.txt",
                "file_path": "contract.txt",
                "page_content": "Vendor ships goods to Buyer.",
            }
        ),
        encoding="utf-8",
    )


def _load_namespace(config: Config, source: Path) -> str:
    _write_document(source)
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=str(source),
    )
    # No document_parsing result on the context: the stage loads the JSON.
    result = DocumentLoadingStage(config, source_directory=source).execute(context)
    assert result.status == PipelineStageStatus.COMPLETED, result.error_message
    (document,) = context.documents
    return document.metadata[REGISTRY_NAMESPACE_KEY]


def test_loading_stage_namespace_carries_additional_suffix(tmp_path: Path) -> None:
    config = Config()
    config.indexing.additional_suffix = "tenant1"

    assert _load_namespace(config, tmp_path) == "default-tenant1"


def test_loading_stage_namespace_carries_store_prefix_tag(tmp_path: Path) -> None:
    config = Config()
    config.indexing.opensearch.entities_index_prefix = "custom-entities"
    tag = store_namespace_tag(config.indexing)
    assert tag

    assert _load_namespace(config, tmp_path) == f"default@{tag}"


def test_a_changes_runs_symlink_fallback_reads_its_own_namespace_alone(
    tmp_path: Path,
) -> None:
    # rv45 ingestion low (mutant): a run of changes given a directory the
    # registry holds through a symlink takes that stored scope, but only one
    # of its own namespace: another namespace's records of the directory
    # are that tenant's (doc ids carry the namespace).
    from unified_kg_rag.application.ingestion.incremental import (
        changes_source_scopes,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    link = tmp_path / "link"
    link.symlink_to(corpus, target_is_directory=True)
    # Namespaces of one length: a prefix check alone tells them apart.
    stored = [f"tenb|{link}"]
    assert changes_source_scopes(str(corpus), "tena", stored) == []
    assert changes_source_scopes(str(corpus), "tenb", stored) == [str(link)]
