# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deduplication changes nothing when the corpus holds no near-copies.

``processing.deduplicate`` drops near-copies and prefers the indexed one;
with no near-copy in any run it drops nothing, so a sequence of corpus runs
and runs of document changes (``insert_documents``) must leave the same
registry with it on as off: the same scopes, files, statuses and content
hashes, whatever spelling of the corpus directory each run names (resolved,
with a trailing slash or a ``.`` part, through a symlink). The documents
are English, Korean, Japanese and Chinese.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from tests.fixtures.doc_status_records import all_records
from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as embedded
from tests.integration.test_embedded_e2e import (
    ANSWER,
    DIM,
    FakeCrossEncoder,
    _config,
    _errors,
    _providers,
    _respond,
)
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import (
    Config,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)

pytestmark = [pytest.mark.property, pytest.mark.usefixtures("no_aws")]

# The fixture that patches every AWS client to fail.
no_aws = embedded.no_aws

_TEXTS = [
    "Vendor supplies widgets to Buyer under the yearly framework agreement.",
    "Depot stores widgets for Carrier in the harbor warehouse every week.",
    "케스트럴 창고는 매주 항구에서 운송사에게 부품을 보관해 준다.",
    "ケストレル社は毎月、港の倉庫から部品を出荷している。",
    "Kestrel 公司每周在港口仓库为承运人储存零件。",
]
_SPELLINGS = ["default", "resolved", "trailing", "dot", "link", "link_trailing"]

# A step: a corpus run of these files, or insert_documents of them, under
# a spelling of the corpus directory.
Step = tuple[str, str, list[int]]


def _ok(context: PipelineContext) -> None:
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)


def _pipeline(root: Path, config: Config) -> tuple[DataIngestionPipeline, object]:
    providers = _providers(
        config, FakeOllama(_respond, ANSWER, DIM), FakeCrossEncoder("rerank")
    )
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=root / "cache",
        ),
        providers=providers,
    )
    for stage in pipeline.stages:
        if isinstance(stage, ps.GraphResolutionStage):
            stage.resolver.entity_resolver.use_process_pool = False
            stage.resolver.relationship_resolver.use_process_pool = False
    return pipeline, providers


def _play(steps: list[Step], deduplicate: bool) -> list[tuple[str, ...]]:
    """The registry after ``steps``, its scopes relative to the run's root."""
    root = Path(tempfile.mkdtemp(prefix="ukr-dedup-pair-"))
    try:
        corpus = root / "corpus"
        corpus.mkdir()
        link = root / "link"
        link.symlink_to(corpus, target_is_directory=True)
        resolved = corpus.resolve().as_posix()
        spelled = {
            "default": None,
            "resolved": resolved,
            "trailing": f"{resolved}/",
            "dot": f"{resolved}/.",
            "link": str(link),
            "link_trailing": f"{link}/",
        }
        base = _config(root / "stores", root / "viz").updated(
            {"processing.deduplicate": deduplicate}
        )
        providers = None
        for kind, spelling, files in steps:
            scope = spelled[spelling]
            if kind == "corpus":
                for path in corpus.iterdir():
                    path.unlink()
                for number in files:
                    (corpus / f"f{number}.txt").write_text(_TEXTS[number])
                config = (
                    base
                    if scope is None
                    else base.updated(
                        {"processing.document_parsing.source_scope": scope}
                    )
                )
                pipeline, providers = _pipeline(root, config)
                try:
                    _ok(pipeline.run(corpus))
                finally:
                    pipeline.close()
            else:
                pipeline, providers = _pipeline(root, base)
                try:
                    _ok(
                        pipeline.insert_documents(
                            [
                                {"id": f"m{number}.txt", "text": _TEXTS[number]}
                                for number in files
                            ],
                            source_scope=scope or resolved,
                        )
                    )
                finally:
                    pipeline.close()
        assert providers is not None
        prefix = resolved.removesuffix("/corpus")
        return sorted(
            (
                record.scope.partition("|")[2].replace(prefix, "<root>"),
                str(record.file_path),
                record.status.value,
                record.content_hash,
            )
            for record in all_records(providers.build_doc_status())  # type: ignore[attr-defined]
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


_step = st.tuples(
    st.sampled_from(["corpus", "corpus", "insert"]),
    st.sampled_from(_SPELLINGS),
    st.lists(st.integers(0, len(_TEXTS) - 1), min_size=1, max_size=3, unique=True),
)


@pytest.mark.slow
@pytest.mark.timeout(600)
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=list(HealthCheck),
    database=None,
)
@given(steps=st.lists(_step, min_size=1, max_size=3))
# rv49: with deduplication on, a corpus run read as a run of changes and
# took over the symlinked scope the registry held, or replaced its own.
@example(steps=[("corpus", "default", [0]), ("corpus", "link", [0])])
@example(
    steps=[
        ("corpus", "link", [0, 1]),
        ("insert", "link", [2]),
        ("corpus", "default", [0, 1]),
    ]
)
def test_deduplication_without_copies_leaves_the_registry_as_without_it(
    steps: list[Step],
) -> None:
    assert _play(steps, True) == _play(steps, False)
