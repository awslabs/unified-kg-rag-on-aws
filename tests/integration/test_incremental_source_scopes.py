# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two source scopes on one index suffix keep their own documents (AWS-free).

Two corpora written to the same index namespace (different source
directories, or different ``document_parsing.source_scope`` values) may both
contain a file at the same relative path. The registry key used to be the
namespace and relative path only, so each corpus's run read the other's
record as its own changed document, pruned its artifacts and re-registered it
under its own scope: the corpora deleted each other's content on every run.

The whole pipeline runs here with a scripted chat model, the in-memory graph
and vector stores, and the in-memory or moto-backed DynamoDB registry.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.aws import DynamoDBDocStatusStore
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.ingestion.delta_detector import compute_doc_id
from unified_kg_rag.domain.models import (
    Config,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.domain.models.config import ChunkingStrategy
from unified_kg_rag.ports import DocStatusPort

pytestmark = pytest.mark.integration

_NAMES = ("Vendor", "Buyer", "Bank", "Depot")


class _Model:
    """Scripted chat model that counts graph-extraction calls per text."""

    def __init__(self) -> None:
        self.extractions: Counter[str] = Counter()

    def respond(self, system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system:
            match = re.search(r"<input_text>\n(.*?)\n</input_text>", human, re.DOTALL)
            assert match, "prompt has no SOURCE TEXT section"
            return self._extraction(match.group(1))
        if "graph refinement specialist" in system:
            return (
                "<refinement_plan><identified_issues></identified_issues>"
                "</refinement_plan>"
            )
        if "community analysis" in system:
            return (
                "<community_name>Trading partners</community_name>"
                "<summary>Partners trade widgets.</summary>"
                "<rating>5.0</rating><rating_explanation>Small.</rating_explanation>"
                "<findings><finding><summary>Central</summary>"
                "<explanation>They trade.</explanation></finding></findings>"
            )
        raise AssertionError(f"unexpected prompt: {system[:80]!r}")

    def _extraction(self, text: str) -> str:
        self.extractions[text.strip()] += 1
        names = [name for name in _NAMES if name in text]
        entities = "".join(
            f"<entity><name>{n}</name><type>ORGANIZATION</type>"
            f"<description>{n} trades widgets</description>"
            f"<confidence>9</confidence><source_text>{n}</source_text></entity>"
            for n in names
        )
        relationships = "".join(
            f"<relationship><source>{a}</source><target>{b}</target>"
            f"<type>WORKS_WITH</type><description>{a} works with {b}</description>"
            f"<strength>8</strength><source_text>{text.strip()}</source_text>"
            "</relationship>"
            for a, b in zip(names, names[1:], strict=False)
        )
        return (
            f"<entities>{entities}</entities>"
            f"<relationships>{relationships}</relationships>"
        )


def scope_test_config(base: Config | None = None) -> Config:
    """``base`` (default ``Config()``) set up for the scripted model."""
    config = base.model_copy(deep=True) if base is not None else Config()
    chunking = config.processing.chunking
    chunking.chunker_type = ChunkingStrategy.SIMPLE
    chunking.fallback_chunk_size = 400
    chunking.chunk_overlap = 0
    chunking.min_chunk_size = 1
    config.processing.translation.enabled = False
    config.processing.gleaning.enabled = False
    config.processing.claim_extraction.enabled = False
    config.processing.max_attempts = 1
    config.fixing.enabled = False
    config.graph.visualization.enabled = False
    return config


class ScopeStack:
    """One shared index suffix: stores, registry and a counting model.

    Defaults to the in-memory stores; ``graph``/``vectors`` inject real
    indexers (the local-stores test), which stay open across runs.
    """

    def __init__(
        self,
        registry: DocStatusPort,
        tmp_path: Path,
        config: Config | None = None,
        graph: Any = None,
        vectors: Any = None,
    ) -> None:
        self.config = config or scope_test_config()
        self.registry = registry
        self.model = _Model()
        self.graph = graph or FakeGraphStore()
        self.vectors = vectors or FakeVectorStore(self.config.indexing.opensearch)
        self.keep_open = graph is not None or vectors is not None
        self.tmp_path = tmp_path
        self.providers = Providers(
            self.config,
            boto_session=MagicMock(),
            llm_factory=ScriptedLLMFactory(self.model.respond),
            embedding_factory=HashingEmbeddingFactory(),
        )

    def run(self, source: Path) -> PipelineContext:
        self.model.extractions.clear()
        pipeline = DataIngestionPipeline(
            self.config,
            PipelineConfig(
                stages_enabled=dict.fromkeys(PipelineStageType, True),
                # A fresh stage cache per run: every run reads its corpus.
                local_directory=self.tmp_path / f"cache-{uuid.uuid4().hex[:8]}",
            ),
            source_directory=source,
            providers=self.providers,
            doc_status=self.registry,
            vector_indexer=self.vectors,
            graph_indexer=self.graph,
        )
        for stage in pipeline.stages:
            if isinstance(stage, ps.GraphResolutionStage):
                stage.resolver.entity_resolver.use_process_pool = False
                stage.resolver.relationship_resolver.use_process_pool = False
            if self.keep_open and isinstance(stage, ps.IndexingStage):
                stage.close = lambda: None  # type: ignore[method-assign]
        context = pipeline.run(source, pipeline_id=f"p{uuid.uuid4().hex[:8]}")
        assert context.status is PipelineStageStatus.COMPLETED
        return context

    def entity_names(self) -> set[str]:
        names = {e.name for e in self.vectors.data.get("entities", {}).values()}
        # The graph holds the same entities as the vector store.
        assert len(self.graph.ids("entities")) == len(names)
        return names

    def texts(self) -> set[str]:
        return {
            u.text.strip() for u in self.vectors.data.get("text_units", {}).values()
        }


@pytest.fixture(params=["fake", "dynamodb"])
def registry(request) -> Iterator[DocStatusPort]:
    if request.param == "fake":
        yield FakeDocStatusStore()
        return
    with mock_aws():
        config = Config()
        config.aws.dynamodb.table_name = "test-doc-status"
        store = DynamoDBDocStatusStore(
            config, boto_session=boto3.Session(region_name="us-east-1")
        )
        _ = store.client
        yield store


def write_corpus(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for path in root.iterdir():
        path.unlink()
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


A_TEXT = "Vendor supplies Buyer."
B_TEXT = "Buyer banks with Bank."


def _scope_of(context: PipelineContext) -> str:
    assert context.incremental_scope is not None
    return context.incremental_scope


def test_two_scopes_with_the_same_relative_path_coexist(registry, tmp_path) -> None:
    stack = ScopeStack(registry, tmp_path)
    source_a = write_corpus(tmp_path / "src-a", {"contract.txt": A_TEXT})
    source_b = write_corpus(tmp_path / "src-b", {"contract.txt": B_TEXT})

    first_a = stack.run(source_a)
    first_b = stack.run(source_b)
    assert first_b.incremental_delta.new and not first_b.incremental_delta.changed
    assert first_b.incremental_delta.deleted == []
    assert stack.texts() == {A_TEXT, B_TEXT}

    for _ in range(2):
        for source, first in ((source_a, first_a), (source_b, first_b)):
            context = stack.run(source)
            delta = context.incremental_delta
            # Unchanged and nothing extracted or deleted: the other scope's
            # contract.txt is a different document.
            assert delta.unchanged == first.incremental_delta.new
            assert delta.new == delta.changed == delta.deleted == []
            assert not stack.model.extractions
            assert stack.texts() == {A_TEXT, B_TEXT}

    records = registry.list_all()
    assert sorted(r.scope for r in records) == sorted(
        [_scope_of(first_a), _scope_of(first_b)]
    )
    assert {r.file_path for r in records} == {"contract.txt"}


def test_deleting_a_file_in_one_scope_removes_only_its_exclusive_artifacts(
    registry, tmp_path
) -> None:
    stack = ScopeStack(registry, tmp_path)
    source_a = write_corpus(
        tmp_path / "src-a",
        {"contract.txt": A_TEXT, "depot.txt": "Depot stores widgets."},
    )
    source_b = write_corpus(tmp_path / "src-b", {"contract.txt": B_TEXT})
    stack.run(source_a)
    stack.run(source_b)
    assert stack.entity_names() == {"Vendor", "Buyer", "Bank", "Depot"}

    (source_a / "contract.txt").unlink()
    context = stack.run(source_a)

    assert len(context.incremental_delta.deleted) == 1
    assert not stack.model.extractions
    # Vendor only came from A's contract.txt; Buyer is shared with B's.
    assert stack.entity_names() == {"Buyer", "Bank", "Depot"}
    assert stack.texts() == {B_TEXT, "Depot stores widgets."}
    assert {(r.file_path, r.scope) for r in registry.list_all()} == {
        ("depot.txt", _scope_of(context)),
        ("contract.txt", _scope_of(stack.run(source_b))),
    }


def _rekey_to_legacy(registry: DocStatusPort) -> dict[str, str]:
    """Rewrite every record under the key used before the source scope was
    part of it; returns ``{legacy_id: current_id}``."""
    moved = {}
    for record in registry.list_all():
        legacy_id = compute_doc_id(record.file_path, "default")
        registry.delete(record.doc_id)
        registry.put(record.model_copy(update={"doc_id": legacy_id}))
        moved[legacy_id] = record.doc_id
    return moved


@pytest.mark.parametrize("legacy_scope", ["same", "none"])
def test_a_legacy_keyed_registry_upgrades_without_reextracting(
    registry, tmp_path, legacy_scope
) -> None:
    stack = ScopeStack(registry, tmp_path)
    source = write_corpus(
        tmp_path / "src-a",
        {"contract.txt": A_TEXT, "depot.txt": "Depot stores widgets."},
    )
    stack.run(source)
    before = {r.file_path: r for r in registry.list_all()}
    moved = _rekey_to_legacy(registry)
    if legacy_scope == "none":
        # Records written before scopes existed carry no scope.
        for legacy_id in moved:
            record = registry.get(legacy_id)
            registry.put(record.model_copy(update={"scope": None}))
    entities, texts = stack.entity_names(), stack.texts()

    context = stack.run(source)

    delta = context.incremental_delta
    assert sorted(delta.unchanged) == sorted(moved.values())
    assert delta.new == delta.changed == delta.deleted == []
    assert not stack.model.extractions
    assert (stack.entity_names(), stack.texts()) == (entities, texts)
    # Adopted under the current key with the same fingerprint and lineage.
    after = {r.file_path: r for r in registry.list_all()}
    assert {r.doc_id for r in after.values()} == set(moved.values())
    for path, record in after.items():
        assert record.model_dump(exclude={"scope"}) == before[path].model_dump(
            exclude={"scope"}
        )
        assert record.scope == _scope_of(context)


def test_a_legacy_record_of_another_scope_is_left_alone(registry, tmp_path) -> None:
    stack = ScopeStack(registry, tmp_path)
    source_a = write_corpus(tmp_path / "src-a", {"contract.txt": A_TEXT})
    source_b = write_corpus(tmp_path / "src-b", {"contract.txt": B_TEXT})
    stack.run(source_b)
    (legacy_id,) = _rekey_to_legacy(registry)
    legacy_b = registry.get(legacy_id)

    context = stack.run(source_a)

    # A's contract.txt is new to A; B's legacy record is not adopted, not
    # deleted and its artifacts are kept.
    assert len(context.incremental_delta.new) == 1
    assert context.incremental_delta.deleted == []
    assert registry.get(legacy_id) == legacy_b
    assert stack.texts() == {A_TEXT, B_TEXT}

    # B's own next run adopts it without re-extracting.
    rerun_b = stack.run(source_b)
    assert rerun_b.incremental_delta.new == rerun_b.incremental_delta.deleted == []
    assert not stack.model.extractions
    assert registry.get(legacy_id) is None
    assert len(registry.list_all()) == 2


def test_an_interrupted_scopeless_adoption_does_not_keep_old_content(
    registry, tmp_path
) -> None:
    stack = ScopeStack(registry, tmp_path)
    source = write_corpus(tmp_path / "src-a", {"contract.txt": A_TEXT})
    stack.run(source)
    (current,) = registry.list_all()
    # A record written before scopes existed, adopted under the current key
    # by a run that stopped before deleting the legacy key.
    registry.put(
        current.model_copy(
            update={"doc_id": compute_doc_id("contract.txt", "default"), "scope": None}
        )
    )

    write_corpus(source, {"contract.txt": "Vendor supplies Depot."})
    context = stack.run(source)

    assert context.incremental_delta.changed == [current.doc_id]
    assert stack.texts() == {"Vendor supplies Depot."}
    assert stack.entity_names() == {"Vendor", "Depot"}
    assert [r.doc_id for r in registry.list_all()] == [current.doc_id]


DEPOT_TEXT = "Depot stores widgets."


def test_a_moved_local_corpus_keeps_its_records(registry, tmp_path) -> None:
    stack = ScopeStack(registry, tmp_path)
    old = write_corpus(
        tmp_path / "old", {"contract.txt": A_TEXT, "depot.txt": DEPOT_TEXT}
    )
    stack.run(old)
    before = {r.file_path: r for r in registry.list_all()}
    new = old.rename(tmp_path / "new")

    moved = stack.run(new)

    # Adopted, not re-extracted: one record per file, under the new scope.
    delta = moved.incremental_delta
    assert len(delta.unchanged) == 2
    assert delta.new == delta.changed == delta.deleted == []
    assert not stack.model.extractions
    after = {r.file_path: r for r in registry.list_all()}
    assert set(after) == {"contract.txt", "depot.txt"}
    for path, record in after.items():
        assert record.scope == _scope_of(moved)
        assert record.model_dump(exclude={"doc_id", "scope"}) == before[
            path
        ].model_dump(exclude={"doc_id", "scope"})

    # An edit and a deletion then prune the old content.
    write_corpus(new, {"contract.txt": "Vendor supplies Bank."})
    edited = stack.run(new)
    assert len(edited.incremental_delta.changed) == 1
    assert len(edited.incremental_delta.deleted) == 1
    assert stack.texts() == {"Vendor supplies Bank."}
    assert stack.entity_names() == {"Vendor", "Bank"}
    assert [r.file_path for r in registry.list_all()] == ["contract.txt"]


def test_a_copied_local_corpus_is_a_separate_corpus(registry, tmp_path, caplog) -> None:
    stack = ScopeStack(registry, tmp_path)
    first = write_corpus(tmp_path / "first", {"contract.txt": A_TEXT})
    second = write_corpus(tmp_path / "second", {"contract.txt": B_TEXT})
    stack.run(first)
    (first_record,) = registry.list_all()

    with caplog.at_level("INFO"):
        context = stack.run(second)

    # The first source still exists: nothing is adopted from it.
    assert len(context.incremental_delta.new) == 1
    assert registry.get(first_record.doc_id) == first_record
    assert stack.texts() == {A_TEXT, B_TEXT}
    shared = [r for r in caplog.records if "Other local corpora" in r.getMessage()]
    assert len(shared) == 1 and first.as_posix() in shared[0].getMessage()


def test_a_fixed_source_scope_never_adopts_a_vanished_directory(
    registry, tmp_path
) -> None:
    stack = ScopeStack(registry, tmp_path)
    old = write_corpus(tmp_path / "old", {"contract.txt": A_TEXT})
    stack.run(old)
    (old_record,) = registry.list_all()
    new = old.rename(tmp_path / "new")
    stack.config.processing.document_parsing.source_scope = "corpus-a"

    context = stack.run(new)

    assert len(context.incremental_delta.new) == 1
    assert registry.get(old_record.doc_id) == old_record


def test_a_delta_of_one_entity_without_relationships_completes(
    registry, tmp_path
) -> None:
    # The delta graph holds one entity and no edge, so community detection
    # runs on an edgeless graph (modularity divided by zero).
    stack = ScopeStack(registry, tmp_path)
    source = write_corpus(tmp_path / "src-a", {"contract.txt": A_TEXT})
    stack.run(source)
    (source / "depot.txt").write_text(DEPOT_TEXT, encoding="utf-8")

    context = stack.run(source)

    assert len(context.incremental_delta.new) == 1
    assert stack.entity_names() == {"Vendor", "Buyer", "Depot"}
    assert stack.texts() == {A_TEXT, DEPOT_TEXT}
