# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two source scopes on one index suffix keep their own documents (AWS-free).

Two corpora written to the same index namespace (different source
directories, or different ``document_parsing.source_scope`` values) may both
contain a file at the same relative path. The registry key used to be the
namespace and relative path only, so each corpus's run read the other's
record as its own changed document, pruned its artifacts and re-registered it
under its own scope: the corpora deleted each other's content on every run.

A corpus that moves to another directory changes its default scope; its old
records are only removed when a run retires the old scope explicitly, never
because a directory is missing on the host that runs.

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
from unified_kg_rag.shared import PipelineExecutionError

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


# --- moving a corpus: explicit retire, never a filesystem guess --------------

C_TEXT = "Buyer banks with Bank."


def _store_snapshot(stack: ScopeStack) -> dict[str, Any]:
    """What a fresh build and an incremental history must agree on."""
    vectors = stack.vectors.data
    return {
        "text_units": stack.texts(),
        "entities": {
            e.name: sorted(e.text_unit_ids or [])
            for e in vectors.get("entities", {}).values()
        },
        "relationships": {
            r.id: sorted(r.text_unit_ids or [])
            for r in vectors.get("relationships", {}).values()
        },
        "graph": {
            collection: stack.graph.ids(collection)
            for collection in ("entities", "relationships")
        },
        "registry": {
            (r.file_path, r.scope): (
                r.content_hash,
                sorted(r.entity_ids),
                sorted(r.relationship_ids),
                sorted(r.text_unit_ids),
            )
            for r in stack.registry.list_all()
        },
    }


def test_retiring_the_old_scope_of_a_moved_corpus_removes_its_content(
    registry, tmp_path, caplog
) -> None:
    stack = ScopeStack(registry, tmp_path)
    other = write_corpus(tmp_path / "other", {"bank.txt": C_TEXT})
    old = write_corpus(
        tmp_path / "old", {"contract.txt": A_TEXT, "depot.txt": DEPOT_TEXT}
    )
    stack.run(other)
    stack.run(old)
    assert stack.entity_names() == {"Vendor", "Buyer", "Bank", "Depot"}

    # Moved, and depot.txt was removed during the move.
    new = old.rename(tmp_path / "new")
    (new / "depot.txt").unlink()
    with caplog.at_level("WARNING"):
        unretired = stack.run(new)
    # Without the flag nothing is deleted: the old records keep Depot, and
    # the run names the old directory.
    assert unretired.incremental_delta.deleted == []
    assert "Depot" in stack.entity_names()
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        old.as_posix() in m and "--retire-source-scope" in m for m in warnings
    ), warnings

    stack.config.indexing.retire_source_scopes = [f"{old.as_posix()}/"]
    retired = stack.run(new)

    delta = retired.incremental_delta
    assert len(delta.deleted) == 2  # both records of the old directory
    assert delta.new == delta.changed == []
    # Depot only came from the old depot.txt; Buyer and Bank are shared with
    # the other corpus and kept.
    assert stack.entity_names() == {"Vendor", "Buyer", "Bank"}
    assert stack.texts() == {A_TEXT, C_TEXT}
    assert {r.scope for r in registry.list_all()} == {
        _scope_of(retired),
        _scope_of(stack.run(other)),
    }

    fresh = ScopeStack(FakeDocStatusStore(), tmp_path / "fresh")
    fresh.run(other)
    fresh.run(new)
    assert _store_snapshot(stack) == _store_snapshot(fresh)


def test_retiring_the_runs_own_scope_fails_without_touching_anything(
    registry, tmp_path
) -> None:
    stack = ScopeStack(registry, tmp_path)
    source = write_corpus(tmp_path / "src-a", {"contract.txt": A_TEXT})
    stack.run(source)
    before = (_store_snapshot(stack), registry.list_all())

    stack.config.indexing.retire_source_scopes = [f"{source.as_posix()}/"]
    with pytest.raises(PipelineExecutionError, match="own source scope"):
        stack.run(source)

    assert (_store_snapshot(stack), registry.list_all()) == before


def test_two_hosts_sharing_a_registry_never_delete_each_others_content(
    registry, tmp_path, caplog
) -> None:
    # Two hosts index their own directory into one registry and namespace.
    # Each host sees only its own directory: the other's is hidden while it
    # runs, which is what a missing directory looked like to #191.
    stack = ScopeStack(registry, tmp_path)
    host_a = write_corpus(tmp_path / "host-a" / "corpus", {"contract.txt": A_TEXT})
    host_b = write_corpus(tmp_path / "host-b" / "corpus", {"contract.txt": B_TEXT})

    def run_alone(source: Path, hidden: Path) -> PipelineContext:
        parked = hidden.rename(hidden.with_name("parked"))
        try:
            return stack.run(source)
        finally:
            parked.rename(hidden)

    run_alone(host_a, host_b)
    with caplog.at_level("WARNING"):
        first_b = run_alone(host_b, host_a)
    assert any(
        host_a.as_posix() in r.getMessage() for r in caplog.records
    ), "the other local scope is reported"
    records = {r.doc_id: r for r in registry.list_all()}

    for _ in range(2):
        for source, hidden in ((host_a, host_b), (host_b, host_a)):
            context = run_alone(source, hidden)
            delta = context.incremental_delta
            assert delta.new == delta.changed == delta.deleted == []
            assert not stack.model.extractions
            assert stack.texts() == {A_TEXT, B_TEXT}
    assert {r.doc_id: r for r in registry.list_all()} == records
    assert len(records) == 2 and _scope_of(first_b) in {
        r.scope for r in records.values()
    }


@pytest.mark.parametrize("retire_value", ["/mnt/corpus/", "/mnt/corpus"])
def test_a_configured_scope_with_a_trailing_slash_can_be_retired(
    registry, tmp_path, retire_value
) -> None:
    # A configured source_scope is stored as given; a retire value is
    # normalized. Both spellings must reach the stored records.
    stack = ScopeStack(registry, tmp_path)
    parsing = stack.config.processing.document_parsing
    source = write_corpus(tmp_path / "src", {"contract.txt": A_TEXT})
    parsing.source_scope = "/mnt/corpus/"
    first = stack.run(source)
    assert _scope_of(first).endswith("|/mnt/corpus/")

    parsing.source_scope = "corpus-renamed"
    stack.config.indexing.retire_source_scopes = [retire_value]
    write_corpus(source, {"depot.txt": DEPOT_TEXT})
    context = stack.run(source)

    assert len(context.incremental_delta.deleted) == 1
    assert {r.scope for r in registry.list_all()} == {_scope_of(context)}
    assert stack.texts() == {DEPOT_TEXT}
    assert stack.entity_names() == {"Depot"}
