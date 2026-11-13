# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What an incremental run deletes when files go missing, end to end.

On the embedded stores with a fake Ollama server: a file the run cannot
read for its type (an optional extra dropped from the image, a loader
plugin not imported) keeps its indexed content as a parse failure does; a
run that would delete more than ``indexing.max_deletion_share`` of its
scope's documents stops before it writes unless confirmed; and the counts
an operator alarms on reach the metrics sink.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.adapters.ingestion.file_loaders import TextFileLoader
from unified_kg_rag.adapters.ingestion.parser import _LOADERS, ParserFactory
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import (
    DocStatus,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.domain.models.config import SeedGraphConfig
from unified_kg_rag.shared.exceptions import ConfirmationDeclinedError

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

no_aws = e2e.no_aws

# A file type no loader reads unless a test registers one.
_PLUGIN = ".note"


class _Sink:
    """A metrics sink keeping every record it was sent."""

    def __init__(self) -> None:
        self.records: list[dict[str, float]] = []

    def emit(self, namespace: str, metrics: Any, dimensions: Any = None) -> None:
        self.records.append(dict(metrics))

    def value(self, name: str) -> float:
        values = [r[name] for r in self.records if name in r]
        assert len(values) == 1, (name, values)
        return values[0]


class _Corpus:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.source = tmp_path / "corpus"
        self.source.mkdir()
        self.config = e2e._config(tmp_path / "stores", tmp_path / "viz")
        self.server = e2e.FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM)
        self.runs = 0

    @property
    def providers(self) -> Providers:
        return e2e._providers(self.config, self.server, e2e.FakeCrossEncoder("r"))

    def run(
        self, sink: _Sink | None = None, confirm_deletions: Any = None
    ) -> PipelineContext:
        self.runs += 1
        cache = self.tmp_path / f"run-{self.runs}"
        pipeline = DataIngestionPipeline(
            self.config,
            PipelineConfig(
                stages_enabled=dict.fromkeys(PipelineStageType, True),
                local_directory=cache,
            ),
            source_directory=self.source,
            providers=self.providers,
            metrics_sink=sink,
            confirm_deletions=confirm_deletions,
        )
        for stage in pipeline.stages:
            if isinstance(stage, ps.GraphResolutionStage):
                stage.resolver.entity_resolver.use_process_pool = False
                stage.resolver.relationship_resolver.use_process_pool = False
        try:
            context = pipeline.run(self.source, pipeline_id=cache.name)
        finally:
            pipeline.close()
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        return context

    def registry(self) -> dict[str, DocStatus]:
        return e2e._registry(self.providers)

    def texts(self) -> set[str]:
        return e2e._stored_texts(self.config, self.providers)


def _with_plugin_loader(corpus: _Corpus, sink: _Sink | None = None) -> None:
    ParserFactory.register_loader(_PLUGIN, TextFileLoader, {}, "Note")
    try:
        corpus.run(sink)
    finally:
        _LOADERS.unregister(_PLUGIN)
    assert _PLUGIN not in ParserFactory.get_supported_extensions()


def test_files_no_loader_reads_keep_what_they_indexed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Old behaviour: the run without the loader recorded the two notes
    # deleted and removed their text units and entities, COMPLETED.
    corpus = _Corpus(tmp_path)
    (corpus.source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (corpus.source / f"handbook{_PLUGIN}").write_text(
        "Depot stores widgets for Vendor."
    )
    (corpus.source / f"faq{_PLUGIN}").write_text("Carrier moves gadgets for Depot.")
    _with_plugin_loader(corpus)
    indexed = corpus.texts()
    assert len(indexed) == 3

    sink = _Sink()
    with caplog.at_level(logging.WARNING):
        context = corpus.run(sink)

    assert corpus.registry() == {
        "contract.txt": DocStatus.PROCESSED,
        f"faq{_PLUGIN}": DocStatus.PROCESSED,
        f"handbook{_PLUGIN}": DocStatus.PROCESSED,
    }
    assert corpus.texts() == indexed
    assert {"Depot", "Carrier"} <= e2e._entity_names(corpus.providers)
    assert context.incremental_delta is not None
    assert context.incremental_delta.deleted == []
    assert len(context.incremental_delta.failed) == 2
    assert sink.value("total_files_skipped") == 2
    assert sink.value("total_documents_deleted") == 0
    skipped = [r.getMessage() for r in caplog.records if "Skipping" in r.getMessage()]
    assert len(skipped) == 1
    assert f"2 '{_PLUGIN}'" in skipped[0] and "keeps what they indexed" in skipped[0]

    # Once the loader is back, the notes are read again, unchanged.
    _with_plugin_loader(corpus)
    assert corpus.texts() == indexed


def test_a_seed_only_scope_with_a_stray_file_still_builds(tmp_path: Path) -> None:
    # A skipped file is not a parse failure: the empty corpus beside seed
    # graphs stays a seed-only run (#509), not a failed parsing stage.
    corpus = _Corpus(tmp_path)
    (corpus.source / "README.unknown").write_text("Not a source document.")
    seed = tmp_path / "seed.jsonl"
    seed.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                {"kind": "node", "id": "v", "name": "Vendor", "type": "ORGANIZATION"},
                {"kind": "node", "id": "p", "name": "Harbor Port", "type": "LOCATION"},
                {"kind": "edge", "source": "v", "target": "p", "type": "SHIPS_FROM"},
            )
        )
    )
    corpus.config.processing.seed_graphs = [
        SeedGraphConfig(name="procure", kind="jsonl", path=str(seed))
    ]

    context = corpus.run()

    assert context.skipped_source_files == [str(corpus.source / "README.unknown")]
    assert {"Vendor", "Harbor Port"} <= e2e._entity_names(corpus.providers)


def _twelve_documents(corpus: _Corpus) -> None:
    for k in range(12):
        (corpus.source / f"d{k:02d}.txt").write_text(f"Vendor {k} supplies Buyer {k}.")


def test_a_mass_deletion_stops_before_it_writes_unless_confirmed(
    tmp_path: Path,
) -> None:
    corpus = _Corpus(tmp_path)
    _twelve_documents(corpus)
    corpus.run()
    indexed, registry = corpus.texts(), corpus.registry()
    assert len(registry) == 12
    # A partial sync: 7 of the 12 files are missing (58% > 50%).
    for k in range(7):
        (corpus.source / f"d{k:02d}.txt").unlink()

    with pytest.raises(ConfirmationDeclinedError, match="No one can confirm"):
        corpus.run()
    questions: list[str] = []

    def no(question: str) -> bool:
        questions.append(question)
        return False

    with pytest.raises(ConfirmationDeclinedError, match="Deletion cancelled"):
        corpus.run(confirm_deletions=no)
    assert "deletes 7 of the 12 documents" in questions[0]
    assert "indexing.max_deletion_share 0.5" in questions[0]
    assert corpus.registry() == registry
    assert corpus.texts() == indexed

    sink = _Sink()
    context = corpus.run(sink, confirm_deletions=lambda question: True)

    assert context.incremental_delta is not None
    assert len(context.incremental_delta.deleted) == 7
    assert sorted(corpus.registry()) == [f"d{k:02d}.txt" for k in range(7, 12)]
    assert sink.value("total_documents_deleted") == 7


def test_deleting_up_to_the_share_or_with_the_check_off_asks_nothing(
    tmp_path: Path,
) -> None:
    corpus = _Corpus(tmp_path)
    _twelve_documents(corpus)
    corpus.run()
    for k in range(6):  # 6 of 12: exactly the share, not over it
        (corpus.source / f"d{k:02d}.txt").unlink()
    sink = _Sink()
    corpus.run(sink)
    assert sink.value("total_documents_deleted") == 6

    # 6 documents left: under the 10 a checked scope needs.
    for k in range(6, 11):
        (corpus.source / f"d{k:02d}.txt").unlink()
    corpus.run()
    assert sorted(corpus.registry()) == ["d11.txt"]


def test_the_check_off_deletes_without_asking(tmp_path: Path) -> None:
    corpus = _Corpus(tmp_path)
    _twelve_documents(corpus)
    corpus.config.indexing.max_deletion_share = None
    corpus.run()
    for k in range(11):
        (corpus.source / f"d{k:02d}.txt").unlink()
    corpus.run()
    assert sorted(corpus.registry()) == ["d11.txt"]


def _with_a_large_seed_graph(corpus: _Corpus, tmp_path: Path) -> None:
    seed = tmp_path / "seed.jsonl"
    seed.write_text(
        "".join(
            json.dumps(
                {"kind": "node", "id": f"n{k}", "name": f"Part {k}", "type": "PRODUCT"}
            )
            + "\n"
            for k in range(60)
        )
    )
    corpus.config.processing.seed_graphs = [
        SeedGraphConfig(name="catalog", kind="jsonl", path=str(seed))
    ]


def test_seed_shards_neither_dilute_the_share_nor_ask(tmp_path: Path) -> None:
    corpus = _Corpus(tmp_path)
    _twelve_documents(corpus)
    _with_a_large_seed_graph(corpus, tmp_path)
    corpus.run()
    shards = len(corpus.registry()) - 12
    assert shards > 12  # more shard records than documents
    # 11 of the 12 documents gone: 92% of the documents, under half of the
    # records with the shards counted.
    for k in range(11):
        (corpus.source / f"d{k:02d}.txt").unlink()
    questions: list[str] = []

    def no(question: str) -> bool:
        questions.append(question)
        return False

    with pytest.raises(ConfirmationDeclinedError, match="Deletion cancelled"):
        corpus.run(confirm_deletions=no)
    assert "deletes 11 of the 12 documents" in questions[0]

    # The seed graph taken out of the configuration: its shards go, unasked.
    (tmp_path / "second").mkdir()
    corpus2 = _Corpus(tmp_path / "second")
    _twelve_documents(corpus2)
    _with_a_large_seed_graph(corpus2, tmp_path)
    corpus2.run()
    corpus2.config.processing.seed_graphs = []
    context = corpus2.run()
    assert context.incremental_delta is not None
    assert len(context.incremental_delta.deleted) == shards
    assert len(context.incremental_delta.seed_deleted) == shards
    assert len(corpus2.registry()) == 12


def test_coverage_counts_reach_the_metrics_sink(tmp_path: Path) -> None:
    corpus = _Corpus(tmp_path)
    for k in range(3):
        (corpus.source / f"d{k}.txt").write_text(f"Vendor {k} supplies Buyer.")
    (corpus.source / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    (corpus.source / f"a{_PLUGIN}").write_text("Depot stores widgets.")
    (corpus.source / f"b{_PLUGIN}").write_text("Carrier moves gadgets.")
    sink = _Sink()

    corpus.run(sink)

    assert sink.value("total_files_failed") == 1
    assert sink.value("total_files_skipped") == 2
    for name in (
        "total_documents_excluded",
        "total_documents_deleted",
        "total_documents_failed",
        "total_documents_retrying",
        "total_documents_without_claims",
    ):
        assert sink.value(name) == 0, name
