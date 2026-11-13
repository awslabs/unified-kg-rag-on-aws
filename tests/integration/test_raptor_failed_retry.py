# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A FAILED document's retry with the raptor methodology (graphrag + raptor,
registry on; docs/design/retrieval-hipporag-raptor.md, "Removal").

Run 1 fails one document's extraction (cut-off output) and records it
FAILED; run 2 retries it. The retry removes the document's text units and
writes them back under the same ids with the same text: the summaries
citing them are kept (no summary asked again, ``refresh: on_reset`` loses
none), while a real deletion in the same run still deletes and marks the
tree pruned. Every outcome, a run killed at any write included, converges
with a fresh build of the same corpus, in English, Korean and Japanese.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from tests.fixtures.fakes.chat_models import CountingResponder
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_raptor_e2e as rt
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - fixture
from tests.integration.test_ingestion_stages import _source_text
from tests.integration.test_name_variant_convergence import _Died, _dying
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
from unified_kg_rag.application.cli.run_ingestion_pipeline import (
    IngestionPipelineRunner,
)
from unified_kg_rag.application.ingestion.summary_tree_stage import SummaryTreeStage
from unified_kg_rag.application.storage.removal import TextUnitRemoval
from unified_kg_rag.domain.ingestion.summary_tree import (
    stored_leaf_entry,
    unit_leaf_entry,
)
from unified_kg_rag.domain.models import (
    Config,
    Constants,
    DocStatus,
    Methodology,
    PipelineStageStatus,
    TextUnit,
)
from unified_kg_rag.domain.models.config import LanguageCode
from unified_kg_rag.domain.retrieval.filters import translated_text_field

pytestmark = pytest.mark.integration

# Extraction output cut off mid-entity: the document is recorded FAILED.
_CUT_OFF = "<entities><entity><type>ORGANIZATION</type></entity><entity><na"

TEXTS = {
    "en": (
        "Zorbix",
        "Zorbix and Vendor trade widgets.\n\nZorbix has a plant in the south.",
    ),
    "ko": (
        "조르빅스",
        "조르빅스와 Vendor는 부품을 거래한다.\n\n조르빅스는 남부에 공장이 있다.",
    ),
    "ja": (
        "ゾルビクス",
        "ゾルビクスとVendorは部品を取引する。\n\nゾルビクスは南部に工場がある。",
    ),
}


def _setup(
    tmp: Path, lang: str, refresh: str, state: dict[str, bool]
) -> tuple[Config, Providers, Path, CountingResponder]:
    name, text = TEXTS[lang]
    config = e2e._config(tmp / "stores", tmp / "viz")
    config.providers.doc_status.enabled = True
    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.RAPTOR]
    config.processing.document_parsing.source_scope = "corpus"
    tree = config.indexing.summary_tree
    tree.min_nodes_to_cluster = 4
    tree.target_children = 3
    tree.refresh = refresh  # type: ignore[assignment]
    first = text.split("\n")[0]

    def respond(system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system and first in _source_text(
            human
        ):
            if state["cut"]:
                return _CUT_OFF
            return e2e.stages._extraction(first, (name, "Vendor"))
        return e2e._respond(system, human)

    calls = CountingResponder(respond, summarize=rt.tree_summary)
    server = FakeOllama(calls, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(config, server, e2e.FakeCrossEncoder("r"))
    source = rt.corpus(tmp)
    (source / "zz.txt").write_text(text)
    return config, providers, source, calls


def _state(providers: Providers) -> tuple[Any, list[str]]:
    return rt.shape(rt.tree_nodes(providers)), sorted(e2e._entity_names(providers))


# A run whose only document fails again has no extraction output and fails:
# another document changes in that run.
_EXTRA = ("mill.txt", "Mill makes iron for Vendor.\n\nMill has an office in the east.")


def _fresh(
    tmp: Path,
    lang: str,
    cut: bool,
    *,
    without: str | None = None,
    extra: bool = False,
) -> Any:
    """The state of a fresh build of the corpus (its zz.txt cut off or not,
    ``without`` one file, with the ``_EXTRA`` change)."""
    config, providers, source, _ = _setup(tmp, lang, "every_run", {"cut": cut})
    if without:
        (source / without).unlink()
    if extra:
        (source / _EXTRA[0]).write_text(_EXTRA[1])
    context = rt.ingest(config, providers, source, tmp / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    return _state(providers)


def _failed_first(
    tmp: Path, lang: str, refresh: str
) -> tuple[Config, Providers, Path, CountingResponder, dict[str, bool]]:
    state = {"cut": True}
    config, providers, source, calls = _setup(tmp, lang, refresh, state)
    context = rt.ingest(config, providers, source, tmp / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert e2e._registry(providers)["zz.txt"] is DocStatus.FAILED
    return config, providers, source, calls, state


@pytest.mark.parametrize("lang", ["en", "ko", "ja"])
@pytest.mark.parametrize("refresh", ["every_run", "on_reset"])
def test_a_retry_keeps_the_summaries_of_text_it_writes_back_unchanged(
    tmp_path: Path, no_aws, lang: str, refresh: str  # noqa: F811 - the fixture
) -> None:
    config, providers, source, calls, state = _failed_first(tmp_path, lang, refresh)
    built = rt.shape(rt.tree_nodes(providers))
    state["cut"] = False
    before = calls.kinds.copy()
    context = rt.ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert e2e._registry(providers)["zz.txt"] is DocStatus.PROCESSED
    # The run JSON counts the retry among the changed documents.
    runner = IngestionPipelineRunner.__new__(IngestionPipelineRunner)
    runner.args = argparse.Namespace(max_failure_rate=None)
    runner.pipeline = SimpleNamespace(executed_stages=frozenset())  # type: ignore[assignment]
    delta = runner._json_result(context)["delta"]
    assert (delta["changed"], delta["retried"]) == (1, 1), delta
    # Neither deleted nor asked again, whatever the refresh.
    assert rt.shape(rt.tree_nodes(providers)) == built
    assert calls.since(before)["summary"] == 0, calls.since(before)
    assert not (rt.tree_record(providers) or {}).get("pruned")
    assert _state(providers) == _fresh(tmp_path / "fresh", lang, cut=False)


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_a_retry_that_fails_again_converges(
    tmp_path: Path, no_aws, lang: str  # noqa: F811 - the fixture
) -> None:
    config, providers, source, _, _ = _failed_first(tmp_path, lang, "every_run")
    (source / _EXTRA[0]).write_text(_EXTRA[1])
    context = rt.ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert e2e._registry(providers)["zz.txt"] is DocStatus.FAILED
    assert not (rt.tree_record(providers) or {}).get("pruned")
    assert _state(providers) == _fresh(tmp_path / "fresh", lang, cut=True, extra=True)


@pytest.mark.parametrize("lang", ["en", "ja"])
def test_a_real_deletion_beside_a_retry_still_prunes_and_rebuilds(
    tmp_path: Path, no_aws, lang: str  # noqa: F811 - the fixture
) -> None:
    config, providers, source, _, state = _failed_first(tmp_path, lang, "every_run")
    mill = {
        hit["_id"]
        for hit in e2e._stored(
            config, providers, config.indexing.opensearch.text_units_index_prefix
        )
        if "Mill" in hit["_source"]["text"]
    }
    state["cut"] = False
    (source / "mill.txt").unlink()
    pruned: list[bool] = []
    original = OpenSearchIndexer._mark_summary_tree_pruned

    def mark(self: OpenSearchIndexer, suffix: str) -> None:
        pruned.append(True)
        original(self, suffix)

    with patch.object(OpenSearchIndexer, "_mark_summary_tree_pruned", mark):
        context = rt.ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert pruned, "the deletion of mill.txt's summaries marks the tree pruned"
    nodes = rt.tree_nodes(providers)
    assert not any(set(n["text_unit_ids"]) & mill for n in nodes)
    assert not (rt.tree_record(providers) or {}).get("pruned")
    assert _state(providers) == _fresh(
        tmp_path / "fresh", lang, cut=False, without="mill.txt"
    )


def test_a_failed_read_of_the_stored_leaves_deletes_their_summaries(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Fail-closed: the removal cannot tell the leaves are unchanged, so the
    # summaries citing them go and the next build asks for them again.
    config, providers, source, calls, state = _failed_first(tmp_path, "en", "every_run")
    state["cut"] = False
    original = OpenSearchIndexer.read_text_unit_fields
    reads = {"n": 0}

    def failing(self: OpenSearchIndexer, *args: Any, **kwargs: Any) -> Any:
        reads["n"] += 1
        if reads["n"] == 1:
            raise ConnectionError("store unavailable")
        return original(self, *args, **kwargs)

    before = calls.kinds.copy()
    with patch.object(OpenSearchIndexer, "read_text_unit_fields", failing):
        context = rt.ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert calls.since(before)["summary"] > 0
    assert not (rt.tree_record(providers) or {}).get("pruned")
    assert _state(providers) == _fresh(tmp_path / "fresh", "en", cut=False)


# (method, nth call, kill after the call, owner): every write of the retry's
# removal, commit and tree build, and the new read of the stored leaves.
KILLS = [
    ("_rewritten_leaves", 1, False, TextUnitRemoval),
    ("_rewritten_leaves", 1, True, TextUnitRemoval),
    ("read_text_unit_fields", 1, True, OpenSearchIndexer),
    ("delete_summary_nodes_citing", 1, False, OpenSearchIndexer),
    ("delete_summary_nodes_citing", 1, True, OpenSearchIndexer),
    ("delete_by_id", 1, True, OpenSearchIndexer),
    ("delete_by_id", 2, True, OpenSearchIndexer),
    ("upsert_entities", 1, True, OpenSearchIndexer),
    ("upsert_relationships", 1, True, SQLiteGraphIndexer),
    ("_execute_core", 1, False, SummaryTreeStage),
]


@pytest.mark.parametrize("lang", ["en", "ko", "ja"])
@pytest.mark.parametrize(
    "kill",
    KILLS,
    ids=lambda k: f"{k[3].__name__}.{k[0]}#{k[1]}{'after' if k[2] else 'before'}",
)
@pytest.mark.parametrize("resume", [True, False])
def test_a_retry_killed_anywhere_converges(
    tmp_path: Path, no_aws, lang: str, kill: tuple[Any, ...], resume: bool  # noqa: F811
) -> None:
    config, providers, source, _, state = _failed_first(tmp_path, lang, "every_run")
    state["cut"] = False
    method, nth, after, owner = kill
    with patch.object(owner, method, _dying(method, nth, after, owner=owner)):
        with pytest.raises(_Died):
            rt.ingest(config, providers, source, tmp_path / "run-2")
    run = "run-2" if resume else "run-3"
    context = rt.ingest(config, providers, source, tmp_path / run)
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert _state(providers) == _fresh(tmp_path / "fresh", lang, cut=False)
    assert not (rt.tree_record(providers) or {}).get("pruned")


@pytest.mark.parametrize("translation", [False, True])
def test_a_text_unit_reads_back_as_the_leaf_the_removal_predicts(
    tmp_path: Path, translation: bool
) -> None:
    # unit_leaf_entry predicts what the store writes: compared with the
    # stored row, an unchanged unit matches and any change to the text the
    # tree reads (its translation into the index language) or to its
    # filters does not, so its summaries are deleted.
    config = Config()
    config.providers.vector.kind = "sqlite"
    config.providers.vector.sqlite_path = str(tmp_path / "vector.sqlite3")
    language = LanguageCode.EN.value
    if translation:
        config.processing.translation.enabled = True
        config.processing.translation.source_language = LanguageCode.EN
        config.processing.translation.target_language = LanguageCode.KO
        config.processing.translation.additional_target_languages = [LanguageCode.JA]
        language = LanguageCode.KO.value
    filters = {Constants.FILTERS.value: {"region": "south"}}
    units = [
        TextUnit(
            id="t-plain", text="Vendor ships widgets.", attributes=filters, n_tokens=3
        ),
        TextUnit(
            id="t-translated",
            text="Vendor ships gears.",
            # A language with no translated field is not stored.
            translated_texts={
                "ko": "벤더는 기어를 출하한다.",
                "ja": "歯車",
                "zh": "齿轮",
            },
            n_tokens=3,
        ),
    ]
    providers = Providers(config, embedding_factory=HashingEmbeddingFactory())
    indexer = providers.build_vector_indexer()
    try:
        indexer.initialize()
        stats = indexer.index_text_units(units)
        assert stats.failed_items == 0, stats.errors
        field = translated_text_field(language)
        rows = indexer.read_text_unit_fields(
            [u.id for u in units], "", fields=["text", field, "attributes"]
        )
    finally:
        indexer.close()
    translated = config.processing.translation.translated_languages
    stored = {str(row["id"]): stored_leaf_entry(row, field) for row in rows or []}
    assert set(stored) == {u.id for u in units}
    for unit in units:
        assert stored[unit.id] == unit_leaf_entry(unit, language, translated)
        changes: list[dict[str, Any]] = [
            {"attributes": {Constants.FILTERS.value: {"region": "north"}}}
        ]
        if translation and unit.translated_texts:
            # The tree reads the translation, not the source text.
            changes.append({"translated_texts": {"ko": "다른 번역."}})
            changes.append({"translated_texts": None})
        else:
            changes.append({"text": unit.text + " Again."})
        for change in changes:
            other = unit.model_copy(update=change)
            assert stored[unit.id] != unit_leaf_entry(other, language, translated)
