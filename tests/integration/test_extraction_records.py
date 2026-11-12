# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-unit extraction records on the embedded stores (no AWS, no server).

Whole-pipeline runs with ``backend: local`` and a fake model server:

- the extraction store sends an edited document's unchanged chunks to the
  model no more, and a gleaned record is not gleaned again;
- a gleaning correction edits only its own text unit's statements, so a full
  build and incremental runs over the same final corpus store the same edges,
  descriptions included, whichever order the documents arrived in.
"""

from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_embedded_e2e import (
    FakeCrossEncoder,
    _config,
    _providers,
)
from tests.integration.test_embedded_e2e import no_aws as _no_aws  # noqa: F401
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.models import (
    Config,
    LanguageCode,
    PipelineConfig,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.shared.utils import generate_stable_id

pytestmark = pytest.mark.integration

NAMES = ("Acme", "Buyer", "Depot")


def _names(text: str) -> list[str]:
    return sorted((n for n in NAMES if re.search(rf"\b{n}\b", text)), key=text.find)


class _Model:
    """Extraction: an edge between consecutive names, SUPPLIES when the text
    says so. Gleaning: "parts" reverses the unit's first edge, "tools"
    retypes it to SUPPLIES."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    def __call__(self, system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system:
            text = stages._source_text(human).strip()
            self.calls["extract"] += 1
            names = _names(text)
            kind = "SUPPLIES" if "supplies" in text else "TRADES_WITH"
            entities = "".join(
                f"<entity><name>{n}</name><type>ORGANIZATION</type>"
                f"<description>{n}: {text}</description><confidence>9</confidence>"
                f"<source_text>{n}</source_text></entity>"
                for n in names
            )
            relationships = "".join(
                f"<relationship><source>{a}</source><target>{b}</target>"
                f"<type>{kind}</type><description>{a}-{b}: {text}</description>"
                f"<strength>8</strength><source_text>{text}</source_text>"
                "</relationship>"
                for a, b in zip(names, names[1:], strict=False)
            )
            return (
                f"<entities>{entities}</entities>"
                f"<relationships>{relationships}</relationships>"
            )
        if "graph refinement specialist" in system:
            text = stages._source_text(human).strip()
            self.calls["glean"] += 1
            names = _names(text)
            issue = ""
            if len(names) >= 2 and ("parts" in text or "tools" in text):
                a, b = names[:2]
                kind = "SUPPLIES" if "supplies" in text else "TRADES_WITH"
                source, target = (b, a) if "parts" in text else (a, b)
                corrected = "SUPPLIES" if "tools" in text else kind
                issue = (
                    "<issue><issue_type>RELATIONSHIP_CORRECTION</issue_type>"
                    f"<details><source>{a}</source><target>{b}</target>"
                    f"<type>{kind}</type><corrected_source>{source}"
                    f"</corrected_source><corrected_target>{target}"
                    f"</corrected_target><corrected_type>{corrected}"
                    "</corrected_type><description>corrected</description>"
                    "</details><justification>x</justification>"
                    f"<text_evidence>{text}</text_evidence></issue>"
                )
            return (
                "<refinement_plan><identified_issues>"
                f"{issue}</identified_issues></refinement_plan>"
            )
        return stages._respond(system, human)


class _Store:
    """One set of embedded stores, ingested into run after run."""

    def __init__(self, root: Path, model: _Model) -> None:
        self.root = root
        self.config: Config = _config(root / "stores", root / "viz")
        self.config.processing.gleaning.enabled = True
        self.config.processing.gleaning.max_rounds = 1
        self.providers = _providers(
            self.config, FakeOllama(model, "x", 32), FakeCrossEncoder("r")
        )
        self.runs = 0

    def ingest(self, corpus: dict[str, str], *, force_rebuild: bool = False) -> None:
        source = self.root / "corpus"
        if source.exists():
            shutil.rmtree(source)
        source.mkdir(parents=True)
        for name, text in corpus.items():
            (source / name).write_text(text)
        self.runs += 1
        cache = self.root / f"run-{self.runs}"
        pipeline = DataIngestionPipeline(
            self.config,
            PipelineConfig(
                stages_enabled=dict.fromkeys(PipelineStageType, True),
                local_directory=cache,
                force_rebuild=force_rebuild,
            ),
            source_directory=source,
            providers=self.providers,
        )
        for stage in pipeline.stages:
            if isinstance(stage, ps.GraphResolutionStage):
                stage.resolver.entity_resolver.use_process_pool = False
                stage.resolver.relationship_resolver.use_process_pool = False
        try:
            context = pipeline.run(source, pipeline_id=cache.name)
        finally:
            pipeline.close()
        assert context.status is PipelineStageStatus.COMPLETED, [
            r.error_message for r in context.stage_results if r.error_message
        ]

    def record_keys(self) -> set[str]:
        """Every record key in the default suffix's extraction store."""
        vector = self.providers.build_vector_indexer()
        try:
            alias = vector._get_name(
                self.config.indexing.opensearch.extraction_records_index_prefix, None
            )
            hits = vector.opensearch_client.search(
                index=alias, body={"size": 1000, "query": {"match_all": {}}}
            )["hits"]["hits"]
        finally:
            vector.close()
        return {hit["_id"] for hit in hits}

    def edges(self) -> list[tuple]:
        graph = self.providers.build_graph_indexer()
        try:
            ids = [i for i, _ in graph.read_entity_names()]
            relationships = graph.read_relationships(
                graph.find_incident_relationship_ids(ids)
            )
        finally:
            graph.close()
        return sorted(
            (
                r.source_name,
                r.target_name,
                r.type,
                r.weight,
                len(r.text_unit_ids or []),
                frozenset((r.description or "").split("\n")),
            )
            for r in relationships
        )


def test_an_edited_document_re_extracts_only_its_changed_chunks(
    tmp_path: Path,
) -> None:
    model = _Model()
    store = _Store(tmp_path, model)
    first = "Acme supplies goods to Buyer here."
    second = "Depot and Buyer trade goods today."
    store.ingest({"a.txt": f"{first} {second}"})
    # Two chunks (60 characters each), each extracted and gleaned once.
    assert model.calls == Counter(extract=2, glean=2)

    model.calls.clear()
    store.ingest({"a.txt": f"{first} {second.replace('today', 'often')}"})

    # The unchanged chunk takes its stored, already gleaned record.
    assert model.calls == Counter(extract=1, glean=1)


def test_a_reset_reuses_the_records_and_force_rebuild_does_not(
    tmp_path: Path,
) -> None:
    model = _Model()
    store = _Store(tmp_path, model)
    corpus = {"a.txt": "Acme supplies goods to Buyer here."}
    store.ingest(corpus)
    edges = store.edges()

    # A reset rebuilds every index but the extraction store.
    store.config.indexing.reset = True
    model.calls.clear()
    store.ingest(corpus)
    assert model.calls == Counter()
    assert store.edges() == edges

    model.calls.clear()
    store.ingest(corpus, force_rebuild=True)
    assert model.calls == Counter(extract=1, glean=1)
    assert store.edges() == edges


def test_a_reset_gives_records_stored_under_other_keys_the_ids_of_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Records an earlier build stored hold the ids its entity_key minted; a
    # reset reusing them rebuilt the graph under those ids, which a new
    # document's statement of the same name no longer takes.
    model = _Model()
    store = _Store(tmp_path, model)
    corpus = {"a.txt": "Acme supplies goods to Buyer here."}
    with monkeypatch.context() as earlier:
        earlier.setattr(
            BaseProcessor,
            "_generate_entity_id",
            staticmethod(lambda name: generate_stable_id(f"earlier:{name}")),
        )
        earlier.setattr(
            BaseProcessor,
            "_generate_relationship_id",
            staticmethod(lambda *names: generate_stable_id(f"earlier:{names}")),
        )
        store.ingest(corpus)

    store.config.indexing.reset = True
    model.calls.clear()
    store.ingest(corpus)

    assert model.calls == Counter()
    graph = store.providers.build_graph_indexer()
    try:
        names = graph.read_entity_names()
        relationships = graph.read_relationships(
            graph.find_incident_relationship_ids([i for i, _ in names])
        )
    finally:
        graph.close()
    assert sorted(names) == sorted(
        (BaseProcessor._generate_entity_id(name), name) for _, name in names
    )
    assert relationships
    for r in relationships:
        assert (r.id, r.source_id, r.target_id) == (
            BaseProcessor._generate_relationship_id(
                r.source_name or "", r.target_name or "", r.type or ""
            ),
            BaseProcessor._generate_entity_id(r.source_name or ""),
            BaseProcessor._generate_entity_id(r.target_name or ""),
        )


def test_a_reset_keeps_only_the_records_its_text_units_name(
    tmp_path: Path,
) -> None:
    store = _Store(tmp_path, _Model())
    kept = "Acme supplies goods to Buyer here."
    store.ingest({"a.txt": kept, "b.txt": "Depot and Buyer trade goods today."})
    before = store.record_keys()

    # b.txt deleted and gleaning switched off (another record fingerprint):
    # the reset's text units name only a.txt's ungleaned records.
    store.config.indexing.reset = True
    store.config.processing.gleaning.enabled = False
    store.ingest({"a.txt": kept})
    after = store.record_keys()

    assert after and not after & before
    # The next reset of the same corpus reuses (and keeps) every one of them.
    store.ingest({"a.txt": kept})
    assert store.record_keys() == after


@pytest.mark.parametrize(
    "runs",
    [
        # A reversal, then another document stating the uncorrected edge.
        [
            {"p.txt": "Acme and Buyer trade parts."},
            {
                "p.txt": "Acme and Buyer trade parts.",
                "n.txt": "Acme and Buyer trade goods.",
            },
        ],
        # The uncorrected edge first, then the reversing document.
        [
            {"n.txt": "Acme and Buyer trade goods."},
            {
                "p.txt": "Acme and Buyer trade parts.",
                "n.txt": "Acme and Buyer trade goods.",
            },
        ],
        # A retype, then a document stating the uncorrected edge.
        [
            {"t.txt": "Acme and Buyer trade tools."},
            {
                "t.txt": "Acme and Buyer trade tools.",
                "n.txt": "Acme and Buyer trade goods.",
            },
        ],
    ],
    ids=["reversal-then-restatement", "restatement-then-reversal", "retype"],
)
def test_corrected_edges_are_the_same_in_full_and_incremental_builds(
    tmp_path: Path, runs: list[dict[str, str]]
) -> None:
    full = _Store(tmp_path / "full", _Model())
    full.ingest(runs[-1])
    incremental = _Store(tmp_path / "incremental", _Model())
    for corpus in runs:
        incremental.ingest(corpus)

    assert incremental.edges() == full.edges()
    # The corrected and the uncorrected statements stay apart, each with
    # its own description.
    descriptions = [edge[-1] for edge in full.edges()]
    assert len(descriptions) == 2
    assert sum("corrected" in d for d in descriptions) == 1


class _Translating(_Model):
    """The extraction model above, and a translator whose every answer
    differs (a model is not deterministic): a translation asked again
    changes the text extraction reads, so its record misses."""

    def __init__(self) -> None:
        super().__init__()
        self.asked = 0

    def __call__(self, system: str, human: str) -> str:
        if "professional translator" in system:
            self.calls["translate"] += 1
            self.asked += 1
            text = human.split("<text>\n", 1)[1].split("\n</text>", 1)[0]
            return f"{text} (v{self.asked})"
        return super().__call__(system, human)


def _translating(tmp_path: Path) -> tuple[_Translating, _Store]:
    model = _Translating()
    store = _Store(tmp_path, model)
    translation = store.config.processing.translation
    translation.enabled = True
    translation.source_language = LanguageCode.KO
    translation.target_language = LanguageCode.EN
    return model, store


TRANSLATED = {
    "a.txt": "Acme supplies goods to Buyer here.",
    "b.txt": "Depot trades with Buyer daily.",
}


@pytest.mark.usefixtures("_no_aws")
def test_a_reset_reuses_the_stored_translations_and_their_extractions(
    tmp_path: Path,
) -> None:
    model, store = _translating(tmp_path)
    store.ingest(TRANSLATED)
    assert model.calls["translate"] == 2
    keys = store.record_keys()
    model.calls.clear()
    store.config.indexing.reset = True
    store.ingest(TRANSLATED)
    # Neither translated nor (the same translation) extracted again, and
    # the records stay named.
    assert model.calls == Counter()
    assert store.record_keys() == keys
    store.ingest(TRANSLATED, force_rebuild=True)
    assert model.calls["translate"] == 2


@pytest.mark.usefixtures("_no_aws")
def test_an_edited_document_translates_only_its_changed_text(tmp_path: Path) -> None:
    model, store = _translating(tmp_path)
    store.ingest(TRANSLATED)
    model.calls.clear()
    store.config.indexing.reset = True
    store.ingest({**TRANSLATED, "b.txt": "Depot trades with Buyer weekly."})
    assert model.calls["translate"] == 1
    assert model.calls["extract"] == 1
