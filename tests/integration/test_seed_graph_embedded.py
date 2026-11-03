# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed graphs imported as virtual documents, end to end on the embedded stores.

``backend: local`` on the SQLite graph, vector and registry files (no
container, no AWS), with the fake Ollama server of ``test_embedded_e2e``. A
JSONL seed beside a small corpus, a seed-only run, an edit that re-indexes
one shard, an LLM settings change that re-imports none, a reset that keeps
the seed, its removal restoring the document-only graph, an interrupted
seed run that resumes and converges, and a seed-only source scope beside a
document scope of the same namespace. The fake model records every prompt:
no seed statement ever reaches it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - fixture
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.ingestion.seed_graph import (
    SEED_RECORD_FINGERPRINT,
    is_seed_document,
    shard_of,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import SeedGraphConfig
from unified_kg_rag.domain.models.pipeline import PIPELINE_METADATA_FILE
from unified_kg_rag.shared.exceptions import ConfirmationDeclinedError

# Every test runs with each AWS entry point patched to fail.
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

CORPUS = {
    "contract.txt": "Vendor supplies widgets to Buyer.",
    "logistics.txt": "Depot stores widgets for Vendor.",
}
SHARDS = 2
# Seed nodes: "Vendor" is also a document entity; the others only the seed
# states. Ids picked so both shards hold nodes.
NODES = [
    {"id": "n-vendor", "name": "Vendor", "type": "ORGANIZATION", "iri": "urn:x:v"},
    {"id": "n-port", "name": "Harbor Port", "type": "LOCATION"},
    {"id": "n-mill", "name": "Ridge Mill", "type": "ORGANIZATION"},
    {"id": "n-yard", "name": "Rail Yard", "type": "LOCATION"},
]
EDGES = [
    {"source": "n-vendor", "target": "n-port", "type": "SHIPS_FROM"},
    {"source": "n-mill", "target": "n-vendor", "type": "SUPPLIES"},
    {"source": "n-yard", "target": "n-port", "type": "CONNECTS"},
]
SEED_NAMES = {node["name"] for node in NODES}


def _shard(node_id: str) -> int:
    return shard_of(node_id, SHARDS)


def test_the_seed_fixture_fills_both_shards() -> None:
    assert {_shard(node["id"]) for node in NODES} == set(range(SHARDS))


def _seed_text(descriptions: dict[str, str] | None = None) -> str:
    descriptions = descriptions or {}
    records = [
        {"kind": "node", **node, "description": descriptions.get(node["id"], "")}
        for node in NODES
    ] + [{"kind": "edge", **edge} for edge in EDGES]
    return "".join(json.dumps(record) + "\n" for record in records)


class _Recording(e2e.FakeOllama):
    """The fake server, keeping every chat prompt it was sent."""

    def __init__(self) -> None:
        super().__init__(self._answer, e2e.ANSWER, e2e.DIM)
        self.prompts: list[str] = []

    def _answer(self, system: str, human: str) -> str:
        self.prompts.append(system + "\n" + human)
        return e2e._respond(system, human)


class _Harness:
    """One namespace on the embedded stores, ingested run after run."""

    def __init__(self, tmp_path: Path, corpus: dict[str, str] = CORPUS) -> None:
        self.tmp_path = tmp_path
        self.source = tmp_path / "corpus"
        self.source.mkdir(parents=True)
        for name, text in corpus.items():
            (self.source / name).write_text(text)
        self.seed = tmp_path / "seed" / "procure.jsonl"
        self.seed.parent.mkdir()
        self.seed.write_text(_seed_text())
        self.config = e2e._config(tmp_path / "stores", tmp_path / "viz")
        # Every LLM stage on: none may see a seed unit.
        self.config.processing.gleaning.enabled = True
        self.config.processing.gleaning.max_rounds = 1
        self.config.processing.claim_extraction.enabled = True
        self.with_seed(True)
        self.server = _Recording()
        self.runs = 0

    def with_seed(self, on: bool) -> None:
        self.config.processing.seed_graphs = (
            [
                SeedGraphConfig(
                    name="procure", kind="jsonl", path=str(self.seed), shards=SHARDS
                )
            ]
            if on
            else []
        )

    @property
    def providers(self) -> Any:
        return e2e._providers(
            self.config, self.server, e2e.FakeCrossEncoder("synthetic-reranker")
        )

    def run(
        self, pipeline_id: str | None = None, confirm: Any = None
    ) -> PipelineContext:
        self.runs += 1
        cache = self.tmp_path / (pipeline_id or f"run-{self.runs}")
        pipeline = DataIngestionPipeline(
            self.config,
            PipelineConfig(
                stages_enabled=dict.fromkeys(PipelineStageType, True),
                local_directory=cache,
            ),
            source_directory=self.source,
            providers=self.providers,
            confirm=confirm,
        )
        for stage in pipeline.stages:
            if isinstance(stage, ps.GraphResolutionStage):
                stage.resolver.entity_resolver.use_process_pool = False
                stage.resolver.relationship_resolver.use_process_pool = False
        try:
            return pipeline.run(self.source, pipeline_id=cache.name)
        finally:
            pipeline.close()

    def ok(
        self, pipeline_id: str | None = None, confirm: Any = None
    ) -> PipelineContext:
        context = self.run(pipeline_id, confirm)
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        return context

    def hits(self, prefix: str) -> list[dict[str, Any]]:
        return [
            hit["_source"] for hit in e2e._stored(self.config, self.providers, prefix)
        ]

    def entities(self) -> dict[str, dict[str, Any]]:
        prefix = self.config.indexing.opensearch.entities_index_prefix
        return {hit["name"]: hit for hit in self.hits(prefix)}

    def graph_names(self) -> set[str]:
        return e2e._entity_names(self.providers)

    def units(self) -> list[dict[str, Any]]:
        return self.hits(self.config.indexing.opensearch.text_units_index_prefix)

    def seed_units(self) -> list[dict[str, Any]]:
        return [u for u in self.units() if u.get("source_kind") == "seed_graph"]

    def registry(self) -> dict[str, DocStatus]:
        return e2e._registry(self.providers)

    def state(self) -> tuple[Any, ...]:
        """What a converged build holds: entities, unit texts, registry."""
        return (
            {
                name: (hit["type"], hit["description"], hit.get("source_kinds"))
                for name, hit in self.entities().items()
            },
            sorted(u["text"] for u in self.units()),
            self.graph_names(),
            self.registry(),
        )

    def assert_no_seed_text_prompted(self) -> None:
        lines = [
            line for unit in self.seed_units() for line in unit["text"].split("\n")
        ]
        assert lines
        for prompt in self.server.prompts:
            assert not any(line in prompt for line in lines)
            assert '"kind":"node"' not in prompt


def _seed_paths() -> dict[str, DocStatus]:
    return {
        f"seed_graphs/procure/{k:04d}-of-{SHARDS:04d}.jsonl": DocStatus.PROCESSED
        for k in range(SHARDS)
    }


def _corpus_paths() -> dict[str, DocStatus]:
    return dict.fromkeys(CORPUS, DocStatus.PROCESSED)


def test_a_seed_beside_a_corpus_lives_through_its_lifecycle(tmp_path: Path) -> None:
    harness = _Harness(tmp_path / "seeded")

    # 1. The seed's shards are documents of the run's scope.
    context = harness.ok()
    loading = next(
        r for r in context.stage_results if r.stage_name == "document_loading"
    )
    assert (
        loading.metrics["seed_documents"],
        loading.metrics["seed_documents_skipped"],
    ) == (SHARDS, 0)
    assert harness.registry() == {**_corpus_paths(), **_seed_paths()}
    assert harness.graph_names() == {"Vendor", "Buyer", "Depot"} | SEED_NAMES
    entities = harness.entities()
    assert entities["Vendor"]["source_kinds"] == ["document", "seed_graph"]
    assert entities["Harbor Port"]["source_kinds"] == ["seed_graph"]
    assert entities["Vendor"].get("iri") == "urn:x:v"
    assert "source_kinds" not in entities["Buyer"]
    # Seed units are stored, tagged, keyed under the seed fingerprint, with
    # no claims record; document units carry no source_kind.
    seed_units = harness.seed_units()
    assert seed_units and all(u["id"].startswith("sg-") for u in seed_units)
    for unit in seed_units:
        keys = unit["attributes"]["extraction_records"]
        assert set(keys) == {"graph"}
        assert keys["graph"].startswith(SEED_RECORD_FINGERPRINT + ":")
    assert {u["text"] for u in harness.units() if "source_kind" not in u} == set(
        CORPUS.values()
    )
    harness.assert_no_seed_text_prompted()

    # 2. An edit of one node re-imports its shard alone.
    harness.seed.write_text(_seed_text({"n-port": "A deep-water port."}))
    context = harness.ok()
    delta = context.incremental_delta
    assert delta is not None
    assert (len(delta.new), len(delta.changed), len(delta.deleted)) == (0, 1, 0)
    (changed,) = context.documents
    assert changed.file_path == (
        f"seed_graphs/procure/{_shard('n-port'):04d}-of-{SHARDS:04d}.jsonl"
    )
    assert "A deep-water port." in harness.entities()["Harbor Port"]["description"]
    prompts = len(harness.server.prompts)

    # 3. An LLM settings change re-extracts the corpus, never the seed.
    harness.config.models.default = "synthetic-llm-2"
    context = harness.ok()
    delta = context.incremental_delta
    assert delta is not None
    assert len(delta.settings_changed) == len(CORPUS)
    assert len(context.documents) == len(CORPUS)
    assert not any(is_seed_document(d) for d in context.documents)
    assert len(harness.server.prompts) > prompts
    assert harness.graph_names() == {"Vendor", "Buyer", "Depot"} | SEED_NAMES

    # 4. A reset rebuilds the namespace with the seed in it.
    harness.config.indexing.reset = True
    harness.ok()
    harness.config.indexing.reset = False
    assert harness.registry() == {**_corpus_paths(), **_seed_paths()}
    assert harness.graph_names() == {"Vendor", "Buyer", "Depot"} | SEED_NAMES
    harness.assert_no_seed_text_prompted()

    # 5. Without the seed, the graph is the document-only build's.
    harness.with_seed(False)
    context = harness.ok()
    assert context.incremental_delta is not None
    assert len(context.incremental_delta.deleted) == SHARDS
    plain = _Harness(tmp_path / "plain")
    plain.with_seed(False)
    plain.config.models.default = "synthetic-llm-2"
    plain.ok()
    assert harness.state() == plain.state()
    assert harness.seed_units() == []


SEED_VENDOR_LINE = "Vendor of record in the procurement master data."


def _document_first(description: str) -> None:
    """``description`` holds the document lines and then the seed's one."""
    lines = description.split("\n")
    assert len(lines) > 1 and lines[-1] == SEED_VENDOR_LINE, lines


@pytest.mark.parametrize("seed_first", [False, True])
def test_a_merged_description_puts_the_document_lines_first(
    tmp_path: Path, seed_first: bool
) -> None:
    # Records merge in unit-id order: the chunker's ids of the document
    # units must sort before the seed's "sg-" ids, whichever run came first.
    harness = _Harness(tmp_path, corpus={} if seed_first else CORPUS)
    harness.seed.write_text(_seed_text({"n-vendor": SEED_VENDOR_LINE}))
    context = harness.ok()
    if seed_first:
        for name, text in CORPUS.items():
            (harness.source / name).write_text(text)
        context = harness.ok()
    units = [u.id for u in context.text_units or []]
    documents = [u for u in units if not u.startswith("sg-")]
    seeds = [u for u in units if u.startswith("sg-")]
    assert documents and max(documents) < min(seeds or ["sg-"])
    vendor = harness.entities()["Vendor"]
    _document_first(vendor["description"])
    graph = harness.providers.build_graph_indexer()
    try:
        (vertex,) = graph.read_entities([vendor["id"]])
    finally:
        graph.close()
    _document_first(vertex.description or "")


def test_a_seed_only_run_makes_no_model_call(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, corpus={})
    harness.ok()
    assert harness.graph_names() == SEED_NAMES
    assert harness.registry() == _seed_paths()
    # Seed-only communities get no report; nothing is extracted.
    assert harness.server.chat_models == []
    assert {name: hit["source_kinds"] for name, hit in harness.entities().items()} == (
        dict.fromkeys(SEED_NAMES, ["seed_graph"])
    )
    # An unchanged seed: nothing to do.
    context = harness.ok()
    assert context.incremental_delta is not None
    assert context.incremental_delta.is_empty
    assert harness.server.chat_models == []


class _Died(BaseException):
    """A crash the pipeline does not catch (the task was killed)."""


@pytest.mark.parametrize("edited_meanwhile", [False, True])
def test_an_interrupted_seed_run_resumes_and_converges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edited_meanwhile: bool
) -> None:
    clean = _Harness(tmp_path / "clean")
    harness = _Harness(tmp_path / "killed")
    edit = {"n-mill": "A saw mill."}
    if edited_meanwhile:
        clean.seed.write_text(_seed_text(edit))
    clean.ok()

    execute = ps.IndexingStage._execute_core
    crash = {"armed": True}

    def killed_once(self: Any, context: PipelineContext) -> Any:
        if crash.pop("armed", False):
            raise _Died("killed while indexing")
        return execute(self, context)

    monkeypatch.setattr(ps.IndexingStage, "_execute_core", killed_once)
    with pytest.raises(_Died):
        harness.run("resumed")
    if edited_meanwhile:
        # The stage cache keys hold the seed's digest: the resume recomputes
        # the stages from the edited seed rather than index the cached ones.
        harness.seed.write_text(_seed_text(edit))
    context = harness.ok("resumed")
    executed = {r.stage_name for r in context.stage_results}
    assert "indexing" in executed
    assert harness.state() == clean.state()
    if edited_meanwhile:
        assert "A saw mill." in harness.entities()["Ridge Mill"]["description"]


def test_a_reset_asks_before_it_drops_another_scopes_seed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    seeded = _Harness(tmp_path, corpus={})
    seeded.config.processing.document_parsing.source_scope = "seeds-procure"
    seeded.ok()
    other: Config = seeded.config.model_copy(deep=True)
    other.processing.document_parsing.source_scope = "documents"
    other.processing.seed_graphs = []
    other.indexing.reset = True
    (seeded.source / "contract.txt").write_text(CORPUS["contract.txt"])
    seeded.config = other
    question = "drops the seed graph documents of source scope(s) seeds-procure"
    # No one to ask (confirm=None), or a no: the reset stops before it writes.
    for confirm in (None, lambda q: False):
        with pytest.raises(ConfirmationDeclinedError, match=re.escape(question)):
            seeded.run(confirm=confirm)
        assert set(seeded.registry()) == set(_seed_paths())
        # Recorded like a failed stage, for a resume.
        (metadata,) = (seeded.tmp_path / f"run-{seeded.runs}").rglob(
            PIPELINE_METADATA_FILE
        )
        (stopped,) = [
            r
            for r in json.loads(metadata.read_text())["stage_results"]
            if r["status"] == PipelineStageStatus.FAILED.value
        ]
        assert question in stopped["error_message"]
    # A cancellation, not a failure: nothing logged an error.
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
    # A yes (run-ingestion --yes): the namespace is the documents' alone.
    asked: list[str] = []
    seeded.ok(confirm=lambda q: asked.append(q) or True)
    assert len(asked) == 1 and question in asked[0]
    assert seeded.registry() == {"contract.txt": DocStatus.PROCESSED}
    assert seeded.graph_names() == {"Vendor", "Buyer"}


def test_a_seed_only_scope_lives_beside_a_document_scope(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, corpus={})
    seeds_source = harness.source
    seeds_config = harness.config
    seeds_config.processing.document_parsing.source_scope = "seeds-procure"
    documents_source = tmp_path / "documents"
    documents_source.mkdir()
    for name, text in CORPUS.items():
        (documents_source / name).write_text(text)
    documents_config: Config = seeds_config.model_copy(deep=True)
    documents_config.processing.document_parsing.source_scope = "documents"
    documents_config.processing.seed_graphs = []

    def use(config: Config, source: Path) -> None:
        harness.config, harness.source = config, source

    harness.ok()
    # The document scope's incremental run diffs its own records only: the
    # seed scope's shards stay, and Vendor is one entity of both.
    use(documents_config, documents_source)
    harness.ok()
    assert harness.registry() == {**_corpus_paths(), **_seed_paths()}
    assert harness.graph_names() == {"Vendor", "Buyer", "Depot"} | SEED_NAMES
    entities = harness.entities()
    assert entities["Vendor"]["source_kinds"] == ["document", "seed_graph"]
    assert entities["Vendor"].get("iri") == "urn:x:v"
    assert "source_kinds" not in entities["Depot"]
    both = harness.state()
    # The seed scope again, unchanged: nothing to do, nothing dropped.
    use(seeds_config, seeds_source)
    context = harness.ok()
    assert context.incremental_delta is not None
    assert context.incremental_delta.is_empty
    assert harness.state() == both

    # A reset of the document scope drops the seed scope's shards: it asks
    # first, and a no leaves both scopes as they were.
    use(documents_config, documents_source)
    documents_config.indexing.reset = True
    asked: list[str] = []
    with pytest.raises(ConfirmationDeclinedError):
        harness.run(confirm=lambda q: asked.append(q) or False)
    assert len(asked) == 1 and "seeds-procure" in asked[0]
    assert harness.state() == both
    # A reset of the seed scope rebuilds the namespace from the seed alone,
    # dropping the document scope: it asks too.
    use(seeds_config, seeds_source)
    seeds_config.indexing.reset = True
    asked.clear()
    harness.ok(confirm=lambda q: asked.append(q) or True)
    assert len(asked) == 1
    assert "dropping the documents of source scope(s) documents" in asked[0]
    assert harness.registry() == _seed_paths()
    assert harness.graph_names() == SEED_NAMES
    assert {name: hit["source_kinds"] for name, hit in harness.entities().items()} == (
        dict.fromkeys(SEED_NAMES, ["seed_graph"])
    )


def test_an_empty_corpus_never_deletes_the_scopes_documents(tmp_path: Path) -> None:
    harness = _Harness(tmp_path / "corpus-run")
    harness.ok()
    indexed = harness.state()
    # The corpus is not in place (a volume not mounted yet, an S3 sync that
    # copied nothing): the seeds alone would delete every document.
    hold = tmp_path / "hold"
    harness.source.rename(hold)
    harness.source.mkdir()
    context = harness.run()
    assert context.status is PipelineStageStatus.FAILED
    error = e2e._errors(context)
    assert "holds 2 documents of source scope" in error
    assert "--retire-source-scope" in error
    assert harness.state() == indexed
    # A reset of the scope from the empty corpus is refused alike.
    harness.config.indexing.reset = True
    context = harness.run(confirm=lambda q: True)
    assert context.status is PipelineStageStatus.FAILED
    assert harness.state() == indexed
    harness.config.indexing.reset = False
    # Back in place: nothing to do.
    harness.source.rmdir()
    hold.rename(harness.source)
    context = harness.ok()
    assert context.incremental_delta is not None
    assert context.incremental_delta.is_empty
    assert harness.state() == indexed


def test_a_seed_added_beside_documents_reports_their_community(
    tmp_path: Path,
) -> None:
    def detection(context: PipelineContext) -> dict[str, Any]:
        return next(
            r for r in context.stage_results if r.stage_name == "community_detection"
        ).metrics

    harness = _Harness(tmp_path / "added")
    harness.with_seed(False)
    harness.ok()
    # The incremental run's graph holds the seed's shards alone, whose
    # units state Vendor; the stored Vendor cites the documents' units.
    harness.with_seed(True)
    context = harness.ok()
    assert context.documents and all(is_seed_document(d) for d in context.documents)
    metrics = detection(context)
    assert metrics["communities_seed_only"] == 0
    assert metrics["reports_generated"] == metrics["communities_detected"] > 0
    # As a full build of documents and seed reports it.
    built = _Harness(tmp_path / "built")
    full = detection(built.ok())
    assert (full["reports_generated"], full["communities_seed_only"]) == (
        metrics["reports_generated"],
        0,
    )


SEED_ONLY = SEED_NAMES - {"Vendor"}


def _sources_named(out: RAGOutput, names: set[str]) -> list[dict[str, Any]]:
    return [
        source
        for source in out.sources
        if source["metadata"].get("name") in names
        or source["metadata"].get("target_name") in names
        or source["metadata"].get("source_name") in names
    ]


def _assert_no_seed_text(out: RAGOutput, strategy: SearchStrategy) -> None:
    """No seed unit reached a TEXT section, its sources or the answer prompt."""
    for result in out.search_results.results:
        if result.retriever_type == "text":
            assert not str(result.source).startswith("sg-"), strategy
            assert '"kind":' not in result.content, strategy
    for source in out.sources:
        if source["metadata"].get("section_type") == "text":
            assert not str(source["source"]).startswith("sg-"), strategy


def test_seed_facts_are_retrieved_and_cited_but_never_as_text(
    tmp_path: Path,
) -> None:
    harness = _Harness(tmp_path)
    harness.ok()
    assert harness.seed_units()
    chain = GraphRAGChain(harness.config, providers=harness.providers)
    asked = len(harness.server.prompts)
    try:
        # Every strategy: seed units are stored but never a text chunk.
        for strategy in SearchStrategy:
            out = _ask(chain, strategy)
            assert out.answer == e2e.ANSWER, strategy
            _assert_no_seed_text(out, strategy)

        # "Harbor Port" is only a seed edge away from Vendor: local search
        # reaches it through the graph and mix through the relationship
        # index; either cites the seed as its provenance, with no chunk.
        for strategy in (SearchStrategy.LOCAL, SearchStrategy.MIX):
            out = _ask(chain, strategy)
            cited = _sources_named(out, SEED_ONLY)
            assert cited, strategy
            named: set[str] = set()
            for source in cited:
                provenance = source["metadata"]["provenance"]
                assert provenance["kind"] == "seed_graph", strategy
                assert source["metadata"]["chunk_id"] is None, strategy
                # The graph's name where the store keeps the document
                # filters (the vector indices, not graph vertices).
                assert provenance["seed_graphs"] in ([], ["procure"]), strategy
                named.update(provenance["seed_graphs"])
            assert named == {"procure"}, strategy
            (vendor,) = [
                s for s in out.sources if s["metadata"].get("name") == "Vendor"
            ]
            assert vendor["metadata"]["provenance"]["kind"] == "document+seed_graph"
            assert vendor["metadata"]["provenance"]["iri"] == "urn:x:v"
            # A candidate entity names its seed graph under either strategy:
            # local reports its graph copy, which takes the hit's filters.
            assert vendor["metadata"]["provenance"]["seed_graphs"] == ["procure"]
            # A document's source is reported as before seed graphs.
            for source in out.sources:
                if source["metadata"].get("name") in {"Buyer", "Depot"}:
                    assert "provenance" not in source["metadata"]

        # source_kinds: document keeps what documents state only.
        out = chain.invoke(
            RAGInput(
                query="Who stores the gadgets for Vendor?",
                search_strategy=SearchStrategy.LOCAL,
                filters={"source_kinds": "document"},
            )
        )
        assert out.sources
        assert _sources_named(out, SEED_ONLY) == []
        _assert_no_seed_text(out, SearchStrategy.LOCAL)
    finally:
        chain.close()
    # The answer prompts held no seed statement either.
    assert len(harness.server.prompts) > asked
    harness.assert_no_seed_text_prompted()


def _ask(chain: GraphRAGChain, strategy: SearchStrategy) -> RAGOutput:
    out = chain.invoke(
        RAGInput(query="Where does Vendor ship from?", search_strategy=strategy)
    )
    assert isinstance(out, RAGOutput), strategy
    return out
