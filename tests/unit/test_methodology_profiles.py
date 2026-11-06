# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Methodology profiles (docs/design.md#46-methodology-profiles): the
catalogue, the stages it selects, the config checks, the build record's
artifacts, the stage cache keys the default leaves alone and the strategies'
declarations."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import unified_kg_rag.adapters.search_strategies  # noqa: F401 - registers them
from tests.unit.test_pipeline_init_resume import _pipeline_for_init, _StubStage
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    _REGISTRY,
    get_strategy_spec,
    routable_strategies,
    serves,
    strategy_methodology,
)
from unified_kg_rag.domain.ingestion import methodologies as mp
from unified_kg_rag.domain.ingestion.cache_keys import (
    answer_settings,
    extraction_fingerprint,
    stage_input_fingerprint,
)
from unified_kg_rag.domain.models import (
    Config,
    IndexArtifact,
    IndexBuildRecord,
    Methodology,
    PipelineStageType,
    SearchStrategy,
)
from unified_kg_rag.domain.models.methodology import (
    DEFAULT_METHODOLOGIES,
    METHODOLOGY_PROFILES,
    UNBUILT_ARTIFACTS,
    available_methodologies,
    methodologies_holding,
)

pytestmark = pytest.mark.unit

S = PipelineStageType
A = IndexArtifact
G, L, H, R = (
    Methodology.GRAPHRAG,
    Methodology.LIGHTRAG,
    Methodology.HIPPORAG,
    Methodology.RAPTOR,
)


def _config(*methodologies: Methodology, **sections) -> Config:
    return Config(
        indexing={"methodologies": [m.value for m in methodologies]}, **sections
    )


# --- catalogue -----------------------------------------------------------


def test_every_buildable_artifact_has_its_stages() -> None:
    for artifact in A:
        if artifact in UNBUILT_ARTIFACTS:
            assert artifact not in mp.ARTIFACT_STAGES
            continue
        assert mp.ARTIFACT_STAGES[artifact], artifact
        assert mp.ARTIFACT_PRODUCERS[artifact] in mp.ARTIFACT_STAGES[artifact]
    assert set(mp.STAGE_DEPENDENCIES) == set(S)


def test_raptor_builds_its_summary_tree_after_indexing() -> None:
    assert available_methodologies() == (G, L, H, R)
    assert UNBUILT_ARTIFACTS == frozenset()
    assert METHODOLOGY_PROFILES[R].required == {A.TEXT_UNITS, A.SUMMARY_TREE}
    assert mp.ARTIFACT_PRODUCERS[A.SUMMARY_TREE] is S.SUMMARY_TREE
    assert mp.STAGE_DEPENDENCIES[S.SUMMARY_TREE] == {S.INDEXING, S.TRANSLATION}
    # Built from what the stores hold: adding it rebuilds nothing.
    assert A.SUMMARY_TREE not in mp.CORPUS_WIDE_ARTIFACTS


def test_raptor_alone_runs_no_graph_stage() -> None:
    # Translation runs too (a no-op unless processing.translation is on):
    # the tree is written in the index language.
    assert mp.selected_stages(_config(R)) == {
        S.DOCUMENT_PARSING,
        S.DOCUMENT_LOADING,
        S.TEXT_CHUNKING,
        S.TRANSLATION,
        S.INDEXING,
        S.SUMMARY_TREE,
    }
    # Beside the default methodologies it adds the tree stage alone.
    assert mp.selected_stages(_config(G, L, R)) - mp.selected_stages(_config(G, L)) == {
        S.SUMMARY_TREE
    }


def test_lightrag_and_hipporag_need_no_community() -> None:
    for methodology in (L, H):
        required = METHODOLOGY_PROFILES[methodology].required
        assert A.RELATIONSHIP_VECTORS in required
        assert not required & {A.COMMUNITIES, A.COMMUNITY_REPORTS}
    assert {A.COMMUNITIES, A.COMMUNITY_REPORTS} <= METHODOLOGY_PROFILES[G].required


def test_stages_for_closes_over_what_each_stage_reads() -> None:
    assert mp.stages_for([A.COMMUNITY_REPORTS]) == {
        S.DOCUMENT_PARSING,
        S.DOCUMENT_LOADING,
        S.TEXT_CHUNKING,
        S.TRANSLATION,
        S.GRAPH_EXTRACTION,
        S.GLEANING,
        S.GRAPH_RESOLUTION,
        S.GRAPH_ANALYSIS,
        S.COMMUNITY_DETECTION,
        S.INDEXING,
    }
    # Text units alone need no graph stage.
    assert mp.stages_for([A.TEXT_UNITS]) == {
        S.DOCUMENT_PARSING,
        S.DOCUMENT_LOADING,
        S.TEXT_CHUNKING,
        S.INDEXING,
    }
    assert mp.stages_for([A.SUMMARY_TREE]) == {
        S.DOCUMENT_PARSING,
        S.DOCUMENT_LOADING,
        S.TEXT_CHUNKING,
        S.TRANSLATION,
        S.INDEXING,
        S.SUMMARY_TREE,
    }


# --- selection -----------------------------------------------------------


def test_the_default_selects_every_stage() -> None:
    config = Config()
    assert config.indexing.methodologies == list(DEFAULT_METHODOLOGIES)
    # Claims are off by default; their switch alone governs their stages.
    # The summary tree is raptor's, not a default methodology.
    assert set(S) - mp.selected_stages(config) == {
        S.CLAIM_EXTRACTION,
        S.CLAIM_RESOLUTION,
        S.SUMMARY_TREE,
    }
    config.processing.claim_extraction.enabled = True
    assert mp.selected_stages(config) == set(S) - {S.SUMMARY_TREE}


def test_lightrag_alone_runs_no_community_detection() -> None:
    for methodology in (L, H):
        stages = mp.selected_stages(_config(methodology))
        assert S.COMMUNITY_DETECTION not in stages
        assert {S.GRAPH_EXTRACTION, S.GRAPH_RESOLUTION, S.INDEXING} <= stages
    assert S.COMMUNITY_DETECTION in mp.selected_stages(_config(G))


def test_claims_follow_their_switch_whatever_the_methodology() -> None:
    config = _config(L, processing={"claim_extraction": {"enabled": True}})
    assert {S.CLAIM_EXTRACTION, S.CLAIM_RESOLUTION} <= mp.selected_stages(config)
    assert any(
        "no selected methodology reads claims" in m
        for m in config.ineffective_settings()
    )
    assert not Config(
        processing={"claim_extraction": {"enabled": True}}
    ).ineffective_settings()


def test_the_pipeline_skips_what_no_selected_methodology_needs() -> None:
    _StubStage.instances = []
    stages, _ = _pipeline_for_init(_config(L))._initialize_stages()
    names = {s.name for s in stages}
    assert "community_detection" not in names
    assert {"graph_extraction", "graph_resolution", "graph_analysis"} <= names
    _StubStage.instances = []
    stages, _ = _pipeline_for_init(Config())._initialize_stages()
    # The default: every stage but the claims, which are off, and the
    # summary tree (raptor).
    assert {s.name for s in stages} == {
        s.value
        for s in S
        if s not in (S.CLAIM_EXTRACTION, S.CLAIM_RESOLUTION, S.SUMMARY_TREE)
    }


def test_an_explicit_stage_list_wins_and_warns(caplog) -> None:
    pipe = _pipeline_for_init(Config())
    listed = {S.DOCUMENT_PARSING, S.DOCUMENT_LOADING, S.TEXT_CHUNKING, S.INDEXING}
    pipe.pipeline_config.stages_enabled = {s: s in listed for s in S}
    with caplog.at_level(logging.WARNING):
        stages, _ = pipe._initialize_stages()
    assert {s.name for s in stages} == {s.value for s in listed}
    warned = caplog.text
    assert "'community_detection' is not in the run's enabled stages" in warned
    assert "graphrag needs it" in warned
    assert "'graph_extraction'" in warned and "graphrag, lightrag needs it" in warned
    # Stages the artifacts are built without warn of nothing.
    assert "'gleaning'" not in warned and "'translation'" not in warned

    # A phase that writes no index leaves nothing out of it.
    caplog.clear()
    pipe = _pipeline_for_init(Config())
    phase = {S.DOCUMENT_PARSING, S.DOCUMENT_LOADING, S.TEXT_CHUNKING}
    pipe.pipeline_config.stages_enabled = {s: s in phase for s in S}
    with caplog.at_level(logging.WARNING):
        pipe._initialize_stages()
    assert "enabled stages" not in caplog.text

    # A listed stage no selected methodology needs runs, with a warning.
    caplog.clear()
    pipe = _pipeline_for_init(_config(L))
    pipe.pipeline_config.stages_enabled = dict.fromkeys(S, True) | {
        S.DOCUMENT_PARSING: False
    }
    with caplog.at_level(logging.WARNING):
        stages, _ = pipe._initialize_stages()
    assert "community_detection" in {s.name for s in stages}
    assert "no selected methodology" in caplog.text


# --- config ----------------------------------------------------------------


def test_an_unknown_methodology_names_the_valid_ones() -> None:
    with pytest.raises(
        ValidationError,
        match="unknown methodology 'lightrg'; valid: graphrag, lightrag, "
        "hipporag, raptor",
    ):
        _config_raw(["lightrg"])
    assert _config_raw(["raptor"]).indexing.methodologies == [R]
    with pytest.raises(ValidationError):
        _config_raw([])
    assert Config(
        indexing={"methodologies": ["LightRAG", "lightrag", " graphrag"]}
    ).indexing.methodologies == [L, G]


def _config_raw(methodologies: list[str]) -> Config:
    return Config(indexing={"methodologies": methodologies})


def test_switches_leave_out_what_they_turn_off() -> None:
    off = Config(graph={"community_detection": {"enabled": False}})
    assert off.effective_methodologies == (L,)
    # A config from before profiles warns of nothing: it is the profile.
    assert not off.ineffective_settings()
    no_vectors = Config(
        indexing={"opensearch": {"build_relationship_vector_index": False}}
    )
    assert no_vectors.effective_methodologies == (G,)
    explicit = _config(G, L, graph={"community_detection": {"enabled": False}})
    assert any(
        "selects graphrag, but graph.community_detection.enabled: false" in m
        for m in explicit.ineffective_settings()
    )
    with pytest.raises(ValidationError, match="needs relationship_vectors"):
        Config(
            indexing={
                "methodologies": ["lightrag"],
                "opensearch": {"build_relationship_vector_index": False},
            }
        )


# --- stage cache keys and manifests ----------------------------------------


def test_the_default_keys_every_stage_as_before() -> None:
    default = Config()
    explicit = _config(G, L)
    for stage in S:
        assert stage_input_fingerprint(default, stage) == stage_input_fingerprint(
            explicit, stage
        )
    assert extraction_fingerprint(default) == extraction_fingerprint(explicit)
    assert "indexing.methodologies" not in answer_settings(default)
    # Another selection changes no stage that runs, but the manifest.
    light = _config(L)
    for stage in (S.TEXT_CHUNKING, S.GRAPH_EXTRACTION, S.GRAPH_RESOLUTION):
        assert stage_input_fingerprint(light, stage) == stage_input_fingerprint(
            default, stage
        )
    assert answer_settings(light)["indexing.methodologies"] == ["lightrag"]


# --- what a run built, and what a namespace holds ---------------------------


def test_built_artifacts_need_their_producing_stage() -> None:
    base = {S.TEXT_CHUNKING, S.GRAPH_EXTRACTION, S.GRAPH_RESOLUTION}
    config = Config()
    assert mp.built_artifacts(config, base | {S.COMMUNITY_DETECTION}) == {
        A.TEXT_UNITS,
        A.ENTITIES,
        A.RELATIONSHIPS,
        A.RELATIONSHIP_VECTORS,
        A.GRAPH,
        A.COMMUNITIES,
        A.COMMUNITY_REPORTS,
    }
    # Community detection did not complete (an explicit stage list).
    assert A.COMMUNITIES not in mp.built_artifacts(config, base)
    config.graph.community_detection.report_generation.enabled = False
    assert A.COMMUNITY_REPORTS not in mp.built_artifacts(
        config, base | {S.COMMUNITY_DETECTION}
    )
    # Community detection run by an explicit list for lightrag: not recorded.
    assert A.COMMUNITIES not in mp.built_artifacts(
        _config(L), base | {S.COMMUNITY_DETECTION}
    )
    assert methodologies_holding(mp.built_artifacts(_config(L), base)) == (L, H)


def _record(*artifacts: IndexArtifact, mode: str = "full") -> IndexBuildRecord:
    return IndexBuildRecord(
        pipeline_id="p",
        mode=mode,
        built_at=datetime(2026, 11, 1, tzinfo=UTC),
        documents=1,
        extraction_fingerprint="f",
    ).with_artifacts(artifacts)


LIGHT = (A.TEXT_UNITS, A.ENTITIES, A.RELATIONSHIPS, A.RELATIONSHIP_VECTORS, A.GRAPH)
FULL = (*LIGHT, A.COMMUNITIES, A.COMMUNITY_REPORTS)


def test_a_record_names_what_its_artifacts_hold() -> None:
    record = _record(*LIGHT)
    assert record.methodologies == [L, H]
    dump = record.model_dump(mode="json")
    assert mp.held_methodologies(dump, Config()) == (L, H)
    assert mp.held_methodologies(_record(*FULL).model_dump(mode="json"), Config()) == (
        G,
        L,
        H,
    )
    # A record from before profiles, or none: what a run of the config
    # builds holds, hipporag (lightrag's artifacts) included.
    legacy = {k: v for k, v in dump.items() if k not in ("artifacts", "methodologies")}
    assert mp.held_methodologies(legacy, Config()) == (G, L, H)
    assert mp.held_methodologies(None, _config(L)) == (L, H)
    no_vectors = Config(
        indexing={"opensearch": {"build_relationship_vector_index": False}}
    )
    assert mp.held_methodologies(None, no_vectors) == (G,)
    no_communities = Config(graph={"community_detection": {"enabled": False}})
    assert mp.held_methodologies(None, no_communities) == (L, H)
    # An artifact a later version names is skipped.
    assert mp.held_methodologies(
        {"artifacts": [*dump["artifacts"], "x"]}, Config()
    ) == (L, H)


def test_an_incremental_record_keeps_what_both_hold() -> None:
    stored = _record(*FULL).model_dump(mode="json")
    # The run stopped building communities (graphrag removed).
    written = _record(*LIGHT, mode="incremental").over(stored)
    assert written.artifacts == list(LIGHT)
    assert written.methodologies == [L, H]
    # A run that built more than the namespace holds claims none of it.
    written = _record(*FULL, mode="incremental").over(
        _record(*LIGHT).model_dump(mode="json")
    )
    assert written.methodologies == [L, H]
    # A full build records what it built.
    assert _record(*FULL).over(
        _record(*LIGHT).model_dump(mode="json")
    ).methodologies == [G, L, H]
    # Over a record from before profiles: this run's.
    legacy = _record(*FULL).model_dump(mode="json")
    legacy.pop("artifacts")
    assert _record(*LIGHT, mode="incremental").over(legacy).artifacts == list(LIGHT)


def test_adding_a_corpus_wide_artifact_is_missing_from_the_record() -> None:
    light = _record(*LIGHT).model_dump(mode="json")
    assert mp.missing_artifacts(light, Config()) == {
        A.COMMUNITIES,
        A.COMMUNITY_REPORTS,
    }
    assert mp.missing_artifacts(light, _config(L)) == frozenset()
    # graphrag kept the relationship vectors: lightrag adds nothing.
    graph_only = _record(*FULL).model_dump(mode="json")
    assert mp.missing_artifacts(graph_only, Config()) == frozenset()
    no_vectors = _record(*(a for a in FULL if a is not A.RELATIONSHIP_VECTORS))
    assert mp.missing_artifacts(no_vectors.model_dump(mode="json"), Config()) == {
        A.RELATIONSHIP_VECTORS
    }
    # Claims are per-document: an incremental run adds them itself.
    claims = Config(processing={"claim_extraction": {"enabled": True}})
    assert mp.missing_artifacts(graph_only, claims) == frozenset()
    # No record, or one from before profiles: nothing to add.
    assert mp.missing_artifacts(None, Config()) == frozenset()
    assert mp.missing_artifacts({"pipeline_id": "p"}, Config()) == frozenset()


# --- strategies --------------------------------------------------------------


def test_every_strategy_declares_its_methodology() -> None:
    expected = {
        SearchStrategy.LOCAL: G,
        SearchStrategy.GLOBAL: G,
        SearchStrategy.DRIFT: G,
        SearchStrategy.SIMPLE: G,
        SearchStrategy.MIX: L,
        SearchStrategy.HYBRID: L,
        SearchStrategy.NAIVE: L,
    }
    for strategy, methodology in expected.items():
        assert get_strategy_spec(strategy).methodology is methodology, strategy
    # Every registered strategy reads one: its own, or a wrapper's base's.
    for strategy in _REGISTRY:
        found = strategy_methodology(strategy, Config())
        assert isinstance(found, Methodology), strategy
        if get_strategy_spec(strategy).wraps is None:
            assert get_strategy_spec(strategy).methodology is found, strategy


def test_iterative_reads_what_its_base_reads() -> None:
    for base, methodology in (("local", G), ("mix", L)):
        config = Config(search={"iterative_search": {"base": base}})
        assert strategy_methodology(SearchStrategy.ITERATIVE, config) is methodology
    # Refused where its base's methodology is not held.
    config = Config(search={"iterative_search": {"base": "local"}})
    assert not serves(SearchStrategy.ITERATIVE, [L, H], config)
    assert serves(SearchStrategy.ITERATIVE, [G], config)


# --- query time ----------------------------------------------------------------


class _Records:
    """A document retriever whose build record the test sets."""

    def __init__(self, record: dict | None) -> None:
        self.record = record
        self.reads = 0

    async def aread_build_record(self, suffix: str | None) -> dict | None:
        self.reads += 1
        if isinstance(self.record, Exception):
            raise self.record
        return self.record


def _chain(config: Config, records: _Records, instances: dict | None = None):
    import threading

    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain.config = config
    chain._cache_lock = threading.RLock()
    chain._held_methodologies = {}
    chain._get_retriever = lambda role: records  # type: ignore[method-assign]

    def instance(strategy: SearchStrategy):
        if instances is not None and strategy in instances:
            found = instances[strategy]
            if isinstance(found, Exception):
                raise found
            return found
        return get_strategy_spec(strategy).strategy_class.__new__(
            get_strategy_spec(strategy).strategy_class
        )

    chain._get_strategy_instance = instance  # type: ignore[method-assign]
    return chain


async def test_a_query_reads_what_the_namespace_holds_before_refusing() -> None:
    from unified_kg_rag.shared import MethodologyNotIndexedError

    records = _Records(_record(*LIGHT).model_dump(mode="json"))
    chain = _chain(Config(), records)
    assert await chain.acheck_methodology(SearchStrategy.MIX, None) == (L, H)
    with pytest.raises(MethodologyNotIndexedError) as refused:
        await chain.acheck_methodology(SearchStrategy.GLOBAL, "tenant-a")
    message = str(refused.value)
    assert "'global' reads the graphrag methodology" in message
    assert "namespace 'tenant-a' does not hold" in message
    assert (
        "holds lightrag, hipporag; its strategies: simple, mix, hybrid, naive, "
        "iterative, ppr, or auto" in message
    )
    # Cached per suffix; a refusal read the record again first.
    assert records.reads == 3
    await chain.acheck_methodology(SearchStrategy.MIX, None)
    assert records.reads == 3
    # graphrag added meanwhile: found before refusing.
    records.record = _record(*FULL).model_dump(mode="json")
    assert await chain.acheck_methodology(SearchStrategy.GLOBAL, None) == (G, L, H)
    # auto is never refused here: it routes among the held ones.
    records.record = _record(*LIGHT).model_dump(mode="json")
    await chain.acheck_methodology(SearchStrategy.AUTO, "tenant-b")


async def test_without_a_readable_record_the_config_stands_in() -> None:
    from types import SimpleNamespace

    from unified_kg_rag.shared import MethodologyNotIndexedError

    # A store error, or no record: what a run of the config builds.
    for record in (RuntimeError("store down"), None):
        chain = _chain(Config(), _Records(record))  # type: ignore[arg-type]
        assert await chain.acheck_methodology(SearchStrategy.GLOBAL) == (G, L, H)
    # A store from before build records, built by the default run, holds the
    # relationship vectors ppr reads: ppr is served, not refused.
    assert await _chain(Config(), _Records(None)).acheck_methodology(
        SearchStrategy.PPR
    ) == (G, L, H)
    chain = _chain(_config(L), _Records(None))
    with pytest.raises(MethodologyNotIndexedError):
        await chain.acheck_methodology(SearchStrategy.LOCAL)
    # A wrapper's instance declares the wrapped strategy's methodology; one
    # that cannot be built is judged by its class.
    light = _Records(_record(*LIGHT).model_dump(mode="json"))
    wrapped = {SearchStrategy.LOCAL: SimpleNamespace(methodology=L)}
    assert await _chain(Config(), light, wrapped).acheck_methodology(
        SearchStrategy.LOCAL
    )
    broken = {SearchStrategy.GLOBAL: RuntimeError("no store")}
    with pytest.raises(MethodologyNotIndexedError):
        await _chain(Config(), light, broken).acheck_methodology(SearchStrategy.GLOBAL)


async def test_naive_and_simple_are_served_where_their_indexes_are_held() -> None:
    from unified_kg_rag.shared import MethodologyNotIndexedError

    # naive reads the text units alone, simple sweeps the indexes held:
    # neither needs its class's methodology.
    tree_only = _Records(_record(A.TEXT_UNITS, A.SUMMARY_TREE).model_dump(mode="json"))
    light = _Records(_record(*LIGHT).model_dump(mode="json"))
    for records in (tree_only, light):
        for strategy in (SearchStrategy.NAIVE, SearchStrategy.SIMPLE):
            assert await _chain(Config(), records).acheck_methodology(strategy)
    with pytest.raises(MethodologyNotIndexedError, match="'local' reads the graphrag"):
        await _chain(Config(), tree_only).acheck_methodology(SearchStrategy.LOCAL)
    # Without the text units they are refused, naming what is missing.
    nothing = _Records(_record().model_dump(mode="json"))
    with pytest.raises(MethodologyNotIndexedError, match="reads the text_units"):
        await _chain(Config(), nothing).acheck_methodology(SearchStrategy.NAIVE)
    # auto may route to them on such a namespace too.
    config = Config(search={"auto_routable_strategies": ["naive", "local"]})
    assert routable_strategies(config, [R]) == (SearchStrategy.NAIVE,)
    assert serves(SearchStrategy.SIMPLE, [L], Config())


async def test_iterative_is_refused_where_its_base_is_not_held() -> None:
    from unified_kg_rag.shared import MethodologyNotIndexedError

    light = _Records(_record(*LIGHT).model_dump(mode="json"))
    over_local = Config(search={"iterative_search": {"base": "local"}})
    with pytest.raises(
        MethodologyNotIndexedError, match="'iterative' reads the graphrag"
    ):
        await _chain(over_local, light).acheck_methodology(SearchStrategy.ITERATIVE)
    over_mix = Config(search={"iterative_search": {"base": "mix"}})
    assert await _chain(over_mix, light).acheck_methodology(SearchStrategy.ITERATIVE)


def test_the_auto_fallback_is_warned_about_once(caplog) -> None:
    import logging

    from unified_kg_rag.adapters.retrieval import strategy_registry

    strategy_registry._WARNED_FALLBACKS.clear()
    config = Config(search={"auto_routable_strategies": ["global", "drift"]})
    with caplog.at_level(logging.DEBUG, logger=strategy_registry.logger.name):
        for _ in range(3):
            assert routable_strategies(config, [R]) == (SearchStrategy.TREE,)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "auto uses tree" in warnings[0].getMessage()
