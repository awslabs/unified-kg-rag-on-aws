# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Methodology profiles (docs/design.md#46-methodology-profiles): the
catalogue, the stages it selects, the config checks, the build record's
artifacts, the stage cache keys the default leaves alone and the strategies'
declarations."""

from __future__ import annotations

import itertools
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
from unified_kg_rag.domain.models.config import merge_over
from unified_kg_rag.domain.models.methodology import (
    ARTIFACT_SWITCHES,
    DEFAULT_METHODOLOGIES,
    METHODOLOGY_PROFILES,
    artifact_switches,
    methodologies_holding,
)
from unified_kg_rag.domain.retrieval import index_prefixes
from unified_kg_rag.ports.indexer import STORED_ITEM_KINDS

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


def _held(record, config: Config) -> tuple[Methodology, ...]:
    """The methodologies a namespace with ``record`` holds."""
    return methodologies_holding(mp.held_artifacts(record, config))


# --- catalogue -----------------------------------------------------------


def test_every_buildable_artifact_has_its_stages() -> None:
    for artifact in A:
        assert mp.ARTIFACT_STAGES[artifact], artifact
        assert mp.ARTIFACT_PRODUCERS[artifact] in mp.ARTIFACT_STAGES[artifact]
    assert set(mp.STAGE_DEPENDENCIES) == set(S)


def test_raptor_is_held_once_its_tree_stage_has_run() -> None:
    # A run that stopped after indexing built the text units, not the tree:
    # the namespace does not serve raptor until the tree stage completes.
    config = _config(R)
    through_indexing = mp.selected_stages(config) - {S.SUMMARY_TREE}
    assert S.INDEXING in through_indexing
    built = mp.built_artifacts(config, through_indexing)
    assert R not in _held(_record(*built).model_dump(mode="json"), config)
    built = mp.built_artifacts(config, mp.selected_stages(config))
    assert R in _held(_record(*built).model_dump(mode="json"), config)


def test_adding_raptor_to_a_built_namespace_rebuilds_nothing() -> None:
    # The tree is built from what the stores hold, so an incremental run that
    # adds raptor reads no document again.
    full = _record(*FULL).model_dump(mode="json")
    assert mp.missing_artifacts(full, _config(G, L, R)) == frozenset()


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
    assert A.COMMUNITIES in METHODOLOGY_PROFILES[G].required
    # Local reads graphrag without the reports; global and drift need them.
    assert A.COMMUNITY_REPORTS in METHODOLOGY_PROFILES[G].optional
    for strategy in (SearchStrategy.GLOBAL, SearchStrategy.DRIFT):
        assert get_strategy_spec(strategy).required_artifacts == (
            METHODOLOGY_PROFILES[G].required | {A.COMMUNITY_REPORTS}
        )


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


def test_a_listed_stage_no_selected_methodology_needs_is_refused() -> None:
    # It would pay for what no query reads (community reports for a
    # lightrag-only namespace, a tree the build record does not list):
    # run-ingestion refuses the stages it names before any model call.
    from unified_kg_rag.application.ingestion.pipeline import unneeded_stage_error

    refused = unneeded_stage_error
    message = refused(_config(L), [S.GRAPH_EXTRACTION, S.COMMUNITY_DETECTION])
    assert message is not None
    assert "Stage 'community_detection'" in message
    assert "indexing.methodologies: lightrag" in message
    assert "add graphrag to indexing.methodologies" in message
    assert "--methodologies lightrag,graphrag" in message

    message = refused(Config(), [S.SUMMARY_TREE])
    assert message is not None and "Stage 'summary_tree'" in message
    assert "--methodologies graphrag,lightrag,raptor" in message
    # With raptor selected the same list runs; so does a stage the config
    # switches off (the run skips it) and every stage the selection needs.
    assert refused(_config(G, L, R), [S.SUMMARY_TREE]) is None
    assert refused(Config(), [S.CLAIM_EXTRACTION, S.CLAIM_RESOLUTION]) is None
    assert refused(Config(), mp.selected_stages(Config())) is None


def test_a_library_stage_list_with_an_unneeded_stage_runs_it_with_a_warning(
    caplog,
) -> None:
    # A PipelineConfig that switches one stage off leaves every other stage
    # listed: the pipeline runs the listed ones and warns.
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


def _setting(key: str, value: bool) -> dict:
    """The config sections that set the dotted ``key`` to ``value``."""
    first, *rest = key.split(".")
    nested: object = value
    for part in reversed(rest):
        nested = {part: nested}
    return {first: nested}


@pytest.mark.parametrize("switch", ARTIFACT_SWITCHES, ids=lambda s: s.key)
def test_every_consumer_reads_the_same_switch_table(switch) -> None:
    """A switch turns the same artifacts off for what a run builds, the
    default index sweep, the effective methodologies and the messages."""
    on, off = Config(**_setting(switch.key, True)), Config(
        **_setting(switch.key, False)
    )
    assert on._switch_on(switch.key) and not off._switch_on(switch.key)
    assert switch.artifacts <= mp.artifacts_to_build(on)
    assert not switch.artifacts & mp.artifacts_to_build(off)
    assert not switch.artifacts & on.artifacts_switched_off
    assert switch.artifacts <= off.artifacts_switched_off
    # The default sweep searches an index of the artifact only when it is on,
    # and a run creates one only then (rebuild or incremental, in a fresh
    # namespace).
    for prefix, artifact in index_prefixes._prefix_artifacts(on):
        if artifact in switch.artifacts:
            assert prefix in index_prefixes.configured_index_prefixes(on)
            assert prefix not in index_prefixes.configured_index_prefixes(off)
            for rebuild in (True, False):
                assert prefix in _created(on, rebuild=rebuild)
                assert prefix not in _created(off, rebuild=rebuild)
    # The pipeline skips the stages only the switched-off artifacts need,
    # even when an explicit stage list names them.
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline

    def skipped(config: Config) -> set[PipelineStageType]:
        return {
            stage
            for stage, allows in DataIngestionPipeline.CONFIG_ENABLED.items()
            if stage is not S.GLEANING and not allows(config)
        }

    assert skipped(off) - skipped(on) == mp.selected_stages(on) - (
        mp.selected_stages(off)
    )
    # A switch leaves out the methodologies that require them.
    needing = {
        m
        for m in DEFAULT_METHODOLOGIES
        if METHODOLOGY_PROFILES[m].required & switch.artifacts
    }
    expected = set(DEFAULT_METHODOLOGIES) - needing
    assert set(off.effective_methodologies) == expected
    assert set(on.effective_methodologies) == set(DEFAULT_METHODOLOGIES)
    if needing:
        with pytest.raises(ValidationError, match=f"{switch.key}: false"):
            Config(
                **merge_over(
                    _setting(switch.key, False),
                    {"indexing": {"methodologies": [m.value for m in needing]}},
                )
            )


def _created(
    config: Config, *, rebuild: bool, live: frozenset[str] = frozenset()
) -> set[str]:
    """The index prefixes a run of ``config`` creates in a namespace whose
    ``live`` prefixes have an index."""
    from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer

    indexer = OpenSearchIndexer.__new__(OpenSearchIndexer)
    indexer.config = config
    indexer.opensearch_config = config.indexing.opensearch
    indexer.analyzer = "standard"
    indexer.language = "en"
    indexer._embedding_dimension = 8
    indexer._index_limits = {}
    indexer._get_name = lambda prefix, suffix: prefix  # type: ignore[method-assign]
    indexer._live_index = lambda alias: alias if alias in live else None  # type: ignore[method-assign]
    return {alias for alias, _ in indexer._mappings_to_create(None, rebuild)}


def test_a_run_creates_only_the_indices_of_what_it_builds() -> None:
    o = Config().indexing.opensearch
    graph_indices = {
        o.entities_index_prefix,
        o.relationships_index_prefix,
        o.community_reports_index_prefix,
        o.claims_index_prefix,
    }
    raptor = _config(R)
    for rebuild in (True, False):
        assert _created(raptor, rebuild=rebuild) == {
            o.text_units_index_prefix,
            o.summary_nodes_index_prefix,
            o.extraction_records_index_prefix,
        }
    # A rebuild replaces the live index of a type it does not build with an
    # empty one (a methodology removed); an incremental run leaves it.
    assert _created(raptor, rebuild=True, live=frozenset(graph_indices)) >= (
        graph_indices
    )
    assert not _created(raptor, rebuild=False, live=frozenset(graph_indices)) & (
        graph_indices
    )
    # Not the summary tree's: its stage writes (and a reset deletes) it.
    assert o.summary_nodes_index_prefix not in _created(
        Config(), rebuild=True, live=frozenset({o.summary_nodes_index_prefix})
    )
    assert _created(_config(L), rebuild=True) == {
        o.text_units_index_prefix,
        o.entities_index_prefix,
        o.relationships_index_prefix,
        o.extraction_records_index_prefix,
    }


@pytest.mark.parametrize(
    "selected",
    [(G, L), (G,), (L,), (H,), (R,), (L, R)],
    ids=lambda m: "+".join(x.value for x in m),
)
def test_the_default_sweep_reads_only_what_a_run_builds(selected) -> None:
    """The default index sweep (and so a filter key check) skips an index
    no effective methodology reads (the entities, relationships and
    reports under raptor alone), not only one a switch turns off."""
    config = _config(*selected)
    built = mp.artifacts_to_build(config)
    expected = [
        prefix
        for prefix, artifact in index_prefixes._prefix_artifacts(config)
        if artifact in built
    ]
    assert index_prefixes.configured_index_prefixes(config) == expected
    if selected == (R,):
        assert expected == [config.indexing.opensearch.text_units_index_prefix]


# The strategies' reads of an optional index: (strategy, artifact, the
# config field of its index prefix).
_OPTIONAL_READS = [
    (SearchStrategy.LOCAL, A.CLAIMS, "claims_index_prefix"),
    (SearchStrategy.LOCAL, A.RELATIONSHIP_VECTORS, "relationships_index_prefix"),
    (SearchStrategy.LOCAL, A.COMMUNITY_REPORTS, "community_reports_index_prefix"),
    (SearchStrategy.PPR, A.RELATIONSHIP_VECTORS, "relationships_index_prefix"),
]


async def _searched_prefixes(
    config: Config,
    strategy: SearchStrategy,
    held: frozenset[IndexArtifact],
    filters: dict | None = None,
) -> set[str]:
    from tests.fixtures.fakes.retrievers import FakeRetriever
    from unified_kg_rag.domain.models import RetrieverRole, SearchQuery

    document, graph = FakeRetriever("document"), FakeRetriever("graph")
    instance = get_strategy_spec(strategy).strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: document,
            RetrieverRole.GRAPH.value: graph,
        },
    )
    instance.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None, **_kw: [
            r for results in results_dict.values() for r in results
        ]
    )
    await instance.asearch(
        SearchQuery(
            query="who founded acme",
            entity_focus=["Acme"],
            filters=filters,
            metadata={index_prefixes.HELD_ARTIFACTS_KEY: sorted(a.value for a in held)},
        )
    )
    return {prefix for call in document.calls for prefix in (call.index_prefixes or [])}


@pytest.mark.parametrize(
    ("strategy", "artifact", "prefix_field"),
    _OPTIONAL_READS,
    ids=lambda v: getattr(v, "value", v),
)
async def test_a_strategy_reads_an_optional_index_only_where_built_and_held(
    strategy: SearchStrategy, artifact: IndexArtifact, prefix_field: str
) -> None:
    """Each switch of ``ARTIFACT_SWITCHES`` that turns the artifact off
    stops the strategy reading its index, as it does the default sweep,
    and so does a namespace that does not hold it (its build record)."""
    switches = [s for s in ARTIFACT_SWITCHES if artifact in s.artifacts]
    assert switches
    every = frozenset(A)
    for switch in switches:
        on = Config(
            **merge_over(
                _setting(switch.key, True),
                _setting("processing.claim_extraction.enabled", True),
            )
        )
        off = Config(
            **merge_over(
                _setting("processing.claim_extraction.enabled", True),
                _setting(switch.key, False),
            )
        )
        prefix = getattr(on.indexing.opensearch, prefix_field)
        # Unfiltered, and under a document filter (the graph walk then
        # passes over the relationships index, where it is read).
        for filters in (None, {"attr_category": "x"}):
            label = f"{switch.key} {filters}"
            assert prefix in await _searched_prefixes(
                on, strategy, every, filters
            ), label
            assert prefix not in await _searched_prefixes(
                on, strategy, every - {artifact}, filters
            ), label
            assert prefix not in await _searched_prefixes(
                off, strategy, every, filters
            ), label


async def test_ppr_seeds_entities_where_the_namespace_holds_no_facts() -> None:
    """Without the relationship vectors the namespace holds, PPR seeds the
    PageRank from entity hits instead of facts, as it does when the
    config builds none."""
    config = Config()
    entities = config.indexing.opensearch.entities_index_prefix
    every = frozenset(A)
    assert entities not in await _searched_prefixes(config, SearchStrategy.PPR, every)
    assert entities in await _searched_prefixes(
        config, SearchStrategy.PPR, every - {A.RELATIONSHIP_VECTORS}
    )


@pytest.mark.parametrize(
    ("setting", "gone"),
    [
        (
            {"graph": {"community_detection": {"enabled": False}}},
            {A.COMMUNITIES, A.COMMUNITY_REPORTS},
        ),
        (
            {
                "graph": {
                    "community_detection": {"report_generation": {"enabled": False}}
                }
            },
            {A.COMMUNITY_REPORTS},
        ),
        (
            {"indexing": {"opensearch": {"build_relationship_vector_index": False}}},
            {A.RELATIONSHIP_VECTORS},
        ),
        ({"processing": {"claim_extraction": {"enabled": False}}}, {A.CLAIMS}),
    ],
    ids=["communities", "reports", "relationship-vectors", "claims"],
)
def test_a_switch_set_false_builds_none_of_its_artifacts(
    setting: dict, gone: set
) -> None:
    # As a user sets it: the run builds every artifact but the switch's, and
    # a message about those artifacts names the setting.
    claims_on = {"processing": {"claim_extraction": {"enabled": True}}}
    on = mp.artifacts_to_build(Config.model_validate(claims_on))
    off = mp.artifacts_to_build(Config.model_validate(merge_over(claims_on, setting)))
    assert on - off == gone
    assert off <= on
    assert _dotted(setting) in artifact_switches(gone)


def _dotted(setting: dict) -> str:
    """The dotted key of a one-key nested setting."""
    ((name, value),) = setting.items()
    return f"{name}.{_dotted(value)}" if isinstance(value, dict) else name


def test_every_switch_is_a_boolean_setting() -> None:
    config = Config()
    for switch in ARTIFACT_SWITCHES:
        assert isinstance(config._switch_on(switch.key), bool), switch.key
        section, _, field = switch.key.rpartition(".")
        owner = config
        for part in section.split("."):
            owner = getattr(owner, part)
        assert type(owner).model_fields[field].annotation is bool, switch.key


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
TREE_ONLY = (A.TEXT_UNITS, A.SUMMARY_TREE)


def test_a_record_names_what_its_artifacts_hold() -> None:
    record = _record(*LIGHT)
    assert record.methodologies == [L, H]
    dump = record.model_dump(mode="json")
    assert _held(dump, Config()) == (L, H)
    assert _held(_record(*FULL).model_dump(mode="json"), Config()) == (
        G,
        L,
        H,
    )
    # A record from before profiles, or none: what a run of the config
    # builds holds, hipporag (lightrag's artifacts) included.
    legacy = {k: v for k, v in dump.items() if k not in ("artifacts", "methodologies")}
    assert _held(legacy, Config()) == (G, L, H)
    assert _held(None, _config(L)) == (L, H)
    no_vectors = Config(
        indexing={"opensearch": {"build_relationship_vector_index": False}}
    )
    assert _held(None, no_vectors) == (G,)
    no_communities = Config(graph={"community_detection": {"enabled": False}})
    assert _held(None, no_communities) == (L, H)
    # An artifact a later version names is skipped.
    assert _held({"artifacts": [*dump["artifacts"], "x"]}, Config()) == (L, H)


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
    # Claims too: an incremental run would extract them for its delta alone,
    # and its record, which keeps what both name, would never name them.
    claims = Config(processing={"claim_extraction": {"enabled": True}})
    assert mp.missing_artifacts(graph_only, claims) == {A.CLAIMS}
    with_claims = _record(*FULL, A.CLAIMS).model_dump(mode="json")
    assert mp.missing_artifacts(with_claims, claims) == frozenset()
    # Report generation off: a run builds no reports, so none are missing.
    no_reports = Config(
        graph={"community_detection": {"report_generation": {"enabled": False}}}
    )
    built = _record(*(a for a in FULL if a is not A.COMMUNITY_REPORTS))
    assert mp.missing_artifacts(built.model_dump(mode="json"), no_reports) == (
        frozenset()
    )
    assert A.COMMUNITIES in mp.artifacts_to_build(no_reports)
    # No record, or one from before profiles: nothing to add.
    assert mp.missing_artifacts(None, Config()) == frozenset()
    assert mp.missing_artifacts({"pipeline_id": "p"}, Config()) == frozenset()


def _profile_configs() -> list[Config]:
    """A config per selection of methodologies and artifact switch values
    (those with no effective methodology left out)."""
    configs = []
    for count in range(1, len(Methodology) + 1):
        for selected in itertools.combinations(list(Methodology), count):
            for values in itertools.product([True, False], repeat=4):
                config = _config(*selected)
                (
                    config.graph.community_detection.enabled,
                    config.graph.community_detection.report_generation.enabled,
                    config.indexing.opensearch.build_relationship_vector_index,
                    config.processing.claim_extraction.enabled,
                ) = values
                if config.effective_methodologies:
                    configs.append(config)
    return configs


def test_every_artifact_an_incremental_run_adds_rebuilds_the_namespace() -> None:
    # CORPUS_WIDE_ARTIFACTS is kept by hand: an incremental run builds what
    # it adds for its delta alone, and its record keeps only what the stored
    # one names, so a run that adds an artifact outside the set (other than
    # the summary tree, whose stage reads every stored text unit) without
    # adding one inside it would never build it for the stored documents.
    # Each pair of configs (one per set of artifacts built): what the second
    # adds to what the first built.
    configs = {mp.artifacts_to_build(c): c for c in _profile_configs()}
    silent = []
    for built, after in itertools.product(configs, configs.values()):
        record = _record(*built).model_dump(mode="json")
        added = mp.artifacts_to_build(after) - built - {A.SUMMARY_TREE}
        if added and not mp.missing_artifacts(record, after):
            silent.append((sorted(built), sorted(added)))
    assert len(configs) > 10
    assert silent == []


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
    assert not serves(SearchStrategy.ITERATIVE, LIGHT, config)
    assert serves(SearchStrategy.ITERATIVE, METHODOLOGY_PROFILES[G].required, config)


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


def _chain(config: Config, records: _Records):
    import threading

    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain.config = config
    chain._cache_lock = threading.RLock()
    chain._held_artifacts = {}
    # The namespace check passes: the records hold no arequire_namespace.
    chain._present_namespaces = set()
    chain._get_retriever = lambda role: records  # type: ignore[method-assign]

    # The check reads the registry: it never builds a strategy.
    chain.built = []  # type: ignore[attr-defined]
    chain._get_strategy_instance = chain.built.append  # type: ignore[attr-defined]
    return chain


async def test_without_reports_local_is_served_and_global_and_drift_refused() -> None:
    from unified_kg_rag.shared import MethodologyNotIndexedError

    # A namespace built with report generation off holds graphrag: local
    # answers as it did before profiles.
    no_reports = (*LIGHT, A.COMMUNITIES)
    record = _record(*no_reports).model_dump(mode="json")
    off = Config(
        graph={"community_detection": {"report_generation": {"enabled": False}}}
    )
    assert G in _held(record, off) and G in off.effective_methodologies
    chain = _chain(off, _Records(record))
    assert G in await chain.acheck_methodology(SearchStrategy.LOCAL)
    # Global and drift, which answer from the reports, are refused naming
    # the switch that builds them, not graphrag (which the namespace holds).
    for strategy in (SearchStrategy.GLOBAL, SearchStrategy.DRIFT):
        with pytest.raises(MethodologyNotIndexedError) as refused:
            await chain.acheck_methodology(strategy)
        message = str(refused.value)
        assert f"'{strategy.value}' reads the community_reports" in message
        assert "graph.community_detection.report_generation.enabled: true" in message
        assert "graphrag methodology" not in message
    # On a namespace without graphrag they are refused as graphrag's.
    with pytest.raises(MethodologyNotIndexedError, match="reads the graphrag"):
        await _chain(
            Config(), _Records(_record(*LIGHT).model_dump(mode="json"))
        ).acheck_methodology(SearchStrategy.GLOBAL)
    # auto never routes to them there.
    assert routable_strategies(off, no_reports) == (
        SearchStrategy.LOCAL,
        SearchStrategy.MIX,
        SearchStrategy.ITERATIVE,
    )
    assert routable_strategies(off) == routable_strategies(off, no_reports)


def test_report_generation_off_warns_that_global_and_drift_are_refused() -> None:
    def warned(config: Config) -> bool:
        return any(
            "report_generation.enabled is false" in m and "global and drift" in m
            for m in config.ineffective_settings()
        )

    assert warned(
        Config(graph={"community_detection": {"report_generation": {"enabled": False}}})
    )
    assert not warned(Config())
    # Without graphrag there is nothing global or drift could read anyway.
    assert not warned(
        Config(
            indexing={"methodologies": ["lightrag"]},
            graph={"community_detection": {"report_generation": {"enabled": False}}},
        )
    )
    assert not warned(Config(graph={"community_detection": {"enabled": False}}))


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
        "iterative, ppr, structured, or auto" in message
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
    # A wrapper reads the wrapped strategy's methodology from the registry,
    # without building a strategy: iterative
    # over mix is served by a LightRAG namespace, over local refused.
    light = _Records(_record(*LIGHT).model_dump(mode="json"))
    config = Config()
    config.search.iterative_search.base = "mix"  # type: ignore[assignment]
    assert await _chain(config, light).acheck_methodology(SearchStrategy.ITERATIVE)
    config.search.iterative_search.base = "local"  # type: ignore[assignment]
    chain = _chain(config, light)
    with pytest.raises(MethodologyNotIndexedError):
        await chain.acheck_methodology(SearchStrategy.ITERATIVE)
    assert chain.built == []


async def test_a_failed_record_read_is_not_cached() -> None:
    from unified_kg_rag.shared import MethodologyNotIndexedError

    # The store fails once: the config (graphrag included) stands in for
    # that query only, and the next one reads the record (no graphrag).
    records = _Records(RuntimeError("store down"))  # type: ignore[arg-type]
    chain = _chain(Config(), records)
    assert G in await chain.acheck_methodology(SearchStrategy.MIX)
    records.record = _record(*LIGHT).model_dump(mode="json")
    assert await chain.acheck_methodology(SearchStrategy.MIX) == (L, H)
    assert records.reads == 2
    with pytest.raises(MethodologyNotIndexedError):
        await chain.acheck_methodology(SearchStrategy.GLOBAL)


async def test_held_artifacts_are_cached_per_suffix() -> None:
    from unified_kg_rag.shared import MethodologyNotIndexedError

    class _PerSuffix(_Records):
        def __init__(self, records: dict[str | None, dict]) -> None:
            super().__init__(None)
            self.records = records

        async def aread_build_record(self, suffix: str | None) -> dict | None:
            self.reads += 1
            return self.records[suffix]

    records = _PerSuffix(
        {
            "tenant-a": _record(*FULL).model_dump(mode="json"),
            "tenant-b": _record(*LIGHT).model_dump(mode="json"),
        }
    )
    chain = _chain(Config(), records)
    assert G in await chain.acheck_methodology(SearchStrategy.GLOBAL, "tenant-a")
    # tenant-b's own record, not tenant-a's cached one, decides.
    with pytest.raises(MethodologyNotIndexedError, match="'tenant-b'"):
        await chain.acheck_methodology(SearchStrategy.GLOBAL, "tenant-b")
    assert G in await chain.acheck_methodology(SearchStrategy.GLOBAL, "tenant-a")


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
    assert routable_strategies(config, TREE_ONLY) == (SearchStrategy.NAIVE,)
    assert serves(SearchStrategy.SIMPLE, LIGHT, Config())


async def test_the_refusal_says_what_adding_the_methodology_costs() -> None:
    from unified_kg_rag.shared import MethodologyNotIndexedError

    tree_only = _Records(_record(A.TEXT_UNITS, A.SUMMARY_TREE).model_dump(mode="json"))
    light = _Records(_record(*LIGHT).model_dump(mode="json"))
    # raptor: the tree is built from the stored text units, no rebuild.
    with pytest.raises(MethodologyNotIndexedError) as refused:
        await _chain(Config(), light).acheck_methodology(SearchStrategy.TREE)
    # The tree is added through indexing.methodologies: a stage list alone
    # builds a tree the build record does not name, which stays refused.
    assert "include it in indexing.methodologies" in str(refused.value)
    assert "no re-extraction" in str(refused.value)
    assert "--enabled-stages" not in str(refused.value)
    # The command it gives selects raptor with what the config selects.
    assert "--methodologies graphrag,lightrag,raptor" in str(refused.value)
    assert "rebuilt" not in str(refused.value)
    # A graph methodology on a raptor-only namespace extracts every unit.
    with pytest.raises(MethodologyNotIndexedError) as refused:
        await _chain(Config(), tree_only).acheck_methodology(SearchStrategy.LOCAL)
    assert "every text unit is extracted" in str(refused.value)
    # Beside stored extractions, only the added model calls are new.
    with pytest.raises(MethodologyNotIndexedError) as refused:
        await _chain(Config(), light).acheck_methodology(SearchStrategy.GLOBAL)
    assert "rebuilt once from its stored extractions" in str(refused.value)


async def test_structured_on_a_tree_only_namespace_names_the_graph_methodologies() -> (
    None
):
    from unified_kg_rag.shared import MethodologyNotIndexedError

    # Regression (rv23): it advised "naming any methodology", raptor too,
    # which builds no graph.
    tree_only = _Records(_record(A.TEXT_UNITS, A.SUMMARY_TREE).model_dump(mode="json"))
    with pytest.raises(MethodologyNotIndexedError) as refused:
        await _chain(Config(), tree_only).acheck_methodology(SearchStrategy.STRUCTURED)
    message = str(refused.value)
    assert "'structured' reads the entities, graph" in message
    assert "naming one of graphrag, lightrag, hipporag." in message
    assert "any methodology" not in message


async def test_the_refusal_advises_by_the_source_scopes_of_the_record() -> None:
    from unified_kg_rag.shared import MethodologyNotIndexedError

    def light(scopes: list[str] | None) -> _Records:
        record = _record(*LIGHT).model_copy(update={"source_scopes": scopes})
        return _Records(record.model_dump(mode="json"))

    async def refusal(records: _Records, strategy: SearchStrategy) -> str:
        with pytest.raises(MethodologyNotIndexedError) as refused:
            await _chain(Config(), records).acheck_methodology(strategy)
        return str(refused.value)

    # Several source scopes: a rebuild from one corpus is refused, so the
    # advice is a namespace of its own or a reset, never a rebuild.
    for strategy in (SearchStrategy.GLOBAL, SearchStrategy.LOCAL):
        shared = await refusal(light(["/corpus-a", "api"]), strategy)
        assert "source scopes /corpus-a, api" in shared
        assert "ingestion refuses" in shared
        assert "indexing.reset: true" in shared
        assert "rebuilt once" not in shared
    # Also where the graph's first build is the cost: a rebuild all the same.
    tree = _record(*TREE_ONLY).model_copy(update={"source_scopes": ["a", "b"]})
    first = await refusal(_Records(tree.model_dump(mode="json")), SearchStrategy.LOCAL)
    assert "source scopes a, b" in first
    assert "every text unit is extracted" not in first
    # One: a re-run over that scope's corpus, also for API-fed documents.
    api = await refusal(light(["api"]), SearchStrategy.GLOBAL)
    assert "over the corpus of the source scope 'api'" in api
    assert "processing.document_parsing.source_scope: 'api'" in api
    assert "rebuilt once from its stored extractions" in api
    # A record without scopes (no registry, an earlier version) names the
    # condition the rebuild needs.
    for old in (light(None), _Records(_record(*LIGHT).model_dump(mode="json"))):
        generic = await refusal(old, SearchStrategy.GLOBAL)
        assert "a namespace of one source scope is rebuilt once" in generic
        assert "also holds other source scopes is not" in generic


def test_a_record_from_before_source_scopes_still_reads() -> None:
    dump = _record(*FULL).model_dump(mode="json")
    del dump["source_scopes"]
    assert IndexBuildRecord.model_validate(dump).source_scopes is None


def test_the_build_record_names_the_source_scopes_the_namespace_holds() -> None:
    from unified_kg_rag.application.ingestion.incremental import (
        recorded_source_scopes,
    )
    from unified_kg_rag.domain.models import DocumentDelta, PipelineContext

    def scopes(reset: bool = False, **delta: list[str]) -> list[str] | None:
        context = PipelineContext.model_construct(
            incremental_scope="ns|/corpus-b",
            incremental_retired_scopes=["s3://bucket/old/"],
            incremental_delta=DocumentDelta(
                stored_scopes=[
                    "ns|/corpus-a",
                    "ns|/corpus-b",
                    "ns|api",
                    "ns|s3://bucket/old",
                    "other|/corpus-c",
                ],
                **delta,
            ),
        )
        return recorded_source_scopes(context, reset)

    # The run's, the others of its namespace, but a retired one (under
    # any spelling) and another namespace's.
    assert scopes(new=["d"]) == ["/corpus-a", "/corpus-b", "api"]
    assert scopes(unchanged=["d"]) == ["/corpus-a", "/corpus-b", "api"]
    # A run that deleted its corpus's last document holds the others only.
    assert scopes(deleted=["d"]) == ["/corpus-a", "api"]
    # A reset deletes the others.
    assert scopes(reset=True, new=["d"]) == ["/corpus-b"]
    # Without the registry: unknown.
    unknown = PipelineContext.model_construct(
        incremental_scope=None, incremental_delta=None
    )
    assert recorded_source_scopes(unknown, False) is None


def test_dropped_methodologies_are_those_a_run_stops_maintaining() -> None:
    full = _record(*FULL).model_dump(mode="json")
    lightrag = Config(indexing={"methodologies": ["lightrag"]})
    assert mp.dropped_methodologies(full, lightrag) == (G,)
    assert mp.dropped_methodologies(full, Config()) == ()
    # Adding is not dropping; a record without artifacts drops nothing.
    light = _record(*LIGHT).model_dump(mode="json")
    assert mp.dropped_methodologies(light, Config()) == ()
    assert mp.dropped_methodologies(None, lightrag) == ()
    del full["artifacts"]
    assert mp.dropped_methodologies(full, lightrag) == ()
    # An artifact switch drops what needs it.
    no_vectors = Config(
        indexing={"opensearch": {"build_relationship_vector_index": False}}
    )
    assert mp.dropped_methodologies(light, no_vectors) == (L, H)


def test_dropped_artifacts_are_the_corpus_wide_ones_a_run_stops_building() -> None:
    full = _record(*FULL, A.CLAIMS).model_dump(mode="json")
    assert mp.dropped_artifacts(full, Config(processing=_CLAIMS_ON)) == frozenset()
    # Report generation off keeps graphrag (local reads it without them),
    # but the record stops naming the reports.
    no_reports = Config(
        graph={"community_detection": {"report_generation": {"enabled": False}}},
        processing=_CLAIMS_ON,
    )
    assert mp.dropped_methodologies(full, no_reports) == ()
    assert mp.dropped_artifacts(full, no_reports) == {A.COMMUNITY_REPORTS}
    assert mp.switches_set_off(no_reports, {A.COMMUNITY_REPORTS}) == [
        "graph.community_detection.report_generation.enabled"
    ]
    assert mp.dropped_artifacts(full, Config()) == {A.CLAIMS}
    assert mp.switches_set_off(Config(), {A.CLAIMS}) == [
        "processing.claim_extraction.enabled"
    ]
    # A methodology left out drops its corpus-wide artifacts; raptor's tree
    # is not one (its stage builds it from the stored text units).
    lightrag = Config(indexing={"methodologies": ["lightrag"]}, processing=_CLAIMS_ON)
    assert mp.dropped_artifacts(full, lightrag) == {
        A.COMMUNITIES,
        A.COMMUNITY_REPORTS,
    }
    assert mp.switches_set_off(lightrag, {A.COMMUNITIES}) == []
    with_tree = _record(*FULL, A.SUMMARY_TREE).model_dump(mode="json")
    assert mp.dropped_methodologies(with_tree, Config()) == (R,)
    assert mp.dropped_artifacts(with_tree, Config()) == frozenset()
    # Nothing to tell without a record that names its artifacts.
    assert mp.dropped_artifacts(None, Config()) == frozenset()
    assert mp.dropped_artifacts({"pipeline_id": "p"}, Config()) == frozenset()


_CLAIMS_ON = {"claim_extraction": {"enabled": True}}


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


def test_the_auto_fallback_is_warned_about_once(caplog, monkeypatch) -> None:
    import logging

    from unified_kg_rag.adapters.retrieval import strategy_registry
    from unified_kg_rag.shared import logging as shared_logging

    monkeypatch.setattr(shared_logging, "_warned_once", set())
    config = Config(search={"auto_routable_strategies": ["global", "drift"]})
    with caplog.at_level(logging.DEBUG, logger=strategy_registry.logger.name):
        for _ in range(3):
            assert routable_strategies(config, TREE_ONLY) == (SearchStrategy.TREE,)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "auto uses tree" in warnings[0].getMessage()


def test_the_artifact_kinds_are_the_vector_stores_item_kinds() -> None:
    assert set(mp.ARTIFACT_KINDS.values()) <= set(STORED_ITEM_KINDS)
    # Every artifact is the vector store's, the graph's or the tree's.
    graph = mp.GRAPH_STORE_ARTIFACTS | {IndexArtifact.RELATIONSHIPS}
    assert set(mp.ARTIFACT_KINDS) | graph | {IndexArtifact.SUMMARY_TREE} == set(
        IndexArtifact
    )


def test_stored_artifacts_shown_names_only_what_the_stores_hold() -> None:
    config = Config()
    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    lightrag_build = {"text_units": 3, "entities": 4, "relationships": 2}
    shown = mp.stored_artifacts_shown(
        config, lightrag_build, graph_holds=True, graph_edges=True, tree=False
    )
    assert shown == {
        IndexArtifact.TEXT_UNITS,
        IndexArtifact.ENTITIES,
        IndexArtifact.RELATIONSHIP_VECTORS,
        IndexArtifact.GRAPH,
        IndexArtifact.RELATIONSHIPS,
    }
    with_reports = {**lightrag_build, "community_reports": 1, "claims": None}
    shown = mp.stored_artifacts_shown(
        config, with_reports, graph_holds=True, graph_edges=None, tree=True
    )
    assert {IndexArtifact.COMMUNITIES, IndexArtifact.COMMUNITY_REPORTS} <= shown
    # Neither the tree (raptor is not selected) nor claims (not extracted).
    assert not {IndexArtifact.SUMMARY_TREE, IndexArtifact.CLAIMS} & shown
    # An empty graph holds no graph artifact; one that cannot tell, the
    # config's.
    assert IndexArtifact.GRAPH not in mp.stored_artifacts_shown(
        config, lightrag_build, graph_holds=False, graph_edges=None, tree=False
    )
    assert IndexArtifact.GRAPH in mp.stored_artifacts_shown(
        config, lightrag_build, graph_holds=None, graph_edges=None, tree=False
    )
    # Without reports to show them, communities are held with the graph.
    config.graph.community_detection.report_generation.enabled = False
    assert IndexArtifact.COMMUNITIES in mp.stored_artifacts_shown(
        config, lightrag_build, graph_holds=True, graph_edges=None, tree=False
    )


def test_stored_artifacts_shown_holds_the_empty_artifacts_of_a_held_graph() -> None:
    # Regression: a corpus that states no claim, or no relationship, leaves
    # its claims or relationship index empty (or not created) after a build
    # that made them; the rewritten record left them out, so the next run
    # rebuilt the namespace for them, or refused its LightRAG queries. Beside
    # held text units and graph, their emptiness tells nothing apart.
    config = Config()
    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    config.processing.claim_extraction.enabled = True
    empty = {"text_units": 3, "entities": 4, "relationships": 0, "claims": None}
    for edges in (False, None):
        shown = mp.stored_artifacts_shown(
            config, empty, graph_holds=True, graph_edges=edges, tree=False
        )
        assert {IndexArtifact.CLAIMS, IndexArtifact.RELATIONSHIP_VECTORS} <= shown
        assert Methodology.LIGHTRAG in methodologies_holding(shown)
    # Beside a stored edge, an empty relationships index was never built
    # (a build with build_relationship_vector_index off): only the claims.
    shown = mp.stored_artifacts_shown(
        config, empty, graph_holds=True, graph_edges=True, tree=False
    )
    assert IndexArtifact.CLAIMS in shown
    assert IndexArtifact.RELATIONSHIP_VECTORS not in shown
    assert Methodology.LIGHTRAG not in methodologies_holding(shown)
    # Not what the config does not build, nor without the text units or graph.
    config.processing.claim_extraction.enabled = False
    assert IndexArtifact.CLAIMS not in mp.stored_artifacts_shown(
        config, empty, graph_holds=True, graph_edges=None, tree=False
    )
    config.processing.claim_extraction.enabled = True
    for documents, graph in (({**empty, "text_units": 0}, True), (empty, False)):
        shown = mp.stored_artifacts_shown(
            config, documents, graph_holds=graph, graph_edges=None, tree=False
        )
        assert not {IndexArtifact.CLAIMS, IndexArtifact.RELATIONSHIP_VECTORS} & shown
