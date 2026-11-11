# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every configuration field is classified for the stage cache keys.

``cache_keys.CONFIG_KEYING`` says, for each field, from which stage it keys
the stage caches or why it keys none. A field added without a decision
fails here, so a setting that changes a stage's output can no longer be left
out of its key by omission.
"""

from __future__ import annotations

from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel

from unified_kg_rag.domain.ingestion import cache_keys
from unified_kg_rag.domain.ingestion.cache_keys import (
    CONFIG_KEYING,
    EXTRACTION_SETTINGS_VERSION,
    Keyed,
    Unkeyed,
    answer_settings,
    changed_settings,
    classify,
    extraction_fingerprint,
    extraction_settings,
    shapes_answers,
    stage_input_fingerprint,
    unit_record_fingerprint,
)
from unified_kg_rag.domain.models import Config, ExtractionSettings, PipelineStageType
from unified_kg_rag.domain.models.config import (
    LanguageCode,
    NeptuneIndexingConfig,
    OpenSearchIndexingConfig,
)

pytestmark = pytest.mark.unit


def _sections(annotation: Any) -> list[type[BaseModel]]:
    found: list[type[BaseModel]] = []
    for arg in get_args(annotation) or (annotation,):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            found.append(arg)
        elif get_origin(arg) is not None:
            found.extend(_sections(arg))
    return found


def _field_paths(model: type[BaseModel], prefix: str = "") -> list[str]:
    """Every non-section field of ``model``'s type tree, as a dotted path."""
    paths: list[str] = []
    for name, field in model.model_fields.items():
        sections = _sections(field.annotation)
        if sections and get_origin(field.annotation) not in (dict, list):
            for section in sections:
                paths.extend(_field_paths(section, f"{prefix}{name}."))
        else:
            paths.append(f"{prefix}{name}")
    return paths


_FIELDS = sorted(set(_field_paths(Config)))

# The fields keyed only once set (cache_keys._KEYED_ONCE_SET), written out:
# compared against the set, a dropped entry (which re-keys every default
# configuration) fails here instead of passing silently.
_KEYED_ONCE_SET = frozenset(
    {
        "indexing.methodologies",
        "processing.graph_extraction.relationship_types",
        "processing.graph_extraction.relationship_patterns",
        "processing.graph_extraction.schema_mode",
        "processing.graph_extraction.schema_source.kind",
        "search.abstention.mode",
        "search.abstention.strip_marker",
        "search.abstention.on_insufficient",
        "memory.max_history_tokens",
        # The raptor methodology's (docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in).
        "indexing.opensearch.summary_nodes_index_prefix",
        "models.roles.summary_tree",
        "custom_prompts.tree_summary_system",
        "custom_prompts.tree_summary_human",
        "search.token_manager.type_budgets.summary",
        *(
            f"indexing.summary_tree.{name}"
            for name in (
                "scope",
                "partition_by",
                "algorithm",
                "target_children",
                "soft_threshold",
                "reduction_dims",
                "max_levels",
                "min_nodes_to_cluster",
                "min_level_shrink",
                "max_cluster_tokens",
                "max_summary_tokens",
                "max_lineage_text_units",
                "min_leaves_per_tree",
                "refresh",
                "random_state",
                "max_leaves_in_memory",
            )
        ),
        *(
            f"search.tree_search.{name}"
            for name in (
                "graph_signal",
                "graph_seed_entities",
                "text_top_k",
                "summary_top_k",
                "max_nodes_per_level",
                "expand_children",
            )
        ),
        # The chunker's and the translator's prompt overrides.
        "custom_prompts.text_chunking_system",
        "custom_prompts.text_chunking_human",
        "custom_prompts.text_translation_system",
        "custom_prompts.text_translation_human",
    }
)


def test_the_fields_keyed_once_set_are_these() -> None:
    assert cache_keys._KEYED_ONCE_SET == _KEYED_ONCE_SET


def test_every_config_field_is_classified() -> None:
    unclassified = [path for path in _FIELDS if classify(path) is None]
    assert not unclassified, (
        "Classify these fields in cache_keys.CONFIG_KEYING: Keyed(stage) from "
        "the first stage whose output they shape, or Unkeyed(reason): "
        f"{unclassified}"
    )


def test_every_entry_names_a_config_field() -> None:
    stale = [
        entry
        for entry in CONFIG_KEYING
        if not any(path == entry or path.startswith(f"{entry}.") for path in _FIELDS)
    ]
    assert not stale, f"CONFIG_KEYING entries that match no field: {stale}"


def test_every_unkeyed_entry_gives_a_reason() -> None:
    assert all(
        entry.reason.strip()
        for entry in CONFIG_KEYING.values()
        if isinstance(entry, Unkeyed)
    )


@pytest.mark.parametrize(
    "section, prefix",
    [
        (OpenSearchIndexingConfig, "indexing.opensearch"),
        (NeptuneIndexingConfig, "indexing.neptune"),
    ],
)
def test_store_prefixes_key_from_parsing(section: type[BaseModel], prefix: str) -> None:
    # store_namespace_tag puts every *_prefix field into the registry
    # namespace, part of each document's identity.
    for name in section.model_fields:
        if name.endswith("_prefix"):
            assert classify(f"{prefix}.{name}") == Keyed(
                PipelineStageType.DOCUMENT_PARSING, per_document=False
            ), name


def _local(**llm: object) -> Config:
    return Config.model_validate({"backend": "local", "providers": {"llm": llm}})


@pytest.mark.parametrize(
    "field, value",
    [
        ("temperature", 0.7),
        ("max_output_tokens", 512),
        ("context_window_size", 4096),
    ],
)
def test_local_model_knobs_key_every_stage(field: str, value: object) -> None:
    # They change what every local model call returns; the allow-list the
    # keys were built from before missed them.
    baseline, changed = _local(), _local(**{field: value})
    for stage in PipelineStageType:
        assert stage_input_fingerprint(changed, stage) != stage_input_fingerprint(
            baseline, stage
        ), stage
    assert extraction_fingerprint(changed) != extraction_fingerprint(baseline)


def test_the_default_output_cap_keys_local_models_too() -> None:
    # A local model is asked for models.default_max_output_tokens
    # (clamped to providers.llm.max_output_tokens); the keys dropped it with
    # the other Bedrock settings, so a changed cap reused cached output.
    capped = Config.model_validate(
        {"backend": "local", "models": {"default_max_output_tokens": 512}}
    )
    for stage in PipelineStageType:
        assert stage_input_fingerprint(capped, stage) != stage_input_fingerprint(
            _local(), stage
        ), stage
    assert extraction_fingerprint(capped) != extraction_fingerprint(_local())
    assert unit_record_fingerprint(capped, gleaned=False) != (
        unit_record_fingerprint(_local(), gleaned=False)
    )


def test_the_moved_output_cap_keys_under_its_former_path() -> None:
    # models.default_max_output_tokens was aws.bedrock.default_max_output_tokens:
    # keyed and recorded under the old path, the move re-keys no stage cache
    # and re-extracts no incrementally indexed document.
    old, new = (
        "aws.bedrock.default_max_output_tokens",
        "models.default_max_output_tokens",
    )
    recorded = extraction_settings(Config()).values
    assert recorded[old] == 16384 and new not in recorded
    assert answer_settings(Config())[old] == 16384
    capped = Config.model_validate({"models": {"default_max_output_tokens": 512}})
    before_the_move = ExtractionSettings(
        version=EXTRACTION_SETTINGS_VERSION, values={**recorded, old: 512}
    )
    assert changed_settings(before_the_move, extraction_settings(capped)) == []
    assert changed_settings(before_the_move, extraction_settings(Config())) == [new]
    assert classify(old) is classify(new)


def test_settings_of_the_inactive_model_provider_do_not_key() -> None:
    stage = PipelineStageType.COMMUNITY_DETECTION
    bedrock = Config.model_validate({"providers": {"llm": {"temperature": 0.9}}})
    assert stage_input_fingerprint(bedrock, stage) == stage_input_fingerprint(
        Config(), stage
    )
    local = Config.model_validate(
        {"backend": "local", "aws": {"bedrock": {"default_effort": "max"}}}
    )
    assert stage_input_fingerprint(local, stage) == stage_input_fingerprint(
        _local(), stage
    )


def test_an_unclassified_field_keys_every_stage(monkeypatch) -> None:
    # Forgetting to classify a field must cost a cache miss, not a stale hit.
    keying = dict(CONFIG_KEYING)
    del keying["processing.max_concurrency"]
    retuned = Config()
    retuned.processing.max_concurrency += 1
    stage = PipelineStageType.DOCUMENT_PARSING
    assert stage_input_fingerprint(retuned, stage) == stage_input_fingerprint(
        Config(), stage
    )
    monkeypatch.setattr(cache_keys, "CONFIG_KEYING", keying)
    assert stage_input_fingerprint(retuned, stage) != stage_input_fingerprint(
        Config(), stage
    )


def test_answer_settings_cover_every_field_but_the_neutral_ones() -> None:
    # run-eval hashes these into its manifest: a field missing here would let
    # two runs that answer differently claim the same configuration. Only
    # fields classified as changing no answer (and the location fields) stay
    # out, so the same evaluation hashes the same in every environment.
    hashed = {
        cache_keys._CURRENT_PATH.get(path, path) for path in answer_settings(Config())
    }
    excluded = {path for path in _FIELDS if not shapes_answers(path)}
    # Plus one derived entry: whether a guardrail is on (its id is a location).
    # Fields added at a default that keeps the earlier behaviour count once
    # set (test_extraction_schema).
    assert hashed == set(_FIELDS) - excluded - _KEYED_ONCE_SET | {
        "aws.bedrock.guardrail.enabled"
    }
    assert _KEYED_ONCE_SET <= set(_FIELDS)
    for path in excluded:
        rule = classify(path)
        assert (
            isinstance(rule, Unkeyed)
            and not rule.shapes_answers
            or (path in cache_keys._LOCATION_FIELDS)
        ), path
    neutral = {"telemetry", cache_keys._WHERE, cache_keys._THROUGHPUT}
    assert {r.reason for r in CONFIG_KEYING.values() if isinstance(r, Unkeyed)} >= (
        neutral
    )


def test_prompt_cache_ttl_shapes_no_answer() -> None:
    # A cache checkpoint's lifetime changes what a query costs, not what it
    # answers: two evaluations differing only in it hash alike.
    path = "aws.bedrock.query_prompt_cache_ttl"
    assert not shapes_answers(path)
    one_hour = Config()
    one_hour.aws.bedrock.query_prompt_cache_ttl = "1h"
    assert answer_settings(one_hour) == answer_settings(Config())


def test_location_fields_are_config_fields() -> None:
    assert cache_keys._LOCATION_FIELDS <= set(_FIELDS)


def test_the_neighbour_order_shapes_answers_but_keys_no_stage() -> None:
    # It changes which entities graph expansion returns (eval manifests and
    # paired comparisons record it), never a cached stage output.
    path = "indexing.neptune.deterministic_order"
    assert shapes_answers(path)
    assert path in answer_settings(Config())
    assert not isinstance(classify(path), Keyed)
    # Its section's entry classifies it: no entry of its own.
    assert path not in CONFIG_KEYING
    assert classify(path) == CONFIG_KEYING["indexing.neptune"]


@pytest.mark.parametrize(
    "fingerprint",
    [
        cache_keys.description_summary_fingerprint,
        cache_keys.community_report_fingerprint,
    ],
)
def test_a_stored_answer_is_keyed_by_the_target_language(fingerprint: Any) -> None:
    # Summaries and reports are written in the target language, which no
    # per-stage setting keys: another one asks the model again.
    other = Config()
    other.processing.translation.target_language = LanguageCode.FR
    assert fingerprint(other) != fingerprint(Config())


def _summarized_otherwise(kind: str) -> Config:
    config = Config()
    summarization = config.processing.graph_extraction.description_summarization
    if kind == "prompt":
        config.custom_prompts.description_summarization_system = "Summarize briefly."
    elif kind == "threshold":
        summarization.force_summary_threshold_tokens = 77
    elif kind == "model":
        config.models.roles.description_summary = "vendor.small-v1"
    else:
        summarization.enabled = False
    return config


@pytest.mark.parametrize("kind", ["prompt", "threshold", "model", "off"])
def test_a_summary_setting_keys_the_summaries_not_the_extraction(kind: str) -> None:
    # Descriptions are summarized once merged, never in a unit's record nor
    # per document: keyed at graph extraction, a new summary threshold asked
    # the model to extract every unit (and gleaning) again.
    config, base = _summarized_otherwise(kind), Config()
    for gleaned, claims in ((False, False), (True, True)):
        assert cache_keys.unit_record_fingerprint(
            config, gleaned=gleaned, claims=claims
        ) == cache_keys.unit_record_fingerprint(base, gleaned=gleaned, claims=claims)
    assert cache_keys.extraction_fingerprint(config) == (
        cache_keys.extraction_fingerprint(base)
    )
    stage = cache_keys.stage_input_fingerprint
    for upstream in (PipelineStageType.GRAPH_EXTRACTION, PipelineStageType.GLEANING):
        assert stage(config, upstream) == stage(base, upstream)
    assert stage(config, PipelineStageType.GRAPH_RESOLUTION) != stage(
        base, PipelineStageType.GRAPH_RESOLUTION
    )
    # A summary records the settings it was made with: one made otherwise
    # is not reused.
    assert cache_keys.description_summary_fingerprint(config) != (
        cache_keys.description_summary_fingerprint(base)
    )


def _keyed_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """The keying of earlier releases: the summary settings at graph
    extraction, per document."""
    earlier = Keyed(PipelineStageType.GRAPH_EXTRACTION)
    for path in cache_keys._SUMMARY_SETTINGS:
        monkeypatch.setitem(cache_keys.CONFIG_KEYING, path, earlier)
    monkeypatch.setattr(cache_keys, "_summary_key", dict)


@pytest.mark.parametrize("kind", [None, "threshold"])
def test_a_scope_recorded_with_the_summary_settings_reads_unchanged(
    kind: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A scope's settings recorded before list the summary settings: an
    # upgrade that drops them re-extracts no document.
    config = _summarized_otherwise(kind) if kind else Config()
    with monkeypatch.context() as patched:
        _keyed_as_before(patched)
        recorded = cache_keys.extraction_settings(config)
    assert any("description_summar" in path for path in recorded.values)
    now = cache_keys.extraction_settings(config)
    assert not any("description_summar" in path for path in now.values)
    assert cache_keys.changed_settings(recorded, now) == []


@pytest.mark.parametrize(
    "configure",
    [
        lambda c: None,
        lambda c: setattr(c.models, "default", "vendor.other-model-v2"),
        lambda c: setattr(c.processing.claim_extraction, "enabled", True),
    ],
    ids=["default", "models-default", "claims"],
)
def test_a_record_stored_while_summaries_keyed_extraction_keeps_its_key(
    configure: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config()
    configure(config)
    now = {
        (gleaned, claims): cache_keys.unit_record_fingerprint(
            config, gleaned=gleaned, claims=claims
        )
        for gleaned in (False, True)
        for claims in (False, True)
    }
    _keyed_as_before(monkeypatch)
    assert now == {
        (gleaned, claims): cache_keys.unit_record_fingerprint(
            config, gleaned=gleaned, claims=claims
        )
        for gleaned in (False, True)
        for claims in (False, True)
    }


def _fast_tier(c: Config) -> None:
    c.models.fast = "vendor.fast-v9"


def _simple_chunks() -> Config:
    # Nothing else a document's extraction reads goes through the tiers.
    config = Config()
    config.processing.chunking.chunker_type = "simple"
    config.processing.translation.enabled = False
    return config


def test_a_tier_only_the_summary_model_reads_keeps_the_record_keys() -> None:
    # A tier changed nothing a document's extraction records, so no warning
    # told the user; keyed through it, a reset extracted every unit again.
    base, config = _simple_chunks(), _simple_chunks()
    _fast_tier(config)
    assert (
        cache_keys.changed_settings(
            cache_keys.extraction_settings(base), cache_keys.extraction_settings(config)
        )
        == []
    )
    for gleaned, claims in ((False, False), (True, True)):
        assert cache_keys.unit_record_fingerprint(
            config, gleaned=gleaned, claims=claims
        ) == cache_keys.unit_record_fingerprint(base, gleaned=gleaned, claims=claims)


@pytest.mark.parametrize(
    ("prompt", "stage"),
    [
        ("DescriptionSummarizationPrompt", PipelineStageType.GRAPH_RESOLUTION),
        ("ClaimExtractionPrompt", PipelineStageType.CLAIM_EXTRACTION),
        ("GraphExtractionPrompt", PipelineStageType.GRAPH_EXTRACTION),
    ],
)
def test_an_edited_builtin_prompt_rekeys_its_stage_and_those_after(
    monkeypatch, prompt: str, stage: PipelineStageType
) -> None:
    # A dev build shares its package version with the build before: a
    # prompt edited between them re-keys the stages its answers shape, so a
    # resumed pipeline does not serve what the earlier prompt produced.
    from unified_kg_rag.domain import prompts

    order = list(PipelineStageType)
    before = {s: stage_input_fingerprint(Config(), s) for s in order}
    prompt_class = getattr(prompts, prompt)
    monkeypatch.setattr(
        prompt_class,
        "system_prompt_template",
        prompt_class.system_prompt_template + " Edited.",
    )
    cache_keys._builtin_prompts.cache_clear()
    try:
        after = {s: stage_input_fingerprint(Config(), s) for s in order}
    finally:
        monkeypatch.undo()
        cache_keys._builtin_prompts.cache_clear()
    changed = [s for s in order if before[s] != after[s]]
    assert changed == order[order.index(stage) :]
