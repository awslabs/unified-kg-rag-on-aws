# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input-aware cache keys for ingestion stage results, and the extraction
settings recorded with what a document produced.

A stage's cached output may only be reused when the inputs that PRODUCED it are
unchanged, so its key carries a fingerprint of those inputs:

* **Every setting counts unless declared otherwise.** ``CONFIG_KEYING``
  classifies every configuration field: :class:`Keyed` from the first stage
  whose output it shapes, or :class:`Unkeyed` with the reason it shapes none
  (a throughput knob, a connection setting, a query-time setting). A stage's
  key hashes every keyed field of it and of every upstream stage (a stage
  consumes its predecessors' output), the package version (built-in prompt
  templates and parsing rules are code) and the corpus manifest. A field the
  table does not cover counts from the first stage, and
  ``tests/unit/test_cache_key_coverage.py`` fails until it is classified.
* **Only settings in effect count.** A block switched off (``enabled:
  false``) contributes its switch alone, except for the fields other
  components read whatever the switch says (``_READ_WHEN_DISABLED``); the
  local model knobs count only while the language models are not Bedrock's
  and the Bedrock ones only while they are, except the output cap every
  model is asked for (``_EVERY_PROVIDER``); a stage's prompt overrides only
  while the stage runs
  (``_IN_EFFECT_WHEN``).

The settings that shape what a document produces (its text units, entities,
relationships and claims) form its extraction settings
(:func:`extraction_settings`): a record of the keyed fields marked
``per_document`` through claim extraction, without the package version, so
an upgrade does not by itself re-extract an incrementally indexed corpus. An
incremental run records the fingerprint of the record on each document it
commits, and the record itself once per registry scope, so the next run can
compare the two field by field (:func:`changed_settings`).

The same classification says which settings shape what a query answers
(:func:`answer_settings`, hashed into the ``run-eval`` manifest): every
setting but the unkeyed ones that change no answer either (where a call goes,
how fast it runs, telemetry, provisioning) and the few keyed ones naming where
a run reads and writes (``_LOCATION_FIELDS``).

The corpus itself is an input too: the source path alone says nothing about
what the files contain, so with a fixed ``pipeline_id`` an edited, added, or
removed file would otherwise resume from the previous corpus's documents (and,
in incremental mode, its delta). The caller passes a corpus manifest
fingerprint (:func:`corpus_manifest_fingerprint`: relative path + size +
content hash per file), folded into every stage's key since parsing is the
first stage. The seed graphs a run imports are part of its corpus: their
converted content is folded into the same fingerprint
(:func:`seed_corpus_fingerprint`).
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from unified_kg_rag import __version__
from unified_kg_rag.domain.models import (
    ChunkingStrategy,
    Config,
    ExtractionSettings,
    ModelRole,
    PipelineStageType,
)
from unified_kg_rag.domain.models.config import SummaryTreeConfig, TreeSearchConfig
from unified_kg_rag.domain.models.provider_config import BEDROCK
from unified_kg_rag.domain.models.schema import (
    EntityTypeItem,
    RelationshipPattern,
    RelationshipPatternItem,
    RelationshipTypeItem,
    entity_type_spec,
    schema_key,
)
from unified_kg_rag.shared.utils.common import compute_hash

FINGERPRINT_LENGTH = 12

# Canonical ingestion order: PipelineStageType is declared in pipeline order.
_STAGE_ORDER: tuple[PipelineStageType, ...] = tuple(PipelineStageType)
_P = PipelineStageType


@dataclass(frozen=True)
class Keyed:
    """A setting that shapes the output of ``stage`` and every later stage.

    ``stage`` ``None``: it shapes every model call, so every stage's output.
    ``per_document`` ``False``: it says where a run reads or writes, or which
    documents it runs, not what a document produces (or shapes only what the
    merged graph does: a description summary), so it keys the run's stage
    caches but is not one of the extraction settings.
    """

    stage: PipelineStageType | None
    per_document: bool = True


@dataclass(frozen=True)
class Unkeyed:
    """A setting that shapes no cached stage output, and why.

    ``shapes_answers`` ``False``: it changes no query answer or score either
    (see :func:`answer_settings`).
    """

    reason: str
    shapes_answers: bool = True


def _neutral(reason: str) -> Unkeyed:
    """An unkeyed setting that changes no answer either."""
    return Unkeyed(reason, shapes_answers=False)


_EVERY_CALL = Keyed(None)
_WHERE = "says where a call goes or with which credentials, not what it returns"
_THROUGHPUT = "changes how long a stage takes, not what it produces"
_TIER = (
    "a tier model: it reaches a stage only through the roles that run on it, "
    "keyed by their resolved model where each role runs"
)
_AT_INDEXING = "read while indexing or querying, and neither caches stage output"
_QUERY = "read at query time only"
_QUERY_PROMPT = Unkeyed("a query-time prompt")


def _keyed_at(stage: PipelineStageType, *paths: str, **kwargs: Any) -> dict[str, Any]:
    return dict.fromkeys(paths, Keyed(stage, **kwargs))


# The description summary's settings: what summarizes a merged description.
_SUMMARY_SETTINGS = (
    "processing.graph_extraction.description_summarization",
    "models.roles.description_summary",
    "custom_prompts.description_summarization_system",
    "custom_prompts.description_summarization_human",
)

# Path prefix -> how the settings under it key the stage caches. The longest
# matching prefix wins. Subtree entries are for sections a stage reads as a
# whole (a new field there is keyed with it) or that no cached stage reads
# (a new field there is unkeyed with them); a mixed section lists its fields.
CONFIG_KEYING: dict[str, Keyed | Unkeyed] = {
    "backend": Unkeyed(
        "a preset: its effect is the provider settings it fills, keyed where "
        "they are read"
    ),
    # --- language models: the knobs that change what any call returns ------
    "providers.llm.kind": _EVERY_CALL,
    "providers.llm.context_window_size": _EVERY_CALL,
    "providers.llm.max_output_tokens": _EVERY_CALL,
    "providers.llm.temperature": _EVERY_CALL,
    "providers.llm.tokenizer": _EVERY_CALL,
    "providers.llm.tiktoken_encoding": _EVERY_CALL,
    "providers.llm.base_url": _neutral(_WHERE),
    "providers.llm.api_key": _neutral(_WHERE),
    "providers.llm.timeout_seconds": _neutral(_THROUGHPUT),
    "models.default_max_output_tokens": _EVERY_CALL,
    "aws.bedrock.model_overrides": _EVERY_CALL,
    "aws.bedrock.enable_1m_context": _EVERY_CALL,
    "aws.bedrock.default_effort": _EVERY_CALL,
    "aws.bedrock.fast_effort": _EVERY_CALL,
    "aws.bedrock.ingestion_effort": _EVERY_CALL,
    "aws.bedrock.region_name": _neutral(_WHERE),
    "aws.bedrock.assumed_role_arn": _neutral(_WHERE),
    "aws.bedrock.enable_global_profile": _neutral(_WHERE),
    # How long Bedrock keeps a query prompt cached: cost and latency, never
    # the answer, so the run-eval manifest leaves it out.
    "aws.bedrock.query_prompt_cache_ttl": _neutral(
        "query chains only; ingestion checkpoints keep the 5-minute TTL"
    ),
    "aws.bedrock.guardrail": Unkeyed("screens the user query and the answer only"),
    "aws.bedrock.guardrail.identifier": _neutral(_WHERE),
    "aws.bedrock.guardrail.version": _neutral(_WHERE),
    "aws.bedrock.guardrail.trace": _neutral(
        "adds trace details to a response, not to what the model returns"
    ),
    # Shapes answers: an ungrounded answer is refused instead of returned.
    "aws.bedrock.guardrail.contextual_grounding": Unkeyed(
        "checks the final answer against its retrieved context only"
    ),
    # --- models: each role keys the stage it runs in, by its resolved id ----
    "models.default": Unkeyed(_TIER),
    "models.fast": Unkeyed(_TIER),
    # Output fixing repairs malformed model output in every stage.
    "models.roles.fixing": _EVERY_CALL,
    "models.roles.chunking": Keyed(_P.TEXT_CHUNKING),
    "models.roles.translation": Keyed(_P.TRANSLATION),
    "models.roles.graph_extraction": Keyed(_P.GRAPH_EXTRACTION),
    "models.roles.description_summary": Keyed(_P.GRAPH_RESOLUTION, per_document=False),
    "models.roles.gleaning": Keyed(_P.GLEANING),
    "models.roles.claim_extraction": Keyed(_P.CLAIM_EXTRACTION),
    "models.roles.community_report": Keyed(_P.COMMUNITY_DETECTION),
    "models.roles.summary_tree": Keyed(_P.SUMMARY_TREE, per_document=False),
    "models.roles.query_translation": Unkeyed(_QUERY),
    "models.roles.entity_extraction": Unkeyed(_QUERY),
    "models.roles.strategy_selection": Unkeyed(_QUERY),
    "models.roles.query_rewrite": Unkeyed(_QUERY),
    "models.roles.answer_generation": Unkeyed(_QUERY),
    "models.roles.community_relevance": Unkeyed(_QUERY),
    "models.roles.global_map": Unkeyed(_QUERY),
    "models.roles.drift_primer": Unkeyed(_QUERY),
    "models.roles.drift_action": Unkeyed(_QUERY),
    "models.roles.query_controller": Unkeyed(_QUERY),
    "models.roles.evaluation": Unkeyed("read by run-eval only"),
    "models.roles.answer_span": Unkeyed("read by run-eval only"),
    "models.roles.pairwise_overlap": Unkeyed("read by run-eval only"),
    "models.embedding": Unkeyed(_AT_INDEXING),
    "models.rerank": Unkeyed(_QUERY),
    # --- the corpus, its identity and the registry it is diffed against ----
    "processing.document_parsing": Keyed(_P.DOCUMENT_PARSING, per_document=False),
    # The registry records the loading stage diffs the corpus against.
    "processing.document_parsing.source_scope": Keyed(
        _P.DOCUMENT_LOADING, per_document=False
    ),
    "processing.ignore_errors": Keyed(_P.DOCUMENT_PARSING, per_document=False),
    # Part of every document's identity: its registry namespace names the
    # stores (see document_identity.store_namespace_tag).
    **_keyed_at(
        _P.DOCUMENT_PARSING,
        "indexing.additional_suffix",
        "indexing.opensearch.text_units_index_prefix",
        "indexing.opensearch.entities_index_prefix",
        "indexing.opensearch.community_reports_index_prefix",
        "indexing.opensearch.relationships_index_prefix",
        "indexing.opensearch.claims_index_prefix",
        "indexing.opensearch.summary_nodes_index_prefix",
        "indexing.opensearch.extraction_records_index_prefix",
        "indexing.neptune.entity_label_prefix",
        "indexing.neptune.community_label_prefix",
        per_document=False,
    ),
    # A reset loads the whole corpus where an incremental run loads its
    # delta; the registry and its retry, quarantine and retirement rules
    # decide what the delta holds.
    **_keyed_at(
        _P.DOCUMENT_LOADING,
        "processing.deduplicate",
        "indexing.reset",
        "indexing.retire_source_scopes",
        "indexing.max_document_failures",
        "indexing.max_document_attempts",
        "providers.doc_status",
        "aws.dynamodb.table_name",
        per_document=False,
    ),
    "aws.dynamodb.create_table_if_missing": _neutral("provisions the table"),
    "aws.dynamodb.billing_mode": _neutral("provisions the table"),
    # --- the per-document stages -------------------------------------------
    **_keyed_at(
        _P.TEXT_CHUNKING,
        "processing.chunking",
        "custom_prompts.text_chunking_system",
        "custom_prompts.text_chunking_human",
    ),
    **_keyed_at(
        _P.TRANSLATION,
        "processing.translation",
        "custom_prompts.text_translation_system",
        "custom_prompts.text_translation_human",
    ),
    **_keyed_at(
        _P.GRAPH_EXTRACTION,
        "processing.graph_extraction",
        "custom_prompts.graph_extraction_system",
        "custom_prompts.graph_extraction_human",
    ),
    # Descriptions are summarized once merged: at resolution, for the report
    # members and at indexing, never in a unit's record nor per document.
    # Keyed with the extraction section once, they changed every record's
    # key and every document's extraction settings, so a new summary
    # threshold asked the model to extract every unit again. A summary
    # records the settings it was made with (description_summary_fingerprint),
    # so a stored one made otherwise is not reused.
    **_keyed_at(
        _P.GRAPH_RESOLUTION,
        "processing.graph_extraction.description_summarization",
        "custom_prompts.description_summarization_system",
        "custom_prompts.description_summarization_human",
        per_document=False,
    ),
    # Where the schema source reads: start-up loads the schema into the
    # inline keys of processing.graph_extraction, keyed with the section.
    "processing.graph_extraction.schema_source.path": _neutral(
        "a location: the schema loaded from it is keyed as the inline keys"
    ),
    "processing.graph_extraction.schema_source.options": _neutral(
        "how the source loads: the schema loaded is keyed as the inline keys"
    ),
    "processing.graph_extraction.schema_source.secret_arn": _neutral(_WHERE),
    # Start-up converts each seed graph into virtual documents; their digest
    # is folded into the corpus fingerprint (seed_corpus_fingerprint), as the
    # loaded schema is keyed through the inline keys. Where a seed is read
    # from keys nothing, and no document re-extracts when a seed changes.
    "processing.seed_graphs": _neutral(
        "the converted seed content keys every stage through the corpus " "fingerprint"
    ),
    **_keyed_at(
        _P.GLEANING,
        "processing.gleaning",
        "custom_prompts.graph_refinement_system",
        "custom_prompts.graph_refinement_human",
    ),
    **_keyed_at(
        _P.GRAPH_RESOLUTION,
        "processing.resolution_method",
        "processing.similarity_threshold",
        "processing.fuzzy_entity_resolution",
    ),
    **_keyed_at(
        _P.CLAIM_EXTRACTION,
        "processing.claim_extraction",
        "custom_prompts.claim_extraction_system",
        "custom_prompts.claim_extraction_human",
    ),
    "processing.max_concurrency": _neutral(_THROUGHPUT),
    "processing.io_workers": _neutral(_THROUGHPUT),
    # --- the corpus-wide stages --------------------------------------------
    "graph.analysis": _neutral("shapes graph statistics, which no later stage reads"),
    **_keyed_at(
        _P.COMMUNITY_DETECTION,
        "graph.community_detection",
        "custom_prompts.community_report_system",
        "custom_prompts.community_report_human",
    ),
    "graph.visualization": _neutral("renders the finished graph"),
    # The summary tree is built from the stored text units after indexing:
    # its settings key that stage alone, never what a document produces.
    **_keyed_at(
        _P.SUMMARY_TREE,
        "indexing.summary_tree",
        "custom_prompts.tree_summary_system",
        "custom_prompts.tree_summary_human",
        per_document=False,
    ),
    # --- indexing, stores and query time -----------------------------------
    # Which stages run and what the build record names, never what a stage
    # that runs produces (its switch keys a stage it turns off, which then
    # caches nothing): a methodology added to a pipeline id resumes from
    # its cached stages. Shapes answers (which strategies a query may use).
    "indexing.methodologies": Unkeyed(
        "selects the stages a run includes, not what one produces"
    ),
    "indexing.community_drift_threshold": Unkeyed(_AT_INDEXING),
    "indexing.community_compaction": _neutral(
        "only logs a recommendation after the commit, never changes what a "
        "stage produces"
    ),
    "indexing.lease_seconds": _neutral(
        "keeps concurrent runs apart, never changes what a stage produces"
    ),
    "indexing.max_failure_rate": _neutral(
        "fails a stage or the run, never changes what a stage produces"
    ),
    "indexing.max_deletion_share": _neutral(
        "stops a run before it writes, never changes what a stage produces"
    ),
    # Which neighbours graph expansion keeps (deterministic_order) included:
    # shapes answers (eval manifests record it), no cached stage output.
    "indexing.neptune": Unkeyed(_AT_INDEXING),
    "indexing.opensearch": Unkeyed(_AT_INDEXING),
    "indexing.opensearch.refresh_after_batch": _neutral(_THROUGHPUT),
    "providers.embedding": Unkeyed(_AT_INDEXING),
    "providers.embedding.base_url": _neutral(_WHERE),
    "providers.embedding.api_key": _neutral(_WHERE),
    "providers.embedding.timeout_seconds": _neutral(_THROUGHPUT),
    "providers.rerank": Unkeyed(_QUERY),
    "providers.rerank.device": _neutral(_THROUGHPUT),
    "providers.rerank.batch_size": _neutral(_THROUGHPUT),
    "providers.graph": Unkeyed(_AT_INDEXING),
    "providers.graph.sqlite_path": _neutral(_WHERE),
    "providers.vector": Unkeyed(_AT_INDEXING),
    "providers.vector.sqlite_path": _neutral(_WHERE),
    "aws.region_name": _neutral(_WHERE),
    "aws.profile_name": _neutral(_WHERE),
    "aws.neptune": _neutral(_WHERE),
    "aws.opensearch": _neutral(_WHERE),
    "aws.s3": _neutral("says where caches and exports are stored"),
    "cache": _neutral("says how and how long caches are stored, not what they hold"),
    "retry": _neutral(_THROUGHPUT),
    "search": Unkeyed(_QUERY),
    "memory": Unkeyed(_QUERY),
    "evaluation": Unkeyed("read by run-eval only"),
    # It narrows auto through search.auto_routable_strategies, which counts.
    "mcp": _neutral("read by run-mcp-server only, to bound its tool calls"),
    "evaluation.outputs_directory": _neutral("says where run-eval writes"),
    "evaluation.save_detailed_results": _neutral("says what run-eval writes"),
    "logging": _neutral("telemetry"),
    "observability": _neutral("telemetry"),
    "custom_prompts.answer_generation_system": _QUERY_PROMPT,
    "custom_prompts.answer_generation_human": _QUERY_PROMPT,
    "custom_prompts.comprehensive_answer_system": _QUERY_PROMPT,
    "custom_prompts.comprehensive_answer_human": _QUERY_PROMPT,
    "custom_prompts.entity_extraction_system": _QUERY_PROMPT,
    "custom_prompts.entity_extraction_human": _QUERY_PROMPT,
    "custom_prompts.keywords_extraction_system": _QUERY_PROMPT,
    "custom_prompts.keywords_extraction_human": _QUERY_PROMPT,
    "custom_prompts.drift_primer_system": _QUERY_PROMPT,
    "custom_prompts.drift_primer_human": _QUERY_PROMPT,
    "custom_prompts.drift_hyde_system": _QUERY_PROMPT,
    "custom_prompts.drift_hyde_human": _QUERY_PROMPT,
    "custom_prompts.drift_action_system": _QUERY_PROMPT,
    "custom_prompts.drift_action_human": _QUERY_PROMPT,
    "custom_prompts.iterative_controller_system": _QUERY_PROMPT,
    "custom_prompts.iterative_controller_human": _QUERY_PROMPT,
    "custom_prompts.query_decomposition_system": _QUERY_PROMPT,
    "custom_prompts.query_decomposition_human": _QUERY_PROMPT,
    "custom_prompts.decomposition_step_system": _QUERY_PROMPT,
    "custom_prompts.decomposition_step_human": _QUERY_PROMPT,
    "custom_prompts.query_rewrite_system": _QUERY_PROMPT,
    "custom_prompts.query_rewrite_human": _QUERY_PROMPT,
    "custom_prompts.strategy_selection_system": _QUERY_PROMPT,
    "custom_prompts.strategy_selection_human": _QUERY_PROMPT,
    "custom_prompts.global_map_system": _QUERY_PROMPT,
    "custom_prompts.global_map_human": _QUERY_PROMPT,
    "custom_prompts.corpus_profile_system": _neutral("read by run-prompt-tuning only"),
    "custom_prompts.corpus_profile_human": _neutral("read by run-prompt-tuning only"),
    "custom_prompts.pairwise_comparison_system": Unkeyed("read by run-eval only"),
    "custom_prompts.pairwise_comparison_human": Unkeyed("read by run-eval only"),
    "custom_prompts.answer_span_extraction_system": Unkeyed("read by run-eval only"),
    "custom_prompts.answer_span_extraction_human": Unkeyed("read by run-eval only"),
    **{
        f"custom_prompts.{key}_{part}": Unkeyed("read by run-eval only")
        for key in (
            "pairwise_overlap",
            "pairwise_differential",
            "assertion_judgement",
            "closed_book_answer",
        )
        for part in ("system", "human")
    },
    **{
        f"custom_prompts.{key}_{part}": _neutral(
            "read by prepare-benchmark assertions only"
        )
        for key in (
            "benchmark_activity_context",
            "benchmark_questions",
            "benchmark_claims",
            "benchmark_assertions",
            "benchmark_validation",
        )
        for part in ("system", "human")
    },
}

# Keyed settings naming where a run reads or writes: they key the stage
# caches (another corpus or registry is another input) but not
# what a query answers, so the same evaluation hashes the same in every
# environment.
_LOCATION_FIELDS = frozenset(
    {
        "processing.document_parsing.source_directory",
        "processing.document_parsing.target_directory",
        "providers.doc_status.sqlite_path",
        "aws.dynamodb.table_name",
    }
)

# What an unclassified field gets: it counts for every stage, so forgetting
# to classify it costs a cache miss, never a stale hit.
_UNCLASSIFIED = _EVERY_CALL

# Fields added to a keyed (or answer-shaping) section whose default keeps
# the behaviour from before they existed: they count only once set to
# something else, so an upgrade re-keys no stage cache or extraction record
# and changes no recorded extraction settings or evaluation manifest. Take a
# field out when its default stops matching the earlier behaviour.
_KEYED_ONCE_SET = frozenset(
    {
        # Unkeyed, but in the evaluation manifest once set otherwise.
        "indexing.methodologies",
        "processing.graph_extraction.relationship_types",
        "processing.graph_extraction.relationship_patterns",
        "processing.graph_extraction.schema_mode",
        "processing.graph_extraction.schema_source.kind",
        "search.abstention.mode",
        "search.abstention.strip_marker",
        "search.abstention.on_insufficient",
        # The raptor methodology's settings (docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in), added with
        # it: at their defaults they change nothing a run or a query did.
        "indexing.opensearch.summary_nodes_index_prefix",
        "models.roles.summary_tree",
        "custom_prompts.tree_summary_system",
        "custom_prompts.tree_summary_human",
        *(f"indexing.summary_tree.{name}" for name in SummaryTreeConfig.model_fields),
        *(f"search.tree_search.{name}" for name in TreeSearchConfig.model_fields),
        "search.token_manager.type_budgets.summary",
        # Overrides of the chunker's and the translator's prompts, added
        # after them: unset, the prompts are the ones keyed before.
        "custom_prompts.text_chunking_system",
        "custom_prompts.text_chunking_human",
        "custom_prompts.text_translation_system",
        "custom_prompts.text_translation_human",
    }
)


def _at_earlier_default(path: str, value: Any) -> bool:
    """Whether ``value`` is the default of a ``_KEYED_ONCE_SET`` field."""
    return path in _KEYED_ONCE_SET and to_jsonable_python(value) == to_jsonable_python(
        _field_default(path)
    )


def classify(path: str) -> Keyed | Unkeyed | None:
    """The ``CONFIG_KEYING`` entry of the longest prefix of ``path`` (or of
    the path a recorded one moved to, ``_RECORDED_AS``), if any."""
    parts = _CURRENT_PATH.get(path, path).split(".")
    for end in range(len(parts), 0, -1):
        entry = CONFIG_KEYING.get(".".join(parts[:end]))
        if entry is not None:
            return entry
    return None


def shapes_answers(path: str) -> bool:
    """Whether the setting at ``path`` can change what a query answers or how
    it is scored: every setting but the unkeyed ones marked otherwise and
    ``_LOCATION_FIELDS``. An unclassified field counts."""
    rule = classify(path)
    if isinstance(rule, Unkeyed) and not rule.shapes_answers:
        return False
    return path not in _LOCATION_FIELDS


def answer_settings(config: Config) -> dict[str, Any]:
    """``{path: value}`` of every setting that can change a query's answer
    or score (:func:`shapes_answers`), a role's model resolved through its
    tier. Ingestion settings count too: an index built otherwise answers
    otherwise. Whether a guardrail screens the queries counts as well
    (``aws.bedrock.guardrail.enabled``): its id is a location, left out, but
    a guarded run blocks and masks what an unguarded one answers."""
    settings = {
        _RECORDED_AS.get(path, path): to_jsonable_python(_resolved(config, path, value))
        for path, value, _ in _leaves(config)
        if shapes_answers(path) and not _at_earlier_default(path, value)
    }
    settings["aws.bedrock.guardrail.enabled"] = config.aws.bedrock.guardrail.enabled
    return settings


def _block_on(config: Config, path: str) -> bool:
    """Whether the block at ``path`` and every block above it are on."""
    node: Any = config
    for name in path.split("."):
        node = getattr(node, name)
        if getattr(node, "enabled", None) is False:
            return False
    return True


# Path prefix -> whether the settings under it are in effect. Their switch
# fields (``kind``, ``enabled``) count either way.
_IN_EFFECT_WHEN: dict[str, Callable[[Config], bool]] = {
    # A role's model counts while the stage it serves runs.
    **{
        f"models.roles.{role}": functools.partial(_block_on, path=block)
        for role, block in {
            "translation": "processing.translation",
            "description_summary": (
                "processing.graph_extraction.description_summarization"
            ),
            "gleaning": "processing.gleaning",
            "claim_extraction": "processing.claim_extraction",
            "community_report": "graph.community_detection.report_generation",
        }.items()
    },
    "models.roles.chunking": lambda c: (
        c.processing.chunking.chunker_type is ChunkingStrategy.INTELLIGENT
    ),
    "providers.llm": lambda c: c.providers.llm.kind != BEDROCK,
    "aws.bedrock": lambda c: c.providers.llm.kind == BEDROCK,
    **{
        f"custom_prompts.{name}_{part}": functools.partial(_block_on, path=block)
        for name, block in {
            "description_summarization": (
                "processing.graph_extraction.description_summarization"
            ),
            "graph_refinement": "processing.gleaning",
            "claim_extraction": "processing.claim_extraction",
            "community_report": "graph.community_detection.report_generation",
            "text_translation": "processing.translation",
        }.items()
        for part in ("system", "human")
    },
    **{
        f"custom_prompts.text_chunking_{part}": lambda c: (
            c.processing.chunking.chunker_type is ChunkingStrategy.INTELLIGENT
        )
        for part in ("system", "human")
    },
}
_SWITCHES = frozenset({"enabled", "kind"})
# Settings every language model reads whatever its provider: the request's
# output cap (clamped to providers.llm's ceiling on a local model).
_EVERY_PROVIDER = frozenset({"models.default_max_output_tokens"})

# Settings moved to another path, keyed and recorded under the path they had
# before: an upgrade then re-keys no stage cache, re-extracts no document and
# changes no evaluation manifest for the move alone.
_RECORDED_AS: dict[str, str] = {
    "models.default_max_output_tokens": "aws.bedrock.default_max_output_tokens",
}
_CURRENT_PATH: dict[str, str] = {old: new for new, old in _RECORDED_AS.items()}

# Fields of a switchable block that other components read whatever its
# ``enabled`` flag says. With translation off, the prompts and the graph
# indexes' analyzers follow ``source_language`` (``index_language``) and the
# indexer maps its translated-text fields from the target languages;
# with reports off, community objects still list at most
# ``max_entities_per_report`` entity names.
_READ_WHEN_DISABLED = frozenset(
    {
        "processing.translation.source_language",
        "processing.translation.target_language",
        "processing.translation.additional_target_languages",
        "graph.community_detection.report_generation.max_entities_per_report",
    }
)


def _leaves(model: BaseModel, prefix: str = "") -> Iterator[tuple[str, Any, bool]]:
    """``(path, value, disabled)`` of every non-section field under ``model``.

    ``disabled``: the field lies in a block switched off with
    ``enabled: false`` (the switch itself excluded).
    """
    off = getattr(model, "enabled", None) is False
    for name in type(model).model_fields:
        value = getattr(model, name)
        path = f"{prefix}{name}"
        if isinstance(value, BaseModel):
            for leaf, leaf_value, disabled in _leaves(value, f"{path}."):
                yield leaf, leaf_value, disabled or off
        else:
            yield path, value, off and name != "enabled"


def _in_effect(config: Config, path: str, disabled: bool) -> bool:
    name = path.rsplit(".", 1)[-1]
    if disabled and path not in _READ_WHEN_DISABLED:
        return False
    if name in _SWITCHES or path in _EVERY_PROVIDER:
        return True
    parts = path.split(".")
    return all(
        _IN_EFFECT_WHEN.get(".".join(parts[:end]), lambda _c: True)(config)
        for end in range(1, len(parts) + 1)
    )


def _settings(config: Config, selected: Callable[[Keyed], bool]) -> dict[str, Any]:
    """``{path: value}`` of the keyed settings in effect that ``selected`` takes."""
    settings: dict[str, Any] = {}
    for path, value, disabled in _leaves(config):
        rule = classify(path) or _UNCLASSIFIED
        if isinstance(rule, Keyed) and selected(rule):
            value = _as_extracted(config, path, value)
            if _in_effect(config, path, disabled) and not _at_earlier_default(
                path, value
            ):
                settings[_RECORDED_AS.get(path, path)] = to_jsonable_python(
                    _resolved(config, path, value)
                )
    return settings


_SCHEMA_PREFIX = "processing.graph_extraction."
_SCHEMA_TYPE_FIELDS = frozenset(
    {f"{_SCHEMA_PREFIX}entity_types", f"{_SCHEMA_PREFIX}relationship_types"}
)
_SCHEMA_PATTERNS_FIELD = f"{_SCHEMA_PREFIX}relationship_patterns"
_SCHEMA_MODE_FIELD = f"{_SCHEMA_PREFIX}schema_mode"


def _type_as_extracted(item: EntityTypeItem | RelationshipTypeItem) -> Any:
    """A schema type as extraction sees it: its label and description.

    A string item as written (as every configuration keyed it before the
    mapping form); a mapping as the string that reads back as the same
    label and description, so a type written either way keys alike, or,
    when no string does (a label with a ':'), as its label and description.
    Never its ``iri``: indexing stores it, extraction does not see it.
    """
    if isinstance(item, str):
        return item
    text = f"{item.label}: {item.description}" if item.description else item.label
    spec = entity_type_spec(text)
    if spec is not None and (spec.label, spec.description) == (
        item.label,
        item.description,
    ):
        return text
    return {"label": item.label, "description": item.description}


def _pattern_as_extracted(item: RelationshipPatternItem) -> Any:
    """A pattern as written, a mapping as the string that reads back as the
    same pattern (else as its fields)."""
    if isinstance(item, str):
        return item
    text = str(item)
    # Always of the form; a relation with a '->' reads back otherwise.
    if RelationshipPattern.parse(text) == item:
        return text
    return item.model_dump()


def _as_extracted(config: Config, path: str, value: Any) -> Any:
    """A setting's value as it reaches extraction.

    Only the schema keys differ from their value: a type or pattern by what
    the prompt and the schema check read of it (``_type_as_extracted``,
    ``_pattern_as_extracted``), and ``schema_mode`` only where extraction
    applies it (``SchemaConformance.applies``): ``strict`` always, ``off``
    once relationship types are declared; with entity types alone ``off``
    and ``align`` extract alike and key as the default. What the schema
    changes only in what indexing stores (its IRIs, ``off`` against
    ``align`` with entity types alone) keys the indexing stage
    (``_schema_iris``), and its digest (``SchemaAnnotator.digest``) is
    recorded per index namespace once the stores hold its annotations.
    """
    if path in _SCHEMA_TYPE_FIELDS:
        return [_type_as_extracted(item) for item in value]
    if path == _SCHEMA_PATTERNS_FIELD:
        return [_pattern_as_extracted(item) for item in value]
    if (
        path == _SCHEMA_MODE_FIELD
        and value != "strict"
        and not config.processing.graph_extraction.relationship_types
    ):
        return _field_default(path)
    return value


_ROLES_PREFIX = "models.roles."


def _resolved(config: Config, path: str, value: Any) -> Any:
    """A setting's value as it takes effect: a role's model id resolved
    through its tier (``Config.model_id``)."""
    if path.startswith(_ROLES_PREFIX):
        return config.model_id(ModelRole(path.removeprefix(_ROLES_PREFIX)))
    return value


def _upstream_of(stage: PipelineStageType) -> Callable[[Keyed], bool]:
    cutoff = _STAGE_ORDER.index(stage)
    return lambda rule: rule.stage is None or _STAGE_ORDER.index(rule.stage) <= cutoff


def _digest(payload: Any) -> str:
    return compute_hash(
        json.dumps(payload, sort_keys=True, default=str), length=FINGERPRINT_LENGTH
    )


_HASH_READ_BYTES = 1 << 20


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_READ_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def is_excluded_source_path(path: Path, root: Path) -> bool:
    """Whether a file found under the corpus ``root`` is outside the corpus.

    Any component of its path below ``root`` that starts with ``.`` (a hidden
    file, ``.git``, a hidden directory) or is ``__pycache__`` excludes it.
    Discovery, the document loader and the corpus fingerprint share this rule,
    so a file is parsed exactly when it counts toward the fingerprint.
    """
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part.startswith(".") or part == "__pycache__" for part in parts)


def corpus_manifest_fingerprint(
    source_directory: str | Path,
    extensions: Iterable[str],
    exclude_directories: Iterable[str | Path] = (),
) -> str:
    """Fingerprint the corpus contents under ``source_directory``.

    Every file the pipeline could ingest contributes its path relative to the
    source root, its size, and a SHA-256 of its bytes. Modification times are
    deliberately not used: an ``aws s3 sync`` into a fresh container or a new
    checkout changes them without changing the content, which would turn every
    phase handoff into a cache miss.

    Args:
        source_directory: Corpus root.
        extensions: File suffixes the pipeline ingests (case-insensitive).
        exclude_directories: Directories the pipeline writes into (cache,
            parsed-JSON export), skipped so its own output never changes the
            fingerprint mid-run.

    Returns:
        A short hex digest; an empty or missing corpus has a fixed digest.
    """
    root = Path(source_directory).resolve()
    suffixes = {ext.lower() for ext in extensions}
    excluded = [Path(d).resolve() for d in exclude_directories]
    entries: list[str] = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (
                not path.is_file()
                or path.suffix.lower() not in suffixes
                or is_excluded_source_path(path, root)
                or any(path.resolve().is_relative_to(d) for d in excluded)
            ):
                continue
            entries.append(
                f"{relative.as_posix()}\t{path.stat().st_size}\t{_file_digest(path)}"
            )
    return compute_hash("\n".join(entries), length=FINGERPRINT_LENGTH)


def seed_corpus_fingerprint(
    corpus_fingerprint: str, seed_digests: Iterable[str] = ()
) -> str:
    """``corpus_fingerprint`` with the seed graphs' content folded in.

    ``seed_digests`` are the converted seeds' digests
    (``SeedConversion.digest``: their shard paths, so their names, and
    texts, so their filters and statements), in configuration order. With
    none it is ``corpus_fingerprint`` itself, so a run without seed graphs
    keeps its keys. A seed read from another location with the same content
    keys alike; an edited seed changes every stage's key.
    """
    digests = list(seed_digests)
    if not digests:
        return corpus_fingerprint
    return compute_hash(
        json.dumps({"corpus": corpus_fingerprint, "seed_graphs": digests}),
        length=FINGERPRINT_LENGTH,
    )


# --- extraction settings -----------------------------------------------------

# Bump when the record's meaning changes in a way its fields do not show (a
# setting removed whose old value shaped the output, say): every recorded
# set of extraction settings then reads as changed.
EXTRACTION_SETTINGS_VERSION = 1
# Values whose JSON is longer than this (prompt overrides, long entity-type
# lists) are recorded by digest, so the record stays small.
_RECORDED_VALUE_MAX_CHARS = 200


def _recorded(value: Any) -> Any:
    text = json.dumps(value, sort_keys=True, default=str)
    if len(text) <= _RECORDED_VALUE_MAX_CHARS:
        return value
    return f"sha256:{compute_hash(text)}"


def extraction_settings(config: Config) -> ExtractionSettings:
    """The settings that shape what a document produces, as a record.

    Every keyed field marked ``per_document`` and in effect, from the model
    knobs through claim extraction: models, prompts, entity types, chunking,
    translation, gleaning, resolution and claims. Not the corpus location,
    the registry or the switches that pick which documents run, and not the
    package version.
    """
    values = _settings(
        config,
        lambda rule: rule.per_document and _upstream_of(_P.CLAIM_EXTRACTION)(rule),
    )
    values.update(_language_input(config))
    return ExtractionSettings(
        version=EXTRACTION_SETTINGS_VERSION,
        values={path: _recorded(value) for path, value in sorted(values.items())},
    )


def _language_input(config: Config) -> dict[str, str]:
    """``{"index_language": ...}`` for a config whose translation settings
    alone do not say the language it is indexed in: translation off with a
    ``target_language`` other than ``source_language``, extracted in the
    target language before #605 and in the source language since. Keyed
    into its extraction settings and stage caches, a namespace extracted
    the earlier way re-extracts (or, where its analyzers differ, stops)
    instead of mixing languages, and a re-run of a pipeline id does not
    resume from target-language stage outputs. Empty for every other
    config, so its keys stay."""
    translation = config.processing.translation
    if translation.enabled or translation.source_language == (
        translation.target_language
    ):
        return {}
    return {"index_language": index_language(config)}


def extraction_fingerprint(config: Config) -> str:
    """Fingerprint of :func:`extraction_settings` (recorded per document)."""
    return extraction_settings(config).fingerprint


_MISSING = object()


def _field_default(path: str) -> Any:
    """The default of the field at ``path`` (a recorded path: see
    ``_RECORDED_AS``), or ``_MISSING``."""
    node: Any = _default_config()
    for name in _CURRENT_PATH.get(path, path).split("."):
        node = getattr(node, name, _MISSING)
        if node is _MISSING:
            return _MISSING
    return node


def _default_value(path: str) -> Any:
    node = _field_default(path)
    if node is _MISSING:
        return _MISSING
    node = _as_extracted(_default_config(), path, node)
    return _recorded(to_jsonable_python(_resolved(_default_config(), path, node)))


@functools.cache
def _default_config() -> Config:
    return Config()


def changed_settings(old: ExtractionSettings, new: ExtractionSettings) -> list[str]:
    """The paths whose value differs between two extraction-settings records.

    Compared field by field. A field only ``new`` records counts as
    unchanged while it holds its default (a field a newer version added,
    whose default keeps the earlier behaviour, or one that came into effect
    at its default); a field only ``old`` records counts as changed while it
    is still a configuration field (it went out of effect) and is ignored
    once removed from the configuration. Records of another
    ``EXTRACTION_SETTINGS_VERSION`` differ as a whole (``["version"]``).
    """
    if old.version != new.version:
        return ["version"]
    changed = []
    for path, value in new.values.items():
        before = old.values.get(path, _MISSING)
        if before is _MISSING:
            before = _default_value(path)
        if before != value:
            changed.append(path)
    for path in old.values.keys() - new.values.keys():
        if _default_value(path) is not _MISSING and _is_extraction_setting(path):
            changed.append(path)
    # Named as configured now (a moved setting is recorded by its old path).
    return sorted(_CURRENT_PATH.get(path, path) for path in changed)


def _is_extraction_setting(path: str) -> bool:
    """Whether ``path`` is keyed as one of the extraction settings (an
    earlier version's record may hold a field keyed otherwise since: the
    description summary settings)."""
    rule = classify(path) or _UNCLASSIFIED
    return (
        isinstance(rule, Keyed)
        and rule.per_document
        and _upstream_of(_P.CLAIM_EXTRACTION)(rule)
    )


# --- unit extraction records -------------------------------------------------

# A text unit's extraction record (``TextUnitExtraction``) is keyed by its
# text, so the settings that only shape the text (chunking, translation) are
# not inputs of it; the language its descriptions are written in is. Bump the
# version when the record format or its parsing rules change.
_RECORD_FORMAT_VERSION = 1


def _summary_settings(config: Config) -> dict[str, Any]:
    """The summary settings (``_SUMMARY_SETTINGS``) in effect, as a unit
    record's fingerprint held them when they were keyed at graph
    extraction."""
    return {
        path: value
        for path, value in _settings(config, lambda rule: True).items()
        if any(path == s or path.startswith(f"{s}.") for s in _SUMMARY_SETTINGS)
    }


def _summary_settings_as_keyed_before(config: Config) -> dict[str, Any]:
    """:func:`_summary_settings` at their defaults, the summary model
    resolved through the run's tiers: how a record's fingerprint held them
    from the release that stopped keying graph extraction with them until
    the current key (:func:`_summary_key`) left the tiers out, so a tier
    only the summary model reads re-keyed every record. A record stored
    under this form is found under it (:func:`legacy_unit_record_fingerprints`)."""
    baseline = config.model_copy(deep=True)
    default = _default_config()
    for path in _SUMMARY_SETTINGS:
        *parents, name = path.split(".")
        target, source = baseline, default
        for part in parents:
            target, source = getattr(target, part), getattr(source, part)
        value = getattr(source, name)
        if isinstance(value, BaseModel):
            value = value.model_copy(deep=True)
        setattr(target, name, value)
    return _summary_settings(baseline)


@functools.cache
def _summary_key() -> dict[str, Any]:
    """The summary part of every record's fingerprint: :func:`_summary_settings`
    of the default configuration, a constant. A record stored under the
    default summary settings, when they were keyed at graph extraction,
    keeps its key, and no summary setting, nor a tier only the summary model
    reads, asks a unit again. A record stored otherwise is found under its
    earlier key (:func:`legacy_unit_record_fingerprints`). (A document's
    extraction settings leave them out: ``changed_settings`` ignores them in
    an earlier record.)"""
    return _summary_settings(_default_config())


def unit_record_fingerprint(
    config: Config, *, gleaned: bool, claims: bool = False
) -> str:
    """Fingerprint the settings a text unit's extraction record depends on.

    The model knobs, graph extraction (entity types, limits, grounding,
    confidence threshold), gleaning when the record is ``gleaned``, claim
    extraction for a unit's ``claims`` record, the target language, and the
    prompt templates in effect: the built-in ones (code) as well as
    ``custom_prompts`` overrides, so editing a template re-keys the records
    it shaped (each hashed with its ``inlined_slots`` filled in). The
    extraction store keys a record by this fingerprint and
    the hash of the unit's text
    (``domain.ingestion.contributions.record_key``).
    """
    return _unit_record_fingerprint(config, gleaned, claims, _summary_key())


def legacy_unit_record_fingerprints(
    config: Config, *, gleaned: bool, claims: bool = False
) -> list[str]:
    """The fingerprints an earlier version stored the same record under,
    where they differ from :func:`unit_record_fingerprint`, in the order
    to look them up.

    While the description summary settings keyed graph extraction, a
    record's fingerprint held them at the run's values; then at their
    defaults resolved through the run's tiers
    (:func:`_summary_settings_as_keyed_before`); it holds them at the
    default configuration's now. A record stored under either earlier form
    is looked up under the fingerprint it was stored under too
    (``ExtractionStore.load``, which stores what it finds there under the
    current key).
    """
    current = unit_record_fingerprint(config, gleaned=gleaned, claims=claims)
    legacy: list[str] = []
    for summary in (
        _summary_settings_as_keyed_before(config),
        _summary_settings(config),
    ):
        key = _unit_record_fingerprint(config, gleaned, claims, summary)
        if key != current and key not in legacy:
            legacy.append(key)
    return legacy


def _unit_record_fingerprint(
    config: Config, gleaned: bool, claims: bool, summary: dict[str, Any]
) -> str:
    from unified_kg_rag.domain.prompts import (
        ClaimExtractionPrompt,
        GraphExtractionPrompt,
        GraphRefinementPrompt,
    )
    from unified_kg_rag.domain.prompts.base import fill_slots

    stages = {_P.GRAPH_EXTRACTION}
    prompts: list[Any] = [GraphExtractionPrompt]
    if gleaned:
        stages.add(_P.GLEANING)
        prompts.append(GraphRefinementPrompt)
    if claims:
        stages.add(_P.CLAIM_EXTRACTION)
        prompts.append(ClaimExtractionPrompt)
    settings = _settings(
        config,
        lambda rule: rule.per_document and (rule.stage is None or rule.stage in stages),
    )
    settings.update(summary)
    return _digest(
        {
            "settings": settings,
            "version": _RECORD_FORMAT_VERSION,
            "gleaned": gleaned,
            "claims": claims,
            # The language the prompts ask for (``index_language``); kept
            # under its old name, so a corpus whose index language is its
            # target language keeps its records.
            "target_language": index_language(config),
            "prompts": [
                compute_hash(
                    fill_slots(
                        resolved.system_prompt_template
                        + "\n"
                        + resolved.human_prompt_template,
                        prompt.inlined_slots(config),
                    )
                )
                for prompt, resolved in (
                    (prompt, prompt.resolve(config.custom_prompts))
                    for prompt in prompts
                )
            ],
        }
    )


# The format of a pre-chunk's boundary record: bump it when the record or the
# way the chunker reads it changes.
_BOUNDARY_FORMAT_VERSION = 1


def chunk_boundary_fingerprint(config: Config) -> str:
    """Fingerprint the settings a pre-chunk's boundary answer depends on.

    The settings of every model call and of chunking (the chunking model
    resolved through its tier, the size range the prompt asks for, the
    length unit), and the boundary prompt (code). The extraction store keys
    the answer by this fingerprint and the hash of the pre-chunk's text
    (``contributions.record_key``), so an unchanged pre-chunk is not sent
    to the model again (a reset, an edited document's untouched part).
    """
    from unified_kg_rag.domain.prompts import TextChunkingPrompt
    from unified_kg_rag.domain.prompts.base import fill_slots

    settings = _settings(
        config,
        lambda rule: rule.per_document
        and (rule.stage is None or rule.stage is _P.TEXT_CHUNKING),
    )
    resolved = TextChunkingPrompt.resolve(config.custom_prompts)
    return _digest(
        {
            "record": "chunk_boundaries",
            "version": _BOUNDARY_FORMAT_VERSION,
            "settings": settings,
            "prompt": compute_hash(
                fill_slots(
                    resolved.system_prompt_template
                    + "\n"
                    + resolved.human_prompt_template,
                    TextChunkingPrompt.inlined_slots(config),
                )
            ),
        }
    )


# The format of a reused description summary or community report: bump it
# when the record or the way its stage reads it changes.
_ANSWER_FORMAT_VERSION = 1


# The format of a text unit's translation record: bump it when the record or
# the way the translator reads it changes.
_TRANSLATION_FORMAT_VERSION = 1


def translation_fingerprint(config: Config) -> str:
    """Fingerprint the settings a text unit's translations depend on beside
    its text: the settings of every model call, the translation model
    resolved through its tier, the source language, the languages it is
    translated into and the translation prompt (code, or its
    ``custom_prompts`` override). The extraction store keys a unit's
    translations by it and the hash of the unit's text
    (``contributions.record_key``), so a reset or an edited document's
    untouched text is not translated again, and extraction, which reads the
    translation, reuses its records too.
    """
    from unified_kg_rag.domain.prompts import TextTranslationPrompt

    translation = config.processing.translation
    # The section's languages are keyed below; its switch is on whenever a
    # unit is translated.
    settings = {
        path: value
        for path, value in _settings(
            config,
            lambda rule: rule.per_document
            and (rule.stage is None or rule.stage is _P.TRANSLATION),
        ).items()
        if not path.startswith("processing.translation.")
    }
    resolved = TextTranslationPrompt.resolve(config.custom_prompts)
    return _digest(
        {
            "record": "translation",
            "version": _TRANSLATION_FORMAT_VERSION,
            "settings": settings,
            "source_language": translation.source_language.value,
            "languages": translation.translated_languages,
            "prompt": compute_hash(
                resolved.system_prompt_template + "\n" + resolved.human_prompt_template
            ),
        }
    )


def _answer_fingerprint(
    config: Config, record: str, role: ModelRole, prompt: Any, section: str | None
) -> str:
    """Fingerprint the settings a corpus-wide model answer depends on beside
    its input text: the settings of every model call, the ``role``'s model
    resolved through its tier, the config ``section`` the stage reads (none
    when the input text holds all of it the answer depends on), the index
    language (``index_language``) and ``prompt`` (code, or its
    ``custom_prompts`` override)."""
    from unified_kg_rag.domain.prompts.base import fill_slots

    keyed: dict[str, Any] = {}
    if section is not None:
        node: Any = config
        for name in section.split("."):
            node = getattr(node, name)
        keyed["section"] = to_jsonable_python(node)
    resolved = prompt.resolve(config.custom_prompts)
    return _digest(
        {
            "record": record,
            "version": _ANSWER_FORMAT_VERSION,
            "settings": _settings(
                config, lambda rule: rule.per_document and rule.stage is None
            ),
            "model": to_jsonable_python(config.model_id(role)),
            **keyed,
            "target_language": index_language(config),
            "prompt": compute_hash(
                fill_slots(
                    resolved.system_prompt_template
                    + "\n"
                    + resolved.human_prompt_template,
                    prompt.inlined_slots(config),
                )
            ),
        }
    )


def description_summary_fingerprint(config: Config) -> str:
    """Fingerprint the settings a description summary depends on beside the
    description (``_answer_fingerprint``). A summarized item records it, so
    a full build reuses a stored summary only when it was made the same way
    (``description_summarizer.reuse_summaries``)."""
    from unified_kg_rag.domain.prompts import DescriptionSummarizationPrompt

    return _answer_fingerprint(
        config,
        "description_summary",
        ModelRole.DESCRIPTION_SUMMARY,
        DescriptionSummarizationPrompt,
        "processing.graph_extraction.description_summarization",
    )


def community_report_fingerprint(config: Config) -> str:
    """Fingerprint the settings a community report depends on beside its
    input (``_answer_fingerprint``). The extraction store keys the report's
    answer by it and the input's text, so a full build that packs a
    community's input again does not ask the model again (a reset of an
    unchanged corpus). Not ``report_generation``: the input as packed
    reflects its packing budgets, and holds the report length the prompt
    asks for, so a budget change re-asks only the reports whose input it
    changed."""
    from unified_kg_rag.domain.prompts import CommunityReportPrompt

    return _answer_fingerprint(
        config,
        "community_report",
        ModelRole.COMMUNITY_REPORT,
        CommunityReportPrompt,
        None,
    )


def index_language(config: Config) -> str:
    """The language (its code) the corpus is indexed in: the target
    language when translation runs, the source language otherwise (the
    corpus is indexed as written).

    The one rule every component follows: text units are embedded and
    searched in it, extraction, gleaning, claims, description summaries,
    community reports and the summary tree are written in it, the graph
    indexes' text fields are analyzed for it, and a query without a
    requested language is processed into it."""
    translation = config.processing.translation
    language = (
        translation.target_language
        if translation.enabled
        else translation.source_language
    )
    return language.value


def tree_summary_fingerprint(config: Config) -> str:
    """Fingerprint the settings one summary of the summary tree depends on
    beside its children's texts: every model call's settings, the
    ``summary_tree`` role's model, the summary length, the index language
    and ``TreeSummaryPrompt`` (code, or its ``custom_prompts`` override). A
    node records the digest of its input under it (``input_key``), so a
    rebuild reuses a summary only when it was made the same way."""
    from unified_kg_rag.domain.prompts import TreeSummaryPrompt
    from unified_kg_rag.domain.prompts.base import fill_slots

    resolved = TreeSummaryPrompt.resolve(config.custom_prompts)
    return _digest(
        {
            "record": "tree_summary",
            "version": _ANSWER_FORMAT_VERSION,
            "settings": _settings(
                config, lambda rule: rule.per_document and rule.stage is None
            ),
            "model": to_jsonable_python(config.model_id(ModelRole.SUMMARY_TREE)),
            "max_summary_tokens": config.indexing.summary_tree.max_summary_tokens,
            # 2: the length hint follows the token counter and an over-long
            # summary is cut at a sentence end (summaries of 1 were mostly
            # cut mid-sentence).
            "length": 2,
            "language": index_language(config),
            "prompt": compute_hash(
                fill_slots(
                    resolved.system_prompt_template
                    + "\n"
                    + resolved.human_prompt_template,
                    TreeSummaryPrompt.inlined_slots(config),
                )
            ),
        }
    )


def summary_tree_fingerprint(config: Config) -> str:
    """Fingerprint what a summary tree is built with beside its leaves: the
    tree settings, the summaries' (``tree_summary_fingerprint``) and the
    embedding model and dimension. The build record keeps it, so a run
    whose fingerprint and leaf set are the stored ones builds nothing."""
    tree = config.indexing.summary_tree.model_dump(
        mode="json", exclude={"refresh", "max_leaves_in_memory"}
    )
    return _digest(
        {
            "record": "summary_tree",
            "tree": tree,
            "summaries": tree_summary_fingerprint(config),
            "embedding": [
                to_jsonable_python(config.embedding_model_id),
                config.models.embedding.dimension,
            ],
        }
    )


# --- stage cache keys --------------------------------------------------------


def stage_input_fingerprint(
    config: Config,
    stage_type: PipelineStageType,
    corpus_fingerprint: str | None = None,
) -> str:
    """Fingerprint the inputs that determine ``stage_type``'s output.

    The keyed settings in effect of the stage and every upstream stage
    (``CONFIG_KEYING``), the package version, and the corpus.

    Args:
        config: The active framework configuration.
        stage_type: Ingestion stage whose inputs are being fingerprinted.
        corpus_fingerprint: :func:`corpus_manifest_fingerprint` of the run's
            source corpus. Every stage consumes the corpus (directly or through
            its predecessors), so it is folded into every key. ``None`` leaves
            the key a function of the configuration only.

    Returns:
        A short hex digest, identical for two equal configurations and
        different as soon as any output-determining input changes.

    Raises:
        ConfigurationError: The indexing stage's key, with a schema source
            not loaded into the inline keys yet (its IRIs key the stage).

    Example:
        >>> fp = stage_input_fingerprint(Config(), PipelineStageType.TEXT_CHUNKING)
        >>> len(fp)
        12
    """
    payload: dict[str, Any] = {
        "settings": _settings(config, _upstream_of(stage_type)),
        "package_version": __version__,
        "corpus_manifest": corpus_fingerprint,
    }
    if _STAGE_ORDER.index(stage_type) >= _STAGE_ORDER.index(_P.TRANSLATION):
        payload.update(_language_input(config))
    if stage_type is _P.INDEXING and (iris := _schema_iris(config)):
        payload["schema_iris"] = iris
    return _digest(payload)


def _schema_iris(config: Config) -> list[list[str]]:
    """``[kind, type key, iri]`` of each IRI indexing writes: what indexing
    alone reads of the schema (its labels and mode are keyed already)."""
    schema = config.processing.graph_extraction.extraction_schema()
    if schema.mode == "off":
        return []
    return [
        [kind, schema_key(spec.label), spec.iri]
        for kind, specs in (
            ("entity", schema.entity_types),
            ("relationship", schema.relationship_types),
        )
        for spec in specs
        if spec.iri
    ]


def stage_cache_key(
    config: Config,
    stage_type: PipelineStageType,
    context_attr: str,
    corpus_fingerprint: str | None = None,
) -> str:
    """Build the cache key for one stage output.

    Args:
        config: The active framework configuration.
        stage_type: Ingestion stage that produces (or produced) the output.
        context_attr: ``PipelineContext`` attribute holding the output, e.g.
            "documents" or "resolved_entities".
        corpus_fingerprint: The run's corpus manifest fingerprint (see
            :func:`stage_input_fingerprint`).

    Returns:
        The context attribute suffixed with the stage's input fingerprint, so a
        configuration change reads as a cache miss instead of a stale hit.

    Example:
        >>> key = stage_cache_key(
        ...     Config(), PipelineStageType.DOCUMENT_LOADING, "documents"
        ... )
        >>> key.startswith("documents-")
        True
    """
    fingerprint = stage_input_fingerprint(config, stage_type, corpus_fingerprint)
    return f"{context_attr}-{fingerprint}"
