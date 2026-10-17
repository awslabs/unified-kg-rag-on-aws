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
  and the Bedrock ones only while they are; a guardrail only while it guards
  ingestion models; a stage's prompt overrides only while the stage runs
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
first stage.
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
    ModelPurpose,
    ModelRole,
    PipelineStageType,
)
from unified_kg_rag.domain.models.provider_config import BEDROCK

from .common import compute_hash

FINGERPRINT_LENGTH = 12

# Canonical ingestion order: PipelineStageType is declared in pipeline order.
_STAGE_ORDER: tuple[PipelineStageType, ...] = tuple(PipelineStageType)
_P = PipelineStageType


@dataclass(frozen=True)
class Keyed:
    """A setting that shapes the output of ``stage`` and every later stage.

    ``stage`` ``None``: it shapes every model call, so every stage's output.
    ``per_document`` ``False``: it says where a run reads or writes, or which
    documents it runs, not what a document produces, so it keys the run's
    stage caches but is not one of the extraction settings.
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


def _keyed_at(stage: PipelineStageType, *paths: str, **kwargs: Any) -> dict:
    return dict.fromkeys(paths, Keyed(stage, **kwargs))


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
    "providers.llm.max_attempts": _neutral(_THROUGHPUT),
    "aws.bedrock.default_max_output_tokens": _EVERY_CALL,
    "aws.bedrock.model_overrides": _EVERY_CALL,
    "aws.bedrock.enable_1m_context": _EVERY_CALL,
    "aws.bedrock.default_effort": _EVERY_CALL,
    "aws.bedrock.fast_effort": _EVERY_CALL,
    "aws.bedrock.ingestion_effort": _EVERY_CALL,
    "aws.bedrock.region_name": _neutral(_WHERE),
    "aws.bedrock.assumed_role_arn": _neutral(_WHERE),
    "aws.bedrock.enable_global_profile": _neutral(_WHERE),
    "aws.bedrock.transient_retry": _neutral(_THROUGHPUT),
    "aws.bedrock.guardrail": _EVERY_CALL,
    "aws.bedrock.guardrail.trace": _neutral(
        "adds trace details to a response, not to what the model returns"
    ),
    "aws.bedrock.guardrail.check_input": Unkeyed("screens the user query only"),
    "aws.bedrock.guardrail.check_output": Unkeyed("screens the answer only"),
    # Output fixing repairs malformed model output in every stage.
    "fixing": _EVERY_CALL,
    # --- models: each role keys the stage it runs in, by its resolved id ----
    "models.default": Unkeyed(_TIER),
    "models.fast": Unkeyed(_TIER),
    "models.roles.fixing": _EVERY_CALL,
    "models.roles.chunking": Keyed(_P.TEXT_CHUNKING),
    "models.roles.translation": Keyed(_P.TRANSLATION),
    "models.roles.graph_extraction": Keyed(_P.GRAPH_EXTRACTION),
    "models.roles.description_summary": Keyed(_P.GRAPH_EXTRACTION),
    "models.roles.gleaning": Keyed(_P.GLEANING),
    "models.roles.claim_extraction": Keyed(_P.CLAIM_EXTRACTION),
    "models.roles.community_report": Keyed(_P.COMMUNITY_DETECTION),
    "models.roles.query_translation": Unkeyed(_QUERY),
    "models.roles.entity_extraction": Unkeyed(_QUERY),
    "models.roles.strategy_selection": Unkeyed(_QUERY),
    "models.roles.query_rewrite": Unkeyed(_QUERY),
    "models.roles.answer_generation": Unkeyed(_QUERY),
    "models.roles.community_relevance": Unkeyed(_QUERY),
    "models.roles.global_map": Unkeyed(_QUERY),
    "models.roles.global_reduce": Unkeyed(_QUERY),
    "models.roles.drift_primer": Unkeyed(_QUERY),
    "models.roles.drift_action": Unkeyed(_QUERY),
    "models.roles.evaluation": Unkeyed("read by run-eval only"),
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
        "aws.dynamodb.enabled",
        "aws.dynamodb.table_name",
        per_document=False,
    ),
    "aws.dynamodb.create_table_if_missing": _neutral("provisions the table"),
    "aws.dynamodb.billing_mode": _neutral("provisions the table"),
    # --- the per-document stages -------------------------------------------
    "processing.chunking": Keyed(_P.TEXT_CHUNKING),
    "processing.translation": Keyed(_P.TRANSLATION),
    **_keyed_at(
        _P.GRAPH_EXTRACTION,
        "processing.graph_extraction",
        "custom_prompts.graph_extraction_system",
        "custom_prompts.graph_extraction_human",
        "custom_prompts.description_summarization_system",
        "custom_prompts.description_summarization_human",
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
    "processing.max_attempts": _neutral(_THROUGHPUT),
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
    # --- indexing, stores and query time -----------------------------------
    "indexing.community_drift_threshold": Unkeyed(_AT_INDEXING),
    "indexing.lease_seconds": _neutral(
        "keeps concurrent runs apart, never changes what a stage produces"
    ),
    "indexing.max_failure_rate": _neutral(
        "fails a stage or the run, never changes what a stage produces"
    ),
    "indexing.neptune": Unkeyed(_AT_INDEXING),
    "indexing.opensearch": Unkeyed(_AT_INDEXING),
    "indexing.opensearch.refresh_after_batch": _neutral(_THROUGHPUT),
    "indexing.opensearch.persist_embedding_cache": _neutral(_THROUGHPUT),
    "indexing.opensearch.embedding_cache_s3_key": _neutral(_WHERE),
    "providers.embedding": Unkeyed(_AT_INDEXING),
    "providers.embedding.base_url": _neutral(_WHERE),
    "providers.embedding.api_key": _neutral(_WHERE),
    "providers.embedding.timeout_seconds": _neutral(_THROUGHPUT),
    "providers.embedding.max_attempts": _neutral(_THROUGHPUT),
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
    "search": Unkeyed(_QUERY),
    "memory": Unkeyed(_QUERY),
    "evaluation": Unkeyed("read by run-eval only"),
    "evaluation.outputs_directory": _neutral("says where run-eval writes"),
    "evaluation.save_detailed_results": _neutral("says what run-eval writes"),
    "logging": _neutral("telemetry"),
    "observability": _neutral("telemetry"),
    "custom_prompts.answer_generation_system": _QUERY_PROMPT,
    "custom_prompts.answer_generation_human": _QUERY_PROMPT,
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
}

# Keyed settings naming where a run reads or writes: they key the stage
# caches (another corpus, registry or guardrail is another input) but not
# what a query answers, so the same evaluation hashes the same in every
# environment.
_LOCATION_FIELDS = frozenset(
    {
        "processing.document_parsing.source_directory",
        "processing.document_parsing.target_directory",
        "providers.doc_status.sqlite_path",
        "aws.dynamodb.table_name",
        "aws.bedrock.guardrail.identifier",
        "aws.bedrock.guardrail.version",
    }
)

# What an unclassified field gets: it counts for every stage, so forgetting
# to classify it costs a cache miss, never a stale hit.
_UNCLASSIFIED = _EVERY_CALL


def classify(path: str) -> Keyed | Unkeyed | None:
    """The ``CONFIG_KEYING`` entry of the longest prefix of ``path``, if any."""
    parts = path.split(".")
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
    otherwise."""
    return {
        path: to_jsonable_python(_resolved(config, path, value))
        for path, value, _ in _leaves(config)
        if shapes_answers(path)
    }


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
            "fixing": "fixing",
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
    "aws.bedrock.guardrail": lambda c: c.aws.bedrock.guardrail.applies_to(
        ModelPurpose.INGESTION
    ),
    **{
        f"custom_prompts.{name}_{part}": functools.partial(_block_on, path=block)
        for name, block in {
            "description_summarization": (
                "processing.graph_extraction.description_summarization"
            ),
            "graph_refinement": "processing.gleaning",
            "claim_extraction": "processing.claim_extraction",
            "community_report": "graph.community_detection.report_generation",
        }.items()
        for part in ("system", "human")
    },
}
_SWITCHES = frozenset({"enabled", "kind"})

# Fields of a switchable block that other components read whatever its
# ``enabled`` flag says. With translation off, extraction, gleaning, claim
# and description prompts still take ``target_language`` and the indexer
# picks its analyzers and translated-text fields from the three languages;
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
    if name in _SWITCHES:
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
            if _in_effect(config, path, disabled):
                settings[path] = to_jsonable_python(_resolved(config, path, value))
    return settings


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
    return ExtractionSettings(
        version=EXTRACTION_SETTINGS_VERSION,
        values={path: _recorded(value) for path, value in sorted(values.items())},
    )


def extraction_fingerprint(config: Config) -> str:
    """Fingerprint of :func:`extraction_settings` (recorded per document)."""
    return extraction_settings(config).fingerprint


_MISSING = object()


def _default_value(path: str) -> Any:
    node: Any = _default_config()
    for name in path.split("."):
        node = getattr(node, name, _MISSING)
        if node is _MISSING:
            return _MISSING
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
        if _default_value(path) is not _MISSING:
            changed.append(path)
    return sorted(changed)


# --- unit extraction records -------------------------------------------------

# A text unit's extraction record (``TextUnitExtraction``) is keyed by its
# text, so the settings that only shape the text (chunking, translation) are
# not inputs of it; the language its descriptions are written in is. Bump the
# version when the record format or its parsing rules change.
_RECORD_FORMAT_VERSION = 1


def unit_record_fingerprint(
    config: Config, *, gleaned: bool, claims: bool = False
) -> str:
    """Fingerprint the settings a text unit's extraction record depends on.

    The model knobs, graph extraction (entity types, limits, grounding,
    confidence threshold), gleaning when the record is ``gleaned``, claim
    extraction for a unit's ``claims`` record, the target language, and the
    prompt templates in effect: the built-in ones (code) as well as
    ``custom_prompts`` overrides, so editing a template re-keys the records
    it shaped. The extraction store keys a record by this fingerprint and
    the hash of the unit's text
    (``domain.ingestion.contributions.record_key``).
    """
    from unified_kg_rag.domain.prompts import (
        ClaimExtractionPrompt,
        GraphExtractionPrompt,
        GraphRefinementPrompt,
    )

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
    return _digest(
        {
            "settings": settings,
            "version": _RECORD_FORMAT_VERSION,
            "gleaned": gleaned,
            "claims": claims,
            "target_language": config.processing.translation.target_language.value,
            "prompts": [
                compute_hash(
                    resolved.system_prompt_template
                    + "\n"
                    + resolved.human_prompt_template
                )
                for resolved in (
                    prompt.resolve(config.custom_prompts) for prompt in prompts
                )
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

    Example:
        >>> fp = stage_input_fingerprint(Config(), PipelineStageType.TEXT_CHUNKING)
        >>> len(fp)
        12
    """
    return _digest(
        {
            "settings": _settings(config, _upstream_of(stage_type)),
            "package_version": __version__,
            "corpus_manifest": corpus_fingerprint,
        }
    )


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
