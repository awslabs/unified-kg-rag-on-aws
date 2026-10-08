# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input-aware cache keys for ingestion stage results.

A stage's cached output may only be reused when the inputs that PRODUCED it are
unchanged. Keying purely on the context attribute name ("documents",
"entities", ...) made every run with the same ``pipeline_id`` a cache hit, so a
changed prompt, model id, or chunking rule silently resumed from output the old
configuration produced — the only escapes were ``--force-rebuild``, a TTL
expiry, or a new ``pipeline_id``.

The key therefore carries a fingerprint of the stage's output-determining
inputs. Two properties matter:

* **Stable for an unchanged configuration**, so resume still works. The
  fingerprint is a hash over a sorted, JSON-canonicalized projection of named
  config paths — no timestamps, no object identity, no dict ordering.
* **Cumulative over the stage order.** A stage consumes its predecessors'
  output, so its result is stale if its OWN inputs changed *or* if any upstream
  stage's inputs changed. Fingerprinting only the stage's own subtree would let
  a chunking change invalidate ``text_units`` while leaving the entities
  extracted from the old chunks looking fresh.

Only output-determining paths are listed. Throughput knobs
(``max_concurrency``, ``chunk_concurrency``, ``batch_size``, ``max_attempts``)
are deliberately excluded: they change how long a stage takes, not what it
produces, and including them would discard an expensive cache on a tuning
change.

The corpus itself is an input too: the source path alone says nothing about
what the files contain, so with a fixed ``pipeline_id`` an edited, added, or
removed file would otherwise resume from the previous corpus's documents (and,
in incremental mode, its delta). The caller passes a corpus manifest
fingerprint (:func:`corpus_manifest_fingerprint`: relative path + size +
content hash per file), folded into every stage's key since parsing is the
first stage.

Known limitation: the built-in prompt templates in ``domain/prompts`` are code,
so editing one is not visible here — only ``custom_prompts`` overrides are.
Bump the ``pipeline_id`` or use ``--force-rebuild`` after editing a template.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from unified_kg_rag.domain.models import Config, PipelineStageType
from unified_kg_rag.domain.models.config import BedrockConfig

from .common import compute_hash

FINGERPRINT_LENGTH = 12

# Canonical ingestion order: PipelineStageType is declared in pipeline order.
_STAGE_ORDER: tuple[PipelineStageType, ...] = tuple(PipelineStageType)

# Config paths whose value determines a stage's OUTPUT. Each stage's model id
# lives inside its own subtree (e.g. `processing.graph_extraction`
# .extraction_model_id), so naming the subtree covers the model. Prompt
# overrides are named per stage rather than taking `custom_prompts` wholesale,
# so editing a retrieval-only prompt does not invalidate ingestion output.
_STAGE_INPUT_PATHS: dict[PipelineStageType, tuple[str, ...]] = {
    PipelineStageType.DOCUMENT_PARSING: (
        "processing.document_parsing",
        "processing.ignore_errors",
        "fixing",
        # LLM knobs that change generated content. Region, role and profile
        # routing are excluded: they change WHERE the call goes, not its output.
        "aws.bedrock.effort",
        "aws.bedrock.fast_effort",
        "aws.bedrock.enable_1m_context",
        "aws.bedrock.guardrail",
        # The per-request output cap truncates long generations, and per-model
        # overrides change a model's capabilities (output cap, effort support).
        "aws.bedrock.default_max_output_tokens",
        "aws.bedrock.model_overrides",
    ),
    PipelineStageType.DOCUMENT_LOADING: ("processing.deduplicate",),
    PipelineStageType.TEXT_CHUNKING: ("processing.chunking",),
    PipelineStageType.TRANSLATION: ("processing.translation",),
    PipelineStageType.GRAPH_EXTRACTION: (
        "processing.graph_extraction",
        "custom_prompts.graph_extraction_system",
        "custom_prompts.graph_extraction_human",
        "custom_prompts.description_summarization_system",
        "custom_prompts.description_summarization_human",
    ),
    PipelineStageType.GLEANING: (
        "processing.gleaning",
        "custom_prompts.graph_refinement_system",
        "custom_prompts.graph_refinement_human",
    ),
    PipelineStageType.GRAPH_RESOLUTION: (
        "processing.resolution_method",
        "processing.similarity_threshold",
    ),
    PipelineStageType.CLAIM_EXTRACTION: (
        "processing.claim_extraction",
        "custom_prompts.claim_extraction_system",
        "custom_prompts.claim_extraction_human",
    ),
    # Claim resolution reuses the resolution knobs already folded in upstream.
    PipelineStageType.CLAIM_RESOLUTION: (),
    # Graph analysis caches the resolved entities and relationships it passes
    # through; `graph.analysis` only shapes centrality and statistics, which
    # no later stage reads (visualization recomputes them), so it is not an
    # output-determining input of this stage or any later one.
    PipelineStageType.GRAPH_ANALYSIS: (),
    PipelineStageType.COMMUNITY_DETECTION: (
        "graph.community_detection",
        "custom_prompts.community_report_system",
        "custom_prompts.community_report_human",
    ),
    PipelineStageType.INDEXING: ("indexing",),
}


# Paths fingerprinted by a derived value instead of the raw attribute.
# "aws.bedrock.effort" (the deprecated name of default_effort) stands for the
# default tier's effort, so an unchanged configuration keeps its cache key.
_DERIVED_PATHS: dict[str, Callable[[Config], Any]] = {
    "aws.bedrock.effort": lambda config: config.aws.bedrock.default_effort,
}


def _resolve_path(config: Config, path: str) -> Any:
    if path in _DERIVED_PATHS:
        return _DERIVED_PATHS[path](config)
    node: Any = config
    for attribute in path.split("."):
        node = getattr(node, attribute)
    return node


def _canonicalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    return value


# Paths added after the key format shipped: folded in only when set away from
# their default, so existing caches stay valid on upgrade.
_OMIT_WHEN_DEFAULT: dict[str, Any] = {
    path: _canonicalize(
        BedrockConfig.model_fields[path.rsplit(".", 1)[1]].get_default(
            call_default_factory=True
        )
    )
    for path in (
        "aws.bedrock.fast_effort",
        "aws.bedrock.default_max_output_tokens",
        "aws.bedrock.model_overrides",
    )
}


# Fields inside a fingerprinted subtree that do not shape stage output and are
# dropped from it: ``source_scope`` only scopes incremental deletion in the
# registry (and the corpus itself is fingerprinted by its manifest).
_NON_OUTPUT_SUBFIELDS: dict[str, frozenset[str]] = {
    "processing.document_parsing": frozenset({"source_scope"}),
}


def _input_paths_through(stage_type: PipelineStageType) -> list[str]:
    """Return the stage's own input paths plus every upstream stage's."""
    cutoff = _STAGE_ORDER.index(stage_type) + 1

    paths: list[str] = []
    for stage in _STAGE_ORDER[:cutoff]:
        paths.extend(_STAGE_INPUT_PATHS.get(stage, ()))
    return paths


_HASH_READ_BYTES = 1 << 20


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_READ_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


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
                or any(part.startswith(".") for part in relative.parts)
                or any(path.resolve().is_relative_to(d) for d in excluded)
            ):
                continue
            entries.append(
                f"{relative.as_posix()}\t{path.stat().st_size}\t{_file_digest(path)}"
            )
    return compute_hash("\n".join(entries), length=FINGERPRINT_LENGTH)


def stage_input_fingerprint(
    config: Config,
    stage_type: PipelineStageType,
    corpus_fingerprint: str | None = None,
) -> str:
    """Fingerprint the configuration that determines ``stage_type``'s output.

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
    projection: list[tuple[str, Any]] = []
    for path in _input_paths_through(stage_type):
        value = _canonicalize(_resolve_path(config, path))
        if path in _NON_OUTPUT_SUBFIELDS and isinstance(value, dict):
            value = {
                k: v for k, v in value.items() if k not in _NON_OUTPUT_SUBFIELDS[path]
            }
        if path in _OMIT_WHEN_DEFAULT and value == _OMIT_WHEN_DEFAULT[path]:
            continue
        projection.append((path, value))
    if corpus_fingerprint is not None:
        projection.append(("corpus_manifest", corpus_fingerprint))
    payload = json.dumps(projection, sort_keys=True, default=str)
    return compute_hash(payload, length=FINGERPRINT_LENGTH)


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
