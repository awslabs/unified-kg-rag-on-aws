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
(``max_concurrency``, ``chunk_concurrency``, ``batch_size``, ``max_retries``)
are deliberately excluded: they change how long a stage takes, not what it
produces, and including them would discard an expensive cache on a tuning
change.

Known limitation: the built-in prompt templates in ``domain/prompts`` are code,
so editing one is not visible here — only ``custom_prompts`` overrides are.
Bump the ``pipeline_id`` or use ``--force-rebuild`` after editing a template.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel

from unified_kg_rag.domain.models import Config, PipelineStageType

from ..logging import get_logger
from .common import compute_hash

logger = get_logger(__name__)

FINGERPRINT_LENGTH = 12

# Canonical ingestion order (mirrors DataIngestionPipeline.STAGE_CLASSES).
_STAGE_ORDER: tuple[PipelineStageType, ...] = (
    PipelineStageType.DOCUMENT_PARSING,
    PipelineStageType.DOCUMENT_LOADING,
    PipelineStageType.TEXT_CHUNKING,
    PipelineStageType.TRANSLATION,
    PipelineStageType.GRAPH_EXTRACTION,
    PipelineStageType.GLEANING,
    PipelineStageType.GRAPH_RESOLUTION,
    PipelineStageType.CLAIM_EXTRACTION,
    PipelineStageType.CLAIM_RESOLUTION,
    PipelineStageType.GRAPH_ANALYSIS,
    PipelineStageType.COMMUNITY_DETECTION,
    PipelineStageType.INDEXING,
)

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
        "aws.bedrock.enable_1m_context",
        "aws.bedrock.guardrail",
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
    PipelineStageType.GRAPH_ANALYSIS: ("graph.analysis",),
    PipelineStageType.COMMUNITY_DETECTION: (
        "graph.community_detection",
        "custom_prompts.community_report_system",
        "custom_prompts.community_report_human",
    ),
    PipelineStageType.INDEXING: ("indexing",),
}


def _resolve_path(config: Config, path: str) -> Any:
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


def _input_paths_through(stage_type: PipelineStageType) -> list[str]:
    """Return the stage's own input paths plus every upstream stage's."""
    try:
        cutoff = _STAGE_ORDER.index(stage_type) + 1
    except ValueError:
        # A stage added without extending _STAGE_ORDER must not silently get a
        # narrower fingerprint than its predecessors, so fold in everything.
        logger.warning(
            "Stage '%s' is not in the canonical cache-key stage order; "
            "fingerprinting all known inputs",
            stage_type,
        )
        cutoff = len(_STAGE_ORDER)

    paths: list[str] = []
    for stage in _STAGE_ORDER[:cutoff]:
        paths.extend(_STAGE_INPUT_PATHS.get(stage, ()))
    return paths


def stage_input_fingerprint(config: Config, stage_type: PipelineStageType) -> str:
    """Fingerprint the configuration that determines ``stage_type``'s output.

    Args:
        config: The active framework configuration.
        stage_type: Ingestion stage whose inputs are being fingerprinted.

    Returns:
        A short hex digest, identical for two equal configurations and
        different as soon as any output-determining input changes.

    Example:
        >>> fp = stage_input_fingerprint(Config(), PipelineStageType.TEXT_CHUNKING)
        >>> len(fp)
        12
    """
    projection = [
        (path, _canonicalize(_resolve_path(config, path)))
        for path in _input_paths_through(stage_type)
    ]
    payload = json.dumps(projection, sort_keys=True, default=str)
    return compute_hash(payload, length=FINGERPRINT_LENGTH)


def stage_cache_key(
    config: Config, stage_type: PipelineStageType, context_attr: str
) -> str:
    """Build the cache key for one stage output.

    Args:
        config: The active framework configuration.
        stage_type: Ingestion stage that produces (or produced) the output.
        context_attr: ``PipelineContext`` attribute holding the output, e.g.
            "documents" or "resolved_entities".

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
    return f"{context_attr}-{stage_input_fingerprint(config, stage_type)}"
