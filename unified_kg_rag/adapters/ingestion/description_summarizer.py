# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-merge LLM re-summarization of over-long descriptions (adapters layer).

The pure merge functions (``domain/ingestion/merge/merger.py`` and the resolver
``_merge_descriptions``) only *concatenate* descriptions — fast, deterministic,
and used in property tests. That leaves a frequently-mentioned entity's
description growing unbounded (one fragment per supporting chunk), which bloats
prompts/embeddings and degrades quality.

This adapter runs the LLM summarization pass that closes the parity gap with MS
GraphRAG ``summarize_descriptions`` and LightRAG
``_handle_entity_relation_summary``. It lives here (not in the domain merge
functions) because it needs Bedrock/LangChain, which the domain layer forbids.

Only descriptions whose estimated token count exceeds the configured threshold
are sent to the LLM — cheap entities skip it entirely. On any LLM failure the
original concatenated description is kept (graceful degradation), and every
non-description field is left untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import boto3
from langchain_core.output_parsers import StrOutputParser

from unified_kg_rag.adapters.aws import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.adapters.aws.token_counter import estimate_token_count
from unified_kg_rag.domain.models import Config, Entity, Relationship
from unified_kg_rag.domain.prompts import DescriptionSummarizationPrompt
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import BatchProcessor

if TYPE_CHECKING:
    from unified_kg_rag.domain.models.config import DescriptionSummarizationConfig

logger = get_logger(__name__)


class DescriptionSummarizer:
    """Re-summarize entity/relationship descriptions that exceed a token budget.

    Built lazily-chain-free: the LLM chain is assembled in ``__init__`` via the
    shared ``setup_chain`` helper (same pattern as ``GraphExtractor`` /
    ``GraphGleaner``), and the LLM fan-out runs through ``BatchProcessor`` for
    consistency with the rest of the ingestion pipeline.
    """

    def __init__(
        self, config: Config, boto_session: boto3.Session | None = None
    ) -> None:
        self.config = config
        self.summarization_config: DescriptionSummarizationConfig = (
            self.config.processing.graph_extraction.description_summarization
        )
        self.target_language = self.config.processing.translation.target_language.value
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )
        self.factory = BedrockLanguageModelFactory(
            config=self.config,
            boto_session=self.boto_session,
            region_name=self.config.aws.bedrock.region_name,
        )
        self.batch_processor = BatchProcessor()
        self.summarizer = setup_chain(
            factory=self.factory,
            model_id=self.summarization_config.summary_model_id,
            prompt_class=DescriptionSummarizationPrompt,
            parser=StrOutputParser(),
            custom_prompts=self.config.custom_prompts,
        )

    def _exceeds_threshold(self, description: str | None) -> bool:
        if not description:
            return False
        return (
            estimate_token_count(description)
            > self.summarization_config.force_summary_threshold_tokens
        )

    def summarize_entities(self, entities: list[Entity]) -> list[Entity]:
        """Re-summarize entity descriptions over the threshold (in place)."""
        if not self.summarization_config.enabled or not entities:
            return entities
        over_threshold = [e for e in entities if self._exceeds_threshold(e.description)]
        if not over_threshold:
            return entities
        summaries = self._summarize_many(
            names=[e.name for e in over_threshold],
            descriptions=[e.description or "" for e in over_threshold],
        )
        for entity, summary in zip(over_threshold, summaries, strict=True):
            if summary:
                entity.description = summary
        return entities

    def summarize_relationships(
        self, relationships: list[Relationship]
    ) -> list[Relationship]:
        """Re-summarize relationship descriptions over the threshold (in place)."""
        if not self.summarization_config.enabled or not relationships:
            return relationships
        over_threshold = [
            r for r in relationships if self._exceeds_threshold(r.description)
        ]
        if not over_threshold:
            return relationships
        summaries = self._summarize_many(
            names=[
                f"{r.source_name} -> {r.target_name} ({r.type})" for r in over_threshold
            ],
            descriptions=[r.description or "" for r in over_threshold],
        )
        for rel, summary in zip(over_threshold, summaries, strict=True):
            if summary:
                rel.description = summary
        return relationships

    def _build_inputs(
        self, names: list[str], descriptions: list[str]
    ) -> list[dict[str, str]]:
        return [
            {
                "entity_name": name,
                "descriptions": description,
                "max_summary_tokens": str(self.summarization_config.max_summary_tokens),
                "language": self.target_language,
                "target_language": self.target_language,
            }
            for name, description in zip(names, descriptions, strict=True)
        ]

    def _summarize_many(
        self, names: list[str], descriptions: list[str]
    ) -> list[str | None]:
        """Run the LLM over each over-threshold item; keep order, degrade safely.

        Returns one entry per input: the summary string, or ``None`` when the
        LLM failed for that item (the caller then keeps the concatenated text).
        """
        inputs = self._build_inputs(names, descriptions)
        logger.info(
            "Summarizing %s over-threshold descriptions (threshold %s tokens)",
            len(inputs),
            self.summarization_config.force_summary_threshold_tokens,
        )

        try:
            results = self.batch_processor.execute_with_fallback(
                items_to_process=inputs,
                prepare_inputs_func=lambda chunk: chunk,
                batch_func=self.summarizer.batch,
                sequential_func=self.summarizer.invoke,
                task_name="Description Summarization",
                run_config=self.config.processing.model_dump(),
                show_progress=False,
            )
        except Exception as e:
            # The whole pass failed (not a per-item error): keep all originals.
            logger.warning(
                "Description summarization failed; keeping concatenated "
                "descriptions: %s",
                e,
            )
            return [None] * len(inputs)

        summaries: list[str | None] = []
        for name, result in zip(names, results, strict=True):
            summary = self._coerce_summary(result)
            if not summary:
                logger.warning(
                    "Empty/failed summary for '%s'; keeping concatenated description",
                    name,
                )
            summaries.append(summary)
        return summaries

    @staticmethod
    def _coerce_summary(result: object) -> str | None:
        """Normalize a chain result into a clean summary string (or None).

        ``BatchProcessor`` inserts an empty ``{}`` (not a string) for any item
        whose per-item LLM call failed during the sequential fallback. We must
        return ``None`` for such non-string results so the caller keeps the
        concatenated original — ``str({})`` would otherwise overwrite the
        description with the literal ``"{}"``.
        """
        if not isinstance(result, str):
            return None
        text = result.strip()
        return text or None
