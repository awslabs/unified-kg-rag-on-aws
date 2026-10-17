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
are sent to the LLM — cheap entities skip it entirely. A description longer
than one call's input budget is summarized map-reduce, as both references do:
its lines are packed into budget-sized groups, each group is summarized, and
the joined summaries are summarized again until they fit one call. On any LLM
failure the original concatenated description is kept (graceful degradation),
and every non-description field is left untouched.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, TypeVar

from langchain_core.output_parsers import StrOutputParser

from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.llm.retry import transient_error_predicate
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config, Entity, ModelPurpose, Relationship
from unified_kg_rag.domain.prompts import DescriptionSummarizationPrompt
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import compute_hash
from unified_kg_rag.shared.utils.langchain import BatchProcessor

if TYPE_CHECKING:
    from unified_kg_rag.domain.models.config import DescriptionSummarizationConfig

logger = get_logger(__name__)

# Input budget of one summarization call (the descriptions, not the prompt),
# at most: a summary model with a smaller context window gets half of what
# the window leaves beside the summary (the rest covers the prompt and the
# token estimate's error), never less than _MIN_INPUT_TOKENS. A longer
# description is summarized in budget-sized groups first.
MAX_INPUT_TOKENS = 12_000
_MIN_INPUT_TOKENS = 1_000
# Map rounds before the final call. Each round shrinks a description by about
# the input budget / max_summary_tokens, so two rounds already cover millions
# of tokens; the cap only bounds a model that will not shorten its input.
_MAX_MAP_ROUNDS = 4


# Attribute of a summarized description: the hash of the description it was
# made from, so a run that derives the same description again keeps the
# summary instead of summarizing it again (see reuse_summaries).
SUMMARIZED_FROM_ATTRIBUTE = "summarized_from"

_Summarized = TypeVar("_Summarized", Entity, Relationship)


def _summarized(item: Entity | Relationship, summary: str) -> None:
    item.attributes = {
        **(item.attributes or {}),
        SUMMARIZED_FROM_ATTRIBUTE: compute_hash(item.description or ""),
    }
    item.description = summary


def needs_summary(config: Config, description: str | None) -> bool:
    """Whether ``description`` is over the summarization threshold (with
    summarization on), so a summarizer is worth building."""
    summarization = config.processing.graph_extraction.description_summarization
    return (
        summarization.enabled
        and bool(description)
        and estimate_token_count(description or "")
        > summarization.force_summary_threshold_tokens
    )


def reuse_summaries(
    items: Sequence[_Summarized], stored: Sequence[_Summarized]
) -> None:
    """Give each item the stored summary made from its description (in place).

    An item whose description equals the one its stored copy's summary was
    made from takes that summary and its ``summarized_from`` mark, so it is
    not summarized again (an LLM call per touched item on every run).
    """
    by_id = {item.id: item for item in stored}
    for item in items:
        previous = by_id.get(item.id)
        source = (
            (previous.attributes or {}).get(SUMMARIZED_FROM_ATTRIBUTE)
            if previous
            else None
        )
        if previous is None or source != compute_hash(item.description or ""):
            continue
        item.description = previous.description
        item.attributes = {**(item.attributes or {}), SUMMARIZED_FROM_ATTRIBUTE: source}


class DescriptionSummarizer:
    """Re-summarize entity/relationship descriptions that exceed a token budget.

    The LLM chain is built once in ``__init__`` with the shared
    ``setup_chain`` helper (as ``GraphExtractor`` and ``GraphGleaner`` do), and
    the calls fan out through ``BatchProcessor`` like the rest of the
    ingestion pipeline.
    """

    def __init__(
        self,
        config: Config,
        *,
        providers: Providers | None = None,
    ) -> None:
        self.config = config
        self.summarization_config: DescriptionSummarizationConfig = (
            self.config.processing.graph_extraction.description_summarization
        )
        self.target_language = self.config.processing.translation.target_language.value
        self.providers = Providers.resolve(config, providers)
        self.factory = self.providers.llm_factory
        self.batch_processor = BatchProcessor.from_processing(
            self.config.processing,
            is_transient_error=transient_error_predicate(self.factory),
        )
        self.summarizer = setup_chain(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            model_id=self.summarization_config.summary_model_id,
            prompt_class=DescriptionSummarizationPrompt,
            parser=StrOutputParser(),
            custom_prompts=self.config.custom_prompts,
        )
        # Local estimate (scaled for the model) rather than the CountTokens
        # API: packing counts every line of every over-long description.
        self.token_counter = self.providers.token_counter(
            self.summarization_config.summary_model_id, api_supported=False
        )

    @cached_property
    def max_input_tokens(self) -> int:
        """One call's input budget for the summary model (see
        ``MAX_INPUT_TOKENS``).

        ``setup_chain`` has already read the model's info, so an unknown
        model has failed by now.
        """
        window = self.factory.get_model_info(
            self.summarization_config.summary_model_id
        ).effective_context_window()
        available = (window - self.summarization_config.max_summary_tokens) // 2
        return min(MAX_INPUT_TOKENS, max(_MIN_INPUT_TOKENS, available))

    def _exceeds_threshold(self, description: str | None) -> bool:
        return needs_summary(self.config, description)

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
                _summarized(entity, summary)
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
                _summarized(rel, summary)
        return relationships

    def _build_inputs(
        self, names: list[str], descriptions: list[str]
    ) -> list[dict[str, str]]:
        return [
            {
                "entity_name": name,
                "descriptions": description,
                "max_summary_tokens": str(self.summarization_config.max_summary_tokens),
                # Extraction and gleaning write descriptions in the target
                # language, so the merged input is already in it.
                "language": self.target_language,
                "target_language": self.target_language,
            }
            for name, description in zip(names, descriptions, strict=True)
        ]

    def _summarize_many(
        self, names: list[str], descriptions: list[str]
    ) -> list[str | None]:
        """Summarize each over-threshold item; keep order, degrade safely.

        A description that fits ``max_input_tokens`` gets one call. A longer
        one is first split into groups of whole lines within the budget, each
        group is summarized, and the newline join of the group summaries
        replaces it, until it fits one call. Returns one entry per input: the
        summary string, or ``None`` when any of its calls failed (the caller
        then keeps the concatenated text).
        """
        logger.info(
            "Summarizing %s over-threshold descriptions (threshold %s tokens)",
            len(names),
            self.summarization_config.force_summary_threshold_tokens,
        )
        current: list[str | None] = list(descriptions)
        for _ in range(_MAX_MAP_ROUNDS):
            groups = {
                index: self._pack(text)
                for index, text in enumerate(current)
                if text is not None
            }
            split = {index: parts for index, parts in groups.items() if len(parts) > 1}
            if not split:
                break
            logger.info(
                "Summarizing %s descriptions over the %s-token input budget in "
                "%s groups",
                len(split),
                self.max_input_tokens,
                sum(len(parts) for parts in split.values()),
            )
            flat = [(index, part) for index, parts in split.items() for part in parts]
            partials = self._invoke_all(
                [names[index] for index, _ in flat], [part for _, part in flat]
            )
            reduced: dict[int, list[str] | None] = {index: [] for index in split}
            for (index, _), partial in zip(flat, partials, strict=True):
                collected = reduced[index]
                if collected is None:
                    continue
                if partial is None:
                    reduced[index] = None
                else:
                    collected.append(partial)
            for index, collected in reduced.items():
                current[index] = None if collected is None else "\n".join(collected)

        pending = [index for index, text in enumerate(current) if text is not None]
        finals = self._invoke_all(
            [names[index] for index in pending],
            [current[index] or "" for index in pending],
        )
        summaries: list[str | None] = [None] * len(names)
        for index, summary in zip(pending, finals, strict=True):
            summaries[index] = summary
        return summaries

    def _pack(self, description: str) -> list[str]:
        """Split ``description`` into groups of whole lines within the budget.

        A description within the budget is one group, unchanged. Otherwise
        lines are packed in order while their summed counts fit (a joined
        group can count a few tokens more than the sum); a single line over
        the budget is a group of its own.
        """
        if self.token_counter.count_tokens(description) <= self.max_input_tokens:
            return [description]
        groups: list[str] = []
        lines: list[str] = []
        used = 0
        for line in description.split("\n"):
            tokens = self.token_counter.count_tokens(line)
            if lines and used + tokens > self.max_input_tokens:
                groups.append("\n".join(lines))
                lines, used = [], 0
            lines.append(line)
            used += tokens
        if lines:
            groups.append("\n".join(lines))
        return groups

    def _invoke_all(
        self, names: list[str], descriptions: list[str]
    ) -> list[str | None]:
        """One summarization call per item; ``None`` for an item that failed."""
        if not names:
            return []
        inputs = self._build_inputs(names, descriptions)
        try:
            results = self.batch_processor.execute_with_fallback(
                items_to_process=inputs,
                prepare_inputs_func=lambda chunk: chunk,
                sequential_func=self.summarizer.invoke,
                task_name="Description Summarization",
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
                logger.warning("Empty/failed summary; keeping concatenated description")
                logger.debug("Empty/failed summary for '%s'", name)
            summaries.append(summary)
        return summaries

    @staticmethod
    def _coerce_summary(result: object) -> str | None:
        """Normalize a chain result into a clean summary string (or None).

        Any non-string result — including ``BATCH_ITEM_FAILED`` for an item
        whose LLM call failed every attempt — yields ``None`` so the caller
        keeps the concatenated original instead of its ``str()``.
        """
        if not isinstance(result, str):
            return None
        text = result.strip()
        return text or None
