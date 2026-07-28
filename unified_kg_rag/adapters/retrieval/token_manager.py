# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Any, ClassVar

import boto3
from botocore.config import Config as BotoConfig
from pydantic import BaseModel, Field

from unified_kg_rag.adapters.aws.bedrock import (
    get_assumed_role_boto_session,
    get_language_model_info,
)
from unified_kg_rag.adapters.aws.token_counter import BedrockTokenCounter
from unified_kg_rag.domain.models import Config, LanguageModelId, RetrievalResult
from unified_kg_rag.domain.retrieval.mixins import MetricsMixin
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

# Placeholder returned by build_context_string when no context sections survive
# optimization. Callers short-circuit answer generation on this sentinel so the
# LLM is never asked to answer from an empty context (which invites fabrication).
EMPTY_CONTEXT_PLACEHOLDER = "No relevant information found."


class SectionType(str, Enum):
    CLAIM = "claim"
    COMMUNITY = "community"
    ENTITY = "entity"
    GENERAL = "general"
    RELATIONSHIP = "relationship"
    TEXT = "text"


class ContextSection(BaseModel):
    content: str = Field(description="The actual text content of the section")
    token_count: int = Field(description="Number of tokens in this section")
    priority: float = Field(description="Priority score for this section")
    section_type: SectionType = Field(
        description="Type of the section (entity, relationship, etc.)"
    )
    source_id: str = Field(
        description="Unique identifier for the source of this section"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the section"
    )


class OptimizedContext(BaseModel):
    sections: list[ContextSection] = Field(
        description="List of context sections included in the optimized context"
    )
    total_tokens: int = Field(
        description="Total number of tokens in the optimized context"
    )
    sections_included: int = Field(
        description="Number of sections included in the optimization"
    )
    sections_excluded: int = Field(
        description="Number of sections excluded from the optimization"
    )
    quality_score: float = Field(
        description="Quality score of the optimized context (0.0 to 1.0)"
    )


class TokenManager(MetricsMixin):
    # Used only when the answer model has no capability record (custom backend)
    # and no explicit budget is configured. Sized for the smallest window this
    # package targets (200K) so it can never overflow a known model.
    FALLBACK_MAX_CONTEXT_TOKENS: ClassVar[int] = 120000
    MIN_DERIVED_CONTEXT_TOKENS: ClassVar[int] = 1024
    PRIORITY_MULTIPLIERS: ClassVar[dict[SectionType, float]] = {
        SectionType.TEXT: 1.3,
        SectionType.ENTITY: 1.2,
        SectionType.RELATIONSHIP: 1.1,
        # Claims are evidentiary (subject/predicate/object assertions about an
        # entity); weight them alongside relationships.
        SectionType.CLAIM: 1.1,
        SectionType.COMMUNITY: 1.0,
        SectionType.GENERAL: 0.8,
    }

    def __init__(
        self, config: Config, boto_session: boto3.Session | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.config = config.search.token_manager

        boto_session = boto_session or boto3.Session(
            profile_name=config.aws.profile_name
        )
        boto_session = get_assumed_role_boto_session(
            boto_session, assumed_role_arn=config.aws.bedrock.assumed_role_arn
        )
        bedrock_client = boto_session.client(
            "bedrock-runtime",
            region_name=config.aws.bedrock.region_name,
            config=BotoConfig(
                retries={"max_attempts": 3},
            ),
        )

        answer_model_id = config.search.answer_generation_model_id
        self._enable_1m_context = config.aws.bedrock.enable_1m_context
        self._token_counter = BedrockTokenCounter(
            model_id=answer_model_id.value,
            client=bedrock_client,
            cache_maxsize=self.config.token_count_cache_size,
        )
        self._max_context_tokens = self._resolve_max_context_tokens(answer_model_id)

    def _resolve_max_context_tokens(self, answer_model_id: LanguageModelId) -> int:
        """Size the prompt-side budget against the answer model's real window.

        The context budget and the model's context window are separate settings
        that must agree: the window has to hold the prompt *and* the generated
        answer, so a budget of window-size leaves no room for output. Deriving it
        from the model's own capabilities keeps the two in sync when the answer
        model changes (a 200K model and a 1M model want very different budgets).
        """
        model_info = get_language_model_info(answer_model_id)
        configured = self.config.max_context_tokens
        if model_info is None:
            # Unknown model (custom backend): honour the explicit value, or fall
            # back to a conservative budget rather than guessing a window.
            if configured is not None:
                return configured
            logger.warning(
                "No capability record for answer model '%s'; using the fallback "
                "context budget of %d tokens. Set "
                "search.token_manager.max_context_tokens explicitly.",
                answer_model_id.value,
                self.FALLBACK_MAX_CONTEXT_TOKENS,
            )
            return self.FALLBACK_MAX_CONTEXT_TOKENS

        headroom = self.config.context_window_headroom_ratio
        window = model_info.effective_context_window(self._enable_1m_context)
        budget_ceiling = int((window - model_info.max_output_tokens) * (1.0 - headroom))
        # A model whose output reservation swallows its window would yield a
        # non-positive ceiling; keep a usable floor instead of returning <= 0,
        # which optimize_context treats as "exclude everything".
        budget_ceiling = max(budget_ceiling, self.MIN_DERIVED_CONTEXT_TOKENS)

        if configured is None:
            logger.debug(
                "Derived context budget of %d tokens for '%s' (window=%d, "
                "output=%d, headroom=%.0f%%)",
                budget_ceiling,
                answer_model_id.value,
                window,
                model_info.max_output_tokens,
                headroom * 100,
            )
            return budget_ceiling
        if configured > budget_ceiling:
            logger.warning(
                "Configured max_context_tokens (%d) exceeds what '%s' can accept "
                "alongside its %d-token output reservation; clamping to %d.",
                configured,
                answer_model_id.value,
                model_info.max_output_tokens,
                budget_ceiling,
            )
            return budget_ceiling
        return configured

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return self._token_counter.count_tokens(text)

    def optimize_context(
        self,
        retrieval_results: list[RetrievalResult],
        query: str,
        max_tokens: int | None = None,
        max_context_tokens_buffer: int = 512,
    ) -> OptimizedContext:
        # A caller-supplied budget is still capped by what the answer model can
        # accept, so a large --top-k / RAGInput.max_tokens cannot overflow it.
        target_tokens = (
            min(max_tokens, self._max_context_tokens)
            if max_tokens
            else (self._max_context_tokens)
        )
        query_tokens = self.count_tokens(query)
        available_tokens = target_tokens - query_tokens - max_context_tokens_buffer

        if available_tokens <= 0:
            logger.error(
                "Token budget exceeded - query: %s, target: %s, buffer: %s",
                query_tokens,
                target_tokens,
                max_context_tokens_buffer,
            )
            return OptimizedContext(
                sections=[],
                total_tokens=0,
                sections_included=0,
                sections_excluded=len(retrieval_results),
                quality_score=0.0,
            )

        all_sections = self._create_context_sections(retrieval_results)
        selected_sections = self._select_optimal_sections(
            all_sections, available_tokens
        )

        total_tokens = sum(section.token_count for section in selected_sections)
        quality_score = self._calculate_quality_score(selected_sections, all_sections)
        sections_included = len(selected_sections)
        sections_excluded = len(all_sections) - sections_included

        self._record_optimization_metrics(
            total_tokens, sections_included, sections_excluded, quality_score
        )

        return OptimizedContext(
            sections=selected_sections,
            total_tokens=total_tokens,
            sections_included=sections_included,
            sections_excluded=sections_excluded,
            quality_score=quality_score,
        )

    def _create_context_sections(
        self, results: list[RetrievalResult]
    ) -> list[ContextSection]:
        return [
            self._create_context_section(result, index)
            for index, result in enumerate(results)
        ]

    def _create_context_section(
        self, result: RetrievalResult, index: int
    ) -> ContextSection:
        section_type_str = result.retriever_type or SectionType.GENERAL.value
        try:
            section_type = SectionType(section_type_str.lower())
        except ValueError:
            # `retriever_type` is a free-form str; an out-of-enum value (a future
            # retriever, or a malformed result) must degrade to GENERAL rather
            # than abort context-building for the whole query. The multiplier
            # lookup below already tolerates unknowns via `.get(..., 1.0)`.
            section_type = SectionType.GENERAL

        base_score = result.score or 0.5
        priority_multiplier = self.PRIORITY_MULTIPLIERS.get(section_type, 1.0)
        priority = base_score * priority_multiplier

        return ContextSection(
            content=result.content,
            token_count=self.count_tokens(result.content),
            priority=priority,
            section_type=section_type,
            source_id=result.source or f"result_{index}",
            metadata=result.metadata or {},
        )

    # Both upstreams (MS GraphRAG `mixed_context`, LightRAG
    # `_apply_token_truncation`) split the context window into PER-TYPE
    # sub-budgets and pack each type independently, so text chunks (which hold the
    # multi-hop answer) are guaranteed representation and can't be crowded out by a
    # flat greedy fill of near-tie RRF scores. The shares themselves are
    # configurable (`search.token_manager.type_budgets`); this maps the enum onto
    # that model's field names so a new SectionType is a compile-time concern
    # rather than a silent zero share.
    _BUDGET_FIELDS: ClassVar[dict[SectionType, str]] = {
        SectionType.TEXT: "text",
        SectionType.ENTITY: "entity",
        SectionType.RELATIONSHIP: "relationship",
        SectionType.CLAIM: "claim",
        SectionType.COMMUNITY: "community",
        SectionType.GENERAL: "general",
    }

    def _type_budget_shares(
        self, present: list[SectionType]
    ) -> dict[SectionType, float]:
        configured = self.config.type_budgets
        return {
            t: float(getattr(configured, self._BUDGET_FIELDS[t], 0.0)) for t in present
        }

    def _type_budgets(
        self, present: list[SectionType], token_budget: int
    ) -> dict[SectionType, int]:
        # Renormalize the shares over the types actually present, so an absent type
        # does not silently shrink the window. If every present type is configured
        # to 0.0, fall back to equal shares rather than emitting an empty context
        # for the whole query — a share of zero should express "deprioritize", not
        # "discard the only evidence there is".
        shares = self._type_budget_shares(present)
        total = sum(shares.values())
        if total <= 0.0:
            shares = dict.fromkeys(present, 1.0)
            total = float(len(present)) or 1.0
        return {t: int(token_budget * s / total) for t, s in shares.items()}

    def _select_optimal_sections(
        self, sections: list[ContextSection], token_budget: int
    ) -> list[ContextSection]:
        # The per-type budget is a HARD CAP, not a floor.
        #
        # This used to pack each type to its sub-budget and then run a second pass
        # that pooled ALL unused budget and offered it to the leftover sections of
        # ANY type by raw priority. Because PRIORITY_MULTIPLIERS favors TEXT (1.3)
        # and every leftover competed in one global pool, whichever type happened to
        # have the most/highest-scoring leftovers absorbed the entire remainder. That
        # made the configured shares advisory rather than binding: one type could
        # take most of the window against a 0.10 share, and once chunk candidates
        # carried real descending rank scores TEXT evicted the community and entity
        # sections almost entirely.
        #
        # Neither upstream pools: MS GraphRAG's mixed_context builds the community /
        # local / text sections against three INDEPENDENT strict token budgets
        # (community_prop, 1 - community_prop - text_unit_prop, text_unit_prop);
        # LightRAG truncates entities / relations / chunks against separate per-type
        # limits. Mirror that: each type is packed only against its own cap, and no
        # type can ever cross into another's share by out-scoring it.
        by_type: dict[SectionType, list[ContextSection]] = {}
        for s in sections:
            if s.token_count > 0:
                by_type.setdefault(s.section_type, []).append(s)
        if not by_type:
            return []
        for lst in by_type.values():
            lst.sort(key=lambda s: s.priority, reverse=True)

        # ONE pass, no leftover re-offering. A type that cannot fill its share
        # (e.g. only 8 chunk candidates survive local search's entity ceiling)
        # leaves the remainder UNUSED — exactly as MS GraphRAG leaves an unfilled
        # text_unit_prop share unused rather than handing it to community reports.
        # Re-offering the remainder to whichever type still has candidates was
        # measured to reproduce the original defect verbatim: community reports
        # kept 0.66 of the window because they were the only type with candidates
        # left. Renormalization over PRESENT types (in _type_budgets) is the only
        # redistribution, and it is static — it cannot depend on how many
        # candidates a type happens to have.
        budgets = self._type_budgets(list(by_type), token_budget)
        selected: list[ContextSection] = []
        for stype, lst in by_type.items():
            cap = budgets[stype]
            sub_used = 0
            for section in lst:
                room = cap - sub_used
                if room <= 0:
                    break
                if section.token_count <= room:
                    selected.append(section)
                    sub_used += section.token_count
                    continue
                # The section overflows the remaining share. Upstream's rows are
                # table rows, so it simply stops; ours are whole retrieved items,
                # and a single community report routinely exceeds a 10% share —
                # dropping the type outright cost 5 gold contexts and shrank the
                # window from 29k to 8.8k chars. Truncate the FIRST overflowing
                # section to fit instead, so every type with candidates is
                # represented, then stop this type (the budget is now spent).
                head = self._truncate_to_tokens(section, room)
                if head is not None:
                    selected.append(head)
                break

        if selected:
            return selected

        # Last resort: every present type's renormalized share came out below the
        # truncation floor, so no type could seat even one item — yet the window
        # itself has room for a whole section. That happens when the window is
        # small relative to the number of present types (shares divide, the floor
        # does not). Returning nothing here would hand the generator an empty
        # context while evidence that fits was on the table, so seat the single
        # highest-priority section instead. This cannot reintroduce the
        # crowding-out defect above: it only runs when the per-type pass selected
        # nothing at all, and it seats exactly one section.
        ranked = sorted(
            (s for lst in by_type.values() for s in lst),
            key=lambda s: s.priority,
            reverse=True,
        )
        for section in ranked:
            if section.token_count <= token_budget:
                return [section]
        head = self._truncate_to_tokens(ranked[0], token_budget)
        return [head] if head is not None else []

    def _truncate_to_tokens(
        self, section: ContextSection, max_tokens: int
    ) -> ContextSection | None:
        """Cut a section's content down to roughly ``max_tokens``, or None if the
        budget is too small to carry anything meaningful.

        Cuts on a whitespace boundary using the section's own measured
        tokens-per-character ratio: this runs inside the packing loop, so a real
        token count per candidate prefix would mean an extra Bedrock count_tokens
        round trip per section. The ratio is exact enough for a budget cap, and the
        result is deliberately conservative (floor + an explicit ellipsis marker).
        """
        if (
            max_tokens < self.config.min_truncated_section_tokens
            or section.token_count <= 0
        ):
            return None
        chars_per_token = len(section.content) / section.token_count
        keep_chars = int(max_tokens * chars_per_token)
        if keep_chars <= 0:
            return None
        head = section.content[:keep_chars].rsplit(" ", 1)[0].rstrip()
        if not head:
            return None
        return section.model_copy(
            update={
                "content": f"{head} …",
                "token_count": max_tokens,
                "metadata": {**section.metadata, "truncated": True},
            }
        )

    @staticmethod
    def _calculate_quality_score(
        selected_sections: list[ContextSection],
        all_sections: list[ContextSection],
    ) -> float:
        if not all_sections or not selected_sections:
            return 0.0

        total_priority = sum(section.priority for section in all_sections)
        selected_priority = sum(section.priority for section in selected_sections)
        priority_coverage = (
            selected_priority / total_priority if total_priority > 0 else 0.0
        )

        all_types = {section.section_type for section in all_sections}
        selected_types = {section.section_type for section in selected_sections}
        type_diversity = len(selected_types) / len(all_types) if all_types else 0.0

        return min((priority_coverage * 0.7) + (type_diversity * 0.3), 1.0)

    def _record_optimization_metrics(
        self,
        total_tokens: int,
        sections_included: int,
        sections_excluded: int,
        quality_score: float,
    ) -> None:
        self._record_metric("optimization_tokens", total_tokens)
        self._record_metric("sections_included", sections_included)
        self._record_metric("sections_excluded", sections_excluded)
        self._record_metric("quality_score", quality_score)

    @staticmethod
    def build_context_string(optimized_context: OptimizedContext) -> str:
        # Group sections by type under labeled
        # headers (mirrors LightRAG's Entities/Relationships/Chunks blocks and MS's
        # sectioned tables) instead of a flat per-item "Source Type/Priority" dump.
        # The Priority float was meaningless prompt noise; dropped. Text chunks carry
        # a citable [id]; the answer prompt can reference them.
        if not optimized_context.sections:
            return EMPTY_CONTEXT_PLACEHOLDER

        order = [
            SectionType.TEXT,
            SectionType.ENTITY,
            SectionType.RELATIONSHIP,
            SectionType.CLAIM,
            SectionType.COMMUNITY,
            SectionType.GENERAL,
        ]
        labels = {
            SectionType.TEXT: "Document Chunks",
            SectionType.ENTITY: "Knowledge Graph — Entities",
            SectionType.RELATIONSHIP: "Knowledge Graph — Relationships",
            SectionType.CLAIM: "Claims",
            SectionType.COMMUNITY: "Community Reports",
            SectionType.GENERAL: "Other Context",
        }
        grouped: dict[SectionType, list[ContextSection]] = {}
        for s in optimized_context.sections:
            grouped.setdefault(s.section_type, []).append(s)

        blocks: list[str] = []
        for stype in order:
            items = grouped.get(stype)
            if not items:
                continue
            lines = [f"##### {labels[stype]}"]
            for s in items:
                lines.append(f"[{s.source_id}] {s.content}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks).strip()
