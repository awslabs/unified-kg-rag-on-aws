# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Any, ClassVar

import boto3
from botocore.config import Config as BotoConfig
from pydantic import BaseModel, Field

from unified_kg_rag.adapters.aws.bedrock import get_assumed_role_boto_session
from unified_kg_rag.adapters.aws.token_counter import BedrockTokenCounter
from unified_kg_rag.domain.models import Config, RetrievalResult
from unified_kg_rag.domain.retrieval.mixins import MetricsMixin
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)


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

        model_id = config.search.answer_generation_model_id.value
        self._token_counter = BedrockTokenCounter(
            model_id=model_id,
            client=bedrock_client,
            cache_maxsize=self.config.token_count_cache_size,
        )

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
        target_tokens = max_tokens or self.config.max_context_tokens
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

    @staticmethod
    def _select_optimal_sections(
        sections: list[ContextSection], token_budget: int
    ) -> list[ContextSection]:
        sorted_sections = sorted(sections, key=lambda s: s.priority, reverse=True)
        selected_sections = []
        used_tokens = 0

        for section in sorted_sections:
            if section.token_count == 0:
                continue

            if used_tokens + section.token_count <= token_budget:
                selected_sections.append(section)
                used_tokens += section.token_count

        return selected_sections

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
        if not optimized_context.sections:
            return "No relevant information found."

        context_parts = []
        for section in optimized_context.sections:
            header = (
                f"## Source Type: {section.section_type.value.upper()} "
                f"(Source ID: {section.source_id}, Priority: {section.priority:.2f})"
            )
            context_parts.append(f"{header}\n{section.content}")

        return "\n\n---\n\n".join(context_parts).strip()
