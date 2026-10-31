# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from collections.abc import Sequence
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config, ModelRole, RetrievalResult
from unified_kg_rag.domain.prompts import AnswerGenerationPrompt
from unified_kg_rag.ports.model_factory import (
    LanguageModelInfo,
    effective_max_output_tokens,
)
from unified_kg_rag.shared import get_logger, observability
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor

logger = get_logger(__name__)

# Placeholder returned by build_context_string when no context sections survive
# optimization. Callers short-circuit answer generation on this sentinel so the
# LLM is never asked to answer from an empty context (which invites fabrication).
EMPTY_CONTEXT_PLACEHOLDER = "No relevant information found."
# Where a truncated section may end: whitespace, or CJK sentence/clause marks
# for text written without spaces.
_TRUNCATION_BREAK = re.compile(r"[\s。．！？、，；]")
# A break is only taken from this trailing share of the kept prefix; further
# back it would throw away most of the room (one space early in Japanese text
# used to leave a few characters), so the cut falls at the character instead.
_TRUNCATION_BACKOFF_RATIO = 0.2


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
    result_index: int | None = Field(
        default=None,
        description="Position of the originating RetrievalResult in the list "
        "passed to optimize_context, so callers can map the selected sections "
        "back to the results they came from",
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


class TokenManager:
    MIN_DERIVED_CONTEXT_TOKENS: ClassVar[int] = 1024
    # Held back from the configured/derived budget for the answer prompt's own
    # text. Not taken from a caller's explicit per-query budget, which already
    # names the context size it wants.
    CONTEXT_BUFFER_TOKENS: ClassVar[int] = 512
    # Concurrent token counts per count_tokens_many call (each may be one
    # CountTokens request).
    COUNT_CONCURRENCY: ClassVar[int] = 8

    def __init__(
        self,
        config: Config,
        *,
        providers: Providers | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config.search.token_manager
        # The token counter comes from the shared provider bundle, so a chain
        # and its strategies reuse one bedrock-runtime client (and a custom
        # backend's counter) instead of each building its own.
        providers = Providers.resolve(config, providers)

        answer_model_id = config.model_id(ModelRole.ANSWER_GENERATION)
        self._enable_1m_context = config.aws.bedrock.enable_1m_context
        answer_model_info = providers.llm_model_info(answer_model_id)
        self._token_counter = providers.token_counter(
            answer_model_id,
            cache_maxsize=self.config.token_count_cache_size,
            api_supported=answer_model_info.supports_count_tokens,
        )
        self._counts_via_api = answer_model_info.supports_count_tokens
        # Reserve what the answer request actually asks for (its max_tokens),
        # not the model maximum, so a capped output leaves room for context.
        answer_output_tokens = effective_max_output_tokens(
            answer_model_info,
            config.aws.bedrock,
            AnswerGenerationPrompt.min_output_tokens,
        )
        self._max_context_tokens = self._resolve_max_context_tokens(
            answer_model_id, answer_model_info, answer_output_tokens
        )

    def _resolve_max_context_tokens(
        self, answer_model_id: str, model_info: LanguageModelInfo, output_tokens: int
    ) -> int:
        """Size the prompt-side budget against the answer model's real window.

        The context budget and the model's context window are separate settings
        that must agree: the window has to hold the prompt *and* the generated
        answer, so a budget of window-size leaves no room for output. Deriving it
        from the model's own capabilities keeps the two in sync when the answer
        model changes (a 200K model and a 1M model want very different budgets).
        """
        configured = self.config.max_context_tokens
        headroom = self.config.context_window_headroom_ratio
        window = model_info.effective_context_window(self._enable_1m_context)
        budget_ceiling = int((window - output_tokens) * (1.0 - headroom))
        # A model whose output reservation swallows its window would yield a
        # non-positive ceiling; keep a usable floor instead of returning <= 0,
        # which optimize_context treats as "exclude everything".
        budget_ceiling = max(budget_ceiling, self.MIN_DERIVED_CONTEXT_TOKENS)

        if configured is None:
            logger.debug(
                "Derived context budget of %d tokens for '%s' (window=%d, "
                "output=%d, headroom=%.0f%%)",
                budget_ceiling,
                answer_model_id,
                window,
                output_tokens,
                headroom * 100,
            )
            return budget_ceiling
        if configured > budget_ceiling:
            logger.warning(
                "Configured max_context_tokens (%d) exceeds what '%s' can accept "
                "alongside its %d-token output reservation; clamping to %d.",
                configured,
                answer_model_id,
                output_tokens,
                budget_ceiling,
            )
            return budget_ceiling
        return configured

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return self._token_counter.count_tokens(text)

    def count_tokens_many(self, texts: Sequence[str]) -> list[int]:
        """Exact counts for ``texts``, in order, with distinct texts counted
        concurrently (at most ``COUNT_CONCURRENCY`` at a time) when each count
        is a CountTokens round trip.

        A mix/hybrid context has hundreds of sections; counting them one round
        trip after another dominated query latency. Without the API (the answer
        model does not accept CountTokens, or rejected it at runtime) a count is
        a local CPU-bound estimate, so they run inline: threads would only add
        pool start-up and GIL contention. Duplicates are counted once (and the
        counter's own cache still applies). Blocking: run it off the event loop
        from async code.
        """
        unique = list(dict.fromkeys(t for t in texts if t))
        # The counter knows whether its counts are round trips right now (the
        # Bedrock one stops calling the API once the model rejects it); one
        # without that attribute is taken at the model's capability record.
        via_api = getattr(self._token_counter, "counts_via_api", self._counts_via_api)
        if len(unique) <= 1 or not via_api:
            counts = {t: self.count_tokens(t) for t in unique}
        else:
            workers = min(self.COUNT_CONCURRENCY, len(unique))
            with ContextThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="count-tokens"
            ) as pool:
                counts = dict(
                    zip(unique, pool.map(self.count_tokens, unique), strict=True)
                )
        return [counts[t] if t else 0 for t in texts]

    def optimize_context(
        self,
        retrieval_results: list[RetrievalResult],
        query: str,
        max_tokens: int | None = None,
        spare_budget_first: Sequence[SectionType] = (),
    ) -> OptimizedContext:
        """The sections of ``retrieval_results`` that fit the context budget.

        ``spare_budget_first`` names the section types that take budget other
        types leave unused before the rest share it (``_select_per_type``).
        """
        with observability.span(
            "kg_rag.context_build", results_in=len(retrieval_results)
        ) as span:
            optimized, budget = self._optimize_context(
                retrieval_results, query, max_tokens, spare_budget_first
            )
            # Tokens used vs the budget the sections had to fit in.
            span.set_attributes(
                {
                    "kg_rag.context.tokens_used": optimized.total_tokens,
                    "kg_rag.context.token_budget": budget,
                    "kg_rag.context.sections_included": optimized.sections_included,
                    "kg_rag.context.sections_excluded": optimized.sections_excluded,
                }
            )
            return optimized

    def _optimize_context(
        self,
        retrieval_results: list[RetrievalResult],
        query: str,
        max_tokens: int | None,
        spare_budget_first: Sequence[SectionType] = (),
    ) -> tuple[OptimizedContext, int]:
        # A caller-supplied budget is still capped by what the answer model can
        # accept, so a large --top-k / RAGInput.max_tokens cannot overflow it.
        # The prompt buffer comes off that ceiling only: subtracting it from a
        # small explicit budget too (max_tokens <= 513) left nothing for context.
        ceiling = self._max_context_tokens - self.CONTEXT_BUFFER_TOKENS
        target_tokens = min(max_tokens, ceiling) if max_tokens else ceiling
        query_tokens = self.count_tokens(query)
        available_tokens = target_tokens - query_tokens

        if available_tokens <= 0:
            logger.error(
                "Token budget exceeded - query: %s, target: %s",
                query_tokens,
                target_tokens,
            )
            return (
                OptimizedContext(
                    sections=[],
                    total_tokens=0,
                    sections_included=0,
                    sections_excluded=len(retrieval_results),
                ),
                available_tokens,
            )

        all_sections = self._create_context_sections(retrieval_results)
        # Each section type present gets a header in the context string
        # (build_context_string); it is paid from the same budget.
        available_tokens -= sum(
            self.count_tokens(self._section_header(section_type))
            for section_type in {s.section_type for s in all_sections}
        )
        selected_sections = self._select_optimal_sections(
            all_sections, max(available_tokens, 0), spare_budget_first
        )

        total_tokens = sum(section.token_count for section in selected_sections)
        sections_included = len(selected_sections)
        sections_excluded = len(all_sections) - sections_included

        return (
            OptimizedContext(
                sections=selected_sections,
                total_tokens=total_tokens,
                sections_included=sections_included,
                sections_excluded=sections_excluded,
            ),
            available_tokens,
        )

    def _create_context_sections(
        self, results: list[RetrievalResult]
    ) -> list[ContextSection]:
        # A section is counted as build_context_string renders it, with its
        # "[source_id] " prefix, so the context stays within the budget.
        counts = self.count_tokens_many(
            [
                self._section_line(result.source or f"result_{index}", result.content)
                for index, result in enumerate(results)
            ]
        )
        return [
            self._create_context_section(result, index, token_count=count)
            for index, (result, count) in enumerate(zip(results, counts, strict=True))
        ]

    def _create_context_section(
        self, result: RetrievalResult, index: int, token_count: int | None = None
    ) -> ContextSection:
        source_id = result.source or f"result_{index}"
        section_type_str = result.retriever_type or SectionType.GENERAL.value
        try:
            section_type = SectionType(section_type_str.lower())
        except ValueError:
            # `retriever_type` is a free-form str; an out-of-enum value (a future
            # retriever, or a malformed result) must degrade to GENERAL rather
            # than abort context-building for the whole query.
            section_type = SectionType.GENERAL

        return ContextSection(
            content=result.content,
            token_count=(
                self.count_tokens(self._section_line(source_id, result.content))
                if token_count is None
                else token_count
            ),
            # Ranks sections only within their own type: each type is packed
            # against its own share (_select_per_type), so a cross-type weight
            # would change nothing there. A 0.0 score stays the lowest.
            priority=result.score,
            section_type=section_type,
            source_id=source_id,
            metadata=result.metadata or {},
            result_index=index,
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

    def _pack_type(
        self, ranked: list[ContextSection], cap: int
    ) -> tuple[list[ContextSection], bool]:
        """``ranked`` packed into ``cap`` tokens in order, and whether all of
        it was seated whole (the type has no candidate left)."""
        packed: list[ContextSection] = []
        used = 0
        for section in ranked:
            room = cap - used
            if section.token_count <= room:
                packed.append(section)
                used += section.token_count
                continue
            # The section overflows the remaining share. Upstream's rows are
            # table rows, so it simply stops; ours are whole retrieved items,
            # and a single community report routinely exceeds a 10% share —
            # dropping the type outright would leave it unrepresented. Truncate
            # the FIRST overflowing section to fit instead, then stop this type
            # (the budget is now spent).
            head = self._truncate_to_tokens(section, room) if room > 0 else None
            if head is not None:
                packed.append(head)
            return packed, False
        return packed, True

    def _select_optimal_sections(
        self,
        sections: list[ContextSection],
        token_budget: int,
        spare_budget_first: Sequence[SectionType] = (),
    ) -> list[ContextSection]:
        # Synthesized sections (global search's ranked map key points or reduce
        # summary) are not one retrieved item of their type: they condense every
        # selected community report, and their producer already sized them
        # (`max_map_reduce_tokens`). Splitting them into the GENERAL share cut
        # them to a fraction of that budget while most of the window stayed
        # unused, so they are seated first and the per-type split covers the
        # remaining budget.
        reserved: list[ContextSection] = []
        remaining = token_budget
        for section in sorted(
            (
                s
                for s in sections
                if s.token_count > 0 and s.metadata.get("synthesized")
            ),
            key=lambda s: s.priority,
            reverse=True,
        ):
            if section.token_count <= remaining:
                reserved.append(section)
                remaining -= section.token_count
                continue
            head = self._truncate_to_tokens(section, remaining)
            if head is not None:
                reserved.append(head)
                remaining -= head.token_count
            break
        evidence = [s for s in sections if not s.metadata.get("synthesized")]
        if not reserved:
            return self._select_per_type(evidence, token_budget, spare_budget_first)
        return reserved + self._select_per_type(
            evidence, remaining, spare_budget_first, fallback=False
        )

    def _select_per_type(
        self,
        sections: list[ContextSection],
        token_budget: int,
        spare_budget_first: Sequence[SectionType] = (),
        fallback: bool = True,
    ) -> list[ContextSection]:
        # Each type is packed in score order against its own share first, so
        # no type can cross into another's share by out-scoring it: scores are
        # only ever compared within a type. Both upstreams start the same way:
        # MS GraphRAG's mixed_context packs the community / local / text
        # sections against independent budgets (community_prop,
        # 1 - community_prop - text_unit_prop, text_unit_prop), and LightRAG caps
        # entities and relations (max_entity_tokens / max_relation_tokens).
        #
        # Budget a type leaves unused because all of its candidates fit is then
        # handed to the types that still have candidates, and they are packed
        # again: first to the ``spare_budget_first`` types (LightRAG's mix/hybrid
        # give chunks what the KG sections did not use: operate.py's
        # available_chunk_tokens = max_total_tokens - KG tokens used), else in
        # proportion to the receivers' shares. A hard cap left 27-34% of the
        # window unused while other types were being cut. The hand-out cannot
        # crowd a type out: it only runs once a type has nothing left to seat,
        # and it goes by share, never by comparing scores across types.
        by_type: dict[SectionType, list[ContextSection]] = {}
        for s in sections:
            if s.token_count > 0:
                by_type.setdefault(s.section_type, []).append(s)
        if not by_type:
            return []
        for lst in by_type.values():
            lst.sort(key=lambda s: s.priority, reverse=True)

        shares = self._type_budget_shares(list(by_type))
        caps = self._type_budgets(list(by_type), token_budget)
        min_spare = self.config.min_truncated_section_tokens
        # A round normally spends the spare or seats a receiver completely, so
        # one round per type (after the first) bounds the loop.
        for _ in range(len(by_type) + 1):
            packed = {t: self._pack_type(lst, caps[t]) for t, lst in by_type.items()}
            spare = token_budget - sum(
                s.token_count for seated, _ in packed.values() for s in seated
            )
            receivers = [t for t, (_, done) in packed.items() if not done]
            if spare < min_spare or not receivers:
                break
            first = [t for t in spare_budget_first if t in receivers]
            if first:
                caps[first[0]] += spare
                continue
            total = sum(shares[t] for t in receivers)
            for t in receivers:
                weight = shares[t] / total if total > 0 else 1.0 / len(receivers)
                caps[t] += int(spare * weight)
        selected = [s for seated, _ in packed.values() for s in seated]

        if selected or not fallback:
            return selected

        # Last resort: every present type's renormalized share came out below the
        # truncation floor, so no type could seat even one item — yet the window
        # itself has room for a whole section. That happens when the window is
        # small relative to the number of present types (shares divide, the floor
        # does not). Returning nothing here would hand the generator an empty
        # context while evidence that fits was on the table, so seat the single
        # highest-priority section instead. This cannot reintroduce the
        # crowding-out defect above: it only runs when the per-type pass selected
        # nothing at all, and it seats exactly one section. Ranked over the input
        # (stable sort), so a score tie goes to the earlier-ranked result rather
        # than to whichever type happened to appear first.
        ranked = sorted(
            (s for s in sections if s.token_count > 0),
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

        Cuts on a whitespace or CJK punctuation boundary near the end of the
        room (else mid-run, for text without one) using the section's own measured
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
        # token_count covers the rendered line, "[source_id] " prefix included.
        prefix_chars = len(self._section_line(section.source_id, "-")) - 1
        chars_per_token = (prefix_chars + len(section.content)) / section.token_count
        keep_chars = int(max_tokens * chars_per_token) - prefix_chars
        if keep_chars <= 0:
            return None
        head = section.content[:keep_chars]
        if keep_chars < len(section.content):
            backoff_start = int(keep_chars * (1 - _TRUNCATION_BACKOFF_RATIO))
            breaks = [m.end() for m in _TRUNCATION_BREAK.finditer(head, backoff_start)]
            if breaks:
                head = head[: breaks[-1]]
        head = head.rstrip()
        if not head:
            return None
        return section.model_copy(
            update={
                "content": f"{head} …",
                "token_count": max_tokens,
                "metadata": {**section.metadata, "truncated": True},
            }
        )

    _SECTION_LABELS: ClassVar[dict[SectionType, str]] = {
        SectionType.TEXT: "Document Chunks",
        SectionType.ENTITY: "Knowledge Graph — Entities",
        SectionType.RELATIONSHIP: "Knowledge Graph — Relationships",
        SectionType.CLAIM: "Claims",
        SectionType.COMMUNITY: "Community Reports",
        SectionType.GENERAL: "Other Context",
    }

    @classmethod
    def _section_header(cls, section_type: SectionType) -> str:
        return f"##### {cls._SECTION_LABELS[section_type]}"

    @staticmethod
    def _section_line(source_id: str, content: str) -> str:
        """How build_context_string renders one section ("" for no content)."""
        return f"[{source_id}] {content}" if content else ""

    @classmethod
    def ranked_section_lines(cls, optimized_context: OptimizedContext) -> list[str]:
        """The sections as ``build_context_string`` renders them, one entry
        per section, highest priority first (ties keep their order)."""
        ranked = sorted(
            optimized_context.sections, key=lambda s: s.priority, reverse=True
        )
        lines = (cls._section_line(s.source_id, s.content) for s in ranked)
        return [line for line in lines if line]

    @classmethod
    def build_context_string(cls, optimized_context: OptimizedContext) -> str:
        # Group sections by type under labeled
        # headers (mirrors LightRAG's Entities/Relationships/Chunks blocks and MS's
        # sectioned tables) instead of a flat per-item "Source Type/Priority" dump.
        # The Priority float was meaningless prompt noise; dropped. Text chunks carry
        # a citable [id]; the answer prompt can reference them.
        if not optimized_context.sections:
            return EMPTY_CONTEXT_PLACEHOLDER

        grouped: dict[SectionType, list[ContextSection]] = {}
        for s in optimized_context.sections:
            grouped.setdefault(s.section_type, []).append(s)

        blocks: list[str] = []
        for stype in cls._SECTION_LABELS:
            items = grouped.get(stype)
            if not items:
                continue
            lines = [cls._section_header(stype)]
            for s in items:
                lines.append(cls._section_line(s.source_id, s.content))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks).strip()
