# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import (
    Config,
    ModelRole,
    RetrievalResult,
    SectionType,
)
from unified_kg_rag.domain.prompts import AnswerGenerationPrompt
from unified_kg_rag.domain.retrieval.citations import neutralize_markers
from unified_kg_rag.ports.model_factory import (
    LanguageModelInfo,
    effective_max_output_tokens,
)
from unified_kg_rag.shared import get_logger, observability
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor
from unified_kg_rag.shared.utils.sentences import CLAUSE_MARKS, UNSPACED_STOPS

logger = get_logger(__name__)

# Placeholder returned by build_context_string when no context sections survive
# optimization. Callers short-circuit answer generation on this sentinel so the
# LLM is never asked to answer from an empty context (which invites fabrication).
EMPTY_CONTEXT_PLACEHOLDER = "No relevant information found."
# Where a truncated section may end: whitespace, or CJK sentence/clause marks
# for text written without spaces.
_TRUNCATION_BREAK = re.compile(f"[\\s{UNSPACED_STOPS}{CLAUSE_MARKS}]")
# A break is only taken from this trailing share of the kept prefix; further
# back it would throw away most of the room (one space early in Japanese text
# used to leave a few characters), so the cut falls at the character instead.
_TRUNCATION_BACKOFF_RATIO = 0.2
# The marker a numbered section is counted with before the numbers are
# known (optimize_context(number_sections=True)), and the last number given:
# as wide as any marker, so the rendered context never exceeds the budget it
# was packed in, and the widest an answer's marker is read at
# (domain/retrieval/citations.py).
_MARKER_PLACEHOLDER = 999
# The times a context past its estimate bound (optimize_context's
# estimated_within) is packed again into a smaller budget; the smallest pack
# still past it then is cut down to it (TokenManager._fitted).
_REFIT_ROUNDS = 4


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
    marker: int | None = Field(
        default=None,
        description="The section's citation number [n] in a numbered context "
        "(optimize_context(number_sections=True)); None for a synthesized "
        "section that names no evidence, and in an unnumbered context",
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
    sections_truncated: int = Field(
        default=0,
        description="Number of the included sections cut to fit the budget",
    )
    numbered: bool = Field(
        default=False,
        description="Whether the sections render with citation markers [n] "
        "instead of their ids (number_sections)",
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
            config.models.default_max_output_tokens,
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
            # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure - LLM token counts, not credentials
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
            # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure - LLM token counts, not credentials
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

    @property
    def context_ceiling(self) -> int:
        """The most context (and question) tokens a budget may take: what the
        answer model accepts (``max_context_tokens``) less the prompt's
        buffer."""
        return self._max_context_tokens - self.CONTEXT_BUFFER_TOKENS

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
        number_sections: bool = False,
        estimated_within: int | None = None,
    ) -> OptimizedContext:
        """The sections of ``retrieval_results`` that fit the context budget.

        ``spare_budget_first`` names the section types that take budget other
        types leave unused before the rest share it (``_select_per_type``).
        ``number_sections`` gives every selected section but a synthesized
        one that names no evidence (``is_citable``) a citation marker ``[1]``, ``[2]``, ... in the order
        ``build_context_string`` renders them, in place of its id prefix
        (docs/design.md §7.1).
        ``estimated_within`` bounds the built context
        (``build_context_string``) at the local estimate
        (``estimate_token_count``) the usage meter counts a call's input at,
        which the budget's counter (a CountTokens API, each section counted
        apart) can count lower or higher: the room a query's input-token
        limit leaves the context. ``max_tokens`` stays the query's own
        budget; a context past the bound is packed again into a budget cut
        in the ratio the whole measured at, until it fits.
        """
        with observability.span(
            "kg_rag.context_build", results_in=len(retrieval_results)
        ) as span:
            optimized, budget = self._optimize_context(
                retrieval_results,
                query,
                max_tokens,
                spare_budget_first,
                number_sections,
                estimated_within,
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
        number_sections: bool = False,
        estimated_within: int | None = None,
    ) -> tuple[OptimizedContext, int]:
        # A caller-supplied budget is still capped by what the answer model can
        # accept, so a large --top-k / RAGInput.max_tokens cannot overflow it.
        # The prompt buffer comes off that ceiling only: subtracting it from a
        # small explicit budget too (max_tokens <= 513) left nothing for context.
        ceiling = self.context_ceiling
        target_tokens = min(max_tokens, ceiling) if max_tokens else ceiling
        query_tokens = self.count_tokens(query)
        available_tokens = target_tokens - query_tokens

        if available_tokens <= 0:
            # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure - LLM token counts, not credentials
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
                    numbered=number_sections,
                ),
                available_tokens,
            )

        all_sections = self._create_context_sections(retrieval_results, number_sections)
        # Each section type seated gets a header in the context string
        # (build_context_string); it is paid from the same budget, reserved
        # for every type present first.
        headers = {
            section_type: self.count_tokens(self._section_header(section_type))
            for section_type in dict.fromkeys(s.section_type for s in all_sections)
        }
        optimized, left = self._packed(
            all_sections, headers, available_tokens, spare_budget_first, number_sections
        )
        if estimated_within is None:
            return optimized, left
        over: tuple[OptimizedContext, int, int] | None = None
        for refits in range(_REFIT_ROUNDS + 1):
            if not optimized.sections:
                break
            built = estimate_token_count(self.build_context_string(optimized))
            if built <= estimated_within:
                return optimized, left
            if over is not None and built >= over[2]:
                # The cut left the built context as it was: it came off the
                # room of the sections cut to fit, whose text still ends at
                # the same break (in Japanese or Chinese the last one may
                # be hundreds of characters back), so a smaller budget
                # would not shrink it either.
                break
            over = optimized, left, built
            if refits == _REFIT_ROUNDS:
                break
            # The budget is in the counter's tokens, the bound in the
            # estimate's: what the pieces used is cut in the ratio the whole
            # measured at (the pieces each rounded down, the separators
            # between them uncounted, a counter that counts otherwise).
            used = optimized.total_tokens + sum(
                headers[t] for t in {s.section_type for s in optimized.sections}
            )
            optimized, left = self._packed(
                all_sections,
                headers,
                used * estimated_within // built,
                spare_budget_first,
                number_sections,
            )
        if over is None:
            return optimized, left
        # The smallest pack still past the bound, cut down to it.
        return (
            self._fitted(over[0], all_sections, estimated_within, number_sections),
            over[1],
        )

    def _fitted(
        self,
        over: OptimizedContext,
        all_sections: list[ContextSection],
        estimated_within: int,
        number_sections: bool,
    ) -> OptimizedContext:
        """The longest run of ``over``'s sections (a pack past
        ``estimated_within``) that fits it, taken in the order packing seats
        them (``_by_rank``): the synthesized sections first, then the best
        of every type, then the second best, ... (scores rank only within a
        type), so the sections each type seated last go first. The run keeps
        its sections as the pack seated them (one the pack cut stays cut
        there); the room past it goes to the first section left out, cut
        down from its whole text, or to none when no cut of it fits. When
        not even the first fits, the first of ``all_sections`` in that order
        that fits alone whole, else the first that fits alone cut down:
        empty only when none does."""

        def context(sections: list[ContextSection]) -> OptimizedContext:
            sections = self._numbered(sections) if number_sections else sections
            return OptimizedContext(
                sections=sections,
                total_tokens=sum(s.token_count for s in sections),
                sections_included=len(sections),
                sections_excluded=len(all_sections) - len(sections),
                sections_truncated=sum(
                    1 for s in sections if self._cut(s, all_sections)
                ),
                numbered=number_sections,
            )

        def fits(sections: list[ContextSection]) -> bool:
            built = self.build_context_string(context(sections))
            return estimate_token_count(built) <= estimated_within

        def cut_beside(
            kept: list[ContextSection], section: ContextSection
        ) -> ContextSection | None:
            """``section`` cut to the room ``kept`` leaves it, if any."""
            if section.result_index is not None:
                section = all_sections[section.result_index]
            top = section.token_count - 1

            def cut_to(room: int) -> ContextSection | None:
                return self._truncate_to_tokens(section, room)

            # The rooms too small to cut the section to (below the
            # truncation floor, or short of its prefix) are one run at the
            # bottom, found without building a context. The cut to the
            # first room above them is the smallest: when it does not fit,
            # no cut does, and the section is passed over at one build.
            uncut = self._most(top, lambda room: cut_to(room) is None)
            least = cut_to(uncut + 1) if uncut < top else None
            if least is None or not fits([*kept, least]):
                return None

            def holds(room: int) -> bool:
                # A room below the run that cuts the section is taken to
                # hold, so the rooms that cut it to fit form one run.
                cut = cut_to(room)
                return cut is None or fits([*kept, cut])

            cut = cut_to(self._most(top, holds))
            return cut if cut is not None and fits([*kept, cut]) else None

        ranked = self._by_rank(over.sections)
        # The estimate only grows with each section added, and with the
        # room a section is cut to: both are found by bisection.
        kept = ranked[: self._most(len(ranked) - 1, lambda n: fits(ranked[:n]))]
        if len(kept) < len(ranked):
            # The first section left out, cut to the room the others leave.
            last = cut_beside(kept, ranked[len(kept)])
            if last is not None:
                kept = [*kept, last]
        if not kept:
            alone = self._by_rank([s for s in all_sections if s.token_count > 0])
            whole = next(([s] for s in alone if fits([s])), None)
            cuts = ([c] for s in alone if (c := cut_beside([], s)) is not None)
            kept = whole or next(cuts, [])
        return context(kept)

    @staticmethod
    def _most(top: int, holds: Callable[[int], bool]) -> int:
        """The largest ``n`` in ``0..top`` for which ``holds(n)`` (taken to
        hold for every smaller one), 0 when none above 0 does."""
        lo, hi = 0, max(top, 0)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if holds(mid) else (lo, mid - 1)
        return lo

    @classmethod
    def _by_rank(cls, sections: list[ContextSection]) -> list[ContextSection]:
        """``sections`` in the order packing seats them: the synthesized ones
        first (``_synthesized``), then the evidence by its rank in its type
        (best first), each rank in input order."""
        rank: dict[SectionType, int] = {}
        keyed: list[tuple[int, ContextSection]] = []
        evidence = [s for s in sections if not s.metadata.get("synthesized")]
        for s in sorted(evidence, key=lambda s: s.priority, reverse=True):
            keyed.append((rank.get(s.section_type, 0), s))
            rank[s.section_type] = rank.get(s.section_type, 0) + 1
        return cls._synthesized(sections) + [
            s for _, s in sorted(keyed, key=lambda k: k[0])
        ]

    @staticmethod
    def _synthesized(sections: list[ContextSection]) -> list[ContextSection]:
        """The synthesized sections of ``sections`` (``metadata``'s
        ``synthesized``), best first: the order they are seated in, ahead of
        the evidence (``_select_optimal_sections``)."""
        return sorted(
            (s for s in sections if s.metadata.get("synthesized")),
            key=lambda s: s.priority,
            reverse=True,
        )

    def _packed(
        self,
        all_sections: list[ContextSection],
        headers: dict[SectionType, int],
        available_tokens: int,
        spare_budget_first: Sequence[SectionType],
        number_sections: bool,
    ) -> tuple[OptimizedContext, int]:
        """``all_sections`` packed into ``available_tokens`` with the headers
        of the types seated (``headers``), and the budget the sections had
        to fit in."""
        selected_sections = self._select_optimal_sections(
            all_sections,
            max(available_tokens - sum(headers.values()), 0),
            spare_budget_first,
            headers,
        )
        if not selected_sections and all_sections:
            # No type fits beside every header: the best section's type is
            # packed alone, paying its own header only (a small budget went
            # to the headers of types it could never seat).
            best = max(all_sections, key=lambda s: s.priority).section_type
            selected_sections = self._select_optimal_sections(
                [s for s in all_sections if s.section_type is best],
                max(available_tokens - headers[best], 0),
                spare_budget_first,
            )
        available_tokens -= sum(
            headers[t] for t in {s.section_type for s in selected_sections}
        )

        if number_sections:
            selected_sections = self._numbered(selected_sections)
        total_tokens = sum(section.token_count for section in selected_sections)
        sections_included = len(selected_sections)
        sections_excluded = len(all_sections) - sections_included

        return (
            OptimizedContext(
                sections=selected_sections,
                total_tokens=total_tokens,
                sections_included=sections_included,
                sections_excluded=sections_excluded,
                # Cut to fit (_truncate_to_tokens): its content is not the
                # whole result's.
                sections_truncated=sum(
                    1 for s in selected_sections if self._cut(s, all_sections)
                ),
                numbered=number_sections,
            ),
            available_tokens,
        )

    @staticmethod
    def _cut(section: ContextSection, all_sections: list[ContextSection]) -> bool:
        """Whether ``section`` was cut to fit (``_truncate_to_tokens``): its
        content is not the whole result's."""
        return (
            section.result_index is not None
            and section.content != all_sections[section.result_index].content
        )

    @classmethod
    def _numbered(cls, sections: list[ContextSection]) -> list[ContextSection]:
        """``sections`` with their markers numbered 1, 2, ... in the order
        ``build_context_string`` renders them (type block, then position).
        Past :data:`_MARKER_PLACEHOLDER` a section is unnumbered: wider
        numbers were counted at no width, and an answer's marker has at most
        three digits (a bracketed year is text), so it could not be cited."""
        order = [
            i
            for stype in cls._SECTION_LABELS
            for i, s in enumerate(sections)
            if s.section_type is stype and s.marker is not None
        ]
        numbered = list(sections)
        for number, i in enumerate(order, start=1):
            numbered[i] = sections[i].model_copy(
                update={"marker": number if number <= _MARKER_PLACEHOLDER else None}
            )
        return numbered

    def _create_context_sections(
        self, results: list[RetrievalResult], number_sections: bool = False
    ) -> list[ContextSection]:
        # A section is counted as build_context_string renders it, with its
        # "[source_id] " prefix where it has one (its "[n] " marker in a
        # numbered context, counted at the widest number), so the context
        # stays within the budget. A numbered context's retrieved text reads
        # its own bracketed numbers ("[2]" footnotes) as "(2)", so an answer
        # copying them cites no section of that number.
        contents = [
            neutralize_markers(result.content) if number_sections else result.content
            for result in results
        ]
        markers = [
            (
                _MARKER_PLACEHOLDER
                if number_sections and self.is_citable(result.metadata)
                else None
            )
            for result in results
        ]
        counts = self.count_tokens_many(
            [
                self._section_line(
                    self._section_type(result),
                    result.source or f"result_{index}",
                    content,
                    marker=marker,
                    numbered=number_sections,
                )
                for index, (result, content, marker) in enumerate(
                    zip(results, contents, markers, strict=True)
                )
            ]
        )
        return [
            self._create_context_section(
                result, index, token_count=count, marker=marker, content=content
            )
            for index, (result, content, count, marker) in enumerate(
                zip(results, contents, counts, markers, strict=True)
            )
        ]

    @staticmethod
    def is_citable(metadata: dict[str, Any] | None) -> bool:
        """Whether a result gets a citation marker in a numbered context:
        retrieved evidence, or a synthesized result that names the evidence
        it was written from (``citable``: global search's key points,
        DRIFT's intermediate answers)."""
        metadata = metadata or {}
        return not metadata.get("synthesized") or bool(metadata.get("citable"))

    @staticmethod
    def _section_type(result: RetrievalResult) -> SectionType:
        # A result built with model_copy(update=...) is not validated.
        return SectionType.of(result.retriever_type)

    def _create_context_section(
        self,
        result: RetrievalResult,
        index: int,
        token_count: int | None = None,
        marker: int | None = None,
        content: str | None = None,
    ) -> ContextSection:
        source_id = result.source or f"result_{index}"
        section_type = self._section_type(result)
        content = result.content if content is None else content

        return ContextSection(
            content=content,
            token_count=(
                self.count_tokens(
                    self._section_line(section_type, source_id, content, marker=marker)
                )
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
            marker=marker,
        )

    # Both upstreams (MS GraphRAG `mixed_context`, LightRAG
    # `_apply_token_truncation`) split the context window into PER-TYPE
    # sub-budgets and pack each type independently, so text chunks (which hold the
    # multi-hop answer) are guaranteed representation and can't be crowded out by a
    # flat greedy fill of near-tie RRF scores. The shares themselves are
    # configurable (`search.token_manager.type_budgets`, one share per
    # SectionType: ContextTypeBudgetConfig.share).

    def _type_budget_shares(
        self, present: list[SectionType]
    ) -> dict[SectionType, float]:
        configured = self.config.type_budgets
        return {t: configured.share(t) for t in present}

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
        headers: Mapping[SectionType, int] | None = None,
    ) -> list[ContextSection]:
        """``sections`` selected for ``token_budget``, beside which the
        ``headers`` of their types are reserved (a type left unseated frees
        its header's)."""
        # Synthesized sections (global search's ranked map key points or reduce
        # summary) are not one retrieved item of their type: they condense every
        # selected community report, and their producer already sized them
        # (`max_map_reduce_tokens`). Splitting them into the GENERAL share cut
        # them to a fraction of that budget while most of the window stayed
        # unused, so they are seated first and the per-type split covers the
        # remaining budget.
        reserved: list[ContextSection] = []
        remaining = token_budget
        for section in self._synthesized([s for s in sections if s.token_count > 0]):
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
        unseated = {
            t: tokens
            for t, tokens in (headers or {}).items()
            if t not in {s.section_type for s in reserved}
        }
        if not reserved:
            return self._select_per_type(
                evidence, token_budget, spare_budget_first, headers=unseated
            )
        return reserved + self._select_per_type(
            evidence, remaining, spare_budget_first, fallback=False, headers=unseated
        )

    def _select_per_type(
        self,
        sections: list[ContextSection],
        token_budget: int,
        spare_budget_first: Sequence[SectionType] = (),
        fallback: bool = True,
        headers: Mapping[SectionType, int] | None = None,
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
        packed = {t: self._pack_type(lst, caps[t]) for t, lst in by_type.items()}
        # A round normally spends the spare or seats a receiver completely, so
        # one round per type bounds the loop.
        bounded = False
        for _ in range(len(by_type)):
            receivers = [t for t, (_, done) in packed.items() if not done]
            used = {
                t: sum(s.token_count for s in seated)
                for t, (seated, _) in packed.items()
            }
            spare = token_budget - sum(used.values())
            if spare < min_spare or not receivers:
                break
            # The spare is lent on top of each receiver's cap: the room a
            # receiver holds but could not seat is in the spare too, for any
            # of them to use. When the receivers then seat past the budget
            # (with the headers of the types they leave unseated, which it
            # reserved), it is lent on top of what each seated instead, the
            # room they held pooled in it (each token lent once).
            lent = self._lent(caps, spare, receivers, shares, spare_budget_first)
            trial = {t: self._pack_type(lst, lent[t]) for t, lst in by_type.items()}
            if self._seated(trial) > token_budget + self._unseated_headers(
                trial, headers
            ):
                bounded = True
                lent = self._lent(
                    {**caps, **{t: used[t] for t in receivers}},
                    spare,
                    receivers,
                    shares,
                    spare_budget_first,
                )
                if max(lent[t] - used[t] for t in receivers) < min_spare:
                    # Each part is too small to cut a section to: the spare
                    # is offered in turn instead.
                    break
                trial = {t: self._pack_type(lst, lent[t]) for t, lst in by_type.items()}
            caps, packed = lent, trial
        if bounded:
            packed = self._topped_up(by_type, packed, token_budget, shares, headers)
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

    @staticmethod
    def _seated(packed: Mapping[SectionType, tuple[list[ContextSection], bool]]) -> int:
        return sum(s.token_count for seated, _ in packed.values() for s in seated)

    @staticmethod
    def _unseated_headers(
        packed: Mapping[SectionType, tuple[list[ContextSection], bool]],
        headers: Mapping[SectionType, int] | None,
    ) -> int:
        """The header room reserved for the types ``packed`` seats nothing
        of: free for the others."""
        return sum(
            tokens
            for t, tokens in (headers or {}).items()
            if not (t in packed and packed[t][0])
        )

    def _topped_up(
        self,
        by_type: dict[SectionType, list[ContextSection]],
        packed: dict[SectionType, tuple[list[ContextSection], bool]],
        token_budget: int,
        shares: dict[SectionType, float],
        headers: Mapping[SectionType, int] | None,
    ) -> dict[SectionType, tuple[list[ContextSection], bool]]:
        """``packed`` with what is left of ``token_budget``, and the header
        room of the types it seats nothing of, offered to the types with
        candidates left in turn by share, largest first, each taking what
        it can seat (a type it seats pays its header from it): parts lent
        in proportion can each fall below what any receiver seats."""
        topped = dict(packed)
        pool = (
            token_budget
            - self._seated(packed)
            + self._unseated_headers(packed, headers)
        )
        receivers = [t for t, (_, done) in packed.items() if not done]
        for t in sorted(receivers, key=lambda t: shares[t], reverse=True):
            seated = topped[t][0]
            fee = 0 if seated else (headers or {}).get(t, 0)
            have = sum(s.token_count for s in seated)
            trial = self._pack_type(by_type[t], have + pool - fee)
            took = sum(s.token_count for s in trial[0]) - have
            if took > 0:
                topped[t] = trial
                pool -= took + fee
        return topped

    @staticmethod
    def _lent(
        caps: dict[SectionType, int],
        spare: int,
        receivers: list[SectionType],
        shares: dict[SectionType, float],
        spare_budget_first: Sequence[SectionType],
    ) -> dict[SectionType, int]:
        """``caps`` with ``spare`` added to the receivers': all of it to the
        first ``spare_budget_first`` type among them, else in proportion to
        their ``shares`` (equally when they are all 0)."""
        lent = dict(caps)
        first = [t for t in spare_budget_first if t in receivers]
        if first:
            lent[first[0]] += spare
            return lent
        total = sum(shares[t] for t in receivers)
        for t in receivers:
            lent[t] += int(
                spare * (shares[t] / total if total > 0 else 1 / len(receivers))
            )
        return lent

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
        # token_count covers the rendered line, any "[source_id] " prefix
        # included.
        prefix_chars = (
            len(
                self._section_line(
                    section.section_type,
                    section.source_id,
                    "-",
                    marker=section.marker,
                )
            )
            - 1
        )
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
        SectionType.QUERY_RESULT: "Graph Query Result",
        SectionType.TEXT: "Document Chunks",
        SectionType.ENTITY: "Knowledge Graph — Entities",
        SectionType.RELATIONSHIP: "Knowledge Graph — Relationships",
        SectionType.CLAIM: "Claims",
        SectionType.COMMUNITY: "Community Reports",
        SectionType.SUMMARY: "Summaries",
        SectionType.GENERAL: "Other Context",
    }

    @classmethod
    def _section_header(cls, section_type: SectionType) -> str:
        return f"##### {cls._SECTION_LABELS[section_type]}"

    # In an unnumbered context, entity and relationship lines carry no
    # "[id] " prefix: nothing reads their ids (sources map back by section),
    # and a UUID prefix is ~9 tokens (estimated; more for a BPE tokenizer) on
    # lines of a few dozen. A numbered context gives every section its "[n] "
    # citation marker instead (about 3 tokens).
    _UNPREFIXED: ClassVar[frozenset[SectionType]] = frozenset(
        {SectionType.ENTITY, SectionType.RELATIONSHIP}
    )

    @classmethod
    def _section_line(
        cls,
        section_type: SectionType,
        source_id: str,
        content: str,
        *,
        marker: int | None = None,
        numbered: bool = False,
    ) -> str:
        """How build_context_string renders one section ("" for no content):
        after its citation marker when it has one, else (and for a
        synthesized section of a ``numbered`` context, which has none) as
        before numbering."""
        if not content:
            return ""
        if marker is not None:
            return f"[{marker}] {content}"
        if numbered or section_type in cls._UNPREFIXED:
            return content
        return f"[{source_id}] {content}"

    @classmethod
    def _rendered(cls, section: ContextSection, numbered: bool) -> str:
        return cls._section_line(
            section.section_type,
            section.source_id,
            section.content,
            marker=section.marker if numbered else None,
            numbered=numbered,
        )

    @classmethod
    def ranked_section_lines(cls, optimized_context: OptimizedContext) -> list[str]:
        """The sections as ``build_context_string`` renders them, one entry
        per section, highest priority first (ties keep their order)."""
        ranked = sorted(
            optimized_context.sections, key=lambda s: s.priority, reverse=True
        )
        lines = (cls._rendered(s, optimized_context.numbered) for s in ranked)
        return [line for line in lines if line]

    @classmethod
    def build_context_string(cls, optimized_context: OptimizedContext) -> str:
        # Group sections by type under labeled
        # headers (mirrors LightRAG's Entities/Relationships/Chunks blocks and MS's
        # sectioned tables) instead of a flat per-item "Source Type/Priority" dump.
        # The Priority float was meaningless prompt noise; dropped. In a numbered
        # context every evidence section starts with its citation marker [n],
        # which the answer prompt cites; otherwise text chunks carry their [id]
        # and entity and relationship lines none (_section_line).
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
            # A blank line between items: an entity, relationship or claim
            # spans several lines and carries no "[id]" prefix, so a single
            # newline ran one item's last line into the next one's first.
            lines = [cls._rendered(s, optimized_context.numbered) for s in items]
            blocks.append(cls._section_header(stype) + "\n" + "\n\n".join(lines))
        return "\n\n".join(blocks).strip()
