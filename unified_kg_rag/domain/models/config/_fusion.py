# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid fusion, reranking, token budgets and query output settings."""

import functools
import logging
import math
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    NonNegativeFloat,
    field_validator,
    model_validator,
)

from ..retrieval import (
    STRATEGY_TRAITS,
    SearchStrategy,
)
from ._common import ConfigSection

# One stdlib logger for every module of the config package, named as the
# package (one module before the split): a filter on it sees them all.
_logger = logging.getLogger("unified_kg_rag.domain.models.config")


class HybridConfig(ConfigSection):
    # The weights are relative: every reader divides by their sum
    # (combine_min_max, local search's seed floor), so they are checked,
    # never rewritten, and an assignment of one is validated as any other.

    lexical_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Weight of the lexical side of a "
            "hybrid search. A hybrid search runs "
            "a lexical and a kNN search on each "
            "index, each fetching at least "
            "normalization_window hits, min-max "
            "normalizes each over its hits, takes "
            "their weighted mean (lexical_weight, "
            "vector_weight) and keeps the "
            "requested number, so the ranking "
            "does not depend on how many hits a "
            "query asks for."
        ),
    )
    vector_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Weight of the vector (kNN) side of a hybrid search. The two "
            "weights are relative (each side counts its weight over their "
            "sum), and their sum must be above 0."
        ),
    )
    normalization_window: int = Field(
        default=50,
        ge=1,
        description=(
            "Hits each side of a hybrid search fetches at least (capped by "
            "the index's max_result_window) for the min-max normalization "
            "before the cut to the requested number: normalized over only "
            "the hits returned, a hit's fused score, and so the ranking, "
            "depended on how many were asked for. A larger window costs "
            "store work per search. Unmeasured choice (no A/B in "
            "docs/concepts/defaults-and-evidence.md)."
        ),
    )

    @model_validator(mode="after")
    def _check_weights(self) -> "HybridConfig":
        if math.isclose(self.lexical_weight + self.vector_weight, 0.0):
            raise ValueError(
                "The sum of vector_weight and lexical_weight cannot be zero."
            )
        return self


# Buckets no strategy emits any more: a weight keyed by one is dropped, with
# a warning (global search fuses only its text units; 0.2.3's emitted the
# candidate and expanded report buckets, and its template listed all three;
# simple fuses one bucket per index, where one weight on all of them would
# change no order).
RETIRED_FUSION_BUCKETS: frozenset[str] = frozenset(
    {
        "opensearch_all",
        "vector_all",
        "opensearch_community_reports",
        "opensearch_candidate_community_reports",
        "opensearch_expanded_community_reports",
    }
)


@functools.cache
def _warn_retired_bucket(old: str) -> None:
    _logger.warning(
        "search.fusion.fusion_weights: no search strategy emits bucket '%s'; "
        "its weight is ignored (remove the key)",
        old,
    )


class FusionConfig(ConfigSection):
    rrf_k: int = Field(
        default=60,
        ge=1,
        description=(
            "The k constant of reciprocal rank fusion: each result adds 1 / "
            "(rrf_k + rank)."
        ),
    )
    diversity_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "MMR lambda for diversity filtering: score = lambda*relevance - "
            "(1-lambda)*max_similarity, with relevance min-max normalized to "
            "[0, 1] over the fused candidates. 1.0 = pure relevance (no diversity), "
            "0.0 = maximum diversity. Lower values penalize redundant results "
            "more strongly. Filtering is skipped at 1.0 (no diversity benefit). "
            "On 200 MuSiQue mix questions 0.7 and 1.0 measured no significant "
            "answer difference against 0.5 (answer_contains -1.0 points "
            "[-3.75, +1.5] and -2.25 [-5.25, +0.5])."
        ),
    )
    fusion_weights: dict[str, NonNegativeFloat] = Field(
        default_factory=dict,
        description=(
            "Per-source-bucket weights: reciprocal rank fusion scales each "
            "bucket's 1/(rrf_k + rank) term. Keys are the source buckets the search "
            "strategies emit: local (graph_entities, text_units, relationships, "
            "bridge_relationships, community_reports, claims), LightRAG "
            "(lightrag_entities, lightrag_relationships, "
            "lightrag_endpoint_entities, lightrag_incident_relationships, "
            "lightrag_chunks, lightrag_linked_chunks (the chunks the entities "
            "cite), lightrag_relation_chunks (those the relationships cite)), "
            "simple (simple_text_units, simple_entities, "
            "simple_relationships, simple_claims, simple_community_reports, "
            "simple_summary_nodes: one per index), global (text_units), "
            "drift (results), "
            "iterative (iterative_round_<n>, round n's base search, 0 the "
            "first; iterative_round_<n>_text, its follow-up queries' text "
            "units; fused once more over the rounds, the base's own buckets "
            "fused within each), ppr (ppr_passages, ppr_seed_text_units, "
            "ppr_facts), tree (tree_text_units, tree_summaries, "
            "tree_graph_summaries, tree_graph_text_units) and structured "
            "(structured_query_result, structured_entities, "
            "structured_relationships, structured_text_units, "
            "structured_text_hits, structured_fallback). A "
            "bucket without a key weighs 1.0 (the default: none); a weight is "
            "0 or more (0 drops the bucket's contribution). A weight on "
            "a retired bucket (opensearch_all, vector_all, "
            "opensearch_community_reports, "
            "opensearch_candidate_community_reports, "
            "opensearch_expanded_community_reports) is dropped, with a warning."
        ),
    )

    @field_validator("fusion_weights", mode="before")
    @classmethod
    def _retired_buckets(cls, value: Any) -> Any:
        """Load a weight keyed by a retired bucket (a dict key the
        moved-key mechanism cannot reach) with a warning."""
        if not isinstance(value, dict):
            return value
        weights = dict(value)
        for old in RETIRED_FUSION_BUCKETS:
            if old in weights:
                _warn_retired_bucket(old)
                weights.pop(old)
        return weights


class RerankingConfig(ConfigSection):
    """How the fused results are reranked with models.rerank.

    providers.rerank.kind picks the reranker; none turns reranking off.
    Bedrock rerank models are not offered in every region, and the Rerank API
    needs its own IAM statement (bedrock:Rerank on Resource '*').
    """

    top_k: int = Field(
        default=100,
        ge=1,
        description=(
            "Fused candidates sent to the reranker, the best fused first; "
            "the others are dropped from the result (their fused scores do "
            "not compare with the reranker's), so keep it at least the "
            "result width a strategy asks for (top_k x retrieval_multiplier, "
            "or the sum of its per-type quotas)."
        ),
    )
    required: bool = Field(
        default=False,
        description=(
            "Fail the query with RerankModelError when the rerank model "
            "cannot be built, the rerank call fails or it returns nothing "
            "usable. false keeps the fused order and reports the outcome in "
            "SearchResult.metadata['rerank']. No effect with "
            "providers.rerank.kind none."
        ),
    )


class ContextTypeBudgetConfig(ConfigSection):
    """Per-section-type share of the answer context window.

    Both upstream methodologies split the context window into per-type
    sub-budgets and pack each type independently, so document chunks (which hold
    the multi-hop answer) are guaranteed representation instead of competing in a
    flat greedy fill of near-tie fusion scores: MS GraphRAG's ``mixed_context``
    packs community / entity / text-unit sections against independent budgets
    (``community_prop`` 0.15, ``text_unit_prop`` 0.5 by default), and LightRAG's
    ``_apply_token_truncation`` trims entities / relations / chunks against
    separate token limits (6000 / 8000 / the remainder, about 16000, of a
    30000 window).

    These are shares of ``max_context_tokens``, not upstream's numbers: at
    30000, LightRAG mix/hybrid (text, entities, relationships) get shares of
    18072, 7228 and 4698 tokens, against upstream's ~16000 / 6000 / 8000. The
    512-token prompt buffer, the question and the section headers come off
    them, so about 17700 / 7050 / 4600 are packed (17746 / 7050 / 4614
    measured).

    Shares are relative weights, not fractions: they are renormalized over the
    section types actually present in a query's candidate set, so an absent type
    does not shrink the usable window. Budget a type leaves unused because all
    of its candidates fit goes to the types that still have candidates, by
    share (LightRAG mix/hybrid give it to chunks first, as upstream does).
    Raise ``text`` to favour verbatim source passages, raise ``community`` to
    favour high-level synthesis.
    """

    text: float = Field(
        default=0.50, ge=0.0, description="Relative share for document chunk sections"
    )
    entity: float = Field(
        default=0.20, ge=0.0, description="Relative share for graph entity sections"
    )
    relationship: float = Field(
        default=0.13,
        ge=0.0,
        description="Relative share for graph relationship sections",
    )
    community: float = Field(
        default=0.10,
        ge=0.0,
        description="Relative share for community report sections",
    )
    claim: float = Field(
        default=0.07,
        ge=0.0,
        description="Relative share for claim (covariate) sections",
    )
    general: float = Field(
        default=0.10,
        ge=0.0,
        description="Relative share for untyped sections — global search's "
        "map-reduce synthesis lands here.",
    )
    summary: float = Field(
        default=0.50,
        ge=0.0,
        description="Relative share for summary-tree sections (the tree "
        "strategy only, beside text's 0.50: at most half the context, and "
        "the text units take what the summaries leave)",
    )
    query_result: float = Field(
        default=0.25,
        ge=0.0,
        description="Relative share for a graph query result (the structured "
        "strategy only): its table of counts, groups, rows or paths, at most "
        "search.structured.rows_in_context long",
    )

    @model_validator(mode="after")
    def validate_at_least_one_positive_share(self) -> "ContextTypeBudgetConfig":
        if not any(
            share > 0.0
            for share in (
                self.text,
                self.entity,
                self.relationship,
                self.community,
                self.claim,
                self.general,
            )
        ):
            raise ValueError(
                "At least one context type budget share must be greater than zero; "
                "all-zero shares would leave every answer context empty."
            )
        return self


class TokenManagerConfig(ConfigSection):
    max_context_tokens: int | None = Field(
        default=30_000,
        ge=1024,
        description=(
            "Retrieval context budget of the answer prompt, in tokens. 30,000 "
            "matches upstream LightRAG's total context budget (MS GraphRAG "
            "uses 12,000); a budget derived from a 1M-token window (~785K) "
            "would never bind, so the per-type budgets would never trim "
            "anything. Always clamped to what models.roles.answer_generation "
            "accepts beside its output reservation "
            "(models.default_max_output_tokens), with a warning. null "
            "derives it from that window minus the output reservation and "
            "context_window_headroom_ratio."
        ),
    )
    context_window_headroom_ratio: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of the answer model's context window held back when "
            "deriving max_context_tokens, covering the system prompt, the "
            "question, conversation history, and tokenizer estimation error."
        ),
    )
    token_count_cache_size: int = Field(
        default=1024,
        ge=1,
        description="Maximum number of entries in the LRU cache of the answer "
        "model's token counter (Bedrock CountTokens, or the local estimate)",
    )
    type_budgets: ContextTypeBudgetConfig = Field(
        default_factory=ContextTypeBudgetConfig,
        description="Per-section-type shares of the context window",
    )
    min_truncated_section_tokens: int = Field(
        default=64,
        ge=1,
        description="A section truncated below this many tokens is dropped rather "
        "than emitted as an unusable fragment of evidence.",
    )


def _reject_auto_strategy(strategies: list[SearchStrategy]) -> list[SearchStrategy]:
    for strategy in strategies:
        reason = STRATEGY_TRAITS[strategy].not_routable_reason
        if reason is not None:
            raise ValueError(
                f"auto_routable_strategies cannot include '{strategy.value}': "
                f"{reason}"
            )
    return strategies


AnswerPromptStyle = Literal["concise", "comprehensive"]


# The answer prompt styles a config that sets none gets; a configured map is
# merged over them, so naming one strategy leaves the others' defaults.
DEFAULT_ANSWER_PROMPT_STYLES: dict[SearchStrategy, AnswerPromptStyle] = {
    SearchStrategy.GLOBAL: "comprehensive"
}


def _over_default_styles(value: Any) -> Any:
    if isinstance(value, dict):
        return {**DEFAULT_ANSWER_PROMPT_STYLES, **value}
    return value


def _reject_auto_style(
    value: dict[SearchStrategy, AnswerPromptStyle],
) -> dict[SearchStrategy, AnswerPromptStyle]:
    if SearchStrategy.AUTO in value:
        raise ValueError(
            "answer_prompt_style is keyed by the strategy a query runs; auto "
            "resolves to one of them, so set the style on those instead"
        )
    return value


class QueryOutputCeilingsConfig(ConfigSection):
    """max_tokens of the short-answer query calls.

    Bedrock reserves input + max_tokens against the tokens-per-minute quota
    when a request starts, so a router that answers one word but asks for
    models.default_max_output_tokens reserves as much as the answer.
    A ceiling never leaves less than the call's thinking budget plus 1,024
    answer tokens (thinking counts toward max_tokens): the effort level's
    budget (low 2,048, medium 4,096, high 8,192, xhigh 16,384, max 32,768)
    on adaptive-thinking and OpenAI reasoning models, the thinking
    budget_tokens on older Claude models, none on a model that does not
    reason. null leaves the call at the default cap. A local server takes
    no thinking budget, so a model of it that may reason (an Ollama model
    listing thinking, any OpenAI-compatible one) keeps its whole cap.
    """

    strategy_selection: int | None = Field(
        default=1024,
        ge=1,
        description=(
            "max_tokens of the auto router call (models.roles."
            "strategy_selection), which answers one strategy name (a few "
            "tokens; a reasoning model still gets its thinking budget plus "
            "1,024)"
        ),
    )
    query_extraction: int | None = Field(
        default=4096,
        ge=1,
        description=(
            "max_tokens of the query entity extraction (local, drift) and "
            "the dual-keyword extraction (mix, hybrid)"
        ),
    )
    community_relevance: int | None = Field(
        default=4096,
        ge=1,
        description=(
            "max_tokens of each report rating of global search's dynamic "
            "selection (search.global_search.use_dynamic_selection), which "
            "answers one number"
        ),
    )
    graph_query: int | None = Field(
        default=4096,
        ge=1,
        description=(
            "max_tokens of the structured strategy's plan call (models.roles."
            "graph_query), which answers one JSON plan of a few hundred tokens"
        ),
    )


class AbstentionConfig(ConfigSection):
    """How an answer declines a question its context does not answer.

    Opt-in: with the defaults the answer prompts and outputs are those of
    earlier releases (only an empty context is answered with the fixed
    no-context reply). Guide: docs/guides/query.md#declined-answers
    """

    mode: Literal["off", "mark"] = Field(
        default="off",
        description=(
            "off: the answer prompts are unchanged, and only an empty context "
            "is reported abstained (metadata.abstained, the fixed no-context "
            "reply). mark: the answer prompt (concise, comprehensive or a "
            "custom_prompts override) also asks the model to start a declined "
            "answer with the fixed token [NO_ANSWER] and say in the query's "
            "language what the context lacks; an answer starting with it is "
            "reported abstained (metadata.abstention_reason: marker) whatever "
            "its language. The instruction changes the answer prompt, so "
            "answers and EM/F1 differ from off: compare mark runs as their own "
            "arm. Opt-in: on multi-hop questions it cost exact-match points "
            "for machine-readable declines (docs/concepts/"
            "defaults-and-evidence.md#declines-and-abstention)."
        ),
    )
    strip_marker: bool = Field(
        default=True,
        description=(
            "Remove the [NO_ANSWER] token from a declined answer (invoke and "
            "stream), leaving the model's explanation; an answer that is only "
            "the token becomes the fixed no-context reply. false returns it "
            "as written."
        ),
    )
    on_insufficient: Literal["hedge", "abstain"] = Field(
        default="hedge",
        description=(
            "What the answer step does when the strategy flagged its context "
            "insufficient (SearchResult.metadata.context_sufficient false; "
            "iterative sets it when its controller still finds the context "
            "insufficient when it stops, other strategies never do). "
            "hedge: the model answers what the context supports and says, in "
            "the query's language, what it does not cover. abstain: the model "
            "is told not to answer and writes the decline in the query's "
            "language; the output is reported abstained "
            "(abstention_reason: insufficient_context). Neither adds a call. "
            "hedge is the default: with iterative, abstain declined many "
            "answerable questions (docs/concepts/"
            "defaults-and-evidence.md#declines-and-abstention)."
        ),
    )


# The strategies whose answers cite by default: every one but these. A
# configured map is merged over them, so naming one strategy leaves the
# others' defaults. global and DRIFT answer from model-written synthesis
# (key points, intermediate answers), and their uncited answers won
# comprehensiveness (docs/concepts/defaults-and-evidence.md#citations).
DEFAULT_CITATION_STRATEGIES: dict[SearchStrategy, bool] = {
    SearchStrategy.GLOBAL: False,
    SearchStrategy.DRIFT: False,
}


def _over_default_citations(value: Any) -> Any:
    if isinstance(value, dict):
        return {**DEFAULT_CITATION_STRATEGIES, **value}
    return value


def _reject_auto_citations(
    value: dict[SearchStrategy, bool],
) -> dict[SearchStrategy, bool]:
    if SearchStrategy.AUTO in value:
        raise ValueError(
            "citations.strategies is keyed by the strategy a query runs; auto "
            "resolves to one of them, so set it on those instead"
        )
    return value


class CitationConfig(ConfigSection):
    """Numbered answer contexts and the answer's citations
    (``RAGOutput.citations``). Guide: docs/guides/query.md#citations
    """

    # An assigned strategies map merges over the default as a loaded one.

    enabled: bool = Field(
        default=True,
        description=(
            "Number the answer context's evidence sections [1], [2], ..., ask "
            "the answer model (concise, comprehensive or a custom_prompts "
            "override) to cite them with those markers in every language, and "
            "return RAGOutput.citations: per number its documents (file name, "
            "relative path, pages), text units and seed-graph provenance, and "
            "whether the answer cites it. The markers stay in the answer text. "
            "false: the context labels passages by id and the prompts are "
            "those of earlier releases; no citations. The setting of every "
            "strategy that strategies does not name. On local and mix, answers "
            "with and without citations scored alike "
            "(docs/concepts/defaults-and-evidence.md#citations)."
        ),
    )
    strategies: Annotated[
        dict[SearchStrategy, bool],
        BeforeValidator(_over_default_citations),
        AfterValidator(_reject_auto_citations),
    ] = Field(
        default_factory=lambda: dict(DEFAULT_CITATION_STRATEGIES),
        description=(
            "Per strategy, whether its answers cite, over enabled (an entry "
            "wins either way). The map is merged over the default {global: "
            "false, drift: false}: on 50 sensemaking questions their answers "
            "without citations won comprehensiveness (global 0.62, drift "
            "0.64, a pairwise judge averaged over both answer orders) "
            "(docs/concepts/defaults-and-evidence.md#citations). {global: "
            "true, drift: true} turns them back on, with the plain citation "
            "instruction. An auto query follows the strategy it routes to, "
            "and iterative follows its base (iterative_search.base) unless "
            "it is named itself."
        ),
    )

    def enabled_for(
        self, strategy: SearchStrategy, base: SearchStrategy | None = None
    ) -> bool:
        """Whether answers of ``strategy`` cite: its ``strategies`` entry,
        else that of the strategy it wraps (``base``: iterative's), else
        ``enabled``."""
        for key in (strategy, base):
            if key is not None and key in self.strategies:
                return self.strategies[key]
        return self.enabled

    lineage_units: int = Field(
        default=10,
        ge=1,
        description=(
            "Text units of a graph section's lineage (entity, relationship, "
            "claim, report, summary) that name its documents, those in the "
            "context first, counting only units the query's suffix and "
            "filters let through. The others are read in batched fetches by "
            "id, in pages that double what a section still lacks, at most 4 "
            "a query (a section reads at most 15 times this many of its "
            "units; one whose units read by then all fail the filters names "
            "no document). Unmeasured: bounds the reads for hub entities and "
            "reports with thousands of units."
        ),
    )
