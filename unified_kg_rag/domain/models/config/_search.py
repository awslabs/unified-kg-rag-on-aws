# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``search`` section that groups the strategy and fusion settings."""

import functools
import logging
import math
from typing import Annotated

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    model_validator,
)

from ..retrieval import (
    SearchStrategy,
)
from ._common import ConfigSection
from ._fusion import (
    DEFAULT_ANSWER_PROMPT_STYLES,
    AbstentionConfig,
    AnswerPromptStyle,
    CitationConfig,
    FusionConfig,
    HybridConfig,
    QueryOutputCeilingsConfig,
    RerankingConfig,
    TokenManagerConfig,
    _over_default_styles,
    _reject_auto_strategy,
    _reject_auto_style,
)
from ._strategies import (
    DriftSearchConfig,
    GlobalSearchConfig,
    GraphExpansionConfig,
    IterativeSearchConfig,
    LightRAGSearchConfig,
    LocalSearchConfig,
    PPRSearchConfig,
    StructuredSearchConfig,
    TreeSearchConfig,
)

# One stdlib logger for every module of the config package, named as the
# package (one module before the split): a filter on it sees them all.
_logger = logging.getLogger("unified_kg_rag.domain.models.config")


class SearchConfig(ConfigSection):
    auto_routable_strategies: Annotated[
        list[SearchStrategy], AfterValidator(_reject_auto_strategy)
    ] = Field(
        default_factory=lambda: [
            SearchStrategy.LOCAL,
            SearchStrategy.MIX,
            SearchStrategy.GLOBAL,
            SearchStrategy.ITERATIVE,
        ],
        min_length=1,
        description=(
            "Strategies the auto router may pick; its prompt describes exactly"
            " these, in this order (under filters, only those applying every "
            "filter key; none applying them refuses the filters). simple is left "
            "out by default because the "
            "graph strategies cover its direct lookups with graph context, and"
            " drift because it costs up to 27 LLM calls a query (local: 2), a "
            "price one misrouted question should not pay; add it to route to "
            "it. iterative, for questions that chain facts through an entity "
            "they do not name, adds latency to the queries it is routed to "
            "(docs/concepts/defaults-and-evidence.md#iterative-follow-up-rounds); "
            "leave it out to keep auto off it. Only the strategies "
            "the queried index namespace serves are offered (on a "
            "LightRAG-only namespace none of local, global and drift), and "
            "when nothing is left local, else mix, else the first strategy "
            "of a held methodology is used (tree on a raptor-only "
            "namespace); with one strategy left no router call is "
            "made. auto itself is rejected, and so are ppr until a "
            "measurement shows where it wins and tree, which answered "
            "better than naive but well below the graph strategies on one "
            "corpus (docs/concepts/defaults-and-evidence.md"
            "#raptor-summary-tree); "
            "every strategy stays selectable explicitly, and one reading "
            "what the namespace does not hold is refused."
        ),
    )
    hybrid: HybridConfig = Field(
        default_factory=HybridConfig, description="Hybrid search configuration"
    )
    fusion: FusionConfig = Field(
        default_factory=FusionConfig, description="Search result fusion configuration"
    )
    reranking: RerankingConfig = Field(
        default_factory=RerankingConfig, description="Reranking configuration"
    )
    global_search: GlobalSearchConfig = Field(
        default_factory=GlobalSearchConfig, description="Global search configuration"
    )
    local_search: LocalSearchConfig = Field(
        default_factory=LocalSearchConfig, description="Local search configuration"
    )
    drift_search: DriftSearchConfig = Field(
        default_factory=DriftSearchConfig, description="Drift search configuration"
    )
    lightrag_search: LightRAGSearchConfig = Field(
        default_factory=LightRAGSearchConfig,
        description="LightRAG dual-level keyword search configuration",
    )
    iterative_search: IterativeSearchConfig = Field(
        default_factory=IterativeSearchConfig,
        description="Iterative bridge follow-up search configuration (opt-in)",
    )
    ppr_search: PPRSearchConfig = Field(
        default_factory=PPRSearchConfig,
        description="HippoRAG 2 Personalized PageRank search configuration",
    )
    tree_search: TreeSearchConfig = Field(
        default_factory=TreeSearchConfig,
        description="RAPTOR summary-tree search configuration (opt-in)",
    )
    structured: StructuredSearchConfig = Field(
        default_factory=StructuredSearchConfig,
        description="Structured graph query (structured strategy) configuration (opt-in)",
    )
    graph_expansion: GraphExpansionConfig = Field(
        default_factory=GraphExpansionConfig,
        description="Graph expansion around the seed entities (every strategy "
        "that walks the graph, on every graph store)",
    )
    token_manager: TokenManagerConfig = Field(
        default_factory=TokenManagerConfig, description="Token management configuration"
    )
    max_concurrency: int = Field(
        default=10,
        ge=1,
        description=(
            "LLM calls and store requests one query keeps in flight at each "
            "level: the global map calls, the dynamic-selection ratings, the "
            "DRIFT primer calls and follow-up local searches; and within a "
            "search, its fetch-by-id batches (ids past the live index's "
            "id_batch_size), the global report browse's scope batches and a "
            "search's filter batches (lists past max_terms_count), and a "
            "filtered graph walk's per-vertex edge requests (each asks both "
            "endpoint sides, so up to twice that many); and the output "
            "guardrail's per-result checks of the model-written results, when "
            "the one check of them all intervenes. Each bound is its "
            "own and they nest: each DRIFT follow-up search, and each batch "
            "the retriever splits again, has one, so up to the square of "
            "this many requests. Per query, so concurrent queries (run-eval "
            "keeps processing.max_concurrency of them in flight) multiply it; "
            "lower it if a query's calls are throttled."
        ),
    )
    store_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Seconds a query waits for one OpenSearch or Neptune request (and "
            "for opening a Neptune connection); a Neptune traversal, including"
            " its wait for a pooled connection, gets twice that. A store that "
            "does not answer fails the query instead of answering from no "
            "context. Ingestion keeps its own, longer limits."
        ),
    )
    query_timeout_seconds: float | None = Field(
        default=180.0,
        gt=0,
        description=(
            "Seconds one query (invoke, search or a stream) may take in all, "
            "from its admission (the read of the namespace's build record) "
            "to the saved answer; past it the query "
            "fails with QueryTimeoutError and its model and store calls in "
            "flight are cancelled (a Bedrock call already sent from a worker "
            "thread runs on through its own attempts, model_max_attempts x "
            "model_timeout_seconds at most, but the framework's transient "
            "retries start no new call past the deadline). A streamed answer "
            "must also deliver its first chunk within model_timeout_seconds "
            "of its model call. Under processing.ignore_errors the query "
            "returns the error answer instead, unless the deadline passes "
            "during its admission, which raises as a refused strategy or "
            "filter does. null disables both deadlines."
        ),
    )
    model_timeout_seconds: float = Field(
        default=90.0,
        gt=0,
        description=(
            "Read timeout of the model clients a query calls (answer and "
            "query-time LLM calls, query embeddings, Bedrock rerank): how long"
            " one request may wait for the server's next bytes, which for a "
            "non-streamed call is its whole generation. Ingestion and "
            "evaluation-judge calls keep the 330 s Bedrock read timeout. On "
            "an OpenAI-compatible or Ollama server it caps "
            "providers.llm.timeout_seconds and providers.embedding."
            "timeout_seconds for query calls."
        ),
    )
    model_max_attempts: int = Field(
        default=2,
        ge=1,
        description=(
            "Attempts, including the first, that botocore makes for one "
            "query-time Bedrock request (read timeouts, throttling, 5xx); "
            "ingestion clients make 6. With model_timeout_seconds it bounds a"
            " stalled call to about model_max_attempts x "
            "model_timeout_seconds. Throttling and model errors past it are "
            "retried with backoff within the retry section's budget "
            "(retry.max_total_seconds)."
        ),
    )
    auto_speculative_extraction: bool = Field(
        default=True,
        description=(
            "For an auto query, start the query entity and keyword "
            "extractions that the routable strategies read, and the query "
            "translation, at the same time as the router call instead of "
            "after it: auto then costs the latency of the strategy it routes "
            "to plus nothing. The router prompt and its decision are "
            "unchanged. An extraction the chosen strategy does not read keeps "
            "running in the background: the query does not wait for it, its "
            "call is in the query's usage from its start (estimated_calls "
            "until it answers), its result fills the "
            "query_extraction_cache_size cache, and it is cancelled at "
            "query_timeout_seconds or when the chain closes. With the default "
            "routable strategies that is one fast-model extraction call per "
            "auto query (two when it routes to global), paid whatever the "
            "route. false runs them after the router, only for the strategy "
            "it picked."
        ),
    )
    query_extraction_cache_size: int = Field(
        default=1024,
        ge=0,
        description=(
            "Query entity and keyword extractions a chain keeps in memory, "
            "keyed by the question, the target language, the model, the "
            "prompt text (a custom_prompts override is a new key) and the "
            "index namespace (no namespace reads by a cache hit what another "
            "was asked; a run-eval run's arms share across namespaces), least "
            "recently used dropped first. A repeated question (run-eval "
            "--repeats, the arms of a strategy comparison, a retry) reuses "
            "the extraction instead of calling the model again, so every "
            "arm searches with the same entities and keywords. A failed "
            "extraction is not kept. 0 disables it."
        ),
    )
    answer_prompt_style: Annotated[
        dict[SearchStrategy, AnswerPromptStyle],
        BeforeValidator(_over_default_styles),
        AfterValidator(_reject_auto_style),
    ] = Field(
        default_factory=lambda: dict(DEFAULT_ANSWER_PROMPT_STYLES),
        description=(
            "Answer prompt per strategy: concise (a one-sentence direct "
            "answer and brief support, custom_prompts.answer_generation_*) "
            "or comprehensive (a multi-paragraph markdown answer that "
            "explains every key point, modelled on MS GraphRAG's global "
            "reduce prompt, custom_prompts.comprehensive_answer_*). The map "
            "is merged over the default {global: comprehensive}; every other "
            "strategy is concise unless named. On sensemaking questions, "
            "comprehensive global answers won comprehensiveness, diversity and "
            "empowerment against concise ones, also with a judge that compares "
            "only what each answer adds, at a higher cost per query, and "
            "lost directness (docs/concepts/defaults-and-evidence.md"
            "#answer-style). drift stays concise: comprehensive drift won the "
            "same way, but its p95 latency came close to the 180 s "
            "query_timeout_seconds, so raise that with {drift: "
            "comprehensive}. {global: concise} restores the earlier default. "
            "An auto query uses the style of the strategy it routes to."
        ),
    )
    output_ceilings: QueryOutputCeilingsConfig = Field(
        default_factory=QueryOutputCeilingsConfig,
        description="max_tokens of the short-answer query calls",
    )
    abstention: AbstentionConfig = Field(
        default_factory=AbstentionConfig,
        description="How an answer declines a question its context does not answer",
    )
    citations: CitationConfig = Field(
        default_factory=CitationConfig,
        description="Numbered answer contexts and the answer's citations",
    )

    def query_model_timeout(self, configured: float) -> float:
        """A query-time model-server request timeout: ``configured`` (the
        provider's own), capped at ``model_timeout_seconds``."""
        return min(configured, self.model_timeout_seconds)

    @model_validator(mode="after")
    def _check_seed_floor(self) -> "SearchConfig":
        if self.local_search.seed_score_ratio > 0 and math.isclose(
            self.hybrid.lexical_weight, 0.0
        ):
            _warn_seed_floor_inert(self.local_search.seed_score_ratio)
        return self


@functools.cache
def _warn_seed_floor_inert(ratio: float) -> None:
    """Warn once that ``seed_score_ratio`` cannot filter a hybrid search
    without its lexical side: every lookup's top kNN hit fuses to 1."""
    _logger.warning(
        "search.local_search.seed_score_ratio %s has no effect on a hybrid "
        "search with search.hybrid.lexical_weight 0: every query entity's top "
        "hit scores 1, so any of them seeds graph expansion",
        ratio,
    )
