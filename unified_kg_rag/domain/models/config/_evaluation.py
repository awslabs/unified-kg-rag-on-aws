# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``evaluation`` section and model prices."""

import logging
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from ..evaluation import EvaluationMetricType, EvaluatorType, RagasMetricType
from ._models import EffortLevel

# One stdlib logger for every module of the config package, named as the
# package (one module before the split): a filter on it sees them all.
_logger = logging.getLogger("unified_kg_rag.domain.models.config")


class ModelPrice(BaseModel):
    """A model's on-demand price, for run-eval's USD cost estimate: a
    language model's token prices, or a rerank model's search-unit price."""

    model_config = ConfigDict(extra="forbid")

    input_per_million_tokens: float | None = Field(
        default=None,
        ge=0,
        description=(
            "USD per million input tokens (a language model's; set with "
            "output_per_million_tokens)."
        ),
    )
    output_per_million_tokens: float | None = Field(
        default=None,
        ge=0,
        description=(
            "USD per million output tokens (a language model's; set with "
            "input_per_million_tokens)."
        ),
    )
    per_thousand_rerank_search_units: float | None = Field(
        default=None,
        ge=0,
        description=(
            "A rerank model's USD per thousand search units (usage "
            "rerank_search_units: one per rerank request of up to 100 "
            "documents, a document of more than 500 estimated tokens with "
            "the query counting as several; Amazon Bedrock's price per "
            "1,000 queries). An entry "
            "sets this, the two token prices, or both."
        ),
    )
    cache_read_per_million_tokens: float | None = Field(
        default=None,
        ge=0,
        description=(
            "USD per million input tokens read from the prompt cache "
            "(usage cache_read_input_tokens; Claude on Bedrock: a tenth of "
            "the input price). null = the input price, an upper bound."
        ),
    )
    cache_write_per_million_tokens: float | None = Field(
        default=None,
        ge=0,
        description=(
            "USD per million input tokens written to the prompt cache "
            "(usage cache_write_input_tokens; Claude on Bedrock: 1.25 x the "
            "input price for the 5-minute cache). null = the input price."
        ),
    )

    @model_validator(mode="after")
    def _priced(self) -> "ModelPrice":
        tokens = (self.input_per_million_tokens, self.output_per_million_tokens)
        if (tokens[0] is None) != (tokens[1] is None):
            raise ValueError(
                "a model price sets both input_per_million_tokens and "
                "output_per_million_tokens, or neither (a rerank model's)"
            )
        if tokens[0] is None and self.per_thousand_rerank_search_units is None:
            raise ValueError(
                "a model price sets input_per_million_tokens and "
                "output_per_million_tokens (a language model), or "
                "per_thousand_rerank_search_units (a rerank model)"
            )
        return self


# Evaluator types no longer built -> why; enabled_evaluators drops them with
# a warning (EvaluatorType keeps the member, so saved runs holding their
# reports still load for run-eval --from-results).
REMOVED_EVALUATORS: dict[str, str] = {
    "langchain": (
        "its correctness and partial_correctness repeated ragas' "
        "answer_correctness: enable ragas for an LLM-judged correctness score"
    ),
}


def _without_removed_evaluators(value: Any) -> Any:
    """``enabled_evaluators`` without the removed types, each warned about."""
    if not isinstance(value, list | tuple):
        return value
    kept = []
    for item in value:
        name = item.value if isinstance(item, Enum) else item
        if isinstance(name, str) and name in REMOVED_EVALUATORS:
            _logger.warning(
                "evaluation.enabled_evaluators: the '%s' evaluator is removed "
                "and ignored: %s",
                name,
                REMOVED_EVALUATORS[name],
            )
            continue
        kept.append(item)
    return kept


class EvaluationConfig(BaseModel):
    outputs_directory: str | Path = Field(
        default="outputs/evaluation",
        description=("Directory the evaluation results are written to."),
    )
    run_subdirectories: bool = Field(
        default=True,
        description=(
            "Write each run-eval invocation's files into a subdirectory of "
            "outputs_directory (or --output-directory) of its own, named "
            "run_<timestamp>_<id> (the run manifest's run_id), so the runs of "
            "different invocations are never mixed in one directory. "
            "run-eval --from-results and --baseline-results read a directory "
            "and its run subdirectories. false writes into the directory "
            "itself (the behaviour before)."
        ),
    )
    enabled_evaluators: Annotated[
        list[EvaluatorType], BeforeValidator(_without_removed_evaluators)
    ] = Field(
        default=[
            EvaluatorType.ANSWER_MATCH,
            EvaluatorType.RETRIEVAL,
            EvaluatorType.GRAPH_AWARE,
        ],
        description=(
            "List of evaluator types to enable for this evaluation run "
            "(run-eval --evaluators overrides it). The default is the "
            "deterministic ones (answer_match, retrieval, graph_aware): free, "
            "no extra, and each skips a query that lacks its dataset fields. "
            "The LLM judges are opt-in (add them to this list): ragas (the "
            "eval extra; answer_correctness and four more metrics, several "
            "judge calls per answer, context_precision one per context up to "
            "ragas_max_contexts) and assertion. 'langchain' was removed "
            "(ragas' answer_correctness measures its correctness): a list "
            "that names it loads with a warning and without it."
        ),
    )
    ragas_metrics: list[RagasMetricType] = Field(
        default=[
            EvaluationMetricType.ANSWER_CORRECTNESS,
            EvaluationMetricType.ANSWER_RELEVANCY,
            EvaluationMetricType.CONTEXT_PRECISION,
            EvaluationMetricType.CONTEXT_RECALL,
            EvaluationMetricType.FAITHFULNESS,
        ],
        description=(
            "The metrics the ragas evaluator scores: answer_correctness, "
            "answer_relevancy, context_precision, context_recall and "
            "faithfulness. Another metric fails the load."
        ),
    )
    max_context_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Token cap on the retrieved context passed to the RAGAS judges. "
            "null (default) uses the answer model's budget "
            "(search.token_manager.max_context_tokens), so faithfulness and "
            "context_recall judge the answer against the context it was given; "
            "with that budget derived from the model window too, no extra cap "
            "applies (the contexts are already the answer model's selection). A "
            "value below the answer budget logs a warning: the judge then sees "
            "less context than the answer model did."
        ),
    )
    judge_effort: EffortLevel | None = Field(
        default="low",
        description=(
            "Reasoning effort for the LLM judges (ragas, assertion, "
            "--pairwise-judge and the answer_span 'extracted' calls) "
            "on adaptive-thinking models (Claude 4.7+). Judge prompts are short "
            "extraction/classification calls, so a low effort keeps each metric "
            "well inside the RAGAS timeout. null inherits the effort of the "
            "judge model's tier (aws.bedrock.default_effort by default). "
            "Ignored by models without adaptive thinking."
        ),
    )
    ragas_timeout: int = Field(
        default=300,
        ge=1,
        description=(
            "RAGAS RunConfig.timeout in seconds: the budget for scoring ONE "
            "metric on ONE sample, covering all of its LLM calls and retries. "
            "A timed-out job yields NaN for that metric. RAGAS' own default "
            "(180s) is too tight for thinking judges."
        ),
    )
    ragas_max_contexts: int | None = Field(
        default=20,
        ge=1,
        description=(
            "Maximum number of top-ranked retrieved contexts per sample scored "
            "by RAGAS context_precision, applied before max_context_tokens. "
            "Context precision makes one judge call per context, so its cost "
            "scales with the context count; strategies that report 100+ "
            "sources time out otherwise. Only that metric is capped (it becomes "
            "context_precision@N); faithfulness and context_recall see every "
            "context within max_context_tokens, and the answer model's context "
            "is unaffected. null disables the cap."
        ),
    )
    ragas_max_workers: int = Field(
        default=8,
        ge=1,
        description=(
            "RAGAS RunConfig.max_workers: concurrent metric jobs. Lower it if "
            "Bedrock throttles the judge model (throttled calls burn the "
            "per-job timeout)."
        ),
    )
    answer_span: Literal["full", "first_line", "extracted"] = Field(
        default="first_line",
        description=(
            "The part of a generated answer the answer_match exact_match and "
            "token_f1 scores read. The benchmarks they come from (SQuAD, "
            "MuSiQue, HotpotQA) score a short predicted answer span, while "
            "the answer prompt writes a one-sentence answer and then its "
            "support. 'first_line' (default, free) scores the first line that "
            "is neither empty nor a Markdown heading: the one-sentence answer, "
            "still longer than a span, so EM under-counts. 'extracted' asks "
            "the models.roles.answer_span model (fast tier) for the short "
            "span, one call per distinct answer, cached for the run; a call "
            "that fails falls back to 'first_line' and is counted in the "
            "summary's answer_span_statistics. Use it, with "
            "answer_normalization official_squad, for numbers close to the "
            "papers'. 'full' scores the whole answer, so the score measures "
            "answer length as much as correctness. answer_contains always "
            "reads the whole answer. "
            "Recorded in the run manifest (answer_scoring) and each "
            "answer_match report."
        ),
    )
    answer_span_output_ceiling: int | None = Field(
        default=4096,
        ge=1,
        description=(
            "max_tokens of each answer_span 'extracted' call, which answers "
            "a short span; it never leaves less than the call's thinking "
            "budget (judge_effort's, see search.output_ceilings) plus 1,024 "
            "tokens. null = models.default_max_output_tokens."
        ),
    )
    answer_normalization: Literal["default", "official_squad", "hotpotqa"] = Field(
        default="default",
        description=(
            "Normalization of the answer_match exact_match and token_f1 scores. "
            "'default' keeps meaning-bearing punctuation (minus signs, decimal "
            "points, 'C++') and folds Unicode; 'official_squad' is the official "
            "SQuAD v1.1 script (lowercase, ASCII punctuation and English "
            "articles removed), which MuSiQue's evaluation uses too; "
            "'hotpotqa' adds the HotpotQA script's rule on top: a prediction or "
            "gold of 'yes', 'no' or 'noanswer' scores token F1 0 unless both "
            "are equal. The scores compare with published numbers only when "
            "the scored text is a short span (answer_span 'extracted'). "
            "answer_contains always uses 'default'."
        ),
    )
    pairwise_overall: bool = Field(
        default=False,
        description=(
            "Ask the run-eval --pairwise-judge judge for an 'overall' winner "
            "too, as LightRAG's evaluation prompt does, reported as a fifth "
            "criterion. A custom_prompts.pairwise_comparison_* override must "
            "then return an 'overall' verdict as well."
        ),
    )
    pairwise_mode: Literal["standard", "differential"] = Field(
        default="standard",
        description=(
            "How the run-eval --pairwise-judge judge reads two answers "
            "(run-eval --pairwise-mode overrides it). 'standard' (default) "
            "judges the whole answers. 'differential' first asks the "
            "models.roles.pairwise_overlap model (fast tier) for the "
            "statements both answers make and those only one makes, then "
            "judges each criterion on the unique statements only, so content "
            "both share, and the length it adds, earns neither side credit; "
            "a pair with no unique statement is a tie without a judge call. "
            "One more fast call per pair. Opt-in until a measured experiment "
            "decides: LLM judges prefer longer answers (Zeng et al. 2025). "
            "Both modes report win rates per answer-length bucket."
        ),
    )
    pairwise_repeats: Literal["first", "all"] = Field(
        default="first",
        description=(
            "Which repeats the run-eval --pairwise-judge judge reads "
            "(run-eval --pairwise-repeats overrides it). 'first' (default) "
            "judges the first repeat's answers: one judged pair per query and "
            "strategy (two judge calls, both orders; three in the differential "
            "mode). 'all' judges repeat r against repeat r for every repeat, "
            "so the win rate also averages the answers' run-to-run variance, "
            "at that cost times --repeats; each query's repeats are averaged "
            "before the queries, and its interval resamples queries."
        ),
    )
    assertion_trials: int = Field(
        default=3,
        ge=1,
        description=(
            "Judge calls per answer of the assertion evaluator: each labels "
            "every assertion of the item supported, contradicted or absent "
            "(assertions shuffled per trial, in an order no earlier trial "
            "used while one is left), and an assertion takes the label a "
            "majority (more than half) of the trials that succeeded gave it. "
            "A deterministic judge (temperature 0) repeats itself on an item "
            "with fewer orders than trials (one assertion, or two with three "
            "trials). A query whose trials all failed records "
            "its metrics as failed, never as 0."
        ),
    )
    retrieval_k: int = Field(
        default=5,
        ge=1,
        description=(
            "Cutoff k for the retrieval evaluator's hit@k / recall@k: the number "
            "of top-ranked distinct files (of the reported sources, in rank "
            "order) compared with reference_sources."
        ),
    )
    save_detailed_results: bool = Field(
        default=True,
        description=(
            "Whether to save detailed evaluation results and reports "
            "(run-eval --from-results and --baseline-results read them)."
        ),
    )
    run_order: Literal["interleaved", "sequential"] = Field(
        default="interleaved",
        description=(
            "How run-eval answers the runs of a comparison (several "
            "strategies, --repeats, --closed-book; run-eval --run-order "
            "overrides it). 'interleaved' (default) answers every query with "
            "every run before the next query, the runs' order alternating "
            "(ABBA), in one batch under processing.max_concurrency, so the "
            "latency and cost deltas do not measure what changed between "
            "runs answered one after another (service load, warm caches). "
            "'sequential' answers each run's queries in turn, the order "
            "before; the comparison then warns that its latency deltas are "
            "confounded with time. Recorded in the run manifest (run_order)."
        ),
    )
    repeat_extractions: Literal["resample", "reuse"] = Field(
        default="resample",
        description=(
            "Whether the run-eval --repeats of a strategy extract the "
            "query's entities and keywords again (run-eval "
            "--repeat-extractions overrides it). 'resample' (default) gives "
            "each repeat its own query extraction cache scope, so repeat_stdev "
            "includes the extraction's run-to-run variance; the arms of one "
            "repeat still share their extractions. It costs the fast-model "
            "extraction calls of a query once per repeat instead of once. "
            "'reuse' shares them across repeats (the behaviour before), so "
            "the repeats only resample the answer. No effect with one repeat "
            "or search.query_extraction_cache_size 0."
        ),
    )
    primary_metric: str | None = Field(
        default=None,
        pattern=r"^[^/\s]+/[^/\s]+$",
        description=(
            "The <evaluator>/<metric> a run-eval comparison declares in "
            "advance as its primary metric (run-eval --primary-metric "
            "overrides it): its paired delta is adjusted only over the "
            "compared strategies, every other metric is Holm-adjusted over "
            "all of them (deltas_vs_baseline.<s>.<metric>.adjusted_p_value, "
            "adjusted_ci_low/high), so reading many metrics does not "
            "multiply the chance of a difference on noise. null = "
            "answer_match/answer_contains when the comparison scored it, "
            "else no primary metric."
        ),
    )
    resamples: int = Field(
        default=2000,
        ge=100,
        description=(
            "Resamples of a run-eval comparison's paired bootstrap intervals "
            "and Monte Carlo sign-flip p-values (comparison bootstrap."
            "resamples). The Holm-adjusted families read at least 20 x tests "
            "/ alpha of them (multiplicity.families.<f>.resamples: 42,000 for "
            "105 tests at 0.05), so a large family is not held above alpha by "
            "the p-value's floor 1 / (1 + resamples); raise it for more "
            "stable unadjusted intervals, at a cost linear in it."
        ),
    )
    prices: dict[str, ModelPrice] = Field(
        default_factory=dict,
        description=(
            "USD prices per model id, as the usage records name them "
            "(usage.by_model: the id the model factory resolved, an inference "
            "profile id included), for the comparison's cost estimate "
            "(cost.strategies.<s>.usd_per_query). Empty (default): no USD "
            "estimate, token counts only. No prices ship with the package: "
            "fill in the current Amazon Bedrock on-demand prices of your "
            "region. input_tokens includes the cache reads and writes; each "
            "is priced at its own rate when the model's price sets one. The "
            "rerank model is priced too (per_thousand_rerank_search_units; "
            "a local cross-encoder at 0): an answer with an unpriced model "
            "that was called, reranker included, has no USD estimate."
        ),
    )
