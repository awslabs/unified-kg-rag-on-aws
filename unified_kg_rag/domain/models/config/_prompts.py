# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``custom_prompts`` section and its template validation."""

import functools
import logging
from string import Formatter

from pydantic import (
    Field,
    model_validator,
)

from ._common import ConfigSection

# One stdlib logger for every module of the config package, named as the
# package (one module before the split): a filter on it sees them all.
_logger = logging.getLogger("unified_kg_rag.domain.models.config")


class CustomPromptConfig(ConfigSection):
    graph_extraction_system: str | None = Field(
        default=None,
        description="Custom system prompt for entity and relationship extraction from text",
    )
    graph_extraction_human: str | None = Field(
        default=None,
        description="Custom human prompt for entity and relationship extraction from text",
    )
    claim_extraction_system: str | None = Field(
        default=None,
        description="Custom system prompt for extracting claims and assertions from documents",
    )
    claim_extraction_human: str | None = Field(
        default=None,
        description="Custom human prompt for extracting claims and assertions from documents",
    )
    description_summarization_system: str | None = Field(
        default=None,
        description="Custom system prompt for re-summarizing over-long merged entity/relationship descriptions",
    )
    description_summarization_human: str | None = Field(
        default=None,
        description="Custom human prompt for re-summarizing over-long merged entity/relationship descriptions",
    )
    text_chunking_system: str | None = Field(
        default=None,
        description="Custom system prompt for the intelligent chunker's "
        "boundary call (processing.chunking.chunker_type: intelligent)",
    )
    text_chunking_human: str | None = Field(
        default=None,
        description="Custom human prompt for the intelligent chunker's "
        "boundary call; it must keep {numbered_text}, {min_chunk_size} and "
        "{max_chunk_size}",
    )
    text_translation_system: str | None = Field(
        default=None,
        description="Custom system prompt for translating text units "
        "(processing.translation)",
    )
    text_translation_human: str | None = Field(
        default=None,
        description="Custom human prompt for translating text units; it must "
        "keep {text} and {target_language}",
    )
    tree_summary_system: str | None = Field(
        default=None,
        description="Custom system prompt for summarizing one cluster of the "
        "summary tree (raptor methodology)",
    )
    tree_summary_human: str | None = Field(
        default=None,
        description="Custom human prompt for summarizing one cluster of the "
        "summary tree; it must keep {passages}",
    )
    graph_refinement_system: str | None = Field(
        default=None,
        description="Custom system prompt for improving and refining extracted graph entities and relationships",
    )
    graph_refinement_human: str | None = Field(
        default=None,
        description="Custom human prompt for improving and refining extracted graph entities and relationships",
    )
    community_report_system: str | None = Field(
        default=None,
        description="Custom system prompt for generating community analysis reports",
    )
    community_report_human: str | None = Field(
        default=None,
        description="Custom human prompt for generating community analysis reports",
    )
    answer_generation_system: str | None = Field(
        default=None,
        description="Custom system prompt for generating answers from knowledge graph",
    )
    answer_generation_human: str | None = Field(
        default=None,
        description="Custom human prompt for generating answers from knowledge graph",
    )
    comprehensive_answer_system: str | None = Field(
        default=None,
        description=(
            "Custom system prompt for the comprehensive answer of the "
            "strategies search.answer_prompt_style sets to comprehensive"
        ),
    )
    comprehensive_answer_human: str | None = Field(
        default=None,
        description=(
            "Custom human prompt for the comprehensive answer of the "
            "strategies search.answer_prompt_style sets to comprehensive"
        ),
    )
    entity_extraction_system: str | None = Field(
        default=None,
        description="Custom system prompt for extracting named entities from user queries",
    )
    entity_extraction_human: str | None = Field(
        default=None,
        description="Custom human prompt for extracting named entities from user queries",
    )
    keywords_extraction_system: str | None = Field(
        default=None,
        description="Custom system prompt for dual-level (high/low) keyword extraction (LightRAG)",
    )
    keywords_extraction_human: str | None = Field(
        default=None,
        description="Custom human prompt for dual-level (high/low) keyword extraction (LightRAG)",
    )
    corpus_profile_system: str | None = Field(
        default=None,
        description="Custom system prompt for corpus profiling during prompt tuning",
    )
    corpus_profile_human: str | None = Field(
        default=None,
        description="Custom human prompt for corpus profiling during prompt tuning",
    )
    drift_primer_system: str | None = Field(
        default=None,
        description="Custom system prompt for the DRIFT primer (intermediate answer + follow-up queries)",
    )
    drift_primer_human: str | None = Field(
        default=None,
        description="Custom human prompt for the DRIFT primer (intermediate answer + follow-up queries)",
    )
    pairwise_comparison_system: str | None = Field(
        default=None,
        description="Custom system prompt for the run-eval --pairwise-judge LLM judge",
    )
    pairwise_comparison_human: str | None = Field(
        default=None,
        description="Custom human prompt for the run-eval --pairwise-judge LLM judge",
    )
    answer_span_extraction_system: str | None = Field(
        default=None,
        description=(
            "Custom system prompt for the run-eval answer-span extractor "
            "(evaluation.answer_span: extracted)"
        ),
    )
    answer_span_extraction_human: str | None = Field(
        default=None,
        description=(
            "Custom human prompt for the run-eval answer-span extractor "
            "(evaluation.answer_span: extracted)"
        ),
    )
    pairwise_overlap_system: str | None = Field(
        default=None,
        description=(
            "Custom system prompt for the shared/unique statement split of "
            "run-eval --pairwise-mode differential"
        ),
    )
    pairwise_overlap_human: str | None = Field(
        default=None,
        description=(
            "Custom human prompt for the shared/unique statement split of "
            "run-eval --pairwise-mode differential"
        ),
    )
    pairwise_differential_system: str | None = Field(
        default=None,
        description=(
            "Custom system prompt for the run-eval --pairwise-mode differential "
            "judge (judges the unique statements)"
        ),
    )
    pairwise_differential_human: str | None = Field(
        default=None,
        description=(
            "Custom human prompt for the run-eval --pairwise-mode differential "
            "judge (judges the unique statements)"
        ),
    )
    assertion_judgement_system: str | None = Field(
        default=None,
        description="Custom system prompt for the assertion evaluator's judge",
    )
    assertion_judgement_human: str | None = Field(
        default=None,
        description="Custom human prompt for the assertion evaluator's judge",
    )
    closed_book_answer_system: str | None = Field(
        default=None,
        description="Custom system prompt for the run-eval --closed-book answers",
    )
    closed_book_answer_human: str | None = Field(
        default=None,
        description="Custom human prompt for the run-eval --closed-book answers",
    )
    benchmark_activity_context_system: str | None = Field(
        default=None,
        description=(
            "Custom system prompt for the personas and tasks of "
            "prepare-benchmark assertions"
        ),
    )
    benchmark_activity_context_human: str | None = Field(
        default=None,
        description=(
            "Custom human prompt for the personas and tasks of "
            "prepare-benchmark assertions"
        ),
    )
    benchmark_questions_system: str | None = Field(
        default=None,
        description=(
            "Custom system prompt for the question generation of "
            "prepare-benchmark assertions"
        ),
    )
    benchmark_questions_human: str | None = Field(
        default=None,
        description=(
            "Custom human prompt for the question generation of "
            "prepare-benchmark assertions"
        ),
    )
    benchmark_claims_system: str | None = Field(
        default=None,
        description=(
            "Custom system prompt for the claim extraction (map step) of "
            "prepare-benchmark assertions"
        ),
    )
    benchmark_claims_human: str | None = Field(
        default=None,
        description=(
            "Custom human prompt for the claim extraction (map step) of "
            "prepare-benchmark assertions"
        ),
    )
    benchmark_assertions_system: str | None = Field(
        default=None,
        description=(
            "Custom system prompt for the assertion synthesis (reduce step) of "
            "prepare-benchmark assertions"
        ),
    )
    benchmark_assertions_human: str | None = Field(
        default=None,
        description=(
            "Custom human prompt for the assertion synthesis (reduce step) of "
            "prepare-benchmark assertions"
        ),
    )
    benchmark_validation_system: str | None = Field(
        default=None,
        description=(
            "Custom system prompt for the assertion validation of "
            "prepare-benchmark assertions"
        ),
    )
    benchmark_validation_human: str | None = Field(
        default=None,
        description=(
            "Custom human prompt for the assertion validation of "
            "prepare-benchmark assertions"
        ),
    )
    drift_hyde_system: str | None = Field(
        default=None,
        description="Custom system prompt for the DRIFT query expansion (HyDE)",
    )
    drift_hyde_human: str | None = Field(
        default=None,
        description="Custom human prompt for the DRIFT query expansion (HyDE)",
    )
    iterative_controller_system: str | None = Field(
        default=None,
        description="Custom system prompt for iterative search's controller, "
        "which judges whether the context answers the question and names "
        "the bridge entities and follow-up queries",
    )
    iterative_controller_human: str | None = Field(
        default=None,
        description="Custom human prompt for iterative search's controller",
    )
    graph_query_system: str | None = Field(
        default=None,
        description="Custom system prompt for the structured strategy's "
        "planner, which writes the question as a graph query plan (JSON)",
    )
    graph_query_human: str | None = Field(
        default=None,
        description="Custom human prompt for the structured strategy's planner",
    )
    drift_action_system: str | None = Field(
        default=None,
        description="Custom system prompt for answering a DRIFT follow-up query "
        "from its local-search context",
    )
    drift_action_human: str | None = Field(
        default=None,
        description="Custom human prompt for answering a DRIFT follow-up query "
        "from its local-search context",
    )
    query_rewrite_system: str | None = Field(
        default=None,
        description="Custom system prompt for rewriting a follow-up message as a "
        "standalone question from the conversation history",
    )
    query_rewrite_human: str | None = Field(
        default=None,
        description="Custom human prompt for rewriting a follow-up message as a "
        "standalone question from the conversation history",
    )
    strategy_selection_system: str | None = Field(
        default=None,
        description="Custom system prompt for selecting search strategy based on user query",
    )
    strategy_selection_human: str | None = Field(
        default=None,
        description="Custom human prompt for selecting search strategy based on user query",
    )
    global_map_system: str | None = Field(
        default=None,
        description="Custom system prompt for the global-search map step "
        "(rate community-report key points 0-100 for relevance)",
    )
    global_map_human: str | None = Field(
        default=None,
        description="Custom human prompt for the global-search map step "
        "(rate community-report key points 0-100 for relevance)",
    )

    @model_validator(mode="after")
    def _check_override_variables(self) -> "CustomPromptConfig":
        """Reject overrides that would fail or run on missing data at call time.

        A placeholder the prompt is not called with raises ``KeyError`` on every
        call (often a literal JSON brace), and an override without a data
        variable such as ``{input_text}`` runs with no document text. Other
        declared variables are tuning knobs an override may leave out.
        """
        # Lazy: the prompt modules reference this config type.
        from unified_kg_rag.domain import prompts

        errors: list[str] = []
        brace_hint = False
        checked: set[str] = set()
        for name in prompts.__all__:
            prompt = getattr(prompts, name)
            key = getattr(prompt, "prompt_key", None)
            # Variants of one prompt share its key (and its overrides).
            if not key or key in checked:
                continue
            checked.add(key)
            system = getattr(self, f"{key}_system")
            human = getattr(self, f"{key}_human")
            if system is None and human is None:
                continue
            used: set[str] = set()
            for part, override, default in (
                ("system", system, prompt.system_prompt_template),
                ("human", human, prompt.human_prompt_template),
            ):
                try:
                    used |= _template_variables(override or default)
                except ValueError as e:
                    errors.append(f"custom_prompts.{key}_{part}: {e}")
                    brace_hint = True
            unknown = used - set(prompt.input_variables)
            if unknown:
                brace_hint = True
                errors.append(
                    f"custom_prompts.{key}_*: unknown variable(s) "
                    f"{', '.join('{' + v + '}' for v in sorted(unknown))}; "
                    f"available: "
                    f"{', '.join('{' + v + '}' for v in prompt.input_variables)}"
                )
            missing = _REQUIRED_PROMPT_VARIABLES.get(key, frozenset()) - used
            if missing:
                errors.append(
                    f"custom_prompts.{key}_*: the system or human template must "
                    f"contain {', '.join('{' + v + '}' for v in sorted(missing))}"
                )
            _warn_missing_output_tags(
                key,
                f"{system or prompt.system_prompt_template}\n"
                f"{human or prompt.human_prompt_template}",
                getattr(prompt, "required_output_tags", ()),
            )
        if errors:
            hint = " Write a literal brace as '{{' or '}}'." if brace_hint else ""
            raise ValueError("; ".join(errors) + "." + hint)
        return self


# The variables that carry each prompt's data: an override must keep them.
_REQUIRED_PROMPT_VARIABLES: dict[str, frozenset[str]] = {
    "answer_generation": frozenset({"query", "context"}),
    "answer_span_extraction": frozenset({"question", "answer"}),
    "assertion_judgement": frozenset({"question", "answer", "assertions"}),
    "benchmark_activity_context": frozenset({"context"}),
    "benchmark_assertions": frozenset({"question", "claims"}),
    "benchmark_claims": frozenset({"question", "text_units"}),
    "benchmark_questions": frozenset({"context"}),
    "benchmark_validation": frozenset({"question", "assertions", "sources"}),
    "closed_book_answer": frozenset({"query"}),
    "comprehensive_answer": frozenset({"query", "context"}),
    "claim_extraction": frozenset({"input_text"}),
    "community_report": frozenset({"entities", "relationships"}),
    "corpus_profile": frozenset({"corpus_sample"}),
    "description_summarization": frozenset({"descriptions"}),
    "tree_summary": frozenset({"passages"}),
    "drift_action": frozenset({"query", "context"}),
    "drift_hyde": frozenset({"query"}),
    "drift_primer": frozenset({"query", "community_reports"}),
    "entity_extraction": frozenset({"query"}),
    "global_map": frozenset({"query", "reports"}),
    "graph_extraction": frozenset({"input_text"}),
    "graph_query": frozenset({"query", "schema"}),
    "graph_refinement": frozenset({"text", "entities", "relationships"}),
    "iterative_controller": frozenset({"query", "context"}),
    "keywords_extraction": frozenset({"query"}),
    "pairwise_comparison": frozenset({"query", "answer1", "answer2"}),
    "pairwise_differential": frozenset({"query", "shared", "unique1", "unique2"}),
    "pairwise_overlap": frozenset({"query", "answer1", "answer2"}),
    "query_rewrite": frozenset({"query", "conversation_history"}),
    "strategy_selection": frozenset({"query", "strategies"}),
}


@functools.cache
def _warn_missing_output_tags(
    key: str, templates: str, required_tags: tuple[str, ...]
) -> None:
    """Warn when an override no longer teaches the XML tags its parser reads.

    A ``*_system`` override replaces the whole built-in system prompt, which is
    where the output schema lives for the XML-parsed prompts; without it the
    model answers in a free format the parser drops. Some users teach the
    format in prose or a human override, so this warns rather than fails.
    Once per override text: a config is validated several times per load
    (each layer, each rebuild under provider flags).
    """
    missing = [tag for tag in required_tags if tag not in templates]
    if missing:
        _logger.warning(
            "custom_prompts.%s_*: the system and human templates do not "
            "mention the output tag(s) %s that the response parser reads; "
            "keep the built-in output format (run-prompt-tuning output does) "
            "or results may be empty or ungrounded",
            key,
            ", ".join(missing),
        )


def _template_variables(template: str) -> set[str]:
    """Placeholder names in an f-string style template (``{{``/``}}`` are literal).

    Raises:
        ValueError: On an unmatched ``{`` or ``}``.
    """
    return {
        field.split(".")[0].split("[")[0]
        for _, field, _, _ in Formatter().parse(template)
        if field is not None
    }
