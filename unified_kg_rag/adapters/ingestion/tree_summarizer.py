# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The summaries of RAPTOR's summary tree (docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in).

One call per cluster (``TreeSummaryPrompt``, ``models.roles.summary_tree``),
fanned out through ``BatchProcessor`` with the run's LLM response cache, as
the other ingestion model calls are. A cluster whose input key (its
children's texts, the language and the summary settings) is a live node's
takes that node's summary and makes no call.

The prompt's length hint comes from the token counter that bounds the
summary: the counter's tokens per word (per character for Japanese, Chinese
and Korean) over the cluster's own texts turn ``max_summary_tokens`` into a
word or character count, so a model that writes what it is asked for stays
within the budget whatever the counter (a script-aware estimate scaled for
the model's tokenizer, or tiktoken) and the language. A summary still over
the budget is cut at its last sentence end within it
(``shared.utils.sentences``), at a character only when its first sentence
is longer than the budget.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser

from unified_kg_rag.adapters.llm.chain_factory import setup_chain, usage_model_id
from unified_kg_rag.adapters.llm.langchain import BatchProcessor
from unified_kg_rag.adapters.llm.retry import transient_error_predicate
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.ingestion.cache_keys import (
    index_language,
    tree_summary_fingerprint,
)
from unified_kg_rag.domain.ingestion.summary_tree import Cluster, summary_input_key
from unified_kg_rag.domain.languages import profiles
from unified_kg_rag.domain.models import (
    Config,
    LanguageCode,
    ModelPurpose,
    ModelRole,
    language_name,
    model_id_value,
)
from unified_kg_rag.domain.prompts import TreeSummaryPrompt
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.usage import record_cache_hit
from unified_kg_rag.shared.utils.scripts import Spacing, script_named
from unified_kg_rag.shared.utils.sentences import sentence_ends

logger = get_logger(__name__)

# Languages whose length hint counts characters: those of a dense script
# (Japanese and Chinese write no spaces between words, a Korean word is
# several tokens); the others' counts words.
_CHARACTER_LANGUAGES = frozenset(
    language.code
    for language in profiles()
    if script_named(language.script).spacing is Spacing.DENSE
)

# The share of max_summary_tokens the length hint asks for: a model's length
# is approximate, and what is over the budget is cut. Claude Haiku wrote about
# 1.4 times the hinted length (603 summaries over MuSiQue at 0.9: 376 output
# tokens each against a 270-token hint, 70% cut), so 0.7 lands the average
# summary within the budget and leaves the cut to the long tail.
_HINT_FILL = 0.7


@dataclass(frozen=True)
class Summary:
    """One cluster's summary, its token count, and whether it was reused."""

    text: str
    n_tokens: int
    input_key: str
    reused: bool = False


class TreeSummarizer:
    """Summarizes the clusters of one level of the summary tree."""

    def __init__(self, config: Config, *, providers: Providers | None = None) -> None:
        self.config = config
        self.tree_config = config.indexing.summary_tree
        self.language_code = index_language(config)
        # Named in the prompt ("Write the summary in Korean"), not by code.
        self.language = language_name(self.language_code)
        self.fingerprint = tree_summary_fingerprint(config)
        self.providers = Providers.resolve(config, providers)
        self.factory = self.providers.llm_factory
        self.model_id = config.model_id(ModelRole.SUMMARY_TREE)
        self.batch_processor = BatchProcessor.from_config(
            config, is_transient_error=transient_error_predicate(self.factory)
        )
        self.chain = setup_chain(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            model_id=self.model_id,
            prompt_class=TreeSummaryPrompt,
            parser=StrOutputParser(),
            custom_prompts=config.custom_prompts,
        )
        # A local estimate (script-aware: Han, Hangul and Kana about one
        # token a character) rather than a CountTokens round trip per text.
        self.token_counter = self.providers.token_counter(
            self.model_id, api_supported=False
        )

    def count_tokens(self, text: str) -> int:
        return self.token_counter.count_tokens(text) if text else 0

    def length_hint(self, texts: Sequence[str]) -> str:
        """How long a summary of ``texts`` (a cluster's children, in the
        index language) may be, as the prompt says it: the words or
        characters that ``max_summary_tokens`` (times ``_HINT_FILL``) holds
        at the token counter's rate over ``texts``."""
        unit, count = "words", lambda text: len(text.split())
        if LanguageCode(self.language_code) in _CHARACTER_LANGUAGES:
            unit, count = "characters", lambda text: len("".join(text.split()))
        sample = "\n\n".join(texts)
        units = count(sample)
        budget = self.tree_config.max_summary_tokens * _HINT_FILL
        if not units:
            return f"{max(1, int(budget))} {unit}"
        tokens_per_unit = max(self.count_tokens(sample), 1) / units
        return f"{max(1, int(budget / tokens_per_unit))} {unit}"

    def input_key(self, cluster: Cluster) -> str:
        return summary_input_key(self.fingerprint, self.language, cluster.child_texts)

    def summarize(
        self,
        clusters: Sequence[Cluster],
        reusable: Mapping[str, tuple[str, int]] | None = None,
    ) -> list[Summary | None]:
        """Each cluster's summary (``None`` where its call failed every
        attempt or answered nothing), in order. A cluster whose input key is
        in ``reusable`` (``{input_key: (text, n_tokens)}``, the live tree's
        nodes) takes that summary."""
        reusable = reusable or {}
        keys = [self.input_key(cluster) for cluster in clusters]
        results: list[Summary | None] = [None] * len(clusters)
        pending: list[int] = []
        for index, key in enumerate(keys):
            stored = reusable.get(key)
            if stored is not None:
                results[index] = Summary(stored[0], stored[1], key, reused=True)
            else:
                pending.append(index)
        if len(pending) < len(clusters):
            # Model calls the run did not make (shared.usage), under the id
            # the calls are recorded by.
            model = usage_model_id(self.factory, model_id_value(self.model_id))
            for _ in range(len(clusters) - len(pending)):
                record_cache_hit(ModelPurpose.INGESTION.value, model)
        if not pending:
            return results
        inputs = [self._chain_input(clusters[index]) for index in pending]
        answers = self.batch_processor.execute_with_fallback(
            items_to_process=inputs,
            prepare_inputs_func=lambda chunk: chunk,
            sequential_func=self.chain.invoke,
            task_name="Summary tree",
            show_progress=False,
        )
        cut = 0
        for index, answer in zip(pending, answers, strict=True):
            text = answer.strip() if isinstance(answer, str) else ""
            if not text:
                logger.warning(
                    "No summary for a cluster of %d nodes at level %d (the "
                    "call failed or answered nothing): its nodes move up",
                    len(clusters[index].members),
                    clusters[index].level,
                )
                continue
            within = self._within_budget(text)
            cut += within != text
            results[index] = Summary(within, self.count_tokens(within), keys[index])
        if cut:
            logger.info(
                "%d of %d summaries were over max_summary_tokens (%d) and were "
                "cut at a sentence end",
                cut,
                len(pending),
                self.tree_config.max_summary_tokens,
            )
        return results

    def _chain_input(self, cluster: Cluster) -> dict[str, str]:
        passages = "\n\n".join(
            f"[{number}] {' '.join(text.split())}"
            for number, text in enumerate(cluster.child_texts, start=1)
        )
        return {
            "passages": passages,
            "target_language": self.language,
            "length_hint": self.length_hint(cluster.child_texts),
        }

    def _within_budget(self, text: str) -> str:
        """``text``, or its longest prefix of whole sentences within
        ``max_summary_tokens``; cut at a character when not even its first
        sentence fits."""
        budget = self.tree_config.max_summary_tokens
        if self.count_tokens(text) <= budget:
            return text
        # The longest fitting prefix (counts grow with the prefix).
        ends = sentence_ends(text)
        low, high = 0, len(ends)
        while low < high:
            middle = (low + high + 1) // 2
            if self.count_tokens(text[: ends[middle - 1]].rstrip()) <= budget:
                low = middle
            else:
                high = middle - 1
        if low:
            return text[: ends[low - 1]].rstrip()
        return self.token_counter.truncate_to_token_limit(text, budget)[0].rstrip()
