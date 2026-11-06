# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The summaries of RAPTOR's summary tree (docs/design.md#65-raptor-summary-tree-raptor-tree-opt-in).

One call per cluster (``TreeSummaryPrompt``, ``models.roles.summary_tree``),
fanned out through ``BatchProcessor`` with the run's LLM response cache, as
the other ingestion model calls are. A cluster whose input key (its
children's texts, the language and the summary settings) is a live node's
takes that node's summary and makes no call. A summary over
``max_summary_tokens`` is cut to it by the token counter.
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

logger = get_logger(__name__)

# Languages written without spaces between words: their length hint is in
# characters (about one a token), the others' in words (about 0.75 a token).
_CHARACTER_LANGUAGES = frozenset({LanguageCode.JA, LanguageCode.ZH, LanguageCode.KO})


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

    def length_hint(self) -> str:
        """How long a summary may be, as the prompt says it."""
        tokens = self.tree_config.max_summary_tokens
        if LanguageCode(self.language_code) in _CHARACTER_LANGUAGES:
            return f"about {tokens} characters"
        return f"about {max(1, int(tokens * 0.75))} words"

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
            text = self._within_budget(text)
            results[index] = Summary(text, self.count_tokens(text), keys[index])
        return results

    def _chain_input(self, cluster: Cluster) -> dict[str, str]:
        passages = "\n\n".join(
            f"[{number}] {' '.join(text.split())}"
            for number, text in enumerate(cluster.child_texts, start=1)
        )
        return {
            "passages": passages,
            "target_language": self.language,
            "length_hint": self.length_hint(),
        }

    def _within_budget(self, text: str) -> str:
        """``text`` cut (at a character) to ``max_summary_tokens``."""
        budget = self.tree_config.max_summary_tokens
        for _ in range(8):
            tokens = self.count_tokens(text)
            if tokens <= budget:
                return text
            text = text[: max(1, int(len(text) * budget / tokens) - 1)].rstrip()
        return text
