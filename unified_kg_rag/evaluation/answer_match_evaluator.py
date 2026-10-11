# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Answer-match evaluator: answer containment, SQuAD-style exact match and token F1.

Deterministic and LLM-free, so scores are reproducible across runs and judge
models. Both the generated answer and each reference are normalized SQuAD-style after
Unicode NFKC (``text_matching.normalize_answer``: lowercase, delete punctuation
except a number's sign, decimal point and percent sign and "C++"-style
suffixes, split dashed ranges and compounds, drop the English articles
``a``/``an``/``the``, collapse whitespace; not identical to the official SQuAD
v1.1 script, so scores are not leaderboard-comparable), then:

- ``answer_contains``: 1.0 if the gold answer appears in the generated answer
  as a whole-word phrase (``text_matching.phrase_in_text``: Korean particles
  tolerated, substring match for single-word CJK phrases). Long-form RAG
  answers rarely equal a short gold span, so this is the headline
  deterministic answer metric. A "yes"/"no" gold is instead compared with the
  first "yes"/"no" that answers the question (not a heading, not the
  determiner in "there is no clause").
- ``exact_match``: 1.0 if the normalized strings are equal.
- ``token_f1``: harmonic mean of token precision/recall over whitespace tokens
  (multiset overlap).

References are the dataset ``answer`` plus optional ``metadata.answer_aliases``
(a string or list of strings); each metric takes the max over references.
A query with no reference is skipped. Tokens are whitespace-delimited, so for
scripts written without spaces (Chinese, Japanese) token F1 degrades to exact
match. Korean is space-delimited but attaches particles to words ("서울은"), so
a correct answer rarely matches a bare gold token exactly; both metrics
under-count for Korean.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Any

from unified_kg_rag.adapters.evaluators.base import (
    SKIP_REASON_EMPTY_REFERENCE,
    SKIPPED_METRICS_KEY,
    BaseGraphRAGEvaluator,
)
from unified_kg_rag.domain.models import (
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
)

from .text_matching import normalize_answer, phrase_in_text

__all__ = [
    "AnswerMatchEvaluator",
    "answer_contains",
    "exact_match",
    "normalize_answer",
    "token_f1",
]


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


_POLARITY = frozenset({"yes", "no"})
_POLARITY_WORD = re.compile(r"(?<![\w-])(yes|no)(?![\w-])")
# "no" answers a question only when punctuation or the line end follows it
# ("No, it may not.", "..., no."); before a word it is a determiner ("there
# is no clause").
_ANSWERING_NO = re.compile(r"\s*(?:$|[^\w\s])")
# "No." before a number abbreviates "number" ("Clause No. 5").
_NUMBER_ABBREVIATION = re.compile(r"\.\s*\d")
_MARKDOWN_HEADING = re.compile(r"\s*#{1,6}(?:\s|$)")


def _leading_polarity(prediction: str) -> str | None:
    """The first "yes"/"no" in the prediction that answers the question.

    Markdown heading lines are skipped, a "no" followed by a word ("there is
    no evidence") is a determiner, not an answer, and "No." before a number
    is the abbreviation. The search runs over whole lines, so an
    abbreviation such as "U.S." cannot cut the answer short.
    """
    text = unicodedata.normalize("NFKC", prediction).lower()
    for line in text.splitlines():
        if _MARKDOWN_HEADING.match(line):
            continue
        for match in _POLARITY_WORD.finditer(line):
            word = match.group(1)
            if word == "yes":
                return word
            rest = line[match.end() :]
            if _ANSWERING_NO.match(rest) and not _NUMBER_ABBREVIATION.match(rest):
                return word
    return None


def answer_contains(prediction: str, reference: str) -> float:
    # A yes/no gold appears in almost any long answer ("no" in "there is no
    # ..."), so it is scored by the answer's leading polarity instead, as
    # HotpotQA does for its comparison questions.
    gold = normalize_answer(reference)
    if gold in _POLARITY:
        return float(_leading_polarity(prediction) == gold)
    return float(phrase_in_text(reference, prediction))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


class AnswerMatchEvaluator(BaseGraphRAGEvaluator):
    """Scores containment, exact match and token F1 against reference answers."""

    def __init__(self, config: Config, **kwargs: Any):
        super().__init__(config, EvaluatorType.ANSWER_MATCH, **kwargs)

    def _initialize_evaluator(self, **kwargs: Any) -> None:
        # Pure, deterministic evaluator — no model to initialize.
        pass

    def metric_types(self) -> list[EvaluationMetricType]:
        return [
            EvaluationMetricType.ANSWER_CONTAINS,
            EvaluationMetricType.EXACT_MATCH,
            EvaluationMetricType.TOKEN_F1,
        ]

    @staticmethod
    def _references(ground_truth: str, metadata: dict[str, Any]) -> list[str]:
        aliases = metadata.get("answer_aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        candidates = [ground_truth, *(a for a in aliases if isinstance(a, str))]
        return [c for c in candidates if c and normalize_answer(c)]

    def evaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        references = self._references(ground_truth, result.metadata)
        if not references:
            return EvaluationReport(
                query_id=query.query_id,
                evaluator_type=self.evaluator_type,
                metrics=[],
                metadata={
                    SKIPPED_METRICS_KEY: {
                        m.value: SKIP_REASON_EMPTY_REFERENCE
                        for m in self.metric_types()
                    }
                },
            )
        answer = result.generated_answer or ""
        contains = max(answer_contains(answer, ref) for ref in references)
        em = max(exact_match(answer, ref) for ref in references)
        f1 = max(token_f1(answer, ref) for ref in references)
        return EvaluationReport(
            query_id=query.query_id,
            evaluator_type=self.evaluator_type,
            metrics=[
                EvaluationMetric(
                    metric_type=EvaluationMetricType.ANSWER_CONTAINS, value=contains
                ),
                EvaluationMetric(
                    metric_type=EvaluationMetricType.EXACT_MATCH, value=em
                ),
                EvaluationMetric(metric_type=EvaluationMetricType.TOKEN_F1, value=f1),
            ],
            evaluation_time=datetime.now(),
            metadata={
                **self._extract_search_metadata(result),
                "num_references": len(references),
            },
        )

    async def aevaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        # Deterministic and CPU-only; reuse the sync path.
        return self.evaluate_single(query, result, ground_truth, **kwargs)
