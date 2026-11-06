# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Answer-match evaluator: answer containment, SQuAD-style exact match and token F1.

Deterministic and LLM-free by default, so scores are reproducible across runs
and judge models. Both the generated answer and each reference are normalized
SQuAD-style after Unicode NFKC (``text_matching.normalize_answer``: lowercase,
delete punctuation except a number's sign, decimal point and percent sign and
"C++"-style suffixes, split dashed ranges and compounds, drop the English
articles ``a``/``an``/``the``, collapse whitespace; not identical to the
official SQuAD v1.1 script). With ``evaluation.answer_normalization:
official_squad`` exact match and token F1 use the official script's
normalization instead (``text_matching.normalize_answer_squad``, MuSiQue's
too); ``hotpotqa`` adds the HotpotQA script's yes/no rule to it.

Exact match and token F1 read the part of the answer ``evaluation.answer_span``
names. The benchmarks they come from score a short predicted span; a RAG
answer is a sentence plus its support, and scored whole (``full``) the two
metrics measure its length as much as its correctness. ``first_line`` (the
default) scores the answer's first line that is neither empty nor a Markdown
heading, the one-sentence answer the answer prompt asks for: closer to a span,
but EM still under-counts. ``extracted`` asks the ``answer_span`` model role
for the span (``AnswerSpanExtractionPrompt``, one call per distinct question
and answer, cached for the evaluator's life); a failed call, an empty reply
or a span that is not in the answer falls back to ``first_line`` and is
recorded in the report (``answer_span_fallback``). Only an extracted span with
an official normalization gives numbers close to the papers'. Each report
records the span mode (``answer_span``) and the text scored
(``scored_span``). Then:

- ``answer_contains``: 1.0 if the gold answer appears in the generated answer
  as a whole-word phrase (``text_matching.phrase_in_text``: Korean particles
  tolerated, substring match for single-word CJK phrases). Long-form RAG
  answers rarely equal a short gold span, so this is the headline
  deterministic answer metric. A yes/no gold ("yes"/"no", Korean
  "예"/"네"/"아니요"/"아니오"/"아닙니다", Chinese "是"/"对"/"否"/"不是"/"没有", Japanese
  "はい"/"いいえ") is instead compared with the polarity of the first yes/no
  word that answers the question (not a heading, not the determiner in
  "there is no clause", not part of a longer word such as "예외").
- ``exact_match``: 1.0 if the normalized span equals the normalized reference.
- ``token_f1``: harmonic mean of token precision/recall (multiset overlap).
  Under ``default`` the tokens are script-aware
  (``text_matching.script_tokens``): whitespace words, but a Han, Hangul or
  Kana letter is a token of its own (character F1, as the CMRC 2018 and
  KorQuAD 1.0 scripts score Chinese and Korean), after a run of Korean
  particles (their closed class) attached to a reference word is stripped. Under ``official_squad`` and ``hotpotqa`` they are
  whitespace words, as in those scripts, so for Chinese and Japanese token
  F1 degrades to exact match there. Under ``hotpotqa`` a span or reference
  that is "yes", "no" or "noanswer" scores 0 unless the two are equal.

References are the dataset ``answer`` plus optional ``metadata.answer_aliases``
(a string or list of strings); each metric takes the max over references.
A query with no reference is skipped. An abstained answer
(``EvaluationResult.abstained``: the no-context reply, or a declined answer
under ``search.abstention``) is a wrong answer: every metric scores 0, with
``abstained`` in the report and no span extraction call. Exact match
compares whole normalized strings, so it under-counts an answer that adds
a Korean particle ("서울은" for "서울") or a Chinese or Japanese copula.
Each report records the token-F1 tokenization (``token_f1_tokens``:
``script_aware_v2`` or ``whitespace``), as does the run manifest's
``answer_scoring``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from unified_kg_rag.adapters.evaluators.base import (
    SKIP_REASON_EMPTY_REFERENCE,
    SKIPPED_METRICS_KEY,
    BaseGraphRAGEvaluator,
    error_digest,
    judge_model_kwargs,
)
from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import (
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
    ModelPurpose,
    ModelRole,
)
from unified_kg_rag.domain.prompts import NO_ANSWER_SPAN, AnswerSpanExtractionPrompt
from unified_kg_rag.shared import EvaluationException, get_logger

from .text_matching import (
    normalize_answer,
    normalize_answer_squad,
    phrase_in_text,
    script_tokens,
)

logger = get_logger(__name__)

# Report-metadata keys: the span mode, the text exact match and token F1
# scored, why an extracted span fell back to the first line, and that the
# answer was a decline (scored 0).
ANSWER_SPAN_KEY = "answer_span"
TOKENS_KEY = "token_f1_tokens"
# The token-F1 tokenization of the default normalization: by script, a
# closed class of Korean particles dropped (``text_matching.script_tokens``).
# Renamed from ``script_aware``, which dropped any one to three Hangul
# syllables after a reference word, so a run scored that way is refused
# against one of now (``saved_results.check_answer_scoring``).
SCRIPT_AWARE_TOKENS = "script_aware_v2"
SCORED_SPAN_KEY = "scored_span"
SPAN_FALLBACK_KEY = "answer_span_fallback"
ABSTAINED_KEY = "abstained"

__all__ = [
    "AnswerMatchEvaluator",
    "answer_contains",
    "exact_match",
    "first_answer_line",
    "normalize_answer",
    "token_f1",
]


def exact_match(
    prediction: str,
    reference: str,
    normalize: Callable[[str], str] = normalize_answer,
) -> float:
    return float(normalize(prediction) == normalize(reference))


# Yes/no answer words, by language, mapped to their polarity. A short
# polarity gold ("예", "是") would match inside unrelated words as a CJK
# substring ("예를 들어", "这是"), so it is never phrase-matched.
_POLARITY = {
    "yes": "yes",
    "no": "no",
    "nope": "no",
    "예": "yes",
    "네": "yes",
    "아니요": "no",
    "아니오": "no",
    "아닙니다": "no",
    "是": "yes",
    "是的": "yes",
    "对": "yes",
    "对的": "yes",
    "否": "no",
    "不是": "no",
    "不是的": "no",
    "没有": "no",
    "はい": "yes",
    "いいえ": "no",
}
# A hyphen (ASCII or U+2010, which NFKC makes of the non-breaking U+2011)
# joins the word to the next one ("no-cap"), so it is not a whole word.
_POLARITY_WORD = re.compile(
    r"(?<![\w\u2010-])("
    + "|".join(sorted(map(re.escape, _POLARITY), key=len, reverse=True))
    + r")(?![\w\u2010-])"
)
# Three words have a second meaning, so they answer a question only when
# punctuation or the line end follows them ("No: it may not.", "**네**, 맞습니다."):
# before a word, "no" is a determiner ("there is no clause") and "네" the
# numeral four ("네 가지"); "예" also means "e.g." before ":" or ")" ("(예: …)").
# Emphasis and quote marks around the word are skipped first; an opening
# quote or emphasis after it starts the next word ("no **cap**", 'no "cap"'),
# so neither counts as the punctuation. Known limit: other punctuation
# before a word still reads as the answer ("no (explicit) cap", "no $500 cap").
# Any other polarity word answers wherever it stands.
_AMBIGUOUS_ANSWER = {
    "no": re.compile(r"\s*[*_\"'`]*\s*(?:$|[^\w\s*_\"'`“‘「『])"),
    "네": re.compile(r"\s*[*_\"'`]*\s*(?:$|[^\w\s*_\"'`“‘「『])"),
    "예": re.compile(r"\s*[*_\"'`]*\s*(?:$|[^\w\s*_\"'`“‘「『:)])"),
}
# A word next to "/" names the options, not the answer ("Yes/No: No").
_OPTION_SLASH = re.compile(r"\s*/")
# "No." before a number abbreviates "number" ("Clause No. 5").
_NUMBER_ABBREVIATION = re.compile(r"\.\s*\d")
_MARKDOWN_HEADING = re.compile(r"\s*#{1,6}(?:\s|$)")


def _leading_polarity(prediction: str) -> str | None:
    """The polarity ("yes"/"no") of the first yes/no word that answers.

    Markdown heading lines are skipped, a polarity word next to "/" lists
    the options ("Yes/No"), "no", "네" or "예" not followed by punctuation or
    the line end ("there is no evidence", "네 가지") or "예" before ":" or ")"
    ("(예: …)") is not an answer, and "No." before a number is the
    abbreviation. Words are matched whole, so "예" in
    "예외" or "是" in "这是" is not one. The search runs over whole lines, so
    an abbreviation such as "U.S." cannot cut the answer short.
    """
    text = unicodedata.normalize("NFKC", prediction).lower()
    for line in text.splitlines():
        if _MARKDOWN_HEADING.match(line):
            continue
        for match in _POLARITY_WORD.finditer(line):
            word = match.group(1)
            rest = line[match.end() :]
            if _OPTION_SLASH.match(rest) or line[: match.start()].rstrip()[-1:] == "/":
                continue
            answering = _AMBIGUOUS_ANSWER.get(word)
            if answering is None:
                return _POLARITY[word]
            if answering.match(rest) and not _NUMBER_ABBREVIATION.match(rest):
                return _POLARITY[word]
    return None


def answer_contains(prediction: str, reference: str) -> float:
    # A yes/no gold appears in almost any long answer ("no" in "there is no
    # ...", "예" in "예외"), so it is scored by the answer's leading polarity
    # instead, as HotpotQA does for its comparison questions.
    gold = normalize_answer(reference)
    if gold in _POLARITY:
        return float(_leading_polarity(prediction) == _POLARITY[gold])
    return float(phrase_in_text(reference, prediction))


def first_answer_line(answer: str) -> str:
    """The answer's first line that is neither empty nor a Markdown heading
    (the whole answer, stripped, when every line is one of those)."""
    for line in answer.splitlines():
        if line.strip() and not _MARKDOWN_HEADING.match(line):
            return line.strip()
    return answer.strip()


# The HotpotQA script scores these only on equality: partial token credit
# for "no" in "no, it does not" would reward a wrong polarity half the time.
_HOTPOTQA_EXACT_ONLY = frozenset({"yes", "no", "noanswer"})


def token_f1(
    prediction: str,
    reference: str,
    normalize: Callable[[str], str] = normalize_answer,
    *,
    yes_no_exact: bool = False,
    script_aware: bool | None = None,
) -> float:
    """Token F1; with ``yes_no_exact`` (HotpotQA's rule) a prediction or
    reference that normalizes to "yes", "no" or "noanswer" scores 0 unless
    the two normalize equal. ``script_aware`` (default: with the default
    ``normalize_answer``) tokenizes by script (``text_matching.script_tokens``:
    a Han, Hangul or Kana letter per token, Korean particles of their
    closed class stripped);
    otherwise tokens are whitespace words, as in the official scripts."""
    if script_aware is None:
        script_aware = normalize is normalize_answer
    normalized_prediction = normalize(prediction)
    normalized_reference = normalize(reference)
    if (
        yes_no_exact
        and normalized_prediction != normalized_reference
        and {normalized_prediction, normalized_reference} & _HOTPOTQA_EXACT_ONLY
    ):
        return 0.0
    if script_aware:
        pred_tokens, ref_tokens = script_tokens(
            normalized_prediction, normalized_reference
        )
    else:
        pred_tokens = normalized_prediction.split()
        ref_tokens = normalized_reference.split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1_tokens(normalization: str) -> str:
    """The token-F1 tokenization of an ``evaluation.answer_normalization``:
    ``script_aware_v2`` (``SCRIPT_AWARE_TOKENS``) under ``default``,
    ``whitespace`` under the official scripts' normalizations."""
    return SCRIPT_AWARE_TOKENS if normalization == "default" else "whitespace"


def answer_scoring(evaluation: Any) -> dict[str, Any]:
    """How exact match and token F1 read an answer under the
    ``EvaluationConfig`` ``evaluation`` (the run manifest's
    ``answer_scoring``): two runs scored differently are not comparable on
    them."""
    return {
        "answer_span": evaluation.answer_span,
        "answer_normalization": evaluation.answer_normalization,
        TOKENS_KEY: token_f1_tokens(evaluation.answer_normalization),
    }


class AnswerMatchEvaluator(BaseGraphRAGEvaluator):
    """Scores containment, exact match and token F1 against reference answers."""

    def __init__(self, config: Config, **kwargs: Any):
        super().__init__(config, EvaluatorType.ANSWER_MATCH, **kwargs)

    def _initialize_evaluator(
        self, providers: Providers | None = None, **kwargs: Any
    ) -> None:
        self.answer_span = self.config.evaluation.answer_span
        # Extracted spans by (question, answer): repeats and strategies that
        # give the same answer reuse it.
        self._spans: dict[str, str] = {}
        self._span_chain: Runnable[Any, Any] | None = None
        if self.answer_span == "extracted":
            providers = Providers.resolve(self.config, providers)
            self._span_chain = setup_chain(
                providers.llm_factory,
                self.config.model_id(ModelRole.ANSWER_SPAN),
                AnswerSpanExtractionPrompt,
                StrOutputParser(),
                custom_prompts=self.config.custom_prompts,
                model_purpose=ModelPurpose.EVALUATION,
                output_ceiling=self.config.evaluation.answer_span_output_ceiling,
                **judge_model_kwargs(self.config),
            )

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
        if self.answer_span == "extracted":
            raise EvaluationException(
                "answer_span 'extracted' calls a model: use aevaluate_single"
            )
        answer = result.generated_answer or ""
        span = first_answer_line(answer) if self.answer_span == "first_line" else answer
        return self._score(query, result, ground_truth, span, {})

    async def aevaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        if self.answer_span != "extracted":
            return self.evaluate_single(query, result, ground_truth, **kwargs)
        if result.abstained or not self._references(ground_truth, result.metadata):
            # Nothing to extract (a decline scores 0): no model call.
            return self._score(query, result, ground_truth, "", {})
        answer = result.generated_answer or ""
        try:
            span = await self._extracted_span(result.question, answer)
        except Exception as e:  # noqa: BLE001 - scored on the first line instead
            # The error may quote the model's span: a digest at WARNING.
            logger.warning(
                "Answer-span extraction failed for '%s' (scoring its first "
                "line): %s",
                query.query_id,
                error_digest(e),
            )
            logger.debug("Answer-span extraction error for '%s': %s", query.query_id, e)
            return self._score(
                query,
                result,
                ground_truth,
                first_answer_line(answer),
                {SPAN_FALLBACK_KEY: str(e) or type(e).__name__},
            )
        return self._score(query, result, ground_truth, span, {})

    async def _extracted_span(self, question: str, answer: str) -> str:
        """The model's short span of ``answer`` ("" when it gives none).

        Raises:
            EvaluationException: The reply is empty, or a span the answer
                does not contain (the model answered the question itself).
        """
        key = hashlib.sha256(f"{question}\0{answer}".encode()).hexdigest()
        if key in self._spans:
            return self._spans[key]
        if self._span_chain is None:
            raise EvaluationException("no span extractor is configured")
        reply = await self._span_chain.ainvoke({"question": question, "answer": answer})
        lines = [line.strip() for line in str(reply).splitlines() if line.strip()]
        if not lines:
            raise EvaluationException("the span extractor returned nothing")
        span = lines[0].strip("\"'`“”‘’").strip()
        if span.upper() == NO_ANSWER_SPAN:
            span = ""
        elif normalize_answer(span) not in _POLARITY and not phrase_in_text(
            span, answer
        ):
            raise EvaluationException(
                f"the extracted span {span!r} is not in the answer"
            )
        self._spans[key] = span
        return span

    def _score(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        span: str,
        span_metadata: dict[str, Any],
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
        if result.abstained:
            # A decline is not an answer, even one that names the gold
            # entity while saying what is missing.
            return EvaluationReport(
                query_id=query.query_id,
                evaluator_type=self.evaluator_type,
                metrics=[
                    EvaluationMetric(metric_type=metric, value=0.0)
                    for metric in self.metric_types()
                ],
                evaluation_time=datetime.now(),
                metadata={
                    **self._extract_search_metadata(result),
                    "num_references": len(references),
                    ANSWER_SPAN_KEY: self.answer_span,
                    ABSTAINED_KEY: True,
                },
            )
        answer = result.generated_answer or ""
        contains = max(answer_contains(answer, ref) for ref in references)
        normalization = self.config.evaluation.answer_normalization
        normalize = (
            normalize_answer if normalization == "default" else normalize_answer_squad
        )
        yes_no_exact = normalization == "hotpotqa"
        script_aware = token_f1_tokens(normalization) == SCRIPT_AWARE_TOKENS
        em = max(exact_match(span, ref, normalize) for ref in references)
        f1 = max(
            token_f1(
                span,
                ref,
                normalize,
                yes_no_exact=yes_no_exact,
                script_aware=script_aware,
            )
            for ref in references
        )
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
                ANSWER_SPAN_KEY: self.answer_span,
                TOKENS_KEY: token_f1_tokens(normalization),
                # The whole answer is in the result already.
                **({} if self.answer_span == "full" else {SCORED_SPAN_KEY: span}),
                **span_metadata,
            },
        )
