# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Answer-match evaluator: answer containment, SQuAD-style exact match and token F1.

Deterministic and LLM-free by default, so scores are reproducible across runs
and judge models. Both the generated answer and each reference are normalized
SQuAD-style after Unicode NFKC (``text_matching.normalize_answer``: lowercase,
delete punctuation except a number's sign, decimal point and percent sign and
"C++"-style suffixes, split dashed ranges and compounds, drop the English
articles ``a``/``an``/``the`` and, for an answer in German, Spanish,
French, Italian or Portuguese, that language's, collapse whitespace; not
identical to the official SQuAD v1.1 script). The answer's language is the
question's (``domain.retrieval.no_context.answer_language``: its script and
function words, else the index language), recorded as ``answer_language``. With ``evaluation.answer_normalization:
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
  and Hebrew and Arabic proclitics tolerated, substring match for
  single-word CJK and Thai phrases). Long-form RAG
  answers rarely equal a short gold span, so this is the headline
  deterministic answer metric. A yes/no gold (a ``LanguageProfile.polarity_words``
  word: "yes"/"no", "sí", "oui"/"non", "ja"/"nein", "sim"/"não",
  "да"/"нет", Korean "예"/"네"/"아니요", Chinese "是"/"否", Japanese
  "はい"/"いいえ", ...) is instead compared with the polarity of the first
  yes/no word that answers the question (not a heading, not the determiner
  in "there is no clause" or the negation in "não há dúvida", not part of a
  longer word such as "예외"). An item
  whose ``metadata.answer_type`` is ``number`` is instead read in the
  answer's first line (its extracted span, with ``extracted``): a gold
  other than 0 matches when that line writes it anywhere, as digits, a
  number word of the answer's language, a Korean native numeral before a
  counter ("세 곳") or a Han numeral ("三社"); a gold of 0 when it states
  there is none ("does not have any", "no hay", 없습니다, 没有) and writes no
  number the question does not (``text_matching.states_number``). Which
  number is the count, and what a condition or comparative bounds, is left
  to the LLM judges. One whose ``answer_type``
  is ``set`` scores 1.0 when every one of its ``metadata.answer_items``
  appears in the answer, in any order or layout; with
  ``metadata.answer_candidates`` (the names it could pick from) the
  candidates the part that answers names (the extracted span, else the
  first line and the list right after it, ``first_answer_block``) must be
  exactly those items, so listing every candidate is no match and a later
  "Delta is fourth" no pick.
- ``exact_match``: 1.0 if the normalized span equals the normalized reference.
- ``token_f1``: harmonic mean of token precision/recall (multiset overlap).
  Under ``default`` the tokens are script-aware
  (``text_matching.script_tokens``): whitespace words, but a Han, Hangul,
  Kana or Thai (Lao, Myanmar, Khmer) letter is a token of its own (character F1, as the CMRC 2018 and
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
``abstained`` in the report and no span extraction call; on an item marked
``metadata.decline_is_correct`` (a question about something the data does
not hold) it is the right answer and scores 1. Exact match
compares whole normalized strings, so it under-counts an answer that adds
a Korean particle ("서울은" for "서울") or a Chinese or Japanese copula.
Each report records the token-F1 tokenization (``token_f1_tokens``:
``script_aware_v2`` or ``whitespace``), as does the run manifest's
``answer_scoring``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime
from functools import partial
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from unified_kg_rag.adapters.evaluators.base import (
    SKIP_REASON_EMPTY_REFERENCE,
    SKIPPED_METRICS_KEY,
    BaseGraphRAGEvaluator,
    judge_effort,
    judge_model_kwargs,
)
from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.ingestion.cache_keys import index_language
from unified_kg_rag.domain.languages import profiles
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
    model_id_value,
)
from unified_kg_rag.domain.prompts import NO_ANSWER_SPAN, AnswerSpanExtractionPrompt
from unified_kg_rag.domain.retrieval.no_context import answer_language
from unified_kg_rag.shared import EvaluationException, get_logger
from unified_kg_rag.shared.logging import log_failure

from .text_matching import (
    normalize_answer,
    normalize_answer_squad,
    number_value,
    phrase_in_text,
    script_tokens,
    states_number,
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
# The language the answer was normalized in (``answer_language``).
ANSWER_LANGUAGE_KEY = "answer_language"
ABSTAINED_KEY = "abstained"
# Dataset metadata that says how a gold answer is matched (structural
# benchmarks): ``answer_type`` "number" (the gold number stated in the
# answer's first line, ``number_matches``), "set" (every one of ``answer_items`` in the answer, in any
# order or layout; with ``answer_candidates``, the names the answer could
# pick from, the ones it states must be exactly ``answer_items``);
# ``decline_is_correct``, a question about something the
# data does not hold, where a decline is the right answer.
ANSWER_TYPE_KEY = "answer_type"
ANSWER_ITEMS_KEY = "answer_items"
ANSWER_CANDIDATES_KEY = "answer_candidates"
DECLINE_IS_CORRECT_KEY = "decline_is_correct"
ANSWER_TYPES = frozenset({"number", "set"})
# Dataset metadata: the other forms a gold answer may take (one string or a
# list of them), each matched as the answer is.
ANSWER_ALIASES_KEY = "answer_aliases"


def answer_aliases(metadata: Mapping[str, Any]) -> list[str]:
    """A dataset row's answer aliases (``metadata.answer_aliases``, a string
    or a list), as a list of its strings."""
    aliases = metadata.get(ANSWER_ALIASES_KEY) or []
    if isinstance(aliases, str):
        aliases = [aliases]
    return [a for a in aliases if isinstance(a, str)]


__all__ = [
    "AnswerMatchEvaluator",
    "answer_contains",
    "exact_match",
    "first_answer_block",
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


# Yes/no answer words of every language (``LanguageProfile.polarity_words``),
# mapped to their polarity. A short polarity gold ("예", "是") would match
# inside unrelated words as a CJK substring ("예를 들어", "这是"), so it is
# never phrase-matched.
_POLARITY = {
    word: polarity
    for language in profiles()
    for word, polarity in language.polarity_words.items()
}
# A hyphen (ASCII or U+2010, which NFKC makes of the non-breaking U+2011)
# joins the word to the next one ("no-cap"), so it is not a whole word.
_POLARITY_WORD = re.compile(
    r"(?<![\w\u2010-])("
    + "|".join(sorted(map(re.escape, _POLARITY), key=len, reverse=True))
    + r")(?![\w\u2010-])"
)
# The words with a second meaning (``LanguageProfile.
# ambiguous_polarity_words``) answer a question only when punctuation or the
# line end follows them ("No: it may not.", "**네**, 맞습니다.", "Não, ele
# não fez."): before a word, "no" is a determiner ("there is no clause"),
# "네" the numeral four ("네 가지"), "não" and "нет" a negation ("não há
# dúvida", "нет долгов"); "예" also means "e.g." before ":" or ")" ("(예:
# …)"), the punctuation its profile lists. Emphasis and quote marks around
# the word are skipped first; an opening quote or emphasis after it starts
# the next word ("no **cap**", 'no "cap"'), so neither counts as the
# punctuation. Known limit: other punctuation before a word still reads as
# the answer ("no (explicit) cap", "no $500 cap"). Any other polarity word
# answers wherever it stands.
_AMBIGUOUS_ANSWER = {
    word: re.compile(
        r"\s*[*_\"'`]*\s*(?:$|[^\w\s*_\"'`“‘「『" + re.escape(other) + "])"
    )
    for language in profiles()
    for word, other in language.ambiguous_polarity_words.items()
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


def answer_contains(
    prediction: str, reference: str, language: str | None = None
) -> float:
    # A yes/no gold appears in almost any long answer ("no" in "there is no
    # ...", "예" in "예외"), so it is scored by the answer's leading polarity
    # instead, as HotpotQA does for its comparison questions.
    gold = normalize_answer(reference)
    if gold in _POLARITY:
        return float(_leading_polarity(prediction) == _POLARITY[gold])
    return float(phrase_in_text(reference, prediction, language))


def number_matches(
    prediction: str,
    reference: str,
    language: str | None = None,
    question: str | None = None,
) -> float:
    """1.0 when ``prediction`` states the number ``reference`` is
    (``text_matching.states_number``: a gold other than 0 written anywhere
    in it; a gold of 0 by a statement of none with no number but those
    ``question`` writes); 0.0 otherwise, and when the reference is no
    number (then ``answer_contains`` reads it)."""
    gold = number_value(reference, language)
    if gold is None:
        return answer_contains(prediction, reference, language)
    return float(states_number(prediction, gold, language, question))


def items_contained(
    prediction: str,
    items: list[str],
    candidates: list[str] | None = None,
    language: str | None = None,
) -> float:
    """1.0 when every one of ``items`` appears in ``prediction`` as a phrase
    (``text_matching.phrase_in_text``), in any order and layout (a comma
    list, a numbered list, one per line). With ``candidates`` (the names
    the answer could pick from, ``items`` among them) the names it states
    must be ``items`` exactly: listing every candidate, or one more than
    the gold, scores 0. A candidate found only inside a longer one it
    states ("Acme" in "Acme Works") is not one it names."""
    if not items:
        return 0.0
    if not candidates:
        return float(all(phrase_in_text(i, prediction, language) for i in items))
    named = {
        c for c in {*candidates, *items} if phrase_in_text(c, prediction, language)
    }
    named = {
        c
        for c in named
        if not any(c != o and phrase_in_text(c, o, language) for o in named)
    }
    return float(named == set(items))


# A list item: a bullet ("-", "*", "+", "•", "・") or an enumerator ("1.",
# "2)", "1、", "一、", "①").
_LIST_ITEM = re.compile(
    r"[ \t]*(?:[-*+•][ \t]+|・[ \t]*|\d{1,3}[.)][ \t]+|\d{1,3}[、．][ \t]*"
    r"|[一二三四五六七八九十]{1,3}、[ \t]*|[\u2460-\u2473][ \t]*)(?=\S)"
)


def first_answer_block(answer: str) -> str:
    """The part of ``answer`` that answers a set question: its first line
    that is neither empty nor a Markdown heading, then the list items that
    follow it (blank lines between them skipped) up to the first line that
    is no list item, each without its bullet or enumerator ("1、Acme" would
    read as one word); the whole answer, stripped, when every line is empty
    or a heading."""
    lines = answer.splitlines()
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if line.strip() and not _MARKDOWN_HEADING.match(line)
        ),
        None,
    )
    if start is None:
        return answer.strip()
    first = _LIST_ITEM.match(lines[start])
    block = [lines[start][first.end() if first else 0 :].strip()]
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        item = _LIST_ITEM.match(line)
        if item is None:
            break
        block.append(line[item.end() :].strip())
    return "\n".join(block)


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
    a letter of a script written without spaces per token, Korean particles of their
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


# The answer_scoring keys of an ``extracted`` span: the model that extracted
# it and its effort (runs saved before they were recorded lack them).
SPAN_MODEL_KEY = "span_model"
SPAN_EFFORT_KEY = "span_effort"


def answer_scoring(config: Config) -> dict[str, Any]:
    """How exact match and token F1 read an answer under ``config`` (the run
    manifest's ``answer_scoring``): the span mode, the normalization and
    its tokenization, and with an ``extracted`` span the model
    (``models.roles.answer_span``) and effort that extracted it. Two runs
    scored differently are not comparable on them."""
    evaluation = config.evaluation
    scoring: dict[str, Any] = {
        "answer_span": evaluation.answer_span,
        "answer_normalization": evaluation.answer_normalization,
        TOKENS_KEY: token_f1_tokens(evaluation.answer_normalization),
    }
    if evaluation.answer_span == "extracted":
        model = config.model_id(ModelRole.ANSWER_SPAN)
        scoring[SPAN_MODEL_KEY] = model_id_value(model)
        scoring[SPAN_EFFORT_KEY] = judge_effort(config, model)
    return scoring


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
        candidates = [ground_truth, *answer_aliases(metadata)]
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
        answer = self.answer_words(result)
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
        answer = self.answer_words(result)
        try:
            span = await self._extracted_span(result.question, answer)
        except Exception as e:  # noqa: BLE001 - scored on the first line instead
            # The error may quote the model's span: a digest at WARNING.
            log_failure(
                logger,
                logging.WARNING,
                "Answer-span extraction failed for '%s' (scoring its first "
                "line): %s",
                query.query_id,
                error=e,
            )
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

    def _contains(
        self,
        answer: str,
        span: str,
        references: list[str],
        result: EvaluationResult,
        span_metadata: dict[str, Any],
        language: str,
    ) -> float:
        """``answer_contains`` as the item's ``answer_type`` reads it: a
        number in the extracted span, else the answer's first line
        (``first_answer_line``; ``number_matches``), a set by every one of
        its ``answer_items`` (with ``answer_candidates``, in the extracted
        span or else the answer's first block, ``first_answer_block``),
        else any reference as a phrase."""
        kind = result.metadata.get(ANSWER_TYPE_KEY)
        if kind == "number":
            read = self._extracted_or(span, span_metadata, first_answer_line(answer))
            return max(
                number_matches(read, ref, language, result.question)
                for ref in references
            )
        if kind == "set":
            items = result.metadata.get(ANSWER_ITEMS_KEY)
            candidates = result.metadata.get(ANSWER_CANDIDATES_KEY)
            if isinstance(items, list) and items:
                if not isinstance(candidates, list):
                    return items_contained(
                        answer, [str(i) for i in items], language=language
                    )
                # The candidates it picks are those of the part that
                # answers: a later "Delta is fourth" is no pick.
                return items_contained(
                    self._extracted_or(span, span_metadata, first_answer_block(answer)),
                    [str(i) for i in items],
                    [str(c) for c in candidates],
                    language,
                )
        return max(answer_contains(answer, ref, language) for ref in references)

    def _extracted_or(
        self, span: str, span_metadata: dict[str, Any], otherwise: str
    ) -> str:
        """The extracted span, or ``otherwise`` without one (another span
        mode, no span, or one that fell back to the first line)."""
        if (
            self.answer_span == "extracted"
            and span
            and SPAN_FALLBACK_KEY not in span_metadata
        ):
            return span
        return otherwise

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
            # entity while saying what is missing; on a question about
            # something the data does not hold it is the right one.
            correct = result.metadata.get(DECLINE_IS_CORRECT_KEY) is True
            return EvaluationReport(
                query_id=query.query_id,
                evaluator_type=self.evaluator_type,
                metrics=[
                    EvaluationMetric(metric_type=metric, value=float(correct))
                    for metric in self.metric_types()
                ],
                evaluation_time=datetime.now(),
                metadata={
                    **self._extract_search_metadata(result),
                    "num_references": len(references),
                    ANSWER_SPAN_KEY: self.answer_span,
                    ABSTAINED_KEY: True,
                    **({DECLINE_IS_CORRECT_KEY: True} if correct else {}),
                },
            )
        answer = self.answer_words(result)
        # Normalized in the answer's language: the question's.
        language = answer_language(result.question, index_language(self.config)).value
        contains = self._contains(
            answer, span, references, result, span_metadata, language
        )
        normalization = self.config.evaluation.answer_normalization
        normalize: Callable[[str], str] = (
            partial(normalize_answer, language=language)
            if normalization == "default"
            else normalize_answer_squad
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
                ANSWER_LANGUAGE_KEY: language,
                # The whole answer is in the result already.
                **({} if self.answer_span == "full" else {SCORED_SPAN_KEY: span}),
                **span_metadata,
            },
        )
