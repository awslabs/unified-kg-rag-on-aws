# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompts of the evaluation judges that the package defines itself."""

from .base import BasePrompt


class PairwiseComparisonPrompt(BasePrompt):
    """Head-to-head answer comparison (the GraphRAG / LightRAG LLM judge).

    The criteria are LightRAG's comprehensiveness, diversity and empowerment
    plus GraphRAG's directness (the control: a longer answer should not win
    it). The judge sees the two answers as "Answer 1" and "Answer 2"; the
    caller runs both orders and keeps only a verdict that survives the swap.
    """

    prompt_key = "pairwise_comparison"
    input_variables = ["query", "answer1", "answer2"]
    output_variables = ["comprehensiveness", "diversity", "empowerment", "directness"]

    system_prompt_template = """You are an expert judge comparing two answers to the same
question on four criteria. Judge only the answers; do not reward length for its own sake.

- comprehensiveness: how much detail the answer provides to cover all aspects of the question.
- diversity: how varied and rich the answer is in offering different perspectives and insights.
- empowerment: how well the answer helps the reader understand the topic and make informed
  judgements.
- directness: how specifically and clearly the answer addresses the question.

For each criterion pick the better answer ("1" or "2"), or "tie" when neither is better.

Output MUST be a single valid JSON object and nothing else, in this form:
{{"comprehensiveness": {{"winner": "1", "reason": "..."}},
 "diversity": {{"winner": "2", "reason": "..."}},
 "empowerment": {{"winner": "tie", "reason": "..."}},
 "directness": {{"winner": "1", "reason": "..."}}}}"""

    human_prompt_template = """Question: {query}

Answer 1:
{answer1}

Answer 2:
{answer2}

Verdict (JSON only):"""


class PairwiseComparisonOverallPrompt(PairwiseComparisonPrompt):
    """The pairwise judge with LightRAG's "overall winner" as a fifth verdict
    (``evaluation.pairwise_overall``); same override key as the four-criteria
    prompt."""

    output_variables = [
        "comprehensiveness",
        "diversity",
        "empowerment",
        "directness",
        "overall",
    ]

    system_prompt_template = """You are an expert judge comparing two answers to the same
question on four criteria, then overall. Judge only the answers; do not reward length for its own sake.

- comprehensiveness: how much detail the answer provides to cover all aspects of the question.
- diversity: how varied and rich the answer is in offering different perspectives and insights.
- empowerment: how well the answer helps the reader understand the topic and make informed
  judgements.
- directness: how specifically and clearly the answer addresses the question.
- overall: the better answer on the four criteria taken together.

For each criterion pick the better answer ("1" or "2"), or "tie" when neither is better.

Output MUST be a single valid JSON object and nothing else, in this form:
{{"comprehensiveness": {{"winner": "1", "reason": "..."}},
 "diversity": {{"winner": "2", "reason": "..."}},
 "empowerment": {{"winner": "tie", "reason": "..."}},
 "directness": {{"winner": "1", "reason": "..."}},
 "overall": {{"winner": "1", "reason": "..."}}}}"""


# What the span extractor answers when the answer gives no answer.
NO_ANSWER_SPAN = "NO_ANSWER"


class AnswerSpanExtractionPrompt(BasePrompt):
    """The short answer span inside a generated answer, for exact match and
    token F1 (``evaluation.answer_span: extracted``).

    Reading-comprehension benchmarks (SQuAD, HotpotQA, MuSiQue) score a short
    predicted span against the gold answer; a RAG answer is a sentence and
    its support. The extractor copies the span the answer commits to, so the
    score reads what the answer says, not how long it is. It never sees the
    gold answer.
    """

    prompt_key = "answer_span_extraction"
    input_variables = ["question", "answer"]

    system_prompt_template = f"""You extract the short answer from a written answer to a question.

Copy, verbatim from the written answer, the shortest span that answers the question: the entity,
name, number, date, place or phrase it gives as the answer, or "yes" or "no" for a yes/no question.
Do not answer the question yourself, correct the answer, or add words that are not in it.
If the written answer does not give an answer (it says the information is missing), output
{NO_ANSWER_SPAN}.

Output only the span (or {NO_ANSWER_SPAN}) on one line, without quotes or explanation."""

    human_prompt_template = """Question: {question}

Written answer:
{answer}

Short answer:"""
