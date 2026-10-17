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
