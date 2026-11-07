# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property: the summary tree's length hint agrees with the token counter
that cuts the summary (``TreeSummarizer.length_hint``).

For any script (Latin, Cyrillic, Hangul, Kana, Han), word shape, budget and
counter scale, a summary written like the cluster's texts in as many words
(or, for Japanese, Chinese and Korean, characters) as the hint names counts
at most ``max_summary_tokens``, and not far below it.
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.adapters.aws.token_counter import BedrockTokenCounter
from unified_kg_rag.adapters.ingestion.tree_summarizer import TreeSummarizer
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import LanguageCode

pytestmark = pytest.mark.property

# (index language, letters, word separator)
_SCRIPTS = [
    (LanguageCode.EN, "abcdefghijklmnopqrstuvwxyz", " "),
    (LanguageCode.RU, "абвгдежзийклмнопрстуф", " "),
    (LanguageCode.KO, "가나다라마바사아자차카타파하창고은행", " "),
    (LanguageCode.JA, "あいうえおかきくけこアイウエオ倉庫部品", ""),
    (LanguageCode.ZH, "东京仓库部品银行地区工业每月保管", ""),
]


@settings(max_examples=60, deadline=None)
@given(
    script=st.sampled_from(_SCRIPTS),
    word_length=st.integers(min_value=1, max_value=12),
    words=st.integers(min_value=20, max_value=400),
    budget=st.integers(min_value=32, max_value=4000),
    multiplier=st.sampled_from([1.0, 1.3, 1.5]),
    data=st.data(),
)
def test_a_summary_of_the_hinted_length_fits_the_budget(
    script: tuple[LanguageCode, str, str],
    word_length: int,
    words: int,
    budget: int,
    multiplier: float,
    data: st.DataObject,
) -> None:
    language, letters, separator = script
    word = "".join(
        data.draw(
            st.lists(
                st.sampled_from(letters), min_size=word_length, max_size=word_length
            )
        )
    )
    config = Config()
    config.processing.translation.enabled = False
    config.processing.translation.source_language = language
    config.indexing.summary_tree.max_summary_tokens = budget
    summarizer = TreeSummarizer(
        config,
        providers=Providers(
            config,
            None,
            llm_factory=ScriptedLLMFactory(lambda system, human: ""),
            token_counter_factory=lambda model_id, **_: BedrockTokenCounter(
                model_id, None, api_supported=False, estimate_multiplier=multiplier
            ),
        ),
    )
    children = [separator.join([word] * words) for _ in range(3)]

    hint = summarizer.length_hint(children)
    match = re.fullmatch(r"(\d+) (words|characters)", hint)
    assert match, hint
    n, unit = int(match.group(1)), match.group(2)
    assert n >= 1
    if unit == "words":
        summary = separator.join([word] * n)
    else:
        text = separator.join([word] * (n // word_length + 1))
        kept, letters_kept = [], 0
        for ch in text:
            if letters_kept == n:
                break
            kept.append(ch)
            letters_kept += not ch.isspace()
        summary = "".join(kept)
    tokens = summarizer.count_tokens(summary)
    per_unit = summarizer.count_tokens(children[0]) / max(
        (
            len(children[0].split())
            if unit == "words"
            else len("".join(children[0].split()))
        ),
        1,
    )

    assert tokens <= budget
    # Tight: the hint asks for 70% of the budget (_HINT_FILL), less the
    # rounding of a unit and of the count over a small sample.
    assert tokens >= 0.65 * budget - 2 * per_unit - 2 or n == 1
