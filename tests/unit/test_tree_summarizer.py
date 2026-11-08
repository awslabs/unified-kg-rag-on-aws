# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The summaries of the summary tree (``adapters.ingestion.tree_summarizer``)
under a scripted model and the local token counters (AWS-free).

The prompt's length hint comes from the token counter that bounds the
summary, so a model that writes what it is asked for is not cut; a summary
over the budget is cut at its last sentence end within it, in English,
Korean and Japanese alike.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.adapters.aws.token_counter import BedrockTokenCounter
from unified_kg_rag.adapters.ingestion.tree_summarizer import TreeSummarizer
from unified_kg_rag.adapters.local.token_counter import LocalTokenCounter
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.ingestion.summary_tree import Cluster, Entry
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import LanguageCode
from unified_kg_rag.shared.utils.sentences import sentence_ends

pytestmark = pytest.mark.unit

# One sentence per language, about as dense as encyclopedia prose.
SENTENCES = {
    LanguageCode.EN: (
        "The depot north of the river stores industrial widgets for the "
        "regional vendor every month."
    ),
    LanguageCode.KO: "강 북쪽의 창고는 매달 지역 벤더를 위해 산업용 위젯을 보관한다.",
    LanguageCode.JA: "川の北にある倉庫は毎月地域の業者のために工業用部品を保管する。",
}
# Languages only the length hint's unit is checked for (Thai ends a sentence
# with no mark).
_HINT_SENTENCES = {
    LanguageCode.TH: "คลังสินค้าทางเหนือของแม่น้ำเก็บชิ้นส่วนให้ผู้ขายทุกเดือน",
    LanguageCode.HI: "नदी के उत्तर का गोदाम हर महीने विक्रेता के लिए पुर्ज़े रखता है।",
}
_HINT = re.compile(r"in (\d+) (words|characters) at most")


def _units(text: str, unit: str) -> int:
    return len(text.split()) if unit == "words" else len("".join(text.split()))


class _Writer:
    """A model that writes whole sentences up to ``scale`` times the hint."""

    def __init__(self, language: LanguageCode, scale: float) -> None:
        self.language = language
        self.scale = scale
        self.answers: list[str] = []

    def __call__(self, system: str, human: str) -> str:
        sentence = SENTENCES[self.language]
        joiner = "" if self.language is LanguageCode.JA else " "
        match = _HINT.search(human)
        assert match, human
        wanted = int(int(match.group(1)) * self.scale)
        sentences: list[str] = []
        while _units(joiner.join([*sentences, sentence]), match.group(2)) <= wanted:
            sentences.append(sentence)
        self.answers.append(joiner.join(sentences or [sentence]))
        return self.answers[-1]


_COUNTERS = {
    # The local estimate, and the estimate as the Bedrock counter scales it
    # for a model whose tokenizer counts ~30% more (Claude 5, no CountTokens).
    "estimate": lambda model_id: LocalTokenCounter(model_id),
    "bedrock-scaled": lambda model_id: BedrockTokenCounter(
        model_id, None, api_supported=False, estimate_multiplier=1.3
    ),
}


def _summarizer(
    language: LanguageCode,
    respond: Callable[[str, str], str],
    counter: str = "bedrock-scaled",
    max_summary_tokens: int = 300,
) -> TreeSummarizer:
    config = Config()
    config.processing.translation.enabled = False
    config.processing.translation.source_language = language
    config.indexing.summary_tree.max_summary_tokens = max_summary_tokens
    config.retry.max_attempts = 1
    providers = Providers(
        config,
        None,
        llm_factory=ScriptedLLMFactory(respond),
        token_counter_factory=lambda model_id, **_: _COUNTERS[counter](model_id),
    )
    return TreeSummarizer(config, providers=providers)


def _cluster(language: LanguageCode, n: int = 6) -> Cluster:
    sentence = {**SENTENCES, **_HINT_SENTENCES}[language]
    members = tuple(
        Entry(
            id=f"u{i}",
            level=0,
            text=" ".join([sentence] * 4),
            vector=np.zeros(2),
            n_tokens=100,
            leaves=frozenset({f"u{i}"}),
            filters={},
        )
        for i in range(n)
    )
    return Cluster(level=1, members=members)


def _ends_a_sentence(text: str) -> bool:
    return bool(sentence_ends(text)) and sentence_ends(text)[-1] == len(text)


@pytest.mark.parametrize("counter", sorted(_COUNTERS))
@pytest.mark.parametrize("language", sorted(SENTENCES))
def test_a_summary_of_the_hinted_length_is_within_the_budget(
    language: LanguageCode, counter: str
) -> None:
    # The model writes as much as it is asked for: the counter that cuts
    # must not find it over the budget (the hint was 225 words for 300
    # tokens, which this counter makes ~390 tokens of English).
    writer = _Writer(language, 1.0)
    summarizer = _summarizer(language, writer, counter)
    [summary] = summarizer.summarize([_cluster(language)])

    assert summary is not None
    assert summary.n_tokens <= 300
    # Not cut: as written, and most of the hinted length used.
    assert summary.text == writer.answers[-1]
    assert _ends_a_sentence(summary.text)
    assert summary.n_tokens >= 300 * 0.55


@pytest.mark.parametrize("counter", sorted(_COUNTERS))
@pytest.mark.parametrize("language", sorted(SENTENCES))
def test_a_model_writing_past_the_hint_as_measured_is_not_cut(
    language: LanguageCode, counter: str
) -> None:
    # Regression (real Bedrock): Claude Haiku wrote ~1.4 times the hinted
    # length, and with the hint at 0.9 of the budget 70% of the summaries
    # were cut. The hint leaves that room.
    writer = _Writer(language, 1.35)
    summarizer = _summarizer(language, writer, counter)
    [summary] = summarizer.summarize([_cluster(language)])

    assert summary is not None
    assert summary.text == writer.answers[-1]
    assert summary.n_tokens <= 300


@pytest.mark.parametrize("language", sorted(SENTENCES))
def test_an_over_long_summary_is_cut_at_its_last_sentence_end_in_the_budget(
    language: LanguageCode,
) -> None:
    writer = _Writer(language, 3.0)
    summarizer = _summarizer(language, writer)
    [summary] = summarizer.summarize([_cluster(language)])

    assert summary is not None
    assert summary.text and writer.answers[-1].startswith(summary.text)
    assert summarizer.count_tokens(writer.answers[-1]) > 300
    assert summary.n_tokens <= 300
    assert summary.n_tokens == summarizer.count_tokens(summary.text)
    assert _ends_a_sentence(summary.text)
    # The longest such prefix: one more sentence would not fit.
    sentence = SENTENCES[language]
    joiner = "" if language is LanguageCode.JA else " "
    assert summarizer.count_tokens(summary.text + joiner + sentence) > 300


@pytest.mark.parametrize("counter", sorted(_COUNTERS))
@pytest.mark.parametrize(
    ("language", "word", "joiner"),
    [
        (LanguageCode.EN, "widgets", " "),
        (LanguageCode.KO, "창고에서", " "),
        (LanguageCode.JA, "倉庫の部品", ""),
    ],
)
def test_a_first_sentence_over_the_budget_is_cut_at_a_character(
    counter: str, language: LanguageCode, word: str, joiner: str
) -> None:
    long_sentence = joiner.join([word] * 400) + ("。" if not joiner else ".")
    summarizer = _summarizer(
        language, lambda system, human: long_sentence, counter=counter
    )
    [summary] = summarizer.summarize([_cluster(language)])

    assert summary is not None
    # The counter's own cut: within the budget and most of it used.
    assert 0.8 * 300 < summary.n_tokens <= 300
    assert long_sentence.startswith(summary.text)


def test_the_hint_counts_characters_for_dense_languages_and_words_otherwise() -> None:
    for language, unit in [
        (LanguageCode.EN, "words"),
        (LanguageCode.KO, "characters"),
        (LanguageCode.JA, "characters"),
        # Thai writes no spaces between words either.
        (LanguageCode.TH, "characters"),
        (LanguageCode.HI, "words"),
    ]:
        summarizer = _summarizer(language, _Writer(language, 1.0))
        hint = summarizer.length_hint(_cluster(language).child_texts)
        assert hint.endswith(f" {unit}"), (language, hint)


@pytest.mark.parametrize(
    ("text", "ends"),
    [
        ("Alpha is 3.5 m tall. Beta, e.g. the one, is not! Gamma", [20, 48]),
        ('He said "go." Then', [13]),
        ("東京は大きい。大阪も「大きい」。京都", [7, 16]),
        ("창고는 위젯을 보관한다. 은행은 돈을 빌려준다? 감사", [13, 26]),
        ("Москва большая. Казань тоже.", [15, 28]),
        ("गोदाम बड़ा है। बैंक भी।", [14, 23]),
        ("المستودع كبير؟ نعم.", [14, 19]),
        ("A heading\nA line.", [9, 17]),
        ("No stop at all", []),
    ],
)
def test_sentence_ends(text: str, ends: list[int]) -> None:
    assert sentence_ends(text) == ends
