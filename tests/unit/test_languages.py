# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The language registry (``domain.languages``): which language a text's
script decides."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.languages import (
    PROFILES,
    LanguageCode,
    LanguageProfile,
    dense_language,
    language_name,
    profile,
)
from unified_kg_rag.shared.utils.scripts import (
    ScriptCounts,
    count_letters,
    script_named,
)

pytestmark = pytest.mark.unit

KO, JA, ZH = LanguageCode.KO, LanguageCode.JA, LanguageCode.ZH


@pytest.mark.parametrize(
    ("hangul", "kana", "han", "expected"),
    [
        (10, 0, 0, KO),
        # Korean with Hanja: a tenth Hangul is Korean.
        (1, 0, 9, KO),
        (1, 0, 10, ZH),
        (0, 1, 9, JA),
        (0, 0, 10, ZH),
        # A Korean word in Japanese text: more Kana than Hangul.
        (2, 7, 4, JA),
        (0, 0, 0, None),
    ],
)
def test_dense_language_by_the_hangul_and_kana_shares(
    hangul: int, kana: int, han: int, expected: LanguageCode | None
) -> None:
    counts = ScriptCounts({"hangul": hangul, "kana": kana, "han": han, "latin": 50})
    assert dense_language(counts) is expected


def test_han_alone_is_japanese_on_a_japanese_index() -> None:
    counts = count_letters(["東京本社"])
    assert dense_language(counts) is ZH
    assert dense_language(counts, JA) is JA
    assert dense_language(counts, KO) is ZH


def test_the_dominant_script_weighs_letters_by_their_token_rate() -> None:
    # Nine Latin letters weigh about two Hangul syllables.
    assert count_letters(["AWS Lambda의 실행 시간"]).dominant() == "dense"
    assert count_letters(["What does 계약 mean?"]).dominant() == "latin"
    assert count_letters(["Кто поставщик Acme?"]).dominant() == "cyrillic"
    assert count_letters(["123 ..."]).dominant() is None


def test_every_code_has_its_profile() -> None:
    assert set(PROFILES) == set(LanguageCode)
    for code, language in PROFILES.items():
        assert language.code is code
        script_named(language.script)  # a script of the table
        assert language.name and language.analyzer and language.decline
    assert len({language.decline for language in PROFILES.values()}) == len(PROFILES)


def test_names_and_unknown_codes() -> None:
    assert language_name("ko") == language_name(KO) == "Korean"
    assert language_name("Brazilian Portuguese") == "Brazilian Portuguese"
    # Text that is no language code reads as English.
    assert profile("Klingon") is PROFILES[LanguageCode.EN]


def test_another_bcp47_code_is_accepted_with_defaults_and_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from unified_kg_rag.domain.models.config import TranslationConfig

    with caplog.at_level("WARNING"):
        config = TranslationConfig(source_language="sw-KE", target_language="sw")
        LanguageCode("sw")
    swahili = config.source_language
    assert swahili is config.target_language is LanguageCode("sw")
    assert swahili.value == "sw" and swahili not in list(LanguageCode)
    warnings = [r for r in caplog.records if "'sw' has no profile" in r.getMessage()]
    assert len(warnings) <= 1  # once per process (another test may have warned)
    language = profile(swahili)
    assert (language.name, language.analyzer) == ("sw", "standard")
    assert language.decline == PROFILES[LanguageCode.EN].decline
    assert language_name("sw") == language_name(swahili) == "sw"
    # A region reads as its language; what is no tag is refused.
    assert LanguageCode("pt-BR") is LanguageCode.PT
    assert LanguageCode("zh_Hant") is LanguageCode.ZH
    with pytest.raises(ValueError):
        TranslationConfig(source_language="Portuguese (Brazil)")


def test_the_function_word_lists_of_one_script_are_disjoint() -> None:
    by_script: dict[str, list[frozenset[str]]] = {}
    for language in PROFILES.values():
        by_script.setdefault(language.script, []).append(language.function_words)
    for lists in by_script.values():
        for i, words in enumerate(lists):
            for other in lists[i + 1 :]:
                assert not words & other


def test_no_word_states_two_meanings_across_languages() -> None:
    # A consumer reads the union of every profile's words: a word may not
    # name two statuses, months or polarities in two languages.
    for facet in ("claim_statuses", "month_words"):
        seen: dict[str, str] = {}
        for language in PROFILES.values():
            for meaning, words in getattr(language, facet).items():
                for word in words:
                    assert seen.setdefault(word, meaning) == meaning, (facet, word)
    polarity: dict[str, str] = {}
    for language in PROFILES.values():
        for word, value in language.polarity_words.items():
            assert polarity.setdefault(word, value) == value, word


# What a profile has to hold for every consumer to cover its language, and
# the gaps known today: a language that needs no entry (its script is its
# alone, or it writes dates in digits) or that waits for its words. A new
# language without a facet fails here until it fills it or lists the gap.
_FACETS = (
    "function_words",
    "claim_statuses",
    "no_value_words",
    "open_date_words",
    "month_words",
    "legal_forms",
    "polarity_words",
    "quote_pairs",
)
_DIGIT_DATES = {"ja", "ko", "zh"}
# Languages added with their analyzer, decline and placeholders only.
_NEW = {"ar", "he", "hi", "th", "vi"}
_GAPS: dict[str, set[str]] = {
    # Function words tell apart languages that share a script only.
    "function_words": {"ar", "he", "hi", "ja", "ko", "ru", "th", "zh"},
    "claim_statuses": {"de", "es", "fr", "it", "pt", "ru"} | _NEW,
    "open_date_words": {"de", "es", "fr", "it", "pt", "ru"} | _NEW,
    # English month names are strptime's own; CJK dates are in digits.
    "month_words": {"en"} | _DIGIT_DATES | _NEW,
    # Dense-script legal forms are in dense_legal_forms.
    "legal_forms": {"ko", "ru", "zh"} | _NEW,
    "polarity_words": {"de", "es", "fr", "it", "pt", "ru"} | _NEW,
    "quote_pairs": {"de", "es", "fr", "it", "ko", "pt", "ru"} | _NEW,
}


@pytest.mark.parametrize("facet", _FACETS)
def test_every_profile_covers_each_facet_or_lists_the_gap(facet: str) -> None:
    missing = {
        code.value
        for code, language in PROFILES.items()
        if not getattr(language, facet)
    }
    assert missing == _GAPS.get(facet, set())


def test_a_profile_is_frozen() -> None:
    with pytest.raises(AttributeError):
        PROFILES[KO].name = "x"  # type: ignore[misc]
    assert isinstance(PROFILES[KO], LanguageProfile)
