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
    text_language,
)
from unified_kg_rag.domain.languages.detection import text_evidence
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
    "articles",
    "number_words",
    "zero_words",
    "absence_phrases",
    "year_words",
    "partitive_words",
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
    # Articles: the languages that write them (ru and the scripts without
    # articles wait).
    "articles": {"ja", "ko", "ru", "zh"} | _NEW,
    # Korean, Japanese and Chinese count with numerals before a counter
    # (counter_numerals).
    "number_words": {"ja", "ko", "zh"} | _NEW,
    # Their words for none, of time and of partitives too (the answer
    # matcher reads the dense scripts' own forms for all three).
    "zero_words": {"ja", "ko", "zh"} | _NEW,
    # German states it with its words for none ("Es gibt keine").
    "absence_phrases": {"de", "ja", "ko", "zh"} | _NEW,
    "year_words": {"ja", "ko", "zh"} | _NEW,
    "partitive_words": {"ja", "ko", "zh"} | _NEW,
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


@pytest.mark.parametrize(
    ("question", "language"),
    [
        # Regression (rv8 surfaces): a tie of function words read as English
        # under an English or Korean index; the elision and the article tell.
        ("Qual è la città dell'accordo?", LanguageCode.IT),
        ("Qual è il prezzo?", LanguageCode.IT),
        ("Qual é o preço?", LanguageCode.PT),
        ("Quel est le prix près d\u2019Avignon ?", LanguageCode.FR),
    ],
)
@pytest.mark.parametrize("fallback", [LanguageCode.EN, LanguageCode.KO])
def test_a_latin_tie_is_broken_by_the_registry(
    question: str, language: LanguageCode, fallback: LanguageCode
) -> None:
    assert text_language(question, fallback) is language


@pytest.mark.parametrize(
    ("question", "language"),
    [
        # Regression (rv9 query LM3): "d'" is French and Italian alike and
        # counted for both, then Italian's article "il" ("a-t-il") won.
        ("Combien d'unités Orion Labs a-t-il rejetées ?", LanguageCode.FR),
        ("D'où vient le lidar ?", LanguageCode.FR),
        ("C'est quoi le lidar ?", LanguageCode.FR),
        ("Combien d'argent ?", LanguageCode.FR),
        ("Qu'a inspecté Orion Labs ?", LanguageCode.FR),
        ("Dov'è il deposito?", LanguageCode.IT),
        ("Quanto costa l'unità?", LanguageCode.IT),
        ("Chi ha firmato l'accordo?", LanguageCode.IT),
        # A shared elision tied French and Italian with Spanish.
        ("¿Dónde vive d'Artagnan?", LanguageCode.ES),
    ],
)
def test_only_an_owned_elision_tells_french_from_italian(
    question: str, language: LanguageCode
) -> None:
    assert text_language(question, LanguageCode.EN) is language


@pytest.mark.parametrize(
    ("question", "language"),
    [
        # Regression (rv9 query LM4): the Latin letters of quoted names
        # outweighed the question's own script.
        ("ما علاقة Microsoft Corporation بشركة OpenAI؟", LanguageCode.AR),
        ("Microsoft Corporation和OpenAI是什么关系？", LanguageCode.ZH),
        ("Microsoft Corporation กับ OpenAI เกี่ยวข้องกันอย่างไร", LanguageCode.TH),
        ("Acme Robotics International Holdings 계약", LanguageCode.KO),
        # Regression (rv10 query M6): a function word inside the quoted
        # name ("of", "The", "and") read the question as English.
        ("Bank of America和The Walt Disney Company是什么关系？", LanguageCode.ZH),
        ("University of Tokyo在哪里？", LanguageCode.ZH),
        ("Procter and Gamble的CEO是谁？", LanguageCode.ZH),
        ("ما علاقة Bank of America بـ The Walt Disney Company؟", LanguageCode.AR),
        ("Bank of America와 The Walt Disney Company의 관계는?", LanguageCode.KO),
        ("Bank of AmericaとThe Walt Disney Companyの関係は？", LanguageCode.JA),
        ("Bank of America का CEO कौन है?", LanguageCode.HI),
        ("Orion Robotics and Zortal Holdings 비교", LanguageCode.KO),
        # Latin function words: an English question quoting a name.
        ("Who is 김철수 at Orion Labs?", LanguageCode.EN),
        ("What did 华为 sign with Acme Robotics in 2023?", LanguageCode.EN),
    ],
)
def test_latin_names_quoted_in_another_script(
    question: str, language: LanguageCode
) -> None:
    assert text_language(question, LanguageCode.EN) is language


# Names in another script a Latin-script question quotes, by that script's
# language.
_QUOTED = {
    KO: "한빛전자",
    JA: "株式会社アクメ",
    ZH: "阿克米集团",
    LanguageCode.RU: "Иван Петров",
}


@pytest.mark.parametrize(
    ("template", "language"),
    [
        # Regression (rv11 query M1): a capitalised word opened a name that
        # took in the question's own words ("Why did Orion Labs" was one).
        ("Why did Orion Labs drop {n}?", LanguageCode.EN),
        ("When did Orion Labs acquire {n}?", LanguageCode.EN),
        ("Does Orion Labs supply {n}?", LanguageCode.EN),
        ("Is Orion Labs a vendor of {n}?", LanguageCode.EN),
        ("How much did Orion Labs pay {n}?", LanguageCode.EN),
        ("What did Acme Corp sign with {n}?", LanguageCode.EN),
        ("Who at Acme Corp negotiated with {n}?", LanguageCode.EN),
        ("What is the relationship between Acme Corp and {n}?", LanguageCode.EN),
        ("How is {n} related to Acme Corp?", LanguageCode.EN),
        ("Compare Acme Corp with {n}", LanguageCode.EN),
        ("What Is The Revenue Of {n} Holdings?", LanguageCode.EN),
        ("Who Is The CEO Of {n}?", LanguageCode.EN),
        # German capitalises its nouns.
        ("Wie viele Lieferanten hat {n}?", LanguageCode.DE),
        ("Wer ist der Geschäftsführer von {n}?", LanguageCode.DE),
        ("Seit wann liefert {n} an Orion Labs?", LanguageCode.DE),
        ("Combien de fournisseurs a {n} ?", LanguageCode.FR),
        ("Pourquoi Orion Labs a-t-il quitté {n} ?", LanguageCode.FR),
        ("¿Cuántos proveedores tiene {n}?", LanguageCode.ES),
        ("¿Quién dirige {n}?", LanguageCode.ES),
        ("Quantos fornecedores tem {n}?", LanguageCode.PT),
        ("Quanti fornitori ha {n}?", LanguageCode.IT),
        ("Ai là giám đốc của {n}?", LanguageCode.VI),
        # Regression (rv11 query M5): a short name in a dense script (or
        # Cyrillic) outweighed the whole Latin question around it.
        ("Where does {n} ship to?", LanguageCode.EN),
        ("Who is the CEO of {n}?", LanguageCode.EN),
        ("Who is {n}?", LanguageCode.EN),
        ("Can You Compare {n} And Acme?", LanguageCode.EN),
    ],
)
@pytest.mark.parametrize("quoted", list(_QUOTED), ids=str)
def test_a_latin_question_quoting_a_name_in_another_script(
    template: str, language: LanguageCode, quoted: LanguageCode
) -> None:
    question = template.format(n=_QUOTED[quoted])
    for fallback in (language, quoted):
        assert text_language(question, fallback) is language


@pytest.mark.parametrize(
    ("question", "language"),
    [
        # rv10 M6 with a lowercase article: "the Walt Disney Company" is a
        # name, so the question keeps no English word.
        ("Bank of America和the Walt Disney Company是什么关系？", ZH),
        ("the Acme 계약 조건은?", KO),
        # More runs than function words (and the opening "What") are the
        # question's own words.
        ("What is 계약 조건 기간 요약?", KO),
        ("Acme의 CEO는 누구인가요?", KO),
        ("Python的for循环怎么写？", ZH),
    ],
)
@pytest.mark.parametrize("fallback", [LanguageCode.EN, KO, ZH])
def test_a_quoted_latin_article_and_a_question_of_runs(
    question: str, language: LanguageCode, fallback: LanguageCode
) -> None:
    assert text_language(question, fallback) is language


@pytest.mark.parametrize(
    ("question", "language"),
    [
        # Regression (rv12 query 1): an acronym that is a Latin function
        # word read the question as that word's language ("ai" is
        # Vietnamese, "os" Portuguese, "es" Spanish, "per" Italian).
        ("AI 모델 추천", KO),
        ("AIとは", JA),
        ("AI是什么", ZH),
        ("OS 설치", KO),
        ("ES 인덱스 설정", KO),
        ("ES6 문법 정리", KO),
        ("PER와 PBR 비교", KO),
        ("Что такое AI?", LanguageCode.RU),
        # A part of a compound is no function word.
        ("to-do 앱 추천", KO),
        ("how-to 가이드", KO),
        # One function word against two runs: the runs' script.
        ("with 문 사용법", KO),
    ],
)
def test_an_acronym_or_a_compound_is_no_function_word(
    question: str, language: LanguageCode
) -> None:
    for fallback in (language, LanguageCode.EN):
        assert text_language(question, fallback) is language


def test_the_evidence_leaves_out_codes_compounds_and_names() -> None:
    evidence = text_evidence(
        [
            "Who sells the AI kit, iPhone, a to-do app and ES6 books to Bank of America 은행?"
        ]
    )
    # "the AI" and "Bank of America" are names, "to-do" and "ES6" compounds.
    assert evidence.words == ("who", "sells", "kit", "a", "app", "and", "books", "to")
    assert evidence.openings == {"latin": 1}
    # A text opening on an acronym or a lowercase word opens on no sentence.
    assert text_evidence(["AI 모델"]).openings == {}
    assert text_evidence(["with 문"]).openings == {}


def test_italian_tiene_is_no_spanish_word() -> None:
    # Regression (rv11 query low): "tiene" is Spanish and Italian.
    assert "tiene" not in profile(LanguageCode.ES).function_words
    assert text_language("Il fornitore tiene molti clienti?", LanguageCode.IT) is (
        LanguageCode.IT
    )


def test_letter_runs_are_counted_by_the_group_dominant_weighs() -> None:
    counts = count_letters(["Who is the CEO of 한빛전자 and 株式会社アクメ?"])
    assert counts.runs == {"latin": 6, "dense": 2}
    # A combining mark continues its run (Devanagari vowel signs).
    assert count_letters(["कौन है"]).runs == {"devanagari": 2}


@pytest.mark.parametrize(
    ("question", "language"),
    [
        # Regression (rv10 query M6): Spanish and Portuguese had no "how
        # many" words, so a count question read as English.
        ("¿Cuántos proveedores tiene Acme?", LanguageCode.ES),
        ("¿Cuantas fábricas hay?", LanguageCode.ES),
        ("Quantos fornecedores a Acme tem?", LanguageCode.PT),
        # An all-Latin question keeps its capitalised function words.
        ("What Acme Robotics sells?", LanguageCode.EN),
    ],
)
@pytest.mark.parametrize("fallback", [LanguageCode.EN, LanguageCode.KO])
def test_count_questions_in_spanish_and_portuguese(
    question: str, language: LanguageCode, fallback: LanguageCode
) -> None:
    assert text_language(question, fallback) is language


def test_the_count_numerals_are_the_profiles() -> None:
    # The answer matcher reads the English, Korean and Han number words from
    # the profiles (tables moved unchanged).
    from unified_kg_rag.evaluation import text_matching

    assert text_matching._ENGLISH_UNITS[19] == "nineteen"
    assert text_matching._ENGLISH_TENS["ninety"] == 90
    assert text_matching._KOREAN_NATIVE_TENS["스무"] == 20
    assert text_matching._HAN_DIGITS["兩"] == 2
    assert set(text_matching._HAN_COUNTERS) >= {"社", "家", "ヶ所"}
    for answer, value in [
        ("twenty-one vendors", 21),
        ("열두 곳", 12),
        ("十二社", 12),
        ("两家", 2),
    ]:
        assert text_matching.leading_number(answer) == value
