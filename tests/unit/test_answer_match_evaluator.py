# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the deterministic answer-match evaluator (AWS-free)."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import (
    Config,
    EvaluationQuery,
    EvaluationResult,
    EvaluatorType,
    ModelRole,
)
from unified_kg_rag.evaluation import (
    AnswerMatchEvaluator,
    EvaluationManager,
    answer_match_evaluator,
    text_matching,
)
from unified_kg_rag.evaluation.answer_match_evaluator import (
    answer_aliases,
    answer_contains,
    exact_match,
    first_answer_line,
    normalize_answer,
    token_f1,
)
from unified_kg_rag.shared import EvaluationException

pytestmark = pytest.mark.unit


def test_answer_aliases_read_a_string_or_a_list_of_strings() -> None:
    # One reader for the evaluator and the run's language check: a string
    # is one alias, and what is no string in a list is no alias.
    assert answer_aliases({"answer_aliases": "서울"}) == ["서울"]
    assert answer_aliases({"answer_aliases": ["東京", 3, None, "Tokyo"]}) == [
        "東京",
        "Tokyo",
    ]
    assert answer_aliases({}) == answer_aliases({"answer_aliases": None}) == []


class TestNormalization:
    def test_squad_normalization(self) -> None:
        assert normalize_answer("The  Vendor, Inc.!") == "vendor inc"

    def test_unicode_punctuation_removed(self) -> None:
        assert normalize_answer("「공급사」는…") == "공급사는"

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [
            ("1,000", "1000"),  # punctuation deleted, not replaced by a space
            ("USD 1,000.", "usd 1000"),
            ("\uff11\uff10\uff10\uff10", "1000"),  # fullwidth digits (NFKC)
            ("Vendor\u2019s", "Vendors"),
            ("the U.S.", "US"),
            ("2,000", "2000"),
            ("a 5-year term", "5 year term"),
            ("3.5%", "3.5 %"),
        ],
    )
    def test_squad_equivalences(self, prediction: str, reference: str) -> None:
        assert exact_match(prediction, reference) == 1.0

    @pytest.mark.parametrize(
        ("text", "normalized"),
        [
            ("3.5 days", "3.5 days"),  # decimal point kept: 3.5 is not 35
            ("1,000,000.", "1000000"),  # thousands separators dropped
            ("1,5", "1,5"),  # a decimal comma is kept
            ("C++ and C#!", "c++ and c#"),
            ("#1 vendor", "1 vendor"),
            ("A", "a"),  # only articles: kept rather than emptied
            ("The.", "the"),
        ],
    )
    def test_meaningful_punctuation_kept(self, text: str, normalized: str) -> None:
        assert normalize_answer(text) == normalized

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [
            ("about 1.5 million users", "15 million"),
            ("The rate is 3.5.", "35 percent"),
            ("Written in C++.", "C"),
            ("The balance is -5.", "5"),
            ("A rate of 3.5% applies.", "3.5 million"),
            ("A rate of 3.5% applies.", "35%"),
            ("From 1990-1995.", "19901995"),
            ("금액은 3억이다.", "3"),
        ],
    )
    def test_different_numbers_and_names_not_contained(
        self, prediction: str, reference: str
    ) -> None:
        assert answer_contains(prediction, reference) == 0.0
        assert exact_match(prediction, reference) == 0.0

    @pytest.mark.parametrize(
        ("prediction", "reference", "expected"),
        [
            ("Yes, the Buyer may terminate.", "yes", 1.0),
            ("No. There is no such clause.", "no", 1.0),
            # the gold appears, but the answer's polarity is the opposite
            ("Yes, although no notice is required.", "no", 0.0),
            # "there is no X" is a determiner, not the answer
            ("There is no clause. Yes, it was signed.", "yes", 1.0),
            ("There is no evidence. So, yes.", "yes", 1.0),
            ("There is no evidence. So, yes.", "no", 0.0),
            ("The contract does not say.", "no", 0.0),
            ("Based on the context, no.", "No", 1.0),
            ("No, the Buyer may not.", "no", 1.0),
            ("**No**", "no", 1.0),
            # a markdown heading before the answer, or an abbreviation in it
            ("## Summary\nYes, it is.", "yes", 1.0),
            ("## Summary\nNo.\nThe clause was removed.", "no", 1.0),
            ("The U.S. law says yes.", "yes", 1.0),
            ("Clause No. 5 says yes.", "yes", 1.0),
            ("Clause No. 5 says yes.", "no", 0.0),
            # options named with "/" are not the answer
            ("Yes/No: No", "no", 1.0),
            ("Yes/No: No", "yes", 0.0),
            ("Answer (yes / no): yes", "yes", 1.0),
            # answering punctuation, a dash and the line end still answer
            ("Answer: No", "no", 1.0),
            ("No—the Buyer pays.", "no", 1.0),
            ("1. No, the Buyer pays.", "no", 1.0),
            ("| Warranty? | Yes |", "yes", 1.0),
            ("No warranty is provided.", "no", 0.0),
            # any punctuation after "no" answers, markdown emphasis included
            ("No: the contract does not allow it.", "no", 1.0),
            ("**No**: the clause forbids it.", "no", 1.0),
            ("No (see Clause 5).", "no", 1.0),
            ("No - it does not.", "no", 1.0),
            ("No: Yes it is", "no", 1.0),
            ("No: Yes it is", "yes", 0.0),
            # emphasis or an opening quote after "no" starts the next word
            ("There is no **cap** on fees. Yes, the Vendor may.", "yes", 1.0),
            ("There is **no** penalty clause. Yes, the Buyer may.", "yes", 1.0),
            ('There is no "cap" on fees. Yes, the Vendor may.', "yes", 1.0),
            ("There is no `cap` field. Yes, it is allowed.", "yes", 1.0),
            ("There is no \u201ccap\u201d on fees. Yes.", "yes", 1.0),
            ('He said "no".', "no", 1.0),
            ("\u201cNo\u201d, it is not.", "no", 1.0),
            ("| Question | Answer |\n|---|---|\n| Warranty? | **No** |", "no", 1.0),
            ("Nope, the Buyer pays.", "no", 1.0),
            # a hyphen joins "no" to the next word, the non-breaking one too
            ("No-cap clause. Yes, it applies.", "yes", 1.0),
            ("No\u2011cap clause. Yes, it applies.", "yes", 1.0),
            ("No\u2010cap clause. Yes, it applies.", "yes", 1.0),
            # known limit: other punctuation before a word still reads as "no"
            ("There is no (explicit) cap; yes, it may.", "no", 1.0),
        ],
    )
    def test_yes_no_gold_scored_by_leading_polarity(
        self, prediction: str, reference: str, expected: float
    ) -> None:
        assert answer_contains(prediction, reference) == expected

    @pytest.mark.parametrize(
        ("prediction", "reference", "expected"),
        [
            # the gold word appears, but the answer is the opposite polarity
            ("아니요, 계약서에는 예외 조항은 없습니다.", "예", 0.0),
            ("아니요, 계약서에는 예외 조항은 없습니다.", "아니요", 1.0),
            ("예, 해지할 수 있습니다.", "예", 1.0),
            ("네, 맞습니다.", "예", 1.0),
            ("아니요, 없습니다.", "아니오", 1.0),
            # inside a longer word, or "네" the numeral, it is no answer
            ("예를 들어 계약서에는 그런 조항이 없습니다.", "예", 0.0),
            ("네트워크 장비 공급 계약입니다.", "네", 0.0),
            ("네 가지 경우가 있습니다. 아니요.", "네", 0.0),
            ("아니요라고 할 수 없습니다. 예, 맞습니다.", "아니요", 0.0),
            ("不是，没有这样的条款。", "是", 0.0),
            ("是的，买方可以终止。", "是", 1.0),
            ("这是一个条款。", "否", 0.0),
            ("不是的，没有。", "否", 1.0),
            ("いいえ、ありません。", "はい", 0.0),
            ("はい、あります。", "はい", 1.0),
            ("いいえ、ありません。", "いいえ", 1.0),
            # "예:" / "예)" abbreviate "for example", not "yes"
            ("계약은 자동 갱신되지 않습니다 (예: 만료 시 종료).", "예", 0.0),
            (
                "예: 공급자가 위반한 경우 해지됩니다. 아니요, 갱신은 없습니다.",
                "아니요",
                1.0,
            ),
            ("예) 2024년 계약서. 아니요.", "아니요", 1.0),
            ("질문 (네/아니요): 아니요", "아니요", 1.0),
            ("질문 (네/아니요): 아니요", "네", 0.0),
            ("질문 2 (네/아니요): 네", "네", 1.0),
            # non-misfires: still answers
            ("**네**, 맞습니다.", "예", 1.0),
            ("답변: 예", "예", 1.0),
            ("1. 예, 구매자가 부담합니다.", "예", 1.0),
            ("예외적으로, 아니요.", "아니요", 1.0),
            ("根据合同第3条，是。", "是", 1.0),
            ("合同规定：是", "是", 1.0),
            ("不是。卖方不承担运费。", "否", 1.0),
            ("아닙니다. 상한이 없습니다.", "아니요", 1.0),
            ("아닙니다. 상한이 없습니다.", "예", 0.0),
            ("没有，合同里没有上限。", "否", 1.0),
            ("没有，合同里没有上限。", "是", 0.0),
            # inside a longer word "没有" is no answer ("有没有": whether)
            ("合同有没有上限？是的，有。", "是", 1.0),
            ("否则，买方可以终止合同。", "否", 0.0),
            ("倉庫にはいります。", "はい", 0.0),
            # "아니요"/"아니오" have no second meaning: any punctuation answers
            ("아니요: 계약서에는 없습니다.", "아니요", 1.0),
            ("**아니요**: 없습니다", "아니요", 1.0),
            ("아니오 (계약 제5조).", "아니오", 1.0),
            ("아니요 계약서에는 그런 내용이 없습니다.", "아니요", 1.0),
            ("**네**: 맞습니다.", "네", 1.0),
            # emphasis or an opening quote after the numeral "네" starts a word
            ("계약에는 네 **가지** 조건이 있습니다. 아니요.", "아니요", 1.0),
            ('네 "가지" 조건이 있습니다. 아니요.', "아니요", 1.0),
            ("- **아니요** — 없습니다.", "아니요", 1.0),
            ("「いいえ」、買主は支払いません。", "いいえ", 1.0),
            ("**是的**，卖方承担。", "是", 1.0),
            ("答案：否。", "否", 1.0),
            ("回答:不是。", "否", 1.0),
        ],
    )
    def test_cjk_yes_no_gold_scored_by_leading_polarity(
        self, prediction: str, reference: str, expected: float
    ) -> None:
        assert answer_contains(prediction, reference) == expected

    def test_korean_number_with_a_unit_is_contained(self) -> None:
        assert answer_contains("계약은 2024년에 체결되었다", "2024") == 1.0
        assert answer_contains("계약 금액은 3억 원이다", "3") == 0.0

    def test_range_is_two_numbers(self) -> None:
        assert token_f1("1990-1995", "1990 1995") == 1.0
        assert answer_contains("It ran from 1990-1995.", "1995") == 1.0

    def test_answer_contains_korean_particle(self) -> None:
        assert (
            answer_contains("공급사의 본사는 서울은 아니고 부산입니다.", "부산") == 1.0
        )
        assert answer_contains("본사는 서울 특별시에 있습니다.", "서울 특별시") == 1.0
        assert answer_contains("본사는 부산에 있습니다.", "서울") == 0.0

    def test_articles_only_as_whole_words(self) -> None:
        assert normalize_answer("Theater an Arena") == "theater arena"

    def test_exact_match_ignores_case_articles_punctuation(self) -> None:
        assert exact_match("the Buyer.", "Buyer") == 1.0
        assert exact_match("Buyer Ltd", "Buyer") == 0.0

    def test_token_f1(self) -> None:
        # pred tokens {vendor, ships, parts}, ref {vendor, ships, tools}
        assert token_f1("Vendor ships parts", "vendor ships tools") == pytest.approx(
            2 / 3
        )
        assert token_f1("unrelated", "vendor") == 0.0
        assert token_f1("", "") == 1.0
        assert token_f1("a", "vendor") == 0.0


class TestScriptAwareTokenF1:
    """Regression: tokens were whitespace words, so a Chinese or Japanese
    answer was one token (F1 0 unless exact) and a Korean word with a
    particle never matched its gold word."""

    @pytest.mark.parametrize(
        ("prediction", "reference", "expected"),
        [
            # 3 of 5 characters / 3 of 3: P 0.6, R 1.
            ("東京都です。", "東京都", 0.75),
            ("北京市是中国的首都", "北京市", 0.5),
            # "서울은" counts as "서울"; then 2 of 10 syllables.
            ("서울은 한국의 수도입니다.", "서울", 2 * 0.2 / 1.2),
            ("서울은", "서울", 1.0),
            ("2024年に締結", "2024年", 2 * 0.4 / 1.4),
            ("대금은 3억 원입니다", "3억 원", 2 * 0.5 / 1.5),
            ("東京", "大阪", 0.0),
            ("The capital is Tokyo.", "Tokyo", 0.5),
        ],
    )
    def test_dense_scripts_score_by_character(
        self, prediction: str, reference: str, expected: float
    ) -> None:
        assert token_f1(prediction, reference) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("prediction", "reference", "expected"),
        [
            # Particles and copula endings of the closed class, alone or
            # in a run, cost nothing.
            ("서울은", "서울", 1.0),
            ("서울에서는", "서울", 1.0),
            ("서울입니다", "서울", 1.0),
            ("서울이었다", "서울", 1.0),
            ("2024년에", "2024년", 1.0),
            ("3명이", "3명", 1.0),
            # Other Hangul after a gold word is another word: 2 of 5 letters.
            ("서울대학교", "서울", 2 * 0.4 / 1.4),
            ("서울특별시", "서울", 2 * 0.4 / 1.4),
            ("이순신장군", "이순신", 2 * 0.6 / 1.6),
            ("부산시청", "부산", 2 * 0.5 / 1.5),
            ("김철수", "김", 2 * (1 / 3) / (4 / 3)),
        ],
    )
    def test_only_korean_particles_are_dropped(
        self, prediction: str, reference: str, expected: float
    ) -> None:
        """Regression: any one to three Hangul syllables after a gold word
        were dropped as a particle, so "서울대학교" scored 1.0 against
        "서울"."""
        assert token_f1(prediction, reference) == pytest.approx(expected)

    def test_containment_stays_lenient(self) -> None:
        # answer_contains (recall-style) keeps the wider rule.
        assert answer_contains("서울특별시에 있다", "서울") == 1.0

    def test_the_official_normalizations_keep_whitespace_tokens(
        self, config: Config
    ) -> None:
        assert token_f1("東京都です", "東京都", script_aware=False) == 0.0
        config.evaluation.answer_normalization = "official_squad"
        report = _report(config, "東京都です", "東京都", {})
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values["token_f1"] == 0.0
        assert report.metadata["token_f1_tokens"] == "whitespace"
        config.evaluation.answer_normalization = "default"
        report = _report(config, "東京都です", "東京都", {})
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values["token_f1"] == pytest.approx(2 * 0.6 / 1.6)
        assert report.metadata["token_f1_tokens"] == "script_aware_v2"


class TestLanguageNormalization:
    @pytest.mark.parametrize(
        ("language", "text", "normalized"),
        [
            ("de", "Der Rhein", "rhein"),
            ("de", "eines Lieferanten", "lieferanten"),
            ("es", "Los Andes", "andes"),
            ("fr", "L'accord de la Banque", "accord de banque"),
            ("fr", "l\u2019usine", "usine"),
            ("it", "Gli Stati e l'Europa", "stati e europa"),
            ("it", "un'altra fabbrica", "altra fabbrica"),
            # Regression (rv8 surfaces M2): every elision is split, only the
            # articles are dropped ("près d'Avignon" read "davignon").
            ("fr", "près d'Avignon", "près d avignon"),
            ("fr", "jusqu'à Lyon", "jusqu à lyon"),
            ("it", "dell'accordo", "dell accordo"),
            ("it", "nell'Italia", "nell italia"),
            # French elides no "un".
            ("fr", "un'usine", "unusine"),
            ("pt", "Os Estados e a Vendedora", "estados e vendedora"),
            # English articles go in every language; only articles stay.
            ("de", "The Beatles", "beatles"),
            ("fr", "La", "la"),
        ],
    )
    def test_the_answers_language_drops_its_articles(
        self, language: str, text: str, normalized: str
    ) -> None:
        assert normalize_answer(text, language) == normalized

    @pytest.mark.parametrize("language", [None, "en", "ko", "ja", "zh", "ru"])
    def test_other_languages_drop_english_articles_only(
        self, language: str | None
    ) -> None:
        # "die" is no English article, "l'" no elision outside fr/it.
        assert normalize_answer("Die Hard and the l'Oreal", language) == (
            "die hard and loreal"
        )
        assert normalize_answer("서울의 The Shilla", language) == "서울의 shilla"

    def test_containment_reads_the_language(self) -> None:
        assert answer_contains("Er fließt durch Rhein und Main.", "der Rhein", "de")
        assert not answer_contains("Er fließt durch Rhein.", "der Rhein")

    @pytest.mark.parametrize(
        ("language", "answer", "number"),
        [
            ("de", "Drei Lieferanten liefern.", 3),
            ("de", "Zwölf.", 12),
            ("fr", "Trois fournisseurs.", 3),
            ("es", "Dos proveedores.", 2),
            ("it", "Quattro fornitori.", 4),
            ("pt", "Três fornecedores e duas fábricas.", 3),
            ("pt", "Duas fábricas.", 2),
            ("ru", "Три поставщика.", 3),
            ("ru", "Двенадцать заводов.", 12),
            # The article forms of one are no count: the digits count.
            ("fr", "Un fournisseur sur 4.", 4),
            ("de", "Eine Fabrik mit 7 Hallen.", 7),
            # Regression (rv9 query M1): a compound read as its first or
            # last word ("Vingt-quatre" 4, "Zwei Millionen" 2).
            ("fr", "Vingt-quatre unités.", 24),
            ("fr", "Vingt et un fournisseurs.", 21),
            ("fr", "Quatre-vingt-dix-sept fournisseurs.", 97),
            ("fr", "Trois cents unités.", 300),
            ("pt", "Vinte e quatro unidades.", 24),
            ("pt", "Duzentos e trinta e dois.", 232),
            ("es", "Treinta y dos proveedores.", 32),
            ("es", "Veinticuatro proveedores.", 24),
            ("es", "Dos mil millones.", 2_000_000_000),
            ("de", "Zwei Millionen Euro.", 2_000_000),
            ("de", "Vierundzwanzig Hallen.", 24),
            ("de", "Einundzwanzig Hallen.", 21),
            ("it", "Duecentotrentadue unità.", 232),
            ("it", "Ventotto fornitori.", 28),
            ("ru", "Двадцать одна единица.", 21),
            ("ru", "Три тысячи единиц.", 3000),
            ("ru", "Двести тридцать два.", 232),
        ],
    )
    def test_number_words_of_the_language_count(
        self, language: str, answer: str, number: int
    ) -> None:
        assert answer_match_evaluator.number_matches(answer, str(number), language)
        # Not read as English.
        if not any(ch.isdigit() for ch in answer):
            assert not answer_match_evaluator.number_matches(answer, str(number))

    @pytest.mark.parametrize(
        ("language", "answer", "number"),
        [
            # Regression (rv10 query M1): English "no" read as 0 in every
            # language, before the language's own number.
            ("pt", "No total, são três fornecedores.", 3),
            ("pt", "No contrato, a Acme tem 3 fornecedores.", 3),
            ("es", "Según el informe, que no detalla fechas, hay 3 proveedores.", 3),
            ("es", "No uno sino tres proveedores.", 3),
            # Each language's own words for none.
            ("fr", "Aucun fournisseur.", 0),
            ("de", "Kein Lieferant.", 0),
            ("de", "Acme hat keine Lieferanten.", 0),
            ("es", "Ningún proveedor.", 0),
            ("pt", "Nenhum fornecedor.", 0),
            ("it", "Nessun fornitore.", 0),
            ("ru", "Ни одного поставщика.", 0),
            # Regression (rv11 query M4): each language's "there is none",
            # read as English's "not ... any" is.
            ("es", "No hay proveedores.", 0),
            ("es", "Acme no tiene proveedores.", 0),
            ("pt", "Não há fornecedores.", 0),
            ("it", "Non ci sono fornitori.", 0),
            ("fr", "Il n'y a pas de fournisseur.", 0),
            ("fr", "Il n\u2019y a pas d\u2019usine.", 0),
            ("ru", "Поставщиков нет.", 0),
            ("ru", "У Acme нет поставщиков.", 0),
            ("en", "Acme doesn't have any vendors.", 0),
            ("en", "Acme does not supply any buyers.", 0),
            # A yes/no answer opening the line is no "none".
            ("ru", "Нет, у Acme три поставщика.", 3),
            ("ru", "**Нет**. Три поставщика.", 3),
            ("es", "Tiene tres proveedores; no hay más.", 3),
            # Low (rv11 query): a CJK word after a year, not only 년/年.
            ("ko", "2019부터 3곳", 3),
            ("ko", "2019 이후 3곳", 3),
            ("ja", "2019以降、3社", 3),
            ("zh", "从2019到现在有3家", 3),
            # Regression (rv10 query M7): a year, the whole of a partitive
            # and "one of them" in the language's own words.
            ("de", "Seit 2019 drei Lieferanten.", 3),
            ("fr", "Depuis 2019, trois fournisseurs.", 3),
            ("es", "En 2019 firmó con tres proveedores.", 3),
            ("es", "En marzo de 2019 firmó con tres proveedores.", 3),
            ("pt", "Desde 2019, dois fornecedores.", 2),
            ("it", "Nel 2019 tre fornitori.", 3),
            ("ru", "В 2019 году три поставщика.", 3),
            ("es", "De los tres proveedores, solo dos firmaron.", 2),
            ("pt", "Dos três fornecedores, só dois assinaram.", 2),
            ("it", "Dei tre fornitori, solo due hanno firmato.", 2),
            ("ru", "Один из них подписал; всего три поставщика.", 3),
            # Low: Italian joins a number with "e", French writes
            # quatre-vingts with a space too.
            ("it", "Mille e cinquecento unità.", 1500),
            ("fr", "Quatre vingts fournisseurs.", 80),
            ("fr", "Quatre vingt dix fournisseurs.", 90),
        ],
    )
    def test_the_languages_own_words_around_a_number(
        self, language: str, answer: str, number: int
    ) -> None:
        assert text_matching.leading_number(answer, language) == number

    @pytest.mark.parametrize(
        ("language", "answer"),
        [
            ("fr", "deux trois"),
            ("de", "zwei und drei"),
            ("es", "un proveedor"),
            # Low (rv10 query): only German writes the ones before the tens.
            ("fr", "deux vingt"),
            ("es", "tres veinte"),
            ("ru", "два двадцать"),
            # Ordinals: their tens count nothing.
            ("fr", "Il est arrivé vingt-troisième."),
            ("ru", "двадцать первый"),
            ("en", "the twenty-first vendor"),
            # A partitive's whole, and "one of them".
            ("de", "Von den drei Lieferanten hat nur einer unterschrieben."),
            ("ru", "один из них"),
        ],
    )
    def test_words_that_state_no_count(self, language: str, answer: str) -> None:
        assert text_matching.leading_number(answer, language) is None

    def test_the_evaluator_normalizes_in_the_questions_language(
        self, config: Config
    ) -> None:
        evaluator = AnswerMatchEvaluator(config)
        result = EvaluationResult(
            query_id="q",
            question="Welcher Fluss fließt durch die Stadt?",
            generated_answer="Der Rhein",
            ground_truth="Rhein",
        )
        report = evaluator.evaluate_single(
            EvaluationQuery(query_id="q", question=result.question), result, "Rhein"
        )
        assert _values(report)["exact_match"] == 1.0
        assert report.metadata["answer_language"] == "de"
        english = _report(config, "Der Rhein", "Rhein", {})
        assert english.metadata["answer_language"] == "en"
        assert _values(english)["exact_match"] == 0.0


def _report(
    config: Config,
    answer: str,
    ground_truth: str,
    metadata: dict,
    question: str = "?",
):
    evaluator = AnswerMatchEvaluator(config)
    result = EvaluationResult(
        query_id="q",
        question=question,
        generated_answer=answer,
        ground_truth=ground_truth,
        metadata=metadata,
    )
    return evaluator.evaluate_single(
        EvaluationQuery(query_id="q", question=question), result, ground_truth
    )


class TestEvaluator:
    def test_max_over_aliases(self, config: Config) -> None:
        report = _report(
            config,
            "Net 30 days",
            "thirty days",
            {"answer_aliases": ["net 30 days", "30 days"]},
        )
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values == {"answer_contains": 1.0, "exact_match": 1.0, "token_f1": 1.0}

    def test_single_string_alias_accepted(self, config: Config) -> None:
        report = _report(config, "USD 500", "", {"answer_aliases": "usd 500"})
        assert {m.metric_type.value: m.value for m in report.metrics} == {
            "answer_contains": 1.0,
            "exact_match": 1.0,
            "token_f1": 1.0,
        }

    def test_no_reference_skipped(self, config: Config) -> None:
        report = _report(config, "anything", "", {})
        assert report.metrics == []
        assert set(report.metadata["skipped_metrics"]) == {
            "answer_contains",
            "exact_match",
            "token_f1",
        }

    def test_article_only_gold_is_scored(self, config: Config) -> None:
        report = _report(config, "A.", "A", {})
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values == {"answer_contains": 1.0, "exact_match": 1.0, "token_f1": 1.0}

    def test_long_answer_contains_gold(self, config: Config) -> None:
        report = _report(
            config, "Payment is due within net 30 days of invoice.", "Net 30 days", {}
        )
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values["answer_contains"] == 1.0
        assert values["exact_match"] == 0.0

    def test_contains_matches_alias(self, config: Config) -> None:
        report = _report(
            config,
            "The Buyer pays USD 1,000.",
            "one thousand",
            {"answer_aliases": ["1000"]},
        )
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values["answer_contains"] == 1.0

    def test_contains_miss(self, config: Config) -> None:
        report = _report(config, "The Vendor ships parts.", "Buyer", {})
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values["answer_contains"] == 0.0

    def test_resolver_maps_type(self) -> None:
        assert (
            EvaluationManager._resolve_evaluator_class(EvaluatorType.ANSWER_MATCH)
            is AnswerMatchEvaluator
        )


# The answer prompt's shape: a one-sentence answer, then its support.
_DEPOT_ANSWER = (
    "## Answer\n\nDepot stores widgets for Vendor.\n"
    "Support: the context links Depot and Vendor through widget flows."
)


def _values(report) -> dict[str, float]:
    return {m.metric_type.value: m.value for m in report.metrics}


class TestAnswerSpan:
    def test_first_line_skips_headings_and_blank_lines(self) -> None:
        assert first_answer_line(_DEPOT_ANSWER) == "Depot stores widgets for Vendor."
        assert first_answer_line("# Only a heading") == "# Only a heading"

    @pytest.mark.parametrize(("span", "f1"), [("full", 1 / 8), ("first_line", 1 / 3)])
    def test_em_and_f1_read_the_configured_span(
        self, config: Config, span: str, f1: float
    ) -> None:
        # Before, EM/F1 always read the whole answer: the supporting
        # sentence diluted F1 to 1/8 for a correct answer.
        config.evaluation.answer_span = span
        report = _report(config, _DEPOT_ANSWER, "Depot", {})
        values = _values(report)
        assert values["token_f1"] == pytest.approx(f1)
        assert values["exact_match"] == 0.0
        # answer_contains always reads the whole answer.
        assert values["answer_contains"] == 1.0
        assert report.metadata["answer_span"] == span

    def test_the_default_is_first_line(self) -> None:
        assert Config().evaluation.answer_span == "first_line"


class _SpanChain:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def ainvoke(self, inputs: dict) -> str:
        self.calls.append(inputs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _extracting(mocker, config: Config, reply: str | Exception):
    config.evaluation.answer_span = "extracted"
    chain = _SpanChain(reply)
    setup = mocker.patch.object(
        answer_match_evaluator, "setup_chain", return_value=chain
    )
    mocker.patch.object(answer_match_evaluator.Providers, "resolve")
    evaluator = AnswerMatchEvaluator(config)
    assert setup.call_args.args[1] == config.model_id(ModelRole.ANSWER_SPAN)
    return evaluator, chain


async def _ascore(evaluator, answer: str, gold: str, question: str = "Who?"):
    result = EvaluationResult(
        query_id="q", question=question, generated_answer=answer, ground_truth=gold
    )
    return await evaluator.aevaluate_single(
        EvaluationQuery(query_id="q", question=question), result, gold
    )


class TestExtractedSpan:
    @pytest.mark.parametrize("span", ["first_line", "extracted"])
    async def test_a_decline_scores_zero_without_a_span_call(
        self, mocker, config, span
    ) -> None:
        # The decline names the gold answer; before, answer_contains gave it 1.
        evaluator, chain = _extracting(mocker, config, "Depot")
        evaluator.answer_span = span
        result = EvaluationResult(
            query_id="q",
            question="Who?",
            generated_answer="The context names Depot but not who runs it.",
            ground_truth="Depot",
            abstained=True,
        )
        report = await evaluator.aevaluate_single(
            EvaluationQuery(query_id="q", question="Who?"), result, "Depot"
        )
        assert _values(report) == {
            "answer_contains": 0.0,
            "exact_match": 0.0,
            "token_f1": 0.0,
        }
        assert report.metadata["abstained"] is True
        assert chain.calls == []

    async def test_a_short_span_scores_like_the_papers(self, mocker, config) -> None:
        evaluator, chain = _extracting(mocker, config, '"Depot"\n')
        report = await _ascore(evaluator, _DEPOT_ANSWER, "Depot")
        assert _values(report)["exact_match"] == 1.0
        assert _values(report)["token_f1"] == 1.0
        assert report.metadata["scored_span"] == "Depot"
        assert chain.calls == [{"question": "Who?", "answer": _DEPOT_ANSWER}]

    async def test_spans_are_cached_per_question_and_answer(
        self, mocker, config
    ) -> None:
        evaluator, chain = _extracting(mocker, config, "Depot")
        await _ascore(evaluator, _DEPOT_ANSWER, "Depot")
        await _ascore(evaluator, _DEPOT_ANSWER, "Depot")
        assert len(chain.calls) == 1
        await _ascore(evaluator, _DEPOT_ANSWER, "Depot", question="Where?")
        assert len(chain.calls) == 2

    @pytest.mark.parametrize(
        "reply", [RuntimeError("throttled"), "", "Harbor"]  # Harbor: not in it
    )
    async def test_a_failed_extraction_falls_back_to_the_first_line(
        self, mocker, config, reply
    ) -> None:
        evaluator, _ = _extracting(mocker, config, reply)
        report = await _ascore(evaluator, _DEPOT_ANSWER, "Depot")
        assert _values(report)["token_f1"] == pytest.approx(1 / 3)
        assert report.metadata["scored_span"] == "Depot stores widgets for Vendor."
        assert report.metadata["answer_span_fallback"]

    async def test_no_answer_scores_an_empty_span(self, mocker, config) -> None:
        evaluator, _ = _extracting(mocker, config, "NO_ANSWER")
        report = await _ascore(evaluator, "The context does not say.", "Depot")
        assert _values(report)["token_f1"] == 0.0
        assert report.metadata["scored_span"] == ""
        assert "answer_span_fallback" not in report.metadata

    async def test_a_yes_no_span_need_not_be_verbatim(self, mocker, config) -> None:
        evaluator, _ = _extracting(mocker, config, "no")
        report = await _ascore(evaluator, "The clause does not apply.", "No")
        assert _values(report)["exact_match"] == 1.0

    async def test_no_reference_makes_no_model_call(self, mocker, config) -> None:
        evaluator, chain = _extracting(mocker, config, "Depot")
        report = await _ascore(evaluator, _DEPOT_ANSWER, "")
        assert report.metrics == [] and chain.calls == []

    def test_the_sync_path_refuses_a_model_call(self, mocker, config) -> None:
        evaluator, _ = _extracting(mocker, config, "Depot")
        with pytest.raises(EvaluationException, match="aevaluate_single"):
            evaluator.evaluate_single(
                EvaluationQuery(query_id="q", question="?"),
                EvaluationResult(
                    query_id="q", question="?", generated_answer="x", ground_truth=""
                ),
                "Depot",
            )

    def test_other_spans_build_no_model(self, mocker, config) -> None:
        setup = mocker.patch.object(answer_match_evaluator, "setup_chain")
        AnswerMatchEvaluator(config)
        setup.assert_not_called()


class TestHotpotQARule:
    @pytest.mark.parametrize(
        ("mode", "f1"), [("official_squad", 0.5), ("hotpotqa", 0.0)]
    )
    def test_yes_no_gets_no_partial_credit(
        self, config: Config, mode: str, f1: float
    ) -> None:
        config.evaluation.answer_normalization = mode
        report = _report(config, "No, it is", "no", {})
        assert _values(report)["token_f1"] == pytest.approx(f1)

    def test_equal_yes_no_still_scores(self, config: Config) -> None:
        config.evaluation.answer_normalization = "hotpotqa"
        assert _values(_report(config, "Yes.", "yes", {}))["token_f1"] == 1.0
        # Other golds keep partial credit, with the official normalization.
        report = _report(config, "Depot and Mill", "Depot", {})
        assert _values(report)["token_f1"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("question", "answer", "gold", "contains"),
    [
        # A number the question states is a condition of the absent thing.
        ("¿Cuántos proveedores hay en el lote 7?", "En el lote 7 no hay.", "0", 1.0),
        ("仕入先は従業員100人以上で何社?", "100人以上の仕入先はありません。", "0", 1.0),
        # One the answer adds after a comparative bounds the count.
        ("Quanti fornitori ha avuto Acme?", "Non ha mai avuto più di tre.", "3", 1.0),
        # No absence: the question's number is the count it confirms.
        ("Does Acme have 4 suppliers?", "Yes, 4.", "4", 1.0),
    ],
)
def test_a_number_item_reads_the_question(
    config: Config, question: str, answer: str, gold: str, contains: float
) -> None:
    report = _report(config, answer, gold, {"answer_type": "number"}, question)
    assert _values(report)["answer_contains"] == contains


def test_answer_contains_reads_through_a_possessive() -> None:
    assert answer_contains("Depot's crew stores the widgets.", "Depot") == 1.0
    assert answer_contains("Depots store the widgets.", "Depot") == 0.0


_SET = {
    "answer_type": "set",
    "answer_items": ["Acme", "Beta", "Gamma"],
    "answer_candidates": ["Acme", "Beta", "Delta", "Gamma", "Zeta"],
}


class TestSetAnswerBlock:
    """With candidates, a set answer is read from the block that answers
    (the extracted span, or the first line and the list right after it),
    so naming another candidate further down is no wrong pick."""

    @pytest.mark.parametrize(
        "answer",
        [
            "The top three are Acme, Beta and Gamma.\nDelta is fourth.",
            "The top three vendors are:\n\n1. Acme\n2. Beta\n3. Gamma\n\n"
            "Delta and Zeta follow with two buyers each.",
            "## Top vendors\n- Acme (5)\n- Beta (4)\n- Gamma (4)\n\nDelta has 3.",
            "상위 세 곳은 다음과 같습니다.\n\n- Acme\n- Beta\n- Gamma\n\n"
            "Delta는 네 번째입니다.",
            "上位三社は次のとおりです。\n・Acme\n・Beta\n・Gamma\nDeltaは四番目です。",
            "前三家供应商：\n1、Acme\n2、Beta\n3、Gamma\nDelta排第四。",
            "1、Acme\n2、Beta\n3、Gamma\nDelta排第四。",
        ],
    )
    def test_a_later_mention_is_no_pick(self, config: Config, answer: str) -> None:
        report = _report(config, answer, "Acme, Beta, Gamma", _SET)
        assert _values(report)["answer_contains"] == 1.0

    @pytest.mark.parametrize(
        "answer",
        [
            "The top three are Acme, Beta and Delta.\nGamma is fourth.",
            "The top three vendors are:\n1. Acme\n2. Delta\n3. Beta\n4. Gamma",
            "The vendors are Acme and Beta.\nGamma and Delta follow.",
        ],
    )
    def test_the_answering_block_decides(self, config: Config, answer: str) -> None:
        report = _report(config, answer, "Acme, Beta, Gamma", _SET)
        assert _values(report)["answer_contains"] == 0.0

    def test_without_candidates_the_whole_answer_counts(self, config) -> None:
        metadata = {k: v for k, v in _SET.items() if k != "answer_candidates"}
        answer = "Acme and Beta lead.\nGamma is third; Delta fourth."
        report = _report(config, answer, "Acme, Beta, Gamma", metadata)
        assert _values(report)["answer_contains"] == 1.0

    async def test_the_extracted_span_is_the_block(self, mocker, config) -> None:
        evaluator, _ = _extracting(mocker, config, "Acme, Beta and Gamma")
        answer = (
            "Based on the supply records, Acme, Beta and Gamma supply the most "
            "buyers, ahead of Delta."
        )
        result = EvaluationResult(
            query_id="q",
            question="Which three?",
            generated_answer=answer,
            ground_truth="Acme, Beta, Gamma",
            metadata=_SET,
        )
        report = await evaluator.aevaluate_single(
            EvaluationQuery(query_id="q", question="Which three?"),
            result,
            "Acme, Beta, Gamma",
        )
        assert _values(report)["answer_contains"] == 1.0
