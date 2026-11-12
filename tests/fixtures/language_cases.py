# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The regression corpus of language detection (a question's and a corpus
sample's), number reading, answer containment, router answers and name
folding: every case the query reviews rv9 to rv14 (and the evaluation and
surfaces reviews since) probed or reported, with the answer they stated as
right.

A case the code answers otherwise is a documented trade-off: it carries
the answer the code gives and why (``Tradeoff``), so a change that moves
it fails like a regression does and says which way it moved. Names are
synthetic (the reviews' customer-like ones replaced). Measured by
``tests/unit/test_language_cases.py``; ``corpus_score`` gives the pass
rate against the stated answers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

__all__ = [
    "CONTAINS",
    "CORPORA",
    "COVERS",
    "DETECTION",
    "DOCUMENT_BREAK",
    "NUMBERS",
    "ROUTER",
    "Case",
    "Tradeoff",
    "corpus_documents",
    "corpus_score",
]


@dataclass(frozen=True)
class Tradeoff:
    """What the code answers instead of a case's stated answer, and why."""

    answer: object
    why: str


@dataclass(frozen=True)
class Case:
    """One probe: ``text`` and the ``expected`` answer the review stated
    (for a number, the gold the answer matches, ``None`` for one that
    states no number); ``context`` is the index language a detection falls
    back to, the answer language a number is read in, or the mention a
    stored name covers; ``question`` is the question a number answers
    (``None``: read without one); ``gold`` is the gold answer a
    containment case scores ``text`` against."""

    text: str
    expected: object
    context: str | None
    source: str
    tradeoff: Tradeoff | None = None
    question: str | None = None
    gold: str | None = None

    @property
    def answer(self) -> object:
        """What the code is pinned to answer."""
        return self.tradeoff.answer if self.tradeoff is not None else self.expected


def corpus_score(
    cases: Iterable[Case], read: Callable[[Case], object]
) -> tuple[int, int]:
    """How many of ``cases`` ``read`` answers as stated, of how many."""
    cases = list(cases)
    return sum(read(case) == case.expected for case in cases), len(cases)


PHRASE = (
    "A lowercase English phrase holding a function word, in a short Korean, "
    "Japanese or Chinese question: one function word against one run, which "
    "no rule tells from an English question quoting a name (rv12 query 1: "
    "debatable)."
)
MANY_NAMES = (
    "More runs of another script than function words: an English question "
    "listing three names, or a spaced one of three words, reads as their "
    "script (old code alike; rv12 query low)."
)
NO_WORD = (
    "No function word: the question's other script decides, not the index "
    "language (old code alike; rv12 query low)."
)
SHOUTED = (
    'Fewer than three all-caps words are acronyms ("ES index", "AI ethics"), '
    "so a two-word shouted question holds no function word: a shouted text "
    "is read lowercased from three words (rv13 query 1)."
)
NO_PROFILE = (
    "A language without a profile (tr, pl, nl, sv, id, el, uk, fa) has no "
    "words of its own to read: it reads as its script's profiled language "
    "or the index language."
)
TITLE_RUN = (
    "A capitalised first word that is no function word opens a name run "
    'with the capitalised words after it ("Has Acme Corp" is fixed by '
    '"has", "¿Por qué Orion Labs", "List The Vendors Of" are not): the '
    "run's words count for nothing."
)
QUOTING = (
    "An English question quoting a name reads as English: the rv11 M5 "
    'matrix pins "Who is the CEO of {n}?" so for every script (the probe\'s '
    "expectation is debatable)."
)
SHARED_WORD = (
    'Italian and Portuguese both write "quanto", but only Italian\'s list '
    "holds it: listed for both it would tie them, and an Italian question "
    'with no other word ("Quanto vale l\'azienda?" on an English index) would '
    "read as the index language; the Portuguese words left are an article "
    '("o") and content words (rv13 query 4, rv14 query 4).'
)
SHARED_TIE = (
    'Its one function word ("como") is Spanish and Portuguese alike: tied, '
    "they leave the index language, a third one too, and so does a tie "
    "beside a word no list holds, a verb or an article only one of them "
    'writes ("Como funciona?" on an English or Korean index, "Como faço '
    'para cancelar?", "Como funciona o contrato?" on a Spanish one; rv14 '
    "query 4, rv15 query)."
)
NO_LISTED_WORD = (
    'No word of the text is in its language\'s list ("Comparar" is a verb, '
    '"tại sao" and "bị" are not in the Vietnamese one): the index language '
    "reads (rv14 query)."
)
URL_LETTERS = (
    "A URL's letters count in the script totals, so a Cyrillic domain "
    "outweighs one Korean word (pre-existing; rv14 query low)."
)
ARTICLES_ONLY = (
    'The one function word is Portuguese ("tem"); the Italian words are '
    'articles ("il", "un"), which decide only between languages tied on '
    "function words (rv13 query 4)."
)
ONE_SCRIPT_NAMES = (
    "In a text of one script its capitalised words are not set apart as "
    'names (German writes its nouns so), so the "of" of a quoted English '
    "name ties the Italian words and the English index wins."
)
QUOTED_SENTENCE = (
    "A quoted German sentence holds more function words than the English "
    "words around it."
)
HAN_BARE = (
    "A Han numeral states a number only before a counter (pre-existing "
    'design: "一万二千" alone may be a code or a year).'
)
SCALE_DIGITS = (
    "Digits before a scale word, a decimal comma or a decimal before a "
    "counter read without the scale (pre-existing; rv11 query low, outside "
    "this change)."
)
NO_QUESTION = (
    "A gold of 0 matches an absence with no number but those the question "
    "writes: read without its question, the absent thing's own number (\"lot "
    '7", "more than 100 employees", "10:30") is a number of its own.'
)
OTHER_NUMBER = (
    "A gold of 0 matches an absence with no other number: one the answer "
    'adds about something else ("3 internal plants") is read as a count it '
    "might state; the LLM judges read that nuance."
)
HOMONYM = (
    '"Семью" is also "семья" (family) in two of its cases, so it is left '
    'out of the numerals: "с семью поставщиками" states no number (rv14 '
    "query 6)."
)
WORD_GAPS = (
    'A word the profiles do not hold ("dozen", the article "un" for one, '
    '"neuf" read as nine, "No." as a label, "ou"/"o" alternatives; '
    "pre-existing, outside this change)."
)


DETECTION: tuple[Case, ...] = (
    Case("AI 모델 추천", "ko", "ko", "rv12 query 1"),
    Case("AI란?", "ko", "ko", "rv12 query 1"),
    Case("AI 에이전트란?", "ko", "ko", "rv12 query 1"),
    Case("AI 윤리 가이드", "ko", "ko", "rv12 query 1"),
    Case("AI 도입 사례는?", "ko", "ko", "rv12 query 1"),
    Case("생성형 AI 활용 사례", "ko", "ko", "rv12 query 1"),
    Case("AIとは", "ja", "ja", "rv12 query 1"),
    Case("AIとは何ですか", "ja", "ja", "rv12 query 1"),
    Case("AIの導入事例", "ja", "ja", "rv12 query 1"),
    Case("AI是什么", "zh", "zh", "rv12 query 1"),
    Case("AI的应用场景", "zh", "zh", "rv12 query 1"),
    Case("OS 업데이트 방법", "ko", "ko", "rv12 query 1"),
    Case("OS 설치", "ko", "ko", "rv12 query 1"),
    Case("ES 인덱스 설정", "ko", "ko", "rv12 query 1"),
    Case("ES 클러스터 상태", "ko", "ko", "rv12 query 1"),
    Case("ES6 문법 정리", "ko", "ko", "rv12 query 1"),
    Case("PER 계산법", "ko", "ko", "rv12 query 1"),
    Case("PER와 PBR 비교", "ko", "ko", "rv12 query 1"),
    Case("EST 시간대", "ko", "ko", "rv12 query 1"),
    Case("COM 포트 설정", "ko", "ko", "rv12 query 1"),
    Case("DEL 키 사용법", "ko", "ko", "rv12 query 1"),
    Case("to-do 앱 추천", "ko", "ko", "rv12 query 1"),
    Case("how-to 가이드", "ko", "ko", "rv12 query 1"),
    Case("what-if 분석", "ko", "ko", "rv12 query 1"),
    Case("proof of concept 일정", "ko", "ko", "rv12 query 1", Tradeoff("en", PHRASE)),
    Case("out of memory 에러 해결", "ko", "ko", "rv12 query 1"),
    Case("state of the art 모델", "ko", "ko", "rv12 query 1", Tradeoff("en", PHRASE)),
    Case("this 키워드", "ko", "ko", "rv12 query 1", Tradeoff("en", PHRASE)),
    Case("with 문 사용법", "ko", "ko", "rv12 query 1"),
    Case("is 연산자", "ko", "ko", "rv12 query 1", Tradeoff("en", PHRASE)),
    Case("OSのアップデート", "ja", "ja", "rv12 query 1"),
    Case("ESクラスタの設定", "ja", "ja", "rv12 query 1"),
    Case("ES集群配置", "zh", "zh", "rv12 query 1"),
    Case("OS升级", "zh", "zh", "rv12 query 1"),
    Case("to-doアプリ", "ja", "ja", "rv12 query 1"),
    Case("DO 모델 성능", "ko", "ko", "rv12 query 1"),
    Case("AI модели", "ru", "ru", "rv12 query 1"),
    Case("Что такое AI?", "ru", "ru", "rv12 query 1"),
    Case("ES кластер", "ru", "ru", "rv12 query 1"),
    Case("ИИ и AI", "ru", "ru", "rv12 query 1"),
    Case("UNO 보드 사용법", "ko", "ko", "rv12 query 1"),
    Case("CHO 회사 실적", "ko", "ko", "rv12 query 1"),
    Case("QUI 시스템", "ko", "ko", "rv12 query 1"),
    Case("AWS AI 서비스 비교", "ko", "ko", "rv12 query 1"),
    Case("AI와 ML 차이", "ko", "ko", "rv12 query 1"),
    Case("AIとMLの違い", "ja", "ja", "rv12 query 1"),
    Case("AI和ML的区别", "zh", "zh", "rv12 query 1"),
    Case("AWS의 S3와 EC2 비교", "ko", "ko", "rv12 probe"),
    Case("A와 B 차이", "ko", "ko", "rv12 probe"),
    Case("A와 B의 차이는?", "ko", "ko", "rv12 probe"),
    Case("AとBの違い", "ja", "ja", "rv12 probe"),
    Case("AとBの違いは？", "ja", "ja", "rv12 probe"),
    Case("A和B的区别", "zh", "zh", "rv12 probe"),
    Case("A和B的区别是什么", "zh", "zh", "rv12 probe"),
    Case("Plan A 또는 Plan B?", "ko", "ko", "rv12 probe"),
    Case("a 값은?", "ko", "ko", "rv12 probe"),
    Case("x와 y의 상관관계", "ko", "ko", "rv12 probe"),
    Case("S3 bucket in us-east-1 설정 방법", "ko", "ko", "rv12 probe"),
    Case("pip install 에러 해결", "ko", "ko", "rv12 probe"),
    Case("what is RAG 설명", "ko", "ko", "rv12 probe", Tradeoff("en", PHRASE)),
    Case("on-premise 이전 비용", "ko", "ko", "rv12 probe"),
    Case("log in 안 됨", "ko", "ko", "rv12 probe"),
    Case("S3 bucket in us-east-1の設定方法", "ja", "ja", "rv12 probe"),
    Case("on-premiseからの移行コスト", "ja", "ja", "rv12 probe"),
    Case("Amazon Bedrockとは何ですか？", "ja", "ja", "rv12 probe"),
    Case("Amazon Bedrock is何ですか？", "ja", "ja", "rv12 probe"),
    Case("on-premise迁移的成本是多少", "zh", "zh", "rv12 probe"),
    Case("Amazon S3 和 EC2 的区别是什么", "zh", "zh", "rv12 probe"),
    Case("log in 失败怎么办", "zh", "zh", "rv12 probe"),
    Case("e-mail 발송 실패", "ko", "ko", "rv12 probe"),
    Case("to-do 리스트 앱", "ko", "ko", "rv12 probe"),
    Case("add-on 설치", "ko", "ko", "rv12 probe"),
    Case("set up 가이드", "ko", "ko", "rv12 probe"),
    Case("sign in 오류", "ko", "ko", "rv12 probe"),
    Case("Как настроить S3 bucket in us-east-1?", "ru", "ru", "rv12 probe"),
    Case("Сравни A и B", "ru", "ru", "rv12 probe"),
    Case("Что такое AWS и EC2?", "ru", "ru", "rv12 probe"),
    Case("AWS или Azure", "ru", "ru", "rv12 probe"),
    Case("log in не работает", "ru", "ru", "rv12 probe"),
    Case("Как сделать log in", "ru", "ru", "rv12 probe"),
    Case("Is 아크미 a vendor?", "ko", "ko", "rv12 probe", Tradeoff("en", QUOTING)),
    Case("Who is the CEO of 한빛전자?", "en", "en", "rv12 probe"),
    Case("Who is 한빛전자?", "en", "en", "rv12 probe"),
    Case("Compare 한빛전자 and 아크미", "en", "en", "rv12 probe"),
    Case(
        "Compare 한빛전자, 아크미 and 오리온",
        "en",
        "en",
        "rv12 probe",
        Tradeoff("ko", MANY_NAMES),
    ),
    Case("Is 株式会社アクメ a supplier of 阿克米集团?", "en", "en", "rv12 probe"),
    Case("What does 株式会社 アクメ make?", "en", "en", "rv12 probe"),
    Case("Who founded 한빛 전자?", "en", "en", "rv12 probe"),
    Case(
        "Who founded 한빛 전자 그룹?",
        "en",
        "en",
        "rv12 probe",
        Tradeoff("ko", MANY_NAMES),
    ),
    Case(
        "What is the relation between Иван Петров and Ольга Смирнова?",
        "en",
        "en",
        "rv12 probe",
    ),
    Case("Who is Иван Петров?", "en", "en", "rv12 probe"),
    Case("Who are Иван, Ольга and Пётр?", "en", "en", "rv12 probe"),
    Case("CEO 한빛전자?", "en", "en", "rv12 probe", Tradeoff("ko", NO_WORD)),
    Case("한빛전자 CEO?", "en", "en", "rv12 probe", Tradeoff("ko", NO_WORD)),
    Case("한빛전자 revenue 2023", "en", "en", "rv12 probe", Tradeoff("ko", NO_WORD)),
    Case("SUPPLIERS OF 한빛전자?", "en", "en", "rv12 probe", Tradeoff("ko", SHOUTED)),
    Case("WHO IS THE CEO OF 한빛전자?", "en", "en", "rv12 probe"),
    Case("Ivan Petrov (Иван Петров) — who is he?", "en", "en", "rv12 probe"),
    Case("The Who played where?", "en", "en", "rv12 probe"),
    Case("Who wrote The Lord of the Rings?", "en", "en", "rv12 probe"),
    Case(
        "Who is 한빛전자's CEO, and what is The Walt Disney Company?",
        "en",
        "en",
        "rv12 probe",
    ),
    Case("Wie viele Lieferanten hat 한빛전자?", "de", "de", "rv12 probe"),
    Case("Welche Verträge hat Die Bahn mit 한빛전자?", "de", "de", "rv12 probe"),
    Case("Wer ist der Geschäftsführer von 한빛전자?", "de", "de", "rv12 probe"),
    Case("Wer Ist Der Geschäftsführer Von 한빛전자?", "de", "de", "rv12 probe"),
    Case("Qui dirige 한빛전자 ?", "fr", "fr", "rv12 probe"),
    Case("Quels Sont Les Fournisseurs De 한빛전자 ?", "fr", "fr", "rv12 probe"),
    Case("¿Quién dirige 한빛전자?", "es", "es", "rv12 probe"),
    Case("¿Cuántos proveedores tiene 한빛전자?", "es", "es", "rv12 probe"),
    Case("Quem dirige a 한빛전자?", "pt", "pt", "rv12 probe"),
    Case("Chi dirige 한빛전자?", "it", "it", "rv12 probe"),
    Case("Ai là CEO của 한빛전자?", "vi", "vi", "rv12 probe"),
    Case("Giám đốc của 株式会社アクメ là ai?", "vi", "vi", "rv12 probe"),
    Case(
        "한빛전자 şirketinin CEO'su kim?",
        "tr",
        "en",
        "rv12 probe",
        Tradeoff("ko", NO_PROFILE),
    ),
    Case(
        "Kto jest prezesem 한빛전자?",
        "pl",
        "en",
        "rv12 probe",
        Tradeoff("ko", NO_PROFILE),
    ),
    Case(
        "Wie is de CEO van 한빛전자?",
        "nl",
        "en",
        "rv12 probe",
        Tradeoff("en", NO_PROFILE),
    ),
    Case(
        "Vem är VD för 한빛전자?", "sv", "en", "rv12 probe", Tradeoff("ko", NO_PROFILE)
    ),
    Case("Siapa CEO 한빛전자?", "id", "en", "rv12 probe", Tradeoff("ko", NO_PROFILE)),
    Case(
        "Is 한빛전자 bigger than 아크미, 오리온 and 미래?",
        "en",
        "en",
        "rv12 probe",
        Tradeoff("ko", MANY_NAMES),
    ),
    Case("Who are the CEOs of 한빛전자, 아크미 and 오리온?", "en", "en", "rv12 probe"),
    Case("한빛전자의 CEO는 who인가요?", "ko", "ko", "rv12 probe"),
    Case("이 문서에서 the vendor는 누구야?", "ko", "ko", "rv12 probe"),
    Case("이 계약의 terms and conditions 요약", "ko", "ko", "rv12 probe"),
    Case("terms and conditions 요약", "ko", "ko", "rv12 probe", Tradeoff("en", PHRASE)),
    Case(
        "terms and conditionsの要約", "ja", "ja", "rv12 probe", Tradeoff("en", PHRASE)
    ),
    Case(
        "terms and conditions的摘要", "zh", "zh", "rv12 probe", Tradeoff("en", PHRASE)
    ),
    Case("the vendor 누구?", "ko", "ko", "rv12 probe", Tradeoff("en", PHRASE)),
    Case("Q&A 정리", "ko", "ko", "rv12 probe"),
    Case("R&D 투자 현황", "ko", "ko", "rv12 probe"),
    Case("Что такое terms and conditions?", "ru", "ru", "rv12 probe"),
    Case(
        "Who is the CEO of 株式会社アクメ",
        "ja",
        "ja",
        "rv12 probe",
        Tradeoff("en", QUOTING),
    ),
    Case("ما هو AWS S3 و EC2؟", "ar", "ar", "rv12 probe"),
    Case(
        "Τι είναι το AWS S3 και το EC2;",
        "el",
        "en",
        "rv12 probe",
        Tradeoff("en", NO_PROFILE),
    ),
    Case("מה זה S3 ו-EC2 של AWS?", "he", "he", "rv12 probe"),
    Case("AWS S3 और EC2 में क्या अंतर है?", "hi", "hi", "rv12 probe"),
    Case("AWS S3 และ EC2 ต่างกันอย่างไร", "th", "th", "rv12 probe"),
    Case("Що таке AWS S3 і EC2?", "uk", "en", "rv12 probe", Tradeoff("ru", NO_PROFILE)),
    Case("Что такое S3 in AWS?", "ru", "ru", "rv12 probe"),
    Case("В чём разница a и b?", "ru", "ru", "rv12 probe"),
    Case("Did Orion Labs sign a contract with 한빛물산?", "en", "en", "rv11 probe"),
    Case("Does Acme Robotics supply 株式会社アクメ?", "en", "en", "rv11 probe"),
    Case("Does Acme Robotics supply 아크미?", "en", "en", "rv11 probe"),
    Case("Is Acme Robotics owned by 阿克米集团?", "en", "en", "rv11 probe"),
    Case("Who Is The CEO Of 한빛물산?", "en", "en", "rv11 probe"),
    Case("How Many Vendors Does 株式会社アクメ Have?", "en", "en", "rv11 probe"),
    Case("What Is The Revenue Of Иван Петров Holdings?", "en", "en", "rv11 probe"),
    Case("When did Acme acquire Ωmega Corp?", "en", "en", "rv11 probe"),
    Case("What is the α coefficient in the Acme model?", "en", "en", "rv11 probe"),
    Case("Where is the 東京 office of Orion Labs?", "en", "en", "rv11 probe"),
    Case("Where Is The 東京 Office?", "en", "en", "rv11 probe"),
    Case("Why did Orion Labs drop 아크미?", "en", "en", "rv11 probe"),
    Case(
        "List The Vendors Of 아크미",
        "en",
        "en",
        "rv11 probe",
        Tradeoff("ko", TITLE_RUN),
    ),
    Case(
        "Explain the relationship between Orion Labs and 株式会社アクメ",
        "en",
        "en",
        "rv11 probe",
    ),
    Case(
        "Summarize 株式会社アクメ's contracts",
        "en",
        "en",
        "rv11 probe",
        Tradeoff("ja", NO_WORD),
    ),
    Case("Has Orion Labs Inc. ever worked with 한빛전자?", "en", "en", "rv11 probe"),
    Case("Can You Compare 한빛전자 And 아크미?", "en", "en", "rv11 probe"),
    Case("iPhone and iPad 비교", "ko", "en", "rv11 probe"),
    Case("eBay and Amazon 관계는?", "ko", "en", "rv11 probe"),
    Case("Compare Hanbit and Orion 전자", "ko", "en", "rv11 probe"),
    Case("Bank of America의 CEO", "ko", "en", "rv11 probe"),
    Case("the Acme 계약 조건은?", "ko", "en", "rv11 probe"),
    Case("Bank of America and Wells Fargoの関係は？", "ja", "en", "rv11 probe"),
    Case("Bank of America and Wells Fargo有什么关系？", "zh", "en", "rv11 probe"),
    Case(
        "Bank of America和the Walt Disney Company是什么关系？", "zh", "en", "rv11 probe"
    ),
    Case("University of Tokyo와 University of Kyoto 비교", "ko", "en", "rv11 probe"),
    Case("Какова связь Bank of America и Wells Fargo?", "ru", "en", "rv11 probe"),
    Case("ما علاقة Bank of America و Wells Fargo؟", "ar", "en", "rv11 probe"),
    Case(
        "Bank of America และ Wells Fargo เกี่ยวข้องกันอย่างไร", "th", "en", "rv11 probe"
    ),
    Case(
        "Ποια είναι η σχέση της Bank of America με την Wells Fargo;",
        "el",
        "en",
        "rv11 probe",
        Tradeoff("en", NO_PROFILE),
    ),
    Case("מה הקשר בין Bank of America ל-Wells Fargo?", "he", "en", "rv11 probe"),
    Case("Bank of America और Wells Fargo का संबंध", "hi", "en", "rv11 probe"),
    Case(
        "Який зв'язок між Bank of America та Wells Fargo?",
        "uk",
        "en",
        "rv11 probe",
        Tradeoff("ru", NO_PROFILE),
    ),
    Case(
        "رابطه Bank of America و Wells Fargo چیست؟",
        "fa",
        "en",
        "rv11 probe",
        Tradeoff("ar", NO_PROFILE),
    ),
    Case(
        "Mối quan hệ giữa Bank of America và Wells Fargo là gì?",
        "vi",
        "en",
        "rv11 probe",
    ),
    Case(
        "Quelle est la relation entre Bank of America et Wells Fargo ?",
        "fr",
        "en",
        "rv11 probe",
    ),
    Case(
        "Wie hängen Bank of America und Wells Fargo zusammen?", "de", "en", "rv11 probe"
    ),
    Case(
        "¿Qué relación hay entre Bank of America y Wells Fargo?",
        "es",
        "en",
        "rv11 probe",
    ),
    Case(
        "Qual é a relação entre Bank of America e Wells Fargo?",
        "pt",
        "en",
        "rv11 probe",
    ),
    Case(
        "Qual è il rapporto tra Bank of America e Wells Fargo?",
        "it",
        "en",
        "rv11 probe",
        Tradeoff("en", ONE_SCRIPT_NAMES),
    ),
    Case("¿Cuántos empleados tiene la empresa?", "es", "en", "rv11 probe"),
    Case("Quanti dipendenti ha l'azienda?", "it", "en", "rv11 probe"),
    Case("Quanto costa il contratto?", "it", "en", "rv11 probe"),
    Case(
        "Quanto custa o contrato?",
        "pt",
        "en",
        "rv11 probe",
        Tradeoff("it", SHARED_WORD),
    ),
    Case("Chi ha firmato il contratto con Acme?", "it", "en", "rv11 probe"),
    Case(
        "Il fornitore tem un contratto?",
        "it",
        "en",
        "rv11 probe",
        Tradeoff("pt", ARTICLES_ONLY),
    ),
    Case("Hay proveedores en Madrid?", "es", "en", "rv11 probe"),
    Case("Há fornecedores em Lisboa?", "pt", "en", "rv11 probe"),
    Case("Ci sono fornitori a Milano?", "it", "en", "rv11 probe"),
    Case("Quanti contratti ha Acme con Orion?", "it", "en", "rv11 probe"),
    Case("Come si chiama il direttore?", "it", "en", "rv11 probe"),
    Case("Como se llama el director?", "es", "en", "rv11 probe"),
    Case("Como se chama o diretor?", "pt", "en", "rv11 probe"),
    Case("Cosa fa Acme?", "it", "en", "rv11 probe"),
    Case("Dove si trova la sede?", "it", "en", "rv11 probe"),
    Case("Il contratto tiene conto delle penali?", "it", "en", "rv11 probe"),
    Case("Why did Orion Labs drop 한빛전자?", "en", "en", "rv11 query 1/5"),
    Case("Why did Orion Labs drop 한빛전자?", "en", "ko", "rv11 query 1/5"),
    Case("When did Orion Labs acquire 한빛전자?", "en", "en", "rv11 query 1/5"),
    Case("When did Orion Labs acquire 한빛전자?", "en", "ko", "rv11 query 1/5"),
    Case("Does Orion Labs supply 한빛전자?", "en", "en", "rv11 query 1/5"),
    Case("Does Orion Labs supply 한빛전자?", "en", "ko", "rv11 query 1/5"),
    Case("Is Orion Labs a vendor of 한빛전자?", "en", "en", "rv11 query 1/5"),
    Case("Is Orion Labs a vendor of 한빛전자?", "en", "ko", "rv11 query 1/5"),
    Case("How much did Orion Labs pay 한빛전자?", "en", "en", "rv11 query 1/5"),
    Case("How much did Orion Labs pay 한빛전자?", "en", "ko", "rv11 query 1/5"),
    Case("What did Acme Corp sign with 한빛전자?", "en", "en", "rv11 query 1/5"),
    Case("What did Acme Corp sign with 한빛전자?", "en", "ko", "rv11 query 1/5"),
    Case(
        "Which products does Acme Corp buy from 한빛전자?", "en", "en", "rv11 query 1/5"
    ),
    Case(
        "Which products does Acme Corp buy from 한빛전자?", "en", "ko", "rv11 query 1/5"
    ),
    Case("Who at Acme Corp negotiated with 한빛전자?", "en", "en", "rv11 query 1/5"),
    Case("Who at Acme Corp negotiated with 한빛전자?", "en", "ko", "rv11 query 1/5"),
    Case(
        "Did Acme Corp terminate the 한빛전자 contract?", "en", "en", "rv11 query 1/5"
    ),
    Case(
        "Did Acme Corp terminate the 한빛전자 contract?", "en", "ko", "rv11 query 1/5"
    ),
    Case(
        "What is the relationship between Acme Corp and 한빛전자?",
        "en",
        "en",
        "rv11 query 1/5",
    ),
    Case(
        "What is the relationship between Acme Corp and 한빛전자?",
        "en",
        "ko",
        "rv11 query 1/5",
    ),
    Case("How is 한빛전자 related to Acme Corp?", "en", "en", "rv11 query 1/5"),
    Case("How is 한빛전자 related to Acme Corp?", "en", "ko", "rv11 query 1/5"),
    Case("Where does 한빛전자 ship to?", "en", "en", "rv11 query 1/5"),
    Case("Where does 한빛전자 ship to?", "en", "ko", "rv11 query 1/5"),
    Case("Who is the CEO of 한빛전자?", "en", "ko", "rv11 query 1/5"),
    Case("Has Acme Corp ever audited 한빛전자?", "en", "en", "rv11 query 1/5"),
    Case("Has Acme Corp ever audited 한빛전자?", "en", "ko", "rv11 query 1/5"),
    Case("Compare Acme Corp with 한빛전자", "en", "en", "rv11 query 1/5"),
    Case("Compare Acme Corp with 한빛전자", "en", "ko", "rv11 query 1/5"),
    Case("Why did Orion Labs drop 株式会社アクメ?", "en", "en", "rv11 query 1/5"),
    Case("Why did Orion Labs drop 株式会社アクメ?", "en", "ja", "rv11 query 1/5"),
    Case("When did Orion Labs acquire 株式会社アクメ?", "en", "en", "rv11 query 1/5"),
    Case("When did Orion Labs acquire 株式会社アクメ?", "en", "ja", "rv11 query 1/5"),
    Case("Does Orion Labs supply 株式会社アクメ?", "en", "en", "rv11 query 1/5"),
    Case("Does Orion Labs supply 株式会社アクメ?", "en", "ja", "rv11 query 1/5"),
    Case("Is Orion Labs a vendor of 株式会社アクメ?", "en", "en", "rv11 query 1/5"),
    Case("Is Orion Labs a vendor of 株式会社アクメ?", "en", "ja", "rv11 query 1/5"),
    Case("How much did Orion Labs pay 株式会社アクメ?", "en", "en", "rv11 query 1/5"),
    Case("How much did Orion Labs pay 株式会社アクメ?", "en", "ja", "rv11 query 1/5"),
    Case("What did Acme Corp sign with 株式会社アクメ?", "en", "en", "rv11 query 1/5"),
    Case("What did Acme Corp sign with 株式会社アクメ?", "en", "ja", "rv11 query 1/5"),
    Case(
        "Which products does Acme Corp buy from 株式会社アクメ?",
        "en",
        "en",
        "rv11 query 1/5",
    ),
    Case(
        "Which products does Acme Corp buy from 株式会社アクメ?",
        "en",
        "ja",
        "rv11 query 1/5",
    ),
    Case(
        "Who at Acme Corp negotiated with 株式会社アクメ?", "en", "en", "rv11 query 1/5"
    ),
    Case(
        "Who at Acme Corp negotiated with 株式会社アクメ?", "en", "ja", "rv11 query 1/5"
    ),
    Case(
        "Did Acme Corp terminate the 株式会社アクメ contract?",
        "en",
        "en",
        "rv11 query 1/5",
    ),
    Case(
        "Did Acme Corp terminate the 株式会社アクメ contract?",
        "en",
        "ja",
        "rv11 query 1/5",
    ),
    Case(
        "What is the relationship between Acme Corp and 株式会社アクメ?",
        "en",
        "en",
        "rv11 query 1/5",
    ),
    Case(
        "What is the relationship between Acme Corp and 株式会社アクメ?",
        "en",
        "ja",
        "rv11 query 1/5",
    ),
    Case("How is 株式会社アクメ related to Acme Corp?", "en", "en", "rv11 query 1/5"),
    Case("How is 株式会社アクメ related to Acme Corp?", "en", "ja", "rv11 query 1/5"),
    Case("Where does 株式会社アクメ ship to?", "en", "en", "rv11 query 1/5"),
    Case("Where does 株式会社アクメ ship to?", "en", "ja", "rv11 query 1/5"),
    Case("Who is the CEO of 株式会社アクメ?", "en", "en", "rv11 query 1/5"),
    Case("Who is the CEO of 株式会社アクメ?", "en", "ja", "rv11 query 1/5"),
    Case("Has Acme Corp ever audited 株式会社アクメ?", "en", "en", "rv11 query 1/5"),
    Case("Has Acme Corp ever audited 株式会社アクメ?", "en", "ja", "rv11 query 1/5"),
    Case("Compare Acme Corp with 株式会社アクメ", "en", "en", "rv11 query 1/5"),
    Case("Compare Acme Corp with 株式会社アクメ", "en", "ja", "rv11 query 1/5"),
    Case("Why did Orion Labs drop 阿克米集团?", "en", "en", "rv11 query 1/5"),
    Case("Why did Orion Labs drop 阿克米集团?", "en", "zh", "rv11 query 1/5"),
    Case("When did Orion Labs acquire 阿克米集团?", "en", "en", "rv11 query 1/5"),
    Case("When did Orion Labs acquire 阿克米集团?", "en", "zh", "rv11 query 1/5"),
    Case("Does Orion Labs supply 阿克米集团?", "en", "en", "rv11 query 1/5"),
    Case("Does Orion Labs supply 阿克米集团?", "en", "zh", "rv11 query 1/5"),
    Case("Is Orion Labs a vendor of 阿克米集团?", "en", "en", "rv11 query 1/5"),
    Case("Is Orion Labs a vendor of 阿克米集团?", "en", "zh", "rv11 query 1/5"),
    Case("How much did Orion Labs pay 阿克米集团?", "en", "en", "rv11 query 1/5"),
    Case("How much did Orion Labs pay 阿克米集团?", "en", "zh", "rv11 query 1/5"),
    Case("What did Acme Corp sign with 阿克米集团?", "en", "en", "rv11 query 1/5"),
    Case("What did Acme Corp sign with 阿克米集团?", "en", "zh", "rv11 query 1/5"),
    Case(
        "Which products does Acme Corp buy from 阿克米集团?",
        "en",
        "en",
        "rv11 query 1/5",
    ),
    Case(
        "Which products does Acme Corp buy from 阿克米集团?",
        "en",
        "zh",
        "rv11 query 1/5",
    ),
    Case("Who at Acme Corp negotiated with 阿克米集团?", "en", "en", "rv11 query 1/5"),
    Case("Who at Acme Corp negotiated with 阿克米集团?", "en", "zh", "rv11 query 1/5"),
    Case(
        "Did Acme Corp terminate the 阿克米集团 contract?", "en", "en", "rv11 query 1/5"
    ),
    Case(
        "Did Acme Corp terminate the 阿克米集团 contract?", "en", "zh", "rv11 query 1/5"
    ),
    Case(
        "What is the relationship between Acme Corp and 阿克米集团?",
        "en",
        "en",
        "rv11 query 1/5",
    ),
    Case(
        "What is the relationship between Acme Corp and 阿克米集团?",
        "en",
        "zh",
        "rv11 query 1/5",
    ),
    Case("How is 阿克米集团 related to Acme Corp?", "en", "en", "rv11 query 1/5"),
    Case("How is 阿克米集团 related to Acme Corp?", "en", "zh", "rv11 query 1/5"),
    Case("Where does 阿克米集团 ship to?", "en", "en", "rv11 query 1/5"),
    Case("Where does 阿克米集团 ship to?", "en", "zh", "rv11 query 1/5"),
    Case("Who is the CEO of 阿克米集团?", "en", "en", "rv11 query 1/5"),
    Case("Who is the CEO of 阿克米集团?", "en", "zh", "rv11 query 1/5"),
    Case("Has Acme Corp ever audited 阿克米集团?", "en", "en", "rv11 query 1/5"),
    Case("Has Acme Corp ever audited 阿克米集团?", "en", "zh", "rv11 query 1/5"),
    Case("Compare Acme Corp with 阿克米集团", "en", "en", "rv11 query 1/5"),
    Case("Compare Acme Corp with 阿克米集团", "en", "zh", "rv11 query 1/5"),
    Case("Why did Orion Labs drop Иван Петров?", "en", "en", "rv11 query 1/5"),
    Case("Why did Orion Labs drop Иван Петров?", "en", "ru", "rv11 query 1/5"),
    Case("When did Orion Labs acquire Иван Петров?", "en", "en", "rv11 query 1/5"),
    Case("When did Orion Labs acquire Иван Петров?", "en", "ru", "rv11 query 1/5"),
    Case("Does Orion Labs supply Иван Петров?", "en", "en", "rv11 query 1/5"),
    Case("Does Orion Labs supply Иван Петров?", "en", "ru", "rv11 query 1/5"),
    Case("Is Orion Labs a vendor of Иван Петров?", "en", "en", "rv11 query 1/5"),
    Case("Is Orion Labs a vendor of Иван Петров?", "en", "ru", "rv11 query 1/5"),
    Case("How much did Orion Labs pay Иван Петров?", "en", "en", "rv11 query 1/5"),
    Case("How much did Orion Labs pay Иван Петров?", "en", "ru", "rv11 query 1/5"),
    Case("What did Acme Corp sign with Иван Петров?", "en", "en", "rv11 query 1/5"),
    Case("What did Acme Corp sign with Иван Петров?", "en", "ru", "rv11 query 1/5"),
    Case(
        "Which products does Acme Corp buy from Иван Петров?",
        "en",
        "en",
        "rv11 query 1/5",
    ),
    Case(
        "Which products does Acme Corp buy from Иван Петров?",
        "en",
        "ru",
        "rv11 query 1/5",
    ),
    Case("Who at Acme Corp negotiated with Иван Петров?", "en", "en", "rv11 query 1/5"),
    Case("Who at Acme Corp negotiated with Иван Петров?", "en", "ru", "rv11 query 1/5"),
    Case(
        "Did Acme Corp terminate the Иван Петров contract?",
        "en",
        "en",
        "rv11 query 1/5",
    ),
    Case(
        "Did Acme Corp terminate the Иван Петров contract?",
        "en",
        "ru",
        "rv11 query 1/5",
    ),
    Case(
        "What is the relationship between Acme Corp and Иван Петров?",
        "en",
        "en",
        "rv11 query 1/5",
    ),
    Case(
        "What is the relationship between Acme Corp and Иван Петров?",
        "en",
        "ru",
        "rv11 query 1/5",
    ),
    Case("How is Иван Петров related to Acme Corp?", "en", "en", "rv11 query 1/5"),
    Case("How is Иван Петров related to Acme Corp?", "en", "ru", "rv11 query 1/5"),
    Case("Where does Иван Петров ship to?", "en", "en", "rv11 query 1/5"),
    Case("Where does Иван Петров ship to?", "en", "ru", "rv11 query 1/5"),
    Case("Who is the CEO of Иван Петров?", "en", "en", "rv11 query 1/5"),
    Case("Who is the CEO of Иван Петров?", "en", "ru", "rv11 query 1/5"),
    Case("Has Acme Corp ever audited Иван Петров?", "en", "en", "rv11 query 1/5"),
    Case("Has Acme Corp ever audited Иван Петров?", "en", "ru", "rv11 query 1/5"),
    Case("Compare Acme Corp with Иван Петров", "en", "en", "rv11 query 1/5"),
    Case("Compare Acme Corp with Иван Петров", "en", "ru", "rv11 query 1/5"),
    Case(
        "Welche Verträge hat Orion Labs mit 한빛전자 geschlossen?",
        "de",
        "de",
        "rv11 query 1",
    ),
    Case("Seit wann liefert 한빛전자 an Orion Labs?", "de", "de", "rv11 query 1"),
    Case("Combien de fournisseurs a 한빛전자 ?", "fr", "fr", "rv11 query 1"),
    Case("Quel est le chiffre d'affaires de 한빛전자 ?", "fr", "fr", "rv11 query 1"),
    Case("Pourquoi Orion Labs a-t-il quitté 한빛전자 ?", "fr", "fr", "rv11 query 1"),
    Case(
        "¿Por qué Orion Labs dejó a 한빛전자?",
        "es",
        "es",
        "rv11 query 1",
        Tradeoff("ko", TITLE_RUN),
    ),
    Case("Quantos fornecedores tem 한빛전자?", "pt", "pt", "rv11 query 1"),
    Case("Quanti fornitori ha 한빛전자?", "it", "it", "rv11 query 1"),
    Case("Ai là giám đốc của 한빛전자?", "vi", "vi", "rv11 query 1"),
    Case("Orion Labs đã ký hợp đồng với 한빛전자 khi nào?", "vi", "vi", "rv11 query 1"),
    Case("Wie viele Lieferanten hat 株式会社アクメ?", "de", "de", "rv11 query 1"),
    Case("Wer ist der Geschäftsführer von 株式会社アクメ?", "de", "de", "rv11 query 1"),
    Case(
        "Welche Verträge hat Orion Labs mit 株式会社アクメ geschlossen?",
        "de",
        "de",
        "rv11 query 1",
    ),
    Case("Seit wann liefert 株式会社アクメ an Orion Labs?", "de", "de", "rv11 query 1"),
    Case("Combien de fournisseurs a 株式会社アクメ ?", "fr", "fr", "rv11 query 1"),
    Case(
        "Quel est le chiffre d'affaires de 株式会社アクメ ?", "fr", "fr", "rv11 query 1"
    ),
    Case(
        "Pourquoi Orion Labs a-t-il quitté 株式会社アクメ ?", "fr", "fr", "rv11 query 1"
    ),
    Case("¿Cuántos proveedores tiene 株式会社アクメ?", "es", "es", "rv11 query 1"),
    Case(
        "¿Por qué Orion Labs dejó a 株式会社アクメ?",
        "es",
        "es",
        "rv11 query 1",
        Tradeoff("ja", TITLE_RUN),
    ),
    Case("¿Quién dirige 株式会社アクメ?", "es", "es", "rv11 query 1"),
    Case("Quantos fornecedores tem 株式会社アクメ?", "pt", "pt", "rv11 query 1"),
    Case("Quanti fornitori ha 株式会社アクメ?", "it", "it", "rv11 query 1"),
    Case("Ai là giám đốc của 株式会社アクメ?", "vi", "vi", "rv11 query 1"),
    Case(
        "Orion Labs đã ký hợp đồng với 株式会社アクメ khi nào?",
        "vi",
        "vi",
        "rv11 query 1",
    ),
    Case("Wie viele Lieferanten hat 阿克米集团?", "de", "de", "rv11 query 1"),
    Case("Wer ist der Geschäftsführer von 阿克米集团?", "de", "de", "rv11 query 1"),
    Case(
        "Welche Verträge hat Orion Labs mit 阿克米集团 geschlossen?",
        "de",
        "de",
        "rv11 query 1",
    ),
    Case("Seit wann liefert 阿克米集团 an Orion Labs?", "de", "de", "rv11 query 1"),
    Case("Combien de fournisseurs a 阿克米集团 ?", "fr", "fr", "rv11 query 1"),
    Case("Quel est le chiffre d'affaires de 阿克米集团 ?", "fr", "fr", "rv11 query 1"),
    Case("Pourquoi Orion Labs a-t-il quitté 阿克米集团 ?", "fr", "fr", "rv11 query 1"),
    Case("¿Cuántos proveedores tiene 阿克米集团?", "es", "es", "rv11 query 1"),
    Case(
        "¿Por qué Orion Labs dejó a 阿克米集团?",
        "es",
        "es",
        "rv11 query 1",
        Tradeoff("zh", TITLE_RUN),
    ),
    Case("¿Quién dirige 阿克米集团?", "es", "es", "rv11 query 1"),
    Case("Quantos fornecedores tem 阿克米集团?", "pt", "pt", "rv11 query 1"),
    Case("Quanti fornitori ha 阿克米集团?", "it", "it", "rv11 query 1"),
    Case("Ai là giám đốc của 阿克米集团?", "vi", "vi", "rv11 query 1"),
    Case(
        "Orion Labs đã ký hợp đồng với 阿克米集团 khi nào?", "vi", "vi", "rv11 query 1"
    ),
    Case("Wie viele Lieferanten hat ООО Ромашка?", "de", "de", "rv11 query 1"),
    Case("Wer ist der Geschäftsführer von ООО Ромашка?", "de", "de", "rv11 query 1"),
    Case(
        "Welche Verträge hat Orion Labs mit ООО Ромашка geschlossen?",
        "de",
        "de",
        "rv11 query 1",
    ),
    Case("Seit wann liefert ООО Ромашка an Orion Labs?", "de", "de", "rv11 query 1"),
    Case("Combien de fournisseurs a ООО Ромашка ?", "fr", "fr", "rv11 query 1"),
    Case("Quel est le chiffre d'affaires de ООО Ромашка ?", "fr", "fr", "rv11 query 1"),
    Case("Pourquoi Orion Labs a-t-il quitté ООО Ромашка ?", "fr", "fr", "rv11 query 1"),
    Case("¿Cuántos proveedores tiene ООО Ромашка?", "es", "es", "rv11 query 1"),
    Case(
        "¿Por qué Orion Labs dejó a ООО Ромашка?",
        "es",
        "es",
        "rv11 query 1",
        Tradeoff("ru", TITLE_RUN),
    ),
    Case("¿Quién dirige ООО Ромашка?", "es", "es", "rv11 query 1"),
    Case("Quantos fornecedores tem ООО Ромашка?", "pt", "pt", "rv11 query 1"),
    Case("Quanti fornitori ha ООО Ромашка?", "it", "it", "rv11 query 1"),
    Case("Ai là giám đốc của ООО Ромашка?", "vi", "vi", "rv11 query 1"),
    Case(
        "Orion Labs đã ký hợp đồng với ООО Ромашка khi nào?", "vi", "vi", "rv11 query 1"
    ),
    Case("Why did Orion Labs drop ООО Ромашка?", "en", "en", "rv11 query 1"),
    Case(
        "Bank of America和The Walt Disney Company是什么关系？", "zh", "en", "rv10 probe"
    ),
    Case("Procter and Gamble的CEO是谁？", "zh", "en", "rv10 probe"),
    Case("University of Tokyo在哪里？", "zh", "en", "rv10 probe"),
    Case(
        "ما علاقة Bank of America بـ The Walt Disney Company؟", "ar", "en", "rv10 probe"
    ),
    Case(
        "Bank of America กับ The Home Depot เกี่ยวข้องกันอย่างไร",
        "th",
        "en",
        "rv10 probe",
    ),
    Case(
        "Bank of America와 The Walt Disney Company의 관계는?", "ko", "en", "rv10 probe"
    ),
    Case(
        "Bank of AmericaとThe Walt Disney Companyの関係は？", "ja", "en", "rv10 probe"
    ),
    Case("Microsoft Corporation和OpenAI是什么关系？", "zh", "en", "rv10 probe"),
    Case("ما علاقة Microsoft Corporation بـ OpenAI؟", "ar", "en", "rv10 probe"),
    Case(
        "Microsoft Corporation กับ OpenAI เกี่ยวข้องกันอย่างไร",
        "th",
        "en",
        "rv10 probe",
    ),
    Case("מה הקשר בין Microsoft Corporation ל-OpenAI?", "he", "en", "rv10 probe"),
    Case("Microsoft Corporation और OpenAI का क्या संबंध है?", "hi", "en", "rv10 probe"),
    Case(
        "Какова связь между Microsoft Corporation и OpenAI?", "ru", "en", "rv10 probe"
    ),
    Case("Microsoft Corporation и OpenAI?", "ru", "en", "rv10 probe"),
    Case("Microsoft Corporation ו-OpenAI?", "he", "en", "rv10 probe"),
    Case("Bank of America का CEO कौन है?", "hi", "en", "rv10 probe"),
    Case("Who is the CEO of Acme?", "en", "en", "rv10 probe"),
    Case("Acme CEO?", "en", "en", "rv10 probe"),
    Case("What does 株式会社アクメ make?", "en", "en", "rv10 probe"),
    Case("Who founded 한빛물산?", "en", "en", "rv10 probe"),
    Case("Who founded Société Générale?", "en", "en", "rv10 probe"),
    Case("What is Café de Flore?", "en", "en", "rv10 probe"),
    Case("How many units did Orion Labs reject?", "en", "en", "rv10 probe"),
    Case("Combien d'unités Orion Labs a-t-il rejetées ?", "fr", "en", "rv10 probe"),
    Case("Qui est le PDG d'Acme ?", "fr", "en", "rv10 probe"),
    Case("Le PDG ?", "fr", "en", "rv10 probe", Tradeoff("en", NO_WORD)),
    Case(
        "Quel est le chiffre d'affaires de la Banca d'Italia ?",
        "fr",
        "en",
        "rv10 probe",
    ),
    Case("Chi è l'amministratore delegato di Acme?", "it", "en", "rv10 probe"),
    Case("Qual è il fatturato della Société Générale?", "it", "en", "rv10 probe"),
    Case("Dov'è il deposito?", "it", "en", "rv10 probe"),
    Case("¿Quién es el director de Acme?", "es", "en", "rv10 probe"),
    Case("¿Cuántos proveedores tiene Acme?", "es", "en", "rv10 probe"),
    Case("Quem é o diretor da Acme?", "pt", "en", "rv10 probe"),
    Case("Quantos fornecedores a Acme tem?", "pt", "en", "rv10 probe"),
    Case("Wer ist der Geschäftsführer von Acme?", "de", "en", "rv10 probe"),
    Case("Wie viele Lieferanten hat Acme?", "de", "en", "rv10 probe"),
    Case("Ai là giám đốc của Acme?", "vi", "en", "rv10 probe"),
    Case("Acme có bao nhiêu nhà cung cấp?", "vi", "en", "rv10 probe"),
    Case("애크미 로보틱스는 누구와 계약했나?", "ko", "en", "rv10 probe"),
    Case("Acme Robotics Inc.의 CEO는?", "ko", "en", "rv10 probe"),
    Case("アクメの社長は誰？", "ja", "en", "rv10 probe"),
    Case("東京本社の所在地", "ja", "en", "rv10 probe"),
    Case("阿克米的首席执行官是谁？", "zh", "en", "rv10 probe"),
    Case("Is Acme's CEO Kim Min-jun?", "en", "en", "rv10 probe"),
    Case(
        "Was ist Acme? asked the CEO",
        "en",
        "en",
        "rv10 probe",
        Tradeoff("de", QUOTED_SENTENCE),
    ),
    Case("pemasok Acme siapa?", "en", "en", "rv10 probe"),
    Case("Wie is de CEO van Acme?", "en", "en", "rv10 probe"),
    Case("Кто такой Ёжиков?", "ru", "en", "rv10 probe"),
    Case("Hanbit Electronics and Orion Electronics 비교", "ko", "en", "rv10 probe"),
    Case("What is 계약 조건 기간 요약?", "ko", "en", "rv12 fix (runs)"),
    Case("Has Acme Corp ever audited 한빛물산?", "en", "en", "rv11 query 1/5"),
    # A shouted text reads its words lowercased (rv13 query 1).
    Case("¿QUIÉN FIRMÓ EL CONTRATO?", "es", "en", "rv13 query 1"),
    Case("¿QUIÉN FIRMÓ EL CONTRATO?", "es", "es", "rv13 query 1"),
    Case("WER HAT DEN VERTRAG UNTERSCHRIEBEN?", "de", "en", "rv13 query 1"),
    Case("QUI A SIGNÉ LE CONTRAT ?", "fr", "en", "rv13 query 1"),
    Case("CHI HA FIRMATO IL CONTRATTO?", "it", "en", "rv13 query 1"),
    Case("QUEM ASSINOU O CONTRATO?", "pt", "en", "rv13 query 1"),
    Case("AI LÀ NGƯỜI KÝ HỢP ĐỒNG?", "vi", "en", "rv13 query 1"),
    Case("WHAT IS THE PENALTY CLAUSE?", "en", "en", "rv13 query 1"),
    Case("WHO SIGNED THE CONTRACT?", "en", "ko", "rv13 query 1"),
    Case("Is the OS supported on ARM?", "en", "ko", "rv13 query 1"),
    Case("AI ethics guide", "en", "ko", "rv13 query 1"),
    Case("ES index mapping", "en", "ko", "rv13 query 1"),
    Case("DO droplet pricing", "en", "en", "rv13 query 1"),
    Case("EST to KST conversion", "en", "en", "rv13 query 1"),
    # The parts of a URL, an email, a path or a dotted name are no
    # function words (rv13 query 3).
    Case("https://example.com 요약", "ko", "ko", "rv13 query 3"),
    Case("example.com 접속 안 됨", "ko", "ko", "rv13 query 3"),
    Case("www.example.com 접속 오류", "ko", "ko", "rv13 query 3"),
    Case("admin@example.com 계정 잠김", "ko", "ko", "rv13 query 3"),
    Case("com.example.app 패키지 오류", "ko", "ko", "rv13 query 3"),
    Case("example.com に接続できない", "ja", "ja", "rv13 query 3"),
    Case("admin@example.com のパスワード", "ja", "ja", "rv13 query 3"),
    Case("https://example.com/docs を要約して", "ja", "ja", "rv13 query 3"),
    Case("support@example.com への問い合わせ方法", "ja", "ja", "rv13 query 3"),
    Case("example.com 无法访问", "zh", "zh", "rv13 query 3"),
    Case("admin@example.com 密码重置", "zh", "zh", "rv13 query 3"),
    Case("https://example.com/docs 页面摘要", "zh", "zh", "rv13 query 3"),
    Case("support@example.com 联系方式", "zh", "zh", "rv13 query 3"),
    Case("example.com.br 접속", "ko", "ko", "rv13 query 3"),
    Case("example.de 접속 불가", "ko", "ko", "rv13 query 3"),
    Case("example.es 사이트", "ko", "ko", "rv13 query 3"),
    Case("example.it 도메인", "ko", "ko", "rv13 query 3"),
    Case("http://example.com/do 경로 오류", "ko", "ko", "rv13 query 3"),
    Case("/usr/local/bin/the 경로", "ko", "ko", "rv13 query 3"),
    Case("C:\\Users\\the 폴더", "ko", "ko", "rv13 query 3"),
    Case("my_the_var 변수", "ko", "ko", "rv13 query 3"),
    Case("NullPointerException at com.example.Main 원인", "ko", "ko", "rv13 query 3"),
    Case("Не открывается example.com", "ru", "ru", "rv13 query 3"),
    Case("لا يفتح example.com", "ar", "ar", "rv13 query 3"),
    Case("Why does example.com.br time out?", "en", "en", "rv13 query 3"),
    Case("Where is the README of github.com/acme/tool?", "en", "en", "rv13 query 3"),
    Case("¿Por qué falla example.com?", "es", "es", "rv13 query 3"),
    Case("o arquivo config.do não carrega", "pt", "pt", "rv13 query 3"),
    Case("file.per 열기", "ko", "ko", "rv13 query 3"),
    Case("app.es 번역 파일", "ko", "ko", "rv13 query 3"),
    Case("app.cho 확장자", "ko", "ko", "rv13 query 3"),
    Case("node.js 설치", "ko", "ko", "rv13 query 3"),
    Case("Vue.js와 React 비교", "ko", "ko", "rv13 query 3"),
    # rv14 query (detection): shouted text, acronyms in another script, URLs and
    # joiners, Spanish and Portuguese words.
    Case("WO IST DAS LAGER DER FIRMA?", "de", "en", "rv14 query"),
    Case("WARUM WURDE DIE LIEFERUNG STORNIERT?", "de", "en", "rv14 query"),
    Case("¿CUÁNTOS PROVEEDORES TIENE LA EMPRESA?", "es", "en", "rv14 query"),
    Case("¿DÓNDE ESTÁ LA OFICINA PRINCIPAL?", "es", "en", "rv14 query"),
    Case("POURQUOI LA LIVRAISON EST-ELLE EN RETARD ?", "fr", "en", "rv14 query"),
    Case("OÙ SE TROUVE LE SIÈGE SOCIAL ?", "fr", "en", "rv14 query"),
    Case("DOVE SI TROVA IL MAGAZZINO PRINCIPALE?", "it", "en", "rv14 query"),
    Case("PERCHÉ LA CONSEGNA È IN RITARDO?", "it", "en", "rv14 query"),
    Case("ONDE FICA O ARMAZÉM PRINCIPAL?", "pt", "en", "rv14 query"),
    Case("QUANTOS FORNECEDORES TEM A EMPRESA?", "pt", "en", "rv14 query"),
    Case(
        "TẠI SAO LÔ HÀNG BỊ CHẬM?",
        "vi",
        "en",
        "rv14 query",
        Tradeoff("en", NO_LISTED_WORD),
    ),
    Case("CÔNG TY CÓ BAO NHIÊU NHÀ CUNG CẤP?", "vi", "en", "rv14 query"),
    Case("ГДЕ НАХОДИТСЯ ГЛАВНЫЙ СКЛАД?", "ru", "en", "rv14 query"),
    Case("ПОЧЕМУ ЗАДЕРЖАЛИ ПОСТАВКУ?", "ru", "en", "rv14 query"),
    Case("WHAT ARE THE PAYMENT TERMS OF THE CONTRACT?", "en", "en", "rv14 query"),
    Case("ΠΟΙΟΣ ΥΠΕΓΡΑΨΕ ΤΗ ΣΥΜΒΑΣΗ;", "en", "en", "rv14 query"),
    Case("WHO SIGNED THE SUPPLY CONTRACT?", "en", "ko", "rv14 query"),
    Case("WHERE IS THE MAIN WAREHOUSE?", "en", "ko", "rv14 query"),
    Case("HOW MANY VENDORS DO WE HAVE?", "en", "de", "rv14 query"),
    Case("QUIÉN ES EL DIRECTOR DE LA EMPRESA?", "es", "de", "rv14 query"),
    Case("¿QUIÉN FIRMÓ?", "es", "en", "rv14 query", Tradeoff("en", SHOUTED)),
    Case("WER UNTERSCHRIEB?", "de", "en", "rv14 query", Tradeoff("en", SHOUTED)),
    Case("POURQUOI PAS?", "fr", "en", "rv14 query", Tradeoff("en", SHOUTED)),
    Case("PERCHÉ NO?", "it", "en", "rv14 query", Tradeoff("en", SHOUTED)),
    Case("ONDE FICA?", "pt", "en", "rv14 query", Tradeoff("en", SHOUTED)),
    Case("ГДЕ СКЛАД?", "ru", "en", "rv14 query"),
    Case("WHO SIGNED?", "en", "de", "rv14 query", Tradeoff("de", SHOUTED)),
    Case("WHERE IS?", "en", "de", "rv14 query", Tradeoff("de", SHOUTED)),
    Case("QUI A SIGNÉ LE CONTRAT AVEC 株式会社アクメ ?", "fr", "en", "rv14 query"),
    Case("¿QUIÉN FIRMÓ EL CONTRATO CON 한빛전자?", "es", "en", "rv14 query"),
    Case("WER IST DER CHEF VON 東京物流?", "de", "en", "rv14 query"),
    Case("WHO IS THE CEO OF 株式会社アクメ?", "en", "en", "rv14 query"),
    Case("AS IS TO BE 프로세스 분석", "ko", "ko", "rv14 query"),
    Case("AS IS / TO BE 비교", "ko", "ko", "rv14 query"),
    Case("DE EN FR 번역 품질 비교", "ko", "ko", "rv14 query"),
    Case("EN JA ES 다국어 지원", "ko", "ko", "rv14 query"),
    Case("PT BR ES 현지화 일정", "ko", "ko", "rv14 query"),
    Case("US EU UK 규제 비교", "ko", "ko", "rv14 query"),
    Case("IT OT IS 보안 점검", "ko", "ko", "rv14 query"),
    Case("AWS EKS IAM 설정 방법", "ko", "ko", "rv14 query"),
    Case("CEO CFO CTO 역할", "ko", "ko", "rv14 query"),
    Case("ESG CSR IT 보고서", "ko", "ko", "rv14 query"),
    Case("NO GO 판단 기준 IT", "ko", "ko", "rv14 query"),
    Case("DO AND DONT 가이드", "ko", "ko", "rv14 query"),
    Case("AS IS TO BE 分析の進め方", "ja", "ja", "rv14 query"),
    Case("DE EN FR の翻訳品質", "ja", "ja", "rv14 query"),
    Case("IT OT IS の違い", "ja", "ja", "rv14 query"),
    Case("AWS EKS IAM の設定", "ja", "ja", "rv14 query"),
    Case("AS IS TO BE 流程分析", "zh", "zh", "rv14 query"),
    Case("DE EN FR 翻译质量", "zh", "zh", "rv14 query"),
    Case("IT OT IS 的区别", "zh", "zh", "rv14 query"),
    Case("AS IS TO BE анализ", "ru", "ru", "rv14 query"),
    Case("DE EN FR перевод", "ru", "ru", "rv14 query"),
    Case(
        "Comparar AS IS TO BE", "es", "en", "rv14 query", Tradeoff("en", NO_LISTED_WORD)
    ),
    Case("Was bedeutet AS IS TO BE?", "de", "en", "rv14 query"),
    Case("https://ejemplo.es/contacto 요약", "ko", "en", "rv14 query"),
    Case("https://example.de/die/der/das ページを要約", "ja", "en", "rv14 query"),
    Case("https://example.it/il-prezzo 摘要", "zh", "en", "rv14 query"),
    Case("info@empresa.com.br 연락처", "ko", "en", "rv14 query"),
    Case("contact@societe.fr に連絡", "ja", "en", "rv14 query"),
    Case("ventas@empresa.es 的联系人", "zh", "en", "rv14 query"),
    Case("C:\\Users\\the\\docs 경로", "ko", "en", "rv14 query"),
    Case("~/work/the/notes.md ファイル", "ja", "en", "rv14 query"),
    Case("/var/log/de/app.log 日志", "zh", "en", "rv14 query"),
    Case("os.path.join 사용법", "ko", "en", "rv14 query"),
    Case("std::vector の使い方", "ja", "en", "rv14 query"),
    Case("is_valid 함수 설명", "ko", "en", "rv14 query"),
    Case("the_value 变量", "zh", "en", "rv14 query"),
    Case("obj.is_empty() 메서드", "ko", "en", "rv14 query"),
    Case("com.example.de.App 클래스", "ko", "en", "rv14 query"),
    Case("git@github.com:org/the-repo.git 클론", "ko", "en", "rv14 query"),
    Case("s3://bucket/de/la/ruta 접근 오류", "ko", "en", "rv14 query"),
    Case(
        "https://пример.рф 요약", "ko", "en", "rv14 query", Tradeoff("ru", URL_LETTERS)
    ),
    Case("сайт пример.рф не открывается", "ru", "en", "rv14 query"),
    Case("https://example.com/docs のページは何ですか", "ja", "en", "rv14 query"),
    Case("Как открыть https://example.com/en/the-docs ?", "ru", "en", "rv14 query"),
    Case("Who owns example.es?", "en", "en", "rv14 query"),
    Case("¿Quién es el dueño de example.com?", "es", "en", "rv14 query"),
    Case("Qui gère contact@exemple.fr ?", "fr", "en", "rv14 query"),
    Case("Wer betreibt die Seite example.com?", "de", "en", "rv14 query"),
    Case("Chi gestisce il sito example.it?", "it", "en", "rv14 query"),
    Case("Quem administra o site example.com.br?", "pt", "en", "rv14 query"),
    Case("Ai quản lý trang example.vn?", "vi", "en", "rv14 query"),
    Case("Кто владеет доменом example.com?", "ru", "en", "rv14 query"),
    Case("What does os.path.join do?", "en", "en", "rv14 query"),
    Case("¿Qué hace la función os.path.join?", "es", "en", "rv14 query"),
    Case("Quem e/ou o que foi afetado?", "pt", "en", "rv14 query"),
    Case("¿Quién y/o qué falló?", "es", "en", "rv14 query"),
    Case("Wer und/oder was ist betroffen?", "de", "en", "rv14 query"),
    Case("Qui et/ou quoi est concerné ?", "fr", "en", "rv14 query"),
    Case("Chi e/o cosa è coinvolto?", "it", "en", "rv14 query"),
    Case("Frage:Wer hat den Vertrag unterschrieben?", "de", "en", "rv14 query"),
    Case("Pregunta:¿quién firmó?", "es", "en", "rv14 query"),
    Case("Question:qui a signé le contrat?", "fr", "en", "rv14 query"),
    Case("Domanda:chi ha firmato il contratto?", "it", "en", "rv14 query"),
    Case("Pergunta:quem assinou o contrato?", "pt", "en", "rv14 query"),
    Case("Wer hat unterschrieben.Wo ist das Lager?", "de", "en", "rv14 query"),
    Case("¿Quién firmó?¿Dónde está?", "es", "en", "rv14 query"),
    Case("Quem assinou/aprovou o contrato?", "pt", "en", "rv14 query"),
    Case("Wer hat den Vertrag unterschrieben/genehmigt?", "de", "en", "rv14 query"),
    Case("Wer ist der Chef?Wo ist das Lager?", "de", "en", "rv14 query"),
    Case("Who signed and/or approved?", "en", "de", "rv14 query"),
    Case("Who is the owner/the operator?", "en", "de", "rv14 query"),
    Case("Q:who signed the deal?", "en", "de", "rv14 query"),
    Case("Como funciona el sistema?", "es", "en", "rv14 query"),
    Case("Como puedo cancelar el pedido?", "es", "en", "rv14 query"),
    Case("Como configurar la cuenta", "es", "en", "rv14 query"),
    Case("Tan grande como Acme", "es", "en", "rv14 query", Tradeoff("en", SHARED_TIE)),
    Case("Como se calcula el precio?", "es", "en", "rv14 query"),
    Case("¿Como se firma un contrato?", "es", "en", "rv14 query"),
    Case("Como esta el proyecto", "es", "en", "rv14 query"),
    Case("Como funciona o sistema?", "pt", "en", "rv14 query"),
    Case("Como posso cancelar o pedido?", "pt", "en", "rv14 query"),
    Case("Como se calcula o preço?", "pt", "en", "rv14 query"),
    Case("Como funciona?", "es", "es", "rv14 query"),
    Case("Como se hace?", "es", "es", "rv14 query"),
    Case("Quanto tempo ci vuole per la consegna?", "it", "en", "rv14 query"),
    Case("Quanto vale l'azienda?", "it", "en", "rv14 query"),
    Case("Quanto costa?", "it", "it", "rv14 query"),
    Case("Quanto dura?", "it", "it", "rv14 query"),
    Case(
        "Quanto custa a licença?", "pt", "en", "rv14 query", Tradeoff("it", SHARED_WORD)
    ),
    Case(
        "Quanto tempo leva a entrega?",
        "pt",
        "en",
        "rv14 query",
        Tradeoff("it", SHARED_WORD),
    ),
    Case("Quanto custa?", "pt", "pt", "rv14 query", Tradeoff("it", SHARED_WORD)),
    Case("Quanto vale?", "pt", "pt", "rv14 query", Tradeoff("it", SHARED_WORD)),
)

NUMBERS: tuple[Case, ...] = (
    Case("No hay duda de que son tres.", 3, "es", "rv12 query 2/4"),
    Case("No hay duda: son tres proveedores.", 3, "es", "rv12 query 2/4"),
    Case(
        "No tiene sentido dudarlo: tiene tres proveedores.", 3, "es", "rv12 query 2/4"
    ),
    Case("Acme tiene tres proveedores; no hay más.", 3, "es", "rv12 query 2/4"),
    Case("No hay más de tres proveedores.", 3, "es", "rv12 query 2/4"),
    Case("No hay menos de 3 proveedores.", 3, "es", "rv12 query 2/4"),
    Case("No tiene más de 3 proveedores.", 3, "es", "rv12 query 2/4"),
    Case("No existe un único proveedor: son 3.", 3, "es", "rv12 query 2/4"),
    Case("No hay proveedores.", 0, "es", "rv12 query 2/4"),
    Case(
        "Il n'y a pas que trois fournisseurs, il y en a cinq.",
        5,
        "fr",
        "rv12 query 2/4",
    ),
    Case("Il n'y a pas de doute : trois fournisseurs.", 3, "fr", "rv12 query 2/4"),
    Case("Il n'y a pas de fournisseur.", 0, "fr", "rv12 query 2/4"),
    Case("Il n'y a pas plus de trois fournisseurs.", 3, "fr", "rv12 query 2/4"),
    Case(
        "Acme n'a pas de concurrent direct mais trois fournisseurs.",
        3,
        "fr",
        "rv12 query 2/4",
    ),
    Case("Non ha senso: sono tre.", 3, "it", "rv12 query 2/4"),
    Case("Non ha mai avuto più di tre fornitori.", 3, "it", "rv12 query 2/4"),
    Case("Non ci sono dubbi: tre fornitori.", 3, "it", "rv12 query 2/4"),
    Case("Non ci sono fornitori.", 0, "it", "rv12 query 2/4"),
    Case("Acme ha tre fornitori, non ha altri partner.", 3, "it", "rv12 query 2/4"),
    Case("Non ha meno di 3 fornitori.", 3, "it", "rv12 query 2/4"),
    Case("Não há dúvida de que são três.", 3, "pt", "rv12 query 2/4"),
    Case("Não tem mais de 3 fornecedores.", 3, "pt", "rv12 query 2/4"),
    Case("Não há fornecedores.", 0, "pt", "rv12 query 2/4"),
    Case("Сомнений нет: их три.", 3, "ru", "rv12 query 2/4"),
    Case("Нет сомнений, их три.", 3, "ru", "rv12 query 2/4"),
    Case("Нет, три.", 3, "ru", "rv12 query 2/4"),
    Case("Поставщиков нет.", 0, "ru", "rv12 query 2/4"),
    Case("У Acme три поставщика, других нет.", 3, "ru", "rv12 query 2/4"),
    Case("Не менее трёх, нет точных данных.", 3, "ru", "rv12 query 2/4"),
    Case("Нет данных о точном числе, но около трёх.", 3, "ru", "rv12 query 2/4"),
    Case("There are no fewer than 3 suppliers.", 3, "en", "rv12 query 2/4"),
    Case(
        "I could not find any other details, but Acme has 3 suppliers.",
        3,
        "en",
        "rv12 query 2/4",
    ),
    Case(
        "Acme has 3 suppliers; it does not have any others.", 3, "en", "rv12 query 2/4"
    ),
    Case("It doesn't have any fewer than 3.", 3, "en", "rv12 query 2/4"),
    Case("Not any.", 0, "en", "rv12 query 2/4"),
    Case("It does not have any suppliers.", 0, "en", "rv12 query 2/4"),
    Case("I'm not aware of any more than three suppliers.", 3, "en", "rv12 query 2/4"),
    Case("2019부터 3곳", 3, "ko", "rv12 query 2/4"),
    Case("2019년부터 3곳", 3, "ko", "rv12 query 2/4"),
    Case("2019以降、3社", 3, "ja", "rv12 query 2/4"),
    Case("从2019到现在有3家", 3, "zh", "rv12 query 2/4"),
    Case("2019至2023年共3家", 3, "zh", "rv12 query 2/4"),
    Case("3社から2社に減った", 3, "ja", "rv12 query 2/4"),
    Case("2019から2社", 2, "ja", "rv12 query 2/4"),
    Case("3까지 늘었다", 3, "ko", "rv12 query 2/4"),
    Case("2,019부터", 2019, "ko", "rv12 query 2/4"),
    Case("1999到2000年有5家", 5, "zh", "rv12 query 2/4"),
    Case("2020年まで3社", 3, "ja", "rv12 query 2/4"),
    Case("3到5家", 3, "zh", "rv12 query 2/4"),
    Case("2000到3000家", 2000, "zh", "rv12 query 2/4"),
    Case("1000부터 2000까지 범위에서 1500명", 1000, "ko", "rv12 query 2/4"),
    Case("1500から2000社", 1500, "ja", "rv12 query 2/4"),
    Case("2000명까지", 2000, "ko", "rv12 query 2/4"),
    Case("3000至5000家", 3000, "zh", "rv12 query 2/4"),
    Case("1990부터 2000까지 3곳", 3, "ko", "rv12 query 2/4"),
    Case("Since 2019, 3 suppliers", 3, "en", "rv12 query 2/4"),
    Case("起2019年3家", 3, "zh", "rv12 query 2/4"),
    Case("No vendors were found.", 0, "en", "rv11 probe"),
    Case("There are no vendors.", 0, "en", "rv11 probe"),
    Case("In 2019, three vendors signed.", 3, "en", "rv11 probe"),
    Case("In March 2019 three vendors signed.", 3, "en", "rv11 probe"),
    Case("Of the three vendors, only one signed.", 1, "en", "rv11 probe"),
    Case("One of them signed; three vendors total.", 3, "en", "rv11 probe"),
    Case("The twenty-first vendor; three in total", 3, "en", "rv11 probe"),
    Case("Early 2024 saw 5 vendors", 5, "en", "rv11 probe"),
    Case("Mid-2024, 5 vendors", 5, "en", "rv11 probe"),
    Case("not any vendor", 0, "en", "rv11 probe"),
    Case("not a single vendor", 0, "en", "rv11 probe"),
    Case("Twenty-one vendors", 21, "en", "rv11 probe"),
    Case("In 5 years they added 3 vendors", 5, "en", "rv11 probe"),
    Case("The first vendor signed, then three more", 3, "en", "rv11 probe"),
    Case("Only one of the vendors signed", 1, "en", "rv11 probe"),
    Case("from 2 to 5 vendors", 2, "en", "rv11 probe"),
    Case("Of all 12 vendors, 3 signed", 3, "en", "rv11 probe"),
    Case("Since 1999 there were 4", 4, "en", "rv11 probe"),
    Case("1,200 vendors", 1200, "en", "rv11 probe"),
    Case("No. 5 vendor", None, "en", "rv11 probe", Tradeoff((5.0,), WORD_GAPS)),
    Case("No vendors.", 0, None, "rv11 probe"),
    Case("Three vendors", 3, None, "rv11 probe"),
    Case("Treinta y dos", None, None, "rv11 probe"),
    Case("No, hay tres proveedores.", 3, "es", "rv11 probe"),
    Case("Acme no tiene proveedores.", 0, "es", "rv11 probe"),
    Case("No tiene ningún proveedor.", 0, "es", "rv11 probe"),
    Case("En 2019 firmó con tres proveedores", 3, "es", "rv11 probe"),
    Case("En marzo de 2019 firmó con tres", 3, "es", "rv11 probe"),
    Case("Desde 2019, 3 proveedores", 3, "es", "rv11 probe"),
    Case("De los tres proveedores, solo dos firmaron", 2, "es", "rv11 probe"),
    Case("Uno de ellos firmó; en total tres", 3, "es", "rv11 probe"),
    Case("el vigésimo primer proveedor; tres en total", 3, "es", "rv11 probe"),
    Case("Ninguno", 0, "es", "rv11 probe"),
    Case("ningún", 0, "es", "rv11 probe"),
    Case(
        "Hay 2.500 proveedores",
        2500,
        "es",
        "rv11 probe",
        Tradeoff((2.5,), SCALE_DIGITS),
    ),
    Case(
        "Hay 2,5 millones",
        2500000.0,
        "es",
        "rv11 probe",
        Tradeoff((1000000.0,), SCALE_DIGITS),
    ),
    Case("Entre los 12 proveedores, 3 firmaron", 3, "es", "rv11 probe"),
    Case("En 5 años añadieron 3", 5, "es", "rv11 probe"),
    Case("En los tres países hay 5 plantas", 5, "es", "rv11 probe"),
    Case("No total, são três fornecedores.", 3, "pt", "rv11 probe"),
    Case("No contrato, a Acme tem 3 fornecedores.", 3, "pt", "rv11 probe"),
    Case("Nenhum fornecedor", 0, "pt", "rv11 probe"),
    Case("Dos três fornecedores, só dois assinaram", 2, "pt", "rv11 probe"),
    Case("Em 2019, três fornecedores", 3, "pt", "rv11 probe"),
    Case("Dos 5 contratos, 2 foram assinados", 2, "pt", "rv11 probe"),
    Case("Dois deles assinaram; três no total", 2, "pt", "rv11 probe"),
    Case("Os dados dos 3 fornecedores", 3, "pt", "rv11 probe"),
    Case("A empresa dos três sócios tem 5 lojas", 3, "pt", "rv11 probe"),
    Case("Aucun fournisseur.", 0, "fr", "rv11 probe"),
    Case("Il n'y a aucun fournisseur", 0, "fr", "rv11 probe"),
    Case("Depuis 2019, trois fournisseurs", 3, "fr", "rv11 probe"),
    Case("En mars 2019, trois fournisseurs", 3, "fr", "rv11 probe"),
    Case("Des trois fournisseurs, seul un a signé", 3, "fr", "rv11 probe"),
    Case("Le chiffre des 3 fournisseurs est 5", 3, "fr", "rv11 probe"),
    Case("Des fournisseurs: 3", 3, "fr", "rv11 probe"),
    Case("L'un d'entre eux a signé; trois au total", 3, "fr", "rv11 probe"),
    Case("Le vingt-troisième fournisseur; trois au total", 3, "fr", "rv11 probe"),
    Case("quatre-vingt-dix fournisseurs", 90, "fr", "rv11 probe"),
    Case("soixante et onze", 71, "fr", "rv11 probe"),
    Case("Les données des 3 usines", 3, "fr", "rv11 probe"),
    Case(
        "Je n'ai trouvé aucune information; 3 usines",
        0,
        "fr",
        "rv11 probe",
        Tradeoff((0.0, 3.0), OTHER_NUMBER),
    ),
    Case("Kein Lieferant", 0, "de", "rv11 probe"),
    Case("Keine Lieferanten", 0, "de", "rv11 probe"),
    Case("Seit 2019 drei Lieferanten", 3, "de", "rv11 probe"),
    Case("Im Jahr 2019 drei Lieferanten", 3, "de", "rv11 probe"),
    Case(
        "Von den drei Lieferanten hat nur einer unterschrieben", 3, "de", "rv11 probe"
    ),
    Case("Von 2019 bis 2023 drei Lieferanten", 3, "de", "rv11 probe"),
    Case("Von den 5 Werken sind 2 aktiv", 2, "de", "rv11 probe"),
    Case("Einer von ihnen; insgesamt drei", 3, "de", "rv11 probe"),
    Case("Vor 5 Jahren drei Lieferanten", 5, "de", "rv11 probe"),
    Case("vierundzwanzig", 24, "de", "rv11 probe"),
    Case("Nach 2020 drei", 3, "de", "rv11 probe"),
    Case("In 3 Werken arbeiten 50", 3, "de", "rv11 probe"),
    Case("Ab 2019 drei", 3, "de", "rv11 probe"),
    Case("Nessun fornitore", 0, "it", "rv11 probe"),
    Case("Nel 2019 tre fornitori", 3, "it", "rv11 probe"),
    Case("Dei tre fornitori, solo due hanno firmato", 2, "it", "rv11 probe"),
    Case("mille e cinquecento", 1500, "it", "rv11 probe"),
    Case("Uno di loro ha firmato; tre in totale", 3, "it", "rv11 probe"),
    Case("I dati dei 3 fornitori", 3, "it", "rv11 probe"),
    Case("Delle 5 fabbriche, 2 sono attive", 2, "it", "rv11 probe"),
    Case("Il ventitreesimo fornitore; tre", 3, "it", "rv11 probe"),
    Case("Non ci sono fornitori", 0, "it", "rv11 probe"),
    Case("Ни одного поставщика", 0, "ru", "rv11 probe"),
    Case("В 2019 году три поставщика", 3, "ru", "rv11 probe"),
    Case("Из трёх поставщиков два", 2, "ru", "rv11 probe"),
    Case("один из них подписал; всего три", 3, "ru", "rv11 probe"),
    Case("двадцать первый поставщик; всего три", 3, "ru", "rv11 probe"),
    Case("С 2019 года три", 3, "ru", "rv11 probe"),
    Case("Из 5 заводов 2 работают", 2, "ru", "rv11 probe"),
    Case("Нет поставщиков", 0, "ru", "rv11 probe"),
    Case("К 2025 году 3 завода", 3, "ru", "rv11 probe"),
    Case("В 3 городах 5 заводов", 3, "ru", "rv11 probe"),
    Case("약 1.5만 명의 직원과 3개 공장", 15000, "ko", "rv11 probe"),
    Case("1,200만 명", 12000000.0, "ko", "rv11 probe"),
    Case("2024년에 세 곳", 3, "ko", "rv11 probe"),
    Case("공급업체는 없습니다", 0, "ko", "rv11 probe"),
    Case("0.5만 명", 5000, "ko", "rv11 probe"),
    Case("3.5개 공장", None, "ko", "rv11 probe", Tradeoff((3.5,), SCALE_DIGITS)),
    Case("1.5배", None, "ko", "rv11 probe", Tradeoff((1.5,), SCALE_DIGITS)),
    Case("3.14", 3.14, "ko", "rv11 probe"),
    Case("1.5억 원", 150000000.0, "ko", "rv11 probe"),
    Case("약 1.5 만 명", 15000, "ko", "rv11 probe"),
    Case("一千零一家", 1001, "zh", "rv11 probe"),
    Case("两千零二十四家", 2024, "zh", "rv11 probe"),
    Case("一万零五百人", 10500, "zh", "rv11 probe"),
    Case("十万零三", 100003, "zh", "rv11 probe", Tradeoff(None, HAN_BARE)),
    Case("零家", 0, "zh", "rv11 probe"),
    Case("零", None, "zh", "rv11 probe"),
    Case("三千零五十家", 3050, "zh", "rv11 probe"),
    Case("二零二四年", None, "zh", "rv11 probe"),
    Case("1.5万人", 15000, "zh", "rv11 probe"),
    Case("第一万零一", None, "zh", "rv11 probe"),
    Case("一百零八将", 108, "zh", "rv11 probe", Tradeoff(None, HAN_BARE)),
    Case("零点五万", None, "zh", "rv11 probe"),
    Case("共有一億人", 100000000.0, "zh", "rv11 probe"),
    Case("1.2.3万", None, "zh", "rv11 probe"),
    Case("1,23万", None, "zh", "rv11 probe"),
    Case("约3,000万元", 30000000.0, "zh", "rv11 probe"),
    Case("1.5亿元", 150000000.0, "zh", "rv11 probe"),
    Case("一億円", 100000000.0, "ja", "rv11 probe"),
    Case("千二百社", 1200, "ja", "rv11 probe"),
    Case("3,000万円", 30000000.0, "ja", "rv11 probe"),
    Case("1.5万人", 15000, "ja", "rv11 probe"),
    Case("2024年に3社", 3, "ja", "rv11 probe"),
    Case("零社", 0, "ja", "rv11 probe"),
    Case("1万2千5百円", 12500, "ja", "rv11 probe"),
    Case("Vingt-quatre fournisseurs", 24, "fr", "rv10 probe"),
    Case("vingt et un", 21, "fr", "rv10 probe"),
    Case("quatre-vingt-dix-sept", 97, "fr", "rv10 probe"),
    Case("soixante-dix", 70, "fr", "rv10 probe"),
    Case("deux cent trente", 230, "fr", "rv10 probe"),
    Case("trois cents", 300, "fr", "rv10 probe"),
    Case("deux mille vingt-quatre", 2024, "fr", "rv10 probe"),
    Case("Il est arrivé vingt-troisième.", None, "fr", "rv10 probe"),
    Case("le vingt-deuxième jour, trois fournisseurs", 3, "fr", "rv10 probe"),
    Case("Un seul fournisseur", 1, "fr", "rv10 probe", Tradeoff(None, WORD_GAPS)),
    Case("L'un d'eux, trois au total", 3, "fr", "rv10 probe"),
    Case(
        "3 millions de clients",
        3000000.0,
        "fr",
        "rv10 probe",
        Tradeoff((3.0, 1000000.0), SCALE_DIGITS),
    ),
    Case("deux millions trois cent mille", 2300000, "fr", "rv10 probe"),
    Case("Le bâtiment neuf a trois fournisseurs", 3, "fr", "rv10 probe"),
    Case("mille", 1000, "fr", "rv10 probe"),
    Case("Cent un", 101, "fr", "rv10 probe"),
    Case("un million", 1000000.0, "fr", "rv10 probe"),
    Case("quatre vingt", 80, "fr", "rv10 probe"),
    Case("dix-huit", 18, "fr", "rv10 probe"),
    Case("dix-sept", 17, "fr", "rv10 probe"),
    Case("cinq cent mille", 500000, "fr", "rv10 probe"),
    Case("vingt-et-un", 21, "fr", "rv10 probe"),
    Case("mille neuf cent quatre-vingt-dix-neuf", 1999, "fr", "rv10 probe"),
    Case("Deux ou trois", None, "fr", "rv10 probe", Tradeoff((2.0, 3.0), WORD_GAPS)),
    Case("Treinta y dos", 32, "es", "rv10 probe"),
    Case("ciento veintiuno", 121, "es", "rv10 probe"),
    Case("doscientos mil", 200000, "es", "rv10 probe"),
    Case("dos mil trescientos", 2300, "es", "rv10 probe"),
    Case("Uno de ellos", None, "es", "rv10 probe"),
    Case("mil millones", 1000000000.0, "es", "rv10 probe"),
    Case("un millón", 1000000.0, "es", "rv10 probe"),
    Case("Veintiún proveedores", 21, "es", "rv10 probe"),
    Case("ciento uno", 101, "es", "rv10 probe"),
    Case("cien mil", 100000, "es", "rv10 probe"),
    Case("El vigésimo primero", None, "es", "rv10 probe"),
    # Alone, with no word after it, a form of one is one (rv23 eval low:
    # "Hay uno." answers 1); before a word it is the article.
    Case("una", 1, "es", "rv10 probe, rv23 eval low"),
    Case("Hay uno.", 1, "es", "rv23 eval low"),
    Case("Solo uno.", 1, "es", "rv23 eval low"),
    Case("Hay un proveedor.", None, "es", "rv23 eval low"),
    Case("Ni uno solo.", 0, "es", "rv23 eval low", question="¿Cuántos proveedores?"),
    Case("Il y en a un.", 1, "fr", "rv23 eval low"),
    Case("C'è uno.", 1, "it", "rv23 eval low"),
    Case("Nur einer.", 1, "de", "rv23 eval low"),
    Case("Es gibt einen.", 1, "de", "rv23 eval low"),
    Case("Bei einer Firma.", None, "de", "rv23 eval low"),
    Case("Há um.", 1, "pt", "rv23 eval low"),
    Case("Há um fornecedor.", None, "pt", "rv23 eval low"),
    Case("Nem um.", 0, "pt", "rv23 eval low", question="Quantos fornecedores?"),
    Case("하나입니다.", 1, "ko", "rv23 eval low"),
    Case("하나도 없습니다.", 0, "ko", "rv23 eval low", question="몇 곳입니까?"),
    Case("하나은행입니다.", None, "ko", "rv23 eval low"),
    Case("셋이에요.", 3, "ko", "rv23 eval low"),
    Case("Not a single one.", 0, "en", "rv23 eval low", question="How many suppliers?"),
    Case("Not a single one but three.", 3, "en", "rv23 eval low"),
    Case("Dos o tres", None, "es", "rv10 probe", Tradeoff((2.0, 3.0), WORD_GAPS)),
    Case("quinientos", 500, "es", "rv10 probe"),
    Case("ciento cincuenta mil", 150000, "es", "rv10 probe"),
    Case("Zwei Millionen", 2000000.0, "de", "rv10 probe"),
    Case("einundzwanzig", 21, "de", "rv10 probe"),
    Case("zweitausenddreihundert", 2300, "de", "rv10 probe"),
    Case("hundertzwei", 102, "de", "rv10 probe"),
    Case("Ein Lieferant", None, "de", "rv10 probe"),
    Case("eine Million", 1000000.0, "de", "rv10 probe"),
    Case("dreihundertfünfundsechzig", 365, "de", "rv10 probe"),
    Case("am vierundzwanzigsten Mai drei Lieferanten", 3, "de", "rv10 probe"),
    Case("zwei und drei", None, "de", "rv10 probe"),
    Case("Eins", 1, "de", "rv10 probe"),
    Case("tausend", 1000, "de", "rv10 probe"),
    Case("Zwanzig Millionen Euro", 20000000.0, "de", "rv10 probe"),
    Case("einhundert", 100, "de", "rv10 probe"),
    Case("zweiundzwanzigtausend", 22000, "de", "rv10 probe"),
    Case("duecentotrentadue", 232, "it", "rv10 probe"),
    Case("ventitré", 23, "it", "rv10 probe"),
    Case("ventotto", 28, "it", "rv10 probe"),
    Case("Mille e cinquecento", 1500, "it", "rv10 probe"),
    Case("duemila", 2000, "it", "rv10 probe"),
    Case("tremila", 3000, "it", "rv10 probe"),
    Case("un milione", 1000000.0, "it", "rv10 probe"),
    Case("centouno", 101, "it", "rv10 probe"),
    Case("Sei fornitori", 6, "it", "rv10 probe"),
    Case("diciottesimo", None, "it", "rv10 probe"),
    Case("due milioni", 2000000.0, "it", "rv10 probe"),
    Case("cinquecento", 500, "it", "rv10 probe"),
    Case("Vinte e quatro", 24, "pt", "rv10 probe"),
    Case("cento e vinte", 120, "pt", "rv10 probe"),
    Case("mil e quinhentos", 1500, "pt", "rv10 probe"),
    Case("duzentas", 200, "pt", "rv10 probe"),
    Case("Uma empresa", None, "pt", "rv10 probe"),
    Case("dois mil e vinte", 2020, "pt", "rv10 probe"),
    Case("dezesseis", 16, "pt", "rv10 probe"),
    Case("um milhão", 1000000.0, "pt", "rv10 probe"),
    Case("Двадцать одна", 21, "ru", "rv10 probe"),
    Case("Три тысячи", 3000, "ru", "rv10 probe"),
    Case("двадцать первый", None, "ru", "rv10 probe"),
    Case("сто двадцать пять", 125, "ru", "rv10 probe"),
    Case("две тысячи двадцать четыре", 2024, "ru", "rv10 probe"),
    Case("один из них", 1, "ru", "rv10 probe"),
    Case("из трёх поставщиков два", 2, "ru", "rv10 probe"),
    Case("пять тысяч", 5000, "ru", "rv10 probe"),
    Case("одна тысяча", 1000, "ru", "rv10 probe"),
    Case("Четырёх", 4, "ru", "rv10 probe"),
    Case("twenty-first", None, "en", "rv10 probe"),
    Case("Three hundred and twelve", 312, "en", "rv10 probe"),
    Case("one of them", 1, "en", "rv10 probe"),
    Case("3 million", 3000000.0, "en", "rv10 probe", Tradeoff((3.0,), SCALE_DIGITS)),
    Case("two thousand twenty-four", 2024, "en", "rv10 probe"),
    Case("a dozen", 12, "en", "rv10 probe", Tradeoff(None, WORD_GAPS)),
    Case("열두 대", 12, "ko", "rv10 probe"),
    Case("1만 2천 명", 12000, "ko", "rv10 probe"),
    Case("만 이천 명", 12000, "ko", "rv10 probe"),
    Case("3만", 30000, "ko", "rv10 probe"),
    Case("십이 대", 12, "ko", "rv10 probe"),
    Case("스물한 명", 21, "ko", "rv10 probe"),
    Case("두 번째", None, "ko", "rv10 probe"),
    Case("十二台", 12, "zh", "rv10 probe"),
    Case("一万二千", 12000, "zh", "rv10 probe", Tradeoff(None, HAN_BARE)),
    Case("3万", 30000, "zh", "rv10 probe"),
    Case("两千", 2000, "zh", "rv10 probe", Tradeoff(None, HAN_BARE)),
    Case("二〇二四年", None, "zh", "rv10 probe"),
    Case("十万八千", 108000, "zh", "rv10 probe", Tradeoff(None, HAN_BARE)),
    Case("一千零一", 1001, "zh", "rv10 probe", Tradeoff(None, HAN_BARE)),
    Case("三千零五十", 3050, "zh", "rv10 probe", Tradeoff(None, HAN_BARE)),
    Case("两万", 20000, "zh", "rv10 probe", Tradeoff(None, HAN_BARE)),
    Case("一億", 100000000.0, "zh", "rv10 probe", Tradeoff(None, HAN_BARE)),
    Case("第三", None, "zh", "rv10 probe"),
    Case("1.5万", 15000, "zh", "rv10 probe"),
    Case("3,000万", 30000000.0, "zh", "rv10 probe"),
    Case("三社", 3, "ja", "rv10 probe"),
    Case("一万二千社", 12000, "ja", "rv10 probe"),
    Case("3万人", 30000, "ja", "rv10 probe"),
    Case("十二件", 12, "ja", "rv10 probe"),
    Case("千二百", 1200, "ja", "rv10 probe", Tradeoff(None, HAN_BARE)),
    Case("百", 100, "ja", "rv10 probe", Tradeoff(None, HAN_BARE)),
    Case("2万5千", 25000, "ja", "rv10 probe"),
    Case(
        "Aunque el informe no indica la fecha, la empresa tiene tres proveedores.",
        3,
        "es",
        "rv10/rv11 findings",
    ),
    Case(
        "El contrato no especifica plazos; Acme tiene 3 proveedores.",
        3,
        "es",
        "rv10/rv11 findings",
    ),
    Case(
        "O relatório não indica; são três fornecedores.", 3, "pt", "rv10/rv11 findings"
    ),
    Case(
        "Il rapporto non lo dice; i fornitori sono tre.", 3, "it", "rv10/rv11 findings"
    ),
    Case("quatre vingt dix", 90, "fr", "rv10/rv11 findings"),
    Case("quatre vingts fournisseurs", 80, "fr", "rv10/rv11 findings"),
    Case(
        "Según el informe, que no detalla fechas, hay 3 proveedores.",
        3,
        "es",
        "rv10/rv11 findings",
    ),
    Case("2019 이후 3곳", 3, "ko", "rv10/rv11 findings"),
    Case("No hay información sobre proveedores.", 0, "es", "rv10/rv11 findings"),
    Case("Нет данных о поставщиках.", 0, "ru", "rv10/rv11 findings"),
    Case(
        "I do not have any information about suppliers.", 0, "en", "rv10/rv11 findings"
    ),
    Case("Нет. Три поставщика.", 3, "ru", "rv12 fix (yes/no word)"),
    Case("No hay.", 0, "es", "rv12 fix (yes/no word)"),
    Case(
        "No hay más de una docena.",
        None,
        "es",
        "rv12 query 2 (comparative)",
        Tradeoff((0.0,), WORD_GAPS),
    ),
    # An absence with the absent thing's own number (rv13 query 2).
    Case(
        "There are not any vendors with more than 100 employees.",
        0,
        "en",
        "rv13 query 2",
        Tradeoff((0.0, 100.0), NO_QUESTION),
    ),
    Case(
        "Acme does not have any plants in Zone 2.",
        0,
        "en",
        "rv13 query 2",
        Tradeoff((0.0, 2.0), NO_QUESTION),
    ),
    Case(
        "Acme doesn't have any suppliers for project 7.",
        0,
        "en",
        "rv13 query 2",
        Tradeoff((0.0, 7.0), NO_QUESTION),
    ),
    Case(
        "There aren't any suppliers in the top 10.",
        0,
        "en",
        "rv13 query 2",
        Tradeoff((0.0, 10.0), NO_QUESTION),
    ),
    Case(
        "Acme does not have any contracts worth over 5 million.",
        0,
        "en",
        "rv13 query 2",
        Tradeoff((0.0, 5.0), NO_QUESTION),
    ),
    Case(
        "No hay proveedores en el lote 7.",
        0,
        "es",
        "rv13 query 2",
        Tradeoff((0.0, 7.0), NO_QUESTION),
    ),
    Case(
        "No hay proveedores con más de 50 empleados.",
        0,
        "es",
        "rv13 query 2",
        Tradeoff((0.0, 50.0), NO_QUESTION),
    ),
    Case(
        "Non ci sono fornitori con più di 50 dipendenti.",
        0,
        "it",
        "rv13 query 2",
        Tradeoff((0.0, 50.0), NO_QUESTION),
    ),
    Case(
        "Non ci sono fornitori nel lotto 7.",
        0,
        "it",
        "rv13 query 2",
        Tradeoff((0.0, 7.0), NO_QUESTION),
    ),
    Case(
        "Il n'y a pas de fournisseur dans le lot 7.",
        0,
        "fr",
        "rv13 query 2",
        Tradeoff((0.0, 7.0), NO_QUESTION),
    ),
    Case(
        "Il n'y a pas de fournisseur avec plus de 50 salariés.",
        0,
        "fr",
        "rv13 query 2",
        Tradeoff((0.0, 50.0), NO_QUESTION),
    ),
    Case(
        "Não há fornecedores no lote 7.",
        0,
        "pt",
        "rv13 query 2",
        Tradeoff((0.0, 7.0), NO_QUESTION),
    ),
    Case(
        "Não há fornecedores com mais de 50 funcionários.",
        0,
        "pt",
        "rv13 query 2",
        Tradeoff((0.0, 50.0), NO_QUESTION),
    ),
    Case(
        "Поставщиков с более чем 50 сотрудниками нет.",
        0,
        "ru",
        "rv13 query 2",
        Tradeoff((0.0, 50.0), NO_QUESTION),
    ),
    Case(
        "Поставщиков в лоте 7 нет.",
        0,
        "ru",
        "rv13 query 2",
        Tradeoff((0.0, 7.0), NO_QUESTION),
    ),
    Case("Не более пяти поставщиков, точно нет шести.", 5, "ru", "rv13 query 2"),
    Case("3 vendors do not have any contracts.", 3, "en", "rv13 query 2"),
    Case(
        "No hay datos de 2019. En 2020 tenía 3 plantas.",
        0,
        "es",
        "rv13 query 2",
        Tradeoff((0.0, 3.0, 2019.0, 2020.0), OTHER_NUMBER),
    ),
    Case("No hay uno sino tres.", 3, "es", "rv13 query 2"),
    Case("No hay más detalles.", 0, "es", "rv13 query 5"),
    Case("Não há mais detalhes.", 0, "pt", "rv13 query 5"),
    Case(
        "I could not find any other details; but Acme has 3 suppliers.",
        3,
        "en",
        "rv13 query 2",
    ),
    Case(
        "Acme does not have any suppliers; it uses 3 internal plants.",
        0,
        "en",
        "rv13 query 2",
        Tradeoff((0.0, 3.0), OTHER_NUMBER),
    ),
    # Comparatives beside an absence (rv13 query 5).
    Case("No hay proveedores; no hay más detalles.", 0, "es", "rv13 query 5"),
    Case("No hay proveedores ni menos derechos.", 0, "es", "rv13 query 5"),
    Case("Non ci sono fornitori, è il caso più diffuso.", 0, "it", "rv13 query 5"),
    Case("Não há fornecedores nem mais detalhes.", 0, "pt", "rv13 query 5"),
    Case(
        "Il n'y a pas de fournisseur, plus des détails suivront.",
        0,
        "fr",
        "rv13 query 5",
    ),
    Case("Поставщиков нет, больше данных нет.", 0, "ru", "rv13 query 5"),
    Case("Поставщиков нет — по большему счёту.", 0, "ru", "rv13 query 5"),
    Case(
        "Acme does not have any suppliers, more thanks to the audit.",
        0,
        "en",
        "rv13 query 5",
    ),
    Case("There are not any vendors, less thankfully.", 0, "en", "rv13 query 5"),
    Case("Нет больше поставщиков.", 0, "ru", "rv13 query 5"),
    # rv14 query (numbers): answers with their question; punctuation inside
    # a number; ranges; Russian oblique numerals; CJK conditions.
    Case(
        "There are not any vendors with more than 1,000 employees.",
        0,
        "en",
        "rv14 query",
        Tradeoff((0.0, 1000.0), NO_QUESTION),
    ),
    Case(
        "Acme does not have any contracts worth more than $1,500,000.",
        0,
        "en",
        "rv14 query",
        Tradeoff((0.0, 1500000.0), NO_QUESTION),
    ),
    Case(
        "There are not any suppliers with revenue above 2,500 units.",
        0,
        "en",
        "rv14 query",
        Tradeoff((0.0, 2500.0), NO_QUESTION),
    ),
    Case(
        "No hay proveedores con una facturación de más de 2,5 millones.",
        0,
        "es",
        "rv14 query",
        Tradeoff((0.0, 1000000.0), NO_QUESTION),
    ),
    Case(
        "Il n'y a pas de fournisseur avec un chiffre d'affaires de 2,5 millions.",
        0,
        "fr",
        "rv14 query",
        Tradeoff((0.0, 1000000.0), NO_QUESTION),
    ),
    Case(
        "Non ci sono fornitori con un fatturato di 2,5 milioni.",
        0,
        "it",
        "rv14 query",
        Tradeoff((0.0, 1000000.0), NO_QUESTION),
    ),
    Case(
        "Não há fornecedores com faturamento de 2,5 milhões.",
        0,
        "pt",
        "rv14 query",
        Tradeoff((0.0, 1000000.0), NO_QUESTION),
    ),
    Case("Поставщиков с выручкой 2,5 млн нет.", 0, "ru", "rv14 query"),
    Case(
        "There are not any deliveries before 10:30.",
        0,
        "en",
        "rv14 query",
        Tradeoff((0.0, 10.0, 30.0), NO_QUESTION),
    ),
    Case(
        "No hay entregas antes de las 10:30.",
        0,
        "es",
        "rv14 query",
        Tradeoff((0.0, 10.0, 30.0), NO_QUESTION),
    ),
    Case(
        "Il n'y a pas de livraison avant 10:30.",
        0,
        "fr",
        "rv14 query",
        Tradeoff((0.0, 10.0, 30.0), NO_QUESTION),
    ),
    Case(
        "There are not any suppliers in lots 3–5.",
        0,
        "en",
        "rv14 query",
        Tradeoff((0.0, 3.0, 5.0), NO_QUESTION),
    ),
    Case(
        "No hay proveedores en los lotes 3–5.",
        0,
        "es",
        "rv14 query",
        Tradeoff((0.0, 3.0, 5.0), NO_QUESTION),
    ),
    Case(
        "Non ci sono fornitori nei lotti 3–5.",
        0,
        "it",
        "rv14 query",
        Tradeoff((0.0, 3.0, 5.0), NO_QUESTION),
    ),
    Case(
        "There are not any vendors with between 100 and 500 employees.",
        0,
        "en",
        "rv14 query",
        Tradeoff((0.0, 100.0, 500.0), NO_QUESTION),
    ),
    Case(
        "There are not any vendors with more than 100 but fewer than 500 employees.",
        0,
        "en",
        "rv14 query",
        Tradeoff((0.0, 100.0, 500.0), NO_QUESTION),
    ),
    Case(
        "No hay proveedores con más de 100 pero menos de 500 empleados.",
        0,
        "es",
        "rv14 query",
        Tradeoff((0.0, 100.0, 500.0), NO_QUESTION),
    ),
    Case(
        "Il n'y a pas de fournisseur avec plus de 100 mais moins de 500 salariés.",
        0,
        "fr",
        "rv14 query",
        Tradeoff((0.0, 100.0, 500.0), NO_QUESTION),
    ),
    Case(
        "Non ci sono fornitori con più di 100 ma meno di 500 dipendenti.",
        0,
        "it",
        "rv14 query",
        Tradeoff((0.0, 100.0, 500.0), NO_QUESTION),
    ),
    Case(
        "Não há fornecedores com mais de 100 mas menos de 500 funcionários.",
        0,
        "pt",
        "rv14 query",
        Tradeoff((0.0, 100.0, 500.0), NO_QUESTION),
    ),
    Case(
        "Поставщиков с более чем 100, но менее чем 500 сотрудниками нет.",
        0,
        "ru",
        "rv14 query",
        Tradeoff((0.0, 100.0, 500.0), NO_QUESTION),
    ),
    Case(
        "There are not any suppliers in Zone 2, Zone 3 or Zone 4.",
        0,
        "en",
        "rv14 query",
        question="How many suppliers are in Zone 2, Zone 3 or Zone 4?",
    ),
    Case(
        "No hay proveedores en las zonas 2, 3 y 4.",
        0,
        "es",
        "rv14 query",
        Tradeoff((0.0, 2.0, 3.0, 4.0), NO_QUESTION),
    ),
    Case(
        "Il n'y a pas de fournisseur dans les lots 2, 3 et 4.",
        0,
        "fr",
        "rv14 query",
        Tradeoff((0.0, 2.0, 3.0, 4.0), NO_QUESTION),
    ),
    Case(
        "Non ci sono fornitori nei lotti 2, 3 e 4.",
        0,
        "it",
        "rv14 query",
        Tradeoff((0.0, 2.0, 3.0, 4.0), NO_QUESTION),
    ),
    Case(
        "Não há fornecedores nos lotes 2, 3 e 4.",
        0,
        "pt",
        "rv14 query",
        Tradeoff((0.0, 2.0, 3.0, 4.0), NO_QUESTION),
    ),
    Case(
        "There are not any vendors with more than 50 employees, according to the 2023 report.",
        0,
        "en",
        "rv14 query",
        question="According to the 2023 report, how many vendors have more than 50 employees?",
    ),
    Case(
        "No hay proveedores con más de 50 empleados, según el informe.",
        0,
        "es",
        "rv14 query",
        Tradeoff((0.0, 50.0), NO_QUESTION),
    ),
    Case(
        "There are not any vendors, among those with more than 100 employees.",
        0,
        "en",
        "rv14 query",
        question="How many vendors have more than 100 employees?",
    ),
    Case(
        "No hay proveedores, al menos con más de 100 empleados.",
        0,
        "es",
        "rv14 query",
        question="¿Cuántos proveedores tienen más de 100 empleados?",
    ),
    Case(
        "Il n'y a aucun fournisseur de plus de 100 salariés.",
        0,
        "fr",
        "rv14 query",
        Tradeoff((0.0, 100.0), NO_QUESTION),
    ),
    Case(
        "Non c'è nessun fornitore con più di 100 dipendenti.",
        0,
        "it",
        "rv14 query",
        Tradeoff((0.0, 100.0), NO_QUESTION),
    ),
    Case(
        "Es gibt keine Lieferanten mit mehr als 100 Mitarbeitern.",
        0,
        "de",
        "rv14 query",
        Tradeoff((0.0, 100.0), NO_QUESTION),
    ),
    Case(
        "Es gibt keinen Lieferanten in Los 7.",
        0,
        "de",
        "rv14 query",
        Tradeoff((0.0, 7.0), NO_QUESTION),
    ),
    Case(
        "Lieferanten mit mehr als 100 Mitarbeitern gibt es keine.",
        0,
        "de",
        "rv14 query",
        question="Wie viele Lieferanten haben mehr als 100 Mitarbeiter?",
    ),
    Case(
        "Im Los 7 gibt es keine Lieferanten.",
        0,
        "de",
        "rv14 query",
        question="Wie viele Lieferanten gibt es im Los 7?",
    ),
    Case(
        "In Zone 2, there are not any vendors.",
        0,
        "en",
        "rv14 query",
        question="How many vendors are there in Zone 2?",
    ),
    Case(
        "For lot 7, Acme does not have any suppliers.",
        0,
        "en",
        "rv14 query",
        question="How many suppliers does Acme have for lot 7?",
    ),
    Case(
        "En el lote 7 no hay proveedores.",
        0,
        "es",
        "rv14 query",
        question="¿Cuántos proveedores hay en el lote 7?",
    ),
    Case(
        "Dans le lot 7, il n'y a pas de fournisseur.",
        0,
        "fr",
        "rv14 query",
        question="Combien de fournisseurs y a-t-il dans le lot 7 ?",
    ),
    Case(
        "Nel lotto 7 non ci sono fornitori.",
        0,
        "it",
        "rv14 query",
        question="Quanti fornitori ci sono nel lotto 7?",
    ),
    Case(
        "No lote 7 não há fornecedores.",
        0,
        "pt",
        "rv14 query",
        question="Quantos fornecedores há no lote 7?",
    ),
    Case(
        "В лоте 7 поставщиков нет.",
        0,
        "ru",
        "rv14 query",
        question="Сколько поставщиков в лоте 7?",
    ),
    Case(
        "Поставщиков, у которых больше 50 сотрудников, нет.",
        0,
        "ru",
        "rv14 query",
        question="Сколько поставщиков, у которых больше 50 сотрудников?",
    ),
    Case(
        "Поставщиков, у которых более 100 сотрудников, нет.",
        0,
        "ru",
        "rv14 query",
        question="Сколько поставщиков, у которых более 100 сотрудников?",
    ),
    Case(
        "Поставщиков с числом сотрудников больше 100 нет.",
        0,
        "ru",
        "rv14 query",
        question="Сколько поставщиков с числом сотрудников больше 100?",
    ),
    Case(
        "Поставщиков старше 5 лет нет.",
        0,
        "ru",
        "rv14 query",
        question="Сколько поставщиков старше 5 лет?",
    ),
    Case(
        "Нет поставщиков с более чем 100 сотрудниками.",
        0,
        "ru",
        "rv14 query",
        Tradeoff((0.0, 100.0), NO_QUESTION),
    ),
    Case(
        "Нет поставщиков в лоте 7.",
        0,
        "ru",
        "rv14 query",
        Tradeoff((0.0, 7.0), NO_QUESTION),
    ),
    Case(
        "Non ha mai avuto più di tre fornitori nel 2023.",
        3,
        "it",
        "rv14 query",
        question="Quanti fornitori ha avuto Acme nel 2023?",
    ),
    Case(
        "Non hanno mai avuto più di due fornitori.",
        2,
        "it",
        "rv14 query",
        question="Quanti fornitori hanno avuto?",
    ),
    Case(
        "Acme non ha avuto più di tre fornitori.",
        3,
        "it",
        "rv14 query",
        question="Quanti fornitori ha avuto Acme?",
    ),
    Case(
        "Non ci sono mai stati più di tre fornitori.",
        3,
        "it",
        "rv14 query",
        question="Quanti fornitori ci sono stati?",
    ),
    Case(
        "No tiene actualmente más de tres proveedores.",
        3,
        "es",
        "rv14 query",
        question="¿Cuántos proveedores tiene Acme?",
    ),
    Case(
        "No hay actualmente más de tres proveedores.",
        3,
        "es",
        "rv14 query",
        question="¿Cuántos proveedores hay?",
    ),
    Case(
        "No tiene ahora más de cinco plantas.",
        5,
        "es",
        "rv14 query",
        question="¿Cuántas plantas tiene Acme?",
    ),
    Case(
        "Não há atualmente mais de três fornecedores.",
        3,
        "pt",
        "rv14 query",
        question="Quantos fornecedores há?",
    ),
    Case(
        "Não tem hoje mais de cinco fábricas.",
        5,
        "pt",
        "rv14 query",
        question="Quantas fábricas a Acme tem?",
    ),
    Case(
        "Acme does not currently have any more than 3 suppliers.", 3, "en", "rv14 query"
    ),
    Case("It does not have any more than 3 suppliers.", 3, "en", "rv14 query"),
    Case("Non ci sono più di tre fornitori.", 3, "it", "rv14 query"),
    Case("Нет более пяти поставщиков.", 5, "ru", "rv14 query"),
    Case(
        "Нет сейчас более пяти поставщиков.",
        5,
        "ru",
        "rv14 query",
        question="Сколько поставщиков у Acme?",
    ),
    Case("No hay proveedores locales, pero hay 4 extranjeros.", 4, "es", "rv14 query"),
    Case("Il n'y a pas de fournisseur local, mais 4 étrangers.", 4, "fr", "rv14 query"),
    Case("Non ci sono fornitori locali, ma 4 esteri.", 4, "it", "rv14 query"),
    Case("Não há fornecedores locais, mas há 4 estrangeiros.", 4, "pt", "rv14 query"),
    Case("Местных поставщиков нет, но есть 4 иностранных.", 4, "ru", "rv14 query"),
    Case(
        "Acme does not have any local suppliers but it has 4 foreign ones.",
        4,
        "en",
        "rv14 query",
    ),
    Case("There is not any doubt: the answer is 6.", 6, "en", "rv14 query"),
    Case("No hay duda — son seis.", 6, "es", "rv14 query"),
    Case("Non c'è dubbio: sono sei.", 6, "it", "rv14 query"),
    Case("Нет сомнений: их шесть.", 6, "ru", "rv14 query"),
    Case("Il n'y a pas de doute, ils sont six.", 6, "fr", "rv14 query"),
    Case("Não há dúvida de que são seis.", 6, "pt", "rv14 query"),
    Case("No hay duda de que son seis.", 6, "es", "rv14 query"),
    Case("Не более пяти поставщиков.", 5, "ru", "rv14 query"),
    Case("Около шести поставщиков.", 6, "ru", "rv14 query"),
    Case("Из семи поставщиков работают все.", 7, "ru", "rv14 query"),
    Case("Свыше восьми поставщиков.", 8, "ru", "rv14 query"),
    Case("До девяти поставщиков.", 9, "ru", "rv14 query"),
    Case("Не менее десяти поставщиков.", 10, "ru", "rv14 query"),
    Case("Компания работает с пятью поставщиками.", 5, "ru", "rv14 query"),
    Case(
        "Компания работает с семью поставщиками.",
        7,
        "ru",
        "rv14 query",
        Tradeoff(None, HOMONYM),
    ),
    Case("С восемью поставщиками.", 8, "ru", "rv14 query"),
    Case("С десятью поставщиками.", 10, "ru", "rv14 query"),
    Case("О пяти поставщиках известно.", 5, "ru", "rv14 query"),
    Case("Около двадцати поставщиков.", 20, "ru", "rv14 query"),
    Case("Около одиннадцати поставщиков.", 11, "ru", "rv14 query"),
    Case("Не более пятнадцати поставщиков.", 15, "ru", "rv14 query"),
    Case("Около тридцати поставщиков.", 30, "ru", "rv14 query"),
    Case("Около ста поставщиков.", 100, "ru", "rv14 query"),
    Case("Около двадцати пяти поставщиков.", 25, "ru", "rv14 query"),
    Case("Пятый поставщик закрыт.", None, "ru", "rv14 query"),
    Case("Поставщиков пять.", 5, "ru", "rv14 query"),
    Case("У компании семь поставщиков.", 7, "ru", "rv14 query"),
    Case(
        "従業員100人以上の仕入先はありません。",
        0,
        "ja",
        "rv14 query",
        question="従業員100人以上の仕入先は何社ありますか？",
    ),
    Case(
        "100人以上の従業員がいる仕入先はいません。",
        0,
        "ja",
        "rv14 query",
        question="100人以上の従業員がいる仕入先は何社ですか？",
    ),
    Case(
        "ロット7には仕入先がありません。",
        0,
        "ja",
        "rv14 query",
        question="ロット7には仕入先が何社ありますか？",
    ),
    Case(
        "직원 100명 이상인 공급업체는 없습니다.",
        0,
        "ko",
        "rv14 query",
        question="직원 100명 이상인 공급업체는 몇 곳입니까?",
    ),
    Case(
        "7번 로트에는 공급업체가 없습니다.",
        0,
        "ko",
        "rv14 query",
        question="7번 로트에는 공급업체가 몇 곳 있습니까?",
    ),
    Case(
        "직원이 100명을 넘는 공급업체는 없습니다.",
        0,
        "ko",
        "rv14 query",
        question="직원이 100명을 넘는 공급업체는 몇 곳입니까?",
    ),
    Case(
        "员工超过100人的供应商没有。",
        0,
        "zh",
        "rv14 query",
        question="员工超过100人的供应商有几家？",
    ),
    Case(
        "没有员工超过100人的供应商。",
        0,
        "zh",
        "rv14 query",
        Tradeoff((0.0, 100.0), NO_QUESTION),
    ),
    Case("第7批次没有供应商。", 0, "zh", "rv14 query"),
    Case(
        "7号批次没有供应商。", 0, "zh", "rv14 query", question="7号批次有几家供应商？"
    ),
    Case("No hay proveedores, más bien distribuidores.", 0, "es", "rv14 query"),
    Case("Non ci sono fornitori più grandi.", 0, "it", "rv14 query"),
    Case("Il n'y a pas de fournisseur plus grand.", 0, "fr", "rv14 query"),
    Case("Acme does not have any more suppliers than Beta.", 0, "en", "rv14 query"),
    Case("Поставщиков больше нет.", 0, "ru", "rv14 query"),
    Case("Больше поставщиков нет.", 0, "ru", "rv14 query"),
    Case("Нет больше пяти поставщиков.", 5, "ru", "rv14 query"),
    Case(
        "Поставщиков больше пяти нет.",
        0,
        "ru",
        "rv14 query",
        Tradeoff((0.0, 5.0), NO_QUESTION),
    ),
    Case(
        "There are not any vendors with fewer than 3 plants; most have 5.",
        0,
        "en",
        "rv14 query",
        Tradeoff((0.0, 3.0, 5.0), OTHER_NUMBER),
    ),
    # The question restating the count the answer confirms (rv14 query 2).
    Case("Yes, 4.", 4, "en", "rv14 query 2", question="Does Acme have 4 suppliers?"),
    Case(
        "Acme has 4 suppliers.",
        4,
        "en",
        "rv14 query 2",
        question="Does Acme have 4 suppliers?",
    ),
    Case(
        "Sí, Acme tiene 4 proveedores.",
        4,
        "es",
        "rv14 query 2",
        question="¿Tiene Acme 4 proveedores?",
    ),
    Case(
        "Acme tiene tres proveedores; no hay más.",
        3,
        "es",
        "rv14 query 2",
        question="¿Tiene Acme tres proveedores?",
    ),
    Case(
        "No hay más de tres proveedores.",
        3,
        "es",
        "rv14 query 2",
        question="¿Cuántos proveedores hay?",
    ),
    Case(
        "No hay proveedores con más de 50 empleados.",
        0,
        "es",
        "rv14 query 2",
        question="¿Cuántos proveedores tienen más de 50 empleados?",
    ),
    Case(
        "Поставщиков с более чем 50 сотрудниками нет.",
        0,
        "ru",
        "rv14 query 2",
        question="Сколько поставщиков с более чем 50 сотрудниками?",
    ),
    Case("没有超过三家供应商。", 3, "zh", "rv14 query 5", question="有几家供应商？"),
    Case(
        "没有超过三家供应商的批次。",
        0,
        "zh",
        "rv14 query 5",
        question="超过三家供应商的批次有几个？",
    ),
    Case("没有三家以上的供应商。", 3, "zh", "rv14 query 5", question="有几家供应商？"),
    Case(
        "没有三家以上的供应商。",
        0,
        "zh",
        "rv14 query 5",
        question="三家以上的供应商有几个批次？",
    ),
    # Read without the question.
    Case(
        "En el lote 7 no hay proveedores.",
        0,
        "es",
        "rv14 query 5",
        Tradeoff((0.0, 7.0), NO_QUESTION),
    ),
    Case(
        "従業員100人以上の仕入先はありません。",
        0,
        "ja",
        "rv14 query 5",
        Tradeoff((0.0, 100.0), NO_QUESTION),
    ),
)

ROUTER: tuple[Case, ...] = (
    Case("local", "local", None, "rv12 query 3"),
    Case("Sure, here is my choice:\n\nlocal", "local", None, "rv12 query 3"),
    Case(
        "local\n\nAlternatives considered:\n- global\n- drift",
        None,
        None,
        "rv12 query 3",
    ),
    Case(
        "local\n\nWhy not the others?\nglobal\nToo broad.", None, None, "rv12 query 3"
    ),
    Case(
        "**local**\n\n| strategy | fit |\n|---|---|\n| local | high |\n| global | low |",
        "local",
        None,
        "rv12 query 3",
    ),
    Case("Strategy: local\nFallback: global", None, None, "rv12 query 3"),
    Case("local\nfallback: global", None, None, "rv12 query 3"),
    Case('{\n  "strategy": "local"\n}', "local", None, "rv12 query 3"),
    Case(
        '{\n  "strategy": "local",\n  "reason": "entity lookup"\n}',
        "local",
        None,
        "rv12 query 3",
    ),
    Case(
        '{\n  "strategy": "local",\n  "fallback": "global"\n}',
        "local",
        None,
        "rv12 query 3",
    ),
    Case('{"strategy": "local", "fallback": "global"}', "local", None, "rv12 query 3"),
    Case(
        '{\n  "strategy": "local",\n  "alternative": "global"\n}',
        "local",
        None,
        "rv12 query 3",
    ),
    Case(
        '{"strategy": "local", "alternative": "global"}', "local", None, "rv12 query 3"
    ),
    Case('```json\n{"strategy": "local"}\n```', "local", None, "rv12 query 3"),
    Case("<think>maybe global</think>\nlocal", "local", None, "rv12 query 3"),
    Case(
        "<think>\nglobal?\n</think>\n\nlocal\n\nglobal would also work but",
        "local",
        None,
        "rv12 query 3",
    ),
    Case("The best strategy is local.", None, None, "rv12 query 3"),
    Case("Answer: local\n\nNote: global", None, None, "rv12 query 3"),
    Case("local.\n\n(global is for corpus-wide themes)", "local", None, "rv12 query 3"),
    Case(
        "Option A: global\nOption B: local\n\nI pick: local", None, None, "rv12 query 3"
    ),
    Case(
        "Option A: global\nOption B: local\n\nI pick option B",
        None,
        None,
        "rv12 query 3",
    ),
    Case(
        "I would pick local over global.\n\nFinal answer:\nlocal",
        "local",
        None,
        "rv12 query 3",
    ),
    Case("Final answer: local\nConfidence: high", "local", None, "rv12 query 3"),
    Case("local\n- global: no\n- drift: no", "local", None, "rv12 query 3"),
    Case("1. local\n2. global", None, None, "rv12 query 3"),
    Case("Not global.\nlocal", "local", None, "rv12 query 3"),
    Case("global\n\nWait, actually:\nlocal", None, None, "rv12 query 3"),
    Case(
        "This is a simple factual lookup, so:\n\nlocal", "local", None, "rv11 query 2"
    ),
    Case("Sure! Here is a simple choice:\nlocal", "local", None, "rv11 query 2"),
    Case("Not global.", None, None, "rv11 query 2"),
    Case("None of these; local is best", None, None, "rv11 query 2"),
    Case("Sure, here is my answer:\n\nlocal", "local", None, "rv11 query 2"),
    Case("This is a simple factual lookup.", None, None, "rv11 query 2"),
    Case(
        "<think>The user asks about X, maybe global.</think>\nlocal",
        "local",
        None,
        "rv10 query 4",
    ),
    Case("**local**", "local", None, "rv10 query 4"),
    Case("Strategy: local", "local", None, "rv10 query 4"),
    Case("`local`\n", "local", None, "rv10 query 4"),
    Case("Local", "local", None, "rv10 query 4"),
    Case(
        "<think>unfinished reasoning about global vs local", None, None, "rv10 query 4"
    ),
    Case("local\n\nReason: entity lookup.", "local", None, "rv10 query 4"),
    Case('{"strategy": "local"}', "local", None, "rv10 query 4"),
    Case("戦略\uff1alocal", "local", None, "rv17 integration 3"),
    Case("策略\uff1aglobal", "global", None, "rv17 integration 3"),
    Case("Strategy\uff1alocal", "local", None, "rv17 integration 3"),
)

COVERS: tuple[Case, ...] = (
    Case("Nguyễn Văn An", True, "Nguyen Van An", "rv11 query 3"),
    Case("Đặng Văn Lâm", True, "Dang Van Lam", "rv11 query 3"),
    Case("Đỗ Hùng Dũng", True, "Do Hung Dung", "rv11 query 3"),
    Case("Łódź Logistics", True, "Lodz Logistics", "rv11 query 3"),
    Case("Yıldız Holding", True, "Yildiz Holding", "rv11 query 3"),
    Case(
        "Søren Kierkegaard Institut", True, "Soren Kierkegaard Institut", "rv11 query 3"
    ),
    Case("Peña Holdings", True, "Pena Holdings", "rv11 query 3"),
    Case("Guðmundsson Holdings", True, "Gudmundsson Holdings", "rv12 query low"),
    Case("İzmir Liman", True, "Izmir Liman", "rv12 query low"),
    Case("Izmir Liman", True, "İzmir Liman", "rv12 query low"),
    Case("Acme Supplies", False, "Zorblax Supplies", "rv11 probe"),
    Case("टाटा मोटर्स", True, "टाटा", "rv23 algo-ret 2"),
    Case("टी-सीरीज़", False, "टाटा", "rv23 algo-ret 2"),
    Case("टीसीएस", False, "टाटा", "rv23 algo-ret 2"),
    Case("ธนาคารกรุงเทพ", True, "กรุงเทพ", "rv23 algo-ret 2"),
    Case("ธนาคารกสิกรไทย", False, "ธนาคารกรุงเทพ", "rv23 algo-ret 2"),
)

# Answer containment (``answer_contains``): ``text`` is the answer,
# ``gold`` the gold answer, ``context`` the answer language and
# ``expected`` the score.
CONTAINS: tuple[Case, ...] = (
    # Thai, Lao, Myanmar and Khmer write no spaces between words: a gold
    # word is a substring of the answer's run, as in Chinese.
    Case(
        "บริษัท Acme ตั้งอยู่ที่กรุงเทพมหานคร",
        1.0,
        "th",
        "rv23 eval 2",
        gold="กรุงเทพมหานคร",
    ),
    Case("เขาอาศัยอยู่ที่กรุงเทพครับ", 1.0, "th", "rv23 eval 2", gold="กรุงเทพ"),
    Case("บริษัทAcmeตั้งอยู่ที่นี่", 1.0, "th", "rv23 eval 2", gold="Acme"),
    Case("เขาอาศัยอยู่ที่เชียงใหม่", 0.0, "th", "rv23 eval 2", gold="กรุงเทพ"),
    Case("ລາວຢູ່ສະຫວັນ", 1.0, None, "rv23 eval 2", gold="ສະຫວັນ"),
    Case("သူရန်ကုန်မှာ", 1.0, None, "rv23 eval 2", gold="ရန်ကုန်"),
    Case("គាត់រស់នៅភ្នំពេញ", 1.0, None, "rv23 eval 2", gold="ភ្នំពេញ"),
    # Hebrew and Arabic write their prefixed particles onto the word.
    Case("Acme נמצאת בירושלים.", 1.0, "he", "rv23 eval 2", gold="ירושלים"),
    Case("Acme נמצאת ובתל אביב.", 1.0, "he", "rv23 eval 2", gold="תל אביב"),
    Case("Acme נמצאת בחיפה.", 0.0, "he", "rv23 eval 2", gold="ירושלים"),
    Case("تقع شركة أكمي في القاهرة وبيروت.", 1.0, "ar", "rv23 eval 2", gold="بيروت"),
    Case("تقع الشركة بالقاهرة.", 1.0, "ar", "rv23 eval 2", gold="القاهرة"),
    Case("تقع الشركة في دمشق.", 0.0, "ar", "rv23 eval 2", gold="بيروت"),
    # Hindi and Vietnamese are spaced: whole words, as before.
    Case("Acme दिल्ली में स्थित है।", 1.0, "hi", "rv23 eval 2", gold="दिल्ली"),
    Case("Acme ở Hà Nội.", 1.0, "vi", "rv23 eval 2", gold="Hà Nội"),
    # A yes/no gold is the polarity of the first yes/no word that answers;
    # a word with a second meaning answers only before punctuation.
    Case("Sim, ele fez isso; não há dúvida.", 0.0, "pt", "rv23 eval 3", gold="não"),
    Case("Não, ele não fez.", 1.0, "pt", "rv23 eval 3", gold="não"),
    Case("Sim.", 1.0, "pt", "rv23 eval 3", gold="sim"),
    Case("O cartão SIM não está incluído.", 0.0, "pt", "rv23 eval 3", gold="sim"),
    Case("Да, у компании нет долгов.", 0.0, "ru", "rv23 eval 3", gold="нет"),
    Case("Нет, долгов нет.", 1.0, "ru", "rv23 eval 3", gold="нет"),
    Case("Долгов нет никаких, да.", 1.0, "ru", "rv23 eval 3", gold="да"),
    Case("Não há dúvida: sim.", 1.0, "pt", "rv23 eval 3", gold="sim"),
    Case("Oui, il n'y a non plus de retard.", 0.0, "fr", "rv23 eval 3", gold="non"),
    Case("Non, il n'a pas signé.", 1.0, "fr", "rv23 eval 3", gold="non"),
    Case("Ja, das Werk hat keine nein-Stimmen.", 0.0, "de", "rv23 eval 3", gold="nein"),
    Case("Das Werk ist ja geschlossen.", 0.0, "de", "rv23 eval 3", gold="ja"),
    Case("Nein, das Werk ist geschlossen.", 1.0, "de", "rv23 eval 3", gold="nein"),
    Case("Sí, lo hizo.", 1.0, "es", "rv23 eval 3", gold="sí"),
    Case("No lo hizo, sí lo anunció.", 1.0, "es", "rv23 eval 3", gold="sí"),
    Case("Sì, non ci sono dubbi.", 0.0, "it", "rv23 eval 3", gold="no"),
    Case("No, non ha firmato.", 1.0, "it", "rv23 eval 3", gold="no"),
    Case("Không, công ty không có nợ.", 1.0, "vi", "rv23 eval 3", gold="không"),
    Case("Công ty có 3 nhà máy.", 0.0, "vi", "rv23 eval 3", gold="có"),
    Case("Có, công ty đã ký.", 1.0, "vi", "rv23 eval 3", gold="có"),
    Case("नहीं, कंपनी ने हस्ताक्षर नहीं किए।", 1.0, "hi", "rv23 eval 3", gold="नहीं"),
    Case("हाँ, कंपनी ने हस्ताक्षर किए।", 0.0, "hi", "rv23 eval 3", gold="नहीं"),
    Case("לא, החברה לא חתמה.", 1.0, "he", "rv23 eval 3", gold="לא"),
    Case("כן, החברה חתמה.", 1.0, "he", "rv23 eval 3", gold="כן"),
    Case("نعم، لا يوجد دين.", 0.0, "ar", "rv23 eval 3", gold="لا"),
    Case("لا، لم توقع الشركة.", 1.0, "ar", "rv23 eval 3", gold="لا"),
    Case("いいえ、違います。", 0.0, "ja", "rv23 eval 3", gold="はい"),
)


# A corpus case's documents are its text split on this character.
DOCUMENT_BREAK = "\f"


def corpus_documents(case: Case) -> list[str]:
    """The documents of a corpus case (:data:`CORPORA`)."""
    return case.text.split(DOCUMENT_BREAK)


def _document(sentences: str) -> str:
    """A document of 2,000 characters, the most a corpus sample reads of
    each (``domain.languages.corpus.SAMPLE_CHARS``)."""
    return (sentences * (2000 // len(sentences) + 1))[:2000]


_ENGLISH_DOCUMENT = _document(
    "The Vendor ships the spare parts to the Buyer within thirty days of the "
    "order. If a delivery is late, the Buyer may claim a credit for each week "
    "of delay, and the Vendor must repair or replace any part that fails "
    "during the warranty period. "
)
_KOREAN_DOCUMENT = _document(
    "공급자는 주문일로부터 삼십 일 이내에 예비 부품을 구매자에게 납품한다. "
    "납품이 지연되면 구매자는 지연된 주마다 공제를 청구할 수 있으며, 공급자는 "
    "보증 기간 중 고장 난 부품을 수리하거나 교체해야 한다. "
)
_JAPANESE_DOCUMENT = _document(
    "供給者は注文日から三十日以内に予備部品を購入者に納品する。納品が遅れた場合、"
    "購入者は遅延した週ごとに控除を請求でき、供給者は保証期間中に故障した部品を"
    "修理または交換しなければならない。"
)

# Technical English with few function words: no Latin language decides.
_TERSE_ENGLISH_DOCUMENT = (
    "Acme Robotics Q3 report: revenue 12M USD, EBITDA 2M USD, headcount 120. "
    "Lidar unit shipments 4,000 units."
)
# English documents naming a file in another script: 8 Hangul (or Hebrew)
# letters among 62.
_QUOTING_KOREAN_NAME = (
    "Umbra Freight delivers lidar units to Kestrel Depot.\nFile '한글문서.txt'.\n"
)
_QUOTING_HEBREW_NAME = (
    "Umbra Freight delivers lidar units to Kestrel Depot.\nFile 'עברית-מסמך.txt'.\n"
)


def _corpus(*documents: tuple[str, int]) -> str:
    return DOCUMENT_BREAK.join(text for text, count in documents for _ in range(count))


# Mixed corpora (the first documents a corpus sample reads): the language
# most of the documents read as. Pooled, one Korean document's Hangul runs
# outweighed the English function words of four to six English ones
# (rv23 surfaces 2), so prompt tuning wrote source_language 'ko' for a
# corpus that is about 85% English.
CORPORA: tuple[Case, ...] = (
    Case(_corpus((_ENGLISH_DOCUMENT, 4), (_KOREAN_DOCUMENT, 1)), "en", None, "rv23"),
    Case(_corpus((_ENGLISH_DOCUMENT, 5), (_KOREAN_DOCUMENT, 1)), "en", None, "rv23"),
    Case(_corpus((_ENGLISH_DOCUMENT, 6), (_KOREAN_DOCUMENT, 1)), "en", None, "rv23"),
    Case(_corpus((_ENGLISH_DOCUMENT, 2), (_KOREAN_DOCUMENT, 1)), "en", None, "rv23"),
    Case(_corpus((_ENGLISH_DOCUMENT, 8), (_KOREAN_DOCUMENT, 1)), "en", None, "rv23"),
    Case(_corpus((_ENGLISH_DOCUMENT, 4), (_JAPANESE_DOCUMENT, 1)), "en", None, "rv23"),
    Case(_corpus((_KOREAN_DOCUMENT, 4), (_ENGLISH_DOCUMENT, 1)), "ko", None, "rv23"),
    Case(_corpus((_JAPANESE_DOCUMENT, 4), (_ENGLISH_DOCUMENT, 1)), "ja", None, "rv23"),
    # As many documents of each: no language holds most of them.
    Case(_corpus((_ENGLISH_DOCUMENT, 1), (_KOREAN_DOCUMENT, 1)), None, None, "rv23"),
    # ... unless the index language is one of those tied.
    Case(_corpus((_ENGLISH_DOCUMENT, 1), (_KOREAN_DOCUMENT, 1)), "ko", "ko", "rv23"),
    # A Latin-dominant document votes Latin even when its words decide no
    # language (terse technical English), and only for a language written in
    # its dominant script: six terse English documents and a Korean one read
    # as Korean, and an English document quoting a Korean or Hebrew file
    # name as Korean or Hebrew (rv24 quality 2).
    Case(
        _corpus((_TERSE_ENGLISH_DOCUMENT, 6), (_KOREAN_DOCUMENT, 1)), "en", "en", "rv24"
    ),
    Case(
        _corpus((_TERSE_ENGLISH_DOCUMENT, 6), (_KOREAN_DOCUMENT, 1)), None, None, "rv24"
    ),
    Case(_QUOTING_KOREAN_NAME, "en", "en", "rv24"),
    Case(_QUOTING_HEBREW_NAME, "en", "en", "rv24"),
    Case(_corpus((_QUOTING_KOREAN_NAME, 5), (_KOREAN_DOCUMENT, 1)), "en", "en", "rv24"),
    # A Korean document quoting English terms still votes Korean.
    Case(
        _corpus((_KOREAN_DOCUMENT, 3), (_TERSE_ENGLISH_DOCUMENT, 1)), "ko", "en", "rv24"
    ),
)
