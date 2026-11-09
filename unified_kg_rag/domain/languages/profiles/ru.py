# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Russian."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import CLAUDE_TOKENIZER, LanguageProfile

# The cardinals by value: zero to nineteen, and the tens and hundreds.
_ONES = {
    "ноль": 0,
    "нуль": 0,
    "один": 1,
    "одна": 1,
    "одно": 1,
    "одну": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
}
_TENS = {
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
    "сто": 100,
    "двести": 200,
    "триста": 300,
    "четыреста": 400,
    "пятьсот": 500,
    "шестьсот": 600,
    "семьсот": 700,
    "восемьсот": 800,
    "девятьсот": 900,
}
# The oblique cases of one to four: the genitive (also the
# prepositional), the dative and the instrumental, with "ё" and "е" (one's
# genitive, dative, instrumental and prepositional, of every gender).
_SMALL_CASES: dict[int, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    1: (("одного", "одной", "одном"), ("одному",), ("одним",)),
    2: (("двух",), ("двум",), ("двумя",)),
    3: (("трёх", "трех"), ("трём", "трем"), ("тремя",)),
    4: (("четырёх", "четырех"), ("четырём", "четырем"), ("четырьмя",)),
}
# "Семью" is also "семья" (family) in two cases: seven reads in its others.
_HOMONYMS = frozenset({"семью"})


def _unit_cases(
    value: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """The genitive, dative and instrumental forms of the unit ``value``
    (one to nine): five and up end in a soft sign ("пяти", "пятью";
    "восьми", "восемью"), their genitive the dative too."""
    if value in _SMALL_CASES:
        return _SMALL_CASES[value]
    word = next(w for w, v in _ONES.items() if v == value)
    genitive = "восьми" if word == "восемь" else word[:-1] + "и"
    return (genitive,), (genitive,), (word + "ю",)


def _cases(word: str, value: int) -> tuple[str, ...]:
    """The oblique cases of the cardinal ``word`` of ``value`` (two and up):
    "пяти", "двадцати", "сорока", "пятидесяти", "пятьюдесятью", "ста",
    "двухсот", "тремястами"."""
    if value < 10:
        genitive, dative, instrumental = _unit_cases(value)
        return (*genitive, *dative, *instrumental)
    if word.endswith("десят"):
        genitive, _, instrumental = _unit_cases(value // 10)
        return (
            *(g + "десяти" for g in genitive),
            *(i + "десятью" for i in instrumental),
        )
    if word.endswith("ь"):
        return word[:-1] + "и", word + "ю"
    if word == "сорок":
        return ("сорока",)
    if word in ("девяносто", "сто"):
        return (word[:-1] + "а",)
    genitive, dative, instrumental = _unit_cases(value // 100)
    return (
        *(g + "сот" for g in genitive),
        *(g + "стах" for g in genitive),
        *(d + "стам" for d in dative),
        *(i + "стами" for i in instrumental),
    )


def _oblique(words: dict[str, int]) -> dict[str, int]:
    """``words`` with the oblique cases of each cardinal from two up, but
    ``_HOMONYMS``; with ``_ONES``, one's too."""
    forms = dict(words)
    for word, value in words.items():
        if value >= 2:
            forms.update(
                {form: value for form in _cases(word, value) if form not in _HOMONYMS}
            )
    if words is _ONES:
        forms.update({form: 1 for cases in _SMALL_CASES[1] for form in cases})
    return forms


PROFILE = LanguageProfile(
    code=LanguageCode.RU,
    name="Russian",
    script="cyrillic",
    analyzer="russian",
    decline=(
        "Не удалось найти в доступных данных информацию, позволяющую ответить "
        "на этот вопрос."
    ),
    no_value_words=("неизвестно", "нет"),
    # Zero to nineteen, with the feminine and neuter forms of one and two
    # and the oblique cases of each (``_oblique``: "около пяти", "с
    # пятью", "двадцати пяти").
    number_words=_oblique(_ONES),
    number_tens=_oblique(_TENS),
    zero_words=("ни одного", "ни одной", "ни один", "ни одна", "ни одно"),
    absence_phrases=("нет",),
    # Masculine, feminine and neuter.
    ordinal_words=frozenset(
        f"{stem}{ending}"
        for stem, endings in (
            ("перв", ("ый", "ая", "ое")),
            ("втор", ("ой", "ая", "ое")),
            ("трет", ("ий", "ья", "ье")),
            ("четвёрт", ("ый", "ая", "ое")),
            ("четверт", ("ый", "ая", "ое")),
            ("пят", ("ый", "ая", "ое")),
            ("шест", ("ой", "ая", "ое")),
            ("седьм", ("ой", "ая", "ое")),
            ("восьм", ("ой", "ая", "ое")),
            ("девят", ("ый", "ая", "ое")),
        )
        for ending in endings
    ),
    # The forms a count takes them in ("одна тысяча", "три тысячи", "пять
    # тысяч").
    number_scales={
        "тысяча": 1000,
        "тысячу": 1000,
        "тысячи": 1000,
        "тысяч": 1000,
        "миллион": 10**6,
        "миллиона": 10**6,
        "миллионов": 10**6,
        "миллиард": 10**9,
        "миллиарда": 10**9,
        "миллиардов": 10**9,
    },
    # Nominative and genitive, and the usual abbreviations.
    month_words={
        "January": ("январь", "января", "янв"),
        "February": ("февраль", "февраля", "фев"),
        "March": ("март", "марта", "мар"),
        "April": ("апрель", "апреля", "апр"),
        "May": ("май", "мая"),
        "June": ("июнь", "июня", "июн"),
        "July": ("июль", "июля", "июл"),
        "August": ("август", "августа", "авг"),
        "September": ("сентябрь", "сентября", "сен", "сент"),
        "October": ("октябрь", "октября", "окт"),
        "November": ("ноябрь", "ноября", "ноя"),
        "December": ("декабрь", "декабря", "дек"),
    },
    # The year mark: "2024 г.", "2024 года".
    date_fillers=("г", "года", "год"),
    chars_per_token={CLAUDE_TOKENIZER: 2.45},
)
