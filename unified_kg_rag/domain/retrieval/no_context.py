# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fixed reply of a query the chain declines without a model call.

When retrieval produced no context (or a decline left no text) the chain
answers with a fixed sentence instead of asking the model. The answer
prompt writes in the language of the query, so the fixed reply does too:
:func:`no_context_answer` picks one of :data:`NO_CONTEXT_ANSWERS` (one per
``LanguageCode``) by the script most of the query's letters are in, and
among the Latin-script languages by their function words. A query whose language it cannot tell
gets the index language's reply when that is written in the query's script,
else English.

The reply is not how a decline is recognised: the output metadata carries
``abstained=True`` (``abstention_reason``) for every declined answer.
"""

from __future__ import annotations

import re

from unified_kg_rag.domain.languages import dense_language
from unified_kg_rag.domain.models import LanguageCode
from unified_kg_rag.shared.utils.scripts import count_letters

__all__ = [
    "NO_CONTEXT_ANSWER",
    "NO_CONTEXT_ANSWERS",
    "answer_language",
    "is_no_context_answer",
    "no_context_answer",
]

NO_CONTEXT_ANSWERS: dict[LanguageCode, str] = {
    LanguageCode.DE: (
        "Ich konnte in den verfügbaren Daten keine relevanten Informationen "
        "finden, um diese Frage zu beantworten."
    ),
    LanguageCode.EN: (
        "I could not find relevant information in the available data to answer "
        "this question."
    ),
    LanguageCode.ES: (
        "No encontré información relevante en los datos disponibles para "
        "responder a esta pregunta."
    ),
    LanguageCode.FR: (
        "Je n'ai trouvé aucune information pertinente dans les données "
        "disponibles pour répondre à cette question."
    ),
    LanguageCode.IT: (
        "Non ho trovato informazioni pertinenti nei dati disponibili per "
        "rispondere a questa domanda."
    ),
    LanguageCode.JA: (
        "利用可能なデータには、この質問に答えるための関連情報が見つかりませんでした。"
    ),
    LanguageCode.KO: "사용 가능한 데이터에서 이 질문에 답할 관련 정보를 찾지 못했습니다.",
    LanguageCode.PT: (
        "Não encontrei informações relevantes nos dados disponíveis para "
        "responder a esta pergunta."
    ),
    LanguageCode.RU: (
        "Не удалось найти в доступных данных информацию, позволяющую ответить "
        "на этот вопрос."
    ),
    LanguageCode.ZH: "在现有数据中未找到回答此问题的相关信息。",
}

# The English reply, the one earlier releases always gave.
NO_CONTEXT_ANSWER = NO_CONTEXT_ANSWERS[LanguageCode.EN]

_LATIN_WORD = re.compile(r"[a-zà-öø-ÿ']+")

# Frequent function words of each Latin-script language, none in another
# list (a word two languages share is left out), so a hit is evidence for
# one language.
_FUNCTION_WORDS: dict[LanguageCode, frozenset[str]] = {
    LanguageCode.EN: frozenset(
        "the what which who whom whose how why when where is are were does did "
        "of and with from this that".split()
    ),
    LanguageCode.DE: frozenset(
        "der die das was wer wie warum wann wo ist sind war wurde ein eine und "
        "mit von nicht welche welcher".split()
    ),
    LanguageCode.ES: frozenset(
        "el los las qué quién cuál cómo cuándo dónde es fue y del".split()
    ),
    LanguageCode.FR: frozenset(
        "les des qui quel quelle comment pourquoi quand où est sont était et "
        "avec du une dans".split()
    ),
    LanguageCode.IT: frozenset(
        "gli che chi quale come perché dove è sono della nel uno".split()
    ),
    LanguageCode.PT: frozenset(
        "os quem qual porque onde é são foi com do uma não".split()
    ),
}
_LATIN_LANGUAGES = frozenset(_FUNCTION_WORDS)


def answer_language(query: str, fallback: LanguageCode | str) -> LanguageCode:
    """The language ``query`` is written in, as far as its script and
    function words tell; ``fallback`` (the index language) when they do not
    and it is written in the query's script (a Latin one for a Latin-script
    query), else English.

    The script most of the query's letters are in decides
    (``ScriptCounts.dominant``: a Korean question quoting an English name is
    Korean, an English one quoting a Korean name English); Han, Hangul and
    Kana read as ``dense_language`` says."""
    fallback = _language(fallback)
    counts = count_letters([query])
    script = counts.dominant()
    if script == "dense":
        return dense_language(counts, fallback) or fallback
    if script == "cyrillic":
        return LanguageCode.RU
    words = _LATIN_WORD.findall(query.lower())
    hits = {
        language: sum(word in vocabulary for word in words)
        for language, vocabulary in _FUNCTION_WORDS.items()
    }
    best = max(hits.values(), default=0)
    if best:
        tied = [language for language, count in hits.items() if count == best]
        if fallback in tied:
            return fallback
        if len(tied) == 1:
            return tied[0]
    return fallback if fallback in _LATIN_LANGUAGES else LanguageCode.EN


def no_context_answer(query: str, fallback: LanguageCode | str) -> str:
    """The fixed decline in the language of ``query`` (``answer_language``)."""
    return NO_CONTEXT_ANSWERS[answer_language(query, fallback)]


def is_no_context_answer(text: str) -> bool:
    """Whether ``text`` is one of the fixed declines (in any language)."""
    return text.strip() in _REPLIES


_REPLIES = frozenset(NO_CONTEXT_ANSWERS.values())


def _language(code: LanguageCode | str) -> LanguageCode:
    try:
        return LanguageCode(code)
    except ValueError:
        return LanguageCode.EN
