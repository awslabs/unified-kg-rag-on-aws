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

from unified_kg_rag.domain.languages import PROFILES, LanguageCode, text_language

__all__ = [
    "NO_CONTEXT_ANSWER",
    "NO_CONTEXT_ANSWERS",
    "answer_language",
    "is_no_context_answer",
    "no_context_answer",
]

# One reply per LanguageCode, its profile's (domain/languages/profiles).
NO_CONTEXT_ANSWERS: dict[LanguageCode, str] = {
    code: language.decline for code, language in PROFILES.items()
}

# The English reply, the one earlier releases always gave.
NO_CONTEXT_ANSWER = NO_CONTEXT_ANSWERS[LanguageCode.EN]


def answer_language(query: str, fallback: LanguageCode | str) -> LanguageCode:
    """The language ``query`` is written in (``domain.languages.text_language``):
    its script's, among the Latin-script languages its function words', else
    ``fallback`` (the index language) when that is written in the query's
    script, else English."""
    return text_language(query, fallback)


def no_context_answer(query: str, fallback: LanguageCode | str) -> str:
    """The fixed decline in the language of ``query`` (``answer_language``)."""
    return NO_CONTEXT_ANSWERS[answer_language(query, fallback)]


def is_no_context_answer(text: str) -> bool:
    """Whether ``text`` is one of the fixed declines (in any language)."""
    return text.strip() in _REPLIES


_REPLIES = frozenset(NO_CONTEXT_ANSWERS.values())
