# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every language of the matrix through the language-aware helpers: the
decline's language, the embedded analyzers, grounding, name normalization,
sentence ends and the summary hint's unit."""

from __future__ import annotations

import pytest

from tests.fixtures.language_matrix import MATRIX, MATRIX_IDS, LanguageSample
from unified_kg_rag.adapters.local.search_engine.analyzers import get_analyzer
from unified_kg_rag.domain.ingestion.entity_grounding import (
    is_grounded,
    is_name_grounded,
    span_length,
)
from unified_kg_rag.domain.languages import LanguageCode, profile, text_language
from unified_kg_rag.domain.models.config import (
    DEFAULT_LANGUAGE_ANALYZERS,
    EntityGroundingConfig,
)
from unified_kg_rag.domain.retrieval.no_context import no_context_answer
from unified_kg_rag.shared.utils import normalize_name
from unified_kg_rag.shared.utils.sentences import sentence_ends

pytestmark = pytest.mark.unit

matrix = pytest.mark.parametrize("sample", MATRIX, ids=MATRIX_IDS)


@matrix
def test_the_question_reads_as_its_language(sample: LanguageSample) -> None:
    for fallback in (LanguageCode.EN, sample.code):
        assert text_language(sample.question, fallback) is sample.code
    assert no_context_answer(sample.question, "en") == profile(sample.code).decline


@matrix
def test_the_language_has_its_analyzer(sample: LanguageSample) -> None:
    assert (
        DEFAULT_LANGUAGE_ANALYZERS[sample.code.value] == profile(sample.code).analyzer
    )


@matrix
def test_the_name_matches_lexically_in_the_text(sample: LanguageSample) -> None:
    analyze = get_analyzer(profile(sample.code).analyzer)
    text_tokens = analyze(sample.text)
    name_tokens = analyze(sample.name)
    # A sentence is several tokens, not one run, and the name is among them.
    assert len(text_tokens) > 5
    assert name_tokens and set(name_tokens) <= set(text_tokens)


_GROUNDING = EntityGroundingConfig()
_BOUNDS = {
    "min_span_tokens": _GROUNDING.min_span_tokens,
    "min_overlap_ratio": _GROUNDING.min_overlap_ratio,
}


@matrix
def test_text_and_name_are_grounded(sample: LanguageSample) -> None:
    sentence = sample.text.split(". ")[0]
    # Long enough to judge, so a span the chunk lacks is caught.
    assert span_length(sentence) >= _GROUNDING.min_span_tokens
    assert is_grounded(sentence, sample.text, **_BOUNDS)
    assert is_name_grounded(sample.name, sample.text)
    other = next(s for s in MATRIX if s.code is not sample.code)
    assert not is_grounded(other.text, sample.text, **_BOUNDS)


@matrix
def test_name_normalization_keeps_each_word_whole(sample: LanguageSample) -> None:
    assert len(normalize_name(sample.name).split()) == len(sample.name.split())


# Thai ends a sentence with a space, not a mark.
_NO_STOP = {LanguageCode.TH}


@matrix
def test_the_first_sentence_ends_inside_the_text(sample: LanguageSample) -> None:
    ends = [end for end in sentence_ends(sample.text) if end < len(sample.text)]
    assert bool(ends) is (sample.code not in _NO_STOP)
