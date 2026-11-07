# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The corpus script check against processing.translation.source_language
(``domain.ingestion.corpus_language``), the loading stage's WARNING and the
setting prompt tuning writes."""

from __future__ import annotations

import logging

import pytest
from langchain_core.documents import Document

from unified_kg_rag.application.cli import run_prompt_tuning
from unified_kg_rag.domain.ingestion.corpus_language import (
    corpus_source_language,
    detected_dense_language,
    script_share,
    source_language_mismatch,
)
from unified_kg_rag.domain.models import Config, LanguageCode

pytestmark = pytest.mark.unit

# Synthetic text, long enough to decide (40 letters and more).
KOREAN = "벤더는 구매자에게 부품을 공급하고 납기와 가격을 계약서에 정한다. " * 8
JAPANESE = "ベンダーは購入者に部品を供給し、納期と価格を契約書で定める。" * 8
CHINESE = "供应商向买方供应零件，并在合同中约定交货期和价格。" * 10
ENGLISH = "The vendor supplies parts to the buyer under a fixed price contract. " * 5
# Korean prose with the Latin terms technical text carries.
KOREAN_TECHNICAL = (
    "벤더는 API 게이트웨이와 Lambda 함수로 주문을 처리하고 S3 버킷에 저장한다. " * 6
)


@pytest.mark.parametrize(
    ("text", "language"),
    [
        (KOREAN, LanguageCode.KO),
        (KOREAN_TECHNICAL, LanguageCode.KO),
        (JAPANESE, LanguageCode.JA),
        (CHINESE, LanguageCode.ZH),
        (ENGLISH, None),
    ],
    ids=["ko", "ko-technical", "ja", "zh", "en"],
)
def test_the_sample_reads_as_its_dense_script_language(
    text: str, language: LanguageCode | None
) -> None:
    assert detected_dense_language(script_share([text])) is language


@pytest.mark.parametrize(
    ("text", "source", "expected"),
    [
        (KOREAN, LanguageCode.EN, "set it to 'ko'"),
        (JAPANESE, LanguageCode.EN, "set it to 'ja'"),
        (CHINESE, LanguageCode.DE, "set it to 'zh'"),
        (ENGLISH, LanguageCode.KO, "only 0% of the sampled letters"),
        (KOREAN, LanguageCode.JA, "reads as Korean"),
    ],
    ids=["ko-as-en", "ja-as-en", "zh-as-de", "en-as-ko", "ko-as-ja"],
)
def test_a_script_the_setting_does_not_name_is_reported(
    text: str, source: LanguageCode, expected: str
) -> None:
    reason = source_language_mismatch(script_share([text]), source)
    assert reason is not None and expected in reason
    assert "processing.translation.source_language" in reason


@pytest.mark.parametrize(
    ("text", "source"),
    [
        (KOREAN, LanguageCode.KO),
        (KOREAN_TECHNICAL, LanguageCode.KO),
        (JAPANESE, LanguageCode.JA),
        (CHINESE, LanguageCode.ZH),
        # One Latin script: English and German are not told apart.
        (ENGLISH, LanguageCode.EN),
        (ENGLISH, LanguageCode.DE),
        # Too few letters decide nothing.
        ("벤더 Vendor 123", LanguageCode.EN),
    ],
)
def test_an_agreeing_or_undecidable_sample_is_not_reported(
    text: str, source: LanguageCode
) -> None:
    assert source_language_mismatch(script_share([text]), source) is None


def test_a_two_sentence_corpus_still_decides() -> None:
    # The sample corpora a first run uses are this small.
    tiny = [
        "벤더는 서울에 있는 구매자에게 전자 부품을 공급한다.",
        "구매자는 계약서에 정한 날까지 대금을 지급한다.",
    ]
    reason = source_language_mismatch(script_share(tiny), LanguageCode.EN)
    assert reason is not None and "set it to 'ko'" in reason


def test_the_sample_is_bounded() -> None:
    # The first documents' first characters only: a corpus is not read whole.
    share = script_share([ENGLISH * 100] * 30 + [KOREAN * 100])
    assert share.dense == 0
    assert share.letters <= 20 * 2000


def test_the_tuned_language_is_the_scripts_else_the_profiles() -> None:
    assert corpus_source_language([KOREAN], "English") is LanguageCode.KO
    assert corpus_source_language([ENGLISH], "German") is LanguageCode.DE
    assert corpus_source_language([ENGLISH], "fr") is LanguageCode.FR
    assert corpus_source_language([ENGLISH], "Klingon") is None


def test_prompt_tuning_writes_the_source_language(caplog) -> None:
    result = {"profile": {"language": "Korean"}, "custom_prompts": {"a": "b"}}
    with caplog.at_level(logging.WARNING):
        tuned = run_prompt_tuning._with_source_language(result, [KOREAN], Config())
    assert list(tuned) == ["profile", "processing", "custom_prompts"]
    assert tuned["processing"] == {"translation": {"source_language": "ko"}}
    # The default 'en' disagrees: said at tuning time too.
    assert "processing.translation.source_language is 'en'" in caplog.text
    unknown = {"profile": {"language": "Klingon"}, "custom_prompts": {}}
    assert run_prompt_tuning._with_source_language(unknown, [ENGLISH], Config()) == (
        unknown
    )


def test_the_loading_stage_warns_of_another_script(caplog) -> None:
    from unified_kg_rag.application.ingestion.pipeline_stages import (
        DocumentLoadingStage,
    )

    stage = object.__new__(DocumentLoadingStage)
    stage.config = Config()
    documents = [Document(page_content=KOREAN, metadata={})]
    with caplog.at_level(logging.WARNING):
        stage._warn_of_another_script(documents)
    assert "Corpus language:" in caplog.text and "set it to 'ko'" in caplog.text
    caplog.clear()
    stage.config = Config(processing={"translation": {"source_language": "ko"}})
    with caplog.at_level(logging.WARNING):
        stage._warn_of_another_script(documents)
    assert "Corpus language" not in caplog.text
