# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The corpus script check against processing.translation.source_language
(``domain.languages.corpus``), the loading stage's WARNING and the
setting prompt tuning writes."""

from __future__ import annotations

import logging

import pytest
from langchain_core.documents import Document

from unified_kg_rag.application.cli import run_prompt_tuning
from unified_kg_rag.domain.languages.corpus import (
    corpus_language,
    corpus_sample,
    corpus_source_language,
    source_language_mismatch,
)
from unified_kg_rag.domain.models import Config, LanguageCode

pytestmark = pytest.mark.unit

# Synthetic text, long enough to decide (40 letters and more).
KOREAN = "벤더는 구매자에게 부품을 공급하고 납기와 가격을 계약서에 정한다. " * 8
JAPANESE = "ベンダーは購入者に部品を供給し、納期と価格を契約書で定める。" * 8
CHINESE = "供应商向买方供应零件，并在合同中约定交货期和价格。" * 10
ENGLISH = "The vendor supplies parts to the buyer under a fixed price contract. " * 5
# Synthetic contract sentences in scripts and languages beyond the dense ones.
RUSSIAN = "Поставщик обязуется поставить товар покупателю в течение тридцати дней. " * 4
THAI = "ผู้ขายตกลงส่งมอบสินค้าให้แก่ผู้ซื้อภายในสามสิบวันหลังจากลงนามในสัญญา " * 4
ARABIC = "يلتزم البائع بتسليم البضاعة إلى المشتري خلال ثلاثين يوما من توقيع العقد " * 4
GERMAN = (
    "Der Lieferant liefert die Ware an den Käufer und die Rechnung wird mit "
    "der Ware gesendet. "
) * 4
# Korean prose with the Latin terms technical text carries.
KOREAN_TECHNICAL = (
    "벤더는 API 게이트웨이와 Lambda 함수로 주문을 처리하고 S3 버킷에 저장한다. " * 6
)
# Korean in mixed script: its stems in Han (Hanja), more Han than Hangul.
KOREAN_HANJA = (
    "供給者는 購買者에게 部品을 供給하고 納期와 價格은 契約書로 定한다. "
    "契約의 當事者는 法律이 定하는 바에 依하여 義務를 진다. "
) * 3


@pytest.mark.parametrize(
    ("text", "language"),
    [
        (KOREAN, LanguageCode.KO),
        (KOREAN_TECHNICAL, LanguageCode.KO),
        (KOREAN_HANJA, LanguageCode.KO),
        (JAPANESE, LanguageCode.JA),
        (CHINESE, LanguageCode.ZH),
        (ENGLISH, LanguageCode.EN),
        (RUSSIAN, LanguageCode.RU),
        (THAI, LanguageCode.TH),
        (ARABIC, LanguageCode.AR),
        (GERMAN, LanguageCode.DE),
        # Too few letters, or no script holding most of them.
        ("벤더 Vendor 123", None),
        ("벤더는 공급한다 " * 3 + "The vendor supplies the buyer " * 3, None),
    ],
    ids=["ko", "ko-technical", "ko-hanja", "ja", "zh", "en", "ru", "th", "ar", "de"]
    + ["few", "mixed"],
)
def test_the_sample_reads_as_its_language(
    text: str, language: LanguageCode | None
) -> None:
    assert corpus_language(corpus_sample([text])) is language


@pytest.mark.parametrize(
    ("text", "source", "expected"),
    [
        (KOREAN, LanguageCode.EN, "set it to 'ko'"),
        (JAPANESE, LanguageCode.EN, "set it to 'ja'"),
        (CHINESE, LanguageCode.DE, "set it to 'zh'"),
        (ENGLISH, LanguageCode.KO, "only 0% of the sampled letters"),
        (KOREAN, LanguageCode.JA, "reads as Korean"),
        # Regression (rv8 design M3): only a dense-script corpus was judged.
        (RUSSIAN, LanguageCode.EN, "set it to 'ru'"),
        (THAI, LanguageCode.EN, "set it to 'th'"),
        (ARABIC, LanguageCode.EN, "set it to 'ar'"),
        (GERMAN, LanguageCode.EN, "reads as German"),
        (ENGLISH, LanguageCode.DE, "set it to 'en'"),
        (RUSSIAN, LanguageCode.KO, "only 0% of the sampled letters are Hangul"),
        (KOREAN, LanguageCode.RU, "only 0% of the sampled letters are Cyrillic"),
    ],
    ids=["ko-as-en", "ja-as-en", "zh-as-de", "en-as-ko", "ko-as-ja"]
    + ["ru-as-en", "th-as-en", "ar-as-en", "de-as-en", "en-as-de"]
    + ["ru-as-ko", "ko-as-ru"],
)
def test_a_script_the_setting_does_not_name_is_reported(
    text: str, source: LanguageCode, expected: str
) -> None:
    reason = source_language_mismatch(corpus_sample([text]), source)
    assert reason is not None and expected in reason
    assert "processing.translation.source_language" in reason


@pytest.mark.parametrize(
    ("text", "source"),
    [
        (KOREAN, LanguageCode.KO),
        (KOREAN_TECHNICAL, LanguageCode.KO),
        (KOREAN_HANJA, LanguageCode.KO),
        (JAPANESE, LanguageCode.JA),
        (CHINESE, LanguageCode.ZH),
        (ENGLISH, LanguageCode.EN),
        (RUSSIAN, LanguageCode.RU),
        (THAI, LanguageCode.TH),
        (ARABIC, LanguageCode.AR),
        (GERMAN, LanguageCode.DE),
        # Too few letters decide nothing.
        ("벤더 Vendor 123", LanguageCode.EN),
    ],
)
def test_an_agreeing_or_undecidable_sample_is_not_reported(
    text: str, source: LanguageCode
) -> None:
    assert source_language_mismatch(corpus_sample([text]), source) is None


def test_a_sample_read_as_the_setting_is_not_reported() -> None:
    # Regression (rv13 query low): a Korean document that is mostly code
    # holds few Hangul letters but reads as Korean; the warning told a
    # source_language of 'ko' to be set to 'ko'.
    code = (
        "```python\n# List all of the buckets that the caller can access\n"
        "for bucket in s3.buckets.all():\n"
        "    print(bucket.name)  # the name of the bucket\n```\n"
        "참고: The bucket name must be unique across all of AWS and it cannot "
        "be changed after it is created.\n"
    )
    korean = (
        "이 가이드는 Amazon S3 버킷 설정 방법을 설명합니다. 먼저 AWS 콘솔에 "
        "로그인한 뒤 S3 페이지를 엽니다.\n"
    )
    sample = corpus_sample([korean + code] * 5)
    assert corpus_language(sample, LanguageCode.KO) is LanguageCode.KO
    assert sample.counts.dense < 0.2 * sample.counts.letters
    assert source_language_mismatch(sample, LanguageCode.KO) is None


def test_a_two_sentence_corpus_still_decides() -> None:
    # The sample corpora a first run uses are this small.
    tiny = [
        "벤더는 서울에 있는 구매자에게 전자 부품을 공급한다.",
        "구매자는 계약서에 정한 날까지 대금을 지급한다.",
    ]
    reason = source_language_mismatch(corpus_sample(tiny), LanguageCode.EN)
    assert reason is not None and "set it to 'ko'" in reason


def test_the_sample_is_bounded() -> None:
    # The first documents' first characters only: a corpus is not read whole.
    sample = corpus_sample([ENGLISH * 100] * 30 + [KOREAN * 100])
    assert sample.counts.dense == 0
    assert sample.counts.letters <= 20 * 2000


def test_korean_with_hanja_is_not_told_to_be_chinese() -> None:
    sample = corpus_sample([KOREAN_HANJA])
    assert sample.counts["han"] > sample.counts["hangul"]
    assert source_language_mismatch(sample, LanguageCode.KO) is None
    reason = source_language_mismatch(sample, LanguageCode.ZH)
    assert reason is not None and "set it to 'ko'" in reason
    # Prompt tuning writes Korean for it, not Chinese.
    assert corpus_source_language([KOREAN_HANJA], "Chinese") is LanguageCode.KO


def test_the_tuned_language_is_the_samples_else_the_profiles() -> None:
    assert corpus_source_language([KOREAN], "English") is LanguageCode.KO
    assert corpus_source_language([GERMAN], "English") is LanguageCode.DE
    assert corpus_source_language([RUSSIAN], "English") is LanguageCode.RU
    # A sample the rule cannot read: the profile's language.
    undecided = "Vendor Buyer Acme Robotics Orion Labs " * 5
    assert corpus_source_language([undecided], "German") is LanguageCode.DE
    assert corpus_source_language([undecided], "fr") is LanguageCode.FR
    assert corpus_source_language([undecided], "Klingon") is None


def test_prompt_tuning_writes_the_source_language(caplog) -> None:
    result = {"profile": {"language": "Korean"}, "custom_prompts": {"a": "b"}}
    with caplog.at_level(logging.WARNING):
        tuned = run_prompt_tuning._with_source_language(result, [KOREAN], Config())
    assert list(tuned) == ["profile", "processing", "custom_prompts"]
    assert tuned["processing"] == {"translation": {"source_language": "ko"}}
    # The default 'en' disagrees: said at tuning time too.
    assert "processing.translation.source_language is 'en'" in caplog.text
    unknown = {"profile": {"language": "Klingon"}, "custom_prompts": {}}
    undecided = ["Vendor Buyer Acme Robotics Orion Labs " * 5]
    assert run_prompt_tuning._with_source_language(unknown, undecided, Config()) == (
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
