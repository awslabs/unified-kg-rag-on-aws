# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The corpus script check against processing.translation.source_language
(``domain.languages.corpus``), the loading stage's WARNING and the
setting prompt tuning writes."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.documents import Document

from unified_kg_rag.application.cli import run_prompt_tuning
from unified_kg_rag.domain.languages.corpus import (
    corpus_language,
    corpus_sample,
    corpus_source_language,
    sample_split,
    source_language_mismatch,
)
from unified_kg_rag.domain.models import Config, DocumentDelta, LanguageCode

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
        # Half of what the letters carry in each (20 Hangul syllables, 80
        # Latin letters: 20 tokens each).
        (
            "벤더는 부품을 공급한다 " * 2
            + "The vendors supply all the parts to their buyers " * 2,
            None,
        ),
        # Most of it Hangul by what the letters carry, though most letters
        # are Latin: the group and its majority are one measure (rv26; the
        # group was weighed in tokens, its majority in letters, so no vote).
        (
            "벤더는 부품을 공급하고 납기를 정한다 " * 2
            + "The vendor supplies parts " * 3,
            LanguageCode.KO,
        ),
    ],
    ids=["ko", "ko-technical", "ko-hanja", "ja", "zh", "en", "ru", "th", "ar", "de"]
    + ["few", "mixed", "token-majority"],
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
        (ENGLISH, LanguageCode.KO, "only 0% of the sampled text"),
        (KOREAN, LanguageCode.JA, "reads as Korean"),
        # Regression (rv8 design M3): only a dense-script corpus was judged.
        (RUSSIAN, LanguageCode.EN, "set it to 'ru'"),
        (THAI, LanguageCode.EN, "set it to 'th'"),
        (ARABIC, LanguageCode.EN, "set it to 'ar'"),
        (GERMAN, LanguageCode.EN, "reads as German"),
        (ENGLISH, LanguageCode.DE, "set it to 'en'"),
        (RUSSIAN, LanguageCode.KO, "only 0% of the sampled text is Hangul"),
        (KOREAN, LanguageCode.RU, "only 0% of the sampled text is Cyrillic"),
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


def test_a_mostly_english_sample_is_reported_with_the_language_it_reads_as() -> None:
    # rv13 query low: a Korean document that is mostly code and English
    # notes told a source_language of 'ko' to be set to 'ko'. Since rv24
    # (quality 2) a document votes for the script group most of its letters
    # are in, and for a language only of that group: this one is Latin, so
    # the warning names English, never the setting itself. Trade-off: two
    # Korean sentences in a document of English code and notes no longer
    # outweigh its English.
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
    assert corpus_language(sample, LanguageCode.KO) is LanguageCode.EN
    assert sample.counts.dense < 0.2 * sample.counts.letters
    reason = source_language_mismatch(sample, LanguageCode.KO)
    assert reason is not None and "set it to 'en'" in reason
    # Korean prose holding most of the letters still reads Korean.
    prose = corpus_sample([korean * 8 + code] * 5)
    assert corpus_language(prose, LanguageCode.KO) is LanguageCode.KO
    assert source_language_mismatch(prose, LanguageCode.KO) is None


def test_a_two_sentence_corpus_still_decides() -> None:
    # The sample corpora a first run uses are this small.
    tiny = [
        "벤더는 서울에 있는 구매자에게 전자 부품을 공급한다.",
        "구매자는 계약서에 정한 날까지 대금을 지급한다.",
    ]
    reason = source_language_mismatch(corpus_sample(tiny), LanguageCode.EN)
    assert reason is not None and "set it to 'ko'" in reason


def test_a_short_dense_document_votes_by_what_its_letters_carry() -> None:
    # Regression (rv25 algo-index low): the threshold counted letters, so a
    # Han character weighed as one Latin letter and a short Chinese document
    # of 38 characters cast no vote. It counts tokens now (ScriptCounts.
    # carried, the token estimate's rates): ten Han characters carry what
    # forty Latin letters do.
    short = "供应商向买方供应零件，并在合同中约定交货期和价格。买方按期付款，供应商开具发票。"
    sample = corpus_sample([short])
    assert sample.counts.letters < 40 and sample.counts.carried >= 10
    assert corpus_language(sample, None) is LanguageCode.ZH
    # Fewer than ten tokens' worth still decide nothing, in any script.
    assert corpus_language(corpus_sample(["供应商向买方"]), None) is None
    assert corpus_language(corpus_sample(["The vendor ships."]), None) is None


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


def test_prompt_tuning_writes_the_language_most_documents_are_in(caplog) -> None:
    # Regression (rv23 surfaces 2): four English documents and one Korean one
    # were tuned as Korean (prompts and analyzers), and the loading stage
    # told an English setting to be 'ko'.
    english, korean = [
        (text * (2000 // len(text) + 1))[:2000] for text in (ENGLISH, KOREAN)
    ]
    texts = [english] * 4 + [korean]
    result = {"profile": {"language": "Korean"}, "custom_prompts": {}}
    with caplog.at_level(logging.WARNING):
        tuned = run_prompt_tuning._with_source_language(result, texts, Config())
    assert tuned["processing"] == {"translation": {"source_language": "en"}}
    assert source_language_mismatch(corpus_sample(texts), LanguageCode.EN) is None
    assert "processing.translation.source_language" not in caplog.text


def test_the_loading_stage_warns_of_another_script(caplog) -> None:
    from unified_kg_rag.application.ingestion.pipeline_stages import (
        DocumentLoadingStage,
    )

    stage = object.__new__(DocumentLoadingStage)
    stage.config = Config()
    documents = [Document(page_content=KOREAN, metadata={})]
    full_build = SimpleNamespace(incremental_delta=None)
    with caplog.at_level(logging.WARNING):
        stage._warn_of_another_script(documents, full_build)  # type: ignore[arg-type]
    assert "Corpus language:" in caplog.text and "set it to 'ko'" in caplog.text
    caplog.clear()
    stage.config = Config(processing={"translation": {"source_language": "ko"}})
    with caplog.at_level(logging.WARNING):
        stage._warn_of_another_script(documents, full_build)  # type: ignore[arg-type]
    assert "Corpus language" not in caplog.text


class _StoredUnits:
    """A vector store holding one page of English text units (two per
    document), or failing to read them."""

    def __init__(self, fails: bool = False) -> None:
        self.fails = fails
        self.page_sizes: list[int | None] = []

    def scan_text_unit_pages(
        self, suffix: str | None, *, fields: list[str], page_size: int | None = None
    ) -> Any:
        self.page_sizes.append(page_size)
        if self.fails:
            raise ConnectionError("store unavailable")
        rows = [
            {"id": f"u{i}", "text": ENGLISH, "document_ids": [f"d{i // 2}"]}
            for i in range(200)
        ]
        return iter([rows[: page_size or len(rows)]])

    def close(self) -> None:
        return None


@pytest.mark.parametrize("fails", [False, True])
def test_an_incremental_run_samples_the_stored_corpus_too(caplog, fails: bool) -> None:
    # rv26 algo-index low: a delta of one short Korean document in an
    # English corpus was the whole sample, and the run was told to set
    # source_language to 'ko'. The unchanged documents are sampled from
    # their stored text, each part its share of the corpus; a failed read
    # falls back to the run's documents.
    from unified_kg_rag.application.ingestion.pipeline_stages import (
        DocumentLoadingStage,
    )

    stage = object.__new__(DocumentLoadingStage)
    stage.config = Config()
    store = _StoredUnits(fails)
    stage.providers = SimpleNamespace(build_vector_indexer=lambda: store)  # type: ignore[assignment]
    short = "벤더는 부품을 공급하고 납기와 가격을 계약서에 정한다."
    delta = DocumentDelta(new=["n1"], unchanged=[f"d{i}" for i in range(500)])
    context = SimpleNamespace(incremental_delta=delta)
    with caplog.at_level(logging.WARNING):
        stage._warn_of_another_script(
            [Document(page_content=short, metadata={})], context  # type: ignore[arg-type]
        )
    assert ("set it to 'ko'" in caplog.text) is fails, caplog.text
    assert store.page_sizes == [5 * 19]
    assert sample_split(1, 500) == (1, 19)
    assert sample_split(30, 0) == (20, 0)
    assert sample_split(0, 3) == (0, 3)
    assert sample_split(10, 10) == (10, 10)
