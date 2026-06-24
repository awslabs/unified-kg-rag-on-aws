# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TextUnitTranslator (AWS-free).

The translator builds a Bedrock LCEL chain via ``setup_chain`` in its
constructor. That factory (and the underlying boto/Bedrock factory) is patched
out and replaced with a deterministic fake chain whose ``.batch`` returns canned
strings. The real per-language fan-out loop, batch input preparation, result
mapping back onto ``TextUnit.translated_texts`` and stat accounting are
exercised. The real ``BatchProcessor`` is used end-to-end (it is pure Python).
"""

from __future__ import annotations

import pytest

from aws_graphrag.adapters.ingestion import translator as tr_module
from aws_graphrag.adapters.ingestion.translator import (
    TextUnitTranslator,
    TranslationStats,
)
from aws_graphrag.domain.models import Config, LanguageCode, TextUnit

pytestmark = pytest.mark.unit


class _FakeChain:
    """Fake LCEL chain: ``batch`` echoes a per-input translation."""

    def __init__(self, transform=None) -> None:
        # transform maps an input dict -> output string.
        self.transform = transform or (
            lambda inp: f"[{inp['target_language'].value}] {inp['text']}"
        )
        self.batch_calls: list[list[dict]] = []

    def batch(self, inputs, *args, **kwargs):
        self.batch_calls.append(inputs)
        return [self.transform(inp) for inp in inputs]

    def invoke(self, inp, *args, **kwargs):
        return self.transform(inp)


def _make_translator(
    mocker, *, target=LanguageCode.EN, additional=None, transform=None
) -> tuple[TextUnitTranslator, _FakeChain]:
    mocker.patch.object(tr_module.boto3, "Session")
    mocker.patch.object(tr_module, "BedrockLanguageModelFactory")
    fake_chain = _FakeChain(transform=transform)
    mocker.patch.object(tr_module, "setup_chain", return_value=fake_chain)

    config = Config()
    config.processing.translation.target_language = target
    config.processing.translation.additional_target_languages = additional
    config.processing.batch_size = 100
    translator = TextUnitTranslator(config, show_progress=False)
    return translator, fake_chain


def _units() -> list[TextUnit]:
    return [
        TextUnit(id="t1", text="Hello world"),
        TextUnit(id="t2", text="Goodbye world"),
    ]


class TestCreateChainInputs:
    def test_pairs_text_with_target_language(self) -> None:
        inputs = TextUnitTranslator._create_chain_inputs(["a", "b"], LanguageCode.KO)
        assert inputs == [
            {"text": "a", "target_language": LanguageCode.KO},
            {"text": "b", "target_language": LanguageCode.KO},
        ]

    def test_empty_texts_yields_empty(self) -> None:
        assert TextUnitTranslator._create_chain_inputs([], LanguageCode.EN) == []


class TestApplyTranslationResult:
    def test_successful_result_populates_dict(self, mocker) -> None:
        tr, _ = _make_translator(mocker)
        tr.stats = TranslationStats(num_total_units=1)
        unit = TextUnit(id="t1", text="x")
        tr._apply_translation_result(unit, "  bonjour  ", LanguageCode.FR)
        assert unit.translated_texts == {"fr": "bonjour"}
        assert tr.stats.num_successful_translations == 1
        assert tr.stats.num_translated == 1

    def test_empty_result_counts_as_failure(self, mocker) -> None:
        tr, _ = _make_translator(mocker)
        tr.stats = TranslationStats(num_total_units=1)
        unit = TextUnit(id="t1", text="x")
        tr._apply_translation_result(unit, "   ", LanguageCode.FR)
        assert unit.translated_texts is None
        assert tr.stats.num_failed_translations == 1
        assert tr.stats.num_successful_translations == 0

    def test_none_result_counts_as_failure(self, mocker) -> None:
        tr, _ = _make_translator(mocker)
        tr.stats = TranslationStats(num_total_units=1)
        unit = TextUnit(id="t1", text="x")
        tr._apply_translation_result(unit, None, LanguageCode.FR)
        assert tr.stats.num_failed_translations == 1

    def test_preserves_existing_translations(self, mocker) -> None:
        tr, _ = _make_translator(mocker)
        tr.stats = TranslationStats(num_total_units=2)
        unit = TextUnit(id="t1", text="x", translated_texts={"en": "hi"})
        tr._apply_translation_result(unit, "salut", LanguageCode.FR)
        assert unit.translated_texts == {"en": "hi", "fr": "salut"}


class TestTranslateTextUnits:
    def test_empty_input_returns_input_unchanged(self, mocker) -> None:
        tr, fake = _make_translator(mocker)
        out = tr.translate_text_units([])
        assert out == []
        assert fake.batch_calls == []  # nothing dispatched

    def test_single_language_maps_results_back(self, mocker) -> None:
        tr, fake = _make_translator(mocker, target=LanguageCode.KO)
        units = _units()
        out = tr.translate_text_units(units)
        assert out is units
        assert units[0].translated_texts == {"ko": "[ko] Hello world"}
        assert units[1].translated_texts == {"ko": "[ko] Goodbye world"}
        # One language -> one batch dispatch with both texts.
        assert len(fake.batch_calls) == 1
        assert [i["text"] for i in fake.batch_calls[0]] == [
            "Hello world",
            "Goodbye world",
        ]

    def test_multi_language_fan_out(self, mocker) -> None:
        tr, fake = _make_translator(
            mocker,
            target=LanguageCode.EN,
            additional=[LanguageCode.KO, LanguageCode.JA],
        )
        units = _units()
        tr.translate_text_units(units)
        # All three languages applied to each unit.
        assert set(units[0].translated_texts) == {"en", "ko", "ja"}
        assert units[0].translated_texts["ja"] == "[ja] Hello world"
        # One batch dispatch per language.
        assert len(fake.batch_calls) == 3

    def test_stats_total_counts_units_times_languages(self, mocker) -> None:
        tr, _ = _make_translator(
            mocker, target=LanguageCode.EN, additional=[LanguageCode.KO]
        )
        tr.translate_text_units(_units())  # 2 units * 2 langs
        assert tr.stats.num_total_units == 4
        assert tr.stats.num_successful_translations == 4
        assert tr.stats.total_processing_time >= 0.0

    def test_empty_translation_recorded_as_failure(self, mocker) -> None:
        # Chain returns empty string for the second unit only.
        def transform(inp):
            return "" if inp["text"] == "Goodbye world" else "ok"

        tr, _ = _make_translator(mocker, target=LanguageCode.KO, transform=transform)
        units = _units()
        tr.translate_text_units(units)
        assert units[0].translated_texts == {"ko": "ok"}
        assert units[1].translated_texts is None
        assert tr.stats.num_successful_translations == 1
        assert tr.stats.num_failed_translations == 1


class TestTranslationStats:
    def test_success_rate_zero_when_no_units(self) -> None:
        assert TranslationStats().success_rate == 0.0

    def test_success_rate_percentage(self) -> None:
        s = TranslationStats(num_total_units=4, num_successful_translations=3)
        assert s.success_rate == pytest.approx(75.0)

    def test_processed_unit_count(self) -> None:
        s = TranslationStats(num_successful_translations=2, num_failed_translations=1)
        assert s.processed_unit_count == 3

    def test_average_processing_time_guards_zero(self) -> None:
        assert TranslationStats().average_processing_time == 0.0

    def test_average_processing_time(self) -> None:
        s = TranslationStats(
            num_successful_translations=2,
            num_failed_translations=2,
            total_processing_time=8.0,
        )
        assert s.average_processing_time == pytest.approx(2.0)
