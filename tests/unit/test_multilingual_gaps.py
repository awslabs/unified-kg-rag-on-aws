# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multilingual / heterogeneous-corpus robustness (AWS-free).

Three audited gaps:

GAP 1 — DRIFT keyword expansion / query refinement must be language-aware so a
        non-English corpus is not expanded with English-only keywords that miss
        the language-analyzed index. ``KeywordExpansionPrompt`` and
        ``QueryRefinementPrompt`` now thread ``target_language``, mirroring the
        LightRAG ``KeywordsExtractionPrompt`` / answer prompts.
GAP 2 — non-UTF-8 source files (e.g. cp949) must be recovered via encoding
        autodetection rather than silently dropped. The parser passes
        ``autodetect_encoding=True`` to the text/CSV loaders, and
        ``Document.from_json_file`` reads via a charset-detecting helper.
GAP 3 — the translation stage gains an ``enabled`` flag and a same-language
        no-op skip so an EN->EN corpus pays no LLM cost.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aws_graphrag.adapters.ingestion.parser import ParserFactory
from aws_graphrag.application.ingestion.pipeline_stages import TranslationStage
from aws_graphrag.domain.models import (
    Config,
    Document,
    LanguageCode,
    PipelineContext,
    TextUnit,
)
from aws_graphrag.domain.models.config import TranslationConfig
from aws_graphrag.domain.models.document import _read_text_autodetect
from aws_graphrag.domain.prompts import KeywordExpansionPrompt, QueryRefinementPrompt

pytestmark = pytest.mark.unit


# A multi-sentence Korean sample; long enough for reliable encoding detection.
_KOREAN_TEXT = (
    "안녕하세요 세계입니다. 한국어 문서를 인코딩 자동 감지로 읽습니다. "
    "추가 문장을 넣어 표본을 충분히 늘려 통계적 감지를 안정화합니다."
)


# --- GAP 1: language-aware DRIFT prompts -------------------------------------


class TestLanguageAwareDriftPrompts:
    def test_keyword_expansion_declares_target_language(self) -> None:
        assert "target_language" in KeywordExpansionPrompt.input_variables

    def test_query_refinement_declares_target_language(self) -> None:
        assert "target_language" in QueryRefinementPrompt.input_variables

    def test_keyword_expansion_renders_with_target_language(self) -> None:
        # resolve() validates the shipped default's variables against the
        # template, so a missing {target_language} slot would raise here.
        resolved = KeywordExpansionPrompt.resolve()
        assert "target_language" in resolved.input_variables
        combined = resolved.system_prompt_template + resolved.human_prompt_template
        assert "{target_language}" in combined

    def test_query_refinement_renders_with_target_language(self) -> None:
        resolved = QueryRefinementPrompt.resolve()
        assert "target_language" in resolved.input_variables
        combined = resolved.system_prompt_template + resolved.human_prompt_template
        assert "{target_language}" in combined

    def test_keyword_expansion_formats_with_a_concrete_language(self) -> None:
        resolved = KeywordExpansionPrompt.resolve()
        rendered = resolved.human_prompt_template.format(
            query="클라우드 아키텍처",
            entities=["AWS"],
            topics=[],
            max_keywords=20,
            target_language="Korean",
        )
        assert "Korean" in rendered


# --- GAP 2: non-UTF-8 file recovery ------------------------------------------


class TestNonUtf8Recovery:
    def test_read_text_autodetect_recovers_cp949(self, tmp_path: Path) -> None:
        p = tmp_path / "korean.txt"
        p.write_bytes(_KOREAN_TEXT.encode("cp949"))
        assert _read_text_autodetect(p) == _KOREAN_TEXT

    def test_read_text_autodetect_passes_through_utf8(self, tmp_path: Path) -> None:
        p = tmp_path / "utf8.txt"
        p.write_text(_KOREAN_TEXT, encoding="utf-8")
        assert _read_text_autodetect(p) == _KOREAN_TEXT

    def test_parser_loads_cp949_text_file(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.txt"
        p.write_bytes(_KOREAN_TEXT.encode("cp949"))
        parser = ParserFactory.create_parser(p, Config())
        document = parser.parse_file(p)
        # The whole file would otherwise be dropped on UnicodeDecodeError; with
        # autodetect it parses and the Korean content survives.
        assert "한국어" in document.content.text

    def test_csv_loader_recovers_cp949(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.csv"
        p.write_bytes(("name,desc\nAWS,클라우드 서비스 제공자\n").encode("cp949"))
        parser = ParserFactory.create_parser(p, Config())
        document = parser.parse_file(p)
        assert "클라우드" in document.content.text

    def test_text_loaders_are_encoding_aware(self) -> None:
        # The .txt/.csv loaders are eligible for the detect-and-retry path.
        from aws_graphrag.adapters.ingestion.parser import _ENCODING_AWARE_LOADERS

        for ext in (".txt", ".csv"):
            loader_class = ParserFactory._loader_configs[ext][0]
            assert issubclass(loader_class, _ENCODING_AWARE_LOADERS)

    def test_document_from_json_file_reads_cp949(self, tmp_path: Path) -> None:
        # Build a real exported Document JSON, re-encode it as cp949, and confirm
        # from_json_file recovers it (it previously hardcoded encoding="utf-8").
        from aws_graphrag.domain.models.document import DocumentContent

        doc = Document(
            document_id="d1",
            file_name="korean.txt",
            file_path="korean.txt",
            file_type="txt",
            total_pages=1,
            page_content=_KOREAN_TEXT,
            content=DocumentContent(text=_KOREAN_TEXT),
        )
        json_str = doc.to_json_file()
        p = tmp_path / "doc.json"
        p.write_bytes(json_str.encode("cp949"))
        loaded = Document.from_json_file(p)
        assert loaded.content.text == _KOREAN_TEXT


# --- GAP 3: translation enable flag + same-language no-op --------------------


class TestTranslationConfigGate:
    def test_enabled_defaults_true(self) -> None:
        assert TranslationConfig().enabled is True

    def test_same_language_no_additional_is_noop(self) -> None:
        cfg = TranslationConfig(
            source_language=LanguageCode.EN, target_language=LanguageCode.EN
        )
        assert cfg.is_noop is True

    def test_different_language_is_not_noop(self) -> None:
        cfg = TranslationConfig(
            source_language=LanguageCode.KO, target_language=LanguageCode.EN
        )
        assert cfg.is_noop is False

    def test_same_language_with_additional_is_not_noop(self) -> None:
        cfg = TranslationConfig(
            source_language=LanguageCode.EN,
            target_language=LanguageCode.EN,
            additional_target_languages=[LanguageCode.KO],
        )
        assert cfg.is_noop is False


class TestTranslationStageSkip:
    @staticmethod
    def _stage(config: Config) -> TranslationStage:
        # Pass a sentinel boto_session so the base __init__ builds no real
        # boto3.Session (AWS-free).
        return TranslationStage(config, boto_session=object())  # type: ignore[arg-type]

    @staticmethod
    def _context(config: Config) -> PipelineContext:
        from datetime import datetime

        from aws_graphrag.domain.models import PipelineStageStatus
        from aws_graphrag.domain.models.config import PipelineConfig

        ctx = PipelineContext(
            pipeline_id="p1",
            config=PipelineConfig(),
            status=PipelineStageStatus.RUNNING,
            start_time=datetime.now(),
            source_directory=Path("."),
        )
        ctx.text_units = [
            TextUnit(id="t1", text="Hello world"),
            TextUnit(id="t2", text="Goodbye world"),
        ]
        return ctx

    def test_skips_when_disabled(self) -> None:
        config = Config()
        config.processing.translation.enabled = False
        stage = self._stage(config)
        assert stage._should_skip() is not None
        # No translator is built for a skipped stage.
        assert stage._translator is None

        ctx = self._context(config)
        in_count, out_count, metrics = stage._execute_core(ctx)
        assert metrics["skipped"] is True
        assert ctx.translated_units == []
        assert stage._translator is None  # still lazy / never constructed
        # Output count equals input so the critical-stage check passes.
        assert out_count == in_count == len(ctx.text_units)

    def test_skips_same_language_noop(self) -> None:
        config = Config()
        config.processing.translation.source_language = LanguageCode.EN
        config.processing.translation.target_language = LanguageCode.EN
        config.processing.translation.additional_target_languages = None
        stage = self._stage(config)
        assert stage._should_skip() is not None

        ctx = self._context(config)
        _, _, metrics = stage._execute_core(ctx)
        assert metrics["skipped"] is True
        assert stage._translator is None

    def test_runs_when_languages_differ(self) -> None:
        config = Config()
        config.processing.translation.source_language = LanguageCode.KO
        config.processing.translation.target_language = LanguageCode.EN
        stage = self._stage(config)
        # Not skipped: a real (KO->EN) run would proceed.
        assert stage._should_skip() is None
