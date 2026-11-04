# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A model answer that corrects itself is parsed by its last complete answer.

The model writes an answer, a line such as "Correction: ... the corrected,
final output:", and the answer again. Parsed as one response, every XML
section became a list of two (graph extraction failed the unit as cut-off
output) and the JSON span from the first ``{`` to the last ``}`` did not
parse ("Extra data": the whole mix/hybrid query failed). A single answer cut
off is still flagged.
"""

from __future__ import annotations

import json
import logging

import pytest
from langchain_core.exceptions import OutputParserException

import unified_kg_rag.adapters.ingestion.graph_extractor as ge_module
from tests.fixtures import self_corrected_outputs as outputs
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.ingestion.claim_extractor import ClaimsOutputParser
from unified_kg_rag.adapters.ingestion.claim_extractor import _checked as claims_checked
from unified_kg_rag.adapters.ingestion.graph_extractor import (
    GraphExtractionOutputParser,
    GraphExtractor,
    _checked,
)
from unified_kg_rag.adapters.llm.langchain import (
    LLMJsonOutputParser,
    RobustXMLOutputParser,
)
from unified_kg_rag.domain.models import Config, TextUnit
from unified_kg_rag.shared.exceptions import LLMOutputTruncatedError
from unified_kg_rag.shared.utils import ensure_list, parse_llm_json

pytestmark = pytest.mark.unit

_EXTRACTION_TAGS = ["entities", "relationships"]
_REPORT_TAGS = ["community_name", "summary", "rating", "rating_explanation", "findings"]


def _extraction_parser() -> GraphExtractionOutputParser:
    return GraphExtractionOutputParser(tags=_EXTRACTION_TAGS)


def _extractor(config: Config, mocker, **answers: list[str]) -> GraphExtractor:
    """A GraphExtractor whose model answers each unit (by its text, as the
    keyword) with its ``answers`` in turn, parsed by the real extraction
    parser; retried at once."""
    mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    mocker.patch.object(ge_module, "create_robust_xml_output_parser")
    mocker.patch.object(ge_module, "setup_chain")
    extractor = GraphExtractor(config)
    extractor.extraction_config.entity_confidence_threshold = 0.0
    extractor.batch_processor.retry_max_wait = 0
    parser = _extraction_parser()
    replies = {text: iter(texts) for text, texts in answers.items()}
    extractor.graph_extractor.invoke = mocker.Mock(
        side_effect=lambda request: parser.parse(next(replies[request["input_text"]]))
    )
    return extractor


def _types(result: dict) -> list[str]:
    return [
        rel["type"]
        for rel in ensure_list(result["relationships"], inner_key="relationship")
    ]


class TestXMLSelfCorrection:
    def test_graph_extraction_parses_the_final_answer(self) -> None:
        result = _checked(_extraction_parser().parse(outputs.EXTRACTION_SELF_CORRECTED))
        names = [e["name"] for e in ensure_list(result["entities"], inner_key="entity")]
        assert names == ["Vendor Corp", "Harbor City", "Buyer Ltd", "Depot Inc"]
        assert _types(result) == outputs.FINAL_RELATIONSHIP_TYPES

    def test_the_extractor_records_the_unit_instead_of_failing_it(
        self, config: Config, mocker
    ) -> None:
        extractor = _extractor(config, mocker, unit=[outputs.EXTRACTION_SELF_CORRECTED])
        unit = TextUnit(id="t1", text="unit")
        records, stats = extractor.extract_units([unit])
        assert stats.failed_text_unit_ids == [] and stats.num_failed_extractions == 0
        assert len(records) == 1 and len(records[0].entities) == 4
        assert sorted(r.type for r in records[0].relationships) == sorted(
            outputs.FINAL_RELATIONSHIP_TYPES
        )

    def test_a_complete_answer_without_names_is_retried_as_malformed(
        self, config: Config, mocker, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Its sections are closed, so it was not cut off: a new attempt can
        # answer (the E2E's units with this error all succeeded on a re-run).
        config.retry.max_attempts = 2
        extractor = _extractor(
            config,
            mocker,
            unit=[outputs.EXTRACTION_UNNAMED, outputs.EXTRACTION_SELF_CORRECTED],
        )
        with pytest.raises(OutputParserException, match="malformed output"):
            _checked(_extraction_parser().parse(outputs.EXTRACTION_UNNAMED))
        unit = TextUnit(id="t1", text="unit")
        with caplog.at_level(logging.DEBUG, logger=ge_module.__name__):
            records, stats = extractor.extract_units([unit])
        assert extractor.graph_extractor.invoke.call_count == 2
        assert stats.failed_text_unit_ids == [] and len(records[0].entities) == 4
        # The response that failed is logged at DEBUG, for diagnosis.
        assert any(
            "Extraction output that failed" in r.getMessage()
            and "<entity_name>" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.DEBUG
        )

    def test_cut_off_output_is_not_retried(self, config: Config, mocker) -> None:
        config.retry.max_attempts = 2
        extractor = _extractor(
            config,
            mocker,
            cut=[outputs.EXTRACTION_TRUNCATED, outputs.EXTRACTION_SELF_CORRECTED],
            whole=[outputs.EXTRACTION_SELF_CORRECTED],
        )
        units = [TextUnit(id="t1", text="cut"), TextUnit(id="t2", text="whole")]
        _, stats = extractor.extract_units(units)
        assert extractor.graph_extractor.invoke.call_count == 2
        assert stats.failed_text_unit_ids == ["t1"]
        assert stats.item_failed_text_unit_ids == ["t1"]

    def test_a_single_truncated_answer_is_still_cut_off_output(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        text = outputs.EXTRACTION_TRUNCATED
        assert _extraction_parser().final_answer(text) == text
        with caplog.at_level(logging.WARNING):
            result = _extraction_parser().parse(text)
        assert "XML output ends with <entities> unclosed" in caplog.text
        with pytest.raises(LLMOutputTruncatedError, match="none of the 1 entities"):
            _checked(result)

    def test_a_correction_cut_off_keeps_the_last_complete_answer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            result = _extraction_parser().parse(outputs.EXTRACTION_CORRECTION_CUT)
        assert "its last <relationships> is unclosed" in caplog.text
        assert _types(_checked(result)) == outputs.FINAL_RELATIONSHIP_TYPES

    @pytest.mark.parametrize(
        "text",
        [
            outputs.EXTRACTION_TRUNCATED,
            outputs.REPORT,
            # The stage tests' one-line report: nested <summary> on its line.
            outputs.REPORT.replace("\n", ""),
            # Prose naming a section tag mid-line is not a second section.
            outputs.EXTRACTION_SELF_CORRECTED.split(outputs.CORRECTION_LINE)[0]
            + "Note: the <relationships> above are complete.\n",
            # A stray end tag of a section that never opened.
            "<line_number>4</line_number>\n</chunk_boundaries>",
        ],
    )
    def test_a_response_repeating_no_section_is_unchanged(self, text: str) -> None:
        tags = [*_EXTRACTION_TAGS, *_REPORT_TAGS, "chunk_boundaries"]
        assert RobustXMLOutputParser(tags=tags).final_answer(text) == text

    def test_a_failed_claims_check_logs_the_response_at_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        text = "<claims>\n<claim>\n<subject>Vendor Corp</subject>\n<obj"
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(LLMOutputTruncatedError):
                claims_checked(ClaimsOutputParser(tags=["claims"]).parse(text))
        assert any(
            r.levelno == logging.DEBUG and text in r.getMessage()
            for r in caplog.records
        )

    def test_claims_parse_the_final_answer(self) -> None:
        result = claims_checked(
            ClaimsOutputParser(tags=["claims"]).parse(outputs.CLAIMS_SELF_CORRECTED)
        )
        claims = ensure_list(result["claims"], inner_key="claim")
        assert [c["claim_type"] for c in claims] == ["OBLIGATION", "PAYMENT"]

    def test_gleaning_parses_the_final_plan(self) -> None:
        result = RobustXMLOutputParser(tags=["refinement_plan"]).parse(
            outputs.GLEANING_SELF_CORRECTED
        )
        plan = result["refinement_plan"]
        assert isinstance(plan, dict)
        issues = ensure_list(plan["identified_issues"], inner_key="issue")
        assert [i["details"]["name"] for i in issues] == ["Depot Inc", "Harbor City"]

    def test_a_community_report_parses_the_final_report(self) -> None:
        result = RobustXMLOutputParser(tags=_REPORT_TAGS).parse(
            outputs.REPORT_SELF_CORRECTED
        )
        assert result["community_name"] == "Supply partners"
        assert result["summary"] == "Vendor Corp supplies Buyer Ltd."
        assert result["findings"]["finding"]["summary"] == "Vendor Corp is central"


class TestJSONSelfCorrection:
    def test_the_last_object_is_the_answer(self) -> None:
        # Before: the span from the first { to the last } held both objects.
        with pytest.raises(json.JSONDecodeError, match="Extra data: line 3 column 1"):
            json.loads(
                outputs.KEYWORDS_SELF_CORRECTED[
                    : outputs.KEYWORDS_SELF_CORRECTED.rindex("}") + 1
                ]
            )
        assert (
            parse_llm_json(outputs.KEYWORDS_SELF_CORRECTED, strict=True)
            == outputs.FINAL_KEYWORDS
        )

    def test_fenced_answers_take_the_last_one(self) -> None:
        raw = (
            '```json\n{"a": 1}\n```\nCorrection: final output:\n```json\n{"a": 2}\n```'
        )
        assert parse_llm_json(raw, strict=True) == {"a": 2}

    def test_the_only_object_that_parses_is_the_answer(self) -> None:
        # The correction was cut off: the first answer is the only one.
        raw = '{"a": 1}\n\nCorrection: final output:\n\n{"a": [2'
        assert parse_llm_json(raw, strict=True) == {"a": 1}

    def test_a_single_truncated_object_still_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json(
                '{"points": [{"description": "x", "score": 9}, {"desc', strict=True
            )
        assert parse_llm_json('{"points": [{"description": "x", "score": 9}, {"d') == {}

    def test_an_object_nested_in_one_that_does_not_parse_is_not_the_answer(
        self,
    ) -> None:
        raw = '{"strategy": "local", "params": {"k": 3},} trailing'
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json(raw, strict=True)

    def test_braces_inside_strings_do_not_split_objects(self) -> None:
        raw = '{"a": "x } {"}\nCorrection:\n{"a": "y { }"}'
        assert parse_llm_json(raw, strict=True) == {"a": "y { }"}

    def test_the_judge_parser_reads_the_last_object(self) -> None:
        parser = LLMJsonOutputParser()
        raw = '{"directness": {"winner": "1"}}\n\nCorrection:\n\n{"directness": {"winner": "2"}}'
        assert parser.parse(raw) == {"directness": {"winner": "2"}}
        with pytest.raises(OutputParserException):
            parser.parse("no verdict")

    def test_the_pairwise_judge_parses_with_it(self, mocker) -> None:
        from unified_kg_rag.evaluation import pairwise

        setup = mocker.patch.object(pairwise, "setup_chain")
        mocker.patch.object(pairwise.Providers, "resolve")
        pairwise.PairwiseJudge(Config())
        assert isinstance(setup.call_args.args[3], LLMJsonOutputParser)
