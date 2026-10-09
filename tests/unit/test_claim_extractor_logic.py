# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ClaimExtractor pure logic (AWS-free).

Covers _get_string_value coercion, _parse_claim_data (missing-field rejection,
stable id, attributes), _parse_extraction_result (malformed/empty handling),
_merge_claims, and _prepare_extraction_inputs returning a dict keyed by unit id
(the as_completed ordering fix). Bedrock/boto wiring is patched out so __init__
never touches AWS, and a single-worker thread pool keeps the executor cheap.
"""

from __future__ import annotations

import logging

import pytest

import unified_kg_rag.adapters.ingestion.claim_extractor as ce_module
from unified_kg_rag.adapters import providers as providers_module
from unified_kg_rag.adapters.ingestion.claim_extractor import (
    ClaimExtractionStats,
    ClaimExtractor,
    _prepare_claim_input_task,
    format_entities_with_limit_task,
)
from unified_kg_rag.domain.models import Claim, Config, Entity, TextUnit

pytestmark = pytest.mark.unit


@pytest.fixture
def extractor(config: Config, mocker) -> ClaimExtractor:
    mocker.patch.object(ce_module, "boto3")
    mocker.patch.object(providers_module, "BedrockLanguageModelFactory")
    mocker.patch.object(ce_module, "create_robust_xml_output_parser")
    mocker.patch.object(ce_module, "setup_chain")
    # Thread pool (not process pool) so patched module globals are inherited and
    # no real subprocess/AWS work happens.
    return ClaimExtractor(config, use_process_pool=False, show_progress=False)


@pytest.fixture
def text_unit() -> TextUnit:
    return TextUnit(id="t1", text="Acme acquired Beta in 2020.")


# --------------------------------------------------------------------------- #
# _get_string_value
# --------------------------------------------------------------------------- #
class TestGetStringValue:
    def test_plain_string_stripped(self) -> None:
        assert ClaimExtractor._get_string_value("  hello  ") == "hello"

    def test_none_is_empty(self) -> None:
        assert ClaimExtractor._get_string_value(None) == ""

    def test_text_key_dict(self) -> None:
        assert ClaimExtractor._get_string_value({"#text": " v "}) == "v"

    def test_single_value_dict_recurses(self) -> None:
        assert ClaimExtractor._get_string_value({"x": "  y "}) == "y"

    def test_int_coerced(self) -> None:
        assert ClaimExtractor._get_string_value(2020) == "2020"


# --------------------------------------------------------------------------- #
# _parse_claim_data
# --------------------------------------------------------------------------- #
class TestParseClaimData:
    def test_valid_claim(self, extractor, text_unit) -> None:
        claim = extractor._parse_claim_data(
            {
                "subject": "Acme",
                "object": "Beta",
                "claim_type": "ACQUISITION",
                "description": "Acme acquired Beta",
                "claim_status": "TRUE",
            },
            text_unit,
        )
        assert claim is not None
        assert claim.subject_name == "Acme"
        assert claim.object_name == "Beta"
        assert claim.type == "ACQUISITION"
        assert claim.status == "TRUE"
        assert claim.text_unit_ids == ["t1"]

    def test_missing_subject_returns_none(self, extractor, text_unit) -> None:
        assert (
            extractor._parse_claim_data(
                {"object": "Beta", "claim_type": "X"}, text_unit
            )
            is None
        )

    def test_missing_type_returns_none(self, extractor, text_unit) -> None:
        assert (
            extractor._parse_claim_data(
                {"subject": "Acme", "object": "Beta"}, text_unit
            )
            is None
        )

    def test_stable_id_for_same_triple(self, extractor, text_unit) -> None:
        a = extractor._parse_claim_data(
            {"subject": "Acme", "object": "Beta", "claim_type": "X"}, text_unit
        )
        b = extractor._parse_claim_data(
            {"subject": "Acme", "object": "Beta", "claim_type": "X"}, text_unit
        )
        assert a.id == b.id

    def test_repeated_tags_coerced_not_stringified(self, extractor, text_unit) -> None:
        # The XML parser returns a list for a repeated tag.
        claim = extractor._parse_claim_data(
            {
                "subject": ["Vendor Co", "Vendor"],
                "object": "Buyer Ltd",
                "claim_type": ["DELIVERY", "OBLIGATION"],
                "claim_status": ["TRUE", "TRUE"],
                "description": [
                    "Ships parts monthly",
                    "Ships parts monthly",
                    "On time",
                ],
                "source_text": ["Vendor ships parts.", "Delivery is monthly."],
            },
            text_unit,
        )
        assert claim is not None
        assert claim.subject_name == "Vendor Co"
        assert claim.type == "DELIVERY"
        assert claim.status == "TRUE"
        assert claim.description == "Ships parts monthly\nOn time"
        assert claim.source_text == "Vendor ships parts.\nDelivery is monthly."
        assert "[" not in claim.description

    def test_id_uses_entity_key_normalization(self, extractor, text_unit) -> None:
        a = extractor._parse_claim_data(
            {"subject": "Vendor  Co.", "object": "Buyer", "claim_type": "X"}, text_unit
        )
        b = extractor._parse_claim_data(
            {"subject": "vendor co", "object": "Buyer", "claim_type": "X"}, text_unit
        )
        assert a.id == b.id

    def test_distinct_dates_get_distinct_ids(self, extractor, text_unit) -> None:
        base = {"subject": "Vendor", "object": "Buyer", "claim_type": "DELIVERY"}
        first = extractor._parse_claim_data(
            {**base, "start_date": "2024-01-01", "end_date": "2024-01-31"}, text_unit
        )
        second = extractor._parse_claim_data(
            {**base, "start_date": "2024-06-01", "end_date": "2024-06-30"}, text_unit
        )
        assert first.id != second.id
        merged = extractor._merge_claims([first, second])
        assert {c.start_date for c in merged} == {"2024-01-01", "2024-06-01"}

    def test_unknown_dates_keep_date_free_id(self, extractor, text_unit) -> None:
        base = {"subject": "Vendor", "object": "Buyer", "claim_type": "DELIVERY"}
        undated = extractor._parse_claim_data(base, text_unit)
        unknown = extractor._parse_claim_data(
            {**base, "start_date": "UNKNOWN", "end_date": "unknown"}, text_unit
        )
        assert undated.id == unknown.id


# --------------------------------------------------------------------------- #
# _parse_extraction_result
# --------------------------------------------------------------------------- #
class TestParseExtractionResult:
    def test_non_dict_returns_empty(self, extractor, text_unit) -> None:
        assert extractor._parse_extraction_result("not a dict", text_unit) == []

    def test_missing_claims_key_returns_empty(self, extractor, text_unit) -> None:
        assert extractor._parse_extraction_result({"other": 1}, text_unit) == []

    def test_single_claim_wrapped(self, extractor, text_unit) -> None:
        result = {
            "claims": {"claim": {"subject": "A", "object": "B", "claim_type": "X"}}
        }
        claims = extractor._parse_extraction_result(result, text_unit)
        assert len(claims) == 1
        assert claims[0].subject_name == "A"

    def test_invalid_claim_skipped(self, extractor, text_unit) -> None:
        result = {
            "claims": {
                "claim": [
                    {"subject": "A", "object": "B", "claim_type": "X"},
                    {"subject": "", "object": "", "claim_type": ""},  # invalid
                ]
            }
        }
        claims = extractor._parse_extraction_result(result, text_unit)
        assert len(claims) == 1


# --------------------------------------------------------------------------- #
# _merge_claims
# --------------------------------------------------------------------------- #
class TestMergeClaims:
    def _claim(self, id_, **kw) -> Claim:
        base = {
            "subject_id": "",
            "subject_name": "Acme",
            "object_id": "",
            "object_name": "Beta",
            "type": "ACQUISITION",
            "status": "TRUE",
        }
        base.update(kw)
        return Claim(id=id_, **base)

    def test_same_id_claims_merged(self, extractor) -> None:
        c1 = self._claim("c1", description="d1", text_unit_ids=["t1"])
        c2 = self._claim("c1", description="d2", text_unit_ids=["t2"])
        merged = extractor._merge_claims([c1, c2])
        assert len(merged) == 1
        assert set(merged[0].text_unit_ids) == {"t1", "t2"}
        assert "d1" in merged[0].description and "d2" in merged[0].description

    def test_distinct_ids_kept(self, extractor) -> None:
        c1 = self._claim("c1", text_unit_ids=["t1"])
        c2 = self._claim("c2", object_name="Gamma", text_unit_ids=["t2"])
        assert len(extractor._merge_claims([c1, c2])) == 2

    def test_empty(self, extractor) -> None:
        assert extractor._merge_claims([]) == []


# --------------------------------------------------------------------------- #
# _prepare_extraction_inputs (keyed by unit id, order-independent)
# --------------------------------------------------------------------------- #
class TestPrepareExtractionInputs:
    def test_returns_dict_keyed_by_unit_id(self, extractor) -> None:
        units = [
            TextUnit(id="t1", text="Acme acquired Beta."),
            TextUnit(id="t2", text="Gamma merged with Delta."),
        ]
        out = extractor._prepare_extraction_inputs(units, all_entities=[])
        assert set(out.keys()) == {"t1", "t2"}
        assert out["t1"]["input_text"] == "Acme acquired Beta."
        assert out["t2"]["input_text"] == "Gamma merged with Delta."

    def test_entity_specs_filtered_by_relevance(self, extractor) -> None:
        units = [TextUnit(id="t1", text="Acme text")]
        entities = [
            Entity(id="e1", name="Acme", text_unit_ids=["t1"]),  # relevant
            Entity(id="e2", name="Other", text_unit_ids=["t2"]),  # not relevant
        ]
        out = extractor._prepare_extraction_inputs(units, all_entities=entities)
        specs = out["t1"]["entity_specs"]
        assert "Acme" in specs
        assert "Other" not in specs

    def test_empty_unit_list_yields_empty_dict(self, extractor) -> None:
        assert extractor._prepare_extraction_inputs([], all_entities=[]) == {}


# --------------------------------------------------------------------------- #
# format_entities_with_limit_task (module-level, AWS-free)
# --------------------------------------------------------------------------- #
class TestFormatEntitiesWithLimit:
    def test_empty_returns_empty_string(self) -> None:
        assert format_entities_with_limit_task([], max_entities=5) == ""

    def test_under_limit_lists_all_names(self) -> None:
        ents = [Entity(id="e1", name="Alice"), Entity(id="e2", name="Bob")]
        out = format_entities_with_limit_task(ents, max_entities=5)
        assert out == "Alice\nBob"

    def test_over_limit_truncates_and_appends_more(self) -> None:
        ents = [
            Entity(id=f"e{i}", name=f"N{i}", text_unit_ids=["t"] * i) for i in range(5)
        ]
        out = format_entities_with_limit_task(ents, max_entities=2)
        # Most-supported entities (more text_unit_ids) sort first.
        assert "N4" in out and "N3" in out
        assert "... and 3 more entities" in out


# --------------------------------------------------------------------------- #
# _prepare_claim_input_task (module-level, AWS-free)
# --------------------------------------------------------------------------- #
class TestPrepareClaimInputTask:
    def test_plain_text_no_entities(self) -> None:
        unit = TextUnit(id="t1", text="raw text")
        out = _prepare_claim_input_task(
            unit, [], {"max_entities_per_prompt": 0, "target_language": "en"}
        )
        assert out == {"input_text": "raw text", "entity_specs": ""}

    def test_translated_text_preferred(self) -> None:
        unit = TextUnit(id="t1", text="original", translated_texts={"en": "translated"})
        out = _prepare_claim_input_task(
            unit, [], {"max_entities_per_prompt": 0, "target_language": "en"}
        )
        assert out["input_text"] == "translated"

    def test_translated_text_missing_target_falls_back_to_text(self) -> None:
        unit = TextUnit(id="t1", text="original", translated_texts={"fr": "francais"})
        out = _prepare_claim_input_task(
            unit, [], {"max_entities_per_prompt": 0, "target_language": "en"}
        )
        assert out["input_text"] == "original"

    def test_relevant_entities_included(self) -> None:
        unit = TextUnit(id="t1", text="body")
        entities = [
            Entity(id="e1", name="Acme", text_unit_ids=["t1"]),
            Entity(id="e2", name="Other", text_unit_ids=["t9"]),
        ]
        out = _prepare_claim_input_task(
            unit, entities, {"max_entities_per_prompt": 5, "target_language": "en"}
        )
        assert "Acme" in out["entity_specs"]
        assert "Other" not in out["entity_specs"]


# --------------------------------------------------------------------------- #
# _get_string_value — multi-key dict branch
# --------------------------------------------------------------------------- #
class TestGetStringValueExtra:
    def test_multi_key_dict_stringified(self) -> None:
        # More than one value and no #text -> str(dict).
        out = ClaimExtractor._get_string_value({"a": "1", "b": "2"})
        assert "a" in out and "b" in out


# --------------------------------------------------------------------------- #
# ClaimExtractionStats derived properties
# --------------------------------------------------------------------------- #
class TestClaimExtractionStats:
    def test_zero_processed(self) -> None:
        assert ClaimExtractionStats().success_rate == 0.0
        assert ClaimExtractionStats().processed_unit_count == 0

    def test_success_rate_computed(self) -> None:
        s = ClaimExtractionStats(num_successful_extractions=3, num_failed_extractions=1)
        assert s.processed_unit_count == 4
        assert s.success_rate == 75.0


# --------------------------------------------------------------------------- #
# _process_extraction_results
# --------------------------------------------------------------------------- #
class TestProcessExtractionResults:
    def test_none_result_counts_failure(self, extractor) -> None:
        units = [TextUnit(id="t1", text="x")]
        claims = extractor._process_extraction_results(units, [None])
        assert claims == []
        assert extractor.stats.num_failed_extractions == 1

    def test_success_accumulates_claims(self, extractor) -> None:
        units = [TextUnit(id="t1", text="x")]
        results = [
            {"claims": {"claim": {"subject": "A", "object": "B", "claim_type": "X"}}}
        ]
        claims = extractor._process_extraction_results(units, results)
        assert len(claims) == 1
        assert extractor.stats.num_successful_extractions == 1

    def test_length_mismatch_warns_and_zips_available(self, extractor) -> None:
        # 2 units but 1 result -> strict=False zip stops at the shorter list.
        units = [TextUnit(id="t1", text="x"), TextUnit(id="t2", text="y")]
        results = [
            {"claims": {"claim": {"subject": "A", "object": "B", "claim_type": "X"}}}
        ]
        claims = extractor._process_extraction_results(units, results)
        assert len(claims) == 1

    def test_parse_failure_counts_failure(self, extractor, mocker) -> None:
        units = [TextUnit(id="t1", text="x")]
        mocker.patch.object(
            extractor, "_parse_extraction_result", side_effect=ValueError("bad")
        )
        claims = extractor._process_extraction_results(units, [{"claims": {}}])
        assert claims == []
        assert extractor.stats.num_failed_extractions == 1


# --------------------------------------------------------------------------- #
# _parse_claim_data — exception path
# --------------------------------------------------------------------------- #
class TestParseClaimDataExceptions:
    def test_exception_during_build_returns_none(
        self, extractor, text_unit, mocker
    ) -> None:
        mocker.patch.object(
            extractor, "_parse_attributes", side_effect=RuntimeError("attr boom")
        )
        out = extractor._parse_claim_data(
            {"subject": "A", "object": "B", "claim_type": "X"}, text_unit
        )
        assert out is None


# --------------------------------------------------------------------------- #
# extract_from_text_units (batch chain mocked)
# --------------------------------------------------------------------------- #
class TestExtractFromTextUnits:
    def test_empty_units_returns_empty(self, extractor) -> None:
        claims, stats = extractor.extract_from_text_units([])
        assert claims == []
        assert stats.num_total_units == 0

    def test_end_to_end_with_mocked_chain(self, extractor, mocker) -> None:
        units = [TextUnit(id="t1", text="Acme acquired Beta.")]
        extractor.claim_extractor.invoke = mocker.Mock(
            side_effect=[
                {
                    "claims": {
                        "claim": {
                            "subject": "Acme",
                            "object": "Beta",
                            "claim_type": "ACQUISITION",
                        }
                    }
                }
            ]
        )
        claims, stats = extractor.extract_from_text_units(units)
        assert len(claims) == 1
        assert claims[0].subject_name == "Acme"
        assert stats.total_claims_extracted == 1
        assert stats.num_total_units == 1

    def test_batch_error_with_ignore_errors_returns_empty(
        self, extractor, mocker
    ) -> None:
        extractor.ignore_errors = True
        fake_bp = mocker.Mock()
        fake_bp.execute_with_fallback.side_effect = RuntimeError("boom")
        extractor.batch_processor = fake_bp
        claims, stats = extractor.extract_from_text_units(
            [TextUnit(id="t1", text="x"), TextUnit(id="t2", text="y")]
        )
        assert claims == []
        # Every unit failed; the stats must say so, not report zero failures.
        assert stats.num_total_units == 2
        assert stats.num_failed_extractions == 2
        assert stats.failed_text_unit_ids == ["t1", "t2"]

    def test_unit_whose_input_prep_failed_counts_as_failed(
        self, extractor, mocker
    ) -> None:
        units = [TextUnit(id="t1", text="x"), TextUnit(id="t2", text="y")]
        mocker.patch.object(
            extractor,
            "_prepare_extraction_inputs",
            return_value={"t1": {"input_text": "x", "entity_specs": ""}},
        )
        extractor.batch_processor = mocker.Mock()
        extractor.batch_processor.execute_with_fallback.return_value = [{"claims": {}}]
        _, stats = extractor.extract_from_text_units(units)
        assert stats.num_successful_extractions == 1
        assert stats.num_failed_extractions == 1
        assert stats.failed_text_unit_ids == ["t2"]

    def test_batch_error_without_ignore_raises(self, extractor, mocker) -> None:
        extractor.ignore_errors = False
        fake_bp = mocker.Mock()
        fake_bp.execute_with_fallback.side_effect = RuntimeError("boom")
        extractor.batch_processor = fake_bp
        with pytest.raises(RuntimeError, match="boom"):
            extractor.extract_from_text_units([TextUnit(id="t1", text="x")])


# --------------------------------------------------------------------------- #
# _log_completion_summary
# --------------------------------------------------------------------------- #
class TestLogCompletionSummary:
    def test_logs_with_failures(self, caplog) -> None:
        stats = ClaimExtractionStats(
            num_total_units=3,
            num_successful_extractions=2,
            num_failed_extractions=1,
            total_claims_extracted=4,
            total_processing_time=1.5,
        )
        with caplog.at_level(logging.INFO):
            ClaimExtractor._log_completion_summary(stats)
        assert "(2/3 units processed)" in caplog.text
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "1 text units failed during extraction" in caplog.text
