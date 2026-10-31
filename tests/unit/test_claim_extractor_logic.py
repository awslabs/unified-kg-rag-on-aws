# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ClaimExtractor pure logic (AWS-free).

Covers _get_string_value coercion, _parse_claim_data (missing-field rejection,
stable id, attributes), _parse_extraction_result (malformed/empty handling),
and _prepare_extraction_inputs returning a dict keyed by unit id
(the as_completed ordering fix). Bedrock/boto wiring is patched out so __init__
never touches AWS, and a single-worker thread pool keeps the executor cheap.
"""

from __future__ import annotations

import logging

import pytest

import unified_kg_rag.adapters.ingestion.claim_extractor as ce_module
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.ingestion.claim_extractor import (
    ClaimExtractionStats,
    ClaimExtractor,
    ClaimsOutputParser,
)
from unified_kg_rag.domain.models import Config, TextUnit
from unified_kg_rag.shared.exceptions import LLMOutputTruncatedError

pytestmark = pytest.mark.unit


@pytest.fixture
def extractor(config: Config, mocker) -> ClaimExtractor:
    mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    mocker.patch.object(ce_module, "create_robust_xml_output_parser")
    mocker.patch.object(ce_module, "setup_chain")
    # Thread pool (not process pool) so patched module globals are inherited and
    # no real subprocess/AWS work happens.
    return ClaimExtractor(config, show_progress=False)


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
    @pytest.mark.parametrize("result", ["not a dict", {"other": 1}])
    def test_output_without_a_claims_section_is_cut_off(
        self, extractor, text_unit, result
    ) -> None:
        # Regression: recorded PROCESSED with no claims, never re-extracted.
        with pytest.raises(LLMOutputTruncatedError, match="no claims section"):
            extractor._parse_extraction_result(result, text_unit)

    @pytest.mark.parametrize("empty", [{}, None, ""])
    def test_an_empty_claims_section_is_no_claims(
        self, extractor, text_unit, empty
    ) -> None:
        # <claims></claims>: the unit states none.
        assert extractor._parse_extraction_result({"claims": empty}, text_unit) == []

    def test_claims_none_of_which_is_complete_are_cut_off(
        self, extractor, text_unit
    ) -> None:
        # "<claims><claim><subject>A</subject><obj" with a normal stop reason.
        with pytest.raises(LLMOutputTruncatedError, match="none of the 1 claims"):
            extractor._parse_extraction_result(
                {"claims": {"claim": {"subject": "A"}}}, text_unit
            )

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

    def test_a_claim_without_fields_is_skipped(self, extractor, text_unit) -> None:
        # <claim>none</claim> parses to a string: skipped, the others kept.
        result = {
            "claims": {
                "claim": [{"subject": "A", "object": "B", "claim_type": "X"}, "none"]
            }
        }
        claims = extractor._parse_extraction_result(result, text_unit)
        assert [c.subject_name for c in claims] == ["A"]


_TWO_CLAIMS = (
    "<claims><claim><subject>A</subject><object>B</object>"
    "<claim_type>OBLIGATION</claim_type><description>A owes B</description>"
    "<source_text>A owes B.</source_text></claim>"
    "<claim><subject>C</subject><object>D</object>"
    "<claim_type>OBLIGATION</claim_type><description>C owes D money</description>"
    "<source_text>C owes D.</source_text></claim></claims>"
)


class TestClaimsOutputParser:
    """A claim the response was cut off inside (no stop reason) is dropped,
    not stored with the fields before the cut."""

    @pytest.mark.parametrize(
        "cut_after", ["<claim_type>OBLIG", "C owes D mo", "<source_text>C o"]
    )
    def test_a_claim_cut_off_inside_is_dropped(
        self, extractor, text_unit, cut_after: str
    ) -> None:
        text = _TWO_CLAIMS[: _TWO_CLAIMS.rindex(cut_after) + len(cut_after)]
        result = ClaimsOutputParser(tags=["claims"]).parse(text)
        claims = extractor._parse_extraction_result(result, text_unit)
        assert [(c.subject_name, c.description) for c in claims] == [("A", "A owes B")]

    def test_a_cut_after_the_last_claim_keeps_every_claim(
        self, extractor, text_unit
    ) -> None:
        text = _TWO_CLAIMS.removesuffix("</claims>")
        result = ClaimsOutputParser(tags=["claims"]).parse(text)
        claims = extractor._parse_extraction_result(result, text_unit)
        assert [c.subject_name for c in claims] == ["A", "C"]

    def test_a_note_after_the_closed_section_keeps_the_last_claim(
        self, extractor, text_unit
    ) -> None:
        # The section is closed, so nothing was cut off: a note after it that
        # names a <claim> tag made the last claim look unclosed and blanked it.
        text = _TWO_CLAIMS + "\nNote: each <claim> above is one obligation."
        result = ClaimsOutputParser(tags=["claims"]).parse(text)
        claims = extractor._parse_extraction_result(result, text_unit)
        assert [c.subject_name for c in claims] == ["A", "C"]

    @pytest.mark.parametrize(
        "missing", ["<object>B</object>", "<claim_type>X</claim_type>"]
    )
    def test_a_closed_section_of_only_an_invalid_claim_is_no_claims(
        self, extractor, text_unit, missing: str
    ) -> None:
        # The model closed </claims>: nothing was cut off, the claim lacks a
        # field and is dropped, rather than the unit failed as cut-off output.
        text = (
            "<claims><claim><subject>A</subject><object>B</object>"
            "<claim_type>X</claim_type><description>A owes B</description>"
            "</claim></claims>"
        ).replace(missing, "")
        result = ClaimsOutputParser(tags=["claims"]).parse(text)
        assert extractor._parse_extraction_result(result, text_unit) == []

    def test_an_output_of_only_a_cut_off_claim_is_cut_off(
        self, extractor, text_unit
    ) -> None:
        # It parsed into one complete-looking claim, "C owes D mo".
        text = _TWO_CLAIMS[_TWO_CLAIMS.index("<claim><subject>C") :]
        text = "<claims>" + text[: text.index("money")]
        result = ClaimsOutputParser(tags=["claims"]).parse(text)
        with pytest.raises(LLMOutputTruncatedError, match="none of the 1 claims"):
            extractor._parse_extraction_result(result, text_unit)


class TestProcessExtractionResults:
    def test_parse_failure_counts_failure(self, extractor, mocker) -> None:
        units = [TextUnit(id="t1", text="x")]
        mocker.patch.object(
            extractor, "_parse_extraction_result", side_effect=ValueError("bad")
        )
        claims = extractor._process_extraction_results(units, [{"claims": {}}])
        assert claims == {}
        assert extractor.stats.num_failed_extractions == 1

    def test_a_parse_failure_discards_the_unit_s_response(
        self, extractor, mocker
    ) -> None:
        # Raised after the response parsed: its response leaves the LLM
        # response cache, so a re-run asks the model again.
        units = [TextUnit(id="t0", text="x"), TextUnit(id="t1", text="y")]
        mocker.patch.object(
            extractor,
            "_parse_extraction_result",
            side_effect=lambda result, unit: [] if unit.id == "t0" else 1 / 0,
        )
        dropped: list[int] = []
        discards = {idx: lambda idx=idx: dropped.append(idx) for idx in range(2)}
        claims = extractor._process_extraction_results(
            units, [{"claims": {}}, {"claims": {}}], discards
        )
        assert claims == {"t0": []} and dropped == [1]
        assert extractor.stats.item_failed_text_unit_ids == ["t1"]


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
# extract_units (batch chain mocked)
# --------------------------------------------------------------------------- #
def _extracted(extractor, units):
    per_unit, stats = extractor.extract_units(units)
    return [c for cs in per_unit.values() for c in cs], stats


class TestExtractUnits:
    def test_empty_units_returns_empty(self, extractor) -> None:
        claims, stats = _extracted(extractor, [])
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
        claims, stats = _extracted(extractor, units)
        assert len(claims) == 1
        assert claims[0].subject_name == "Acme"
        assert stats.total_claims_extracted == 1
        assert stats.num_total_units == 1

    def test_a_cut_off_response_fails_the_unit_and_is_not_retried(
        self, extractor, mocker
    ) -> None:
        # Regression: the unit was recorded PROCESSED with zero claims. Now
        # it fails after one call, counted as its own error once the
        # known-good request is answered.
        units = [TextUnit(id="t1", text="Acme acquired Beta.")]
        whole = {
            "claims": {"claim": {"subject": "A", "object": "B", "claim_type": "X"}}
        }
        invoke = mocker.Mock(
            side_effect=[{"claims": {"claim": {"subject": "Acme"}}}, whole]
        )
        extractor.claim_extractor.invoke = invoke
        per_unit, stats = extractor.extract_units(units)
        assert per_unit == {}
        assert stats.failed_text_unit_ids == ["t1"]
        assert stats.item_failed_text_unit_ids == ["t1"]
        assert invoke.call_count == 2  # the unit, then the known-good request

    def test_batch_error_with_ignore_errors_returns_empty(
        self, extractor, mocker
    ) -> None:
        extractor.ignore_errors = True
        fake_bp = mocker.Mock()
        fake_bp.execute_with_fallback.side_effect = RuntimeError("boom")
        extractor.batch_processor = fake_bp
        claims, stats = _extracted(
            extractor, [TextUnit(id="t1", text="x"), TextUnit(id="t2", text="y")]
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
        _, stats = _extracted(extractor, units)
        assert stats.num_successful_extractions == 1
        assert stats.num_failed_extractions == 1
        assert stats.failed_text_unit_ids == ["t2"]

    def test_a_unit_whose_claims_fail_to_build_discards_its_response(
        self, extractor, mocker
    ) -> None:
        # extract_units hands the batch's discards on: without them the
        # response a unit's claims failed to build from stayed cached, and a
        # re-run with the same pipeline id was served it again.
        dropped: list[int] = []

        def execute(**kwargs):  # type: ignore[no-untyped-def]
            for idx in range(2):
                kwargs["discards"][idx] = lambda idx=idx: dropped.append(idx)
            return [{"claims": {}}, {"claims": {}}]

        extractor.batch_processor = mocker.Mock()
        extractor.batch_processor.execute_with_fallback.side_effect = execute
        mocker.patch.object(
            extractor,
            "_parse_extraction_result",
            side_effect=lambda result, unit: [] if unit.id == "t0" else 1 / 0,
        )
        _, stats = extractor.extract_units(
            [TextUnit(id=f"t{i}", text="x") for i in range(2)]
        )
        assert dropped == [1]
        assert stats.item_failed_text_unit_ids == ["t1"]

    def test_batch_error_without_ignore_raises(self, extractor, mocker) -> None:
        extractor.ignore_errors = False
        fake_bp = mocker.Mock()
        fake_bp.execute_with_fallback.side_effect = RuntimeError("boom")
        extractor.batch_processor = fake_bp
        with pytest.raises(RuntimeError, match="boom"):
            _extracted(extractor, [TextUnit(id="t1", text="x")])


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


def test_claim_prompt_defines_one_type_taxonomy() -> None:
    # Only <claim_type> is output, so the prompt lists exactly one set of types.
    from unified_kg_rag.domain.prompts.graph_extraction import ClaimExtractionPrompt

    system = ClaimExtractionPrompt.resolve().system_prompt_template
    assert "# CLAIM TYPES" in system
    assert "Claim Categories" not in system
    assert "FACTUAL_ASSERTION" not in system
