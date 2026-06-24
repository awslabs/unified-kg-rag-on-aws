# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for ClaimExtractor pure logic (AWS-free).

Covers _get_string_value coercion, _parse_claim_data (missing-field rejection,
stable id, attributes), _parse_extraction_result (malformed/empty handling),
_merge_claims, and _prepare_extraction_inputs returning a dict keyed by unit id
(the as_completed ordering fix). Bedrock/boto wiring is patched out so __init__
never touches AWS, and a single-worker thread pool keeps the executor cheap.
"""

from __future__ import annotations

import pytest

import aws_graphrag.adapters.ingestion.claim_extractor as ce_module
from aws_graphrag.adapters.ingestion.claim_extractor import ClaimExtractor
from aws_graphrag.domain.models import Claim, Config, Entity, TextUnit

pytestmark = pytest.mark.unit


@pytest.fixture
def extractor(config: Config, mocker) -> ClaimExtractor:
    mocker.patch.object(ce_module, "boto3")
    mocker.patch.object(ce_module, "BedrockLanguageModelFactory")
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
