# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The custom analyzers the OpenSearch indexer defines in its indices.

How ``cjk_script_split`` tokenizes is pinned against a real OpenSearch in
``tests/integration/test_store_parity.py`` (LOCAL_STORES); these tests pin
where its definition goes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from unified_kg_rag.adapters.local.search_engine.analyzers import (
    _ANALYZERS,
    STANDARD,
    analyzes_alike,
    get_analyzer,
)
from unified_kg_rag.adapters.storage.opensearch_analysis import (
    CJK_SCRIPT_SPLIT,
    CUSTOM_ANALYZERS,
    with_analysis,
)
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import LanguageCode

pytestmark = pytest.mark.unit

_IAC_CONFIG = Path(__file__).resolve().parents[2] / "iac" / "config.py"


@pytest.fixture
def indexer(config: Config) -> OpenSearchIndexer:
    inst = OpenSearchIndexer.__new__(OpenSearchIndexer)
    inst.config = config
    inst.opensearch_config = config.indexing.opensearch
    inst.language = LanguageCode.KO.value
    inst.analyzer = CJK_SCRIPT_SPLIT
    inst._embedding_dimension = 8
    inst._index_limits = {}
    config.processing.translation.source_language = LanguageCode.KO
    config.indexing.opensearch.language_analyzers = {"ko": CJK_SCRIPT_SPLIT}
    return inst


def test_an_index_whose_fields_use_it_defines_it(indexer: OpenSearchIndexer) -> None:
    # Without the definition OpenSearch refuses the mapping ("analyzer
    # [cjk_script_split] has not been configured").
    for mapping in (
        indexer._get_text_units_mapping(),
        indexer._get_entities_mapping(),
        indexer._get_relationships_mapping(),
        indexer._get_community_reports_mapping(),
        indexer._get_claims_mapping(),
        indexer._get_summary_nodes_mapping(),
    ):
        analysis = mapping["settings"]["analysis"]
        assert analysis["analyzer"][CJK_SCRIPT_SPLIT] == (
            CUSTOM_ANALYZERS[CJK_SCRIPT_SPLIT]["analyzer"][CJK_SCRIPT_SPLIT]
        )
        assert analysis["char_filter"] == dict(
            CUSTOM_ANALYZERS[CJK_SCRIPT_SPLIT]["char_filter"]
        )


def test_an_index_without_custom_analyzers_defines_none(
    indexer: OpenSearchIndexer,
) -> None:
    indexer.analyzer = "nori"
    indexer.config.indexing.opensearch.language_analyzers = {"ko": "nori"}
    assert "analysis" not in indexer._get_text_units_mapping()["settings"]
    # The extraction records hold no text field.
    indexer.analyzer = CJK_SCRIPT_SPLIT
    assert "analysis" not in indexer._get_extraction_records_mapping()["settings"]


def test_a_translated_field_s_analyzer_is_defined(indexer: OpenSearchIndexer) -> None:
    # Only the translated Korean field names it (the source text is English).
    indexer.analyzer = "english"
    indexer.language = LanguageCode.EN.value
    indexer.config.processing.translation.source_language = LanguageCode.EN
    indexer.config.processing.translation.additional_target_languages = [
        LanguageCode.KO
    ]
    mapping = indexer._get_text_units_mapping()
    assert mapping["mappings"]["properties"]["translated_text_ko"]["analyzer"] == (
        CJK_SCRIPT_SPLIT
    )
    assert CJK_SCRIPT_SPLIT in mapping["settings"]["analysis"]["analyzer"]


def test_index_settings_analysis_is_kept_beside_it() -> None:
    # A user's own analysis settings (index_settings.analysis) reach the index
    # with the definitions added, and a definition of the same name is theirs.
    own = {
        "analysis": {
            "analyzer": {"mine": {"type": "standard"}},
            "char_filter": {f"{CJK_SCRIPT_SPLIT}_boundary": {"type": "mapping"}},
        },
        "number_of_shards": 1,
    }
    settings = with_analysis(
        own, {"text": {"type": "text", "analyzer": CJK_SCRIPT_SPLIT}}
    )
    assert settings["number_of_shards"] == 1
    assert set(settings["analysis"]["analyzer"]) == {"mine", CJK_SCRIPT_SPLIT}
    assert settings["analysis"]["char_filter"] == own["analysis"]["char_filter"]
    # The caller's settings are not changed.
    assert set(own["analysis"]["analyzer"]) == {"mine"}


def test_a_multi_field_s_analyzer_is_defined() -> None:
    properties = {
        "name": {
            "type": "text",
            "fields": {"ko": {"type": "text", "analyzer": CJK_SCRIPT_SPLIT}},
        }
    }
    assert "analysis" in with_analysis({}, properties)


def test_the_cdk_maps_korean_to_an_analyzer_every_engine_has() -> None:
    # The IaC is a separate project: its default is read as text. Its Korean
    # analyzer is one the indexer defines, and the embedded engine knows it:
    # cjk's width fold with the English stop words kept, as OpenSearch's
    # (test_store_parity pins the tokens against a container).
    source = _IAC_CONFIG.read_text()
    (default,) = re.findall(r'opensearch_korean_analyzer: str = "([^"]+)"', source)
    assert default == CJK_SCRIPT_SPLIT
    assert f'"nori" if opensearch_nori_package else "{CJK_SCRIPT_SPLIT}"' in source
    for name in CUSTOM_ANALYZERS:
        assert name in _ANALYZERS
        assert not analyzes_alike(name, STANDARD)
        assert get_analyzer(name)("ＡＷＳ를 the") == ["aws", "를", "the"]


def _char_filter(text: str) -> str:
    """``text`` through cjk_script_split's char filter, its Java pattern
    read by Python's re (the classes are \\x{...} code points)."""
    (spec,) = CUSTOM_ANALYZERS[CJK_SCRIPT_SPLIT]["char_filter"].values()
    pattern = re.sub(
        r"\\x\{([0-9A-F]+)\}", lambda m: chr(int(m.group(1), 16)), spec["pattern"]
    )
    return re.sub(pattern, spec["replacement"], text)


@pytest.mark.parametrize(
    ("text", "split"),
    [
        # Halfwidth Katakana is Kana: no space before the Hangul, as the
        # embedded engine (which folds width first) keeps ナ한.
        ("ｶﾀｶﾅ한글", "ｶﾀｶﾅ한글"),
        ("カタカナ한글", "カタカナ한글"),
        ("AWS를", "AWS 를"),
        ("한글ﾃｽﾄ", "한글ﾃｽﾄ"),
    ],
)
def test_the_split_reads_halfwidth_kana_as_a_dense_script(
    text: str, split: str
) -> None:
    assert _char_filter(text) == split
