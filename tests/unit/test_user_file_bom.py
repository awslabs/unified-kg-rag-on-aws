# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A UTF-8 byte-order mark in a file the user supplies is not content.

Excel's "CSV UTF-8" and Notepad write one, common for Korean and Japanese
files. The corpus loaders always read past it; the other user-supplied files
(an evaluation dataset, a JSON schema file, an uploaded dataset, the corpus
text of ``run-prompt-tuning``) read it as ``json.loads``'s "Unexpected UTF-8
BOM" or as a character of the text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unified_kg_rag.adapters.schema_sources import (
    SchemaSourceContext,
    get_schema_source,
)
from unified_kg_rag.application.cli import run_prompt_tuning
from unified_kg_rag.application.cli.console import evaluations
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import SchemaSourceConfig
from unified_kg_rag.evaluation.evaluation_manager import EvaluationManager
from unified_kg_rag.shared.utils.files import read_user_text

pytestmark = pytest.mark.unit

_BOM = b"\xef\xbb\xbf"
_DATASET = [{"question": "공급업체는 누구입니까?", "ground_truth": "한빛상사"}]


def _with_bom(path: Path, text: str) -> Path:
    path.write_bytes(_BOM + text.encode("utf-8"))
    return path


def test_read_user_text_drops_a_leading_mark_only(tmp_path: Path) -> None:
    path = _with_bom(tmp_path / "a.txt", "한빛상사﻿")
    assert read_user_text(path) == "한빛상사﻿"
    assert read_user_text(_with_bom(tmp_path / "b.txt", "")) == ""


def test_an_evaluation_dataset_with_a_mark_loads(tmp_path: Path) -> None:
    path = _with_bom(tmp_path / "eval.json", json.dumps(_DATASET, ensure_ascii=False))
    queries, _ = EvaluationManager.load_data(path)
    assert queries[0].question == "공급업체는 누구입니까?"
    assert len(queries) == 1


def test_an_uploaded_dataset_with_a_mark_is_read(tmp_path: Path) -> None:
    path = _with_bom(tmp_path / "eval.json", json.dumps(_DATASET, ensure_ascii=False))
    assert evaluations._read_json(path, 2**20) == _DATASET


def test_a_json_schema_file_with_a_mark_loads(tmp_path: Path) -> None:
    schema = {"relationship_types": ["공급: 부품을 공급한다"]}
    path = _with_bom(tmp_path / "schema.json", json.dumps(schema, ensure_ascii=False))
    document = get_schema_source("file")(
        SchemaSourceContext(SchemaSourceConfig(kind="file", path=str(path)), Config())
    )
    assert document.relationship_types is not None
    assert document.relationship_types == schema["relationship_types"]


def test_prompt_tuning_corpus_text_keeps_no_mark(tmp_path: Path) -> None:
    _with_bom(tmp_path / "a.txt", "한빛상사는 부품을 공급한다.")
    (tmp_path / "b.md").write_text("plain", encoding="utf-8")
    texts = run_prompt_tuning.load_corpus_texts(tmp_path, max_docs=10)
    assert sorted(texts) == ["plain", "한빛상사는 부품을 공급한다."]
