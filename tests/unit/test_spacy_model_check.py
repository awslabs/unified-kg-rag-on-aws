# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The spaCy model check of the Markdown/HTML loader.

unstructured downloads its spaCy model on first use, waiting up to 120 s,
and tried again for every file: with no network each Markdown or HTML file
waited and failed alike. The loader keeps the first failure and fails every
later file at once. Fake ``unstructured`` modules stand in, so the tests need
neither the extra nor a network.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from unified_kg_rag.adapters.ingestion import file_loaders
from unified_kg_rag.adapters.ingestion.file_loaders import (
    SPACY_MODEL,
    UnstructuredPartitionLoader,
    require_spacy_model,
)
from unified_kg_rag.shared.exceptions import DataProcessingError

pytestmark = pytest.mark.unit


class _Unstructured:
    """Fake ``unstructured.nlp.tokenize`` and ``unstructured.partition.md``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, error: Exception | None):
        self.error = error
        self.tokenized = 0
        self.partitioned = 0
        tokenize = types.ModuleType("unstructured.nlp.tokenize")
        tokenize.sent_tokenize = self._sent_tokenize  # type: ignore[attr-defined]
        partition = types.ModuleType("unstructured.partition.md")
        partition.partition_md = self._partition  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "unstructured.nlp.tokenize", tokenize)
        monkeypatch.setitem(sys.modules, "unstructured.partition.md", partition)

    def _sent_tokenize(self, text: str) -> list[str]:
        self.tokenized += 1
        if self.error is not None:
            raise self.error
        return [text]

    def _partition(self, filename: str) -> list[str]:
        self.partitioned += 1
        return ["Heading", "Body text."]


@pytest.fixture
def model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_loaders, "_spacy_problem", None)
    find_spec = importlib.util.find_spec

    def without_model(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == SPACY_MODEL:
            return None
        return find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", without_model)


@pytest.mark.usefixtures("model_missing")
def test_a_failed_download_is_tried_once_and_every_file_fails_at_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _Unstructured(
        monkeypatch, RuntimeError("Failed to download spaCy model: timed out")
    )
    source = tmp_path / "a.md"
    source.write_text("# Heading\n\nBody text.\n", encoding="utf-8")

    for _ in range(3):
        with pytest.raises(DataProcessingError) as raised:
            UnstructuredPartitionLoader(source, partition="md").load()
        message = str(raised.value)
        assert SPACY_MODEL in message
        assert "timed out" in message
        assert "docker/Dockerfile" in message

    assert fake.tokenized == 1
    assert fake.partitioned == 0


@pytest.mark.usefixtures("model_missing")
def test_a_download_that_succeeds_is_not_tried_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _Unstructured(monkeypatch, None)
    source = tmp_path / "a.md"
    source.write_text("# Heading\n", encoding="utf-8")

    for _ in range(2):
        (document,) = UnstructuredPartitionLoader(source, partition="md").load()
        assert document.page_content == "Heading\n\nBody text."

    assert (fake.tokenized, fake.partitioned) == (1, 2)


def test_an_installed_model_is_not_downloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_loaders, "_spacy_problem", None)
    find_spec = importlib.util.find_spec

    def with_model(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == SPACY_MODEL:
            return importlib.util.spec_from_loader(SPACY_MODEL, loader=None)
        return find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", with_model)
    fake = _Unstructured(monkeypatch, RuntimeError("no network"))

    require_spacy_model()

    assert fake.tokenized == 0
