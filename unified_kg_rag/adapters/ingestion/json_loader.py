# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free JSON source loader.

LangChain's ``JSONLoader`` needs the optional ``jq`` package (not a project
dependency), and its text mode rejects JSON objects, so every ``.json`` source
failed to parse. This loader uses the standard library only.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document as LangChainDocument

JsonTextMode = Literal["pretty", "strings"]


class JsonTextLoader(BaseLoader):
    """Load a JSON file as text for knowledge-graph extraction.

    * A top-level array yields one LangChain document (one page) per element,
      so record-style files keep their record boundaries.
    * Any other top-level value yields a single document.

    ``text_mode`` controls how a value is rendered:

    * ``"pretty"`` (default): indented JSON with non-ASCII kept as-is. Keys
      stay visible, which gives the extractor the field semantics
      (``"buyer": "Acme"``). A top-level string is used verbatim.
    * ``"strings"``: only the string leaves, one per line, in document order
      (drops keys, numbers and booleans; useful for prose-heavy payloads).

    Metadata carries ``source`` (the file path) and ``seq_num`` (1-based element
    index), matching the keys ``JSONLoader`` emitted.
    """

    def __init__(
        self,
        file_path: str | Path,
        encoding: str | None = None,
        text_mode: JsonTextMode = "pretty",
    ) -> None:
        if text_mode not in ("pretty", "strings"):
            raise ValueError(
                f"text_mode must be 'pretty' or 'strings', got '{text_mode}'"
            )
        self.file_path = Path(file_path)
        # utf-8-sig also accepts a leading BOM, which json.loads would reject.
        self.encoding = encoding or "utf-8-sig"
        self.text_mode = text_mode

    def lazy_load(self) -> Iterator[LangChainDocument]:
        raw = self.file_path.read_text(encoding=self.encoding).lstrip("\ufeff")
        data = json.loads(raw)
        items = data if isinstance(data, list) else [data]
        for seq_num, item in enumerate(items, start=1):
            text = self._render(item)
            if not text.strip():
                continue
            yield LangChainDocument(
                page_content=text,
                metadata={"source": str(self.file_path), "seq_num": seq_num},
            )

    def _render(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if self.text_mode == "strings":
            return "\n".join(s for s in _iter_strings(value) if s.strip())
        return json.dumps(value, ensure_ascii=False, indent=2)


def _iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)
