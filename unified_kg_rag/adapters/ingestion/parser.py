# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.document_loaders.base import BaseLoader
from pydantic import BaseModel, Field

from unified_kg_rag.domain.models import Config, Document
from unified_kg_rag.shared import DataProcessingError, get_logger
from unified_kg_rag.shared.utils import convert_langchain_to_document

logger = get_logger(__name__)

# UnstructuredHTMLLoader / UnstructuredMarkdownLoader import fine but raise at
# .load() time if the heavy optional `unstructured` package is absent (it is not
# a runtime dependency). Detect it once so .md/.html/.htm are only advertised as
# supported when they can actually be parsed — instead of failing mid-ingestion
# with "No module named 'unstructured'".
_UNSTRUCTURED_AVAILABLE = importlib.util.find_spec("unstructured") is not None


class ParsingStats(BaseModel):
    num_total_files: int = Field(default=0)
    num_successful_files: int = Field(default=0)
    num_failed_files: int = Field(default=0)
    total_processing_time: float = Field(default=0.0)

    @property
    def success_rate(self) -> float:
        if self.num_total_files == 0:
            return 0.0
        return (self.num_successful_files / self.num_total_files) * 100


class BaseParser(ABC):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.stats = ParsingStats()

    @abstractmethod
    def parse_file(
        self, file_path: str | Path, index_value: str | None = None
    ) -> Document:
        pass

    def parse_files(
        self, file_paths: list[str | Path], index_value: str | None = None
    ) -> list[Document]:
        documents = []
        self.stats = ParsingStats(num_total_files=len(file_paths))

        for file_path in file_paths:
            try:
                doc = self.parse_file(file_path, index_value)
                documents.append(doc)
                self.stats.num_successful_files += 1
            except Exception as e:
                logger.error("Failed to parse '%s': %s", file_path, e)
                self.stats.num_failed_files += 1

        logger.info("Parsing completed - Success rate: %.2f%%", self.stats.success_rate)
        return documents


class FileParser(BaseParser):
    def __init__(
        self,
        config: Config,
        loader_class: type[BaseLoader],
        loader_kwargs: dict | None = None,
        file_type_name: str = "file",
    ):
        super().__init__(config)
        self.loader_class = loader_class
        self.loader_kwargs = loader_kwargs or {}
        self.file_type_name = file_type_name

    def _detect_encoding(self, file_path: str | Path) -> str | None:
        """Detect a text file's encoding with charset-normalizer.

        Returns the detected encoding name (logged at debug), or None if
        detection fails so the caller can fall back to latin-1.
        """
        from charset_normalizer import from_path

        best = from_path(file_path).best()
        if best is None:
            return None
        logger.debug("Detected encoding '%s' for '%s'", best.encoding, file_path)
        return best.encoding

    def _load_with_encoding(
        self, file_path: str | Path, encoding: str | None
    ) -> list[Any]:
        loader_kwargs = {**self.loader_kwargs, "file_path": str(file_path)}
        if encoding is not None and issubclass(
            self.loader_class, _ENCODING_AWARE_LOADERS
        ):
            loader_kwargs["encoding"] = encoding
        loader = self.loader_class(**loader_kwargs)
        return loader.load()

    @staticmethod
    def _is_decode_error(exc: BaseException) -> bool:
        # TextLoader/CSVLoader wrap a UnicodeDecodeError in a RuntimeError
        # ("Error loading <path>"), so inspect the exception and its cause chain.
        seen: set[int] = set()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen:
            if isinstance(cur, UnicodeDecodeError):
                return True
            seen.add(id(cur))
            cur = cur.__cause__ or cur.__context__
        return False

    def parse_file(
        self, file_path: str | Path, index_value: str | None = None
    ) -> Document:
        try:
            try:
                langchain_docs = self._load_with_encoding(file_path, None)
            except Exception as exc:
                # Non-UTF-8 file: detect and retry instead of dropping it. A
                # whole non-UTF-8 corpus would otherwise fail 100% silently.
                if not (
                    issubclass(self.loader_class, _ENCODING_AWARE_LOADERS)
                    and self._is_decode_error(exc)
                ):
                    raise
                detected = self._detect_encoding(file_path) or "latin-1"
                langchain_docs = self._load_with_encoding(file_path, detected)
            document = convert_langchain_to_document(
                langchain_docs, file_path, index_value=index_value
            )

            if (
                document.content
                and hasattr(document.content, "text")
                and isinstance(document.content.text, str)
                and document.content.text.strip() == ""
            ):
                raise DataProcessingError(
                    f"Parsed text is empty for '{self.file_type_name}' '{file_path}'"
                )

            return document
        except Exception as e:
            raise DataProcessingError(
                f"Failed to parse '{self.file_type_name}' '{file_path}': {e}"
            ) from e


# Text-based loaders accept an ``encoding`` kwarg. When a non-UTF-8 file
# (e.g. cp949, latin-1) raises UnicodeDecodeError, FileParser re-detects the
# encoding with charset-normalizer and retries with an explicit ``encoding=``.
# We deliberately do NOT use LangChain's ``autodetect_encoding=True`` because it
# imports ``chardet`` (an extra dependency); charset-normalizer is already
# vendored (it ships with requests) and is reused here and in
# ``Document._read_text_autodetect`` for one consistent detection path.
_ENCODING_AWARE_LOADERS = (CSVLoader, TextLoader)


class ParserFactory:
    # Loaders with no heavy optional deps — always available.
    _loader_configs: dict[str, tuple[type[BaseLoader], dict, str]] = {
        ".csv": (CSVLoader, {}, "CSV"),
        ".json": (JSONLoader, {"jq_schema": "."}, "JSON"),
        ".pdf": (PyPDFLoader, {}, "PDF"),
        ".txt": (TextLoader, {}, "Text"),
    }
    # Markdown/HTML need the optional `unstructured` package; register them only
    # when it is importable so the supported-format list matches runtime reality.
    if _UNSTRUCTURED_AVAILABLE:
        _loader_configs.update(
            {
                ".htm": (UnstructuredHTMLLoader, {}, "HTML"),
                ".html": (UnstructuredHTMLLoader, {}, "HTML"),
                ".markdown": (UnstructuredMarkdownLoader, {}, "Markdown"),
                ".md": (UnstructuredMarkdownLoader, {}, "Markdown"),
            }
        )

    @classmethod
    def create_parser(cls, file_path: str | Path, config: Config) -> BaseParser:
        extension = Path(file_path).suffix.lower()

        if extension not in cls._loader_configs:
            hint = ""
            if extension in {".md", ".markdown", ".htm", ".html"}:
                hint = (
                    " (.md/.html require the optional 'unstructured' package, which "
                    "is not installed)"
                )
            raise DataProcessingError(f"Unsupported file type: '{extension}'{hint}")

        loader_class, loader_kwargs, file_type_name = cls._loader_configs[extension]
        return FileParser(config, loader_class, loader_kwargs, file_type_name)

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        return list(cls._loader_configs.keys())
