# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langchain_core.document_loaders import BaseLoader

from unified_kg_rag.adapters.ingestion.document_converter import (
    convert_langchain_to_document,
)
from unified_kg_rag.adapters.ingestion.file_loaders import (
    CsvFileLoader,
    PdfFileLoader,
    TextFileLoader,
    UnstructuredPartitionLoader,
)
from unified_kg_rag.adapters.ingestion.json_loader import JsonTextLoader
from unified_kg_rag.domain.models import Config, Document
from unified_kg_rag.shared import DataProcessingError, get_logger
from unified_kg_rag.shared.exceptions import causes

logger = get_logger(__name__)

# UnstructuredPartitionLoader imports fine but raises at .load() time if the
# heavy optional `unstructured` package is absent (it is not a runtime
# dependency). Detect it once so .md/.html/.htm are only advertised as
# supported when they can actually be parsed — instead of failing mid-ingestion
# with "No module named 'unstructured'".
_UNSTRUCTURED_AVAILABLE = importlib.util.find_spec("unstructured") is not None
# The formats the optional extra adds, and how to install it.
UNSTRUCTURED_EXTENSIONS = frozenset({".htm", ".html", ".markdown", ".md"})
UNSTRUCTURED_INSTALL_HINT = (
    "install the optional 'unstructured' extra: "
    "`uv sync --extra unstructured` or "
    "`pip install -e '.[unstructured]'` in a source checkout"
)


class BaseParser(ABC):
    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def parse_file(
        self, file_path: str | Path, index_value: str | None = None
    ) -> Document:
        pass


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
        # A registered loader may wrap the UnicodeDecodeError (langchain-community
        # loaders raise RuntimeError("Error loading <path>") from it), so inspect
        # the exception and its cause chain.
        return any(isinstance(cause, UnicodeDecodeError) for cause in causes(exc))

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

            if not document.page_content.strip():
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
# Detection runs only after a decode error, so a file that decodes keeps the
# text it always had (charset-normalizer is a declared dependency, reused in
# ``Document._read_text_autodetect`` for one consistent detection path).
_ENCODING_AWARE_LOADERS = (CsvFileLoader, JsonTextLoader, TextFileLoader)


class ParserFactory:
    # Loaders with no heavy optional deps — always available.
    _loader_configs: dict[str, tuple[type[BaseLoader], dict, str]] = {
        ".csv": (CsvFileLoader, {}, "CSV"),
        # Stdlib-only: LangChain's JSONLoader needs the optional `jq` package.
        ".json": (JsonTextLoader, {}, "JSON"),
        ".pdf": (PdfFileLoader, {}, "PDF"),
        ".txt": (TextFileLoader, {}, "Text"),
    }
    # Markdown/HTML need the optional `unstructured` package; register them only
    # when it is importable so the supported-format list matches runtime reality.
    if _UNSTRUCTURED_AVAILABLE:
        _loader_configs.update(
            {
                ".htm": (UnstructuredPartitionLoader, {"partition": "html"}, "HTML"),
                ".html": (UnstructuredPartitionLoader, {"partition": "html"}, "HTML"),
                ".markdown": (
                    UnstructuredPartitionLoader,
                    {"partition": "md"},
                    "Markdown",
                ),
                ".md": (UnstructuredPartitionLoader, {"partition": "md"}, "Markdown"),
            }
        )

    @classmethod
    def register_loader(
        cls,
        extension: str,
        loader_class: type[BaseLoader],
        loader_kwargs: dict | None = None,
        file_type_name: str | None = None,
    ) -> None:
        """Register a custom loader for a file extension (extension seam).

        Any ``langchain_core`` ``BaseLoader`` subclass works — ``FileParser``
        only calls ``loader_class(file_path=..., **loader_kwargs).load()``. This
        lets a user add a new format (or override a built-in one) without
        editing the factory::

            from langchain_core.document_loaders import BaseLoader

            class XmlTextLoader(BaseLoader):
                def __init__(self, file_path): ...
                def lazy_load(self): ...  # yield langchain Documents

            ParserFactory.register_loader(".xml", XmlTextLoader)

        The new extension is automatically picked up by
        ``get_supported_extensions()`` and by the loader's discovery filter, so
        files of that type are then both discovered and parsed.

        Args:
            extension: file extension including the leading dot (case-insensitive).
            loader_class: a ``langchain_core`` ``BaseLoader`` subclass.
            loader_kwargs: kwargs passed to the loader constructor (besides the
                file path), e.g. ``{"text_mode": "strings"}`` for ``JsonTextLoader``.
            file_type_name: human-readable name for logs (defaults to the
                extension without its dot, upper-cased).
        """
        ext = extension.lower()
        if not ext.startswith("."):
            raise ValueError(
                f"extension must include the leading dot, got '{extension}'"
            )
        if not (
            isinstance(loader_class, type) and issubclass(loader_class, BaseLoader)
        ):
            raise TypeError(
                f"loader_class must be a langchain BaseLoader subclass, got "
                f"{loader_class!r}"
            )
        cls._loader_configs[ext] = (
            loader_class,
            loader_kwargs or {},
            file_type_name or ext.lstrip(".").upper(),
        )

    @classmethod
    def create_parser(cls, file_path: str | Path, config: Config) -> BaseParser:
        extension = Path(file_path).suffix.lower()

        if extension not in cls._loader_configs:
            hint = ""
            if extension in UNSTRUCTURED_EXTENSIONS:
                hint = f"; to parse it, {UNSTRUCTURED_INSTALL_HINT}"
            raise DataProcessingError(f"Unsupported file type: '{extension}'{hint}")

        loader_class, loader_kwargs, file_type_name = cls._loader_configs[extension]
        return FileParser(config, loader_class, loader_kwargs, file_type_name)

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        return list(cls._loader_configs.keys())
