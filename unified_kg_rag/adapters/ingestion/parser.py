# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import hashlib
import importlib.util
import inspect
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from langchain_core.document_loaders import BaseLoader

from unified_kg_rag.adapters.ingestion.document_converter import (
    convert_langchain_to_document,
)
from unified_kg_rag.adapters.ingestion.file_loaders import (
    DOCX_INSTALL_HINT,
    CsvFileLoader,
    DocxFileLoader,
    HtmlFileLoader,
    PdfFileLoader,
    TextFileLoader,
    UnstructuredPartitionLoader,
)
from unified_kg_rag.adapters.ingestion.json_loader import JsonTextLoader
from unified_kg_rag.domain.models import Config, Document
from unified_kg_rag.shared import DataProcessingError, get_logger
from unified_kg_rag.shared.exceptions import causes

logger = get_logger(__name__)

# Markdown and HTML are read in the core install (Markdown as written, HTML
# through the standard library's parser); with the optional `unstructured`
# extra installed, its partitioners read them instead (the richer parser:
# Markdown rendered to its text, HTML elements by their layout). Detected once:
# its loader imports fine without the package and would fail at .load().
_UNSTRUCTURED_AVAILABLE = importlib.util.find_spec("unstructured") is not None
# .docx needs the optional `docx` extra (python-docx), registered only when it
# is importable; without it a .docx is skipped naming the extra.
_DOCX_AVAILABLE = importlib.util.find_spec("docx") is not None
# The formats an optional extra adds that this install cannot read, and how
# to install the extra.
MISSING_EXTRA_FORMATS: dict[str, str] = (
    {} if _DOCX_AVAILABLE else {".docx": DOCX_INSTALL_HINT}
)


class FileParser:
    """Parses one source file into a ``Document`` with a registered loader
    (``ParserFactory``), retrying a decode error of a text format with the
    encoding charset-normalizer detects."""

    def __init__(
        self,
        config: Config,
        loader_class: type[BaseLoader],
        loader_kwargs: dict[str, Any] | None = None,
        file_type_name: str = "file",
    ):
        self.config = config
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
_ENCODING_AWARE_LOADERS = (
    CsvFileLoader,
    HtmlFileLoader,
    JsonTextLoader,
    TextFileLoader,
)


class ParserFactory:
    """The file-extension registry of document loaders.

    Maps each extension (``.pdf``, ``.txt``, ...) to a LangChain loader;
    ``create_parser`` builds the parser for one file and
    ``get_supported_extensions`` lists what ingestion discovers. Add or
    override a format with :meth:`register_loader`.
    """

    # Loaders with no heavy optional deps — always available.
    _loader_configs: dict[str, tuple[type[BaseLoader], dict[str, Any], str]] = {
        ".csv": (CsvFileLoader, {}, "CSV"),
        ".htm": (HtmlFileLoader, {}, "HTML"),
        ".html": (HtmlFileLoader, {}, "HTML"),
        # Stdlib-only: LangChain's JSONLoader needs the optional `jq` package.
        ".json": (JsonTextLoader, {}, "JSON"),
        ".markdown": (TextFileLoader, {}, "Markdown"),
        ".md": (TextFileLoader, {}, "Markdown"),
        ".pdf": (PdfFileLoader, {}, "PDF"),
        ".txt": (TextFileLoader, {}, "Text"),
    }
    if _DOCX_AVAILABLE:
        _loader_configs[".docx"] = (DocxFileLoader, {}, "Word")
    # The richer Markdown/HTML parser, in place of the core ones, when the
    # optional `unstructured` package is importable.
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
        loader_kwargs: dict[str, Any] | None = None,
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
    def create_parser(cls, file_path: str | Path, config: Config) -> FileParser:
        extension = Path(file_path).suffix.lower()

        if extension not in cls._loader_configs:
            hint = ""
            if extension in MISSING_EXTRA_FORMATS:
                hint = f"; to parse it, {MISSING_EXTRA_FORMATS[extension]}"
            raise DataProcessingError(f"Unsupported file type: '{extension}'{hint}")

        loader_class, loader_kwargs, file_type_name = cls._loader_configs[extension]
        return FileParser(config, loader_class, loader_kwargs, file_type_name)

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        return list(cls._loader_configs.keys())

    @classmethod
    def loader_spec(
        cls, extension: str
    ) -> tuple[type[BaseLoader], dict[str, Any], str]:
        """The ``(loader_class, loader_kwargs, file_type_name)`` registered for
        ``extension``, what :meth:`create_parser` builds a parser from.

        Raises:
            DataProcessingError: No loader reads the extension.
        """
        ext = extension.lower()
        if ext not in cls._loader_configs:
            raise DataProcessingError(f"Unsupported file type: '{ext}'")
        return cls._loader_configs[ext]

    @classmethod
    def parser_fingerprint(cls, extension: str) -> str:
        """What a file of ``extension`` parses to depends on, for its source
        digest (``compute_source_digest``): this package's version and parser
        modules, the registered loader (class, module source, kwargs) and the
        installed versions of the parsing libraries. An upgrade of any of
        them parses every file again once.

        Raises:
            DataProcessingError: No loader reads the extension.
        """
        loader_class, loader_kwargs, file_type_name = cls.loader_spec(extension)
        return _fingerprint(
            loader_class, repr(sorted(loader_kwargs.items())), file_type_name
        )


# The libraries the built-in loaders read files with; their versions are
# part of every parser fingerprint.
_PARSING_DISTRIBUTIONS = (
    "charset-normalizer",
    "langchain-core",
    "pypdf",
    "python-docx",
    "unstructured",
)


@functools.lru_cache(maxsize=32)
def _fingerprint(
    loader_class: type[BaseLoader], loader_kwargs: str, file_type_name: str
) -> str:
    from unified_kg_rag import __version__

    sources = []
    for module in {
        __name__,
        convert_langchain_to_document.__module__,
        loader_class.__module__,
    }:
        try:
            sources.append(Path(inspect.getfile(sys.modules[module])).read_bytes())
        except (KeyError, OSError, TypeError):
            sources.append(module.encode("utf-8"))
    versions = []
    for distribution in _PARSING_DISTRIBUTIONS:
        try:
            versions.append(f"{distribution}={metadata.version(distribution)}")
        except metadata.PackageNotFoundError:
            versions.append(f"{distribution}=-")
    digest = hashlib.sha256(
        repr(
            (
                __version__,
                f"{loader_class.__module__}.{loader_class.__qualname__}",
                loader_kwargs,
                file_type_name,
                versions,
            )
        ).encode("utf-8")
    )
    for source in sorted(sources):
        digest.update(source)
    return digest.hexdigest()
