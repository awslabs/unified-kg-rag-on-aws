# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import hashlib
import inspect
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, NamedTuple

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
from unified_kg_rag.shared.registry import Registry, module_installed

logger = get_logger(__name__)

# Markdown and HTML are read in the core install (Markdown as written, HTML
# through the standard library's parser); with the optional `unstructured`
# extra installed, its partitioners read them instead (the richer parser:
# Markdown rendered to its text, HTML elements by their layout). A preference,
# not a requirement, detected once: its loader imports fine without the
# package and would fail at .load().
_UNSTRUCTURED_AVAILABLE = module_installed("unstructured")


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

    def _detect_encoding(self, file_path: str | Path) -> str:
        """The encoding a text file that is not UTF-8 is read in: the
        configured ``processing.document_parsing.encoding``, else the one
        charset-normalizer detects (logged at debug).

        Raises:
            DataProcessingError: Nothing is configured and no encoding is
                detected: the bytes are not text (a binary file with a text
                suffix), which latin-1 would decode into garbage text units.
        """
        configured = self.config.processing.document_parsing.encoding
        if configured is not None:
            return configured
        from charset_normalizer import from_path

        best = from_path(file_path).best()
        if best is None:
            raise DataProcessingError(
                "not text in any detected encoding (set "
                "processing.document_parsing.encoding to read it in one)"
            )
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
                langchain_docs = self._load_with_encoding(
                    file_path, self._detect_encoding(file_path)
                )
            document = convert_langchain_to_document(
                langchain_docs, file_path, index_value=index_value
            )

            if not document.page_content.strip():
                raise DataProcessingError(
                    f"Parsed text is empty for '{self.file_type_name}' '{file_path}'"
                )

            return document
        except Exception as e:
            raise DataProcessingError.caused_by(
                f"Failed to parse '{self.file_type_name}' '{file_path}'", e
            ) from e


# Text-based loaders accept an ``encoding`` kwarg. When a non-UTF-8 file
# (e.g. cp949, latin-1) raises UnicodeDecodeError, FileParser retries with an
# explicit ``encoding=``: the configured one, else the one charset-normalizer
# detects (none detected: the file fails, it is not text).
# Detection runs only after a decode error, so a file that decodes keeps the
# text it always had (charset-normalizer is a declared dependency, reused by
# ``Document.from_json_file``, which also fails on bytes it detects no
# encoding of: one rule for what is text).
_ENCODING_AWARE_LOADERS = (
    CsvFileLoader,
    HtmlFileLoader,
    JsonTextLoader,
    TextFileLoader,
)


class LoaderSpec(NamedTuple):
    """What a file extension is read with: ``loader_class(file_path=...,
    **loader_kwargs)``, named ``file_type_name`` in logs."""

    loader_class: type[BaseLoader]
    loader_kwargs: dict[str, Any]
    file_type_name: str


# The document loaders by file extension, under the one registry rule
# (shared.registry); a loader of an optional extra declares it (requires=),
# and a file of its extension is skipped naming the extra when it is missing.
_LOADERS: Registry[LoaderSpec] = Registry(
    "file loader", pattern=re.compile(r"\.[a-z0-9][a-z0-9_+-]*")
)
for _extension, _spec in {
    ".csv": LoaderSpec(CsvFileLoader, {}, "CSV"),
    ".htm": LoaderSpec(HtmlFileLoader, {}, "HTML"),
    ".html": LoaderSpec(HtmlFileLoader, {}, "HTML"),
    # Stdlib-only: LangChain's JSONLoader needs the optional `jq` package.
    ".json": LoaderSpec(JsonTextLoader, {}, "JSON"),
    ".markdown": LoaderSpec(TextFileLoader, {}, "Markdown"),
    ".md": LoaderSpec(TextFileLoader, {}, "Markdown"),
    ".pdf": LoaderSpec(PdfFileLoader, {}, "PDF"),
    ".txt": LoaderSpec(TextFileLoader, {}, "Text"),
    # The richer Markdown/HTML parser in place of the core ones.
    **(
        {
            ".htm": LoaderSpec(
                UnstructuredPartitionLoader, {"partition": "html"}, "HTML"
            ),
            ".html": LoaderSpec(
                UnstructuredPartitionLoader, {"partition": "html"}, "HTML"
            ),
            ".markdown": LoaderSpec(
                UnstructuredPartitionLoader, {"partition": "md"}, "Markdown"
            ),
            ".md": LoaderSpec(
                UnstructuredPartitionLoader, {"partition": "md"}, "Markdown"
            ),
        }
        if _UNSTRUCTURED_AVAILABLE
        else {}
    ),
}.items():
    _LOADERS.register(_extension, _spec)
_LOADERS.register(
    ".docx",
    LoaderSpec(DocxFileLoader, {}, "Word"),
    requires=("docx", DOCX_INSTALL_HINT),
)


class ParserFactory:
    """The file-extension registry of document loaders.

    Maps each extension (``.pdf``, ``.txt``, ...) to a LangChain loader;
    ``create_parser`` builds the parser for one file and
    ``get_supported_extensions`` lists what ingestion discovers. Add or
    override a format with :meth:`register_loader`.
    """

    @classmethod
    def register_loader(
        cls,
        extension: str,
        loader_class: type[BaseLoader],
        loader_kwargs: dict[str, Any] | None = None,
        file_type_name: str | None = None,
        *,
        requires: tuple[str, str] | None = None,
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
            extension: file extension including the leading dot
                (case-insensitive; a dot, then letters, digits, ``_``, ``+``
                or ``-``).
            loader_class: a ``langchain_core`` ``BaseLoader`` subclass.
            loader_kwargs: kwargs passed to the loader constructor (besides the
                file path), e.g. ``{"text_mode": "strings"}`` for ``JsonTextLoader``.
            file_type_name: human-readable name for logs (defaults to the
                extension without its dot, upper-cased).
            requires: ``(module, how to install it)`` of the optional package
                the loader imports: without it, a file of the extension is
                skipped with that hint.
        """
        ext = _LOADERS.check_name(extension)
        if not (
            isinstance(loader_class, type) and issubclass(loader_class, BaseLoader)
        ):
            raise TypeError(
                f"loader_class must be a langchain BaseLoader subclass, got "
                f"{loader_class!r}"
            )
        _LOADERS.register(
            ext,
            LoaderSpec(
                loader_class,
                loader_kwargs or {},
                file_type_name or ext.lstrip(".").upper(),
            ),
            requires=requires,
        )

    @classmethod
    def create_parser(cls, file_path: str | Path, config: Config) -> FileParser:
        loader_class, loader_kwargs, file_type_name = cls.loader_spec(
            Path(file_path).suffix
        )
        return FileParser(config, loader_class, loader_kwargs, file_type_name)

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """The extensions a loader of this install reads."""
        return [ext for ext in _LOADERS if _LOADERS.missing(ext) is None]

    @staticmethod
    def install_hint(extension: str) -> str | None:
        """How to install the extra that reads ``extension``, when a loader
        of it declares one this install lacks."""
        missing = _LOADERS.missing(extension)
        return None if missing is None else missing[1]

    @classmethod
    def loader_spec(cls, extension: str) -> LoaderSpec:
        """The ``(loader_class, loader_kwargs, file_type_name)`` registered for
        ``extension``, what :meth:`create_parser` builds a parser from.

        Raises:
            DataProcessingError: No loader of this install reads the
                extension (naming the extra to install, if one does).
        """
        ext = extension.lower()
        spec = _LOADERS.get(ext)
        hint = cls.install_hint(ext)
        if spec is None or hint is not None:
            remedy = f"; to parse it, {hint}" if hint else ""
            raise DataProcessingError(f"Unsupported file type: '{ext}'{remedy}")
        return spec

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

    @classmethod
    def non_utf8_encoding(cls, extension: str, config: Config) -> str | None:
        """The configured encoding a file of ``extension`` that does not
        decode as UTF-8 is read in (``processing.document_parsing.encoding``),
        for its source digest; ``None`` when none is configured (detected per
        file) or its loader takes no encoding.

        Raises:
            DataProcessingError: No loader reads the extension.
        """
        loader_class, _, _ = cls.loader_spec(extension)
        if not issubclass(loader_class, _ENCODING_AWARE_LOADERS):
            return None
        return config.processing.document_parsing.encoding


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
