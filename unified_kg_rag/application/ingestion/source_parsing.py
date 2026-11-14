# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reading source files into documents, once per change.

Each file's *source digest* (``compute_source_digest``: its bytes and the
parser that reads them) goes into its document's metadata, and the
doc-status registry records it with the document. An incremental run reads
only the files whose digest is not the one a PROCESSED record holds: for the
others an *unparsed* document stands in, carrying the record's content hash,
which is all the delta compares (``delta_detector.compute_content_hash``).
One that the delta still selects (new extraction settings, a reset) is read
before anything processes it (:meth:`SourceFileParser.read_unparsed`), so
an unparsed document never reaches a stage that reads its text.

Parsing runs in-process until the files left are expected to take longer
than starting a process pool, measured per file type as they are read (from
its second file: the first also pays its loader's imports): a corpus of
text files never starts one, one of PDFs parses in parallel.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

from unified_kg_rag.adapters.ingestion.parser import FileParser, ParserFactory
from unified_kg_rag.domain.ingestion.delta_detector import (
    assign_document_identity,
    assign_registry_source,
    compute_content_hash,
    document_doc_id,
)
from unified_kg_rag.domain.ingestion.document_identity import (
    REGISTRY_SOURCE_KEY,
    SOURCE_DIGEST_KEY,
    UNPARSED_CONTENT_HASH_KEY,
    compute_source_digest,
    is_unparsed,
    store_namespace_tag,
)
from unified_kg_rag.domain.models import (
    PENDING_CONTENT_HASH,
    Config,
    Constants,
    DocStatus,
    Document,
)
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.logging import error_log_text, log_error
from unified_kg_rag.shared.utils import default_max_workers
from unified_kg_rag.shared.utils.concurrency import process_pool, shut_down_after

logger = get_logger(__name__)

# Seconds a process pool takes to start and import the parsers: parsing the
# rest of the files in one pays off once they are expected to take longer
# in-process. A cold forkserver pool with the parser module preloaded
# started in 0.72 s on a 16-vCPU host (2026-11-08, 200 text files: 0.014 s
# in-process, 0.08 s in a started pool).
_POOL_START_SECONDS = 1.0

# Bytes of a source file hashed at a time (a large PDF is never held whole).
_READ_CHUNK_BYTES = 1 << 20

LoaderSpec = tuple[type, dict[str, Any], str]


def _parse(
    config: Config, spec: LoaderSpec, path: Path, index_value: str | None
) -> Document:
    loader_class, loader_kwargs, file_type_name = spec
    return FileParser(config, loader_class, loader_kwargs, file_type_name).parse_file(
        path, index_value
    )


# The run's config in a pool worker (_init_worker), so a task sends only its
# file and loader.
_WORKER_CONFIG: Config | None = None


def _init_worker(config: Config) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _parse_in_worker(
    spec: LoaderSpec, path: Path, index_value: str | None
) -> Document | str:
    """``path`` parsed in a pool worker, or why it failed (an exception of a
    loader may not pickle back): :func:`_failure`."""
    assert _WORKER_CONFIG is not None
    try:
        return _parse(_WORKER_CONFIG, spec, path, index_value)
    except Exception as e:  # noqa: BLE001 - reported as a failed file
        return _failure(e)


def _failure(error: Exception) -> str:
    """Why a file failed to parse, as the ERROR line reporting it may say it
    (``error_log_text``: a loader's message may quote the file's content);
    the message itself at DEBUG."""
    logger.debug("Parse failure: %s", error)
    return error_log_text(error)


# Source digests computed while the corpus fingerprint read each file
# (:func:`fingerprint_digest`), by path, inode, size, modification and
# change times, parser fingerprint and the encoding the digest keys
# (``ParserFactory.non_utf8_encoding``): the parsing stage reads each file
# once per run, not twice. The change time is set by every write and cannot
# be set back, so a file rewritten at the same size and modification time
# (a sync that copies an object's second-precision time) is read again.
_StatKey = tuple[str, int, int, int, int, str, str | None]
_SOURCE_DIGESTS: dict[_StatKey, str] = {}
_SOURCE_DIGESTS_HELD = 1 << 17


def _stat_key(path: Path, parser_fingerprint: str, encoding: str | None) -> _StatKey:
    stat = path.stat()
    return (
        str(path.resolve()),
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        parser_fingerprint,
        encoding,
    )


def fingerprint_digest(path: Path, config: Config | None = None) -> str:
    """``path``'s SHA-256 (hex) for the corpus fingerprint
    (``corpus_manifest_fingerprint``), with its source digest computed in
    the same read when a loader reads its extension (kept for
    :meth:`SourceFileParser.source_digest`; with ``config``, keyed with the
    encoding it configures for the extension, as that method keys it)."""
    extension = path.suffix.lower()
    try:
        parser_fingerprint: str | None = ParserFactory.parser_fingerprint(extension)
        encoding = (
            ParserFactory.non_utf8_encoding(extension, config)
            if config is not None
            else None
        )
    except Exception:  # noqa: BLE001 - no loader: no source digest to keep
        parser_fingerprint = encoding = None
    plain = hashlib.sha256()

    def read() -> Iterator[bytes]:
        with path.open("rb") as handle:
            while chunk := handle.read(_READ_CHUNK_BYTES):
                plain.update(chunk)
                yield chunk

    if parser_fingerprint is None:
        for _ in read():
            pass
        return plain.hexdigest()
    key = _stat_key(path, parser_fingerprint, encoding)
    source = compute_source_digest(read(), parser_fingerprint, encoding)
    if len(_SOURCE_DIGESTS) >= _SOURCE_DIGESTS_HELD:
        _SOURCE_DIGESTS.clear()
    _SOURCE_DIGESTS[key] = source
    return plain.hexdigest()


class SourceFileParser:
    """Parses a corpus's source files into identified documents.

    Every document gets its identity against ``source_directory``
    (``assign_document_identity``) and its source digest.
    """

    def __init__(self, config: Config, source_directory: Path) -> None:
        self.config = config
        self.source_directory = Path(source_directory)
        self._fingerprints: dict[str, tuple[str, str | None]] = {}

    @property
    def index_value(self) -> str | None:
        return self.config.processing.document_parsing.index_value

    def source_digest(self, path: Path) -> str:
        """``path``'s source digest (its bytes and its extension's parser,
        with the configured encoding it reads bytes in that are not UTF-8).

        Raises:
            OSError: The file cannot be read.
            DataProcessingError: No loader reads its extension.
        """
        extension = path.suffix.lower()
        reading = self._fingerprints.get(extension)
        if reading is None:
            reading = self._fingerprints[extension] = (
                ParserFactory.parser_fingerprint(extension),
                ParserFactory.non_utf8_encoding(extension, self.config),
            )
        fingerprint, encoding = reading
        if held := _SOURCE_DIGESTS.get(_stat_key(path, fingerprint, encoding)):
            return held
        with path.open("rb") as file:
            return compute_source_digest(
                iter(lambda: file.read(_READ_CHUNK_BYTES), b""),
                fingerprint,
                encoding,
            )

    def _identify(self, document: Document, digest: str | None) -> Document:
        if digest is not None:
            document.metadata[SOURCE_DIGEST_KEY] = digest
        assign_document_identity(
            document,
            self.source_directory,
            self.config.indexing.additional_suffix,
            store_namespace_tag(self.config.indexing),
        )
        return document

    def unparsed(self, path: Path, digest: str, content_hash: str) -> Document:
        """The document standing in for ``path``, not parsed: the identity
        a parse would give it, its source digest and ``content_hash``."""
        metadata: dict[str, Any] = {UNPARSED_CONTENT_HASH_KEY: content_hash}
        if self.index_value is not None:
            metadata[Constants.INDEX.value] = self.index_value
        document = Document(
            page_content="",
            document_id=f"unparsed:{path.name}",
            file_name=path.name,
            file_path=str(path),
            metadata=metadata,
        )
        return self._identify(document, digest)

    def parse(
        self, paths: Sequence[Path], digests: dict[Path, str] | None = None
    ) -> tuple[dict[Path, Document], list[str]]:
        """The documents of ``paths`` that parsed, by path, and the files
        that failed. ``digests`` are their source digests, computed here when
        one is missing (``None``: none is recorded)."""
        documents: dict[Path, Document] = {}
        failed: list[str] = []
        for path, result in zip(paths, self._parse_all(list(paths)), strict=True):
            if not isinstance(result, Document):
                logger.error("Failed to parse '%s': %s", path, result)
                failed.append(str(path))
                continue
            digest = None
            if digests is not None:
                digest = digests.get(path)
                if digest is None:
                    try:
                        digest = self.source_digest(path)
                    except OSError:
                        digest = None
            documents[path] = self._identify(result, digest)
        return documents, failed

    def _parse_all(self, paths: list[Path]) -> list[Document | str]:
        """Each file parsed, or why it failed: in-process while the files
        left would parse faster than a pool starts, then in a pool."""
        results: list[Document | str | None] = [None] * len(paths)
        left = Counter(path.suffix.lower() for path in paths)
        # Seconds and files parsed so far, per file type, after its first
        # file: that one also pays the imports and caches of its loader.
        spent: dict[str, tuple[float, int]] = {}
        warmed: set[str] = set()
        workers = default_max_workers()
        for index, path in enumerate(paths):
            remaining = len(paths) - index
            expected = sum(
                count * spent[ext][0] / spent[ext][1]
                for ext, count in left.items()
                if ext in spent
            )
            if workers > 1 and remaining > 1 and expected > _POOL_START_SECONDS:
                logger.info(
                    "Parsing %d files in %d processes (about %.0f s in one)",
                    remaining,
                    min(workers, remaining),
                    expected,
                )
                self._parse_in_pool(paths, results, index, min(workers, remaining))
                break
            extension = path.suffix.lower()
            start = time.perf_counter()
            results[index] = self._parse_here(path)
            if extension in warmed:
                seconds, count = spent.get(extension, (0.0, 0))
                spent[extension] = (seconds + time.perf_counter() - start, count + 1)
            warmed.add(extension)
            left[extension] -= 1
        return [result if result is not None else "not parsed" for result in results]

    def _parse_here(self, path: Path) -> Document | str:
        try:
            parser = ParserFactory.create_parser(path, self.config)
            return parser.parse_file(path, self.index_value)
        except Exception as e:  # noqa: BLE001 - reported as a failed file
            return _failure(e)

    def _parse_in_pool(
        self,
        paths: list[Path],
        results: list[Document | str | None],
        start: int,
        workers: int,
    ) -> None:
        """Parse ``paths[start:]`` into ``results`` in a pool of ``workers``.

        A file whose task does not reach a worker (its loader does not
        pickle: a class defined in a script) is parsed in-process, as is
        every file left when the pool breaks (a worker killed). A loader's
        own error comes back as the worker's answer (``_parse_in_worker``).
        """
        futures: dict[int, Future[Document | str]] = {}
        try:
            with shut_down_after(
                process_pool(
                    workers,
                    initializer=_init_worker,
                    initargs=(self.config,),
                    preload=(__name__,),
                )
            ) as executor:
                for index in range(start, len(paths)):
                    try:
                        spec = ParserFactory.loader_spec(paths[index].suffix)
                    except Exception:  # noqa: BLE001 - its parse reports it
                        continue
                    futures[index] = executor.submit(
                        _parse_in_worker, spec, paths[index], self.index_value
                    )
                for index, future in futures.items():
                    try:
                        results[index] = future.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as e:  # noqa: BLE001 - parsed in-process
                        logger.debug(
                            "Parsing '%s' in-process: its task did not reach a "
                            "worker (%s)",
                            paths[index],
                            e,
                        )
        except BrokenProcessPool as e:
            log_error(
                logger,
                logging.WARNING,
                "The parsing processes stopped (%s): parsing the rest in-process",
                error=e,
            )
        for index in range(start, len(paths)):
            if results[index] is None:
                results[index] = self._parse_here(paths[index])

    def read_unparsed(
        self,
        documents: list[Document],
        selected: Callable[[Document], bool] = lambda _: True,
    ) -> tuple[list[Document], dict[str, str]]:
        """``documents`` with the unparsed ones ``selected`` accepts parsed,
        each in its place, and the content hash of each of those by its
        registry id. A parsed document keeps the registry source scope its
        stand-in had. One that fails to parse is dropped (logged as an
        error)."""
        wanted = {
            index
            for index, document in enumerate(documents)
            if is_unparsed(document) and selected(document)
        }
        if not wanted:
            return documents, {}
        logger.info(
            "Reading %d unchanged source files this run processes again",
            len(wanted),
        )
        stubs = [documents[index] for index in sorted(wanted)]
        results = iter(self._parse_all([Path(stub.file_path) for stub in stubs]))
        out: list[Document] = []
        hashes: dict[str, str] = {}
        for index, document in enumerate(documents):
            if index not in wanted:
                out.append(document)
                continue
            result = next(results)
            if not isinstance(result, Document):
                logger.error("Failed to parse '%s': %s", document.file_path, result)
                continue
            read = self._identify(result, document.metadata.get(SOURCE_DIGEST_KEY))
            if (source := document.metadata.get(REGISTRY_SOURCE_KEY)) is not None:
                assign_registry_source(read, str(source))
            hashes[document_doc_id(read)] = compute_content_hash(read)
            out.append(read)
        return out, hashes


def reusable_records(
    parser: SourceFileParser,
    digests: dict[Path, str],
    store: Any,
    source_scope: str,
) -> dict[Path, Document]:
    """The unparsed documents of the files whose source digest (in
    ``digests``) a PROCESSED record of ``store`` (a ``DocStatusPort``)
    holds: those files need no parse. One batched read."""
    probes: dict[Path, Document] = {}
    for path, digest in digests.items():
        probe = parser.unparsed(path, digest, PENDING_CONTENT_HASH)
        assign_registry_source(probe, source_scope)
        probes[path] = probe
    stored = store.get_many({document_doc_id(p) for p in probes.values()})
    reusable: dict[Path, Document] = {}
    for path, probe in probes.items():
        record = stored.get(document_doc_id(probe))
        if (
            record is not None
            and record.status is DocStatus.PROCESSED
            and record.content_hash != PENDING_CONTENT_HASH
            and record.source_digest == digests[path]
        ):
            # Its id derives from the content hash: assigned again with the
            # record's.
            probe.metadata[UNPARSED_CONTENT_HASH_KEY] = record.content_hash
            assign_registry_source(probe, source_scope)
            reusable[path] = probe
    return reusable
