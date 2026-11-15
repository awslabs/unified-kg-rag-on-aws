# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import logging
import re
import time
import unicodedata
from collections.abc import Callable
from concurrent.futures import as_completed
from functools import partial
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from unified_kg_rag.domain.ingestion.base_resolver import jaccard, lsh_params
from unified_kg_rag.domain.ingestion.cache_keys import is_excluded_source_path
from unified_kg_rag.domain.languages import profiles
from unified_kg_rag.domain.models import Document
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import ConfigurationError
from unified_kg_rag.shared.logging import log_failure
from unified_kg_rag.shared.utils import default_max_workers
from unified_kg_rag.shared.utils.concurrency import (
    ContextThreadPoolExecutor,
    preload_in_forkserver,
    process_pool,
    shut_down_after,
)
from unified_kg_rag.shared.utils.scripts import script_named, script_of

logger = get_logger(__name__)

# Its pool's tasks live here: imported by the forkserver whichever pool of
# the run (document loading, graph resolution) starts it.
preload_in_forkserver(__name__)


def document_shingles(content: str, n_grams: int) -> frozenset[str]:
    """The character ``n_grams`` of ``content`` (empty for a text shorter
    than one): what deduplication compares."""
    return frozenset(
        content[j : j + n_grams] for j in range(len(content) - n_grams + 1)
    )


def compute_minhash(
    doc_info: tuple[int, str], num_permutations: int, n_grams: int
) -> tuple[int, MinHash, str] | None:
    """A document's MinHash of its character ``n_grams`` and the digest of
    its numbers (:func:`numbers_digest`), or None for a text shorter than
    one shingle."""
    doc_index, content = doc_info
    shingles = document_shingles(content or "", n_grams)
    if not shingles:
        return None

    minhash = MinHash(num_perm=num_permutations)
    minhash.update_batch([shingle.encode("utf8") for shingle in shingles])

    return doc_index, minhash, numbers_digest(content)


_HAN = script_named("han")


def _han_numerals() -> str:
    """The Han numerals of the language profiles (``counter_numerals``,
    ``numeral_multipliers``; 〇, a numeral of no script's letters, too)."""
    return "".join(
        dict.fromkeys(
            ch
            for language in profiles()
            for ch in (*language.counter_numerals, *language.numeral_multipliers)
            if len(ch) == 1 and script_of(ch) in (_HAN, None)
        )
    )


# A run of digits of any script, or of Han numerals (三万, 二〇二六, 三ヶ月,
# 壹拾万). One numeral alone is read too, whatever follows it: the units a
# number stands before are an open list (月, ヶ月, 平方米, 折, ...). It is
# also part of ordinary words (一起 "together", 统一 "unify"), so an edit of
# such a word keeps a copy: an extra document indexed, not one lost.
# Hangul number words (삼백만) are ordinary syllables and are not read.
_NUMBER = re.compile(rf"\d+|[{_han_numerals()}]+")


def _number(run: str) -> str:
    """``run`` with every decimal digit as its ASCII digit (Arabic-Indic,
    Devanagari, ... digits read as the same number; zeros kept)."""
    return "".join(str(unicodedata.digit(ch)) if ch.isdecimal() else ch for ch in run)


def numbers_digest(text: str) -> str:
    """A digest of the numbers ``text`` holds, in their order: two documents
    with the same numbers (amounts, dates, ids) in the same order share it.

    Template documents (two contracts of one form) differ in a few amounts
    and dates within a text of thousands of characters, so their shingle
    similarity alone reads them as copies of each other; two dates or line
    amounts swapped make another document too.
    """
    numbers = [
        _number(run) for run in _NUMBER.findall(unicodedata.normalize("NFKC", text))
    ]
    return hashlib.sha256("\x00".join(numbers).encode("utf-8")).hexdigest()


# Near-duplicate detection of ``processing.deduplicate``: an LSH index of
# 128-permutation MinHashes of the documents' character 3-gram sets proposes
# pairs (the resolver's layout, ``lsh_params``), their exact Jaccard
# similarity decides.
DEDUP_MINHASH_PERMUTATIONS = 128
DEDUP_N_GRAMS = 3
# The dropped documents a WARNING names at most (the rest are counted).
_DROPPED_NAMED = 20


class DirectoryLoader:
    """Load pre-parsed ``Document`` JSON files from a directory (recursively).

    Used by the loading stage when the parsing stage did not run; the stage
    also calls :meth:`deduplicate_documents` on the run's documents.
    """

    SUPPORTED_EXTENSIONS = frozenset({".json"})

    def __init__(self, source_directory: str | Path):
        self.source_directory = Path(source_directory).resolve()
        self.max_workers = default_max_workers()
        self.failed_files: list[str] = []

    def load(self) -> list[Document]:
        start_time = time.time()
        logger.info("Starting document loading from: '%s'", self.source_directory)

        self._validate_directory()
        discovered_files = self.discover_files()

        if not discovered_files:
            logger.warning("No supported files found in '%s'.", self.source_directory)
            return []

        logger.info("Discovered %s files", len(discovered_files))

        docs, failed_paths = self._load_files_concurrently(discovered_files)
        self.failed_files = sorted(map(str, failed_paths))

        success_rate = (
            (len(docs) / len(discovered_files)) * 100 if discovered_files else 0
        )
        elapsed_time = time.time() - start_time

        logger.info(
            "Document loading completed in %.2fs. Success: %s/%s (%.1f%%)",
            elapsed_time,
            len(docs),
            len(discovered_files),
            success_rate,
        )

        if self.failed_files:
            logger.warning("Failed to load %s files", len(self.failed_files))

        return docs

    def _validate_directory(self) -> None:
        if not self.source_directory.exists():
            raise FileNotFoundError(
                f"Directory does not exist: '{self.source_directory}'"
            )
        if not self.source_directory.is_dir():
            raise ConfigurationError(
                f"Path is not a valid directory: '{self.source_directory}'"
            )

    def discover_files(self) -> list[Path]:
        resolved_root = self.source_directory.resolve()
        files = [
            p
            for p in self.source_directory.rglob("*")
            if self._is_valid_file(p, resolved_root)
        ]
        return sorted(files)

    def _is_valid_file(self, path: Path, resolved_root: Path | None = None) -> bool:
        # Hidden paths and __pycache__ follow the rule discovery and the
        # corpus fingerprint use; only the part below the root is tested, so
        # a hidden ancestor of the source directory does not hide the corpus.
        return (
            path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            and not is_excluded_source_path(path, self.source_directory, resolved_root)
        )

    def _load_files_concurrently(
        self, file_paths: list[Path]
    ) -> tuple[list[Document], list[Path]]:
        """Load every file; documents come back in ``file_paths`` order.

        Completion order would make which copy deduplication keeps, and the
        order of everything downstream, vary from run to run.
        """
        documents, failed_paths = [], []

        with ContextThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                (path, executor.submit(self._load_and_enrich_single, path))
                for path in file_paths
            ]
            for path, future in futures:
                try:
                    if doc := future.result():
                        documents.append(doc)
                    else:
                        failed_paths.append(path)
                except Exception as e:  # noqa: BLE001 - isolate one file's failure
                    log_failure(
                        logger,
                        logging.ERROR,
                        "Error processing file '%s': %s",
                        path,
                        error=e,
                        traceback=True,
                    )
                    failed_paths.append(path)

        return documents, failed_paths

    def _load_and_enrich_single(self, file_path: Path) -> Document | None:
        try:
            doc = self.load_single(file_path)
            doc.metadata.update(
                {
                    "source_directory": str(self.source_directory),
                    "relative_path": str(file_path.relative_to(self.source_directory)),
                    "file_extension": file_path.suffix,
                    "file_size": file_path.stat().st_size,
                }
            )
            return doc
        except Exception as e:  # noqa: BLE001 - isolate one file's failure
            log_failure(
                logger,
                logging.WARNING,
                "Failed to load document '%s': %s",
                file_path,
                error=e,
            )
            return None

    def load_single(self, file_path: Path) -> Document:
        return Document.from_json_file(file_path)

    def deduplicate_documents(
        self,
        documents: list[Document],
        *,
        threshold: float,
        keep: Callable[[Document], bool] | None = None,
        avoid: Callable[[Document], bool] | None = None,
    ) -> list[Document]:
        """MinHash near-duplicate removal for documents loaded elsewhere.

        Two documents are near-duplicates when their 3-gram similarity is
        at least ``threshold`` and they hold the same numbers
        (:func:`numbers_digest`): a form filled in twice is two documents.
        Of two near-duplicates the earlier one is kept, except that a document
        ``keep`` accepts wins over one it rejects (an incremental run keeps
        the already-indexed copy) and one ``avoid`` accepts loses to any other
        (a copy whose indexing failed). The survivors keep their input order;
        a WARNING names each dropped document and the one it copies.
        """
        if len(documents) < 2:
            return documents

        logger.info("Starting deduplication for %s documents", len(documents))

        signatures = self._compute_minhashes(documents)
        if not signatures:
            logger.warning("No MinHashes computed, skipping deduplication")
            return documents

        # Rank = keep order: kept documents first, avoided ones last, then
        # input order.
        order = sorted(
            range(len(documents)),
            key=lambda i: (
                not (keep and keep(documents[i])),
                bool(avoid and avoid(documents[i])),
            ),
        )
        rank = {doc_index: position for position, doc_index in enumerate(order)}
        duplicates = self._find_duplicates(
            signatures, [d.page_content for d in documents], rank, threshold
        )
        if not duplicates:
            logger.info("No duplicates found")
            return documents
        duplicate_indices = set(duplicates)
        named = [
            f"'{_label(documents[index])}' (of '{_label(documents[original])}', "
            f"similarity {similarity:.2f})"
            for index, (original, similarity) in sorted(duplicates.items())
        ]
        logger.warning(
            "Deduplication dropped %d near-duplicate document(s) "
            "(processing.deduplicate_similarity_threshold %.2f): %s%s",
            len(named),
            threshold,
            ", ".join(named[:_DROPPED_NAMED]),
            (
                f" and {len(named) - _DROPPED_NAMED} more"
                if len(named) > _DROPPED_NAMED
                else ""
            ),
        )

        unique_docs = [
            doc for i, doc in enumerate(documents) if i not in duplicate_indices
        ]

        logger.info(
            "Deduplication completed: removed %s duplicates, kept %s unique documents",
            len(duplicate_indices),
            len(unique_docs),
        )

        return unique_docs

    def _compute_minhashes(
        self, documents: list[Document]
    ) -> dict[int, tuple[MinHash, str]]:
        """Each document's MinHash and numbers digest, by its index."""
        minhashes: dict[int, tuple[MinHash, str]] = {}
        compute_func = partial(
            compute_minhash,
            num_permutations=DEDUP_MINHASH_PERMUTATIONS,
            n_grams=DEDUP_N_GRAMS,
        )

        with shut_down_after(
            process_pool(self.max_workers, preload=(__name__,))
        ) as executor:
            tasks = {
                executor.submit(compute_func, (i, doc.page_content)): i
                for i, doc in enumerate(documents)
            }

            failures = 0
            for future in as_completed(tasks):
                try:
                    if result := future.result():
                        minhashes[result[0]] = (result[1], result[2])
                except Exception as e:  # noqa: BLE001 - isolate one document's failure
                    failures += 1
                    log_failure(
                        logger,
                        logging.WARNING,
                        "MinHash computation failed for document %s: %s",
                        tasks[future],
                        error=e,
                    )

        # A document with no MinHash is silently excluded from the LSH index and
        # can never be flagged as a duplicate, so a high failure rate means dedup
        # quietly under-removes. Surface the aggregate at error level (not just
        # per-doc warnings) so a half-failed pass does not read as successful.
        if failures and tasks:
            rate = failures / len(tasks)
            log = logger.error if rate >= 0.1 else logger.warning
            log(
                "MinHash computation failed for %s/%s documents (%.0f%%); "
                "those documents are excluded from deduplication",
                failures,
                len(tasks),
                rate * 100,
            )

        logger.debug("Computed MinHashes for %s documents", len(minhashes))
        return minhashes

    def _find_duplicates(
        self,
        signatures: dict[int, tuple[MinHash, str]],
        texts: list[str],
        rank: dict[int, int],
        threshold: float,
    ) -> dict[int, tuple[int, float]]:
        """The indices to drop, each with the index it copies and their
        similarity: of each near-duplicate pair, the later-ranked one.

        The LSH index proposes pairs, at the threshold 99% of them; the
        exact Jaccard similarity of their ``texts``' shingle sets decides
        (the MinHash estimate is off by a few hundredths either way). Only
        proposed documents' shingle sets are built.
        """
        lsh = MinHashLSH(
            threshold=threshold,
            num_perm=DEDUP_MINHASH_PERMUTATIONS,
            params=lsh_params(threshold, DEDUP_MINHASH_PERMUTATIONS),
        )
        for doc_idx, (minhash, _) in signatures.items():
            lsh.insert(doc_idx, minhash)

        shingles: dict[int, frozenset[str]] = {}

        def shingles_of(index: int) -> frozenset[str]:
            if index not in shingles:
                shingles[index] = document_shingles(texts[index], DEDUP_N_GRAMS)
            return shingles[index]

        duplicates: dict[int, tuple[int, float]] = {}
        for doc_idx in sorted(signatures, key=rank.__getitem__):
            if doc_idx in duplicates:
                continue
            minhash, numbers = signatures[doc_idx]
            for other in lsh.query(minhash):
                if (
                    not isinstance(other, int)
                    or other in duplicates
                    or rank[doc_idx] >= rank[other]
                    or signatures[other][1] != numbers
                ):
                    continue
                similarity = jaccard(shingles_of(doc_idx), shingles_of(other))
                if similarity >= threshold:
                    duplicates[other] = (doc_idx, similarity)
        return duplicates


def _label(document: Document) -> str:
    """How a WARNING names ``document``: its file name, else its path."""
    return document.file_name or document.file_path
