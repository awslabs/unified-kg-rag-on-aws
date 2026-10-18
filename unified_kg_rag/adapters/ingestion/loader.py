# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from unified_kg_rag.domain.models import Document
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import default_max_workers
from unified_kg_rag.shared.utils.cache_keys import is_excluded_source_path
from unified_kg_rag.shared.utils.concurrency import (
    ContextThreadPoolExecutor,
    cancelling_on_error,
)

logger = get_logger(__name__)


def compute_minhash(
    doc_info: tuple[int, str], num_permutations: int, n_grams: int
) -> tuple[int, MinHash] | None:
    doc_index, content = doc_info
    if not content or len(content) < n_grams:
        return None

    shingles = {content[j : j + n_grams] for j in range(len(content) - n_grams + 1)}
    if not shingles:
        return None

    minhash = MinHash(num_perm=num_permutations)
    minhash.update_batch([shingle.encode("utf8") for shingle in shingles])

    return doc_index, minhash


# Near-duplicate detection parameters of ``processing.deduplicate``: two
# documents are duplicates when the Jaccard similarity of their character
# 3-gram sets, estimated with a 128-permutation MinHash, is at least 0.9.
DEDUP_SIMILARITY_THRESHOLD = 0.9
DEDUP_MINHASH_PERMUTATIONS = 128
DEDUP_N_GRAMS = 3


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
            raise ValueError(
                f"Path is not a valid directory: '{self.source_directory}'"
            )

    def discover_files(self) -> list[Path]:
        files = [p for p in self.source_directory.rglob("*") if self._is_valid_file(p)]
        return sorted(files)

    def _is_valid_file(self, path: Path) -> bool:
        # Hidden paths and __pycache__ follow the rule discovery and the
        # corpus fingerprint use; only the part below the root is tested, so
        # a hidden ancestor of the source directory does not hide the corpus.
        return (
            path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            and not is_excluded_source_path(path, self.source_directory)
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
                except Exception as e:
                    logger.error("Error processing file '%s': %s", path, e)
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
        except Exception as e:
            logger.warning("Failed to load document '%s': %s", file_path, e)
            return None

    def load_single(self, file_path: Path) -> Document:
        return Document.from_json_file(file_path)

    def deduplicate_documents(
        self,
        documents: list[Document],
        keep: Callable[[Document], bool] | None = None,
        avoid: Callable[[Document], bool] | None = None,
    ) -> list[Document]:
        """MinHash near-duplicate removal for documents loaded elsewhere.

        Of two near-duplicates the earlier one is kept, except that a document
        ``keep`` accepts wins over one it rejects (an incremental run keeps
        the already-indexed copy) and one ``avoid`` accepts loses to any other
        (a copy whose indexing failed). The survivors keep their input order.
        """
        if len(documents) < 2:
            return documents

        logger.info("Starting deduplication for %s documents", len(documents))

        minhashes = self._compute_minhashes(documents)
        if not minhashes:
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
        duplicate_indices = self._find_duplicate_indices(minhashes, rank)
        if not duplicate_indices:
            logger.info("No duplicates found")
            return documents

        unique_docs = [
            doc for i, doc in enumerate(documents) if i not in duplicate_indices
        ]

        logger.info(
            "Deduplication completed: removed %s duplicates, kept %s unique documents",
            len(duplicate_indices),
            len(unique_docs),
        )

        return unique_docs

    def _compute_minhashes(self, documents: list[Document]) -> dict[int, MinHash]:
        minhashes = {}
        compute_func = partial(
            compute_minhash,
            num_permutations=DEDUP_MINHASH_PERMUTATIONS,
            n_grams=DEDUP_N_GRAMS,
        )

        with (
            ProcessPoolExecutor(max_workers=self.max_workers) as executor,
            cancelling_on_error(executor),
        ):
            tasks = {
                executor.submit(compute_func, (i, doc.page_content)): i
                for i, doc in enumerate(documents)
            }

            failures = 0
            for future in as_completed(tasks):
                try:
                    if result := future.result():
                        minhashes[result[0]] = result[1]
                except Exception as e:
                    failures += 1
                    logger.warning(
                        "MinHash computation failed for document %s: %s",
                        tasks[future],
                        e,
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

    def _find_duplicate_indices(
        self, minhashes: dict[int, MinHash], rank: dict[int, int]
    ) -> set[int]:
        """Indices to drop: of each near-duplicate pair, the later-ranked one.

        The Jaccard check runs here: it is a comparison of two short
        signatures, cheaper than handing them to a worker process.
        """
        lsh = MinHashLSH(
            threshold=DEDUP_SIMILARITY_THRESHOLD, num_perm=DEDUP_MINHASH_PERMUTATIONS
        )
        for doc_idx, minhash in minhashes.items():
            lsh.insert(doc_idx, minhash)

        duplicate_indices = set()
        for doc_idx, minhash in minhashes.items():
            for other in lsh.query(minhash):
                if (
                    isinstance(other, int)
                    and rank[doc_idx] < rank[other]
                    and minhash.jaccard(minhashes[other]) >= DEDUP_SIMILARITY_THRESHOLD
                ):
                    duplicate_indices.add(other)
        return duplicate_indices
