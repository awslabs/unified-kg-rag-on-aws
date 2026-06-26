# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import fnmatch
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document as BaseDocument

from unified_kg_rag.adapters.ingestion.parser import ParserFactory
from unified_kg_rag.domain.models import Config, Document
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import compute_hash, default_max_workers

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
    for shingle in shingles:
        minhash.update(shingle.encode("utf8"))

    return doc_index, minhash


def compute_jaccard_similarity(
    pair: tuple[int, int], minhashes: dict[int, MinHash]
) -> tuple[int, int, float] | None:
    idx1, idx2 = pair
    minhash1 = minhashes.get(idx1)
    minhash2 = minhashes.get(idx2)

    if minhash1 and minhash2:
        return idx1, idx2, minhash1.jaccard(minhash2)
    return None


class DirectoryLoader(BaseLoader):
    DEFAULT_SUPPORTED_EXTENSIONS = {".csv", ".json", ".md", ".tsv", ".txt"}
    DEFAULT_EXCLUDE_PATTERNS = {"**/.*", "**/*.pyc", "**/__pycache__/**"}

    def __init__(
        self,
        source_directory: str | Path,
        config: Config | None = None,
        recursive: bool = True,
        supported_extensions: set[str] | None = None,
        exclude_patterns: set[str] | None = None,
        deduplicate: bool = False,
        similarity_threshold: float = 0.9,
        minhash_permutations: int = 128,
        n_grams_size: int = 3,
        max_workers: int | None = None,
        compute_dir_hash: bool = False,
        parse_files: bool = False,
    ):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if minhash_permutations <= 0:
            raise ValueError("minhash_permutations must be positive")
        if n_grams_size <= 0:
            raise ValueError("n_grams_size must be positive")

        self.source_directory = Path(source_directory).resolve()
        self.config = config
        self.recursive = recursive
        self.supported_extensions = (
            supported_extensions or self.DEFAULT_SUPPORTED_EXTENSIONS
        )
        self.exclude_patterns = exclude_patterns or self.DEFAULT_EXCLUDE_PATTERNS
        self.deduplicate = deduplicate
        self.similarity_threshold = similarity_threshold
        self.minhash_permutations = minhash_permutations
        self.n_grams_size = n_grams_size
        self.max_workers = max_workers or default_max_workers()
        self.compute_dir_hash = compute_dir_hash
        self.parse_files = parse_files

        if self.parse_files and self.config:
            parser_extensions = set(ParserFactory.get_supported_extensions())
            self.supported_extensions = self.supported_extensions.union(
                parser_extensions
            )

        self.failed_files: list[str] = []
        self.directory_hash: str | None = None

    def load(self) -> list[BaseDocument]:
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

        if self.deduplicate and docs:
            docs = self._deduplicate_documents(docs)

        if self.compute_dir_hash:
            self.directory_hash = self._compute_dir_hash(discovered_files)

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

        # Document is a BaseDocument subclass; widen the invariant list type to
        # match the BaseLoader.load() supertype signature.
        return list(docs)

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
        pattern = "**/*" if self.recursive else "*"
        files = [
            p for p in self.source_directory.glob(pattern) if self._is_valid_file(p)
        ]
        return sorted(files)

    def _is_valid_file(self, path: Path) -> bool:
        return (
            path.is_file()
            and path.suffix.lower() in self.supported_extensions
            and not any(fnmatch.fnmatch(str(path), p) for p in self.exclude_patterns)
        )

    def _load_files_concurrently(
        self, file_paths: list[Path]
    ) -> tuple[list[Document], list[Path]]:
        documents, failed_paths = [], []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self._load_and_enrich_single, path): path
                for path in file_paths
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
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
        if self.parse_files and self.config and file_path.suffix.lower() != ".json":
            parser = ParserFactory.create_parser(file_path, self.config)
            return parser.parse_file(
                file_path, self.config.processing.document_parsing.index_value
            )
        return Document.from_json_file(file_path)

    def _deduplicate_documents(self, documents: list[Document]) -> list[Document]:
        if len(documents) < 2:
            return documents

        logger.info("Starting deduplication for %s documents", len(documents))

        minhashes = self._compute_minhashes(documents)
        if not minhashes:
            logger.warning("No MinHashes computed, skipping deduplication")
            return documents

        duplicate_indices = self._find_duplicate_indices(minhashes)
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
            num_permutations=self.minhash_permutations,
            n_grams=self.n_grams_size,
        )

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = {
                executor.submit(compute_func, (i, doc.content.text or "")): i
                for i, doc in enumerate(documents)
                if doc.content
            }

            for future in as_completed(tasks):
                try:
                    if result := future.result():
                        minhashes[result[0]] = result[1]
                except Exception as e:
                    logger.warning(
                        "MinHash computation failed for document %s: %s",
                        tasks[future],
                        e,
                    )

        logger.debug("Computed MinHashes for %s documents", len(minhashes))
        return minhashes

    def _find_duplicate_indices(self, minhashes: dict[int, MinHash]) -> set[int]:
        lsh = MinHashLSH(
            threshold=self.similarity_threshold, num_perm=self.minhash_permutations
        )

        for doc_idx, minhash in minhashes.items():
            lsh.insert(doc_idx, minhash)

        candidate_pairs = set()
        for doc_idx in minhashes:
            duplicates = lsh.query(minhashes[doc_idx])
            for dup_idx in duplicates:
                if isinstance(dup_idx, int) and doc_idx < dup_idx:
                    candidate_pairs.add(tuple(sorted((doc_idx, dup_idx))))

        if not candidate_pairs:
            return set()

        logger.debug(
            "Found %s candidate pairs for similarity check", len(candidate_pairs)
        )

        duplicate_indices = set()
        similarity_func = partial(compute_jaccard_similarity, minhashes=minhashes)

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pair = {
                executor.submit(similarity_func, pair): pair for pair in candidate_pairs
            }

            for future in as_completed(future_to_pair):
                try:
                    if result := future.result():
                        _, j, sim = result
                        if sim >= self.similarity_threshold:
                            duplicate_indices.add(j)
                except Exception as e:
                    logger.warning(
                        "Similarity computation failed for pair %s: %s",
                        future_to_pair[future],
                        e,
                    )

        return duplicate_indices

    def _compute_dir_hash(self, files: list[Path]) -> str:
        if not files:
            return compute_hash("", algorithm="sha256", length=16)

        file_info_parts = []
        for file_path in files:
            try:
                stat = file_path.stat()
                info = (
                    f"{file_path.relative_to(self.source_directory)}:"
                    f"{stat.st_mtime_ns}:{stat.st_size}"
                )
                file_info_parts.append(info)
            except Exception as e:
                logger.warning("Could not stat file for hash '%s': %s", file_path, e)

        content = "\n".join(file_info_parts)
        hash_value = compute_hash(content, algorithm="sha256", length=16)
        logger.debug("Computed directory hash: '%s'", hash_value)
        return hash_value
