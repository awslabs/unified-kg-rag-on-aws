# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import statistics
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from tqdm import tqdm

from unified_kg_rag.adapters.ingestion.length_units import (
    GrowingText,
    TokenBudgetTextSplitter,
    chunk_length_function,
)
from unified_kg_rag.adapters.llm.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from unified_kg_rag.adapters.llm.langchain import BATCH_ITEM_FAILED, BatchProcessor
from unified_kg_rag.adapters.llm.retry import transient_error_predicate
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.ingestion.chunk_length import max_chars
from unified_kg_rag.domain.models import (
    ChunkingStrategy,
    ChunkLengthUnit,
    Config,
    Document,
    ModelPurpose,
    ModelRole,
    TextUnit,
)
from unified_kg_rag.domain.prompts import TextChunkingPrompt
from unified_kg_rag.shared import DataProcessingError, get_logger
from unified_kg_rag.shared.utils import generate_stable_id

logger = get_logger(__name__)


class ChunkingStats(BaseModel):
    num_total_documents: int = Field(
        default=0, description="Total number of documents scheduled for processing"
    )
    num_successful_documents: int = Field(
        default=0, description="Number of documents successfully chunked"
    )
    num_failed_documents: int = Field(
        default=0,
        description="Number of documents that encountered errors during chunking",
    )
    num_pre_chunks_processed: int | None = Field(
        default=None,
        description="Number of pre-chunks sent to LLM for intelligent chunking",
    )
    total_chunks_created: int = Field(
        default=0,
        description="Total number of text chunks generated across all documents",
    )
    total_processing_time: float = Field(
        default=0.0, description="Total time spent on chunking operations (in seconds)"
    )
    llm_processing_failures: int | None = Field(
        default=None, description="Number of pre-chunks that failed LLM processing"
    )
    fallback_chunks_used: int | None = Field(
        default=None,
        description="Number of chunks created using fallback splitting method",
    )
    num_chunk_chars: list[int] = Field(
        default_factory=list,
        description="Character count distribution for all generated chunks",
    )

    @property
    def processed_document_count(self) -> int:
        return self.num_successful_documents + self.num_failed_documents

    @property
    def average_processing_time(self) -> float:
        return self.total_processing_time / max(1, self.processed_document_count)

    @property
    def success_rate(self) -> float:
        if self.num_total_documents == 0:
            return 0.0
        return (self.num_successful_documents / self.num_total_documents) * 100

    @property
    def llm_failure_rate(self) -> float:
        if not self.num_pre_chunks_processed or self.num_pre_chunks_processed == 0:
            return 0.0
        if not self.llm_processing_failures:
            return 0.0
        return (self.llm_processing_failures / self.num_pre_chunks_processed) * 100

    @property
    def chunk_stats(self) -> dict[str, int | float]:
        if not self.num_chunk_chars:
            return {"total": 0, "min": 0, "avg": 0.0, "median": 0.0, "max": 0}

        return {
            "total": sum(self.num_chunk_chars),
            "min": min(self.num_chunk_chars),
            "avg": statistics.mean(self.num_chunk_chars),
            "median": statistics.median(self.num_chunk_chars),
            "max": max(self.num_chunk_chars),
        }

    def add_num_chunk_chars(self, num_chunk_chars: int) -> None:
        if hasattr(self, "__dict__") and "num_chunk_chars" in self.__dict__:
            self.__dict__["num_chunk_chars"].append(num_chunk_chars)
        else:
            self.num_chunk_chars.append(num_chunk_chars)


class ChunkQualityValidator:
    def __init__(
        self,
        min_chunk_size: int = 50,
        max_chunk_size: int = 5000,
        length_function: Callable[[str], int] = len,
    ) -> None:
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        # The measure the size limits are in; the *_num_chars metrics count
        # characters whatever it is.
        self.length_function = length_function

    def validate_chunks(self, chunks: list[str]) -> dict[str, Any]:
        if not chunks:
            return {"is_valid": False, "issues": ["No chunks generated"], "metrics": {}}

        issues = []
        metrics = {
            "total_chunks": len(chunks),
            "avg_num_chars": sum(len(chunk) for chunk in chunks) / len(chunks),
            "min_num_chars": min(len(chunk) for chunk in chunks),
            "max_num_chars": max(len(chunk) for chunk in chunks),
            "empty_chunks": sum(1 for chunk in chunks if not chunk.strip()),
            "oversized_chunks": sum(
                1
                for chunk in chunks
                if self.length_function(chunk) > self.max_chunk_size
            ),
            # A document's only chunk cannot be merged with anything, so a
            # short document is not an undersized-chunk problem.
            "undersized_chunks": (
                sum(
                    1
                    for chunk in chunks
                    if self.length_function(chunk) < self.min_chunk_size
                )
                if len(chunks) > 1
                else 0
            ),
        }

        if metrics["empty_chunks"] > 0:
            issues.append(f"{metrics['empty_chunks']} empty chunks found")
        if metrics["oversized_chunks"] > 0:
            issues.append(f"{metrics['oversized_chunks']} chunks exceed maximum size")
        if metrics["undersized_chunks"] > len(chunks) * 0.5:
            issues.append(f"Too many undersized chunks: {metrics['undersized_chunks']}")

        return {"is_valid": len(issues) == 0, "issues": issues, "metrics": metrics}


class LineBasedBoundaryProcessor:
    def __init__(
        self,
        max_line_miss_rate: float,
    ):
        self.max_line_miss_rate = max_line_miss_rate

    def extract_line_numbers(self, boundaries: dict[str, Any] | list[Any]) -> list[int]:
        if not boundaries or self._is_single_chunk_request(boundaries):
            return []
        line_numbers = self._extract_line_numbers_from_response(boundaries)
        return sorted({num for num in line_numbers if isinstance(num, int) and num > 1})

    @staticmethod
    def _is_single_chunk_request(boundaries: dict[str, Any] | list[Any]) -> bool:
        if isinstance(boundaries, dict):
            return boundaries.get("single_chunk") in [True, "true"]
        return False

    def _extract_line_numbers_from_response(
        self, boundaries: dict[str, Any] | list[Any]
    ) -> list[int]:
        line_numbers = []
        if isinstance(boundaries, dict):
            chunk_boundaries = boundaries.get("chunk_boundaries")
            if chunk_boundaries:
                line_numbers.extend(
                    self._extract_from_chunk_boundaries(chunk_boundaries)
                )
            if "line_number" in boundaries:
                line_numbers.extend(
                    self._extract_from_line_number_field(boundaries["line_number"])
                )
        elif isinstance(boundaries, list):
            for item in boundaries:
                if isinstance(item, int):
                    line_numbers.append(item)
                elif isinstance(item, dict) and "line_number" in item:
                    line_numbers.extend(
                        self._extract_from_line_number_field(item["line_number"])
                    )
        return line_numbers

    def _extract_from_chunk_boundaries(self, chunk_boundaries: Any) -> list[int]:
        line_numbers = []

        if isinstance(chunk_boundaries, list):
            for item in chunk_boundaries:
                if isinstance(item, int):
                    line_numbers.append(item)
                elif isinstance(item, dict):
                    if "line_number" in item:
                        line_numbers.extend(
                            self._extract_from_line_number_field(item["line_number"])
                        )
                    elif "#text" in item:
                        try:
                            line_numbers.append(int(item["#text"]))
                        except (ValueError, TypeError):
                            pass
        elif isinstance(chunk_boundaries, dict):
            if "line_number" in chunk_boundaries:
                line_numbers.extend(
                    self._extract_from_line_number_field(
                        chunk_boundaries["line_number"]
                    )
                )

        return line_numbers

    @staticmethod
    def _extract_from_line_number_field(line_number_field: Any) -> list[int]:
        line_numbers = []

        if isinstance(line_number_field, int):
            line_numbers.append(line_number_field)
        elif isinstance(line_number_field, list):
            for item in line_number_field:
                if isinstance(item, int):
                    line_numbers.append(item)
                elif isinstance(item, dict) and "#text" in item:
                    try:
                        line_numbers.append(int(item["#text"]))
                    except (ValueError, TypeError):
                        pass
                elif isinstance(item, str):
                    try:
                        line_numbers.append(int(item))
                    except (ValueError, TypeError):
                        pass
        elif isinstance(line_number_field, dict) and "#text" in line_number_field:
            try:
                line_numbers.append(int(line_number_field["#text"]))
            except (ValueError, TypeError):
                pass
        elif isinstance(line_number_field, str):
            try:
                line_numbers.append(int(line_number_field))
            except (ValueError, TypeError):
                pass

        return line_numbers

    @staticmethod
    def convert_line_numbers_to_indices(
        lines: list[str], line_numbers: list[int]
    ) -> tuple[list[int], int]:
        if not line_numbers:
            return [], 0

        chunk_indices, missed_count = [], 0
        cumulative_length = 0

        for i, line in enumerate(lines):
            line_num = i + 1

            if line_num in line_numbers:
                chunk_indices.append(cumulative_length)

            cumulative_length += len(line)
            if i < len(lines) - 1:
                cumulative_length += 1

        max_line_num = len(lines)
        missed_count = sum(1 for num in line_numbers if num > max_line_num)

        return sorted(set(chunk_indices)), missed_count

    def validate_error_rate(self, missed_count: int, total_line_numbers: int) -> bool:
        if total_line_numbers == 0:
            return True
        return (missed_count / total_line_numbers) <= self.max_line_miss_rate


# LangChain's default separators (paragraph, line, word, character) plus a
# split after CJK sentence terminators, so Chinese and Japanese text, which has
# no spaces, is not cut mid-sentence at the character limit. The lookbehind is
# zero-width, so the terminator stays at the end of its sentence; text without
# these characters splits exactly as with the defaults.
_SPLITTER_SEPARATORS = ["\n\n", "\n", "(?<=[。．｡！？；])", " ", ""]


class ChunkProcessor:
    """Folds undersized chunks into their neighbours.

    Adjacent chunks are joined by their span in the source text when it is
    given (exact: the splitter's overlap appears once and the original
    whitespace between chunks is kept). Without a source, or when a chunk
    cannot be located in it, the overlap of the next chunk with the previous
    one (at most ``chunk_overlap`` characters) is dropped and the pieces are
    joined with a separator, so words at the seam are never fused.

    ``min_chunk_size``/``max_chunk_size`` are in ``length_function``'s unit;
    ``chunk_overlap`` bounds the shared text in characters.
    """

    _JOIN_SEPARATOR = "\n"

    def __init__(
        self,
        min_chunk_size: int,
        max_chunk_size: int,
        fallback_splitter: RecursiveCharacterTextSplitter | None = None,
        chunk_overlap: int = 0,
        length_function: Callable[[str], int] = len,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.fallback_splitter = fallback_splitter
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def merge_small_chunks(
        self,
        chunks: list[str],
        source: str | None = None,
        *,
        overlap: int | None = None,
    ) -> list[str]:
        """Merge undersized chunks into their neighbours.

        ``overlap`` is the most text consecutive chunks may share (defaults to
        ``chunk_overlap``, the splitter's overlap). Pass ``0`` for disjoint
        pieces, e.g. slices cut at line boundaries, so text that legitimately
        repeats across a seam is never mistaken for overlap and dropped.
        """
        if not chunks:
            return []

        max_overlap = self.chunk_overlap if overlap is None else overlap
        spans = self._locate_spans(chunks, source, max_overlap) if source else None

        def joined(first: int, last: int) -> str:
            if spans is not None and source is not None:
                return source[spans[first][0] : spans[last][1]]
            text = chunks[first]
            for k in range(first + 1, last + 1):
                text = self._join_pair(text, chunks[k], max_overlap)
            return text

        size = self.length_function
        groups: list[tuple[int, int]] = []
        i = 0
        while i < len(chunks):
            last = i
            # Each candidate extends the current text, so it is built and
            # measured from what it adds.
            current = GrowingText(chunks[i], size)
            while last + 1 < len(chunks) and current.size < self.min_chunk_size:
                if spans is None:
                    candidate = self._join_pair(
                        current.text, chunks[last + 1], max_overlap
                    )
                else:
                    candidate = joined(i, last + 1)
                candidate_size = current.size_of(candidate)
                if candidate_size > self.max_chunk_size:
                    break
                current.extend(candidate, candidate_size)
                last += 1
            groups.append((i, last))
            i = last + 1

        if len(groups) > 1:
            tail_first, tail_last = groups[-1]
            if size(joined(tail_first, tail_last)) < self.min_chunk_size:
                prev_first = groups[-2][0]
                if size(joined(prev_first, tail_last)) <= self.max_chunk_size:
                    groups[-2:] = [(prev_first, tail_last)]

        return [joined(first, last) for first, last in groups]

    def _locate_spans(
        self, chunks: list[str], source: str, max_overlap: int
    ) -> list[tuple[int, int]] | None:
        """Find each chunk's (start, end) in ``source``, in order.

        A chunk starts after the previous one and overlaps it by at most
        ``max_overlap`` characters, which bounds the search so repeated text
        earlier in the source cannot be matched. Each chunk must also end
        strictly after the previous one: a chunk that repeats the previous
        chunk's tail verbatim would otherwise match that tail and be dropped
        from the merged span, so the search is retried past the previous chunk.
        Returns None if any chunk is not found, so the caller falls back to
        joining the chunk strings.
        """
        spans: list[tuple[int, int]] = []
        for chunk in chunks:
            if not spans:
                start = source.find(chunk)
                if start < 0:
                    return None
                spans.append((start, start + len(chunk)))
                continue
            prev_start, prev_end = spans[-1]
            search_from = max(prev_start + 1, prev_end - max_overlap)
            start = source.find(chunk, search_from)
            if 0 <= start and start + len(chunk) <= prev_end:
                start = source.find(chunk, prev_end)
            if start < 0:
                return None
            spans.append((start, start + len(chunk)))
        return spans

    def _join_pair(self, left: str, right: str, max_overlap: int) -> str:
        overlap = self._seam_overlap(left, right, max_overlap)
        if overlap:
            # The remainder starts at whitespace (or is empty): no fused words.
            return left + right[overlap:]
        return left + self._JOIN_SEPARATOR + right

    def _seam_overlap(self, left: str, right: str, max_overlap: int) -> int:
        """Length of the longest suffix of ``left`` that prefixes ``right``.

        Bounded by ``max_overlap`` and accepted only at whitespace (or chunk)
        boundaries on both sides, so a coincidental match of a few characters
        inside a word is not mistaken for the splitter's overlap.
        """
        limit = min(max_overlap, len(left), len(right))
        for size in range(limit, 0, -1):
            if not left.endswith(right[:size]):
                continue
            left_ok = size == len(left) or left[-size - 1].isspace()
            right_ok = size == len(right) or right[size].isspace()
            if left_ok and right_ok:
                return size
        return 0


class BaseChunker(ABC):
    def __init__(self, config: Config, show_progress: bool = True) -> None:
        self.config = config
        self.chunking_config = self.config.processing.chunking
        self.show_progress = show_progress
        self.stats = ChunkingStats()
        # Every size setting is in this unit; positions and overlaps inside
        # the text stay in characters.
        self.length_unit = self.chunking_config.length_unit
        self.length_function = chunk_length_function(self.length_unit)

        self.fallback_splitter = self._create_splitter(
            self.chunking_config.fallback_chunk_size,
            self.chunking_config.chunk_overlap,
            length_unit=self.length_unit,
        )
        self.chunk_processor = ChunkProcessor(
            self.chunking_config.min_chunk_size,
            self.chunking_config.max_chunk_size,
            fallback_splitter=self.fallback_splitter,
            chunk_overlap=max_chars(
                self.chunking_config.chunk_overlap, self.length_unit
            ),
            length_function=self.length_function,
        )
        self.quality_validator = ChunkQualityValidator(
            min_chunk_size=self.chunking_config.min_chunk_size,
            max_chunk_size=self.chunking_config.max_chunk_size,
            length_function=self.length_function,
        )

    @abstractmethod
    def _chunk_single_document(self, doc: Document) -> list[TextUnit]:
        pass

    @staticmethod
    def _create_splitter(
        size: int,
        overlap: int,
        *,
        length_unit: ChunkLengthUnit = ChunkLengthUnit.CHARS,
    ) -> RecursiveCharacterTextSplitter:
        try:
            if length_unit == ChunkLengthUnit.TOKENS:
                return TokenBudgetTextSplitter(
                    chunk_size=size,
                    chunk_overlap=overlap,
                    separators=_SPLITTER_SEPARATORS,
                    is_separator_regex=True,
                )
            return RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                length_function=len,
                separators=_SPLITTER_SEPARATORS,
                is_separator_regex=True,
            )
        except Exception as e:
            raise DataProcessingError(f"Failed to create text splitter: {e}") from e

    def chunk_documents(self, documents: list[Document]) -> list[TextUnit]:
        try:
            logger.info("Starting chunking process for %s documents", len(documents))
            self.stats = ChunkingStats(num_total_documents=len(documents))
            start_time = time.time()
            all_text_units: list[TextUnit] = []

            with tqdm(
                total=len(documents),
                desc=f"Chunking Documents ({self.__class__.__name__})",
                unit="doc",
                disable=None if self.show_progress else True,
            ) as pbar:
                for doc in documents:
                    self._process_document_with_stats(doc, all_text_units)
                    self._update_progress_bar(pbar)

            self._log_completion_summary(time.time() - start_time)
            return all_text_units
        except Exception as e:
            logger.error("Failed to chunk documents: %s", e)
            raise DataProcessingError(f"Failed to chunk documents: {e}") from e

    def _process_document_with_stats(
        self,
        doc: Document,
        all_text_units: list[TextUnit],
    ) -> None:
        doc_start_time = time.time()
        try:
            if doc.is_error:
                self.stats.num_failed_documents += 1
                return

            text_units = self._chunk_single_document(doc)
            if text_units:
                all_text_units.extend(text_units)
                self.stats.num_successful_documents += 1
                self.stats.total_chunks_created += len(text_units)
                for unit in text_units:
                    self.stats.add_num_chunk_chars(len(unit.text))
            else:
                logger.warning("No chunks generated for document '%s'", doc.file_name)
                self.stats.num_failed_documents += 1
        except DataProcessingError as e:
            logger.error(
                "Failed to chunk document '%s': %s", doc.file_name, e, exc_info=False
            )
            self.stats.num_failed_documents += 1
        finally:
            self.stats.total_processing_time += time.time() - doc_start_time

    def _update_progress_bar(self, pbar: tqdm) -> None:
        pbar.set_postfix(
            {
                "Success": f"{self.stats.success_rate:.2f}%",
                "Avg Time": f"{self.stats.average_processing_time:.2f}s",
            }
        )
        pbar.update(1)

    def _log_completion_summary(self, total_time: float) -> None:
        if not self.stats:
            return
        chunk_stats = self.stats.chunk_stats
        logger.info(
            "Chunking completed - Total time: %.2fs, Success rate: %.2f%% (%s/%s)",
            total_time,
            self.stats.success_rate,
            self.stats.num_successful_documents,
            self.stats.num_total_documents,
        )
        logger.info(
            "Total chunks: %s, Chars (Min/Avg/Max): %s/%.0f/%s",
            self.stats.total_chunks_created,
            chunk_stats["min"],
            chunk_stats["avg"],
            chunk_stats["max"],
        )
        if self.stats.num_failed_documents > 0:
            logger.warning(
                "Failed to process %s documents", self.stats.num_failed_documents
            )

    def _create_text_unit(
        self,
        chunk_text: str,
        document: Document,
        chunk_id: int,
        total_chunks: int,
        method: str,
        pre_chunk_id: int,
    ) -> TextUnit:
        try:
            n_tokens = self._calculate_token_count(chunk_text)

            attributes = {
                "file_name": document.file_name,
                "file_path": document.file_path,
                "chunk_id": chunk_id,
                "total_chunks": total_chunks,
                "chunking_method": method,
                "pre_chunk_id": pre_chunk_id,
            }

            if "index" in document.metadata:
                attributes["index"] = document.metadata["index"]

            if "filters" in document.metadata:
                attributes["filters"] = document.metadata["filters"]

            text_unit_id_content = f"text_unit:{document.document_id}:{chunk_id}"
            text_unit_id = generate_stable_id(text_unit_id_content)

            return TextUnit(
                id=text_unit_id,
                short_id=text_unit_id[:8],
                text=chunk_text,
                document_ids=[document.document_id],
                entity_ids=[],
                relationship_ids=[],
                community_ids=[],
                n_tokens=n_tokens,
                attributes=attributes,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        except Exception as e:
            raise DataProcessingError(f"Failed to create text unit: {e}") from e

    @staticmethod
    def _calculate_token_count(text: str) -> int:
        # Chunk sizes are budgeted for the embedding model, and Bedrock's
        # CountTokens API does not accept embedding models.
        return estimate_token_count(text)


class SimpleTextChunker(BaseChunker):
    def __init__(
        self,
        config: Config,
        show_progress: bool = True,
        *,
        providers: Providers | None = None,
    ) -> None:
        # No model calls: ``providers`` is accepted so the factory can pass
        # one bundle to either chunker.
        super().__init__(config, show_progress=show_progress)
        logger.debug("Initialized SimpleTextChunker")

    def _chunk_single_document(self, doc: Document) -> list[TextUnit]:
        try:
            content = doc.page_content
            if not content:
                return []

            chunks = self.fallback_splitter.split_text(content)
            chunks = self.chunk_processor.merge_small_chunks(chunks, source=content)

            validation = self.quality_validator.validate_chunks(chunks)
            if not validation["is_valid"]:
                logger.warning(
                    "Chunk quality issues for document '%s': %s",
                    doc.file_name,
                    validation["issues"],
                )

            return [
                self._create_text_unit(chunk, doc, i + 1, len(chunks), "simple", i + 1)
                for i, chunk in enumerate(chunks)
            ]
        except Exception as e:
            raise DataProcessingError(
                f"Failed to chunk single document '{doc.file_name}': {e}"
            ) from e


class IntelligentTextChunker(BaseChunker):
    def __init__(
        self,
        config: Config,
        show_progress: bool = True,
        *,
        providers: Providers | None = None,
    ) -> None:
        super().__init__(config, show_progress=show_progress)
        self.providers = Providers.resolve(config, providers)

        self.factory = self.providers.llm_factory
        self.batch_processor = BatchProcessor.from_config(
            self.config,
            is_transient_error=transient_error_predicate(self.factory),
        )

        robust_xml_output_parser = create_robust_xml_output_parser(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            enable_output_fixing=self.config.fixing.enabled,
            output_fixing_model_id=self.config.model_id(ModelRole.FIXING),
            output_tags=["chunk_boundaries"],
        )
        self.chunker = setup_chain(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            model_id=self.config.model_id(ModelRole.CHUNKING),
            prompt_class=TextChunkingPrompt,
            parser=robust_xml_output_parser,
            custom_prompts=self.config.custom_prompts,
        )

        self.pre_splitter = self._create_splitter(
            self.chunking_config.pre_chunk_size,
            self.chunking_config.pre_chunk_overlap,
            length_unit=self.length_unit,
        )
        self.line_boundary_processor = LineBasedBoundaryProcessor(
            max_line_miss_rate=self.chunking_config.max_marker_miss_rate,
        )
        # Pre-chunks and boundary answers per document of the running
        # chunk_documents call (keyed by id(document)).
        self._planned: dict[int, tuple[list[str], dict[int, Any]]] = {}
        logger.debug(
            "Initialized IntelligentTextChunker with model %s",
            self.config.model_id(ModelRole.CHUNKING),
        )

    def chunk_documents(self, documents: list[Document]) -> list[TextUnit]:
        """Chunk ``documents`` with one boundary-model batch for all of them.

        The pre-chunks of every document that need the model are requested
        together, so a corpus of short documents does not run one model call
        at a time; the documents are then assembled one by one.
        """
        planned: dict[int, list[str]] = {}
        for doc in documents:
            if not doc.is_error:
                planned[id(doc)] = self._pre_chunks_of(doc)
        responses = self._request_boundaries(list(planned.values()), "all documents")
        self._planned = {
            key: (pre_chunks, response)
            for (key, pre_chunks), response in zip(
                planned.items(), responses, strict=True
            )
        }
        try:
            return super().chunk_documents(documents)
        finally:
            self._planned = {}

    def _pre_chunks_of(self, doc: Document) -> list[str]:
        content = doc.page_content
        return self._create_pre_chunks(content) if content else []

    def _chunk_single_document(self, doc: Document) -> list[TextUnit]:
        try:
            plan = self._planned.get(id(doc))
            pre_chunks, responses = (
                plan if plan is not None else (self._pre_chunks_of(doc), None)
            )
            if not pre_chunks:
                return []

            final_chunks_info = self._process_pre_chunks(
                pre_chunks, doc.file_name, responses
            )

            chunks = [text for _, text, _ in final_chunks_info]

            validation = self.quality_validator.validate_chunks(chunks)
            if not validation["is_valid"]:
                logger.warning(
                    "Chunk quality issues for document '%s': %s",
                    doc.file_name,
                    validation["issues"],
                )

            return [
                self._create_text_unit(
                    text, doc, i + 1, len(final_chunks_info), method, src_id
                )
                for i, (method, text, src_id) in enumerate(final_chunks_info)
            ]
        except Exception as e:
            raise DataProcessingError(
                f"Failed to chunk single document '{doc.file_name}': {e}"
            ) from e

    def _needs_llm(self, pre_chunk: str) -> bool:
        """Whether a pre-chunk is worth a boundary call.

        Any split of a pre-chunk shorter than 2 * min_chunk_size leaves at
        least one piece under min_chunk_size, and merge_small_chunks folds it
        back, so the result is the pre-chunk itself (unless it is oversized
        and must be split anyway).
        """
        processor = self.chunk_processor
        size = self.length_function(pre_chunk)
        return size >= 2 * processor.min_chunk_size or size > processor.max_chunk_size

    def _request_boundaries(
        self, pre_chunk_lists: list[list[str]], label: str
    ) -> list[dict[int, Any]]:
        """Boundary answers for each list's pre-chunks that need the model.

        One batch call covers every list. A list's answers map a pre-chunk's
        index to the parsed response, or None when the call failed (the
        pre-chunk is then split by size).
        """
        requests = [
            (n, i)
            for n, pre_chunks in enumerate(pre_chunk_lists)
            for i, pre_chunk in enumerate(pre_chunks)
            if self._needs_llm(pre_chunk)
        ]
        answers: list[dict[int, Any]] = [{} for _ in pre_chunk_lists]
        if not requests:
            return answers
        try:
            results = self.batch_processor.execute_with_fallback(
                items_to_process=[pre_chunk_lists[n][i] for n, i in requests],
                prepare_inputs_func=self._create_chain_inputs,
                sequential_func=self.chunker.invoke,
                task_name=f"Processing boundaries for {label}",
                show_progress=self.show_progress,
            )
        except Exception as e:
            logger.error("Failed to get boundary results for %s: %s", label, e)
            results = [None] * len(requests)
        for (n, i), result in zip(requests, results, strict=True):
            answers[n][i] = None if result is BATCH_ITEM_FAILED else result
        return answers

    def _process_pre_chunks(
        self,
        pre_chunks: list[str],
        doc_name: str,
        responses: dict[int, Any] | None = None,
    ) -> list[tuple[str, str, int]]:
        """Chunk one document's pre-chunks, keeping each under max_chunk_size.

        ``responses`` are the boundary answers by pre-chunk index (see
        :meth:`_request_boundaries`); requested here when not given.
        """
        logger.debug(
            "Processing %s pre-chunks for document '%s'", len(pre_chunks), doc_name
        )
        self.stats.num_pre_chunks_processed = (
            self.stats.num_pre_chunks_processed or 0
        ) + len(pre_chunks)
        if responses is None:
            responses = self._request_boundaries([pre_chunks], f"'{doc_name}'")[0]

        final_chunks: list[tuple[str, str, int]] = []
        for i, pre_chunk in enumerate(pre_chunks):
            if i not in responses:
                final_chunks.append(("unsplit", pre_chunk, i + 1))
                continue
            chunks = self._get_chunks_from_response(pre_chunk, responses[i])
            if chunks:
                # Every answer shape (no boundaries, single_chunk, only line
                # 1, out-of-range lines) can leave a piece over the limit.
                final_chunks.extend(
                    ("llm", chunk, i + 1)
                    for chunk in self._split_large_line_chunks(chunks)
                )
            else:
                self._handle_llm_failure(pre_chunk, i, final_chunks)
        return final_chunks

    def _get_chunks_from_response(
        self, pre_chunk: str, llm_response: dict[str, Any] | None
    ) -> list[str] | None:
        return (
            self._chunk_with_llm_boundaries(pre_chunk, llm_response)
            if llm_response
            else None
        )

    def _handle_llm_failure(
        self,
        pre_chunk: str,
        chunk_index: int,
        final_chunks: list[tuple[str, str, int]],
    ) -> None:
        logger.debug(
            "LLM processing failed for pre-chunk %s, using fallback", chunk_index + 1
        )
        self.stats.llm_processing_failures = (
            self.stats.llm_processing_failures or 0
        ) + 1
        fallback_chunks = self._create_fallback_chunks(pre_chunk)
        self.stats.fallback_chunks_used = (self.stats.fallback_chunks_used or 0) + len(
            fallback_chunks
        )
        final_chunks.extend(
            [("fallback", chunk, chunk_index + 1) for chunk in fallback_chunks]
        )

    def _split_large_line_chunks(self, chunks: list[str]) -> list[str]:
        if not self.fallback_splitter:
            return chunks

        final_chunks = []
        max_size = self.chunking_config.max_chunk_size

        for chunk in chunks:
            if self.length_function(chunk) > max_size:
                split_chunks = self.fallback_splitter.split_text(chunk)
                final_chunks.extend(split_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _chunk_with_llm_boundaries(
        self, text: str, response: dict[str, Any]
    ) -> list[str] | None:
        try:
            chunk_boundaries = response.get("chunk_boundaries")
            if chunk_boundaries is None:
                return [text]

            lines = text.split("\n")

            line_numbers = self.line_boundary_processor.extract_line_numbers(
                chunk_boundaries
            )
            if not line_numbers:
                return [text]

            indices, missed = (
                self.line_boundary_processor.convert_line_numbers_to_indices(
                    lines, line_numbers
                )
            )

            if not self.line_boundary_processor.validate_error_rate(
                missed, len(line_numbers)
            ):
                logger.debug(
                    "Line boundary error rate too high: %s/%s",
                    missed,
                    len(line_numbers),
                )
                return None

            if not indices:
                return [text]

            return self._extract_chunks_from_line_indices(text, indices)

        except Exception as e:
            # The pre-chunk is split by size instead (_handle_llm_failure).
            logger.warning("Failed LLM line-based boundary chunking: %s", e)
            return None

    def _extract_chunks_from_line_indices(
        self, text: str, indices: list[int]
    ) -> list[str]:
        if not indices:
            return [text] if text.strip() else []

        chunks = []
        start_pos = 0

        for end_pos in sorted(set(indices)):
            if end_pos > start_pos:
                chunk = text[start_pos:end_pos].strip()
                if chunk:
                    chunks.append(chunk)
            start_pos = end_pos

        if start_pos < len(text):
            final_chunk = text[start_pos:].strip()
            if final_chunk:
                chunks.append(final_chunk)

        # Line-boundary slices are disjoint: allow no overlap between them.
        return self.chunk_processor.merge_small_chunks(chunks, source=text, overlap=0)

    def _create_pre_chunks(self, content: str) -> list[str]:
        try:
            return self.pre_splitter.split_text(content)
        except Exception as e:
            logger.warning("Failed to create pre-chunks: %s", e)
            return [content] if content else []

    def _create_fallback_chunks(self, pre_chunk: str) -> list[str]:
        chunks = (
            self.fallback_splitter.split_text(pre_chunk)
            if self.fallback_splitter
            else [pre_chunk]
        )
        return self.chunk_processor.merge_small_chunks(chunks, source=pre_chunk)

    def _create_chain_inputs(self, chunks: list[str]) -> list[dict[str, Any]]:
        inputs = []

        for chunk in chunks:
            lines = chunk.split("\n")
            numbered_lines = []
            for i, line in enumerate(lines, 1):
                numbered_lines.append(f"{i}: {line}")

            numbered_text = "\n".join(numbered_lines)
            min_size, max_size = self._prompt_sizes(chunk)

            inputs.append(
                {
                    "numbered_text": numbered_text,
                    "min_chunk_size": min_size,
                    "max_chunk_size": max_size,
                }
            )

        return inputs

    def _prompt_sizes(self, chunk: str) -> tuple[int, int]:
        """The chunk size range for the boundary prompt, in characters.

        The prompt asks for characters (the model sees lines and text, not
        token counts), so a token range is converted at this pre-chunk's own
        characters-per-token ratio. The pieces are measured in tokens again
        afterwards (merge and oversize split).
        """
        config = self.chunking_config
        if self.length_unit != ChunkLengthUnit.TOKENS:
            return config.min_chunk_size, config.max_chunk_size
        ratio = len(chunk) / max(estimate_token_count(chunk), 1)
        return (
            max(1, round(config.min_chunk_size * ratio)),
            max(1, round(config.max_chunk_size * ratio)),
        )

    def _log_completion_summary(self, total_time: float) -> None:
        super()._log_completion_summary(total_time)
        if self.stats and (self.stats.llm_processing_failures or 0) > 0:
            logger.info(
                "LLM fallback rate: %.2f%%, Fallback chunks: %s",
                self.stats.llm_failure_rate,
                self.stats.fallback_chunks_used,
            )


class ChunkerFactory:
    @staticmethod
    def create_chunker(
        config: Config,
        chunker_type: ChunkingStrategy = ChunkingStrategy.INTELLIGENT,
        show_progress: bool = True,
        *,
        providers: Providers | None = None,
    ) -> BaseChunker:
        logger.info("Creating chunker of type: '%s'", chunker_type.value)
        if chunker_type == ChunkingStrategy.SIMPLE:
            return SimpleTextChunker(config, show_progress, providers=providers)
        if chunker_type == ChunkingStrategy.INTELLIGENT:
            return IntelligentTextChunker(config, show_progress, providers=providers)
        raise DataProcessingError(f"Unknown chunker type: '{chunker_type}'")
