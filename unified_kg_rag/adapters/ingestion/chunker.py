# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import statistics
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from langchain.text_splitter import (
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic import BaseModel, Field
from tqdm import tqdm

from unified_kg_rag.adapters.aws import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.bedrock import get_assumed_role_boto_session
from unified_kg_rag.adapters.aws.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from unified_kg_rag.adapters.aws.token_counter import (
    BedrockTokenCounter,
    estimate_token_count,
)
from unified_kg_rag.domain.models import ChunkingStrategy, Config, Document, TextUnit
from unified_kg_rag.domain.prompts import TextChunkingPrompt
from unified_kg_rag.shared import DataProcessingError, get_logger
from unified_kg_rag.shared.utils import (
    BatchProcessor,
    generate_stable_id,
)

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
    def __init__(self, min_chunk_size: int = 50, max_chunk_size: int = 5000) -> None:
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

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
                1 for chunk in chunks if len(chunk) > self.max_chunk_size
            ),
            "undersized_chunks": sum(
                1 for chunk in chunks if len(chunk) < self.min_chunk_size
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
        try:
            if not boundaries:
                return []

            if self._is_single_chunk_request(boundaries):
                return []

            line_numbers = self._extract_line_numbers_from_response(boundaries)

            valid_line_numbers = [
                num for num in line_numbers if isinstance(num, int) and num > 1
            ]
            return sorted(set(valid_line_numbers))

        except Exception as e:
            logger.debug("Failed to extract line numbers: %s", e)
            return []

    @staticmethod
    def _is_single_chunk_request(boundaries: dict[str, Any] | list[Any]) -> bool:
        if isinstance(boundaries, dict):
            return boundaries.get("single_chunk") in [True, "true"]
        return False

    def _extract_line_numbers_from_response(
        self, boundaries: dict[str, Any] | list[Any]
    ) -> list[int]:
        line_numbers = []

        try:
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

        except Exception as e:
            logger.debug("Failed to extract line numbers from response: %s", e)

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
        try:
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

        except Exception as e:
            logger.debug("Failed to convert line numbers to indices: %s", e)
            return [], len(line_numbers)

    def validate_error_rate(self, missed_count: int, total_line_numbers: int) -> bool:
        if total_line_numbers == 0:
            return True
        return (missed_count / total_line_numbers) <= self.max_line_miss_rate


class ChunkProcessor:
    def __init__(
        self,
        min_chunk_size: int,
        max_chunk_size: int,
        fallback_splitter: RecursiveCharacterTextSplitter | None = None,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.fallback_splitter = fallback_splitter

    def merge_small_chunks(self, chunks: list[str]) -> list[str]:
        if not chunks:
            return []

        try:
            merged_chunks = []
            i = 0
            while i < len(chunks):
                current_chunk = chunks[i]
                j = i + 1
                while j < len(chunks) and len(current_chunk) < self.min_chunk_size:
                    if len(current_chunk) + len(chunks[j]) <= self.max_chunk_size:
                        current_chunk += chunks[j]
                        j += 1
                    else:
                        break
                merged_chunks.append(current_chunk)
                i = j
            return self._merge_final_small_chunk_if_needed(merged_chunks)
        except Exception as e:
            logger.debug("Failed to merge small chunks: %s", e)
            return chunks

    def _merge_final_small_chunk_if_needed(self, chunks: list[str]) -> list[str]:
        if len(chunks) > 1 and len(chunks[-1]) < self.min_chunk_size:
            last_chunk = chunks.pop()
            if len(chunks[-1]) + len(last_chunk) <= self.max_chunk_size:
                chunks[-1] += last_chunk
            else:
                chunks.append(last_chunk)
        return chunks


class BaseChunker(ABC):
    def __init__(
        self,
        config: Config,
        show_progress: bool = True,
        boto_session: boto3.Session | None = None,
    ) -> None:
        self.config = config
        self.chunking_config = self.config.processing.chunking
        self.show_progress = show_progress
        self.stats = ChunkingStats()

        session = boto_session or boto3.Session(profile_name=config.aws.profile_name)
        session = get_assumed_role_boto_session(
            session, assumed_role_arn=config.aws.bedrock.assumed_role_arn
        )
        bedrock_client = session.client(
            "bedrock-runtime",
            region_name=config.aws.bedrock.region_name,
            config=BotoConfig(retries={"max_attempts": 3}),
        )
        self._token_counter = BedrockTokenCounter(
            model_id=config.indexing.opensearch.embedding_model_id.value,
            client=bedrock_client,
        )

        self.fallback_splitter = self._create_splitter(
            self.chunking_config.fallback_chunk_size, self.chunking_config.chunk_overlap
        )
        self.chunk_processor = ChunkProcessor(
            self.chunking_config.min_chunk_size,
            self.chunking_config.max_chunk_size,
            fallback_splitter=self.fallback_splitter,
        )
        self.quality_validator = ChunkQualityValidator(
            min_chunk_size=self.chunking_config.min_chunk_size,
            max_chunk_size=self.chunking_config.max_chunk_size,
        )

    @abstractmethod
    def _chunk_single_document(self, doc: Document) -> list[TextUnit]:
        pass

    @staticmethod
    def _create_splitter(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
        try:
            return RecursiveCharacterTextSplitter(
                chunk_size=size, chunk_overlap=overlap, length_function=len
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
                disable=not self.show_progress,
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
            if doc.is_error or not doc.content:
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
                covariate_ids={},
                community_ids=[],
                n_tokens=n_tokens,
                attributes=attributes,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        except Exception as e:
            raise DataProcessingError(f"Failed to create text unit: {e}") from e

    def _calculate_token_count(self, text: str) -> int:
        try:
            return self._token_counter.count_tokens(text)
        except Exception:
            return estimate_token_count(text)

    def _extract_document_content(self, doc: Document) -> str | None:
        try:
            if not doc.content:
                return None
            content_type = self.config.processing.chunking.content_type

            primary_content = getattr(doc.content, content_type, None)
            if isinstance(primary_content, str) and primary_content:
                return primary_content

            fallback_content = getattr(doc.content, "text", None)
            if isinstance(fallback_content, str) and fallback_content:
                return fallback_content

            return None

        except Exception as e:
            logger.warning(
                "Failed to extract document content from '%s': %s", doc.file_name, e
            )
            return None


class SimpleTextChunker(BaseChunker):
    def __init__(
        self,
        config: Config,
        show_progress: bool = True,
        boto_session: boto3.Session | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, show_progress=show_progress, boto_session=boto_session)
        logger.debug("Initialized SimpleTextChunker")

    def _chunk_single_document(self, doc: Document) -> list[TextUnit]:
        try:
            content = self._extract_document_content(doc)
            if not content:
                return []

            chunks = self.fallback_splitter.split_text(content)
            chunks = self.chunk_processor.merge_small_chunks(chunks)

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
        boto_session: boto3.Session | None = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, show_progress=show_progress, boto_session=boto_session)
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )
        self.ignore_errors = self.config.processing.ignore_errors

        self.factory = BedrockLanguageModelFactory(
            config=self.config,
            boto_session=self.boto_session,
            region_name=self.config.aws.bedrock.region_name,
        )
        self.batch_processor = BatchProcessor()

        robust_xml_output_parser = create_robust_xml_output_parser(
            factory=self.factory,
            enable_output_fixing=self.config.fixing.enabled,
            output_fixing_model_id=self.config.fixing.fixing_model_id,
        )
        self.chunker = setup_chain(
            factory=self.factory,
            model_id=self.chunking_config.chunking_model_id,
            prompt_class=TextChunkingPrompt,
            parser=robust_xml_output_parser,
        )

        self.pre_splitter = self._create_splitter(
            self.chunking_config.pre_chunk_size, self.chunking_config.pre_chunk_overlap
        )
        self.line_boundary_processor = LineBasedBoundaryProcessor(
            max_line_miss_rate=self.chunking_config.max_marker_miss_rate,
        )
        logger.debug(
            "Initialized IntelligentTextChunker with model %s",
            self.chunking_config.chunking_model_id,
        )

    def _chunk_single_document(self, doc: Document) -> list[TextUnit]:
        try:
            content = self._extract_document_content(doc)
            if not content:
                return []

            pre_chunks = self._create_pre_chunks(content)
            if not pre_chunks:
                logger.warning("No pre-chunks created for document '%s'", doc.file_name)
                return []

            final_chunks_info = self._process_pre_chunks(pre_chunks, doc.file_name)

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

    def _process_pre_chunks(
        self, pre_chunks: list[str], doc_name: str
    ) -> list[tuple[str, str, int]]:
        try:
            logger.debug(
                "Processing %s pre-chunks for document '%s'", len(pre_chunks), doc_name
            )
            self.stats.num_pre_chunks_processed = (
                self.stats.num_pre_chunks_processed or 0
            ) + len(pre_chunks)
            boundary_results = self._get_boundary_results(pre_chunks, doc_name)

            final_chunks = []
            for i, (pre_chunk, llm_response) in enumerate(
                zip(pre_chunks, boundary_results, strict=True)
            ):
                chunks = self._get_chunks_from_response(pre_chunk, llm_response)
                if chunks:
                    final_chunks.extend([("llm", chunk, i + 1) for chunk in chunks])
                else:
                    self._handle_llm_failure(pre_chunk, i, final_chunks)
            return final_chunks
        except Exception as e:
            logger.error("Failed to process pre-chunks for '%s': %s", doc_name, e)
            return [("fallback", chunk, i + 1) for i, chunk in enumerate(pre_chunks)]

    def _get_boundary_results(
        self, pre_chunks: list[str], doc_name: str
    ) -> list[dict[str, Any] | None]:
        try:
            return self.batch_processor.execute_with_fallback(
                items_to_process=pre_chunks,
                prepare_inputs_func=self._create_chain_inputs,
                batch_func=self.chunker.batch,
                sequential_func=self.chunker.invoke,
                task_name=f"Processing boundaries for '{doc_name}'",
                run_config=self.config.processing.model_dump(),
                show_progress=self.show_progress,
            )
        except Exception as e:
            if not self.ignore_errors:
                raise

            logger.error("Failed to get boundary results for '%s': %s", doc_name, e)
            return [None] * len(pre_chunks)

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

        try:
            final_chunks = []
            max_size = self.chunking_config.max_chunk_size

            for chunk in chunks:
                if len(chunk) > max_size:
                    split_chunks = self.fallback_splitter.split_text(chunk)
                    final_chunks.extend(split_chunks)
                else:
                    final_chunks.append(chunk)

            return final_chunks

        except Exception as e:
            logger.debug("Failed to split large line chunks: %s", e)
            return chunks

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
            logger.debug("Failed LLM line-based boundary chunking: %s", e)
            return None

    def _extract_chunks_from_line_indices(
        self, text: str, indices: list[int]
    ) -> list[str]:
        try:
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

            merged_chunks = self.chunk_processor.merge_small_chunks(chunks)

            return self._split_large_line_chunks(merged_chunks)

        except Exception as e:
            logger.debug("Failed to extract chunks from line indices: %s", e)
            return [text] if text.strip() else []

    def _create_pre_chunks(self, content: str) -> list[str]:
        try:
            content_type = self.config.processing.chunking.content_type
            if content_type in ["markdown", "html"]:
                logger.debug("Creating structured pre-chunks for %s", content_type)
                return self._create_structured_pre_chunks(content, content_type)
            return self.pre_splitter.split_text(content)
        except Exception as e:
            logger.warning("Failed to create pre-chunks: %s", e)
            return [content] if content else []

    def _create_structured_pre_chunks(
        self, content: str, content_type: str
    ) -> list[str]:
        try:
            splitter = self._get_structured_splitter(content_type)
            return self._merge_structured_chunks(splitter.split_text(content))
        except Exception as e:
            logger.warning(
                "Failed to create structured pre-chunks for %s: %s", content_type, e
            )
            return self.pre_splitter.split_text(content)

    def _merge_structured_chunks(self, structured_chunks: list[Any]) -> list[str]:
        try:
            final_pre_chunks = []
            i = 0
            while i < len(structured_chunks):
                current_content, i = self._merge_small_structured_chunks(
                    structured_chunks, i
                )
                if (
                    len(current_content)
                    > self.config.processing.chunking.pre_chunk_size
                ):
                    final_pre_chunks.extend(
                        self.pre_splitter.split_text(current_content)
                    )
                else:
                    final_pre_chunks.append(current_content)
                i += 1
            return final_pre_chunks
        except Exception as e:
            logger.warning("Failed to merge structured chunks: %s", e)
            return [chunk.page_content for chunk in structured_chunks]

    def _merge_small_structured_chunks(
        self, chunks: list[Any], start_idx: int
    ) -> tuple[str, int]:
        i = start_idx
        content = chunks[i].page_content
        config = self.config.processing.chunking

        while len(content) < config.min_chunk_size and (i + 1) < len(chunks):
            next_content = chunks[i + 1].page_content
            if len(content) + len(next_content) <= config.pre_chunk_size:
                content += "\n\n" + next_content
                i += 1
            else:
                break
        return content, i

    def _create_fallback_chunks(self, pre_chunk: str) -> list[str]:
        try:
            chunks = (
                self.fallback_splitter.split_text(pre_chunk)
                if self.fallback_splitter
                else [pre_chunk]
            )
            return self.chunk_processor.merge_small_chunks(chunks)
        except Exception as e:
            logger.debug("Failed to create fallback chunks: %s", e)
            return [pre_chunk]

    def _create_chain_inputs(self, chunks: list[str]) -> list[dict[str, Any]]:
        try:
            config = self.config.processing.chunking
            inputs = []

            for chunk in chunks:
                lines = chunk.split("\n")
                numbered_lines = []
                for i, line in enumerate(lines, 1):
                    numbered_lines.append(f"{i}: {line}")

                numbered_text = "\n".join(numbered_lines)

                inputs.append(
                    {
                        "numbered_text": numbered_text,
                        "min_chunk_size": config.min_chunk_size,
                        "max_chunk_size": config.max_chunk_size,
                    }
                )

            return inputs
        except Exception as e:
            logger.debug("Failed to create chain inputs: %s", e)
            return []

    @staticmethod
    def _get_structured_splitter(
        content_type: str,
    ) -> MarkdownHeaderTextSplitter | HTMLHeaderTextSplitter:
        try:
            if content_type == "markdown":
                return MarkdownHeaderTextSplitter(
                    headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
                    strip_headers=False,
                )
            return HTMLHeaderTextSplitter(
                headers_to_split_on=[("h1", "H1"), ("h2", "H2"), ("h3", "H3")]
            )
        except Exception as e:
            raise DataProcessingError(
                f"Failed to create structured splitter for {content_type}: {e}"
            ) from e

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
        boto_session: boto3.Session | None = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> BaseChunker:
        logger.info("Creating chunker of type: '%s'", chunker_type.value)
        if chunker_type == ChunkingStrategy.SIMPLE:
            return SimpleTextChunker(
                config, show_progress, boto_session=boto_session, **kwargs
            )
        if chunker_type == ChunkingStrategy.INTELLIGENT:
            return IntelligentTextChunker(config, boto_session, show_progress, **kwargs)
        raise DataProcessingError(f"Unknown chunker type: '{chunker_type}'")
