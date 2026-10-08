# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import shutil
import statistics
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from langchain_core.runnables import Runnable
from pydantic import ValidationError

from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.application.retrieval.rag_chain import (
    NO_CONTEXT_ANSWER,
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
    EvaluatorType,
    RetrieverRole,
)
from unified_kg_rag.shared import EvaluationException, get_logger
from unified_kg_rag.shared.utils import BATCH_ITEM_FAILED, BatchProcessor

from .answer_match_evaluator import AnswerMatchEvaluator
from .base import FAILED_METRICS_KEY, SKIPPED_METRICS_KEY, BaseEvaluator
from .graph_aware_evaluator import GraphAwareEvaluator
from .retrieval_evaluator import RetrievalEvaluator
from .source_resolver import SourceFileResolver, TextUnitFileResolver, file_name_of

logger = get_logger(__name__)

# The installed ``unified_kg_rag`` package directory (for the run manifest).
_PACKAGE_DIR = Path(__file__).resolve().parents[1]

# Per reported source, in rank order: (file names it names directly,
# text-unit ids whose files it derives from).
SourceProvenance = list[tuple[list[str], list[str]]]


class EvaluationManager:
    @staticmethod
    def _resolve_evaluator_class(
        evaluator_type: EvaluatorType,
    ) -> Callable[..., BaseEvaluator] | None:
        """Resolve an evaluator class lazily (registry, but import-on-use).

        The langchain/ragas adapter evaluators import `evaluation.base`, which
        runs this package's __init__; importing them at module load here would
        create a circular import (manager -> adapter -> evaluation.base ->
        __init__ -> manager). Resolving inside the method defers the import to
        instantiation time, when the package is fully initialized — while
        keeping the declarative type->class registry.
        """
        if evaluator_type is EvaluatorType.LANGCHAIN:
            from unified_kg_rag.adapters.evaluators.langchain_evaluator import (
                LangChainEvaluator,
            )

            return LangChainEvaluator
        if evaluator_type is EvaluatorType.RAGAS:
            from unified_kg_rag.adapters.evaluators.ragas_evaluator import (
                RagasEvaluator,
            )

            return RagasEvaluator
        if evaluator_type is EvaluatorType.GRAPH_AWARE:
            return GraphAwareEvaluator
        if evaluator_type is EvaluatorType.RETRIEVAL:
            return RetrievalEvaluator
        if evaluator_type is EvaluatorType.ANSWER_MATCH:
            return AnswerMatchEvaluator
        # Defensive: a future EvaluatorType with no mapping resolves to None and
        # is skipped by the caller. mypy sees the enum as exhaustive today, hence
        # the ignore — the branch is real once a new member is added.
        return None  # type: ignore[unreachable]

    def __init__(
        self,
        config: Config,
        rag_chain: Runnable | None = None,
        source_resolver: SourceFileResolver | None = None,
        *,
        providers: Providers | None = None,
    ) -> None:
        self.config = config
        if rag_chain is None:
            raise EvaluationException("RAG chain not provided for evaluation.")
        self.rag_chain = rag_chain
        # One provider bundle for every evaluator: the injected one, else the
        # chain's (so judges share its session and any injected factory), else
        # a default Bedrock bundle.
        chain_providers = getattr(rag_chain, "providers", None)
        if not isinstance(chain_providers, Providers):
            chain_providers = None
        self.providers = Providers.resolve(config, providers or chain_providers)
        self.source_resolver = source_resolver or self._default_source_resolver(
            config, rag_chain
        )
        self.evaluators: dict[EvaluatorType, BaseEvaluator] = {}
        self._initialize_evaluators()
        # One attempt per batch: the RAG chain already retries transient
        # Bedrock/store errors itself (and botocore under it), so retrying here
        # only re-runs deterministic failures (invalid filter, validation) with
        # minutes of backoff. A failed item still gets its sequential fallback.
        self.batch_processor = BatchProcessor(
            max_attempts=1, max_concurrency=config.processing.max_concurrency
        )

    @staticmethod
    def _default_source_resolver(
        config: Config, rag_chain: Runnable
    ) -> SourceFileResolver | None:
        """Resolve graph-source lineage through the chain's own document store.

        Reuses the retriever the chain binds to the DOCUMENT role (including an
        injected ``retriever_builders`` backend), so attribution reads the same
        indices the answers were retrieved from. Other runnables get no
        resolver: only sources that name a file directly are attributable.
        """
        if not isinstance(rag_chain, GraphRAGChain):
            return None
        return TextUnitFileResolver(
            config, lambda: rag_chain._get_retriever(RetrieverRole.DOCUMENT)
        )

    def _initialize_evaluators(self) -> None:
        """Build every enabled evaluator; fail fast unless ignore_errors.

        An evaluator that cannot be built (e.g. no Bedrock access for the
        judge) or rejects its configuration aborts the run with an
        ``EvaluationException`` — silently dropping it would publish a summary
        without metrics the user asked for. With ``processing.ignore_errors``
        it is dropped instead and recorded in ``dropped_evaluators`` (and the
        run manifest).
        """
        self.dropped_evaluators: dict[str, str] = {}
        for evaluator_type in self.config.evaluation.enabled_evaluators:
            evaluator_class = self._resolve_evaluator_class(evaluator_type)
            if not evaluator_class:
                logger.warning("Unknown evaluator type: '%s'", evaluator_type)
                self.dropped_evaluators[str(evaluator_type)] = "unknown evaluator type"
                continue

            try:
                evaluator = evaluator_class(
                    config=self.config, providers=self.providers
                )
                reason = (
                    None if evaluator.validate_config() else "invalid configuration"
                )
            except Exception as e:
                evaluator, reason = None, f"initialization failed: {e}"
            if evaluator is not None and reason is None:
                self.evaluators[evaluator_type] = evaluator
                continue
            message = f"'{evaluator_type.value}' evaluator {reason}"
            if not self.config.processing.ignore_errors:
                raise EvaluationException(
                    f"{message} (set processing.ignore_errors to drop it and "
                    "continue)"
                )
            logger.error("Dropping %s", message)
            self.dropped_evaluators[evaluator_type.value] = str(reason)

        if not self.evaluators:
            logger.warning("No evaluators were successfully initialized")
        else:
            logger.info("Initialized %s evaluators", len(self.evaluators))

    @staticmethod
    def load_data(
        eval_data_path: str | Path, base_metadata: dict[str, Any] | None = None
    ) -> tuple[list[EvaluationQuery], list[EvaluationGroundTruth]]:
        """Load and validate an evaluation dataset (a JSON array of objects).

        Fails fast with an ``EvaluationException`` naming the item index and
        query id on any malformed item — a missing/blank ``question``, a
        duplicate id, a field of the wrong type, or ``metadata`` the RAG chain
        would reject (e.g. an unknown ``search_strategy``) — instead of
        skipping it or failing per query mid-run. An empty dataset is an error.
        """
        if not eval_data_path:
            raise ValueError("Evaluation data path is required.")

        try:
            with open(eval_data_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error("Evaluation data file not found: '%s'", eval_data_path)
            raise
        except json.JSONDecodeError as e:
            logger.error("Error decoding JSON from '%s': %s", eval_data_path, e)
            raise

        if not isinstance(data, list):
            raise EvaluationException(
                f"Evaluation data '{eval_data_path}' must be a JSON array of "
                f"objects, got {type(data).__name__}."
            )
        if not data:
            raise EvaluationException(
                f"Evaluation data '{eval_data_path}' contains no queries."
            )

        cleaned_base_metadata = {
            k: v for k, v in (base_metadata or {}).items() if v is not None
        }
        queries: list[EvaluationQuery] = []
        ground_truths: list[EvaluationGroundTruth] = []
        seen_ids: set[str] = set()

        for i, item in enumerate(data):
            query, gt = EvaluationManager._parse_item(i, item, cleaned_base_metadata)
            if query.query_id in seen_ids:
                raise EvaluationException(
                    f"Invalid evaluation item at index {i}: duplicate query_id "
                    f"'{query.query_id}'."
                )
            seen_ids.add(query.query_id)
            queries.append(query)
            if gt is not None:
                ground_truths.append(gt)

        logger.info(
            "Loaded %s queries and %s ground truths from '%s'.",
            len(queries),
            len(ground_truths),
            eval_data_path,
        )
        return queries, ground_truths

    @staticmethod
    def _parse_item(
        index: int, item: Any, base_metadata: dict[str, Any]
    ) -> tuple[EvaluationQuery, EvaluationGroundTruth | None]:
        """Validate one dataset item; raise with its index and query id."""
        if not isinstance(item, dict):
            raise EvaluationException(
                f"Invalid evaluation item at index {index}: expected an object, "
                f"got {type(item).__name__}."
            )
        query_id = str(item.get("query_id", item.get("id", f"q_{index}")))
        where = f"Invalid evaluation item at index {index} (query_id '{query_id}')"

        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise EvaluationException(
                f"{where}: 'question' must be a non-empty string."
            )
        item_metadata = item.get("metadata", {})
        if not isinstance(item_metadata, dict):
            raise EvaluationException(f"{where}: 'metadata' must be an object.")
        if not isinstance(item_metadata.get("answerable", True), bool):
            raise EvaluationException(
                f"{where}: 'metadata.answerable' must be true or false."
            )

        final_metadata = {**base_metadata, **item_metadata}
        rag_fields = {
            k: v for k, v in final_metadata.items() if k in RAGInput.model_fields
        }
        try:
            # Validate what the RAG chain will receive now, not per query later.
            RAGInput.model_validate({**rag_fields, "query": question})
            query = EvaluationQuery(
                query_id=query_id,
                question=question,
                category=item.get("category"),
                difficulty=item.get("difficulty"),
                metadata=final_metadata,
            )
            # Build a ground truth when ANY ground-truth signal is present —
            # not only a textual answer — so graph-aware evaluation works on
            # datasets that supply only expected_entities/relationships.
            answer = item.get("answer")
            expected_entities = item.get("expected_entities") or []
            expected_relationships = item.get("expected_relationships") or []
            reference_sources = item.get("reference_sources") or []
            gt = None
            if (
                answer is not None
                or expected_entities
                or expected_relationships
                or reference_sources
            ):
                gt = EvaluationGroundTruth(
                    query_id=query_id,
                    ground_truth=str(answer) if answer is not None else "",
                    reference_sources=reference_sources,
                    expected_entities=expected_entities,
                    expected_relationships=expected_relationships,
                )
        except ValidationError as e:
            raise EvaluationException(f"{where}: {e}") from e
        return query, gt

    async def evaluate_dataset(
        self,
        queries: list[EvaluationQuery],
        ground_truths: list[EvaluationGroundTruth],
        show_progress: bool = True,
        *,
        dataset_path: str | Path | None = None,
        cli_args: dict[str, Any] | None = None,
    ) -> tuple[list[EvaluationResult], list[EvaluationReport], EvaluationSummary]:
        """Answer, score and summarize a dataset; the summary carries the manifest.

        ``dataset_path`` (the file the queries were loaded from) and
        ``cli_args`` are recorded in ``summary.run_manifest`` when given.
        """
        start_time = datetime.now()
        logger.info("Starting evaluation for %s queries", len(queries))

        try:
            results = await self._generate_answers(queries, show_progress)
            reports = await self._evaluate_results(queries, results, ground_truths)
            end_time = datetime.now()
            summary = self._generate_summary(
                queries, results, reports, start_time, end_time
            )
            summary.run_manifest = self.build_run_manifest(
                dataset_path,
                cli_args,
                queries=queries,
                ground_truths=ground_truths,
            )

            logger.info(
                "Evaluation completed: %s/%s queries processed",
                summary.successful_evaluations,
                summary.total_queries,
            )
            return results, reports, summary
        except Exception as e:
            logger.error("Dataset evaluation failed: %s", e)
            raise EvaluationException(f"Dataset evaluation failed: {e}") from e

    async def _generate_answers(
        self, queries: list[EvaluationQuery], show_progress: bool
    ) -> list[EvaluationResult]:
        def prepare_inputs(query_batch: list[EvaluationQuery]) -> list[dict[str, Any]]:
            return [{"query": q.question, **q.metadata} for q in query_batch]

        raw_results = await self.batch_processor.aexecute_with_fallback(
            items_to_process=queries,
            prepare_inputs_func=prepare_inputs,
            batch_func=self.rag_chain.abatch,
            sequential_func=self.rag_chain.ainvoke,
            task_name="Answer Generation",
            show_progress=show_progress,
        )

        results = []
        provenance: list[SourceProvenance] = []
        for query, raw_result in zip(queries, raw_results, strict=True):
            provenance.append(self._extract_source_provenance(raw_result))
            try:
                rag_metadata = self._extract_from_result(raw_result, "metadata", {})
                error_message = self._detect_generation_error(raw_result, rag_metadata)
                answer = self._extract_from_result(raw_result, "answer", "")
                results.append(
                    EvaluationResult(
                        query_id=query.query_id,
                        question=query.question,
                        generated_answer=answer,
                        ground_truth="",
                        retrieved_contexts=self._extract_from_result(
                            raw_result, "sources", []
                        ),
                        enable_thinking=rag_metadata.get("enable_thinking", False),
                        search_strategy=rag_metadata.get("search_strategy"),
                        response_time=rag_metadata.get("processing_time"),
                        search_type=query.metadata.get("search_type"),
                        top_k=query.metadata.get("top_k"),
                        retrieval_multiplier=query.metadata.get("retrieval_multiplier"),
                        metadata=query.metadata,
                        error=error_message is not None,
                        error_message=error_message,
                        abstained=error_message is None
                        and self._is_abstention(answer, rag_metadata),
                    )
                )
            except Exception as e:
                logger.error(
                    "Failed to process result for query '%s': %s", query.query_id, e
                )
                results.append(
                    EvaluationResult(
                        query_id=query.query_id,
                        question=query.question,
                        generated_answer="",
                        ground_truth="",
                        metadata={"error": str(e)},
                        error=True,
                        error_message=str(e),
                    )
                )
        await self._attribute_sources(queries, results, provenance)
        return results

    async def _attribute_sources(
        self,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
        provenance: list[SourceProvenance],
    ) -> None:
        """Set ``retrieved_source_ids``: per source, the file names it maps to.

        A source that names no file directly (entity, relationship, community
        report) is attributed to the files of the text units in its lineage,
        resolved in one batched lookup per index suffix. A source that still
        maps to no file stays ``[]`` (unattributable).
        """
        resolved: dict[str | None, dict[str, str]] = {}
        if self.source_resolver is not None:
            pending: dict[str | None, set[str]] = defaultdict(set)
            for query, sources in zip(queries, provenance, strict=True):
                for files, unit_ids in sources:
                    if not files:
                        pending[query.metadata.get("suffix")].update(unit_ids)
            for suffix, wanted in pending.items():
                if not wanted:
                    continue
                try:
                    resolved[suffix] = await self.source_resolver.aresolve(
                        sorted(wanted), suffix
                    )
                except Exception as e:  # noqa: BLE001 - attribution is best-effort
                    logger.warning(
                        "Could not resolve %s text-unit ids to files (suffix "
                        "'%s'); those sources stay unattributable: %s",
                        len(wanted),
                        suffix,
                        e,
                    )
        for query, result, sources in zip(queries, results, provenance, strict=True):
            mapping = resolved.get(query.metadata.get("suffix"), {})
            result.retrieved_source_ids = [
                files or sorted({mapping[u] for u in unit_ids if u in mapping})
                for files, unit_ids in sources
            ]

    @staticmethod
    def _is_abstention(answer: Any, rag_metadata: Any) -> bool:
        """True for the chain's fixed no-context reply (flag, or the exact text)."""
        if isinstance(rag_metadata, dict) and rag_metadata.get("abstained"):
            return True
        return isinstance(answer, str) and answer.strip() == NO_CONTEXT_ANSWER

    @staticmethod
    def _is_unanswerable(query_or_result: EvaluationQuery | EvaluationResult) -> bool:
        return query_or_result.metadata.get("answerable") is False

    @staticmethod
    def _detect_generation_error(raw_result: Any, rag_metadata: Any) -> str | None:
        """Return why answer generation failed, or None if it succeeded.

        Two failure shapes reach here without raising: the RAG chain's
        ignore_errors fallback (answer=DEFAULT_ERROR_MESSAGE, metadata
        ``{"error": True}``) and ``BATCH_ITEM_FAILED`` (or ``None``) for an
        item that failed even sequentially. Neither is a real answer, so
        neither may be scored.
        """
        if raw_result is None or raw_result is BATCH_ITEM_FAILED:
            return "answer generation returned no result"
        if not (isinstance(rag_metadata, dict) and rag_metadata.get("error")):
            return None
        detail = None
        if isinstance(raw_result, RAGOutput) and raw_result.search_results:
            detail = raw_result.search_results.metadata.get("error")
        return str(detail) if detail else "RAG chain returned an error response"

    @staticmethod
    def _extract_source_provenance(raw_result: Any) -> SourceProvenance:
        """Per reported source, in rank order: file names and text-unit lineage.

        File names come from the chunk attributes the indexer stores
        (``file_name`` / ``file_path``, top-level or under ``attributes``).
        Lineage is the source's ``text_unit_ids`` (entities, relationships,
        community reports) or, for a text unit without a file name, its own id.
        Document ids are not used: they are content hashes no dataset names.
        """
        if isinstance(raw_result, RAGOutput):
            sources: Any = raw_result.sources
        elif isinstance(raw_result, dict):
            sources = raw_result.get("sources")
        else:
            return []
        if not isinstance(sources, list):
            return []

        provenance: SourceProvenance = []
        for source in sources:
            files: list[str] = []
            unit_ids: list[str] = []
            if isinstance(source, dict):
                metadata = source.get("metadata")
                payloads = [source]
                if isinstance(metadata, dict):
                    payloads.append(metadata)
                for payload in payloads:
                    if name := file_name_of(payload):
                        files.append(name)
                    lineage = payload.get("text_unit_ids") or []
                    if isinstance(lineage, str):
                        lineage = [lineage]
                    if isinstance(lineage, list | tuple):
                        unit_ids.extend(str(u) for u in lineage if u)
                if (
                    isinstance(metadata, dict)
                    and metadata.get("section_type") == SectionType.TEXT.value
                ):
                    own_id = metadata.get("chunk_id") or metadata.get("source_id")
                    if own_id:
                        unit_ids.append(str(own_id))
            provenance.append(
                (list(dict.fromkeys(files)), list(dict.fromkeys(unit_ids)))
            )
        return provenance

    def create_lean_context_strings(
        self, sources_list: list[dict[str, Any]]
    ) -> list[str]:
        lean_contexts = []
        translated_key = (
            f"translated_text_{self.config.processing.translation.target_language}"
        )
        desired_fields = [
            "description",
            "full_content",
            # `content` is RetrievalResult's text field — the vector retriever populates
            # it and nothing else, so without it those results fell through to
            # _create_minimal_info and were stored as id+score with no text.
            "content",
            "name",
            "summary",
            translated_key,
        ]

        for item in sources_list:
            # ``content`` is the exact (possibly budget-truncated) text the
            # answer model saw; it already embeds the section's name/summary/
            # description, so re-adding those metadata fields would duplicate
            # them — and for a truncated section would credit text the model
            # never read. Fall back to field extraction only when it is empty.
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                lean_contexts.append(content)
                continue
            if self._is_truncated(item):
                lean_contexts.append(self._create_minimal_info(item))
                continue
            payloads_to_search = self._get_payloads_to_search(item)
            lean_item = self._extract_fields(payloads_to_search, desired_fields)

            if lean_item:
                lean_contexts.append(str(lean_item))
            else:
                lean_contexts.append(self._create_minimal_info(item))

        return lean_contexts

    @staticmethod
    def _is_truncated(item: dict[str, Any]) -> bool:
        metadata = item.get("metadata")
        return isinstance(metadata, dict) and bool(metadata.get("truncated"))

    @staticmethod
    def _get_payloads_to_search(item: dict[str, Any]) -> list[dict[str, Any]]:
        payloads = []

        if isinstance(item.get("metadata", {}).get("attributes"), dict):
            payloads.append(item["metadata"]["attributes"])

        if isinstance(item.get("metadata"), dict):
            payloads.append(item["metadata"])

        payloads.append(item)
        return payloads

    @staticmethod
    def _extract_fields(
        payloads: list[dict[str, Any]],
        desired_fields: list[str],
    ) -> dict[str, Any]:
        lean_item = {}

        for field in desired_fields:
            if field in lean_item:
                continue

            for payload in payloads:
                if field in payload and payload[field]:
                    lean_item[field] = payload[field]
                    break

        return lean_item

    @staticmethod
    def _create_minimal_info(item: dict[str, Any]) -> str:
        minimal_info = {"source": item.get("source"), "score": item.get("score")}
        return str({k: v for k, v in minimal_info.items() if v is not None})

    def _extract_from_result(
        self, raw_result: Any, key: str, default: Any = None
    ) -> Any:
        if raw_result is BATCH_ITEM_FAILED:
            return default
        if isinstance(raw_result, RAGOutput):
            if key == "sources" and hasattr(raw_result, key):
                sources_list = getattr(raw_result, key, [])
                return self.create_lean_context_strings(sources_list)
            return getattr(raw_result, key, default)

        if isinstance(raw_result, dict):
            value = raw_result.get(key, default)
            if key == "sources" and isinstance(value, list):
                return self.create_lean_context_strings(value)
            return value

        return default if key != "answer" else str(raw_result)

    async def _evaluate_results(
        self,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
        ground_truths: list[EvaluationGroundTruth],
    ) -> list[EvaluationReport]:
        all_reports: list[EvaluationReport] = []
        ground_truth_map = {gt.query_id: gt.ground_truth for gt in ground_truths}
        gt_obj_map = {gt.query_id: gt for gt in ground_truths}

        for result in results:
            result.ground_truth = ground_truth_map.get(result.query_id, "")
            # Thread graph-aware expectations onto the result so the
            # GraphAwareEvaluator can score entity/relationship coverage without
            # changing the evaluator signature.
            if gt := gt_obj_map.get(result.query_id):
                # Copy so a downstream in-place mutation of result.metadata does
                # not corrupt the shared ground-truth lists.
                result.metadata["expected_entities"] = list(gt.expected_entities)
                result.metadata["expected_relationships"] = [
                    dict(rel) if isinstance(rel, dict) else rel
                    for rel in gt.expected_relationships
                ]
                result.metadata["reference_sources"] = list(gt.reference_sources)

        # A failed answer generation (RAG error fallback text or an empty
        # sentinel) is not an answer: scoring it would let LLM judges grade the
        # apology text. Exclude it from every evaluator; the summary counts it
        # as failed and its metrics as skipped.
        # An item marked metadata.answerable=false has no answer to grade; it is
        # scored only on whether the chain abstained (abstention_statistics).
        scorable = [
            (query, res)
            for query, res in zip(queries, results, strict=True)
            if not res.error and not self._is_unanswerable(query)
        ]
        errored = sum(1 for res in results if res.error)
        if errored:
            logger.warning(
                "Excluding %s/%s queries from scoring: answer generation failed",
                errored,
                len(results),
            )
        scorable_queries = [query for query, _ in scorable]
        scorable_results = [res for _, res in scorable]
        gt_list = [res.ground_truth for res in scorable_results]

        # Graph-aware coverage emits nothing when the dataset carries no
        # expected_entities/relationships. Say so once (metric_outcomes also
        # counts the skips); info, not warning, since it is enabled by default.
        if EvaluatorType.GRAPH_AWARE in self.evaluators and not any(
            gt.expected_entities or gt.expected_relationships for gt in ground_truths
        ):
            logger.info(
                "graph_aware evaluator is enabled but no dataset row supplies "
                "expected_entities/expected_relationships — no coverage metric "
                "will be reported. Add them to the evaluation dataset to measure "
                "entity/relationship recall."
            )

        if not scorable_results:
            return all_reports

        for evaluator_type, evaluator in self.evaluators.items():
            try:
                reports = await evaluator.aevaluate_batch(
                    scorable_queries, scorable_results, gt_list
                )
                all_reports.extend(reports)
            except Exception as e:
                if not self.config.processing.ignore_errors:
                    raise
                logger.error(
                    "Failed to run '%s' evaluation: %s", evaluator_type.value, e
                )
                # Record the crash per query (as the per-query path in
                # BaseEvaluator does) so the summary counts the metrics as
                # failed instead of silently reporting nothing.
                all_reports.extend(
                    evaluator._create_empty_report(
                        query.query_id, reason=f"Evaluator crashed: {e}"
                    )
                    for query in scorable_queries
                )

        return all_reports

    def _generate_summary(
        self,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
        reports: list[EvaluationReport],
        start_time: datetime,
        end_time: datetime,
    ) -> EvaluationSummary:
        successful_evaluations = sum(
            1 for r in results if r.generated_answer and not r.error
        )
        response_times = [
            r.response_time for r in results if r.response_time and not r.error
        ]
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0.0
        )

        return EvaluationSummary(
            total_queries=len(queries),
            successful_evaluations=successful_evaluations,
            failed_evaluations=len(queries) - successful_evaluations,
            average_response_time=avg_response_time,
            metric_statistics=self._calculate_metric_statistics(reports),
            metric_outcomes=self._calculate_metric_outcomes(results, reports),
            abstention_statistics=self._calculate_abstention_statistics(results),
            grouped_statistics=self._calculate_grouped_statistics(
                queries, results, reports
            ),
            evaluation_start_time=start_time,
            evaluation_end_time=end_time,
            configuration=self.config.evaluation.model_dump(),
        )

    @classmethod
    def _calculate_grouped_statistics(
        cls,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
        reports: list[EvaluationReport],
    ) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
        """Metric statistics per actual strategy, category and difficulty.

        A pooled mean hides that ``auto`` answers queries with different
        strategies, and mixes easy and hard questions; grouping makes runs and
        strategies comparable.
        """
        attributes: dict[str, dict[str, str | None]] = {
            q.query_id: {"category": q.category, "difficulty": q.difficulty}
            for q in queries
        }
        for r in results:
            attributes.setdefault(r.query_id, {})["search_strategy"] = r.search_strategy

        grouped: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        for dimension in ("search_strategy", "category", "difficulty"):
            buckets: dict[str, list[EvaluationReport]] = defaultdict(list)
            for report in reports:
                value = attributes.get(report.query_id, {}).get(dimension)
                if value:
                    buckets[str(value)].append(report)
            stats = {
                value: cls._calculate_metric_statistics(bucket)
                for value, bucket in sorted(buckets.items())
            }
            if any(stats.values()):
                grouped[dimension] = stats
        return grouped

    _LIBRARIES = (
        "ragas",
        "langchain",
        "langchain-core",
        "langchain-aws",
        "langchain-community",
    )

    def build_run_manifest(
        self,
        eval_data_path: str | Path | None = None,
        cli_args: dict[str, Any] | None = None,
        *,
        queries: list[EvaluationQuery] | None = None,
        ground_truths: list[EvaluationGroundTruth] | None = None,
    ) -> dict[str, Any]:
        """Record what produced a run so two summaries can be compared."""
        enabled = set(self.evaluators) or set(self.config.evaluation.enabled_evaluators)
        uses_judge = bool(enabled & {EvaluatorType.LANGCHAIN, EvaluatorType.RAGAS})
        evaluation = self.config.evaluation
        dataset: dict[str, Any] = {}
        if eval_data_path is not None:
            path = Path(eval_data_path)
            dataset["path"] = str(path)
            dataset["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if queries is not None:
            # Hash of the parsed dataset, so library callers that never read a
            # file (and files differing only in formatting) are comparable.
            dataset["num_queries"] = len(queries)
            dataset["content_sha256"] = self._sha256_json(
                {
                    "queries": [q.model_dump(mode="json") for q in queries],
                    "ground_truths": [
                        gt.model_dump(mode="json") for gt in ground_truths or []
                    ],
                }
            )
        git_sha, git_dirty = self._git_state()
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "package_version": self._package_version("unified-kg-rag-on-aws"),
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "config_sha256": self._sha256_json(self.config.model_dump(mode="json")),
            "library_versions": {
                name: self._package_version(name) for name in self._LIBRARIES
            },
            "dataset": dataset,
            "cli_args": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in (cli_args or {}).items()
            },
            "models": {
                "answer_generation": self.config.search.answer_generation_model_id,
                "evaluation_judge": (
                    evaluation.evaluation_model_id if uses_judge else None
                ),
                "evaluation_embedding": (
                    evaluation.embedding_model_id.value
                    if EvaluatorType.RAGAS in enabled
                    else None
                ),
            },
            "enabled_evaluators": sorted(e.value for e in enabled),
            "dropped_evaluators": dict(self.dropped_evaluators),
        }

    @staticmethod
    def _sha256_json(data: Any) -> str:
        encoded = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _package_version(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    @staticmethod
    def _git_state() -> tuple[str | None, bool | None]:
        """``(HEAD sha, dirty)`` of the checkout this package runs from.

        Only a repository that tracks this package's own files counts: an
        install inside some other repository (e.g. a host project's ``.venv``)
        reports ``(None, None)`` rather than that repository's HEAD. ``dirty``
        is True when tracked files differ from HEAD.
        """
        git = shutil.which("git")
        if git is None:
            return None, None

        def _git(*args: str) -> str:
            return subprocess.run(  # noqa: S603 - fixed argv, no shell
                [git, *args],
                cwd=_PACKAGE_DIR,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()

        try:
            _git("ls-files", "--error-unmatch", "__init__.py")
            sha = _git("rev-parse", "HEAD")
            dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
        except (OSError, subprocess.SubprocessError):
            return None, None
        return sha or None, dirty

    def _calculate_metric_outcomes(
        self,
        results: list[EvaluationResult],
        reports: list[EvaluationReport],
    ) -> dict[str, dict[str, dict[str, int]]]:
        """Count scored / failed / skipped per evaluator and metric.

        Only ``scored`` values enter ``metric_statistics``; this makes the
        excluded ones visible so a mean over 3 of 50 queries is not mistaken
        for a mean over 50. Queries whose answer generation failed, and items
        marked ``metadata.answerable=false``, are counted as ``skipped`` for
        every metric of every enabled evaluator.
        """
        outcomes: dict[str, dict[str, dict[str, int]]] = {}

        def _bump(evaluator: str, metric: str, kind: str, n: int = 1) -> None:
            counts = outcomes.setdefault(evaluator, {}).setdefault(
                metric, {"scored": 0, "failed": 0, "skipped": 0}
            )
            counts[kind] += n

        unscored = sum(1 for r in results if r.error or self._is_unanswerable(r))
        for evaluator_type, evaluator in self.evaluators.items():
            for metric_type in evaluator.metric_types():
                _bump(evaluator_type.value, metric_type.value, "skipped", unscored)

        for report in reports:
            evaluator_name = report.evaluator_type.value
            for metric in report.metrics:
                _bump(evaluator_name, metric.metric_type.value, "scored")
            for kind, key in (
                ("failed", FAILED_METRICS_KEY),
                ("skipped", SKIPPED_METRICS_KEY),
            ):
                for metric_name in report.metadata.get(key) or {}:
                    _bump(evaluator_name, metric_name, kind)
        return outcomes

    @classmethod
    def _calculate_abstention_statistics(
        cls, results: list[EvaluationResult]
    ) -> dict[str, Any]:
        answered = [r for r in results if not r.error]
        if not answered:
            return {}

        def _rate(group: list[EvaluationResult]) -> dict[str, Any]:
            abstained = sum(1 for r in group if r.abstained)
            return {
                "abstained": abstained,
                "answered": len(group),
                "abstention_rate": abstained / len(group),
            }

        by_strategy: dict[str, list[EvaluationResult]] = defaultdict(list)
        for r in answered:
            if r.search_strategy:
                by_strategy[r.search_strategy].append(r)
        stats: dict[str, Any] = {
            **_rate(answered),
            "per_strategy": {k: _rate(v) for k, v in sorted(by_strategy.items())},
        }
        unanswerable = [r for r in answered if cls._is_unanswerable(r)]
        if unanswerable:
            correct = sum(1 for r in unanswerable if r.abstained)
            stats["unanswerable"] = {
                "total": len(unanswerable),
                "correct_abstentions": correct,
                "accuracy": correct / len(unanswerable),
            }
        return stats

    @staticmethod
    def _calculate_metric_statistics(
        reports: list[EvaluationReport],
    ) -> dict[str, dict[str, float]]:
        metric_values = defaultdict(list)
        for report in reports:
            for metric in report.metrics:
                metric_values[metric.metric_type].append(metric.value)

        statistics_dict: dict[str, dict[str, float]] = {}
        for metric_type, values in metric_values.items():
            if not values:
                continue
            statistics_dict[metric_type.value] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
        return statistics_dict

    def save_results(
        self,
        results: list[EvaluationResult],
        reports: list[EvaluationReport],
        summary: EvaluationSummary,
        outputs_dir: str | Path,
    ) -> None:
        if isinstance(outputs_dir, str):
            outputs_dir = Path(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        # A random suffix keeps runs started in the same second (parallel
        # strategies, CI matrices) from overwriting each other's files.
        timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        # Name files after the strategy when every answered query used the same
        # one, so runs of different strategies are distinguishable on disk.
        strategies = {r.search_strategy for r in results if r.search_strategy}
        stem = f"{strategies.pop()}_{timestamp}" if len(strategies) == 1 else timestamp

        if self.config.evaluation.save_detailed_results:
            self._save_json(
                outputs_dir / f"evaluation_results_{stem}.json",
                [r.model_dump() for r in results],
            )
            self._save_json(
                outputs_dir / f"evaluation_reports_{stem}.json",
                [r.model_dump() for r in reports],
            )

        self._save_json(
            outputs_dir / f"evaluation_summary_{stem}.json", summary.model_dump()
        )
        logger.info("Evaluation results saved to '%s'", outputs_dir)

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
