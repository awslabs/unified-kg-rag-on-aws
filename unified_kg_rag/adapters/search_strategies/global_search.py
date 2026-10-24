# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MS GraphRAG global search over a cut of the community hierarchy.

The reports of one cut (``community_depth``, see
``domain.ingestion.community_hierarchy``), or of a root-down dynamic
selection, are all mapped into scored key points in token-sized batches;
the ranked points, packed into a token budget, are the answer context.
"""

import asyncio
import math
import time
from typing import Any

from langchain_core.exceptions import ContextOverflowError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig

from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
    as_float,
    is_fatal_retrieval_error,
)
from unified_kg_rag.adapters.retrieval.hybrid_scorer import RerankTrace
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.adapters.store_errors import is_bad_request
from unified_kg_rag.domain.ingestion.community_hierarchy import cut_filter
from unified_kg_rag.domain.models import (
    Config,
    ModelRole,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
    language_name,
)
from unified_kg_rag.domain.prompts import (
    CommunityRelevancePrompt,
    GlobalMapPrompt,
    MapReduceSummaryPrompt,
)
from unified_kg_rag.domain.retrieval.strategy_registry import register_strategy
from unified_kg_rag.shared import (
    ConfigurationError,
    LanguageModelError,
    LanguageModelTimeoutError,
    get_logger,
    observability,
)
from unified_kg_rag.shared.utils import parse_llm_json, safe_float_parse, text_digest
from unified_kg_rag.shared.utils.concurrency import gather_bounded
from unified_kg_rag.shared.utils.langchain import BatchProcessor

logger = get_logger(__name__)


class _MapPoint:
    """A scored key point produced by the global-search map step.

    Lightweight value object (not a Pydantic model) so the map/filter/rank/pack
    plumbing stays internal to this adapter. ``batch`` is the index of the map
    batch that produced it (-1 when unknown); ``report_ids`` are the ids of
    the reports the map model says the point comes from (MS GraphRAG's
    ``[Data: Reports (ids)]``), as it wrote them.
    """

    __slots__ = ("batch", "description", "report_ids", "score")

    def __init__(
        self,
        description: str,
        score: int,
        batch: int = -1,
        report_ids: list[str] | None = None,
    ) -> None:
        self.description = description
        self.score = score
        self.batch = batch
        self.report_ids = report_ids or []


def _record(degraded: list[str] | None, error: Exception) -> None:
    if degraded is not None:
        degraded.append(type(error).__name__)


@register_strategy(
    SearchStrategy.GLOBAL,
    required_roles=(RetrieverRole.DOCUMENT,),
    needs_community_reports=True,
)
class GlobalSearchStrategy(BaseSearchStrategy):
    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        **kwargs: Any,
    ) -> None:
        super().__init__(config, retrievers, **kwargs)
        self.global_search_config = config.search.global_search
        self.ignore_errors = config.processing.ignore_errors
        self.max_concurrency = config.search.max_concurrency
        self.target_language = language_name(
            config.processing.translation.target_language
        )

        factory = self.providers.llm_factory

        str_output_parser = StrOutputParser()
        # Built only for dynamic community selection, which rates reports.
        if self.global_search_config.use_dynamic_selection:
            self.community_relevance_scorer: Runnable = setup_chain(
                factory=factory,
                model_id=self.config.model_id(ModelRole.COMMUNITY_RELEVANCE),
                prompt_class=CommunityRelevancePrompt,
                parser=str_output_parser,
                custom_prompts=config.custom_prompts,
            )
        # MAP step: rate community-report key points (0-100) for the query. Cheap
        # model by default since rating is cheap. StrOutputParser + a robust JSON
        # parse below keeps map output handling fault-tolerant.
        self.map_rater: Runnable = setup_chain(
            factory=factory,
            model_id=self.config.model_id(ModelRole.GLOBAL_MAP),
            prompt_class=GlobalMapPrompt,
            parser=str_output_parser,
            custom_prompts=config.custom_prompts,
        )
        # REDUCE step: synthesize the final answer from the ranked key points.
        self.map_reducer: Runnable = setup_chain(
            factory=factory,
            model_id=self.config.model_id(ModelRole.GLOBAL_REDUCE),
            prompt_class=MapReduceSummaryPrompt,
            parser=str_output_parser,
            custom_prompts=config.custom_prompts,
        )
        # One prepared input per map LLM call (each input packs up to
        # ``map_context_tokens`` of reports), max_concurrency of them in
        # flight. The map chain already retries transient Bedrock errors as a
        # query chain, so BatchProcessor calls each input once (max_attempts=1).
        self.batch_processor = BatchProcessor(
            max_concurrency=config.search.max_concurrency, max_attempts=1
        )

    async def asearch(
        self, query: SearchQuery, config: RunnableConfig | None = None
    ) -> SearchResult:
        start_time = time.time()
        rerank = self._new_rerank_trace()
        logger.info(
            "Global search started - query: %s ('%s')",
            text_digest(query.query),
            query.search_type.value,
        )
        # Per-query counts reported in the search metadata.
        stats: dict[str, Any] = {
            "global_community_depth": self.global_search_config.community_depth,
            "global_dynamic_selection": (
                self.global_search_config.use_dynamic_selection
            ),
            "global_level_capped": False,
        }
        # Type names of the failures ignore_errors degraded.
        degraded: list[str] = []

        if self.global_search_config.use_dynamic_selection:
            reports = await self._select_dynamically(
                query, stats, config=config, degraded=degraded
            )
        else:
            reports = await self._cut_reports(query, stats)
        stats["global_selected_report_count"] = len(reports)
        if not reports:
            if not stats.get("global_rated_report_count"):
                await self._require_report_hierarchy(query)
            logger.warning("No results found for query: %s", text_digest(query.query))

        results = reports
        no_relevant_points = False
        if reports and self.global_search_config.enable_map_reduce:
            results = await self._apply_map_reduce(
                reports, query, config=config, degraded=degraded, stats=stats
            )
            # Only an all-below-threshold map phase empties a non-empty input.
            no_relevant_points = not results

        if results and self.global_search_config.text_unit_slots:
            results = results + await self._text_units(results, query, rerank)

        processing_time = time.time() - start_time
        logger.info(
            "Search completed - retrieved: %s results (%.3fs)",
            len(results),
            processing_time,
        )
        return SearchResult(
            query=query,
            results=results,
            total_results=len(results),
            search_strategy="global_search",
            processing_time=processing_time,
            metadata={
                **stats,
                "map_reduce_applied": self._was_map_reduce_applied(results),
                "map_reduce_no_relevant_points": no_relevant_points,
                "rerank": rerank.summary(),
                # The chain sets error.type on the strategy span from it.
                **({"degraded_error": degraded[0]} if degraded else {}),
            },
        )

    # ------------------------------------------------------------------ #
    # Report selection
    # ------------------------------------------------------------------ #

    async def _report_cap(self, suffix: str | None) -> int:
        """Reports one cut or selection step contributes at most.

        A store request returns at most ``max_query_size`` hits, and no more
        than the reports index's live ``max_result_window``; the cut probe
        asks for one more than the cap (``_browse_reports``), so the cap is
        at most the smaller of the two less one: at a cap of the page size a
        cut of exactly the cap could not be told apart from a larger one.
        """
        page = self.config.indexing.opensearch.max_query_size
        if (retriever := self.document_retriever) is not None:
            limits = await retriever.aindex_limits(
                self.config.indexing.opensearch.community_reports_index_prefix, suffix
            )
            page = min(page, limits.max_result_window)
        return max(1, min(self.global_search_config.max_reports_per_level, page - 1))

    async def _scope_terms(self, scope: dict[str, Any], suffix: str | None) -> int:
        """The values one ``terms`` clause of the reports index takes, or 0
        when every list of ``scope`` fits in one."""
        retriever = self.document_retriever
        assert retriever is not None
        prefix = self.config.indexing.opensearch.community_reports_index_prefix
        terms = (await retriever.aindex_limits(prefix, suffix)).max_terms_count
        fits = all(not isinstance(v, list) or len(v) <= terms for v in scope.values())
        return 0 if fits else terms

    async def _browse_reports(
        self, query: SearchQuery, scope: dict[str, Any], label: str
    ) -> tuple[list[RetrievalResult], bool]:
        """Up to the cap of the reports matching ``scope`` (and the caller's
        filters), most important (``rank``, then id) first, and whether that
        is all.

        One more report than the cap is asked for, so a set of exactly the
        cap is complete (``_report_cap`` leaves room for it in one request).
        A scope list past ``max_terms_count`` (the children of many relevant
        reports) is asked in batches of that many, each for the probe, at
        most ``search.max_concurrency`` at once, and merged before the cut.
        """
        retriever = self.document_retriever
        if retriever is None:
            return [], True
        cap = await self._report_cap(query.suffix)
        probe = cap + 1
        scopes = [scope]
        if terms := await self._scope_terms(scope, query.suffix):
            key = next(
                k for k, v in scope.items() if isinstance(v, list) and len(v) > terms
            )
            values = scope[key]
            scopes = [
                {**scope, key: values[i : i + terms]}
                for i in range(0, len(values), terms)
            ]
        parts = await gather_bounded(
            self.config.search.max_concurrency,
            (
                self._safe_aretrieve(
                    retriever,
                    SearchQuery(
                        query="",
                        search_type=SearchType.LEXICAL,
                        top_k=probe,
                        index_prefixes=[
                            self.config.indexing.opensearch.community_reports_index_prefix
                        ],
                        suffix=query.suffix,
                        filters=self._scoped_filters(query, **part),
                    ),
                    label,
                )
                for part in scopes
            ),
        )
        reports = [report for part in parts for report in part]
        reports.sort(
            key=lambda r: (
                -(as_float((r.metadata or {}).get("rank")) or 0.0),
                str((r.metadata or {}).get("id") or r.source),
            )
        )
        return reports[:cap], len(reports) < probe

    async def _most_similar_reports(
        self,
        query: SearchQuery,
        scope: dict[str, Any],
        label: str,
        stats: dict[str, Any],
    ) -> list[RetrievalResult]:
        """The cap of the reports matching ``scope`` most similar to the query:
        the part of a set larger than the cap a query can afford. That is the
        only use of similarity in global search."""
        retriever = self.document_retriever
        if retriever is None:
            return []
        cap = await self._report_cap(query.suffix)
        if terms := await self._scope_terms(scope, query.suffix):
            # A query with text is one search, and a terms clause takes at
            # most max_terms_count values: the browse's reports (by rank) are
            # mapped instead.
            stats["global_level_capped"] = True
            logger.warning(
                "%s: more than %s community reports, of more communities than "
                "a similarity search can scope (the reports index's "
                "max_terms_count, %s); mapping the %s with the highest rank.",
                label,
                cap,
                terms,
                cap,
            )
            return []
        similar = query.model_copy(
            update={
                "top_k": cap,
                "retrieval_multiplier": 1,
                "index_prefixes": [
                    self.config.indexing.opensearch.community_reports_index_prefix
                ],
                "filters": self._scoped_filters(query, **scope),
            }
        )
        ranked = await self._safe_aretrieve(
            retriever, similar, f"{label} (most similar)"
        )
        stats["global_level_capped"] = True
        logger.warning(
            "%s: more than %s community reports; mapping the %s most similar "
            "to the query (search.global_search.max_reports_per_level, at most "
            "indexing.opensearch.max_query_size - 1 and the reports index's "
            "max_result_window - 1).",
            label,
            cap,
            cap,
        )
        return ranked

    async def _fetch_reports(
        self,
        query: SearchQuery,
        scope: dict[str, Any],
        label: str,
        stats: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Every report matching ``scope``, or the cap of them most similar
        to the query when there are more."""
        reports, complete = await self._browse_reports(query, scope, label)
        if complete:
            return reports
        return await self._most_similar_reports(query, scope, label, stats) or reports

    async def _cut_reports(
        self, query: SearchQuery, stats: dict[str, Any]
    ) -> list[RetrievalResult]:
        """The reports of the ``community_depth`` cut (MS GraphRAG's fixed
        ``community_level``): every community at that depth plus the leaves
        above it.

        A cut with more reports than the cap gives way to the deepest
        shallower cut that fits, which still covers every entity, in fewer,
        broader reports. Only when even the root cut does not fit are the
        reports of the configured cut most similar to the query mapped.
        """
        depth = self.global_search_config.community_depth
        label = "Community report cut"
        with observability.span("kg_rag.global.select", depth=depth) as span:
            first, complete = await self._browse_reports(
                query, cut_filter(depth), label
            )
            reports, used = first, depth
            while not complete and used > 0:
                used -= 1
                reports, complete = await self._browse_reports(
                    query, cut_filter(used), label
                )
            if not complete:
                used = depth
                reports = (
                    await self._most_similar_reports(
                        query, cut_filter(depth), label, stats
                    )
                    or first
                )
            elif used < depth:
                logger.warning(
                    "The depth-%s cut has more than %s community reports; "
                    "mapping the depth-%s cut (%s reports), which covers the "
                    "same entities in broader reports "
                    "(search.global_search.max_reports_per_level)",
                    depth,
                    await self._report_cap(query.suffix),
                    used,
                    len(reports),
                )
            stats["global_cut_depth"] = used
            span.set_attributes({"kg_rag.reports": len(reports), "kg_rag.cut": used})
        logger.info("Global search cut at depth %s: %s reports", used, len(reports))
        return reports

    async def _select_dynamically(
        self,
        query: SearchQuery,
        stats: dict[str, Any],
        *,
        config: RunnableConfig | None = None,
        degraded: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """MS GraphRAG dynamic community selection, root down.

        Rates the root reports; the children of each relevant report are
        rated next, down to the leaves, and a relevant child drops its
        parent. A depth with no relevant report and nothing queued rates the
        whole next depth, down to ``community_depth`` (MS GraphRAG's
        ``max_level``, which bounds only that fallback). Returns the relevant
        reports, most relevant first.
        """
        max_depth = self.global_search_config.community_depth
        threshold = self.global_search_config.relevance_threshold
        semaphore = asyncio.Semaphore(self.max_concurrency)
        relevant: dict[str, tuple[float, int, RetrievalResult]] = {}
        rated = 0
        depth = 0
        with observability.span(
            "kg_rag.global.select", depth=max_depth, dynamic=True
        ) as span:
            queue = await self._fetch_reports(
                query, {"depth": 0}, "Root community reports", stats
            )
            while queue:
                scores = await asyncio.gather(
                    *(
                        self._rate_report(r, query, semaphore, config=config)
                        for r in queue
                    ),
                    return_exceptions=True,
                )
                children: list[str] = []
                # As in the map phase: one failed rating (a throttled call)
                # keeps its report, a step none of whose ratings succeeded is
                # the model not answering.
                failures = [o for o in scores if isinstance(o, BaseException)]
                for failure in failures:
                    if not isinstance(failure, Exception) or is_fatal_retrieval_error(
                        failure
                    ):
                        raise failure
                if failures and len(failures) == len(scores) and not self.ignore_errors:
                    raise failures[0]
                for report, outcome in zip(queue, scores, strict=True):
                    rated += 1
                    score = threshold  # MS GraphRAG keeps a report it could not rate
                    if isinstance(outcome, Exception):
                        logger.warning("Community rating failed: %s", outcome)
                        _record(degraded, outcome)
                    elif not isinstance(outcome, BaseException):
                        score = outcome
                    if score < threshold:
                        continue
                    metadata = report.metadata or {}
                    relevant[self._community_id(report)] = (score, rated, report)
                    children.extend(metadata.get("child_community_ids") or [])
                    if parent := metadata.get("parent_community_id"):
                        relevant.pop(str(parent), None)
                depth += 1
                if children:
                    queue = await self._fetch_reports(
                        query,
                        {"community_id": list(dict.fromkeys(children))},
                        "Child community reports",
                        stats,
                    )
                elif not relevant and depth <= max_depth:
                    queue = await self._fetch_reports(
                        query, {"depth": depth}, "Community reports", stats
                    )
                else:
                    queue = []
            span.set_attributes(
                {"kg_rag.rated_reports": rated, "kg_rag.reports": len(relevant)}
            )
        stats["global_rated_report_count"] = rated
        logger.info(
            "Dynamic community selection rated %s reports, kept %s",
            rated,
            len(relevant),
        )
        ranked = sorted(relevant.values(), key=lambda e: (-e[0], e[1]))
        # Copies: the score is the rating, not the store's retrieval score.
        return [report.model_copy(update={"score": s}) for s, _, report in ranked]

    async def _rate_report(
        self,
        report: RetrievalResult,
        query: SearchQuery,
        semaphore: asyncio.Semaphore,
        *,
        config: RunnableConfig | None = None,
    ) -> float:
        """The report's relevance to the query on a 0-1 scale (rating / 10)."""
        async with semaphore:
            llm_output = await self.community_relevance_scorer.ainvoke(
                {"community_summary": report.content, "query": query.query}, config
            )
        parsed = safe_float_parse(llm_output, default_value=5.0)
        return (parsed / 10.0) if parsed is not None else 0.0

    @staticmethod
    def _community_id(report: RetrievalResult) -> str:
        return str((report.metadata or {}).get("community_id") or report.source)

    async def _require_report_hierarchy(self, query: SearchQuery) -> None:
        """Raise when the stored reports predate the hierarchy fields.

        Such reports match no cut, so global search would answer from nothing
        on every query rather than on this one.

        Raises:
            ConfigurationError: A report exists but carries no ``depth``.
        """
        retriever = self.document_retriever
        if retriever is None:
            return
        probe = SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=1,
            index_prefixes=[
                self.config.indexing.opensearch.community_reports_index_prefix
            ],
            suffix=query.suffix,
        )
        found = await self._safe_aretrieve(retriever, probe, "Community report probe")
        if found and (found[0].metadata or {}).get("depth") is None:
            raise ConfigurationError(
                "The community reports were indexed without their place in the "
                "community hierarchy, which global search selects them by. "
                "Re-run ingestion with indexing.reset: true to re-index them."
            )

    async def _text_units(
        self,
        results: list[RetrievalResult],
        query: SearchQuery,
        rerank: RerankTrace,
    ) -> list[RetrievalResult]:
        """``text_unit_slots`` text units of the reports' communities, reranked."""
        community_ids = [
            self._community_id(r)
            for r in results
            if r.retriever_type == SectionType.COMMUNITY.value
        ]
        if not community_ids or self.document_retriever is None:
            return []
        slots = self.global_search_config.text_unit_slots
        search_query = query.model_copy(
            update={
                "top_k": slots,
                "retrieval_multiplier": 1,
                "index_prefixes": [
                    self.config.indexing.opensearch.text_units_index_prefix
                ],
                "filters": self._scoped_filters(
                    query, community_ids=list(dict.fromkeys(community_ids))
                ),
            }
        )
        chunks = await self._safe_aretrieve(
            self.document_retriever, search_query, "Community text units"
        )
        if not chunks:
            return []
        return await self._fuse_and_rerank(
            {"text_units": chunks},
            rerank_trace=rerank,
            top_k=slots,
            query=query.query,
        )

    # ------------------------------------------------------------------ #
    # Map-reduce
    # ------------------------------------------------------------------ #

    async def _apply_map_reduce(
        self,
        results: list[RetrievalResult],
        query: SearchQuery,
        *,
        config: RunnableConfig | None = None,
        degraded: list[str] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """MS GraphRAG global-search map-reduce over every selected report.

        MAP — rate the key points of each token-sized batch of reports (0-100).
        FILTER+RANK — drop low-scored points, sort by score descending.
        PACK — take ranked points up to ``max_map_reduce_tokens``.
        REDUCE — synthesize the answer from the ranked, packed points.

        The result is the synthesized item followed by the reports the
        packed points cite (``_evidence``), best point first: the evidence the
        points came from, for the answer budget left over and the reported
        sources. Below ``map_reduce_min_results`` reports the results pass
        through unchanged. If the map phase yields no usable scored points
        (e.g. every map call failed to parse), it degrades to the legacy
        concat-and-reduce path so global search never hard-fails.

        If the map phase DID rate the reports but every key point scored at or
        below ``map_relevance_threshold``, the reports were judged irrelevant to
        the query: this returns no results (MS GraphRAG answers with its no-data
        response here), so the chain's empty-context guard answers "cannot
        answer" instead of synthesizing from reports the map step rejected.
        That verdict needs every batch to have been rated: when some batches
        failed (throttling, timeout) or returned unparseable output, their
        reports were never judged, so it degrades to concat-and-reduce over
        just those unrated reports instead.

        The type name of each failure ``ignore_errors`` degraded is appended
        to ``degraded``; the map counts go to ``stats``.
        """
        reports = [
            r for r in results if r.retriever_type == SectionType.COMMUNITY.value
        ]
        if len(reports) < self.global_search_config.map_reduce_min_results:
            return results
        stats = {} if stats is None else stats

        # Counting can be one CountTokens request per report: off the loop.
        batches, report_tokens = await asyncio.to_thread(self._map_batches, reports)
        stats["global_mapped_report_count"] = len(reports)
        stats["global_map_call_count"] = len(batches)
        stats["global_map_report_tokens"] = report_tokens
        logger.info(
            "Global map: %s reports (%s report tokens) in %s map calls",
            len(reports),
            report_tokens,
            len(batches),
        )
        try:
            with observability.span(
                "kg_rag.global.map",
                reports=len(reports),
                map_calls=len(batches),
                report_tokens=report_tokens,
            ):
                map_points, unrated = await self._run_map_phase(
                    batches, query, config=config, degraded=degraded
                )
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.error("Map phase failed: %s", e)
            _record(degraded, e)
            map_points, unrated = [], list(reports)

        ranked_points = self._filter_and_rank_points(map_points)
        if not ranked_points:
            if unrated:
                logger.warning(
                    "No relevant map key point among the rated reports, but %s/%s "
                    "reports were never rated (failed or unparseable map calls); "
                    "degrading to concat-and-reduce over the unrated reports.",
                    len(unrated),
                    len(reports),
                )
                return await self._concat_reduce(
                    unrated, query, config=config, degraded=degraded
                )
            logger.warning(
                "All %s map key points scored at or below threshold %s; "
                "no relevant community data for the query, returning no context.",
                len(map_points),
                self.global_search_config.map_relevance_threshold,
            )
            return []

        # Counting can be one CountTokens request per point: off the loop.
        packed_points = await asyncio.to_thread(
            self._pack_points_within_budget, ranked_points
        )
        return await self._reduce_from_points(
            packed_points,
            self._evidence(packed_points, batches),
            query,
            config=config,
            degraded=degraded,
        )

    def _map_batches(
        self, reports: list[RetrievalResult]
    ) -> tuple[list[list[RetrievalResult]], int]:
        """``reports`` packed in order into batches of ``map_context_tokens``.

        A report that alone exceeds the budget gets a batch of its own (MS
        GraphRAG's ``build_community_context``). Returns the batches and the
        reports' token count.
        """
        budget = self.global_search_config.map_context_tokens
        costs = self.token_manager.count_tokens_many(
            [self._format_reports([r]) for r in reports]
        )
        batches: list[list[RetrievalResult]] = []
        used = 0
        for report, cost in zip(reports, costs, strict=True):
            if not batches or used + cost > budget:
                batches.append([])
                used = 0
            batches[-1].append(report)
            used += cost
        return batches, sum(costs)

    @classmethod
    def _evidence(
        cls, points: list[_MapPoint], batches: list[list[RetrievalResult]]
    ) -> list[RetrievalResult]:
        """The reports the packed points came from, best point first.

        A point credits the reports of its batch it cites by id (their
        ordinal in the batch, ``_format_reports``); one that cites none of
        them (no ids, or ids the batch does not hold) credits its whole
        batch. Each report scores its best crediting point (0-1). Crediting
        every report of a batch made a point drawn from one report put up to
        a dozen unrelated ones in the answer context and sources.
        """
        best: dict[tuple[int, int], int] = {}
        for point in points:
            if not 0 <= point.batch < len(batches):
                continue
            batch = batches[point.batch]
            cited = set(point.report_ids)
            credited = [
                i for i in range(len(batch)) if cls._report_ordinal(i) in cited
            ] or range(len(batch))
            for i in credited:
                key = (point.batch, i)
                best[key] = max(best.get(key, 0), point.score)
        return [
            batches[b][i].model_copy(update={"score": best[(b, i)] / 100.0})
            for b, i in sorted(best, key=lambda key: (-best[key], key))
        ]

    async def _run_map_phase(
        self,
        report_batches: list[list[RetrievalResult]],
        query: SearchQuery,
        *,
        config: RunnableConfig | None = None,
        degraded: list[str] | None = None,
    ) -> tuple[list[_MapPoint], list[RetrievalResult]]:
        """Fan one map call per batch of reports and parse the scored points.

        BatchProcessor runs the calls concurrently with graceful per-item
        fallback. Returns the scored points (each tagged with its batch) and
        the reports of every batch that was NOT rated (the call failed, or its
        output was not parseable map JSON), so the caller can tell "rated
        irrelevant" apart from "never rated". An unrated batch is recorded in
        ``degraded``; when every map call failed (other than on a prompt
        longer than the model's context) and ``ignore_errors`` is off, the
        first error is raised (as DRIFT does when a step fails entirely), not
        answered from the unjudged reports.
        """

        def prepare_inputs(
            batches: list[list[RetrievalResult]],
        ) -> list[dict[str, Any]]:
            return [
                {
                    "query": query.query,
                    "reports": self._format_reports(batch),
                    "target_language": self.target_language,
                }
                for batch in batches
            ]

        # BatchProcessor turns an item that failed every attempt into
        # BATCH_ITEM_FAILED, and every attempt runs through sequential_func,
        # so fatal errors are recorded there: a model the
        # caller cannot access must fail the query, not degrade it to
        # "unrated" and an answer from the raw reports.
        fatal_errors: list[Exception] = []
        # A 400 leaves its batch unrated (a vLLM/llama.cpp overflow), but one
        # every batch got is the model refusing the request (an unsupported
        # parameter, an unknown model id): with a fast map model the answer
        # call, on another model, would not fail on it.
        bad_requests: list[Exception] = []
        errors: list[Exception] = []

        def sequential_func(single_input: dict[str, Any]) -> str:
            try:
                return str(self.map_rater.invoke(single_input, config))
            except Exception as e:
                errors.append(e)
                if is_bad_request(e):
                    bad_requests.append(e)
                elif is_fatal_retrieval_error(e):
                    fatal_errors.append(e)
                raise

        raw_outputs = await asyncio.to_thread(
            self.batch_processor.execute_with_fallback,
            items_to_process=report_batches,
            prepare_inputs_func=prepare_inputs,
            sequential_func=sequential_func,
            task_name="global_search_map",
            show_progress=False,
        )
        if fatal_errors:
            raise fatal_errors[0]

        points: list[_MapPoint] = []
        unrated: list[RetrievalResult] = []
        failed = 0
        for i, batch in enumerate(report_batches):
            raw = raw_outputs[i] if i < len(raw_outputs) else None
            failed += not isinstance(raw, str)
            # A failed item is BATCH_ITEM_FAILED, not a string.
            batch_points = (
                self._parse_map_payload(raw) if isinstance(raw, str) and raw else None
            )
            if batch_points is None:
                unrated.extend(batch)
                continue
            for point in batch_points:
                point.batch = i
            points.extend(batch_points)
        report_count = sum(len(batch) for batch in report_batches)
        if bad_requests and len(unrated) == report_count:
            raise bad_requests[0]
        # A call BatchProcessor gave up on after call_timeout_seconds never
        # raised inside sequential_func: it fails without an error of its own.
        timeout = LanguageModelTimeoutError(
            "map call gave up after " f"{self.batch_processor.call_timeout_seconds}s"
        )
        # A prompt too long for the model is that batch's own problem (it is
        # answered from its unrated reports); any other failure of every map
        # call is the model not answering.
        refusals = [e for e in errors if not isinstance(e, ContextOverflowError)]
        if failed == len(report_batches) and not self.ignore_errors:
            if refusals:
                raise refusals[0]
            if not errors:
                raise timeout
        if unrated:
            logger.warning(
                "Global map phase left %s/%s reports unrated (failed or "
                "unparseable map calls)",
                len(unrated),
                report_count,
            )
            _record(
                degraded,
                (
                    errors[0]
                    if errors
                    else (
                        timeout if failed else LanguageModelError("map output not JSON")
                    )
                ),
            )
        return points, unrated

    @staticmethod
    def _report_ordinal(index: int) -> str:
        """The id of a batch's ``index``-th report in the map prompt."""
        return str(index + 1)

    @classmethod
    def _format_reports(cls, reports: list[RetrievalResult]) -> str:
        """``reports`` for the map prompt, each headed by its ordinal in the
        batch: short ids a model copies back reliably, where a community
        UUID cost tokens per report and came back garbled."""
        return "\n\n".join(
            f"--- Report (id: {cls._report_ordinal(i)}) ---\n{report.content}"
            for i, report in enumerate(reports)
        )

    @staticmethod
    def _parse_map_payload(raw: str) -> list[_MapPoint] | None:
        """Parse a map JSON payload into scored key points.

        Tolerates the LLM wrapping the JSON in prose or code fences. ``None``
        when ``raw`` is not map JSON (the batch counts as unrated); an empty
        list means the batch WAS rated and yielded no key points.
        """
        payload = parse_llm_json(raw)
        if not payload:
            return None

        points: list[_MapPoint] = []
        for item in payload.get("points", []) or []:
            if not isinstance(item, dict):
                continue
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            score = safe_float_parse(str(item.get("score", 0)), default_value=0.0)
            # Guard against non-finite scores BEFORE int(): the LLM map output is
            # untrusted, and parse_llm_json (json.loads) accepts Infinity/-Infinity/
            # NaN while a string like "1e999" overflows float() to inf. int(inf)
            # raises OverflowError and int(nan) raises ValueError, which would
            # abort the whole query. Coerce non-finite to 0 (below the relevance
            # threshold, so the point is simply dropped).
            if score is None or not math.isfinite(score):
                score_int = 0
            else:
                score_int = int(score)
            score_int = max(0, min(100, score_int))
            raw_ids = item.get("report_ids")
            report_ids = (
                [str(rid).strip() for rid in raw_ids]
                if isinstance(raw_ids, list)
                else []
            )
            points.append(
                _MapPoint(
                    description=description, score=score_int, report_ids=report_ids
                )
            )
        return points

    def _filter_and_rank_points(self, points: list[_MapPoint]) -> list[_MapPoint]:
        threshold = self.global_search_config.map_relevance_threshold
        kept = [p for p in points if p.score > threshold]
        kept.sort(key=lambda p: p.score, reverse=True)
        return kept

    def _pack_points_within_budget(self, points: list[_MapPoint]) -> list[_MapPoint]:
        budget = self.global_search_config.max_map_reduce_tokens
        packed: list[_MapPoint] = []
        used = 0
        # Exact counts, taken concurrently rather than one request per point.
        costs = self.token_manager.count_tokens_many([p.description for p in points])
        for point, cost in zip(points, costs, strict=True):
            if packed and used + cost > budget:
                break
            packed.append(point)
            used += cost
        logger.debug(
            "Packed %s/%s ranked key points into the %s-token reduce budget",
            len(packed),
            len(points),
            budget,
        )
        return packed

    async def _reduce_from_points(
        self,
        points: list[_MapPoint],
        results: list[RetrievalResult],
        query: SearchQuery,
        *,
        config: RunnableConfig | None = None,
        degraded: list[str] | None = None,
    ) -> list[RetrievalResult]:
        synthesis_input = "\n\n".join(
            f"- (relevance {p.score}) {p.description}" for p in points
        )
        if not self.global_search_config.reduce_with_llm:
            # Hand the ranked points to the answer model directly: it already
            # synthesizes from its context, so a separate reduce LLM only adds a
            # call and a second rewrite that can drop or contradict facts.
            points_result = RetrievalResult(
                content=(
                    "Key points from the community reports, ranked by relevance "
                    "to the query (0-100):\n\n" + synthesis_input
                ),
                score=1.0,
                source="synthesized_key_points",
                retriever_type=SectionType.GENERAL.value,
                metadata={
                    # Map-LLM output, not retrieved evidence: in the answer
                    # context, excluded from the reported sources.
                    "synthesized": True,
                    "source_results_count": len(results),
                    "ranked_key_points": len(points),
                },
            )
            return [points_result] + results
        try:
            summary = await self.map_reducer.ainvoke(
                {
                    "summaries": synthesis_input,
                    "query": query.query,
                    "target_language": self.target_language,
                },
                config,
            )
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.error("Map-reduce synthesis failed: %s", e)
            _record(degraded, e)
            return results

        summary_result = RetrievalResult(
            content=summary,
            score=1.0,
            source="synthesized_summary",
            retriever_type=SectionType.GENERAL.value,
            metadata={
                # LLM output, not retrieved evidence: kept in the answer context
                # (design §6.1.1) but excluded from the reported sources.
                "synthesized": True,
                "source_results_count": len(results),
                "ranked_key_points": len(points),
            },
        )
        return [summary_result] + results

    async def _concat_reduce(
        self,
        results: list[RetrievalResult],
        query: SearchQuery,
        *,
        config: RunnableConfig | None = None,
        degraded: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Legacy direct concat-and-reduce path (map-reduce degradation target).

        Without ``reduce_with_llm`` the reports pass through unchanged: the
        answer model reads them directly instead of a summary of them.
        """
        if not self.global_search_config.reduce_with_llm:
            return results
        try:
            context = "\n\n---\n\n".join([r.content for r in results])
            summary = await self.map_reducer.ainvoke(
                {
                    "summaries": context,
                    "query": query.query,
                    "target_language": self.target_language,
                },
                config,
            )
            summary_result = RetrievalResult(
                content=summary,
                score=1.0,
                source="synthesized_summary",
                retriever_type=SectionType.GENERAL.value,
                metadata={
                    "synthesized": True,
                    "source_results_count": len(results),
                },
            )
            return [summary_result] + results
        except Exception as e:
            if not self.ignore_errors:
                raise

            logger.error("Map-reduce synthesis failed: %s", e)
            _record(degraded, e)
            return results

    def _was_map_reduce_applied(self, final_results: list[RetrievalResult]) -> bool:
        if not self.global_search_config.enable_map_reduce:
            return False
        return any("synthesized" in (r.source or "") for r in final_results)
