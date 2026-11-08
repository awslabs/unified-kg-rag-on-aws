# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MS GraphRAG global search over a cut of the community hierarchy.

The reports of one cut (``community_depth``, see
``domain.ingestion.community_hierarchy``), or of a root-down dynamic
selection, are all mapped into scored key points in token-sized batches;
the ranked points, packed into a token budget, are the answer context.
"""

import asyncio
import json
import re
import time
from typing import Any

from langchain_core.exceptions import ContextOverflowError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig

from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
    evidence_lineage,
    record_degradation,
)
from unified_kg_rag.adapters.retrieval.hybrid_scorer import RerankTrace
from unified_kg_rag.adapters.retrieval.strategy_registry import register_strategy
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.adapters.storage.filter_schema import (
    document_filters,
)
from unified_kg_rag.adapters.store_errors import (
    is_bad_request,
    is_fatal_retrieval_error,
)
from unified_kg_rag.domain.ingestion.community_hierarchy import cut_filter
from unified_kg_rag.domain.models import (
    Config,
    Methodology,
    ModelRole,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.domain.models.methodology import REPORT_READER_ARTIFACTS
from unified_kg_rag.domain.prompts import CommunityRelevancePrompt, GlobalMapPrompt
from unified_kg_rag.domain.retrieval.degradation import DegradationStep
from unified_kg_rag.shared import (
    ConfigurationError,
    LanguageModelError,
    get_logger,
    observability,
)
from unified_kg_rag.shared.utils import (
    parse_llm_json,
    parse_llm_score,
    safe_float_parse,
    text_digest,
)
from unified_kg_rag.shared.utils.concurrency import gather_bounded

logger = get_logger(__name__)

# The source id of each ranked key point (``_KEY_POINTS_SOURCE_<rank>``).
_KEY_POINTS_SOURCE = "synthesized_key_points"


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


# A rating's number: the first one in the answer that is not part of a
# mention of the scale, in English ("1-10", "1 to 10", "out of 10", "/10")
# or in the corpus language the model may answer in: a range ("1~10",
# "1から10", "1到10", "1에서 10") or the maximum ("10점 만점에", "10点満点中",
# "满分10分", "10段階", "10分制").
_FIRST_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SCALE_MENTION = re.compile(
    r"(?<![\d.])[01]\s*(?:-|\u2013|~|\uff5e|\u301c|to|から|到|至|에서|부터)\s*"
    r"10{1,2}(?![\d.])"
    r"|(?:\bout\s+of|/|满分|滿分)\s*10{1,2}(?![\d.])"
    r"|(?<![\d.])10{1,2}\s*(?:점\s*(?:만점|중|척도)|点\s*満点|段階|分制|分满分|分滿分)",
    re.IGNORECASE,
)


# A backslash escape JSON allows (kept), or a lone backslash (dropped): a
# model writes ``\'`` or ``\_`` in a description.
_ESCAPE = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})?')
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
# One key point as the map prompt asks for it, read without a JSON parser:
# its description runs to the ``", "score"`` that follows it, so a quote the
# model left unescaped inside it does not end it.
_MAP_POINT = re.compile(
    r'"description"\s*:\s*"(?P<description>.*?)"\s*,\s*"score"\s*:\s*'
    r'"?(?P<score>-?\d+(?:\.\d+)?)"?'
    r'(?:\s*,\s*"report_ids"\s*:\s*\[(?P<ids>[^\]]*)\])?',
    re.DOTALL,
)
_REPORT_ID = re.compile(r"[^\s,\"']+")


def _json_string(body: str) -> str:
    """A JSON string's ``body`` decoded, or as written when it does not
    decode."""
    try:
        decoded = json.loads(f'"{body}"', strict=False)
    except ValueError:
        return body.replace('\\"', '"')
    return decoded if isinstance(decoded, str) else body


def _repaired_map_payload(raw: str) -> dict[str, Any] | None:
    """Map JSON a model wrote with a defect ``json.loads`` refuses, read as
    MS GraphRAG reads it (``try_parse_json_object`` repairs it with
    ``json_repair``), for the defects seen in map outputs: control
    characters (a raw line break) inside a string, an invalid backslash
    escape, a trailing comma, the points as a bare list, or a quote left
    unescaped inside a description (the points are then read one by one,
    ``_MAP_POINT``). ``None`` when no point can be read."""
    text = (raw or "").strip()
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start == -1 or end <= start:
        return None
    text = text[start : end + 1]
    text = _ESCAPE.sub(lambda m: m.group(0) if m.group(1) else "", text)
    text = _TRAILING_COMMA.sub(r"\1", text)
    try:
        parsed = json.loads(text, strict=False)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        parsed = {"points": parsed}
    if isinstance(parsed, dict) and isinstance(parsed.get("points"), list):
        return parsed
    points = [
        {
            "description": _json_string(m.group("description")),
            "score": m.group("score"),
            "report_ids": _REPORT_ID.findall(m.group("ids") or ""),
        }
        for m in _MAP_POINT.finditer(text)
    ]
    return {"points": points} if points else None


def _unparsed_shape(raw: str) -> str:
    """What a map output that is not map JSON looks like, for the log: its
    length and the parser's complaint, never its text."""
    text = (raw or "").strip()
    if not text:
        return "empty"
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return f"{len(text)} characters, no JSON object"
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError as e:
        return f"{len(text)} characters, {type(e).__name__}: {e}"
    return f"{len(text)} characters, a {type(parsed).__name__} without 'points'"


@register_strategy(
    SearchStrategy.GLOBAL,
    required_roles=(RetrieverRole.DOCUMENT,),
    required_artifacts=REPORT_READER_ARTIFACTS,
)
class GlobalSearchStrategy(BaseSearchStrategy):
    methodology = Methodology.GRAPHRAG

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

        factory = self.providers.llm_factory

        str_output_parser = StrOutputParser()
        # Built only for dynamic community selection, which rates reports.
        if self.global_search_config.use_dynamic_selection:
            self.community_relevance_scorer: Runnable[Any, Any] = setup_chain(
                factory=factory,
                model_id=self.config.model_id(ModelRole.COMMUNITY_RELEVANCE),
                prompt_class=CommunityRelevancePrompt,
                parser=str_output_parser,
                custom_prompts=config.custom_prompts,
                output_ceiling=config.search.output_ceilings.community_relevance,
            )
        # MAP step: rate community-report key points (0-100) for the query. Cheap
        # model by default since rating is cheap. StrOutputParser + a robust JSON
        # parse below keeps map output handling fault-tolerant.
        self.map_rater: Runnable[Any, Any] = setup_chain(
            factory=factory,
            model_id=self.config.model_id(ModelRole.GLOBAL_MAP),
            prompt_class=GlobalMapPrompt,
            parser=str_output_parser,
            custom_prompts=config.custom_prompts,
        )

    def filtered_index_prefixes(self, strategy: SearchStrategy) -> list[str]:
        # The reports, and with text_unit_slots their communities' text units.
        o = self.config.indexing.opensearch
        return [o.community_reports_index_prefix] + (
            [o.text_units_index_prefix]
            if self.global_search_config.text_unit_slots
            else []
        )

    def reserved_filter_keys(self, strategy: SearchStrategy) -> frozenset[str]:
        # The report selection's scope (the cut, or the dynamic browse's
        # depths and children) and the text units' communities.
        c = self.global_search_config
        keys = (
            {"depth", "community_id"} if c.use_dynamic_selection else set(cut_filter(0))
        )
        if c.text_unit_slots:
            keys.add("community_ids")
        return frozenset(keys)

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
        if self.global_search_config.use_dynamic_selection:
            reports = await self._select_dynamically(query, stats, config=config)
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
                reports, query, config=config, stats=stats
            )
            # Only an all-below-threshold map phase empties a non-empty input.
            no_relevant_points = not results

        if results and not self._was_map_reduce_applied(results):
            # The reports as they are (no map, or the map left them unrated):
            # each cites what its findings cite (the key points' reports
            # carry it from the map-reduce already).
            results = await self._cite_report_findings(query, results)

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
                -(safe_float_parse((r.metadata or {}).get("rank")) or 0.0),
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
        reports of the configured cut most similar to the query mapped, and
        at once under a filter on a report key (``community_id``, ``level``,
        ``rating``; a key the reports index declares, not a document filter
        or a text-unit key such as ``n_tokens``): a shallower cut's reports
        pass it or not on their own, not by covering the configured cut's
        (document filters pass a report through any member's documents, so
        they still do).
        """
        depth = self.global_search_config.community_depth
        label = "Community report cut"
        # A key only the text units declare (text_unit_slots) leaves the
        # reports alone. Declared by the document retriever; one without a
        # schema declares every key.
        filters = query.filters or {}
        report_keys = set(filters) - set(document_filters(filters))
        declared = (
            documents.filter_fields_for(
                [self.config.indexing.opensearch.community_reports_index_prefix]
            )
            if report_keys and (documents := self.document_retriever) is not None
            else None
        )
        step_down = not any(
            declared is None or declared.declares(key) for key in report_keys
        )
        with observability.span("kg_rag.global.select", depth=depth) as span:
            first, complete = await self._browse_reports(
                query, cut_filter(depth), label
            )
            reports, used = first, depth
            while not complete and step_down and used > 0:
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
        unrated = 0
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
                        record_degradation(
                            DegradationStep.COMMUNITY_RATING,
                            "community rating failed",
                            outcome,
                        )
                        unrated += 1
                    elif outcome is None:
                        logger.warning(
                            "Community rating held no number; report %s kept "
                            "unrated",
                            report.source,
                        )
                        record_degradation(
                            DegradationStep.COMMUNITY_RATING,
                            "community rating not a number",
                            LanguageModelError.__name__,
                        )
                        unrated += 1
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
        stats["global_unrated_report_count"] = unrated
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
    ) -> float | None:
        """The report's relevance to the query on a 0-1 scale (rating / 10),
        ``None`` when the answer holds no number.

        The prompt asks for a bare 1-10 rating; a model that writes "8/10",
        "Score: 2" or a reason after the number is read by its first number
        outside a mention of the scale ("On a scale of 1-10, I'd rate 8" is
        8, not 1), clamped to 1-10 (an "85" read as a percentage is still
        the top rating). A float() of the whole answer read each of those as
        the default 5 and "nan" as NaN, which no threshold drops."""
        async with semaphore:
            llm_output = await self.community_relevance_scorer.ainvoke(
                {"community_summary": report.content, "query": query.query}, config
            )
        found = _FIRST_NUMBER.search(_SCALE_MENTION.sub(" ", str(llm_output)))
        if found is None:
            return None
        return min(max(float(found.group()), 1.0), 10.0) / 10.0

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
        """``text_unit_slots`` text units of the reports' communities, reranked.

        The query has text, so it is one search: its ``community_ids`` are
        those of the best-ranked reports one ``terms`` clause of the text
        units index takes."""
        community_ids = list(
            dict.fromkeys(
                self._community_id(r)
                for r in results
                if r.retriever_type == SectionType.COMMUNITY.value
            )
        )
        if not community_ids or self.document_retriever is None:
            return []
        prefix = self.config.indexing.opensearch.text_units_index_prefix
        limits = await self.document_retriever.aindex_limits(prefix, query.suffix)
        slots = self.global_search_config.text_unit_slots
        search_query = query.model_copy(
            update={
                "top_k": slots,
                "retrieval_multiplier": 1,
                "index_prefixes": [prefix],
                "filters": self._scoped_filters(
                    query, community_ids=community_ids[: limits.max_terms_count]
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
        stats: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """MS GraphRAG global-search map-reduce over every selected report.

        MAP — rate the key points of each token-sized batch of reports (0-100).
        FILTER+RANK — drop low-scored points, sort by score descending.
        PACK — take ranked points up to ``max_map_reduce_tokens``.
        REDUCE — hand the ranked, packed points to the answer model, one
        citable context section each (``_reduce_from_points``).

        The result is the key points followed by the reports the
        packed points cite (``_evidence``), best point first: the evidence the
        points came from, for the answer budget left over and the reported
        sources (whole reports up to ``evidence_tokens`` when it is set). Below
        ``map_reduce_min_results`` reports the results pass through unchanged.
        If the map phase yields no usable scored points (e.g. every map call
        failed to parse), the reports pass through as they are, so global
        search never hard-fails.

        If the map phase DID rate the reports but every key point scored at or
        below ``map_relevance_threshold``, the reports were judged irrelevant to
        the query: this returns no results (MS GraphRAG answers with its no-data
        response here), so the chain's empty-context guard answers "cannot
        answer" instead of synthesizing from reports the map step rejected.
        That verdict needs every batch to have been rated: when some batches
        failed (throttling, timeout) or returned unparseable output, their
        reports were never judged, so it returns just those unrated reports
        instead.

        Each failure ``ignore_errors`` absorbed is recorded in the query's
        degradation ledger; the map counts go to ``stats``.
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
        # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure - LLM token counts, not credentials
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
                    batches, query, config=config
                )
        except Exception as e:
            # A fatal error (no model access) fails the query even under
            # ignore_errors, as DRIFT's.
            if not self.ignore_errors or is_fatal_retrieval_error(e):
                raise
            logger.error("Map phase failed: %s", e)
            record_degradation(DegradationStep.GLOBAL_MAP, "map phase failed", e)
            map_points, unrated = [], list(reports)

        ranked_points = self._filter_and_rank_points(map_points)
        if not ranked_points:
            if unrated:
                logger.warning(
                    "No relevant map key point among the rated reports, but %s/%s "
                    "reports were never rated (failed or unparseable map calls); "
                    "degrading to the unrated reports as they are.",
                    len(unrated),
                    len(reports),
                )
                return unrated
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
        evidence = self._evidence(packed_points, batches)
        # Read before the evidence cap: a point names its reports' sources
        # even when the cap leaves the reports out of the context.
        lineages = await self._findings_lineages(query, evidence)
        if self.global_search_config.evidence_tokens is not None:
            evidence = await asyncio.to_thread(self._cap_evidence, evidence)
            stats["global_evidence_reports"] = len(evidence)
        return self._reduce_from_points(packed_points, evidence, batches, lineages)

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
    def _credited(
        cls, point: _MapPoint, batches: list[list[RetrievalResult]]
    ) -> list[int]:
        """The ordinals in its batch of the reports ``point`` credits: those
        of its batch it cites by id (their ordinal in the batch,
        ``_format_reports``), or its whole batch when it cites none of them
        (no ids, or ids the batch does not hold)."""
        if not 0 <= point.batch < len(batches):
            return []
        batch = batches[point.batch]
        cited = set(point.report_ids)
        return [
            i for i in range(len(batch)) if cls._report_ordinal(i) in cited
        ] or list(range(len(batch)))

    @classmethod
    def _evidence(
        cls, points: list[_MapPoint], batches: list[list[RetrievalResult]]
    ) -> list[RetrievalResult]:
        """The reports the packed points came from (``_credited``), best
        point first. Each report scores its best crediting point (0-1).
        Crediting every report of a batch made a point drawn from one report
        put up to a dozen unrelated ones in the answer context and sources.
        """
        best: dict[tuple[int, int], int] = {}
        for point in points:
            for i in cls._credited(point, batches):
                key = (point.batch, i)
                best[key] = max(best.get(key, 0), point.score)
        return [
            batches[b][i].model_copy(update={"score": best[(b, i)] / 100.0})
            for b, i in sorted(best, key=lambda key: (-best[key], key))
        ]

    def _cap_evidence(self, evidence: list[RetrievalResult]) -> list[RetrievalResult]:
        """The leading ``evidence`` reports whose text fits in
        ``evidence_tokens``, whole reports in order (best point first)."""
        budget = self.global_search_config.evidence_tokens or 0
        if budget <= 0 or not evidence:
            return []
        costs = self.token_manager.count_tokens_many([r.content for r in evidence])
        kept: list[RetrievalResult] = []
        used = 0
        for report, cost in zip(evidence, costs, strict=True):
            if used + cost > budget:
                break
            kept.append(report)
            used += cost
        # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure - LLM token counts, not credentials
        logger.debug(
            "Kept %s/%s cited reports (%s tokens) within evidence_tokens %s",
            len(kept),
            len(evidence),
            used,
            budget,
        )
        return kept

    async def _run_map_phase(
        self,
        report_batches: list[list[RetrievalResult]],
        query: SearchQuery,
        *,
        config: RunnableConfig | None = None,
    ) -> tuple[list[_MapPoint], list[RetrievalResult]]:
        """Fan one map call per batch of reports and parse the scored points.

        At most ``search.max_concurrency`` calls run at once, each awaited on
        the event loop, so a query deadline or a closed chain cancels the
        calls in flight and starts none of those still queued (as DRIFT's).
        Returns the scored points (each tagged with its batch) and the reports
        of every batch that was NOT rated (the call failed, or its output was
        not parseable map JSON), so the caller can tell "rated irrelevant"
        apart from "never rated". An unrated batch is recorded in the
        degradation ledger; when every map call failed (other than on a prompt
        longer than the model's context) and ``ignore_errors`` is off, the
        first error is raised (as DRIFT does when a step fails entirely), not
        answered from the unjudged reports.
        """
        # A model the caller cannot access must fail the query, not degrade it
        # to "unrated" and an answer from the raw reports.
        fatal_errors: list[Exception] = []
        # A 400 leaves its batch unrated (a vLLM/llama.cpp overflow), but one
        # every batch got is the model refusing the request (an unsupported
        # parameter, an unknown model id): with a fast map model the answer
        # call, on another model, would not fail on it.
        bad_requests: list[Exception] = []
        errors: list[Exception] = []

        async def rate(batch: list[RetrievalResult]) -> str | None:
            # The map chain already retries transient model errors as a query
            # chain. An output that is not map JSON, even repaired
            # (``_parse_map_payload``), is asked for once more: on a real
            # model about 1% of batches came back so, each leaving its
            # reports unrated (one global answer in five lost a batch).
            raw: str | None = None
            for attempt in (1, 2):
                try:
                    raw = str(
                        await self.map_rater.ainvoke(
                            {
                                "query": query.query,
                                "reports": self._format_reports(batch),
                                "target_language": self.query_language(query),
                            },
                            config,
                        )
                    )
                except Exception as e:  # noqa: BLE001 - classified below
                    errors.append(e)
                    if is_bad_request(e):
                        bad_requests.append(e)
                    elif is_fatal_retrieval_error(e):
                        fatal_errors.append(e)
                    return None
                if self._parse_map_payload(raw) is not None:
                    return raw
                logger.warning(
                    "Global map output for %s reports is not map JSON (%s)%s",
                    len(batch),
                    _unparsed_shape(raw),
                    "; asking once more" if attempt == 1 else "",
                )
            return raw

        raw_outputs = await gather_bounded(
            self.max_concurrency, (rate(batch) for batch in report_batches)
        )
        if fatal_errors:
            raise fatal_errors[0]

        points: list[_MapPoint] = []
        unrated: list[RetrievalResult] = []
        for i, (batch, raw) in enumerate(zip(report_batches, raw_outputs, strict=True)):
            batch_points = self._parse_map_payload(raw) if raw else None
            if batch_points is None:
                unrated.extend(batch)
                continue
            for point in batch_points:
                point.batch = i
            points.extend(batch_points)
        report_count = sum(len(batch) for batch in report_batches)
        if bad_requests and len(unrated) == report_count:
            raise bad_requests[0]
        # A prompt too long for the model is that batch's own problem (it is
        # answered from its unrated reports); any other failure of every map
        # call is the model not answering.
        refusals = [e for e in errors if not isinstance(e, ContextOverflowError)]
        if len(errors) == len(report_batches) and refusals and not self.ignore_errors:
            raise refusals[0]
        if unrated:
            logger.warning(
                "Global map phase left %s/%s reports unrated (failed or "
                "unparseable map calls)",
                len(unrated),
                report_count,
            )
            record_degradation(
                DegradationStep.GLOBAL_MAP,
                f"{len(unrated)}/{report_count} reports unrated",
                errors[0] if errors else LanguageModelError.__name__,
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

        Tolerates the LLM wrapping the JSON in prose or code fences, and
        repairs the defects ``_repaired_map_payload`` reads. ``None`` when
        ``raw`` is not map JSON, an object with a ``points`` list (the
        batch counts as unrated); an empty list means the batch WAS rated
        and yielded no key points. (An object without ``points``, such as
        the last point of a bare list, read as rated, lost the batch's
        points without a trace.)
        """
        payload: dict[str, Any] | None = parse_llm_json(raw, answer_keys=("points",))
        if not isinstance((payload or {}).get("points"), list):
            payload = _repaired_map_payload(raw)
        if payload is None:
            return None

        points: list[_MapPoint] = []
        for item in payload.get("points", []) or []:
            if not isinstance(item, dict):
                continue
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            # A non-finite score is 0: below the relevance threshold, so the
            # point is dropped instead of int() aborting the query.
            score_int = parse_llm_score(item.get("score", 0))
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

    def _reduce_from_points(
        self,
        points: list[_MapPoint],
        results: list[RetrievalResult],
        batches: list[list[RetrievalResult]] | None = None,
        lineages: dict[str, dict[str, list[str]]] | None = None,
    ) -> list[RetrievalResult]:
        """The ranked ``points``, one context section each, before
        ``results``.

        The answer model reads the points directly: it already synthesizes
        from its context, so a separate reduce call would only add a call and
        a second rewrite that can drop or contradict facts. Each point is a
        section of its own with the lineage of the reports it credits
        (``_credited``): the text units and documents of the records their
        findings cite (``lineages``, ``_findings_lineages``), or of a
        report's community where its findings cite none. The answer cites
        it ``[n]`` as MS GraphRAG's reduce cites ``[Data: Reports (...)]``,
        and the citation names the reports' documents even when the
        evidence budget left the reports themselves out of the context. A
        report among ``results`` carries the same lineage
        (``_with_findings_lineages``), so its own citation agrees with the
        points'.
        """
        batches = batches or []
        out: list[RetrievalResult] = []
        for rank, point in enumerate(points, start=1):
            credited = [batches[point.batch][i] for i in self._credited(point, batches)]
            lineage = evidence_lineage(credited, lineages)
            cites = bool(lineage["text_unit_ids"] or lineage["document_ids"])
            out.append(
                RetrievalResult(
                    content=(
                        "Key point from the community reports (relevance "
                        f"{point.score}/100): {point.description}"
                    ),
                    score=point.score / 100.0,
                    source=f"{_KEY_POINTS_SOURCE}_{rank}",
                    retriever_type=SectionType.GENERAL.value,
                    metadata={
                        # Map-LLM output, not retrieved evidence: in the
                        # answer context (citable by its reports' lineage),
                        # screened by the output guardrail, never reported as
                        # a source.
                        "synthesized": True,
                        "citable": cites,
                        "key_point_rank": rank,
                        "ranked_key_points": len(points),
                        "source_results_count": len(results),
                        "report_ids": [
                            str(r.source) for r in credited if r.source is not None
                        ],
                        **lineage,
                    },
                )
            )
        return out + self._with_findings_lineages(results, lineages or {})

    def _was_map_reduce_applied(self, final_results: list[RetrievalResult]) -> bool:
        if not self.global_search_config.enable_map_reduce:
            return False
        return any("synthesized" in (r.source or "") for r in final_results)
