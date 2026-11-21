# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-query audit record (``domain.models.audit.QueryAuditRecord``).

``GraphRAGChain`` writes one per query, answered or searched, from the one
place every query ends (``GraphRAGChain._record_query``): ``invoke``,
``ainvoke``, ``search``, ``asearch``, the streams and ``aquery_events``, so
``run-rag``, ``run-eval``, the MCP server and the console each get it with
no code of their own. A query that failed, timed out, was refused, whose
stream was closed early or whose caller cancelled it gets one too (a
cancelled one is ``stopped``, or ``failed`` with ``timeout`` when its
deadline had passed); a host that ends a query before the chain ran it (an
MCP call that timed out or whose client left waiting for a call slot)
records it through the same path (``GraphRAGChain.record_unrun_query``).

It is an INFO record of the ``unified_kg_rag.audit`` logger, its fields the
record's own keys (``logging.log_format: structured`` writes them as one
JSON object). Nothing is built while that logger is off for INFO: set it to
WARNING in ``logging.library_levels`` to turn the records off, or route the
logger to a handler of its own to keep them apart from the other logs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from unified_kg_rag import __version__
from unified_kg_rag.domain.ingestion.cache_keys import answer_settings_fingerprint
from unified_kg_rag.domain.models.audit import (
    AuditBuild,
    AuditEvidence,
    AuditOutcome,
    AuditText,
    QueryAuditRecord,
)
from unified_kg_rag.domain.models.retrieval import SearchType
from unified_kg_rag.domain.models.usage import LLMUsage
from unified_kg_rag.domain.pricing import snapshot_cost_usd
from unified_kg_rag.shared.exceptions import ErrorClass
from unified_kg_rag.shared.logging import get_channel_logger
from unified_kg_rag.shared.observability import MAX_RECORDED_IDS
from unified_kg_rag.shared.utils.common import compute_hash

if TYPE_CHECKING:
    from unified_kg_rag.domain.models import Config

__all__ = [
    "AUDIT_LOGGER_NAME",
    "audit_enabled",
    "emit",
    "evidence_of",
    "outcome_of",
    "query_record",
]

AUDIT_LOGGER_NAME = "unified_kg_rag.audit"
# The event (message) of every record.
AUDIT_EVENT = "query_audit"

_logger = get_channel_logger(AUDIT_LOGGER_NAME)
_level_logger = logging.getLogger(AUDIT_LOGGER_NAME)
# A full SHA-256: a stored answer is checked against its record.
_DIGEST_LENGTH = 64


def audit_enabled() -> bool:
    """Whether the audit logger takes INFO records (none is built if not)."""
    return _level_logger.isEnabledFor(logging.INFO)


def emit(record: QueryAuditRecord) -> None:
    """Write ``record`` on the audit logger."""
    _logger.info(AUDIT_EVENT, **record.model_dump(mode="json"))


def outcome_of(
    error_class: ErrorClass | None,
    *,
    raised: bool,
    stopped: bool = False,
    blocked: bool = False,
    abstained: bool = False,
) -> AuditOutcome:
    """How a query ended: ``stopped`` (its reader left), else by its error
    class (``refused`` for a caller's, else ``failed`` when ``raised``,
    ``error`` for the error answer of ``processing.ignore_errors``), else
    ``blocked``, ``abstained`` or ``answered``."""
    if stopped:
        return AuditOutcome.STOPPED
    if error_class is not None:
        if error_class.is_caller_error:
            return AuditOutcome.REFUSED
        return AuditOutcome.FAILED if raised else AuditOutcome.ERROR
    if blocked:
        return AuditOutcome.BLOCKED
    if abstained:
        return AuditOutcome.ABSTAINED
    return AuditOutcome.ANSWERED


def query_record(
    config: Config,
    *,
    query_id: str,
    trace_id: str | None,
    started: float,
    mode: Literal["rag", "search"],
    fields: Mapping[str, Any],
    namespace: str | None,
    strategy: str | None,
    outcome: AuditOutcome,
    error_class: ErrorClass | None,
    degraded: bool,
    query_limited: bool,
    build: Mapping[str, Any] | None,
    output: Any,
    usage: Mapping[str, Any] | None,
) -> QueryAuditRecord:
    """The audit record of one query: ``fields`` are its input's (a
    ``RAGInput``'s, or those a refused input gave), ``started`` its
    ``time.monotonic()`` start, ``build`` the build record its admission
    read, ``output`` its ``RAGOutput`` or ``SearchOutput`` (``None``: it
    raised or stopped) and ``usage`` its usage snapshot."""
    spent = _usage(usage)
    cost, unpriced = snapshot_cost_usd(usage, config.models.prices)
    requested = fields.get("search_strategy")
    search_type = fields.get("search_type") or config.search.search_type
    top_k = fields.get("top_k")
    filters = fields.get("filters")
    return QueryAuditRecord(
        query_id=query_id,
        trace_id=trace_id,
        finished_at=datetime.now(UTC),
        latency_ms=max(0.0, (time.monotonic() - started) * 1000),
        mode=mode,
        namespace=namespace,
        user_id=_text_field(fields.get("user_id")),
        conversation_id=_text_field(fields.get("conversation_id")),
        requested_strategy=(
            None if requested is None else str(getattr(requested, "value", requested))
        ),
        strategy=strategy,
        search_type=_search_type(search_type),
        top_k=top_k if isinstance(top_k, int) and not isinstance(top_k, bool) else None,
        # A filter's keys name what it scoped the query by; its values (a
        # folder, a file name) stay out, as the question's text does.
        filter_keys=(
            _bounded(sorted(str(k) for k in filters))[0]
            if isinstance(filters, Mapping)
            else []
        ),
        outcome=outcome,
        error_class=None if error_class is None else error_class.value,
        degraded=degraded,
        query_limited=query_limited,
        package_version=__version__,
        config_fingerprint=answer_settings_fingerprint(config),
        build=_build(build),
        question=_digest(fields.get("query")),
        answer=_digest(getattr(output, "answer", None)),
        evidence=evidence_of(output),
        models=sorted(spent.by_model) if spent else [],
        usage=spent,
        cost_usd=cost,
        unpriced_models=unpriced,
    )


def _search_type(value: Any) -> str | None:
    """The :class:`SearchType` value ``value`` names; ``None`` for one that
    names none (a refused input's: its text is not recorded)."""
    try:
        return SearchType(getattr(value, "value", value)).value
    except ValueError:
        return None


def _text_field(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _digest(text: Any) -> AuditText | None:
    if not isinstance(text, str):
        return None
    return AuditText(length=len(text), sha256=compute_hash(text, _DIGEST_LENGTH))


def _usage(usage: Mapping[str, Any] | None) -> LLMUsage | None:
    if not usage:
        return None
    try:
        return LLMUsage.model_validate(dict(usage))
    except ValidationError:
        return None


def _build(record: Mapping[str, Any] | None) -> AuditBuild | None:
    """The fields of a build record (an ``IndexBuildRecord`` dump) that
    name the build; ``None`` without one that names its run."""
    if not isinstance(record, Mapping) or not isinstance(
        record.get("pipeline_id"), str
    ):
        return None
    base = record.get("full_build")
    try:
        return AuditBuild(
            pipeline_id=record["pipeline_id"],
            mode=str(record.get("mode") or ""),
            built_at=record.get("built_at"),
            package_version=record.get("package_version"),
            extraction_fingerprint=record.get("extraction_fingerprint"),
            full_build_pipeline_id=(
                base.get("pipeline_id") if isinstance(base, Mapping) else None
            ),
        )
    except ValidationError:
        # A record an earlier version wrote otherwise: its run's id stands.
        return AuditBuild(pipeline_id=record["pipeline_id"], mode="")


def _bounded(ids: Iterable[Any]) -> tuple[list[str], bool]:
    """The distinct string ids of ``ids`` in order, at most
    ``MAX_RECORDED_IDS``, and whether more were left out."""
    kept = list(dict.fromkeys(str(i) for i in ids if i))
    return kept[:MAX_RECORDED_IDS], len(kept) > MAX_RECORDED_IDS


def evidence_of(output: Any) -> AuditEvidence | None:
    """The ids of what ``output`` rests on: an answer's sources and the
    sections it cites, a search's results."""
    if output is None:
        return None
    sources = getattr(output, "sources", None)
    if sources is None:
        # A search: its ranked results.
        results = list(getattr(output.search_results, "results", None) or [])
        source_ids, cut = _bounded(r.source for r in results)
        return AuditEvidence(sources=len(results), source_ids=source_ids, truncated=cut)
    cited = [c for c in getattr(output, "citations", None) or [] if c.cited]
    source_ids, cut = _bounded(s.get("source") for s in sources)
    cited_sources, cut_sources = _bounded(c.source_id for c in cited)
    units, cut_units = _bounded(u for c in cited for u in c.text_unit_ids)
    documents, cut_documents = _bounded(
        d.document_id for c in cited for d in c.documents
    )
    return AuditEvidence(
        sources=len(sources),
        source_ids=source_ids,
        cited_source_ids=cited_sources,
        cited_text_unit_ids=units,
        cited_document_ids=documents,
        truncated=cut or cut_sources or cut_units or cut_documents,
    )
