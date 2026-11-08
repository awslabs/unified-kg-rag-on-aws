# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a query answered without: its degraded steps.

A query step that fails without failing the query (a store read a strategy
absorbs, a map call of global search, a DRIFT follow-up, the query
translation under ``processing.ignore_errors``) is one :class:`Degradation`
in the query's :class:`DegradationLedger`. The answer reports the ledger as
``RAGOutput.metadata["degraded"]`` (each entry's ``step``, ``reason`` and
``error_type``) and ``metadata["degraded_error"]``, the error type of the
first entry that names one.

A failed read of a lineage only the reported sources use (the findings
lineage of a community report when the strategy's answers do not cite,
``search.citations``) degrades no answer: it is recorded apart
(:meth:`DegradationLedger.record_unresolved`) and reported as
``metadata["citations_unresolved"]``.
"""

from __future__ import annotations

import threading
from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "DEGRADED_ERROR_KEY",
    "DEGRADED_KEY",
    "UNRESOLVED_KEY",
    "Degradation",
    "DegradationLedger",
    "DegradationStep",
]

# RAGOutput/SearchResult metadata keys.
DEGRADED_KEY = "degraded"
DEGRADED_ERROR_KEY = "degraded_error"
UNRESOLVED_KEY = "citations_unresolved"


class DegradationStep:
    """The ``step`` names of the framework's own entries (a third-party
    strategy may record its own)."""

    # A store read a strategy answered without (``_safe_aretrieve``).
    RETRIEVAL = "retrieval"
    # The query processing calls (query_processing) and the whole step.
    TRANSLATION = "translation"
    ENTITY_EXTRACTION = "entity_extraction"
    KEYWORD_EXTRACTION = "keyword_extraction"
    QUERY_PROCESSING = "query_processing"
    # auto's router, the follow-up rewrite, the context budgeting.
    ROUTING = "routing"
    FOLLOW_UP_REWRITE = "follow_up_rewrite"
    CONTEXT_OPTIMIZATION = "context_optimization"
    # Global search's community rating and map phase.
    COMMUNITY_RATING = "community_rating"
    GLOBAL_MAP = "global_map"
    # DRIFT's query expansion, primer and follow-up actions.
    DRIFT_EXPANSION = "drift_expansion"
    DRIFT_PRIMER = "drift_primer"
    DRIFT_ACTION = "drift_action"
    # Iterative search's controller.
    ITERATIVE_CONTROLLER = "iterative_controller"
    # The findings lineage of community reports (cited answers only).
    FINDINGS_LINEAGE = "findings_lineage"
    # The structured strategy's graph query, answered by its fallback (the
    # reason: planner_error, execution_error, timeout, compile_error,
    # unlinked_mention, unknown_value).
    STRUCTURED_QUERY = "structured_query"


class Degradation(BaseModel):
    """One failed step a query answered without."""

    step: str = Field(description="What failed (a DegradationStep name)")
    reason: str = Field(
        description="How: the failed read's label, or what the model returned"
    )
    error_type: str | None = Field(
        default=None,
        description="The class name of the error; None for an output that "
        "could not be read",
    )


class DegradationLedger:
    """The degraded steps of one query (or of one search within it).

    A ledger opened inside another (``parent``) passes each entry on to it:
    a strategy's search reports its own entries, the query all of them.
    Safe to record into from the query's tasks and worker threads.
    """

    def __init__(self, parent: DegradationLedger | None = None) -> None:
        self._parent = parent
        self._lock = threading.Lock()
        self._entries: list[Degradation] = []
        self._unresolved: list[Degradation] = []

    def record(
        self,
        step: str,
        reason: str,
        error: BaseException | str | None = None,
    ) -> Degradation:
        """Record that ``step`` failed (``error``: the exception, or its
        type name)."""
        entry = Degradation(step=step, reason=reason, error_type=_type_name(error))
        self._add(entry, unresolved=False)
        return entry

    def record_unresolved(
        self, step: str, reason: str, error: BaseException | str | None = None
    ) -> Degradation:
        """Record a failed read of a lineage the answer does not cite with:
        its sources are less precise, the answer is not degraded."""
        entry = Degradation(step=step, reason=reason, error_type=_type_name(error))
        self._add(entry, unresolved=True)
        return entry

    def _add(self, entry: Degradation, *, unresolved: bool) -> None:
        with self._lock:
            (self._unresolved if unresolved else self._entries).append(entry)
        if self._parent is not None:
            self._parent._add(entry, unresolved=unresolved)

    @property
    def entries(self) -> list[Degradation]:
        with self._lock:
            return list(self._entries)

    @property
    def unresolved(self) -> list[Degradation]:
        with self._lock:
            return list(self._unresolved)

    @property
    def first_error_type(self) -> str | None:
        """The error type of the first entry that names one."""
        return next((e.error_type for e in self.entries if e.error_type), None)

    def metadata(self) -> dict[str, Any]:
        """The ledger's keys of a result's metadata: ``degraded`` and
        ``degraded_error`` when a step was degraded, ``citations_unresolved``
        when a lineage read failed; empty otherwise."""
        out: dict[str, Any] = {}
        if entries := self.entries:
            out[DEGRADED_KEY] = [e.model_dump() for e in entries]
            if first := self.first_error_type:
                out[DEGRADED_ERROR_KEY] = first
        if self.unresolved:
            out[UNRESOLVED_KEY] = True
        return out


def _type_name(error: BaseException | str | None) -> str | None:
    if error is None or isinstance(error, str):
        return error
    return type(error).__name__
