# guard: G1
# as: adapters/search_strategies/drift_search.py
# expect: clean
import logging

from ...domain.retrieval.degradation import DegradationLedger, DegradationStep

logger = logging.getLogger(__name__)


def _mut(ledger: DegradationLedger, exc: Exception) -> None:
    # A follow-up that fails degrades the answer gracefully.
    logger.warning("DRIFT follow-up failed; answering without it: %s", exc)
    ledger.record(DegradationStep.DRIFT_ACTION, "follow_up_failed", exc)
