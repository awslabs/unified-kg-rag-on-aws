# guard: G5
# as: evaluation/report.py
# expect: copy
from unified_kg_rag.domain.ingestion.document_identity import RELATIVE_PATH_KEY as _K


def _mut(m):
    return m["attributes"][_K]
