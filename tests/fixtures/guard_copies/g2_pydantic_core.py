# guard: G2
# as: adapters/ingestion/graph_extractor.py
# expect: copy
from pydantic_core import from_json


def _mut(answer):
    return from_json(answer, allow_partial=True)
