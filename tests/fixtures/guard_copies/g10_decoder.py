# guard: G10
# as: domain/retrieval/graph_query.py
# expect: copy
import json


def _mut(raw):
    return json.JSONDecoder().decode(raw)
