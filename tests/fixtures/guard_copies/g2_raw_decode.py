# guard: G2
# as: adapters/search_strategies/drift_search.py
# expect: copy
import json


def _mut(text):
    obj, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
    return obj
