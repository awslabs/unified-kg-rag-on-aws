# guard: G2
# as: adapters/ingestion/claim_extractor.py
# expect: copy
import json
import re

_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _mut(llm_answer: str) -> dict:
    match = _OBJECT.search(llm_answer)
    if match is None:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
