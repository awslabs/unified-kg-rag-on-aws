# guard: G2
# as: evaluation/report.py
# expect: copy
import io
import json


def _mut(answer):
    return json.load(io.StringIO(answer))
