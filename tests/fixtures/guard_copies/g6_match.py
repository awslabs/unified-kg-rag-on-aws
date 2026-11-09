# guard: G6
# as: evaluation/report.py
# expect: copy
def _mut(v):
    match v:
        case str():
            return [v]
    return list(v)
