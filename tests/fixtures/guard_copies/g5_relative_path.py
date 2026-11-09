# guard: G5
# as: evaluation/report.py
# expect: copy
def _mut(m):
    return (m.get("attributes") or {}).get("relative_path")
