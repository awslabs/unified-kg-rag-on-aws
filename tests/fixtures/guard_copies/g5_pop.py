# guard: G5
# as: evaluation/report.py
# expect: copy
def _mut(m):
    return m.pop("file_path", None)
