# guard: G7
# as: evaluation/report.py
# expect: copy
import io


def _mut(p, t):
    with io.open(p, "w", encoding="utf-8") as f:
        f.write(t)
