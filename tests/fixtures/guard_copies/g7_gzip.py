# guard: G7
# as: evaluation/report.py
# expect: copy
import gzip


def _mut(p, t):
    with gzip.open(p, "wt") as f:
        f.write(t)
