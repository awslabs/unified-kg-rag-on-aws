# guard: G7
# as: evaluation/report.py
# expect: copy
import tempfile
from pathlib import Path


def _mut(p, t):
    with tempfile.NamedTemporaryFile("w", dir=p.parent, delete=False) as f:
        f.write(t)
    Path(f.name).rename(p)
