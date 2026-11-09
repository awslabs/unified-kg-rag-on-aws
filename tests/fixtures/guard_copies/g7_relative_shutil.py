# guard: G7
# as: evaluation/report.py
# expect: copy
import shutil as _sh


def _mut(a, b):
    _sh.move(a, b)
