# guard: G7
# as: evaluation/report.py
# expect: clean
import dataclasses


def _f(cfg, other):
    return dataclasses.replace(cfg)
