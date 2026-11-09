# guard: G5
# as: evaluation/report.py
# expect: copy
import operator


def _mut(m):
    return operator.itemgetter("file_name")(m)
