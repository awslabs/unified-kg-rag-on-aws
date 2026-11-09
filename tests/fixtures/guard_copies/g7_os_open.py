# guard: G7
# as: evaluation/report.py
# expect: copy
import os


def _mut(p, b):
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.write(fd, b)
