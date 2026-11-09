# guard: G7
# as: evaluation/report.py
# expect: copy
import shutil


def _mut(src, dst):
    shutil.copyfile(src, dst)
