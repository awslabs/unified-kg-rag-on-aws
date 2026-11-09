# guard: G4
# as: shared/utils/common.py
# expect: clean
import re as _re

_ZW = _re.compile("[\u200b-\u200f\ufeff]")
