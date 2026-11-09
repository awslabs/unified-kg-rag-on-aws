# guard: G1
# as: application/presentation/answer_view.py
# expect: copy
from pydantic import Field as _F

_MUT = _F(default=None, alias="citations_unresolved")
