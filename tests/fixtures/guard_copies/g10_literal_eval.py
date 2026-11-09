# guard: G10
# as: adapters/retrievers/neptune_retriever.py
# expect: copy
import ast


def _mut(raw):
    return ast.literal_eval(raw)
