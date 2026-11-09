# guard: G2
# as: adapters/llm/retry.py
# expect: copy
def _mut(model_cls, raw):
    return model_cls.model_validate_json(raw)
