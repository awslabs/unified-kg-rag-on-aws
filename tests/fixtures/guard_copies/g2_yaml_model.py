# guard: G2
# as: application/prompts/tuner.py
# expect: copy
import yaml


def _mut(answer):
    return yaml.safe_load(answer)
