# 0006. No Bedrock batch inference

Status: accepted (recorded 2026-11-19).

**Context.** Bedrock batch inference runs a set of prompts as one
asynchronous job at a lower price, which could cut the cost of extraction
and report generation.

**Decision.** Not used. It supports a limited set of models, and it would
add a second execution path to the pipeline (staging inputs, submitting and
polling jobs, resuming a half-done job) beside the one every stage, cache
and budget uses now.

**Rejected.** A batch mode for the ingestion stages.

**Evidence.** None measured: the decision rests on the model coverage and
the pipeline cost above. Revisit it with a measured saving on a real build.
