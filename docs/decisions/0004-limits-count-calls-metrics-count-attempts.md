# 0004. Query limits count logical calls; usage, metrics and run budgets count attempts

Status: accepted (recorded 2026-11-19).

**Context.** A query's model calls may be retried (a stream cut before its
answer text) and resent by the client (botocore, the OpenAI client). A
limit, a usage record and a budget can each count either the calls the
query plans or the requests sent.

**Decision.** `search.query_max_model_calls` and
`search.query_max_input_tokens` count the calls the query plans: a retry
reuses the room its failed attempt held. The query's usage
(`metadata.usage`), the metrics and an ingestion run's budgets
(`processing.max_run_*`) count every attempt sent, as the bill does.

**Rejected.** Limits on attempts (a throttled minute would refuse answers
whose plan fits); usage and budgets on logical calls (retries would vanish
from cost and throttling).

**Evidence.** [Model-call and input-token limits](../guides/query.md#model-call-and-input-token-limits);
the metrics table of the [operator runbook](../operations.md#alarms).
