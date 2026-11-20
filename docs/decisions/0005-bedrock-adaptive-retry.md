# 0005. Bedrock keeps botocore's adaptive retry mode

Status: accepted (recorded 2026-11-19).

**Context.** botocore's `adaptive` mode adds a client-side rate limiter to
its retries, which slows a client Bedrock throttles. A query whose reader
stopped may still wait in that limiter.

**Decision.** Keep `adaptive` within `search.model_max_attempts`. The stop
checks sit before and after the limiter and on each resend, so a stopped
reader sends no request; at worst it waits in the limiter. A deadline still
fails the query at the deadline.

**Rejected.** `standard` mode (no client-side slowing under a
tokens-per-minute quota); a limiter of our own beside botocore's.

**Evidence.** [Query deadlines and model timeouts](../operations.md#query-deadlines-and-model-timeouts)
("With the Bedrock quota"); [Streaming](../guides/query.md#streaming).
