# 0007. A cancelled or disconnected caller is a stopped reader

Status: accepted (recorded 2026-11-19).

**Context.** A caller can leave a query before its answer: a console client
disconnects, an `asyncio.timeout` wraps `astream`, a task is cancelled,
`Ctrl-C` arrives while a stream waits.

**Decision.** The query stops for that reader: its model responses are
closed, no request is sent after the stop (resends included), nothing is
written to memory, and the query metrics record its latency and usage up
to there as no failure. A call concurrent queries share goes on until each
of their readers stopped. A deadline is not a stop: it still fails the
query.

**Rejected.** Recording a stop as a failure (an alarm per client
disconnect); letting the query's calls run to the end (spend for an answer
nobody reads).

**Evidence.** [Streaming](../guides/query.md#streaming).
