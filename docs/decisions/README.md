# Decision records

Each record is one maintainer decision that a review or a contributor
might otherwise reopen: its context, the decision, the options rejected and
the evidence it rests on. A record changes only by a new record that
supersedes it (say which in its Status line); the code and the guides keep
describing the current behaviour.

| # | Decision | Status |
|---|---|---|
| [0001](./0001-simple-strategy-kept.md) | The `simple` strategy stays, fixed, until a re-measure decides | Accepted |
| [0002](./0002-visualization-export-full-builds.md) | Visualization data is exported after full builds only, and built on demand | Accepted |
| [0003](./0003-guards-are-tripwires.md) | The single-owner guards are tripwires, not coverage | Accepted |
| [0004](./0004-limits-count-calls-metrics-count-attempts.md) | Query limits count logical calls; usage, metrics and run budgets count attempts | Accepted |
| [0005](./0005-bedrock-adaptive-retry.md) | Bedrock keeps botocore's adaptive retry mode | Accepted |
| [0006](./0006-no-bedrock-batch-inference.md) | No Bedrock batch inference | Accepted |
| [0007](./0007-cancelled-caller-is-a-stopped-reader.md) | A cancelled or disconnected caller is a stopped reader, not a failure | Accepted |
| [0008](./0008-npm-audit-accepted-advisories.md) | npm audit gates with an expiring list of accepted advisories | Accepted |
