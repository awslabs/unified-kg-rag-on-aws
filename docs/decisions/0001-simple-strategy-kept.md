# 0001. The `simple` strategy stays until a re-measure decides

Status: accepted (recorded 2026-11-19).

**Context.** `simple` searches every index the namespace holds by vector
and lexical search, without the graph. On 200 MuSiQue questions it lost to
`naive` (`answer_contains` 0.528 against 0.605, 2026-11-08): one search
over every index kept `top_k` × `retrieval_multiplier` hits in all, two or
three per index, before fusion and the reranker saw them.

**Decision.** Keep `simple`, fixed: each index keeps its own
`top_k` × `retrieval_multiplier` candidates, the union is fused and
reranked, and the summary tree is read when it is built. Re-measure it on
the MuSiQue set, the sensemaking set and a claim-lookup corpus; deprecate it
(one release with a warning, then removal) only if it wins no question type.

**Rejected.** Removing it on the 2026-11-08 result: that run measured the
unfixed strategy, and the lookups it may serve best (claims, identifiers)
were never measured.

**Evidence.** [Sensemaking strategies](../concepts/defaults-and-evidence.md#sensemaking-strategies);
[Design §6.1](../design/retrieval-graphrag-lightrag.md#61-graphrag-methodology-adapterssearch_strategies).
