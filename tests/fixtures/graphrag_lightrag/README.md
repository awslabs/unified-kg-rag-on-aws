# GraphRAG and LightRAG fidelity fixtures

`upstream.json` holds the answers of the original MS GraphRAG and LightRAG
code on synthetic inputs, for `tests/unit/test_graphrag_lightrag_fidelity.py`.
Only the originals' answers are stored: the inputs come from
`tests/fixtures/graphrag_lightrag_fidelity.py` (plain Python, SHA-256
draws) on both sides.

The originals were run from their source at the commits pinned in
`.github/workflows/fidelity.yml` (microsoft/graphrag `6d02c23`, package
3.1.0, MIT; HKUDS/LightRAG `4e1f952`, version string 1.5.4, MIT), each
function executed on its own from the checkout, never installed:

- `polling`: LightRAG `utils.pick_by_weighted_polling` on 60 sets of 1-12
  chunk lineages of 0-8 chunks each, at 1-8 chunks for the first item and 1
  or 2 for the last;
- `sort_context`: MS GraphRAG `graph_context/sort_context.sort_context` on
  30 communities of 3-14 entities, given every relationship and no token
  cap, read back as the order its context lists the entities and
  relationships in;
- `report_edges`: MS GraphRAG
  `graph_context/context_builder._prepare_reports_at_level` on the same
  communities, the relationships it hands `sort_context`.

`test_the_recorded_answers_are_the_originals` checks them against
checkouts (`GRAPHRAG_SRC=<checkout> LIGHTRAG_SRC=<checkout> uv run --extra
eval pytest tests/unit/test_graphrag_lightrag_fidelity.py`); it is skipped
otherwise. To record them again after a pin moves, run
`python -m tests.fixtures.graphrag_lightrag_fidelity` with the same
variables.
