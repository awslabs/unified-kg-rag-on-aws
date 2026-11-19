# 0002. Visualization data is exported after full builds only

Status: accepted (recorded 2026-11-19).

**Context.** `visualization_data.json` holds the namespace's whole graph and
community hierarchy, read from the stores. An incremental run's own graph is
only its delta, so an export there means a pass over every stored item for
each delta.

**Decision.** A full build writes the export at its end
(`graph.visualization.export_data`); an incremental run writes none.
`run-visualization` and the console's Graph page build the data from the
stores, with the same writer, when no export names the build the stores
hold.

**Rejected.** Exporting after every incremental run (a full store pass per
delta, whether anyone renders it); exporting the delta alone (a partial
graph under the namespace's name).

**Evidence.** [Design §10](../design/surfaces.md#10-visualization--analytics);
the `export_data` entry of the [configuration reference](../reference/configuration.md).
