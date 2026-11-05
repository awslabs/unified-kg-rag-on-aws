# Visualization (`run-visualization`)

A **standalone** renderer that draws from an exported visualization-data
JSON; it does **not** re-run ingestion or touch AWS. Every flag with its
default: [CLI Reference](../reference/cli.md#run-visualization).

## Examples

```bash
# Render all renderers
run-visualization --data-path visualization_data.json --output-directory ./viz --config-path config.yaml

# Only the interactive renderer
run-visualization --data-path visualization_data.json --renderers interactive --config-path config.yaml
```

The two renderers are **`interactive`** (pyvis; it needs the `viz` extra,
`uv sync --extra viz`, [Install](../getting-started/aws.md#install)) and
**`static`** (dependency-free HTML with inline SVG; it runs on a core
install), configured under `graph.visualization` (`interactive.*`,
`static.*`, `centrality.*`; every key: [Configuration
Reference](../reference/configuration.md)).
Ingestion itself renders nothing. Another renderer plugs in through
`@register_renderer` ([Design Doc §15](../design.md#15-extension-guide)).

## Input and output

- **The input.** The community-detection stage of `run-ingestion` writes
  `visualization_data.json` on every run whose graph covers the whole corpus:
  a full build, a reset, or an incremental run in which every document is new
  or changed. An incremental run that skips unchanged documents writes
  nothing (it logs an INFO line) and leaves an earlier file in place; run one
  full build (`indexing.reset: true`) to visualize the whole graph.
- **Where.** `graph.visualization.outputs_directory`, by default
  `<cache directory>/<pipeline_id>/visualization/`. S3 cache sync uploads the
  `.json` only; re-render the HTML locally.
- **What.** Nodes (colored by their level-0 `community_id`), edges and the
  community hierarchy, without vectors. Writing it calls no model;
  `run-visualization` computes a seeded spring layout from the graph
  topology when it renders, and the `centrality` metrics only when `static`
  is among the renderers (the interactive page does not read them; on a
  5,000-node graph an `interactive`-only run took 22 s instead of 71 s). A
  metric that is off or failed to compute is left out of the centrality
  chart, not drawn as 0.
- **Files.** `interactive` writes `interactive_graph.html` and, with a
  community hierarchy, `community_hierarchy.html`. `static` writes
  `degree_distribution.html`, `centrality_comparison.html` (the top 25 nodes
  by degree, with their degree, betweenness and PageRank) and
  `community_size_distribution.html` (level 0): each is one page with an
  SVG chart and a tooltip on every bar, which needs no script or network.
- **Size.** The interactive graph keeps the top `interactive.max_nodes` nodes
  by degree (default 2000) and lays out only those.
- Edge width and opacity are log-scaled to the graph's own weight range.
- Only files a renderer wrote are reported (an earlier file of the same name
  is removed first); when nothing is rendered (an empty graph), or a
  renderer fails while others write their files, the command exits `1`
  (the JSON output names it in `failed_renderers`).
