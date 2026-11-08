# Visualization (`run-visualization`)

A **standalone** renderer that draws from an exported visualization-data
JSON; it does **not** re-run ingestion or touch AWS. Every flag with its
default: [CLI Reference](../reference/cli.md#run-visualization).

## Examples

```bash
# Render the newest export of the default namespace (run-ingestion prints its path)
run-visualization --suffix default --output-directory ./viz --config-path config.yaml

# Render all renderers of one file
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
`@register_renderer` ([Design Doc §15](./extend.md#15-extension-guide)).

## Input and output

- **The input.** `run-ingestion` writes `visualization_data.json` at the end
  of every run that indexed, read from the index namespace's stores, and
  prints its path (`visualization_data` in its JSON result): the whole graph
  as queries see it, also after an incremental run and in a namespace other
  corpora share. The entities, claims and community reports are read a page
  at a time from the vector store, each entity's communities from the graph
  store, and the relationships from the vector store (the graph store
  without a relationships index); a namespace without community detection
  (`indexing.methodologies` without `graphrag`) has an empty hierarchy. A
  failed export is logged and never fails the run. The file names its
  namespace first (`"namespace"`), then the build the stores held
  (`"built_at"`, the build record's). The export runs once the run gave up
  the namespace's lease (it only reads), and a run that wrote, deleted and
  re-annotated nothing keeps an export of the build the stores still hold
  (an export of an earlier build is written again).
- **Which file.** `--data-path`, else the newest `visualization_data.json`
  under `graph.visualization.outputs_directory`, `outputs/visualization` and
  `--cache-directory` (default `cache`), of `--suffix`'s namespace when given
  (exit 2 when there is none).
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
