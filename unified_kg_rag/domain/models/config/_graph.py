# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``graph`` section: centrality, reports, community detection, visualization."""

from pathlib import Path

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)


class CentralityConfig(BaseModel):
    calculate_degree: bool = Field(
        default=True,
        description="Calculate degree centrality to measure node connectivity",
    )
    calculate_betweenness: bool = Field(
        default=True,
        description=(
            "Compute betweenness centrality (bridge nodes) for the "
            "visualization; ingestion itself computes no centrality. On graphs"
            " larger than betweenness_auto_sample_threshold nodes it is "
            "sampled instead of computed exactly."
        ),
    )
    calculate_pagerank: bool = Field(
        default=True,
        description="Calculate PageRank centrality to rank node importance",
    )
    pagerank_alpha: float = Field(
        default=0.85, ge=0.0, le=1.0, description="PageRank damping factor"
    )
    pagerank_max_iter: int = Field(
        default=100,
        ge=1,
        description="Maximum iterations for PageRank convergence",
    )
    betweenness_k: int | None = Field(
        default=None,
        ge=1,
        description="Sample size for betweenness calculation (None for all nodes)",
    )
    betweenness_auto_sample_threshold: int = Field(
        default=2000,
        ge=1,
        description="When betweenness_k is None (exact) AND the graph has more "
        "nodes than this, automatically switch to sampled betweenness (using "
        "this value as the pivot-sample size) instead of exact all-pairs "
        "shortest paths. Exact betweenness is O(V*E) and stalls on large real "
        "graphs; sampling keeps the analysis phase bounded. Raise this to force "
        "exact computation on larger graphs.",
    )
    betweenness_seed: int = Field(
        default=42,
        description="Random seed for sampled betweenness (only used when "
        "betweenness_k is set). Fixed for reproducibility — sampled betweenness "
        "picks random pivots, so without a seed centrality is non-deterministic.",
    )


class ReportGenerationConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description=(
            "Enable automatic community report generation. Off: global and "
            "drift search, which answer from the reports, are refused (local "
            "reads graphrag without them; a WARNING at load says so). A run "
            "turning it off for a namespace that holds reports is refused "
            "while the namespace holds other source scopes' documents."
        ),
    )
    max_entities_per_report: int = Field(
        default=50, ge=1, description="Maximum entities per community report"
    )
    max_report_context_tokens: int = Field(
        default=4000,
        ge=1,
        description="Token budget for the entity/relationship (and, with "
        "claim extraction on, claim) context packed into a community report "
        "prompt. As MS GraphRAG's sort_context, relationships are packed by "
        "combined endpoint degree (most connected first), each with its two "
        "entities and their claims, then the entities with no relationship "
        "in the community, up to this budget. max_entities_per_report remains "
        "an additional upper bound on entity count. MS GraphRAG's default is "
        "8000 with max_cluster_size 10: raise this (more report input tokens "
        "per call) when reports of large communities keep few relationships.",
    )
    content_length: str = Field(
        default="medium",
        pattern="^(short|medium|long)$",
        description="Report length, as the number of findings the report asks "
        "for: short (5-7), medium (7-10), long (10-15)",
    )
    enable_sub_community_rollup: bool = Field(
        default=True,
        description="When a parent community's entity/relationship context is "
        "cut at max_report_context_tokens, add the summaries of its child "
        "communities' reports (largest first, within a budget of "
        "max_report_context_tokens of their own, so the prompt holds at most "
        "about twice it) after the context kept, so what was cut is still "
        "summarized. MS GraphRAG substitutes the child reports for raw "
        "context instead. Reports are then generated level by level, the "
        "finest first (a level waits for the one below); false generates "
        "every level at once and only cuts the context.",
    )


class CommunityDetectionConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description=(
            "Run Leiden community detection + LLM community-report generation "
            "during indexing, when indexing.methodologies selects graphrag "
            "(its strategies need the reports). false leaves graphrag out of "
            "what ingestion builds, as indexing.methodologies: [lightrag] "
            "does: no Leiden pass and no report calls."
        ),
    )
    resolution: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description=(
            "Leiden modularity resolution: higher gives more, smaller "
            "communities (and more report calls); lower gives fewer, larger "
            "ones."
        ),
    )
    random_state: int = Field(
        default=42,
        ge=1,
        description="Leiden random seed (a positive integer; the native Leiden "
        "rejects 0). The same graph and seed give the same communities.",
    )
    max_levels: int = Field(
        default=5, ge=1, description=("Maximum depth of the " "community hierarchy.")
    )
    trials: int = Field(
        default=3,
        ge=1,
        description="Number of independent Leiden runs to find best partition "
        "(MS GraphRAG runs 1).",
    )
    extra_forced_iterations: int = Field(
        default=2,
        ge=0,
        description="Additional optimization iterations after convergence "
        "(MS GraphRAG runs 0).",
    )
    max_cluster_size: int = Field(
        default=20,
        ge=2,
        description="Maximum members per community before it is split: "
        "entities at level 0, child communities above. A larger Leiden "
        "community is re-partitioned "
        "on its own subgraph, recursively, until every piece fits or Leiden "
        "finds no finer split, and stays the parent of its pieces (as in MS "
        "GraphRAG's hierarchical Leiden). Without the "
        "bound, modularity's resolution limit grows communities with the "
        "corpus beyond what one report's max_report_context_tokens can "
        "describe (or a parent's roll-up can summarize). MS GraphRAG's default "
        "is 10 with a larger report budget; 20 typically keeps a leaf's "
        "entities and relationships, or a parent's child summaries, within the "
        "default 4000-token report context.",
    )
    min_community_size: int = Field(
        default=3,
        ge=1,
        description=(
            "Minimum entities per level-0 community: a smaller one merges into"
            " the neighbour it shares the most edge weight with, unless that "
            "would exceed max_cluster_size. A connected component of fewer "
            "entities (an isolated entity, a pair) has no neighbour and gets "
            "no community or report. 1 = cluster every component."
        ),
    )
    report_generation: ReportGenerationConfig = Field(
        default_factory=ReportGenerationConfig,
        description="Community report generation configuration",
    )

    @model_validator(mode="after")
    def _check_cluster_size_bounds(self) -> "CommunityDetectionConfig":
        if self.min_community_size > self.max_cluster_size:
            raise ValueError(
                f"min_community_size ({self.min_community_size}) must not exceed "
                f"max_cluster_size ({self.max_cluster_size})"
            )
        return self


class InteractiveGraphSettings(BaseModel):
    """The interactive pyvis graph (``interactive_graph.html``)."""

    height: str = Field(default="900px", description="Canvas height (CSS).")
    width: str = Field(default="100%", description="Canvas width (CSS).")
    physics_enabled: bool = Field(
        default=False, description="Run the pyvis physics simulation."
    )
    cdn_resources: str = Field(
        default="in_line", description="pyvis cdn_resources: in_line, local or remote."
    )
    background_color: str = Field(default="#f8fafc", description="Canvas colour.")
    font_family: str = Field(default="Arial", description="Label font.")
    tooltip_delay: int = Field(default=200, ge=0, description="Tooltip delay (ms).")
    max_nodes: int | None = Field(
        default=2000,
        ge=0,
        description="Keep the top-N nodes by degree; 0 or null disables the cap "
        "(a browser cannot lay out tens of thousands of nodes).",
    )


class StaticChartSettings(BaseModel):
    """The static charts (dependency-free HTML with inline SVG)."""

    figure_width: int = Field(default=900, ge=100, description="Chart width (px).")
    figure_height: int = Field(default=600, ge=100, description="Chart height (px).")
    font: str = Field(default="Arial, sans-serif", description="Chart font.")
    background_color: str = Field(default="#fdfdfd", description="Chart background.")
    log_scale_x: bool = Field(
        default=False, description="Logarithmic x axis on distributions."
    )


class VisualizationConfig(BaseModel):
    centrality: CentralityConfig = Field(
        default_factory=CentralityConfig,
        description="Node centrality metrics the static renderer's centrality "
        "chart plots (computed at render time; no ingestion stage reads them). "
        "The interactive graph sizes nodes by degree, not by these.",
    )
    export_data: bool = Field(
        default=True,
        description="Write visualization_data.json (the index namespace's "
        "graph nodes, edges and community hierarchy, read from its stores) at "
        "the end of a full build, so run-visualization can render it, once "
        "the run gave up the namespace's lease. One pass over the stored "
        "items: no embedding or model call, no layout or centrality "
        "(run-visualization computes them). An incremental run writes none "
        "(run-visualization and the console build it from the stores); a "
        "run that wrote nothing keeps an export of the build the stores "
        "hold.",
    )
    outputs_directory: str | Path | None = Field(
        default=None,
        description="Directory ingestion writes visualization_data.json to. "
        "When unset, '<cache directory>/<pipeline_id>/visualization' "
        "(--cache-directory), so the file is synced to S3 with the cache, or "
        "'outputs/visualization' without a cache directory.",
    )
    interactive: InteractiveGraphSettings = Field(
        default_factory=InteractiveGraphSettings,
        description="Settings of the interactive pyvis graph.",
    )
    static: StaticChartSettings = Field(
        default_factory=StaticChartSettings,
        description="Settings of the static HTML/SVG charts.",
    )


class GraphConfig(BaseModel):
    community_detection: CommunityDetectionConfig = Field(
        default_factory=CommunityDetectionConfig,
        description="Community detection configuration",
    )
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
        description="Visualization configuration",
    )
