# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from typing import Any

import networkx as nx
import numpy as np
from bokeh.models import (  # type: ignore[attr-defined]
    ColumnDataSource,
    HoverTool,
    NumeralTickFormatter,
    Title,
)
from bokeh.palettes import Blues8
from bokeh.plotting import figure, output_file, save
from bokeh.transform import dodge

from aws_graphrag.domain.ingestion.graph_analyzer import CentralityMetrics
from aws_graphrag.domain.models import Community
from aws_graphrag.shared import get_logger

logger = get_logger(__name__)


class StaticRenderer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.figure_width = config.get("figure_width", 900)
        self.figure_height = config.get("figure_height", 600)
        self.font = config.get("font", "Arial, sans-serif")
        self.background_color = config.get("background_color", "#fdfdfd")
        self.color_palette = config.get("color_palette", Blues8)
        self.log_scale_x = config.get("log_scale_x", False)

    def plot_degree_distribution(self, graph: nx.Graph, outputs_path: str) -> None:
        if not graph or graph.number_of_nodes() == 0:
            logger.warning("Cannot plot degree distribution for an empty graph.")
            return

        logger.info("Creating degree distribution plot...")
        degrees = [d for _, d in graph.degree()]
        if not degrees:
            logger.warning("Graph has no degrees to plot.")
            return

        unique_degrees = set(degrees)
        num_bins = max(1, min(50, len(unique_degrees)))

        if self.log_scale_x and min(degrees) > 0:
            log_min = np.log10(min(degrees))
            log_max = np.log10(max(degrees))
            bins = np.logspace(log_min, log_max, num_bins + 1)
            hist, edges = np.histogram(degrees, bins=bins)
        else:
            hist, edges = np.histogram(degrees, bins=num_bins)

        bar_width = (edges[1] - edges[0]) if len(edges) > 1 else 1.0
        spacing = bar_width * 0.1
        adjusted_width = bar_width - spacing

        centers = (edges[:-1] + edges[1:]) / 2
        left_positions = centers - adjusted_width / 2
        right_positions = centers + adjusted_width / 2

        source = ColumnDataSource(
            data={"top": hist, "left": left_positions, "right": right_positions}
        )

        figure_kwargs = {}
        if self.log_scale_x:
            figure_kwargs["x_axis_type"] = "log"

        p = self._create_figure(
            "Degree Distribution", "Degree", "Number of Nodes", **figure_kwargs
        )
        p.quad(
            top="top",
            bottom=0,
            left="left",
            right="right",
            source=source,
            fill_color="#3b82f6",
            line_color=None,
            alpha=0.85,
        )

        p.add_tools(
            HoverTool(
                tooltips=[
                    ("Degree Range", "@left{0.0} - @right{0.0}"),
                    ("Count", "@top{0,0}"),
                ]
            )
        )

        self._save_plot(p, outputs_path, "Degree Distribution")

    def plot_centrality_comparison(
        self, centrality_data: dict[str, CentralityMetrics], output_path: str
    ) -> None:
        if not centrality_data:
            logger.warning("No centrality data to plot.")
            return

        logger.info("Creating centrality comparison plot...")
        sorted_nodes = sorted(
            centrality_data.values(), key=lambda x: x.degree or 0, reverse=True
        )
        top_nodes = [
            n
            for n in sorted_nodes
            if hasattr(n, "node_name") and n.node_name is not None
        ][:25]

        if not top_nodes:
            logger.warning("No nodes with names found for centrality plot.")
            return

        node_names = [n.node_name for n in top_nodes]
        metrics = ["degree", "betweenness", "pagerank"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        data: dict[str, Any] = {"nodes": node_names}
        for metric in metrics:
            data[metric] = [getattr(n, metric, 0) or 0 for n in top_nodes]

        source = ColumnDataSource(data=data)
        p = self._create_figure(
            "Centrality Comparison (Top 25 Nodes by Degree)",
            "Nodes",
            "Score",
            x_range=node_names,
            width=1200,
            height=700,
        )

        dodge_width = 0.2
        for i, (metric, color) in enumerate(zip(metrics, colors, strict=True)):
            p.vbar(
                x=dodge("nodes", i * dodge_width - dodge_width * 1.5, range=p.x_range),
                top=metric,
                width=dodge_width * 0.95,
                source=source,
                legend_label=metric.title(),
                color=color,
                alpha=0.8,
                line_color="white",
            )

        p.xaxis.major_label_orientation = 0.6
        p.xaxis.major_label_text_font_size = "9pt"
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        p.yaxis.formatter = NumeralTickFormatter(format="0.00")

        hover_tooltips = [("Node", "@nodes")] + [
            (m.title(), f"@{m}{{0.000}}") for m in metrics
        ]
        p.add_tools(HoverTool(tooltips=hover_tooltips, mode="vline"))

        self._save_plot(p, output_path, "Centrality Comparison")

    def plot_community_size_distribution(
        self, communities: list[Community], output_path: str
    ) -> None:
        if not communities:
            logger.warning("No community data to plot.")
            return

        logger.info("Creating community size distribution plot...")
        sizes = [
            c.size
            for c in communities
            if c.level == "0" and c.size is not None and c.size > 0
        ]
        if not sizes:
            logger.warning("No valid community sizes found for visualization.")
            return

        unique_sizes = set(sizes)
        bins = max(1, min(50, len(unique_sizes)))
        if self.log_scale_x and min(sizes) > 0:
            log_min = np.log10(min(sizes))
            log_max = np.log10(max(sizes))
            log_bins = np.logspace(log_min, log_max, bins + 1)
            hist, edges = np.histogram(sizes, bins=log_bins)
        else:
            hist, edges = np.histogram(sizes, bins=bins)

        bar_width = (edges[1] - edges[0]) if len(edges) > 1 else 1.0
        spacing = bar_width * 0.1
        adjusted_width = bar_width - spacing

        centers = (edges[:-1] + edges[1:]) / 2
        left_positions = centers - adjusted_width / 2
        right_positions = centers + adjusted_width / 2

        source = ColumnDataSource(
            data={"top": hist, "left": left_positions, "right": right_positions}
        )

        figure_kwargs = {}
        if self.log_scale_x:
            figure_kwargs["x_axis_type"] = "log"

        p = self._create_figure(
            "Community Size Distribution (Level 0)",
            "Community Size (Number of Nodes)",
            "Number of Communities",
            **figure_kwargs,
        )
        p.quad(
            top="top",
            bottom=0,
            left="left",
            right="right",
            source=source,
            fill_color="#3b82f6",
            line_color=None,
            alpha=0.85,
        )

        p.yaxis.formatter = NumeralTickFormatter(format="0.0")

        p.add_tools(
            HoverTool(
                tooltips=[
                    ("Size Range", "@left{0.0} - @right{0.0}"),
                    ("Count", "@top{0,0}"),
                ]
            )
        )

        self._save_plot(p, output_path, "Community Size Distribution")

    def _create_figure(
        self, title: str, x_label: str, y_label: str, **kwargs: Any
    ) -> figure:
        defaults = {
            "width": self.figure_width,
            "height": self.figure_height,
            "tools": "pan,wheel_zoom,box_zoom,reset,save",
            "background_fill_color": self.background_color,
            "border_fill_color": None,
        }
        defaults.update(kwargs)

        p = figure(**defaults)

        p.title = Title(
            text=title,
            align="center",
            text_font=self.font,
            text_font_size="16pt",
            text_color="#27272a",
        )

        p.xaxis.axis_label = x_label
        p.yaxis.axis_label = y_label
        p.xaxis.axis_label_text_font = self.font
        p.yaxis.axis_label_text_font = self.font
        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.xaxis.axis_label_text_color = "#3f3f46"
        p.yaxis.axis_label_text_color = "#3f3f46"

        p.xaxis.major_label_text_font = self.font
        p.yaxis.major_label_text_font = self.font
        p.xaxis.major_label_text_font_size = "10pt"
        p.yaxis.major_label_text_font_size = "10pt"
        p.yaxis.formatter = NumeralTickFormatter(format="0,0")

        p.grid.grid_line_color = "#e5e5e5"
        p.grid.grid_line_alpha = 0.5
        p.grid.minor_grid_line_color = None
        p.grid.grid_line_dash = [4, 4]

        p.xaxis.axis_line_color = "#a1a1aa"
        p.yaxis.axis_line_color = "#a1a1aa"

        if hasattr(p.y_range, "start"):
            p.y_range.start = 0
        if hasattr(p.x_range, "range_padding"):
            p.x_range.range_padding = 0
        p.outline_line_color = None

        return p

    @staticmethod
    def _save_plot(plot: figure, path: str, title: str) -> None:
        output_file(path, title=title)
        save(plot, title=title)
        logger.info(f"Static plot '{title}' saved to {path}")
