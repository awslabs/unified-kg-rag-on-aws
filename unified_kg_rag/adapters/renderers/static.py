# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static charts as self-contained HTML pages with an inline SVG.

No plotting package and no script: each chart is one HTML file a browser
opens offline, with a native tooltip (an SVG ``<title>``) on every bar. Every
text that reaches the page — node names, the configured font and colour —
is HTML-escaped, so a name such as ``<script>`` renders as text.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from unified_kg_rag.domain.ingestion.graph_analyzer import CentralityMetrics
from unified_kg_rag.domain.models import Community
from unified_kg_rag.domain.models.config import StaticChartSettings
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

# How many nodes the centrality chart shows, by degree.
TOP_CENTRALITY_NODES = 25
# Width (px) of a centrality chart node label; a longer name is cut with an
# ellipsis and shown whole in its tooltip.
_NODE_LABEL_WIDTH = 190

_BAR_COLOR = "#3b82f6"
_METRIC_COLORS = {
    "degree": "#1f77b4",
    "betweenness": "#ff7f0e",
    "pagerank": "#2ca02c",
}
_TEXT_COLOR = "#27272a"
_AXIS_LABEL_COLOR = "#3f3f46"
_AXIS_COLOR = "#a1a1aa"
_GRID_COLOR = "#e5e5e5"


def _attr(value: object) -> str:
    """``value`` escaped for a double-quoted attribute."""
    return escape(str(value), quote=True)


def _text(value: object) -> str:
    """``value`` escaped for element content."""
    return escape(str(value), quote=False)


def _label(
    text: str,
    *,
    css_class: str,
    box: tuple[float, float, float, float],
    align: str = "center",
    size: int = 12,
    color: str = _AXIS_LABEL_COLOR,
    rotate: tuple[float, float, float] | None = None,
    tooltip: str | None = None,
) -> str:
    """One line of text in an HTML box (``x, y, width, height``) inside the
    SVG, aligned by CSS, so its width need not be known; ``rotate`` is
    ``(degrees, cx, cy)``. Overflowing text ends in an ellipsis."""
    x, y, width, height = box
    transform = ""
    if rotate is not None:
        degrees, cx, cy = rotate
        transform = f' transform="rotate({degrees} {cx:.2f} {cy:.2f})"'
    title = f' title="{_attr(tooltip)}"' if tooltip is not None else ""
    return (
        f'<foreignObject x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
        f'height="{height:.2f}"{transform}>'
        f'<div xmlns="http://www.w3.org/1999/xhtml" class="{css_class}"{title} '
        f'style="font-size:{size}px;line-height:{height:.0f}px;color:{color};'
        f"text-align:{align};white-space:nowrap;overflow:hidden;"
        f'text-overflow:ellipsis">{_text(text)}</div></foreignObject>'
    )


def _fmt_number(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}" if abs(value) >= 1 else f"{value:.2g}"


def _nice_step(span: float, target: int, integer: bool) -> float:
    raw = span / max(1, target)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = magnitude * next(m for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    return max(1.0, math.ceil(step)) if integer else step


def _y_ticks(top: float, integer: bool) -> tuple[float, list[float]]:
    """The axis maximum (at least ``top``) and its ticks from 0."""
    if top <= 0:
        top = 1.0
    step = _nice_step(top, 5, integer)
    count = math.ceil(top / step - 1e-9)
    return step * count, [step * i for i in range(count + 1)]


def _x_ticks(low: float, high: float, log: bool) -> list[float]:
    if log:
        decades = range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
        ticks = [
            m * 10.0**k
            for k in decades
            for m in (1, 2, 5)
            if low <= m * 10.0**k <= high * (1 + 1e-9)
        ]
        return ticks if len(ticks) >= 2 else [low, high]
    step = _nice_step(high - low, 6, integer=False)
    first = math.ceil(low / step - 1e-9) * step
    return [
        first + step * i
        for i in range(int((high - first) / step + 1e-9) + 1)
        if first + step * i <= high + 1e-9
    ]


@dataclass(frozen=True)
class _Frame:
    """The plot area of a chart, in SVG pixels."""

    width: int
    height: int
    left: int
    right: int
    top: int
    bottom: int

    @property
    def x0(self) -> int:
        return self.left

    @property
    def x1(self) -> int:
        return self.width - self.right

    @property
    def y0(self) -> int:
        return self.height - self.bottom

    @property
    def y1(self) -> int:
        return self.top


class StaticRenderer:
    def __init__(self, config: StaticChartSettings | Mapping[str, Any]) -> None:
        settings = StaticChartSettings.model_validate(config)
        self.figure_width = settings.figure_width
        self.figure_height = settings.figure_height
        self.font = settings.font
        self.background_color = settings.background_color
        self.log_scale_x = settings.log_scale_x

    def plot_degree_distribution(self, graph: nx.Graph, outputs_path: str) -> None:
        if not graph or graph.number_of_nodes() == 0:
            logger.warning("Cannot plot degree distribution for an empty graph.")
            return

        logger.info("Creating degree distribution plot...")
        degrees = [d for _, d in graph.degree()]
        if not degrees:
            logger.warning("Graph has no degrees to plot.")
            return

        self._save_histogram(
            degrees,
            outputs_path,
            chart="degree_distribution",
            title="Degree Distribution",
            x_label="Degree",
            y_label="Number of Nodes",
            range_label="Degree range",
        )

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
        ][:TOP_CENTRALITY_NODES]

        if not top_nodes:
            logger.warning("No nodes with names found for centrality plot.")
            return

        node_names = self._unique_factors([str(n.node_name) for n in top_nodes])
        # A metric that is off or failed to compute is None: plotted as 0 it
        # read as a measured zero. A metric no node has is left out, a node
        # without a value gets no bar.
        metrics = [
            m
            for m in _METRIC_COLORS
            if any(self._metric_value(n, m) is not None for n in top_nodes)
        ]
        if not metrics:
            logger.warning("No centrality values to plot.")
            return

        title = f"Centrality Comparison (Top {TOP_CENTRALITY_NODES} Nodes by Degree)"
        # Wider than the histograms: 25 groups of rotated labels.
        frame = _Frame(
            width=1200,
            height=700,
            left=170,
            right=30,
            top=90,
            bottom=180,
        )
        top_value = max(
            v
            for n in top_nodes
            for m in metrics
            if (v := self._metric_value(n, m)) is not None
        )
        y_max, y_ticks = _y_ticks(top_value, integer=False)
        parts = self._open_chart(frame, title)
        parts += self._y_axis(frame, y_max, y_ticks, "Score", lambda v: f"{v:.2f}")

        band = (frame.x1 - frame.x0) / len(top_nodes)
        bar_width = band * 0.8 / len(metrics)
        for i, (node, name) in enumerate(zip(top_nodes, node_names, strict=True)):
            group_left = frame.x0 + band * i + band * 0.1
            for j, metric in enumerate(metrics):
                value = self._metric_value(node, metric)
                if value is None:
                    continue
                height = (frame.y0 - frame.y1) * value / y_max
                parts.append(
                    f'<rect class="bar" data-metric="{metric}" '
                    f'x="{group_left + bar_width * j:.2f}" '
                    f'y="{frame.y0 - height:.2f}" width="{bar_width * 0.95:.2f}" '
                    f'height="{height:.2f}" fill="{_METRIC_COLORS[metric]}" '
                    f'fill-opacity="0.85"><title>'
                    f"{_text(name)}&#10;{metric.title()}: {value:.3f}"
                    "</title></rect>"
                )
            # Right-aligned, its end under the group, turned 35 degrees.
            cx, ly = frame.x0 + band * (i + 0.5), frame.y0 + 10
            parts.append(
                _label(
                    name,
                    css_class="tick-label",
                    box=(cx - _NODE_LABEL_WIDTH, ly - 8, _NODE_LABEL_WIDTH, 16),
                    align="right",
                    rotate=(-35, cx, ly),
                    tooltip=name,
                )
            )
        parts += self._axis_label_x(frame, "Nodes")
        parts += self._legend(frame, metrics)
        self._save_chart(parts, frame, output_path, title, "centrality_comparison")

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

        self._save_histogram(
            sizes,
            output_path,
            chart="community_size_distribution",
            title="Community Size Distribution (Level 0)",
            x_label="Community Size (Number of Nodes)",
            y_label="Number of Communities",
            range_label="Size range",
        )

    @staticmethod
    def _metric_value(node: CentralityMetrics, metric: str) -> float | None:
        value = getattr(node, metric, None)
        if value is None or not math.isfinite(value):
            return None
        return float(value)

    def _save_histogram(
        self,
        values: Sequence[float],
        path: str,
        *,
        chart: str,
        title: str,
        x_label: str,
        y_label: str,
        range_label: str,
    ) -> None:
        num_bins = max(1, min(50, len(set(values))))
        # A log axis needs a positive, non-empty range.
        log = self.log_scale_x and min(values) > 0 and min(values) < max(values)
        if log:
            bins: Any = np.logspace(
                np.log10(min(values)), np.log10(max(values)), num_bins + 1
            )
            # 10 ** log10(x) can round below x, which would drop the largest
            # values from the last bin.
            bins[0], bins[-1] = min(values), max(values)
        else:
            bins = num_bins
        hist, edges = np.histogram(values, bins=bins)
        low, high = float(edges[0]), float(edges[-1])

        frame = _Frame(
            width=self.figure_width,
            height=self.figure_height,
            left=80,
            right=30,
            top=60,
            bottom=70,
        )

        def x_pos(value: float) -> float:
            if log:
                share = (math.log10(value) - math.log10(low)) / (
                    math.log10(high) - math.log10(low)
                )
            else:
                share = (value - low) / (high - low)
            return frame.x0 + share * (frame.x1 - frame.x0)

        y_max, y_ticks = _y_ticks(float(hist.max()), integer=True)
        parts = self._open_chart(frame, title)
        parts += self._y_axis(frame, y_max, y_ticks, y_label, _fmt_number)
        for tick in _x_ticks(low, high, log):
            x = x_pos(tick)
            parts.append(
                _label(
                    _fmt_number(tick),
                    css_class="tick-label",
                    box=(x - 40, frame.y0 + 6, 80, 16),
                )
            )
        for count, left, right in zip(hist, edges[:-1], edges[1:], strict=True):
            if count == 0:
                continue
            x_left, x_right = x_pos(float(left)), x_pos(float(right))
            gap = (x_right - x_left) * 0.05
            height = (frame.y0 - frame.y1) * float(count) / y_max
            parts.append(
                f'<rect class="bar" x="{x_left + gap:.2f}" '
                f'y="{frame.y0 - height:.2f}" '
                f'width="{max(0.5, x_right - x_left - 2 * gap):.2f}" '
                f'height="{height:.2f}" fill="{_BAR_COLOR}" fill-opacity="0.85">'
                f"<title>{range_label}: {left:.1f} - {right:.1f}&#10;"
                f"Count: {int(count):,}</title></rect>"
            )
        parts += self._axis_label_x(frame, x_label)
        self._save_chart(parts, frame, path, title, chart)

    def _open_chart(self, frame: _Frame, title: str) -> list[str]:
        return [
            f'<rect width="{frame.width}" height="{frame.height}" '
            f'fill="{_attr(self.background_color)}"/>',
            _label(
                title,
                css_class="chart-title",
                box=(0, 10, frame.width, 30),
                size=20,
                color=_TEXT_COLOR,
            ),
        ]

    @staticmethod
    def _y_axis(
        frame: _Frame,
        y_max: float,
        ticks: list[float],
        label: str,
        fmt: Callable[[float], str],
    ) -> list[str]:
        parts = []
        for tick in ticks:
            y = frame.y0 - (frame.y0 - frame.y1) * tick / y_max
            parts.append(
                f'<line x1="{frame.x0}" x2="{frame.x1}" y1="{y:.2f}" y2="{y:.2f}" '
                f'stroke="{_GRID_COLOR}" stroke-dasharray="4 4"/>'
                + _label(
                    fmt(tick),
                    css_class="tick-label",
                    box=(frame.x0 - 70, y - 8, 62, 16),
                    align="right",
                )
            )
        cx, cy = frame.x0 - 60, (frame.y0 + frame.y1) / 2
        span = frame.y0 - frame.y1
        parts.append(
            f'<line x1="{frame.x0}" x2="{frame.x0}" y1="{frame.y1}" '
            f'y2="{frame.y0}" stroke="{_AXIS_COLOR}"/>'
            f'<line x1="{frame.x0}" x2="{frame.x1}" y1="{frame.y0}" '
            f'y2="{frame.y0}" stroke="{_AXIS_COLOR}"/>'
            + _label(
                label,
                css_class="axis-label",
                box=(cx - span / 2, cy - 12, span, 24),
                size=14,
                rotate=(-90, cx, cy),
            )
        )
        return parts

    @staticmethod
    def _axis_label_x(frame: _Frame, label: str) -> list[str]:
        return [
            _label(
                label,
                css_class="axis-label",
                box=(frame.x0, frame.height - 36, frame.x1 - frame.x0, 24),
                size=14,
            )
        ]

    @staticmethod
    def _legend(frame: _Frame, metrics: list[str]) -> list[str]:
        parts = ['<g class="legend">']
        for i, metric in enumerate(metrics):
            y = 52 + 20 * i
            x = frame.x1 - 130
            parts.append(
                f'<rect x="{x}" y="{y - 10}" width="12" height="12" '
                f'fill="{_METRIC_COLORS[metric]}"/>'
                + _label(
                    metric.title(),
                    css_class="legend-label",
                    box=(x + 18, y - 12, 110, 16),
                    align="left",
                    color=_TEXT_COLOR,
                )
            )
        parts.append("</g>")
        return parts

    @staticmethod
    def _unique_factors(names: list[str]) -> list[str]:
        """Make the centrality chart's labels unique, keeping their order.

        Display names repeat (two claims with the same subject, type and
        object, or two entities sharing a name), and two bars of one label
        would read as one node. A repeat gets the first free `` (2)``,
        `` (3)``, ... suffix.
        """
        seen: set[str] = set()
        unique: list[str] = []
        for name in names:
            factor, n = name, 1
            while factor in seen:
                n += 1
                factor = f"{name} ({n})"
            seen.add(factor)
            unique.append(factor)
        return unique

    def _save_chart(
        self, parts: list[str], frame: _Frame, path: str, title: str, chart: str
    ) -> None:
        width, height = frame.width, frame.height
        body = "\n".join(parts)
        page = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f"<title>{_text(title)}</title>\n</head>\n"
            '<body style="margin:0">\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" class="chart" '
            f'data-chart="{chart}" role="img" aria-label="{_attr(title)}" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'font-family="{_attr(self.font)}" style="max-width:100%;height:auto">\n'
            f"{body}\n</svg>\n</body>\n</html>\n"
        )
        Path(path).write_text(page, encoding="utf-8")
        logger.info("Static plot '%s' saved to %s", title, path)
