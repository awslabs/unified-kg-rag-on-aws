# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from unified_kg_rag.ingestion import (
    CentralityMetrics,
    CommunityMetrics,
    GraphStatistics,
)
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)


class HTMLExporter:
    def create_report(self, outputs_dir: Path, data: dict[str, Any]) -> None:
        logger.info("Creating HTML report in %s", outputs_dir)
        report_path = outputs_dir / "graph_analysis_report.html"

        try:
            html_content = self._generate_html(data)
            with report_path.open("w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info("HTML report created successfully: %s", report_path)
        except Exception as e:
            logger.exception("Failed to create HTML report: %s", e)

    def _generate_html(self, data: dict[str, Any]) -> str:
        head = self._generate_head()
        body = self._generate_body(data)
        return f"<!DOCTYPE html><html lang='en'>{head}{body}</html>"

    @staticmethod
    def _generate_head() -> str:
        css = """
        :root {
            --bg-color: #f8fafc; --fg-color: #0f172a; --card-bg: #ffffff;
            --primary-color: #3b82f6; --primary-hover: #2563eb; --border-color: #e5e7eb;
            --text-color: #374151; --text-light: #6b7280; --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0; padding: 40px 20px; background-color: var(--bg-color); color: var(--text-color);
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 40px; }
        h1 { font-size: 2.8em; color: var(--fg-color); margin: 0 0 10px 0; }
        header p { font-size: 1.2em; color: var(--text-light); margin: 0; }
        section { background: var(--card-bg); padding: 30px; border-radius: 12px; box-shadow: var(--shadow); margin-bottom: 30px; }
        h2 { font-size: 1.8em; color: var(--fg-color); border-bottom: 1px solid var(--border-color); padding-bottom: 15px; margin: 0 0 25px 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; }
        .card { display: flex; align-items: center; background: #f9fafb; padding: 20px; border-radius: 10px; border: 1px solid var(--border-color); }
        .card .icon { margin-right: 15px; color: var(--primary-color); }
        .card .text h3 { margin: 0 0 5px 0; color: var(--text-light); font-size: 1em; font-weight: 500; }
        .card .text .value { font-size: 2em; font-weight: 600; color: var(--fg-color); margin: 0; }
        .viz-links { display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; }
        .viz-link { display: flex; align-items: center; background-color: var(--primary-color); color: white; padding: 12px 25px; text-decoration: none; border-radius: 8px; font-weight: 500; transition: all 0.2s ease; }
        .viz-link:hover { background-color: var(--primary-hover); transform: translateY(-2px); box-shadow: var(--shadow); }
        .viz-link .icon { margin-right: 8px; }
        .tabs { display: flex; border-bottom: 1px solid var(--border-color); margin-bottom: 20px; }
        .tab-button { padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 1em; color: var(--text-light); position: relative; }
        .tab-button.active { color: var(--primary-color); font-weight: 600; }
        .tab-button.active::after { content: ''; position: absolute; bottom: -1px; left: 0; right: 0; height: 2px; background: var(--primary-color); }
        .tab-content { display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.5s; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 14px; border-bottom: 1px solid var(--border-color); text-align: left; }
        th { font-weight: 600; color: var(--text-color); font-size: 0.9em; text-transform: uppercase; }
        tr:last-child td { border-bottom: none; }
        footer { text-align: center; color: var(--text-light); font-style: italic; margin-top: 40px; }
        """
        return f"""
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Graph Analysis Report</title>
            <style>{css}</style>
        </head>
        """

    def _generate_body(self, data: dict[str, Any]) -> str:
        stats_html = self._render_stats(
            data.get("graph_stats") or GraphStatistics(num_nodes=0, num_edges=0),
            data.get("community_metrics")
            or CommunityMetrics(
                modularity=0.0,
                num_communities=0,
                average_community_size=0.0,
                largest_community_size=0,
                smallest_community_size=0,
                community_size_distribution={},
            ),
        )
        centrality_html, centrality_js = self._render_centrality(
            data.get("centrality_data") or {}
        )
        report_date = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

        return f"""
        <body>
            <div class="container">
                <header>
                    <h1>Graph Analysis Report</h1>
                    <p>An overview of the generated graph structure and key metrics.</p>
                </header>

                <section>
                    <h2>Overall Statistics</h2>
                    <div class="grid">{stats_html}</div>
                </section>

                <section>
                    <h2>Visualizations</h2>
                    <div class="viz-links">
                        <a href="interactive_graph.html" class="viz-link" target="_blank">{self._get_icon('network')} Interactive Network</a>
                        <a href="community_hierarchy.html" class="viz-link" target="_blank">{self._get_icon('hierarchy')} Community Hierarchy</a>
                        <a href="degree_distribution.html" class="viz-link" target="_blank">{self._get_icon('bar_chart')} Degree Distribution</a>
                        <a href="centrality_comparison.html" class="viz-link" target="_blank">{self._get_icon('bar_chart')} Centrality Comparison</a>
                        <a href="community_size_distribution.html" class="viz-link" target="_blank">{self._get_icon('pie_chart')} Community Sizes</a>
                    </div>
                </section>

                <section>
                    <h2>Centrality Analysis</h2>
                    {centrality_html}
                </section>

                <footer>Report generated on {report_date}</footer>
            </div>
            <script>{centrality_js}</script>
        </body>
        """

    def _render_stats(
        self,
        stats: GraphStatistics,
        community_metrics: CommunityMetrics | None = None,
    ) -> str:
        if not stats:
            return "<p>Statistics not available.</p>"

        cards_data = {
            "Nodes": (f"{stats.num_nodes:,}", "nodes"),
            "Edges": (f"{stats.num_edges:,}", "edges"),
            "Density": (f"{stats.density:.4f}" if stats.density else "N/A", "density"),
            "Avg. Clustering": (
                (
                    f"{stats.average_clustering:.4f}"
                    if stats.average_clustering
                    else "N/A"
                ),
                "clustering",
            ),
            "Components": (f"{stats.num_connected_components:,}", "components"),
            "Communities": (
                (
                    f"{community_metrics.num_communities:,}"
                    if community_metrics
                    else "N/A"
                ),
                "communities",
            ),
            "Modularity": (
                (f"{community_metrics.modularity:.4f}" if community_metrics else "N/A"),
                "modularity",
            ),
        }

        return "".join(f"""<div class="card">
                <div class="icon">{self._get_icon(icon_name)}</div>
                <div class="text">
                    <h3>{title}</h3>
                    <p class="value">{value}</p>
                </div>
            </div>""" for title, (value, icon_name) in cards_data.items())

    @staticmethod
    def _render_centrality(
        centrality_data: dict[str, CentralityMetrics],
    ) -> tuple[str, str]:
        if not centrality_data:
            return "<p>Centrality data not available.</p>", ""

        metrics = ["degree", "betweenness", "pagerank", "closeness", "eigenvector"]
        tabs_html = ""
        content_html = ""
        is_first = True

        for metric in metrics:
            if not any(getattr(c, metric, None) for c in centrality_data.values()):
                continue

            def _metric_key(node: Any, metric_name: str = metric) -> float:
                return getattr(node, metric_name, None) or -1

            sorted_nodes = sorted(
                centrality_data.values(),
                key=_metric_key,
                reverse=True,
            )[:10]

            active_class = "active" if is_first else ""
            tabs_html += f'<button class="tab-button {active_class}" onclick="openTab(event, \'{metric}\')">{metric.title()}</button>'

            rows = ""
            for i, node_metrics in enumerate(sorted_nodes):
                score = getattr(node_metrics, metric)
                score_str = f"{score:.5f}" if score is not None else "N/A"
                node_name = node_metrics.node_name or node_metrics.node_id
                rows += (
                    f"<tr><td>{i+1}</td><td>{node_name}</td><td>{score_str}</td></tr>"
                )

            content_html += f"""
            <div id="{metric}" class="tab-content {active_class}">
                <table>
                    <tr><th>Rank</th><th>Node Name</th><th>Score</th></tr>
                    {rows}
                </table>
            </div>
            """
            is_first = False

        full_html = f'<div class="tabs">{tabs_html}</div>{content_html}'

        js = """
        function openTab(evt, tabName) {
            let i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; }
            tablinks = document.getElementsByClassName("tab-button");
            for (i = 0; i < tablinks.length; i++) { tablinks[i].className = tablinks[i].className.replace(" active", ""); }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        """
        return full_html, js

    @staticmethod
    def _get_icon(name: str) -> str:
        icons = {
            "nodes": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="3"></circle><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="18" r="3"></circle><path d="M12 8v7m-6 0h12"></path></svg>',
            "edges": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.59 13.41c.44-.44.44-1.16 0-1.6L6 7.24a1.12 1.12 0 0 0-1.58 0l-1.18 1.18a1.12 1.12 0 0 0 0 1.58l4.56 4.56a1.12 1.12 0 0 0 1.58 0l1.18-1.18z"></path><path d="M18.82 10.18l-1.18-1.18a1.12 1.12 0 0 0-1.58 0L11.5 13.56a1.12 1.12 0 0 0 0 1.58l1.18 1.18a1.12 1.12 0 0 0 1.58 0l4.56-4.56a1.12 1.12 0 0 0 0-1.58z"></path></svg>',
            "density": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"></path><path d="M7 17a4 4 0 0 0 4-4 4 4 0 0 0-4-4 4 4 0 0 0-4 4 4 4 0 0 0 4 4z"></path><path d="M17 7a4 4 0 0 0-4 4 4 4 0 0 0 4 4 4 4 0 0 0 4-4 4 4 0 0 0-4-4z"></path></svg>',
            "clustering": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"></path><circle cx="12" cy="12" r="3"></circle><path d="M12 5v2m0 10v2m-7-7h2m10 0h2m-8.5-5.5-1.42-1.42M18.92 18.92l-1.42-1.42m-12 0 1.42-1.42m12.08-12.08-1.42 1.42"></path></svg>',
            "components": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 7.5a2.5 2.5 0 0 0-4.16-1.9L13 9.44l-1.84-1.84a2.5 2.5 0 1 0-3.32 0l-1.84 1.84-3.16-3.84a2.5 2.5 0 1 0-1.6 4.34l3.16 3.84-1.84 1.84a2.5 2.5 0 1 0 0 3.32l1.84-1.84 3.84 3.16a2.5 2.5 0 1 0 4.34-1.6L14.56 13l1.84-1.84 3.84 3.16a2.5 2.5 0 1 0 1.6-4.34L18.16 13l3.84-3.16A2.5 2.5 0 0 0 21 7.5z"></path></svg>',
            "communities": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>',
            "modularity": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s-8-4.5-8-12V5l8-3 8 3v5"></path><path d="m22 12-4.2 2.1"></path><path d="M16 22V12"></path><path d="M22 17h-6"></path><path d="M12 2v20"></path></svg>',
            "network": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg>',
            "hierarchy": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 12v4m0-8v2m-4 4h8m-8-4h8m-8-4h8"></path><rect x="3" y="3" width="18" height="18" rx="2"></rect></svg>',
            "bar_chart": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20V10M18 20V4M6 20V16"></path></svg>',
            "pie_chart": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8.11 2.99"></path><path d="M22 12A10 10 0 0 0 12 2v10z"></path></svg>',
        }
        return icons.get(name, "")
