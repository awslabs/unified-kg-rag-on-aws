# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import json
from typing import Any, ClassVar

import networkx as nx
import numpy as np
from pyvis.network import Network

from aws_graphrag.adapters.ingestion.community_detector import HierarchicalCommunity
from aws_graphrag.shared import get_logger

logger = get_logger(__name__)


class InteractiveRenderer:
    NODE_SIZE_BASE: ClassVar[float] = 15
    NODE_SIZE_FACTOR: ClassVar[float] = 30
    MIN_EDGE_WIDTH: ClassVar[float] = 0.8
    MAX_EDGE_WIDTH: ClassVar[float] = 7.0
    MIN_EDGE_OPACITY: ClassVar[float] = 0.3
    MAX_EDGE_OPACITY: ClassVar[float] = 0.8

    def __init__(self, config: dict[str, Any]) -> None:
        self.height = config.get("height", "900px")
        self.width = config.get("width", "100%")
        self.physics_enabled = config.get("physics_enabled", True)
        self.cdn_resources = config.get("cdn_resources", "in_line")
        self.background_color = config.get("background_color", "#f8fafc")
        self.font_family = config.get("font_family", "Arial")
        self.tooltip_delay = config.get("tooltip_delay", 200)

    def create_network_visualization(
        self, graph: nx.Graph, layout: dict[str, tuple[float, float]], outputs_path: str
    ) -> None:
        if not graph or graph.number_of_nodes() == 0:
            logger.warning("Cannot create visualization for an empty or invalid graph.")
            return

        logger.info(
            "Creating interactive visualization for %s nodes...",
            graph.number_of_nodes(),
        )
        net = self._init_network(directed=isinstance(graph, nx.DiGraph))

        if self.physics_enabled and not layout:
            physics_options = self._get_physics_options()
            net.set_options(json.dumps(physics_options))
        else:
            net.toggle_physics(False)

        self._add_nodes(net, graph, layout)
        self._add_edges(net, graph)

        net.save_graph(outputs_path)
        logger.info("Interactive visualization saved to %s", outputs_path)

    def _init_network(self, directed: bool = False) -> Network:
        return Network(
            height=self.height,
            width=self.width,
            notebook=False,
            directed=directed,
            cdn_resources=self.cdn_resources,
            bgcolor=self.background_color,
            font_color=False,
        )

    def _add_nodes(
        self, net: Network, graph: nx.Graph, layout: dict[str, tuple[float, float]]
    ) -> None:
        community_ids = {
            attrs.get("community_id")
            for _, attrs in graph.nodes(data=True)
            if attrs.get("community_id") is not None
        }
        colors = self._generate_palette(len(community_ids))
        color_map = dict(zip(sorted(community_ids), colors, strict=True))

        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1

        for node, attrs in graph.nodes(data=True):
            node_id = str(node)
            comm_id = attrs.get("community_id")
            degree = degrees.get(node, 0)
            if attrs.get("node_type") == "claim":
                display_name = attrs.get("type", node_id)
            else:
                display_name = attrs.get("name", node_id)

            color = color_map.get(comm_id, "#94a3b8")
            node_size = self.NODE_SIZE_BASE + self.NODE_SIZE_FACTOR * np.log1p(
                degree
            ) / np.log1p(max_degree)

            pos = layout.get(node_id)
            node_x, node_y = (pos[0] * 1200, pos[1] * 1200) if pos else (None, None)

            if attrs.get("node_type") == "claim":
                title_parts = [
                    f"Node: {attrs.get('subject_name', node_id)} -> {attrs.get('object_name', node_id)} ({attrs.get('type', 'undirected').lower()})"
                ]
            else:
                title_parts = [f"Node: {attrs.get('name', node_id)}"]

            for key, val in attrs.items():
                if key not in ["name", "x", "y", "fx", "fy"] and isinstance(
                    val, (str | int | float)
                ):
                    clean_key = key.replace("_", " ").title()
                    title_parts.append(f"{clean_key}: {val}")

            title_parts.append(f"Connections: {degree}")
            title = "\n".join(title_parts)
            net.add_node(
                n_id=node_id,
                label=display_name,
                title=title,
                size=node_size,
                x=node_x,
                y=node_y,
                color=color,
                shape="dot",
                borderWidth=2,
                borderWidthSelected=4,
                font={
                    "size": 16,
                    "color": "#1e293b",
                    "face": self.font_family,
                },
                shadow=True,
            )

    @staticmethod
    def _generate_palette(num_colors: int) -> list[str]:
        if num_colors == 0:
            return []

        base_colors = [
            "#3b82f6",
            "#ef4444",
            "#10b981",
            "#f97316",
            "#8b5cf6",
            "#14b8a6",
            "#ec4899",
            "#6366f1",
            "#84cc16",
            "#06b6d4",
            "#f43f5e",
            "#d946ef",
            "#22c55e",
            "#f59e0b",
            "#0ea5e9",
        ]

        if num_colors <= len(base_colors):
            return base_colors[:num_colors]

        return [
            f"hsl({int(i * (360 / num_colors))}, 70%, 55%)" for i in range(num_colors)
        ]

    def _add_edges(self, net: Network, graph: nx.Graph) -> None:
        for u, v, attrs in graph.edges(data=True):
            weight = attrs.get("weight", 1.0)

            if attrs.get("edge_type") in ["is_subject_of", "is_object_of"]:
                title_parts = ["Edge: claim"]
            else:
                title_parts = [
                    f"Edge: {attrs.get('source_name', u)} -> {attrs.get('target_name', v) } ({attrs.get('type', 'undirected').lower()})\nStrength: {weight:.2f}"
                ]
            for key, val in attrs.items():
                if key != "weight" and isinstance(val, (str | int | float)):
                    clean_key = key.replace("_", " ").title()
                    title_parts.append(f"{clean_key}: {val}")
            title = "\n".join(title_parts)

            normalized_weight = min(weight, 1.0)
            edge_width = (
                self.MIN_EDGE_WIDTH
                + (self.MAX_EDGE_WIDTH - self.MIN_EDGE_WIDTH) * normalized_weight
            )
            edge_opacity = (
                self.MIN_EDGE_OPACITY
                + (self.MAX_EDGE_OPACITY - self.MIN_EDGE_OPACITY) * normalized_weight
            )

            net.add_edge(
                source=str(u),
                to=str(v),
                title=title,
                value=weight,
                width=edge_width,
                color={
                    "color": "#94a3b8",
                    "opacity": edge_opacity,
                    "highlight": "#334155",
                    "hover": "#475569",
                },
                smooth={"type": "dynamic"},
            )

    def _get_physics_options(self) -> dict[str, Any]:
        return {
            "physics": {
                "enabled": True,
                "barnesHut": {
                    "theta": 0.5,
                    "gravitationalConstant": -5000,
                    "centralGravity": 0.1,
                    "springLength": 200,
                    "springConstant": 0.05,
                    "damping": 0.3,
                    "avoidOverlap": 0.8,
                },
                "minVelocity": 0.75,
                "solver": "barnesHut",
                "stabilization": {"iterations": 1500, "updateInterval": 50},
                "timestep": 0.5,
                "adaptiveTimestep": True,
            },
            "interaction": {
                "dragNodes": True,
                "dragView": True,
                "hideEdgesOnDrag": True,
                "hideNodesOnDrag": False,
                "hover": True,
                "hoverConnectedEdges": True,
                "keyboard": {"enabled": True},
                "multiselect": True,
                "navigationButtons": True,
                "selectConnectedEdges": False,
                "tooltipDelay": self.tooltip_delay,
                "zoomView": True,
                "zoomSpeed": 1.5,
            },
        }

    def create_community_hierarchy(
        self, communities: list[HierarchicalCommunity], outputs_path: str
    ) -> None:
        if not communities:
            logger.warning("No hierarchical community data to visualize.")
            return

        logger.info(
            "Creating community hierarchy visualization for %s communities.",
            len(communities),
        )
        net = self._init_network(directed=True)
        net.set_options(json.dumps(self._get_hierarchy_options()))

        all_comm_ids = {comm.community_id for comm in communities}
        level_colors = {
            0: "#3b82f6",
            1: "#10b981",
            2: "#f97316",
            3: "#ec4899",
            4: "#8b5cf6",
            5: "#14b8a6",
            6: "#ef4444",
        }

        for comm in communities:
            size = len(comm.nodes)
            level = int(comm.level)
            color = level_colors.get(level, "#64748b")

            node_size = 25 + np.log1p(size) * 10
            label = f"Level {comm.level}\n{size} nodes"
            title = f"Community ID: {comm.community_id}\nHierarchy Level: {comm.level}\nMember Nodes: {size}"

            net.add_node(
                comm.community_id,
                label=label,
                title=title,
                size=node_size,
                level=level,
                color=color,
                shape="box",
                borderWidth=2,
                font={
                    "size": 14,
                    "color": "white",
                    "face": self.font_family,
                    "strokeWidth": 0.5,
                    "strokeColor": "#00000033",
                },
                margin=12,
                shapeProperties={"borderRadius": 6},
            )

        for comm in communities:
            if comm.parent_id and comm.parent_id in all_comm_ids:
                net.add_edge(
                    comm.parent_id,
                    comm.community_id,
                    color={"color": "#475569", "opacity": 0.9},
                    width=2.5,
                    arrows={
                        "to": {"enabled": True, "scaleFactor": 1.2, "type": "arrow"}
                    },
                )

        net.save_graph(outputs_path)
        logger.info("Community hierarchy visualization saved to %s", outputs_path)

    def _get_hierarchy_options(self) -> dict[str, Any]:
        return {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "sortMethod": "directed",
                    "direction": "UD",
                    "levelSeparation": 180,
                    "nodeSpacing": 180,
                    "treeSpacing": 220,
                }
            },
            "physics": {"enabled": False},
            "edges": {
                "smooth": {
                    "type": "cubicBezier",
                    "forceDirection": "vertical",
                    "roundness": 0.7,
                }
            },
            "interaction": {
                "navigationButtons": True,
                "tooltipDelay": self.tooltip_delay,
                "zoomView": True,
            },
        }
