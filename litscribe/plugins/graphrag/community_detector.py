"""Community detection using Leiden algorithm.

Falls back to connected components if graspologic is not available
or if the graph is too small for meaningful community detection.
"""
from __future__ import annotations

import logging

import networkx as nx

logger = logging.getLogger(__name__)

MIN_NODES_FOR_LEIDEN = 5  # Below this, just use connected components


def detect_communities(G: nx.Graph, resolution: float = 1.0, seed: int = 42) -> list[dict]:
    """Detect communities in a knowledge graph.

    Uses Leiden algorithm (via graspologic) for graphs with >= 5 nodes,
    falls back to connected components for smaller graphs.
    """
    if G.number_of_nodes() == 0:
        return []

    if G.number_of_nodes() < MIN_NODES_FOR_LEIDEN:
        return _connected_components_fallback(G)

    try:
        return _leiden_communities(G, resolution, seed)
    except Exception as e:
        logger.warning(f"Leiden failed, falling back to connected components: {e}")
        return _connected_components_fallback(G)


def _leiden_communities(G: nx.Graph, resolution: float, seed: int) -> list[dict]:
    """Run Leiden algorithm via graspologic."""
    from graspologic.partition import leiden

    # leiden() returns dict: node_id -> community_id
    partition = leiden(G, resolution=resolution, random_seed=seed)

    # Group nodes by community
    communities_map: dict[int, list[str]] = {}
    for node_id, comm_id in partition.items():
        communities_map.setdefault(comm_id, []).append(str(node_id))

    communities = []
    for comm_id, entity_ids in sorted(communities_map.items()):
        communities.append({
            "community_id": str(comm_id),
            "level": 0,
            "entity_ids": entity_ids,
            "paper_ids": _collect_paper_ids(G, entity_ids),
            "summary": "",
            "parent_id": None,
            "children_ids": [],
        })

    return communities


def _connected_components_fallback(G: nx.Graph) -> list[dict]:
    """Simple fallback using connected components."""
    communities = []
    for i, component in enumerate(nx.connected_components(G)):
        entity_ids = list(component)
        communities.append({
            "community_id": str(i),
            "level": 0,
            "entity_ids": entity_ids,
            "paper_ids": _collect_paper_ids(G, entity_ids),
            "summary": "",
            "parent_id": None,
            "children_ids": [],
        })
    return communities


def _collect_paper_ids(G: nx.Graph, entity_ids: list[str]) -> list[str]:
    """Collect paper_ids from edges touching these entities."""
    paper_ids = set()
    for eid in entity_ids:
        for _, _, data in G.edges(eid, data=True):
            for pid in data.get("paper_ids", []):
                paper_ids.add(pid)
    return list(paper_ids)
