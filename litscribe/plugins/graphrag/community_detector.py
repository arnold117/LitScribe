"""Community detector — finds communities in the entity graph via connected components.

This is a stub implementation using NetworkX connected components.
For production, port to a Leiden/Louvain algorithm.
"""
from __future__ import annotations

import networkx as nx


def detect_communities(G: nx.Graph) -> list[dict]:
    """Detect communities using connected components (stub for Leiden algorithm).

    Returns a list of community dicts with keys: community_id, level, entity_ids,
    paper_ids, summary, parent_id, children_ids.
    """
    communities: list[dict] = []
    for i, component in enumerate(nx.connected_components(G)):
        communities.append({
            "community_id": str(i),
            "level": 0,
            "entity_ids": list(component),
            "paper_ids": [],
            "summary": "",
            "parent_id": None,
            "children_ids": [],
        })
    return communities
