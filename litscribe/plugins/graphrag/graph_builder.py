"""Graph builder — constructs a NetworkX co-occurrence graph from entities and mentions."""
from __future__ import annotations

import networkx as nx


def build_graph(entities: list[dict], mentions: list[dict]) -> nx.Graph:
    """Build an undirected weighted co-occurrence graph.

    Nodes are entities; edges connect pairs of entities that appear in the
    same paper, weighted by the number of shared papers.
    """
    G: nx.Graph = nx.Graph()

    for e in entities:
        G.add_node(e["entity_id"], **{k: v for k, v in e.items() if k != "entity_id"})

    # Group entity IDs by paper
    paper_entities: dict[str, list[str]] = {}
    for m in mentions:
        paper_entities.setdefault(m["paper_id"], []).append(m["entity_id"])

    for paper_id, eids in paper_entities.items():
        for i, a in enumerate(eids):
            for b in eids[i + 1:]:
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                    G[a][b]["paper_ids"].append(paper_id)
                else:
                    G.add_edge(a, b, weight=1, edge_type="co_occurs", paper_ids=[paper_id])

    return G
