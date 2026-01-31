"""Knowledge graph construction using NetworkX.

This module builds a knowledge graph from extracted entities, mentions,
and paper metadata. The graph captures relationships between papers,
entities (methods, datasets, metrics, concepts), and their connections.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from agents.state import EntityMention, ExtractedEntity, GraphEdge, PaperSummary

logger = logging.getLogger(__name__)


def build_knowledge_graph(
    entities: List[ExtractedEntity],
    mentions: List[EntityMention],
    papers: List[PaperSummary],
    include_citation_edges: bool = True,
) -> nx.MultiDiGraph:
    """Build a knowledge graph from extracted data.

    The graph contains:
    - Paper nodes: Research papers with metadata
    - Entity nodes: Methods, datasets, metrics, concepts
    - Edges:
        - mentions: Paper -> Entity (paper mentions entity)
        - co_occurs: Entity <-> Entity (entities co-occur in same paper)
        - cites: Paper -> Paper (if citation data available)

    Args:
        entities: List of extracted entities
        mentions: List of entity mentions
        papers: List of paper summaries
        include_citation_edges: Whether to add citation edges

    Returns:
        NetworkX MultiDiGraph
    """
    G = nx.MultiDiGraph()

    # Build entity lookup
    entities_by_id = {e["entity_id"]: e for e in entities}

    # Build paper lookup
    papers_by_id = {p["paper_id"]: p for p in papers}

    # Build mentions lookup: paper_id -> list of entity_ids
    mentions_by_paper: Dict[str, List[str]] = defaultdict(list)
    for mention in mentions:
        mentions_by_paper[mention["paper_id"]].append(mention["entity_id"])

    # Add paper nodes
    for paper in papers:
        paper_id = paper["paper_id"]
        G.add_node(
            paper_id,
            node_type="paper",
            title=paper.get("title", ""),
            year=paper.get("year", 0),
            citations=paper.get("citations", 0),
            source=paper.get("source", ""),
            relevance_score=paper.get("relevance_score", 0.0),
        )

    # Add entity nodes
    for entity in entities:
        entity_id = entity["entity_id"]
        G.add_node(
            entity_id,
            node_type="entity",
            name=entity["name"],
            entity_type=entity["entity_type"],
            frequency=entity.get("frequency", 1),
            aliases=entity.get("aliases", []),
            description=entity.get("description", ""),
        )

    # Add mention edges (paper -> entity)
    for mention in mentions:
        paper_id = mention["paper_id"]
        entity_id = mention["entity_id"]

        if paper_id in G.nodes and entity_id in G.nodes:
            G.add_edge(
                paper_id,
                entity_id,
                edge_type="mentions",
                section=mention.get("section", ""),
                confidence=mention.get("confidence", 1.0),
            )

    # Add co-occurrence edges (entity <-> entity)
    co_occurrence_counts: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for paper_id, entity_ids in mentions_by_paper.items():
        unique_entities = list(set(entity_ids))
        # Create pairs of co-occurring entities
        for i, eid1 in enumerate(unique_entities):
            for eid2 in unique_entities[i + 1 :]:
                # Sort to ensure consistent ordering
                pair = tuple(sorted([eid1, eid2]))
                co_occurrence_counts[pair].add(paper_id)

    for (eid1, eid2), paper_ids in co_occurrence_counts.items():
        weight = len(paper_ids)
        if weight > 0 and eid1 in G.nodes and eid2 in G.nodes:
            G.add_edge(
                eid1,
                eid2,
                edge_type="co_occurs",
                weight=weight,
                paper_ids=list(paper_ids),
            )
            # Add reverse edge for undirected co-occurrence
            G.add_edge(
                eid2,
                eid1,
                edge_type="co_occurs",
                weight=weight,
                paper_ids=list(paper_ids),
            )

    # Add citation edges if requested
    if include_citation_edges:
        _add_citation_edges(G, papers)

    logger.info(
        f"Built knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )

    return G


def _add_citation_edges(
    G: nx.MultiDiGraph,
    papers: List[PaperSummary],
) -> None:
    """Add citation edges between papers.

    Note: This relies on citation data being available in paper metadata.
    For now, we create inferred citation relationships based on year
    and shared entities (older papers with shared entities likely cited).

    Args:
        G: Graph to add edges to
        papers: List of papers
    """
    # Group papers by year
    by_year: Dict[int, List[str]] = defaultdict(list)
    for paper in papers:
        year = paper.get("year", 0)
        if year > 0:
            by_year[year].append(paper["paper_id"])

    # For each paper, check if it likely cites older papers with shared entities
    paper_entities: Dict[str, Set[str]] = defaultdict(set)

    for node, data in G.nodes(data=True):
        if data.get("node_type") == "paper":
            # Get entities mentioned by this paper
            for _, target, edge_data in G.out_edges(node, data=True):
                if edge_data.get("edge_type") == "mentions":
                    paper_entities[node].add(target)

    # Create potential citation edges (newer -> older with shared entities)
    sorted_years = sorted(by_year.keys())

    for i, year in enumerate(sorted_years):
        for paper_id in by_year[year]:
            entities = paper_entities.get(paper_id, set())

            # Check older papers
            for older_year in sorted_years[:i]:
                for older_paper_id in by_year[older_year]:
                    older_entities = paper_entities.get(older_paper_id, set())

                    # If they share entities, create a weak citation link
                    shared = entities & older_entities
                    if len(shared) >= 2:  # At least 2 shared entities
                        G.add_edge(
                            paper_id,
                            older_paper_id,
                            edge_type="likely_cites",
                            weight=len(shared),
                            shared_entities=list(shared),
                        )


def compute_graph_statistics(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Compute basic statistics about the knowledge graph.

    Args:
        G: Knowledge graph

    Returns:
        Dict of statistics
    """
    # Count node types
    paper_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "paper")
    entity_nodes = sum(
        1 for _, d in G.nodes(data=True) if d.get("node_type") == "entity"
    )

    # Count entity types
    entity_type_counts: Dict[str, int] = defaultdict(int)
    for _, data in G.nodes(data=True):
        if data.get("node_type") == "entity":
            entity_type_counts[data.get("entity_type", "unknown")] += 1

    # Count edge types
    edge_type_counts: Dict[str, int] = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_type_counts[data.get("edge_type", "unknown")] += 1

    # Compute centrality for entities
    entity_ids = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entity"]

    if entity_ids:
        # Create subgraph of entities only (for degree centrality)
        entity_subgraph = G.subgraph(entity_ids).copy()
        degree_centrality = nx.degree_centrality(entity_subgraph)
        top_entities = sorted(
            degree_centrality.items(), key=lambda x: x[1], reverse=True
        )[:10]
    else:
        top_entities = []

    # Get entity names for top entities
    top_entity_names = []
    for eid, centrality in top_entities:
        name = G.nodes[eid].get("name", eid)
        etype = G.nodes[eid].get("entity_type", "")
        top_entity_names.append((name, etype, centrality))

    return {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "paper_count": paper_nodes,
        "entity_count": entity_nodes,
        "entity_types": dict(entity_type_counts),
        "edge_types": dict(edge_type_counts),
        "top_entities": top_entity_names,
        "density": nx.density(G) if G.number_of_nodes() > 1 else 0,
    }


def get_entity_subgraph(
    G: nx.MultiDiGraph,
    entity_type: Optional[str] = None,
) -> nx.Graph:
    """Extract entity-only subgraph for community detection.

    Args:
        G: Full knowledge graph
        entity_type: Optional filter for entity type

    Returns:
        Undirected graph of entities with co-occurrence edges
    """
    # Get entity nodes
    entity_nodes = []
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "entity":
            if entity_type is None or data.get("entity_type") == entity_type:
                entity_nodes.append(node)

    # Create undirected subgraph
    subgraph = nx.Graph()

    for node in entity_nodes:
        subgraph.add_node(node, **G.nodes[node])

    # Add co-occurrence edges (undirected)
    added_edges = set()
    for u, v, data in G.edges(data=True):
        if data.get("edge_type") == "co_occurs":
            if u in entity_nodes and v in entity_nodes:
                edge_key = tuple(sorted([u, v]))
                if edge_key not in added_edges:
                    subgraph.add_edge(u, v, weight=data.get("weight", 1))
                    added_edges.add(edge_key)

    return subgraph


def export_graph_edges(G: nx.MultiDiGraph) -> List[GraphEdge]:
    """Export graph edges in state-compatible format.

    Args:
        G: Knowledge graph

    Returns:
        List of GraphEdge dicts
    """
    edges = []

    for u, v, data in G.edges(data=True):
        edge = GraphEdge(
            source_id=u,
            target_id=v,
            edge_type=data.get("edge_type", "unknown"),
            weight=data.get("weight", 1.0),
            paper_ids=data.get("paper_ids", []),
        )
        edges.append(edge)

    return edges


def get_papers_for_entity(G: nx.MultiDiGraph, entity_id: str) -> List[str]:
    """Get all papers that mention a given entity.

    Args:
        G: Knowledge graph
        entity_id: Entity node ID

    Returns:
        List of paper IDs
    """
    paper_ids = []

    for node, _, data in G.in_edges(entity_id, data=True):
        if data.get("edge_type") == "mentions":
            paper_ids.append(node)

    return paper_ids


def get_entities_for_paper(G: nx.MultiDiGraph, paper_id: str) -> List[str]:
    """Get all entities mentioned by a paper.

    Args:
        G: Knowledge graph
        paper_id: Paper node ID

    Returns:
        List of entity IDs
    """
    entity_ids = []

    for _, target, data in G.out_edges(paper_id, data=True):
        if data.get("edge_type") == "mentions":
            entity_ids.append(target)

    return entity_ids
