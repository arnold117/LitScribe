"""Community detection using Leiden algorithm.

This module detects communities in the knowledge graph using the Leiden
algorithm, which improves upon Louvain by guaranteeing well-connected
communities. It also supports hierarchical community detection at
multiple resolution levels.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set
import hashlib

import networkx as nx
import numpy as np

from agents.state import Community

logger = logging.getLogger(__name__)


def detect_communities_leiden(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: int = 42,
) -> Dict[str, int]:
    """Detect communities using Leiden algorithm.

    Args:
        G: Undirected graph (entity co-occurrence graph)
        resolution: Resolution parameter (higher = more communities)
        seed: Random seed for reproducibility

    Returns:
        Dict mapping node_id to community_id
    """
    if G.number_of_nodes() == 0:
        return {}

    if G.number_of_nodes() == 1:
        return {list(G.nodes())[0]: 0}

    # Try graspologic first (Microsoft implementation)
    try:
        from graspologic.partition import leiden

        logger.debug("Using graspologic Leiden implementation")

        # Convert to adjacency matrix
        nodes = list(G.nodes())
        adj_matrix = nx.to_numpy_array(G, nodelist=nodes)

        # Run Leiden
        partition = leiden(adj_matrix, resolution=resolution, random_seed=seed)

        # Map back to node IDs
        return {nodes[i]: int(partition[i]) for i in range(len(nodes))}

    except ImportError:
        logger.warning("graspologic not available, trying python-louvain fallback")

    # Fallback to python-louvain (Louvain algorithm, similar results)
    try:
        import community as community_louvain

        logger.debug("Using python-louvain fallback")
        return community_louvain.best_partition(G, resolution=resolution, random_state=seed)

    except ImportError:
        logger.warning("python-louvain not available, using simple connected components")

    # Final fallback: use connected components as communities
    components = list(nx.connected_components(G))
    partition = {}
    for i, component in enumerate(components):
        for node in component:
            partition[node] = i

    return partition


def build_community_hierarchy(
    G: nx.Graph,
    resolutions: List[float] = [0.5, 1.0, 2.0],
    min_community_size: int = 2,
) -> List[Community]:
    """Build hierarchical community structure at multiple resolutions.

    Lower resolutions produce fewer, larger communities (coarse).
    Higher resolutions produce more, smaller communities (fine).

    Args:
        G: Undirected graph (entity co-occurrence graph)
        resolutions: List of resolution parameters (low to high)
        min_community_size: Minimum nodes for a valid community

    Returns:
        List of Community objects with hierarchy information
    """
    if G.number_of_nodes() == 0:
        return []

    communities: List[Community] = []
    community_counter = 0

    # Store partitions at each level
    level_partitions: List[Dict[str, int]] = []

    # Detect communities at each resolution
    for level, resolution in enumerate(sorted(resolutions)):
        partition = detect_communities_leiden(G, resolution=resolution)
        level_partitions.append(partition)

        # Group nodes by community
        community_nodes: Dict[int, Set[str]] = defaultdict(set)
        for node, comm_id in partition.items():
            community_nodes[comm_id].add(node)

        # Create Community objects for this level
        for comm_id, nodes in community_nodes.items():
            if len(nodes) < min_community_size:
                continue

            # Generate unique community ID
            community_id = f"comm_{level}_{community_counter}"
            community_counter += 1

            # Get papers associated with these entities
            papers = set()
            for node in nodes:
                if "paper_ids" in G.nodes[node]:
                    papers.update(G.nodes[node].get("paper_ids", []))
                # Also check if we stored paper info differently
                for key in ["papers", "mentioned_in"]:
                    if key in G.nodes[node]:
                        papers.update(G.nodes[node][key])

            community = Community(
                community_id=community_id,
                level=level,
                entities=list(nodes),
                papers=list(papers),
                summary="",  # Will be filled by summarizer
                parent_id=None,  # Will be computed below
                children_ids=[],  # Will be computed below
            )
            communities.append(community)

    # Build hierarchy relationships
    _build_hierarchy_relationships(communities, level_partitions)

    logger.info(
        f"Built community hierarchy: {len(communities)} communities "
        f"across {len(resolutions)} levels"
    )

    return communities


def _build_hierarchy_relationships(
    communities: List[Community],
    level_partitions: List[Dict[str, int]],
) -> None:
    """Compute parent-child relationships between community levels.

    A community at level L is a parent of a community at level L+1 if
    the child's entities are a subset of the parent's entities.

    Args:
        communities: List of communities to update
        level_partitions: Partition at each level
    """
    # Group communities by level
    by_level: Dict[int, List[Community]] = defaultdict(list)
    for comm in communities:
        by_level[comm["level"]].append(comm)

    levels = sorted(by_level.keys())

    # For each level (except the lowest), find parent at level-1
    for i in range(1, len(levels)):
        current_level = levels[i]
        parent_level = levels[i - 1]

        for child in by_level[current_level]:
            child_entities = set(child["entities"])

            # Find best parent (most overlap)
            best_parent = None
            best_overlap = 0

            for parent in by_level[parent_level]:
                parent_entities = set(parent["entities"])
                overlap = len(child_entities & parent_entities)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_parent = parent

            if best_parent and best_overlap > 0:
                child["parent_id"] = best_parent["community_id"]
                best_parent["children_ids"].append(child["community_id"])


def get_community_by_id(
    communities: List[Community],
    community_id: str,
) -> Optional[Community]:
    """Get a community by its ID.

    Args:
        communities: List of communities
        community_id: Community ID to find

    Returns:
        Community or None
    """
    for comm in communities:
        if comm["community_id"] == community_id:
            return comm
    return None


def get_communities_at_level(
    communities: List[Community],
    level: int,
) -> List[Community]:
    """Get all communities at a specific level.

    Args:
        communities: List of communities
        level: Level to filter

    Returns:
        Communities at that level
    """
    return [c for c in communities if c["level"] == level]


def get_root_communities(communities: List[Community]) -> List[Community]:
    """Get top-level communities (no parent).

    Args:
        communities: List of communities

    Returns:
        Root communities
    """
    return [c for c in communities if c["parent_id"] is None]


def get_leaf_communities(communities: List[Community]) -> List[Community]:
    """Get bottom-level communities (no children).

    Args:
        communities: List of communities

    Returns:
        Leaf communities
    """
    return [c for c in communities if not c["children_ids"]]


def compute_community_stats(
    communities: List[Community],
    G: Optional[nx.Graph] = None,
) -> Dict[str, Any]:
    """Compute statistics about detected communities.

    Args:
        communities: List of communities
        G: Optional graph for additional stats

    Returns:
        Dict of statistics
    """
    if not communities:
        return {
            "total_communities": 0,
            "levels": 0,
            "avg_size": 0,
        }

    levels = set(c["level"] for c in communities)
    sizes = [len(c["entities"]) for c in communities]

    stats = {
        "total_communities": len(communities),
        "levels": len(levels),
        "communities_per_level": {
            level: len(get_communities_at_level(communities, level)) for level in levels
        },
        "avg_size": float(np.mean(sizes)) if sizes else 0.0,  # Convert to Python float
        "min_size": int(min(sizes)) if sizes else 0,
        "max_size": int(max(sizes)) if sizes else 0,
        "root_communities": len(get_root_communities(communities)),
        "leaf_communities": len(get_leaf_communities(communities)),
    }

    return stats


def filter_communities_by_entities(
    communities: List[Community],
    entity_type: str,
    entities_by_id: Dict[str, Dict],
) -> List[Community]:
    """Filter communities to only include entities of a specific type.

    Args:
        communities: List of communities
        entity_type: Entity type to filter
        entities_by_id: Entity lookup dict

    Returns:
        Filtered communities
    """
    filtered = []

    for comm in communities:
        matching_entities = [
            eid
            for eid in comm["entities"]
            if eid in entities_by_id
            and entities_by_id[eid].get("entity_type") == entity_type
        ]

        if matching_entities:
            filtered_comm = comm.copy()
            filtered_comm["entities"] = matching_entities
            filtered.append(filtered_comm)

    return filtered
