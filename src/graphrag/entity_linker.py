"""Entity linking across papers using embedding similarity.

This module links equivalent entities across papers by computing
embedding similarity and clustering similar entities.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from agents.state import ExtractedEntity

logger = logging.getLogger(__name__)

# Lazy load sentence transformer to avoid import overhead
_embedding_model = None


def get_embedding_model():
    """Get or initialize the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence transformer model for entity linking...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def compute_entity_embeddings(
    entities: List[ExtractedEntity],
) -> Dict[str, np.ndarray]:
    """Compute embeddings for entity names and descriptions.

    Args:
        entities: List of extracted entities

    Returns:
        Dict mapping entity_id to embedding vector
    """
    model = get_embedding_model()

    # Create text representation for each entity
    texts = []
    entity_ids = []

    for entity in entities:
        # Combine name, aliases, and description for richer embedding
        name = entity["name"]
        aliases = " ".join(entity.get("aliases", [])[:3])
        desc = entity.get("description", "")[:200]

        text = f"{name}. {aliases}. {desc}"
        texts.append(text)
        entity_ids.append(entity["entity_id"])

    if not texts:
        return {}

    # Compute embeddings
    embeddings = model.encode(texts, show_progress_bar=False)

    # Create mapping
    return {eid: emb for eid, emb in zip(entity_ids, embeddings)}


def compute_similarity_matrix(
    embeddings: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings: Dict of entity_id -> embedding

    Returns:
        Tuple of (similarity matrix, list of entity IDs in order)
    """
    if not embeddings:
        return np.array([]), []

    entity_ids = list(embeddings.keys())
    vectors = np.array([embeddings[eid] for eid in entity_ids])

    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
    normalized = vectors / norms

    # Compute cosine similarity
    similarity = np.dot(normalized, normalized.T)

    return similarity, entity_ids


def cluster_similar_entities(
    similarity_matrix: np.ndarray,
    entity_ids: List[str],
    threshold: float = 0.85,
) -> List[Set[str]]:
    """Cluster entities with similar embeddings.

    Args:
        similarity_matrix: Pairwise similarity matrix
        entity_ids: List of entity IDs corresponding to matrix rows/cols
        threshold: Minimum similarity to consider entities equivalent

    Returns:
        List of sets, where each set contains equivalent entity IDs
    """
    if len(entity_ids) <= 1:
        return [set(entity_ids)] if entity_ids else []

    # Build graph: add edge between entities with similarity >= threshold
    G = nx.Graph()
    G.add_nodes_from(range(len(entity_ids)))

    for i in range(len(entity_ids)):
        for j in range(i + 1, len(entity_ids)):
            if similarity_matrix[i, j] >= threshold:
                G.add_edge(i, j)

    # Find connected components as clusters
    clusters = []
    for component in nx.connected_components(G):
        clusters.append({entity_ids[idx] for idx in component})

    return clusters


def select_canonical_entity(
    cluster: Set[str],
    entities_by_id: Dict[str, ExtractedEntity],
) -> str:
    """Select the canonical entity from a cluster.

    Selection criteria (in order):
    1. Highest frequency (mentioned in most papers)
    2. Most paper references
    3. Shortest name (likely more canonical)

    Args:
        cluster: Set of entity IDs in the cluster
        entities_by_id: Dict mapping entity_id to entity

    Returns:
        Entity ID of the canonical entity
    """
    candidates = [(eid, entities_by_id[eid]) for eid in cluster if eid in entities_by_id]

    if not candidates:
        return list(cluster)[0]

    # Sort by: frequency (desc), paper count (desc), name length (asc)
    candidates.sort(
        key=lambda x: (
            -x[1].get("frequency", 0),
            -len(x[1].get("paper_ids", [])),
            len(x[1].get("name", "")),
        )
    )

    return candidates[0][0]


async def link_entities(
    entities: List[ExtractedEntity],
    similarity_threshold: float = 0.85,
) -> Tuple[Dict[str, str], List[ExtractedEntity]]:
    """Link equivalent entities across papers.

    This function:
    1. Computes embeddings for all entities
    2. Clusters similar entities
    3. Selects canonical entity for each cluster
    4. Returns mapping and merged entity list

    Args:
        entities: List of extracted entities
        similarity_threshold: Minimum similarity to link entities (0-1)

    Returns:
        Tuple of:
        - Dict mapping original entity_id to canonical entity_id
        - List of merged (deduplicated) entities
    """
    if not entities:
        return {}, []

    logger.info(f"Linking {len(entities)} entities with threshold {similarity_threshold}")

    # Build lookup
    entities_by_id = {e["entity_id"]: e for e in entities}

    # Group entities by type (only link within same type)
    by_type: Dict[str, List[ExtractedEntity]] = {}
    for entity in entities:
        etype = entity["entity_type"]
        if etype not in by_type:
            by_type[etype] = []
        by_type[etype].append(entity)

    # Process each type separately
    id_mapping: Dict[str, str] = {}  # original -> canonical
    merged_entities: Dict[str, ExtractedEntity] = {}  # canonical_id -> merged entity

    for entity_type, type_entities in by_type.items():
        if len(type_entities) <= 1:
            # No linking needed for single entity
            for e in type_entities:
                id_mapping[e["entity_id"]] = e["entity_id"]
                merged_entities[e["entity_id"]] = e.copy()
            continue

        logger.debug(f"Processing {len(type_entities)} entities of type {entity_type}")

        # Compute embeddings
        embeddings = compute_entity_embeddings(type_entities)

        # Compute similarity
        similarity, entity_ids = compute_similarity_matrix(embeddings)

        # Cluster
        clusters = cluster_similar_entities(similarity, entity_ids, similarity_threshold)

        # Process clusters
        for cluster in clusters:
            canonical_id = select_canonical_entity(cluster, entities_by_id)

            # Create merged entity
            canonical = entities_by_id[canonical_id].copy()

            # Merge all cluster members
            all_paper_ids = set()
            all_aliases = set()
            total_frequency = 0

            for eid in cluster:
                entity = entities_by_id[eid]
                all_paper_ids.update(entity.get("paper_ids", []))
                all_aliases.update(entity.get("aliases", []))
                all_aliases.add(entity["name"])
                total_frequency += entity.get("frequency", 1)

                # Map to canonical
                id_mapping[eid] = canonical_id

            # Update canonical entity
            canonical["paper_ids"] = list(all_paper_ids)
            canonical["aliases"] = list(all_aliases - {canonical["name"]})
            canonical["frequency"] = total_frequency

            merged_entities[canonical_id] = canonical

    logger.info(
        f"Entity linking complete: {len(entities)} -> {len(merged_entities)} entities"
    )

    return id_mapping, list(merged_entities.values())


def update_mentions_with_linking(
    mentions: List[Dict],
    id_mapping: Dict[str, str],
) -> List[Dict]:
    """Update entity mentions with linked entity IDs.

    Args:
        mentions: List of entity mentions
        id_mapping: Mapping from original to canonical entity IDs

    Returns:
        Updated mentions with canonical entity IDs
    """
    updated = []
    for mention in mentions:
        updated_mention = mention.copy()
        original_id = mention["entity_id"]
        updated_mention["entity_id"] = id_mapping.get(original_id, original_id)
        updated.append(updated_mention)
    return updated
