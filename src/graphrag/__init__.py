"""GraphRAG module for LitScribe.

This module implements a knowledge graph-based approach to literature synthesis,
inspired by Microsoft's GraphRAG paper. It extracts entities from papers,
builds a knowledge graph, detects communities, and generates hierarchical summaries.

Components:
- entity_extractor: LLM-based entity extraction from papers
- entity_linker: Cross-paper entity linking using embeddings
- graph_builder: NetworkX-based knowledge graph construction
- community_detector: Leiden algorithm for community detection
- summarizer: Hierarchical summarization of communities
- integration: GraphRAG agent for LangGraph workflow
"""

from graphrag.entity_extractor import (
    extract_entities_batch,
    extract_entities_from_paper,
)
from graphrag.entity_linker import link_entities
from graphrag.graph_builder import build_knowledge_graph, compute_graph_statistics
from graphrag.community_detector import (
    build_community_hierarchy,
    detect_communities_leiden,
)
from graphrag.summarizer import (
    generate_global_summary,
    summarize_all_communities,
    summarize_community,
)
from graphrag.integration import graphrag_agent, run_graphrag_pipeline

__all__ = [
    # Entity extraction
    "extract_entities_from_paper",
    "extract_entities_batch",
    # Entity linking
    "link_entities",
    # Graph building
    "build_knowledge_graph",
    "compute_graph_statistics",
    # Community detection
    "detect_communities_leiden",
    "build_community_hierarchy",
    # Summarization
    "summarize_community",
    "summarize_all_communities",
    "generate_global_summary",
    # Integration
    "graphrag_agent",
    "run_graphrag_pipeline",
]
