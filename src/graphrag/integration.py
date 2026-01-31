"""GraphRAG integration with LangGraph workflow.

This module provides the graphrag_agent node for the LangGraph workflow,
orchestrating the full GraphRAG pipeline: entity extraction, linking,
graph construction, community detection, and hierarchical summarization.
"""

import logging
from typing import Any, Dict, List, Optional

from agents.state import (
    Community,
    ExtractedEntity,
    KnowledgeGraphData,
    LitScribeState,
    PaperSummary,
)
from graphrag.community_detector import (
    build_community_hierarchy,
    compute_community_stats,
    get_root_communities,
)
from graphrag.entity_extractor import extract_entities_batch
from graphrag.entity_linker import link_entities, update_mentions_with_linking
from graphrag.graph_builder import (
    build_knowledge_graph,
    compute_graph_statistics,
    export_graph_edges,
    get_entity_subgraph,
)
from graphrag.summarizer import (
    generate_global_summary,
    summarize_all_communities,
)

logger = logging.getLogger(__name__)


async def run_graphrag_pipeline(
    papers: List[PaperSummary],
    parsed_documents: Dict[str, Dict[str, Any]],
    research_question: str,
    llm_config: Optional[Dict[str, Any]] = None,
    entity_batch_size: int = 10,
    max_concurrent: int = 5,
    similarity_threshold: float = 0.85,
    community_resolutions: List[float] = [0.5, 1.0, 2.0],
    min_community_size: int = 2,
) -> KnowledgeGraphData:
    """Run the complete GraphRAG pipeline.

    Pipeline steps:
    1. Extract entities from all papers (batched)
    2. Link equivalent entities across papers
    3. Build knowledge graph
    4. Detect communities at multiple resolutions
    5. Generate hierarchical summaries
    6. Generate global summary

    Args:
        papers: List of analyzed paper summaries
        parsed_documents: Dict of paper_id -> parsed document
        research_question: Original research question
        llm_config: LLM configuration
        entity_batch_size: Batch size for entity extraction
        max_concurrent: Max concurrent LLM calls
        similarity_threshold: Threshold for entity linking
        community_resolutions: Resolution parameters for community detection
        min_community_size: Minimum community size

    Returns:
        KnowledgeGraphData with all extracted information
    """
    logger.info(f"Starting GraphRAG pipeline for {len(papers)} papers")

    # Step 1: Extract entities
    logger.info("Step 1/6: Extracting entities from papers...")
    entities, mentions = await extract_entities_batch(
        papers,
        parsed_documents,
        batch_size=entity_batch_size,
        max_concurrent=max_concurrent,
        llm_config=llm_config,
    )
    logger.info(f"Extracted {len(entities)} entities with {len(mentions)} mentions")

    if not entities:
        logger.warning("No entities extracted, returning empty graph")
        return KnowledgeGraphData(
            entities={},
            mentions=[],
            edges=[],
            communities=[],
            global_summary="No entities were extracted from the analyzed papers.",
            stats={"node_count": 0, "edge_count": 0},
        )

    # Step 2: Link entities
    logger.info("Step 2/6: Linking equivalent entities...")
    id_mapping, linked_entities = await link_entities(
        entities, similarity_threshold=similarity_threshold
    )
    linked_mentions = update_mentions_with_linking(mentions, id_mapping)
    logger.info(f"Linked to {len(linked_entities)} unique entities")

    # Build entity lookup
    entities_by_id = {e["entity_id"]: e for e in linked_entities}

    # Step 3: Build knowledge graph
    logger.info("Step 3/6: Building knowledge graph...")
    graph = build_knowledge_graph(linked_entities, linked_mentions, papers)
    graph_stats = compute_graph_statistics(graph)
    logger.info(
        f"Built graph with {graph_stats['node_count']} nodes, "
        f"{graph_stats['edge_count']} edges"
    )

    # Step 4: Detect communities
    logger.info("Step 4/6: Detecting communities...")
    entity_graph = get_entity_subgraph(graph)

    # Add paper_ids to entity nodes for community paper tracking
    for entity in linked_entities:
        eid = entity["entity_id"]
        if eid in entity_graph.nodes:
            entity_graph.nodes[eid]["paper_ids"] = entity.get("paper_ids", [])

    communities = build_community_hierarchy(
        entity_graph,
        resolutions=community_resolutions,
        min_community_size=min_community_size,
    )
    community_stats = compute_community_stats(communities)
    logger.info(
        f"Detected {community_stats['total_communities']} communities "
        f"across {community_stats['levels']} levels"
    )

    # Step 5: Generate community summaries
    logger.info("Step 5/6: Generating community summaries...")
    papers_by_id = {p["paper_id"]: p for p in papers}

    communities = await summarize_all_communities(
        communities,
        entities_by_id,
        papers_by_id,
        research_question,
        max_concurrent=max_concurrent,
        llm_config=llm_config,
    )

    # Step 6: Generate global summary
    logger.info("Step 6/6: Generating global summary...")
    global_summary = await generate_global_summary(
        communities,
        entities_by_id,
        papers,
        research_question,
        llm_config=llm_config,
    )

    # Export edges
    edges = export_graph_edges(graph)

    # Compile statistics
    stats = {
        **graph_stats,
        **community_stats,
        "entity_types": graph_stats.get("entity_types", {}),
    }

    logger.info("GraphRAG pipeline complete")

    return KnowledgeGraphData(
        entities=entities_by_id,
        mentions=linked_mentions,
        edges=edges,
        communities=communities,
        global_summary=global_summary,
        stats=stats,
    )


async def graphrag_agent(state: LitScribeState) -> Dict[str, Any]:
    """GraphRAG agent node for LangGraph workflow.

    This agent runs after critical_reading and before synthesis.
    It builds a knowledge graph from analyzed papers and extracts
    structural insights to enhance the synthesis.

    Args:
        state: Current workflow state

    Returns:
        State update dict with knowledge_graph and next agent
    """
    # Check if GraphRAG is enabled
    if not state.get("graphrag_enabled", True):
        logger.info("GraphRAG disabled, skipping to synthesis")
        return {
            "current_agent": "synthesis",
            "knowledge_graph": None,
        }

    papers = state.get("analyzed_papers", [])

    if not papers:
        logger.warning("No analyzed papers found, skipping GraphRAG")
        return {
            "current_agent": "synthesis",
            "knowledge_graph": None,
        }

    logger.info(f"GraphRAG agent processing {len(papers)} papers")

    parsed_documents = state.get("parsed_documents", {})
    research_question = state.get("research_question", "")
    llm_config = state.get("llm_config", {})
    batch_size = state.get("batch_size", 20)

    try:
        knowledge_graph = await run_graphrag_pipeline(
            papers=papers,
            parsed_documents=parsed_documents,
            research_question=research_question,
            llm_config=llm_config,
            entity_batch_size=min(batch_size, 10),  # Smaller batches for entity extraction
            max_concurrent=5,
        )

        # Log summary
        stats = knowledge_graph.get("stats", {})
        logger.info(
            f"GraphRAG complete: {stats.get('entity_count', 0)} entities, "
            f"{stats.get('total_communities', 0)} communities"
        )

        return {
            "current_agent": "synthesis",
            "knowledge_graph": knowledge_graph,
        }

    except Exception as e:
        logger.error(f"GraphRAG pipeline failed: {e}", exc_info=True)
        # Continue to synthesis without GraphRAG data
        return {
            "current_agent": "synthesis",
            "knowledge_graph": None,
            "errors": state.get("errors", []) + [f"GraphRAG error: {str(e)}"],
        }


def get_graphrag_insights_for_synthesis(
    knowledge_graph: Optional[KnowledgeGraphData],
) -> Dict[str, Any]:
    """Extract insights from GraphRAG for synthesis agent.

    This provides structured information that the synthesis agent
    can use to enhance the literature review.

    Args:
        knowledge_graph: GraphRAG output data

    Returns:
        Dict with structured insights
    """
    if not knowledge_graph:
        return {
            "has_graphrag": False,
        }

    entities = knowledge_graph.get("entities", {})
    communities = knowledge_graph.get("communities", [])
    stats = knowledge_graph.get("stats", {})

    # Get top entities by type
    by_type: Dict[str, List] = {
        "method": [],
        "dataset": [],
        "metric": [],
        "concept": [],
    }

    for entity in entities.values():
        etype = entity.get("entity_type", "")
        if etype in by_type:
            by_type[etype].append({
                "name": entity.get("name", ""),
                "frequency": entity.get("frequency", 0),
                "paper_count": len(entity.get("paper_ids", [])),
            })

    # Sort by frequency
    for etype in by_type:
        by_type[etype].sort(key=lambda x: x["frequency"], reverse=True)
        by_type[etype] = by_type[etype][:10]  # Top 10

    # Get root community summaries
    root_communities = get_root_communities(communities)
    community_insights = []
    for comm in root_communities[:5]:  # Top 5
        community_insights.append({
            "summary": comm.get("summary", ""),
            "entity_count": len(comm.get("entities", [])),
            "paper_count": len(comm.get("papers", [])),
        })

    return {
        "has_graphrag": True,
        "global_summary": knowledge_graph.get("global_summary", ""),
        "top_methods": by_type["method"],
        "top_datasets": by_type["dataset"],
        "top_metrics": by_type["metric"],
        "top_concepts": by_type["concept"],
        "community_insights": community_insights,
        "stats": stats,
    }
