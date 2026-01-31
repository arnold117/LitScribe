"""Hierarchical summarization of communities.

This module generates summaries for detected communities, working from
leaf communities up to generate hierarchical summaries that capture
the research landscape at multiple granularities.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from agents.state import Community, ExtractedEntity, PaperSummary
from graphrag.prompts import (
    COMMUNITY_SUMMARY_PROMPT,
    COMMUNITY_SUMMARY_SYSTEM,
    GLOBAL_SUMMARY_PROMPT,
    GLOBAL_SUMMARY_SYSTEM,
)

logger = logging.getLogger(__name__)


def _format_entities_for_prompt(
    entity_ids: List[str],
    entities_by_id: Dict[str, ExtractedEntity],
    max_entities: int = 15,
) -> str:
    """Format entities for inclusion in prompt.

    Args:
        entity_ids: List of entity IDs in community
        entities_by_id: Entity lookup dict
        max_entities: Maximum entities to include

    Returns:
        Formatted string
    """
    lines = []

    # Sort by frequency (most mentioned first)
    sorted_ids = sorted(
        entity_ids,
        key=lambda eid: entities_by_id.get(eid, {}).get("frequency", 0),
        reverse=True,
    )[:max_entities]

    for eid in sorted_ids:
        entity = entities_by_id.get(eid)
        if entity:
            etype = entity.get("entity_type", "unknown")
            name = entity.get("name", eid)
            desc = entity.get("description", "")[:100]
            freq = entity.get("frequency", 0)
            lines.append(f"- [{etype.upper()}] {name} (freq: {freq}): {desc}")

    return "\n".join(lines) if lines else "No entities found."


def _format_papers_for_prompt(
    paper_ids: List[str],
    papers_by_id: Dict[str, PaperSummary],
    max_papers: int = 10,
) -> str:
    """Format papers for inclusion in prompt.

    Args:
        paper_ids: List of paper IDs in community
        papers_by_id: Paper lookup dict
        max_papers: Maximum papers to include

    Returns:
        Formatted string
    """
    lines = []

    # Sort by citations (most cited first)
    sorted_ids = sorted(
        paper_ids,
        key=lambda pid: papers_by_id.get(pid, {}).get("citations", 0),
        reverse=True,
    )[:max_papers]

    for pid in sorted_ids:
        paper = papers_by_id.get(pid)
        if paper:
            title = paper.get("title", "Unknown")[:80]
            year = paper.get("year", "")
            citations = paper.get("citations", 0)
            findings = paper.get("key_findings", [])[:2]
            findings_str = "; ".join(findings) if findings else ""

            lines.append(f"- {title} ({year}, {citations} cites)")
            if findings_str:
                lines.append(f"  Key findings: {findings_str}")

    return "\n".join(lines) if lines else "No papers found."


def _format_relationships_for_prompt(
    entity_ids: List[str],
    entities_by_id: Dict[str, ExtractedEntity],
) -> str:
    """Format entity relationships for prompt.

    Args:
        entity_ids: List of entity IDs in community
        entities_by_id: Entity lookup dict

    Returns:
        Formatted string describing relationships
    """
    # Group entities by type
    by_type: Dict[str, List[str]] = {}
    for eid in entity_ids:
        entity = entities_by_id.get(eid)
        if entity:
            etype = entity.get("entity_type", "unknown")
            if etype not in by_type:
                by_type[etype] = []
            by_type[etype].append(entity.get("name", eid))

    lines = []

    # Methods
    if "method" in by_type:
        methods = by_type["method"][:5]
        lines.append(f"Methods: {', '.join(methods)}")

    # Datasets
    if "dataset" in by_type:
        datasets = by_type["dataset"][:5]
        lines.append(f"Datasets: {', '.join(datasets)}")

    # Metrics
    if "metric" in by_type:
        metrics = by_type["metric"][:3]
        lines.append(f"Metrics: {', '.join(metrics)}")

    # Concepts
    if "concept" in by_type:
        concepts = by_type["concept"][:3]
        lines.append(f"Concepts: {', '.join(concepts)}")

    return "\n".join(lines) if lines else "No relationships identified."


async def summarize_community(
    community: Community,
    entities_by_id: Dict[str, ExtractedEntity],
    papers_by_id: Dict[str, PaperSummary],
    research_question: str,
    llm_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a summary for a single community.

    Args:
        community: Community to summarize
        entities_by_id: Entity lookup dict
        papers_by_id: Paper lookup dict
        research_question: Original research question
        llm_config: LLM configuration

    Returns:
        Summary text
    """
    from agents.tools import call_llm

    # Format inputs
    entities_list = _format_entities_for_prompt(
        community["entities"], entities_by_id
    )
    papers_list = _format_papers_for_prompt(
        community["papers"], papers_by_id
    )
    relationships = _format_relationships_for_prompt(
        community["entities"], entities_by_id
    )

    # Build prompt
    prompt = COMMUNITY_SUMMARY_PROMPT.format(
        research_question=research_question,
        entities_list=entities_list,
        papers_list=papers_list,
        relationships=relationships,
    )

    try:
        response = await call_llm(
            messages=[
                {"role": "system", "content": COMMUNITY_SUMMARY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            model=llm_config.get("model") if llm_config else None,
        )
        return response.strip()

    except Exception as e:
        logger.error(f"Failed to summarize community {community['community_id']}: {e}")
        # Return a basic summary
        entity_names = [
            entities_by_id[eid].get("name", eid)
            for eid in community["entities"][:5]
            if eid in entities_by_id
        ]
        return f"This community contains research on: {', '.join(entity_names)}."


async def summarize_all_communities(
    communities: List[Community],
    entities_by_id: Dict[str, ExtractedEntity],
    papers_by_id: Dict[str, PaperSummary],
    research_question: str,
    max_concurrent: int = 3,
    llm_config: Optional[Dict[str, Any]] = None,
) -> List[Community]:
    """Generate summaries for all communities.

    Processes communities from leaf level up, so parent summaries
    can potentially incorporate child summaries.

    Args:
        communities: List of communities
        entities_by_id: Entity lookup dict
        papers_by_id: Paper lookup dict
        research_question: Original research question
        max_concurrent: Maximum concurrent LLM calls
        llm_config: LLM configuration

    Returns:
        Communities with summaries filled in
    """
    if not communities:
        return []

    # Sort by level (highest/leaf first)
    sorted_communities = sorted(communities, key=lambda c: c["level"], reverse=True)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def summarize_with_limit(community: Community) -> None:
        async with semaphore:
            if not community.get("summary"):  # Only summarize if empty
                summary = await summarize_community(
                    community,
                    entities_by_id,
                    papers_by_id,
                    research_question,
                    llm_config,
                )
                community["summary"] = summary

    # Process level by level (to ensure children are summarized before parents)
    levels = sorted(set(c["level"] for c in sorted_communities), reverse=True)

    for level in levels:
        level_communities = [c for c in sorted_communities if c["level"] == level]
        logger.info(f"Summarizing {len(level_communities)} communities at level {level}")

        await asyncio.gather(
            *[summarize_with_limit(c) for c in level_communities],
            return_exceptions=True,
        )

    return communities


async def generate_global_summary(
    communities: List[Community],
    entities_by_id: Dict[str, ExtractedEntity],
    papers: List[PaperSummary],
    research_question: str,
    llm_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a global summary from community summaries.

    This creates a high-level overview of the entire research landscape
    by synthesizing community summaries.

    Args:
        communities: Communities with summaries
        entities_by_id: Entity lookup dict
        papers: List of papers
        research_question: Original research question
        llm_config: LLM configuration

    Returns:
        Global summary text
    """
    from agents.tools import call_llm

    # Get root communities (or all if hierarchy is flat)
    root_communities = [c for c in communities if c["parent_id"] is None]
    if not root_communities:
        root_communities = communities[:5]  # Take first 5 if no hierarchy

    # Format community summaries
    community_summaries = []
    for i, comm in enumerate(root_communities[:10], 1):  # Max 10 communities
        summary = comm.get("summary", "No summary available.")
        entity_count = len(comm["entities"])
        paper_count = len(comm["papers"])
        community_summaries.append(
            f"**Community {i}** ({entity_count} entities, {paper_count} papers):\n{summary}"
        )

    # Compute statistics
    all_entities = list(entities_by_id.values())

    methods = [e["name"] for e in all_entities if e["entity_type"] == "method"]
    datasets = [e["name"] for e in all_entities if e["entity_type"] == "dataset"]

    # Sort by frequency
    method_counts = {}
    dataset_counts = {}
    for e in all_entities:
        if e["entity_type"] == "method":
            method_counts[e["name"]] = e.get("frequency", 0)
        elif e["entity_type"] == "dataset":
            dataset_counts[e["name"]] = e.get("frequency", 0)

    top_methods = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_datasets = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Format prompt
    prompt = GLOBAL_SUMMARY_PROMPT.format(
        research_question=research_question,
        community_summaries="\n\n".join(community_summaries),
        num_papers=len(papers),
        num_entities=len(all_entities),
        num_communities=len(communities),
        top_methods=", ".join([m[0] for m in top_methods]) or "None identified",
        top_datasets=", ".join([d[0] for d in top_datasets]) or "None identified",
    )

    try:
        response = await call_llm(
            messages=[
                {"role": "system", "content": GLOBAL_SUMMARY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            model=llm_config.get("model") if llm_config else None,
        )
        return response.strip()

    except Exception as e:
        logger.error(f"Failed to generate global summary: {e}")
        # Return a basic summary
        return (
            f"This literature review analyzed {len(papers)} papers on '{research_question}'. "
            f"The analysis identified {len(all_entities)} key entities "
            f"organized into {len(communities)} research communities."
        )
