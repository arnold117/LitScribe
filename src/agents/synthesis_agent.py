"""Synthesis Agent for LitScribe.

This agent is responsible for:
1. Theme identification - finding common themes across papers
2. Gap analysis - identifying research gaps and future directions
3. Literature review generation - writing the final review narrative
4. Citation formatting - generating proper citations

Phase 7.5 Enhancement:
- GraphRAG deep integration: uses knowledge graph communities as themes
- Skips redundant LLM theme identification when GraphRAG data available
- Leverages entity relationships for richer synthesis
"""

import json
import logging
from typing import Any, Dict, List, Optional

from agents.errors import LLMError
from agents.prompts import (
    CITATION_FORMAT_PROMPT,
    GAP_ANALYSIS_PROMPT,
    GRAPHRAG_LITERATURE_REVIEW_PROMPT,
    LITERATURE_REVIEW_PROMPT,
    THEME_IDENTIFICATION_PROMPT,
    format_summaries_for_prompt,
    get_language_instruction,
)
from agents.state import (
    Community,
    ExtractedEntity,
    KnowledgeGraphData,
    LitScribeState,
    PaperSummary,
    SynthesisOutput,
    ThemeCluster,
)
from agents.tools import call_llm

logger = logging.getLogger(__name__)


def communities_to_themes(
    communities: List[Community],
    entities: Dict[str, ExtractedEntity],
    analyzed_papers: List[PaperSummary],
) -> List[ThemeCluster]:
    """Convert GraphRAG communities to ThemeCluster format.

    This enables deep integration: GraphRAG communities become themes directly,
    skipping redundant LLM theme identification.

    Args:
        communities: List of detected communities from GraphRAG
        entities: Entity dictionary for looking up entity names
        analyzed_papers: List of analyzed papers for extracting key points

    Returns:
        List of ThemeCluster objects derived from communities
    """
    themes = []
    paper_findings_map = {
        p["paper_id"]: p.get("key_findings", [])
        for p in analyzed_papers
    }

    # Sort communities by level (lower = higher hierarchy) and size
    sorted_communities = sorted(
        communities,
        key=lambda c: (c.get("level", 0), -len(c.get("papers", []))),
    )

    # Use top-level communities as themes (level 0 or 1)
    top_communities = [c for c in sorted_communities if c.get("level", 0) <= 1]

    # Limit to 6 themes max
    for community in top_communities[:6]:
        community_id = community.get("community_id") or "unknown"
        entity_ids = community.get("entities") or []
        paper_ids = community.get("papers") or []
        summary = community.get("summary") or ""

        # Build theme name from top entities (defensive access)
        entity_names = []
        for eid in entity_ids[:3]:
            if eid and eid in entities:
                entity = entities[eid]
                entity_names.append(entity.get("name") or eid)
            elif eid:
                entity_names.append(str(eid))

        if entity_names:
            theme_name = " / ".join(entity_names)
        else:
            theme_name = f"Research Cluster {community_id[:8]}"

        # Use community summary as description
        description = summary if summary else f"Papers exploring {theme_name}"

        # Extract key points from papers in this community
        key_points = []
        for pid in paper_ids[:5]:
            findings = paper_findings_map.get(pid, [])
            key_points.extend(findings[:2])
        key_points = key_points[:8]  # Limit key points

        themes.append(ThemeCluster(
            theme=theme_name,
            description=description,
            paper_ids=paper_ids,
            key_points=key_points,
        ))

    # If no communities, return empty list (will trigger fallback)
    if not themes:
        logger.warning("No communities found for theme conversion")

    logger.info(f"Converted {len(themes)} communities to themes")
    return themes


def format_knowledge_graph_context(
    knowledge_graph: KnowledgeGraphData,
    max_entities: int = 20,
) -> str:
    """Format knowledge graph data for inclusion in prompts.

    Args:
        knowledge_graph: The knowledge graph data
        max_entities: Maximum entities to include

    Returns:
        Formatted string with key entities and relationships
    """
    lines = []

    # Global summary
    global_summary = knowledge_graph.get("global_summary", "")
    if global_summary:
        lines.append("## Knowledge Graph Summary")
        lines.append(global_summary)
        lines.append("")

    # Key entities by type
    entities = knowledge_graph.get("entities", {})
    if entities:
        lines.append("## Key Entities")

        # Group by type
        by_type: Dict[str, List[ExtractedEntity]] = {}
        for entity in entities.values():
            etype = entity.get("entity_type", "other")
            if etype not in by_type:
                by_type[etype] = []
            by_type[etype].append(entity)

        # Sort each group by frequency and take top entities
        entity_count = 0
        for etype in ["method", "dataset", "metric", "concept", "other"]:
            if etype not in by_type:
                continue
            sorted_entities = sorted(
                by_type[etype],
                key=lambda e: e.get("frequency", 0),
                reverse=True,
            )
            lines.append(f"\n### {etype.title()}s")
            for entity in sorted_entities[:5]:
                name = entity.get("name") or "Unknown"
                desc = (entity.get("description") or "")[:100]
                freq = entity.get("frequency") or 0
                lines.append(f"- **{name}** (freq: {freq}): {desc}")
                entity_count += 1
                if entity_count >= max_entities:
                    break
            if entity_count >= max_entities:
                break

    # Community summaries
    communities = knowledge_graph.get("communities") or []
    if communities:
        lines.append("\n## Research Clusters")
        for comm in communities[:5]:
            summary = comm.get("summary") or "Unnamed cluster"
            paper_count = len(comm.get("papers") or [])
            lines.append(f"- {summary} ({paper_count} papers)")

    return "\n".join(lines)


async def identify_themes(
    analyzed_papers: List[PaperSummary],
    research_question: str,
    model: Optional[str] = None,
    tracker=None,
) -> List[ThemeCluster]:
    """Identify major themes across analyzed papers.

    Uses LLM to find patterns and group papers into thematic clusters.

    Args:
        analyzed_papers: List of analyzed paper summaries
        research_question: The original research question
        model: LLM model to use

    Returns:
        List of identified theme clusters
    """
    if not analyzed_papers:
        return []

    paper_summaries = format_summaries_for_prompt(analyzed_papers)

    prompt = THEME_IDENTIFICATION_PROMPT.format(
        research_question=research_question,
        num_papers=len(analyzed_papers),
        paper_summaries=paper_summaries,
    )

    try:
        response = await call_llm(prompt, model=model, temperature=0.4, max_tokens=2000, tracker=tracker, agent_name="synthesis")

        # Parse JSON
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        themes_data = json.loads(response)

        if not isinstance(themes_data, list):
            raise ValueError("Expected JSON array of themes")

        themes = []
        for theme_dict in themes_data[:6]:  # Max 6 themes
            themes.append(ThemeCluster(
                theme=theme_dict.get("theme", "Unknown Theme"),
                description=theme_dict.get("description", ""),
                paper_ids=theme_dict.get("paper_ids", []),
                key_points=theme_dict.get("key_points", []),
            ))

        logger.info(f"Identified {len(themes)} themes across papers")
        return themes

    except (json.JSONDecodeError, LLMError) as e:
        logger.warning(f"Theme identification failed: {e}")
        # Fallback: create a single "General Findings" theme
        return [ThemeCluster(
            theme="General Findings",
            description="Synthesized findings from analyzed papers",
            paper_ids=[p["paper_id"] for p in analyzed_papers],
            key_points=[
                finding
                for p in analyzed_papers
                for finding in p.get("key_findings", [])[:2]
            ][:10],
        )]


async def analyze_gaps(
    analyzed_papers: List[PaperSummary],
    themes: List[ThemeCluster],
    research_question: str,
    model: Optional[str] = None,
    tracker=None,
) -> Dict[str, List[str]]:
    """Identify research gaps and future directions.

    Args:
        analyzed_papers: List of analyzed paper summaries
        themes: Identified themes
        research_question: The original research question
        model: LLM model to use

    Returns:
        Dict with "gaps" and "future_directions" lists
    """
    if not analyzed_papers:
        return {"gaps": [], "future_directions": []}

    paper_summaries = format_summaries_for_prompt(analyzed_papers)
    themes_text = "\n".join(
        f"- {t['theme']}: {t['description']}"
        for t in themes
    )

    prompt = GAP_ANALYSIS_PROMPT.format(
        research_question=research_question,
        paper_summaries=paper_summaries,
        themes=themes_text,
    )

    try:
        response = await call_llm(prompt, model=model, temperature=0.4, max_tokens=1000, tracker=tracker, agent_name="synthesis")

        # Parse JSON
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        result = json.loads(response)
        return {
            "gaps": result.get("gaps", [])[:5],
            "future_directions": result.get("future_directions", [])[:5],
        }

    except (json.JSONDecodeError, LLMError) as e:
        logger.warning(f"Gap analysis failed: {e}")
        # Fallback based on paper limitations
        limitations = []
        for paper in analyzed_papers:
            limitations.extend(paper.get("limitations", [])[:2])
        return {
            "gaps": limitations[:5] if limitations else ["Gap analysis not available"],
            "future_directions": ["Further research needed"],
        }


async def generate_review(
    analyzed_papers: List[PaperSummary],
    themes: List[ThemeCluster],
    gaps: List[str],
    research_question: str,
    review_type: str = "narrative",
    target_words: int = 2000,
    model: Optional[str] = None,
    language: str = "en",
    tracker=None,
) -> str:
    """Generate the literature review narrative.

    Args:
        analyzed_papers: List of analyzed paper summaries
        themes: Identified themes
        gaps: Research gaps
        research_question: The original research question
        review_type: Type of review (narrative, systematic, scoping)
        target_words: Target word count
        model: LLM model to use
        language: Output language code ("en", "zh", etc.)

    Returns:
        Generated review text
    """
    if not analyzed_papers:
        return "No papers available for literature review."

    paper_summaries = format_summaries_for_prompt(analyzed_papers)
    themes_text = "\n\n".join(
        f"**{t['theme']}**\n{t['description']}\nKey points:\n" +
        "\n".join(f"  - {p}" for p in t["key_points"][:5])
        for t in themes
    )
    gaps_text = "\n".join(f"- {g}" for g in gaps)

    prompt = LITERATURE_REVIEW_PROMPT.format(
        review_type=review_type,
        research_question=research_question,
        num_papers=len(analyzed_papers),
        paper_summaries=paper_summaries,
        themes=themes_text,
        gaps=gaps_text,
        word_count=target_words,
    ) + get_language_instruction(language)

    try:
        response = await call_llm(
            prompt,
            model=model,
            temperature=0.5,
            max_tokens=min(4000, target_words * 2),
            tracker=tracker,
            agent_name="synthesis",
        )
        logger.info(f"Generated review with {len(response.split())} words")
        return response.strip()

    except LLMError as e:
        logger.error(f"Review generation failed: {e}")
        # Fallback: create basic summary
        summary_parts = [
            f"# Literature Review: {research_question}\n\n",
            "## Summary\n\n",
            f"This review synthesizes findings from {len(analyzed_papers)} papers.\n\n",
            "## Key Findings\n\n",
        ]
        for paper in analyzed_papers[:5]:
            summary_parts.append(f"**{paper['title']}** ({paper['year']})\n")
            for finding in paper.get("key_findings", [])[:2]:
                summary_parts.append(f"- {finding}\n")
            summary_parts.append("\n")

        return "".join(summary_parts)


async def generate_graphrag_review(
    analyzed_papers: List[PaperSummary],
    themes: List[ThemeCluster],
    gaps: List[str],
    research_question: str,
    knowledge_graph_context: str,
    global_summary: str,
    review_type: str = "narrative",
    target_words: int = 2500,
    model: Optional[str] = None,
    language: str = "en",
    tracker=None,
) -> str:
    """Generate literature review enhanced with GraphRAG knowledge.

    This function uses the knowledge graph context (entities, relationships,
    community summaries) to produce a richer, more connected synthesis.

    Args:
        analyzed_papers: List of analyzed paper summaries
        themes: Theme clusters (derived from GraphRAG communities)
        gaps: Research gaps
        research_question: The original research question
        knowledge_graph_context: Formatted knowledge graph data
        global_summary: GraphRAG global summary
        review_type: Type of review (narrative, systematic, scoping)
        target_words: Target word count
        model: LLM model to use
        language: Output language code ("en", "zh", etc.)

    Returns:
        Generated review text with GraphRAG enhancement
    """
    if not analyzed_papers:
        return "No papers available for literature review."

    paper_summaries = format_summaries_for_prompt(analyzed_papers)
    themes_text = "\n\n".join(
        f"**{t['theme']}**\n{t['description']}\nKey points:\n"
        + "\n".join(f"  - {p}" for p in t["key_points"][:5])
        for t in themes
    )
    gaps_text = "\n".join(f"- {g}" for g in gaps)

    prompt = GRAPHRAG_LITERATURE_REVIEW_PROMPT.format(
        review_type=review_type,
        research_question=research_question,
        num_papers=len(analyzed_papers),
        paper_summaries=paper_summaries,
        themes=themes_text,
        gaps=gaps_text,
        knowledge_graph_context=knowledge_graph_context,
        global_summary=global_summary,
        word_count=target_words,
    ) + get_language_instruction(language)

    try:
        response = await call_llm(
            prompt,
            model=model,
            temperature=0.5,
            max_tokens=min(5000, target_words * 2),
            tracker=tracker,
            agent_name="synthesis",
        )
        logger.info(f"Generated GraphRAG-enhanced review with {len(response.split())} words")
        return response.strip()

    except LLMError as e:
        logger.error(f"GraphRAG review generation failed: {e}, falling back to standard")
        # Fallback to standard review
        return await generate_review(
            analyzed_papers=analyzed_papers,
            themes=themes,
            gaps=gaps,
            research_question=research_question,
            review_type=review_type,
            target_words=target_words,
            model=model,
            language=language,
            tracker=tracker,
        )


async def format_citations(
    analyzed_papers: List[PaperSummary],
    style: str = "APA",
    model: Optional[str] = None,
    tracker=None,
) -> List[str]:
    """Format paper citations in the specified style.

    Args:
        analyzed_papers: List of analyzed paper summaries
        style: Citation style (APA, MLA, Chicago, IEEE)
        model: LLM model to use

    Returns:
        List of formatted citations
    """
    if not analyzed_papers:
        return []

    # Prepare paper info for formatting
    papers_info = []
    for paper in analyzed_papers:
        authors = paper.get("authors", [])
        if isinstance(authors, list):
            authors_str = ", ".join(authors[:3])
            if len(paper.get("authors", [])) > 3:
                authors_str += " et al."
        else:
            authors_str = str(authors)

        info = {
            "title": paper.get("title", "Unknown"),
            "authors": authors_str,
            "year": paper.get("year", "n.d."),
            "venue": paper.get("venue", ""),
            "paper_id": paper.get("paper_id", ""),
        }
        papers_info.append(info)

    papers_text = "\n".join(
        f"- Title: {p['title']}\n  Authors: {p['authors']}\n  Year: {p['year']}\n  Venue: {p['venue']}\n  ID: {p['paper_id']}"
        for p in papers_info
    )

    prompt = CITATION_FORMAT_PROMPT.format(
        citation_style=style,
        papers=papers_text,
    )

    try:
        response = await call_llm(prompt, model=model, temperature=0.1, max_tokens=2000, tracker=tracker, agent_name="synthesis")
        citations = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return citations

    except LLMError as e:
        logger.warning(f"Citation formatting failed: {e}")
        # Fallback: simple format
        return [
            f"{p['authors']} ({p['year']}). {p['title']}. {p['venue']}."
            for p in papers_info
        ]


async def synthesis_agent(state: LitScribeState) -> Dict[str, Any]:
    """Main entry point for the Synthesis Agent.

    This function is called by the LangGraph workflow to generate
    the final literature review synthesis.

    Phase 7.5 Enhancement:
    - When knowledge_graph is available from GraphRAG, uses communities
      directly as themes (skipping redundant LLM theme identification)
    - Enriches review with entity relationships and community summaries

    Args:
        state: Current workflow state

    Returns:
        State updates with synthesis output
    """
    analyzed_papers = state.get("analyzed_papers", [])
    research_question = state["research_question"]
    review_type = state.get("review_type", "narrative")
    language = state.get("language", "en")
    errors = list(state.get("errors", []))
    llm_config = state.get("llm_config", {})
    model = llm_config.get("model")
    from utils.token_tracker import get_tracker
    tracker = get_tracker()

    # Phase 7.5: Check for GraphRAG knowledge graph
    knowledge_graph = state.get("knowledge_graph")
    use_graphrag = (
        knowledge_graph is not None
        and knowledge_graph.get("communities")
        and len(knowledge_graph.get("communities", [])) > 0
    )

    if not analyzed_papers:
        error_msg = "No analyzed papers available for synthesis"
        logger.warning(error_msg)
        errors.append(error_msg)
        return {
            "synthesis": SynthesisOutput(
                themes=[],
                gaps=["No papers analyzed"],
                future_directions=[],
                review_text="Unable to generate review - no papers analyzed.",
                citations_formatted=[],
                word_count=0,
                papers_cited=0,
            ),
            "errors": errors,
            "current_agent": "complete",
        }

    logger.info(f"Synthesis Agent starting: {len(analyzed_papers)} papers to synthesize")
    if use_graphrag:
        logger.info(
            f"GraphRAG enabled: {len(knowledge_graph.get('communities', []))} communities, "
            f"{len(knowledge_graph.get('entities', {}))} entities"
        )

    try:
        # Step 1: Identify themes (or use GraphRAG communities)
        kg_context = None
        if use_graphrag:
            # Deep integration: convert communities to themes directly
            themes = communities_to_themes(
                communities=knowledge_graph.get("communities", []),
                entities=knowledge_graph.get("entities", {}),
                analyzed_papers=analyzed_papers,
            )

            if themes:
                logger.info(f"Using {len(themes)} themes from GraphRAG communities")
                # Prepare knowledge graph context for review generation
                kg_context = format_knowledge_graph_context(knowledge_graph)
            else:
                # GraphRAG communities existed but produced no themes (e.g., all level > 1)
                # Fall back to LLM theme identification
                logger.warning("GraphRAG produced no usable themes, falling back to LLM")
                use_graphrag = False
                themes = await identify_themes(analyzed_papers, research_question, model, tracker=tracker)
        else:
            # Fallback: LLM-based theme identification
            themes = await identify_themes(analyzed_papers, research_question, model, tracker=tracker)
            kg_context = None

        # Step 2: Analyze gaps (enhanced with GraphRAG context if available)
        gap_analysis = await analyze_gaps(analyzed_papers, themes, research_question, model, tracker=tracker)
        gaps = gap_analysis["gaps"]
        future_directions = gap_analysis["future_directions"]

        # Step 3: Generate review (with GraphRAG enhancement if available)
        if use_graphrag and kg_context:
            review_text = await generate_graphrag_review(
                analyzed_papers=analyzed_papers,
                themes=themes,
                gaps=gaps,
                research_question=research_question,
                knowledge_graph_context=kg_context,
                global_summary=knowledge_graph.get("global_summary", ""),
                review_type=review_type,
                target_words=2500,  # Slightly longer for richer synthesis
                model=model,
                language=language,
                tracker=tracker,
            )
        else:
            review_text = await generate_review(
                analyzed_papers=analyzed_papers,
                themes=themes,
                gaps=gaps,
                research_question=research_question,
                review_type=review_type,
                target_words=2000,
                model=model,
                language=language,
                tracker=tracker,
            )

        # Step 4: Format citations
        citations = await format_citations(analyzed_papers, style="APA", model=model, tracker=tracker)

        # Build synthesis output
        synthesis = SynthesisOutput(
            themes=themes,
            gaps=gaps,
            future_directions=future_directions,
            review_text=review_text,
            citations_formatted=citations,
            word_count=len(review_text.split()),
            papers_cited=len(analyzed_papers),
        )

        # Step 5: Citation grounding check (Phase 9.5)
        grounding_report = None
        try:
            from analysis.citation_grounding import check_citation_grounding
            grounding_report = check_citation_grounding(review_text, analyzed_papers)
            if grounding_report["ungrounded_count"] > 0:
                logger.warning(
                    f"Citation grounding: {grounding_report['ungrounded_count']} ungrounded citations "
                    f"(rate={grounding_report['grounding_rate']:.2%})"
                )
            else:
                logger.info(
                    f"Citation grounding: {grounding_report['grounded_count']}/{grounding_report['total_citations']} "
                    f"citations grounded (100%)"
                )
        except Exception as e:
            logger.warning(f"Citation grounding check failed: {e}")

        mode_str = "GraphRAG-enhanced" if use_graphrag else "standard"
        logger.info(
            f"Synthesis complete ({mode_str}): {synthesis['word_count']} words, "
            f"{len(themes)} themes, {len(gaps)} gaps"
        )

        result = {
            "synthesis": synthesis,
            "errors": errors,
            "current_agent": "complete",
        }
        if grounding_report:
            result["_citation_grounding"] = grounding_report
        return result

    except Exception as e:
        error_msg = f"Synthesis Agent failed: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
        return {
            "synthesis": SynthesisOutput(
                themes=[],
                gaps=[error_msg],
                future_directions=[],
                review_text=f"Synthesis failed: {e}",
                citations_formatted=[],
                word_count=0,
                papers_cited=0,
            ),
            "errors": errors,
            "current_agent": "complete",
        }


# Export for use in graph.py
__all__ = [
    "synthesis_agent",
    "identify_themes",
    "analyze_gaps",
    "generate_review",
    "generate_graphrag_review",
    "format_citations",
    # GraphRAG integration helpers
    "communities_to_themes",
    "format_knowledge_graph_context",
]
