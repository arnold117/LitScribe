"""Synthesis Agent for LitScribe.

This agent is responsible for:
1. Theme identification - finding common themes across papers
2. Gap analysis - identifying research gaps and future directions
3. Literature review generation - writing the final review narrative
4. Citation formatting - generating proper citations
"""

import json
import logging
from typing import Any, Dict, List, Optional

from agents.errors import LLMError
from agents.prompts import (
    CITATION_FORMAT_PROMPT,
    GAP_ANALYSIS_PROMPT,
    LITERATURE_REVIEW_PROMPT,
    THEME_IDENTIFICATION_PROMPT,
    format_summaries_for_prompt,
)
from agents.state import LitScribeState, PaperSummary, SynthesisOutput, ThemeCluster
from agents.tools import call_llm

logger = logging.getLogger(__name__)


async def identify_themes(
    analyzed_papers: List[PaperSummary],
    research_question: str,
    model: Optional[str] = None,
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
        response = await call_llm(prompt, model=model, temperature=0.4, max_tokens=2000)

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
        response = await call_llm(prompt, model=model, temperature=0.4, max_tokens=1000)

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
    )

    try:
        response = await call_llm(
            prompt,
            model=model,
            temperature=0.5,
            max_tokens=min(4000, target_words * 2),
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


async def format_citations(
    analyzed_papers: List[PaperSummary],
    style: str = "APA",
    model: Optional[str] = None,
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
        response = await call_llm(prompt, model=model, temperature=0.1, max_tokens=2000)
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

    Args:
        state: Current workflow state

    Returns:
        State updates with synthesis output
    """
    analyzed_papers = state.get("analyzed_papers", [])
    research_question = state["research_question"]
    review_type = state.get("review_type", "narrative")
    errors = list(state.get("errors", []))
    llm_config = state.get("llm_config", {})
    model = llm_config.get("model")

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

    try:
        # Step 1: Identify themes
        themes = await identify_themes(analyzed_papers, research_question, model)

        # Step 2: Analyze gaps
        gap_analysis = await analyze_gaps(analyzed_papers, themes, research_question, model)
        gaps = gap_analysis["gaps"]
        future_directions = gap_analysis["future_directions"]

        # Step 3: Generate review
        review_text = await generate_review(
            analyzed_papers=analyzed_papers,
            themes=themes,
            gaps=gaps,
            research_question=research_question,
            review_type=review_type,
            target_words=2000,
            model=model,
        )

        # Step 4: Format citations
        citations = await format_citations(analyzed_papers, style="APA", model=model)

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

        logger.info(
            f"Synthesis complete: {synthesis['word_count']} words, "
            f"{len(themes)} themes, {len(gaps)} gaps"
        )

        return {
            "synthesis": synthesis,
            "errors": errors,
            "current_agent": "complete",
        }

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
    "format_citations",
]
