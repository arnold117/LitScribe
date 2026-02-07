"""Self-Review Agent for LitScribe (Phase 9.1).

This agent runs after synthesis to assess review quality:
1. Relevance check - flags papers from unrelated fields
2. Coverage assessment - identifies missing subtopics
3. Coherence evaluation - checks narrative flow
4. Claim support - flags unsupported or weak claims

Design principles:
- Single LLM call for efficiency
- Never blocks the workflow (fallback on any failure)
- Assessment only in Phase 9.1 (no loop-back to discovery)
"""

import json
import logging
from typing import Any, Dict, List, Optional

from agents.errors import LLMError
from agents.prompts import (
    SELF_REVIEW_PROMPT,
    format_papers_for_self_review,
)
from agents.state import (
    LitScribeState,
    ReviewAssessment,
    SynthesisOutput,
)
from agents.tools import call_llm

logger = logging.getLogger(__name__)


def _build_review_summary(synthesis: SynthesisOutput, max_words: int = 3000) -> str:
    """Truncate review text to fit within prompt budget.

    Args:
        synthesis: SynthesisOutput with review_text
        max_words: Maximum word count for the summary

    Returns:
        Truncated review text
    """
    review_text = synthesis.get("review_text", "")
    words = review_text.split()
    if len(words) <= max_words:
        return review_text
    return " ".join(words[:max_words]) + f"\n\n[... truncated, {len(words)} words total]"


def _build_fallback_assessment(error_msg: str) -> ReviewAssessment:
    """Build a neutral fallback assessment when self-review fails.

    Returns neutral 0.5 scores so the workflow is never blocked.

    Args:
        error_msg: Error message to include in suggestions

    Returns:
        ReviewAssessment with neutral scores
    """
    return ReviewAssessment(
        overall_score=0.5,
        relevance_score=0.5,
        coverage_score=0.5,
        coherence_score=0.5,
        coverage_gaps=[],
        irrelevant_papers=[],
        weak_claims=[],
        suggestions=[f"Self-review could not complete: {error_msg}"],
        needs_additional_search=False,
        additional_queries=[],
    )


async def assess_review_quality(
    research_question: str,
    analyzed_papers: List[Dict[str, Any]],
    synthesis: SynthesisOutput,
    model: Optional[str] = None,
) -> ReviewAssessment:
    """Assess the quality of a generated literature review.

    Makes a single LLM call to evaluate relevance, coverage,
    coherence, and claim support.

    Args:
        research_question: The original research question
        analyzed_papers: List of analyzed paper summaries
        synthesis: The synthesis output to assess
        model: LLM model to use (optional)

    Returns:
        ReviewAssessment with scores and findings
    """
    # Build prompt inputs
    paper_list = format_papers_for_self_review(analyzed_papers, max_chars=6000)
    review_summary = _build_review_summary(synthesis, max_words=3000)

    themes = synthesis.get("themes", [])
    themes_text = "\n".join(
        f"- {t.get('theme', 'Unknown')}: {t.get('description', '')[:200]}"
        for t in themes
    )

    gaps = synthesis.get("gaps", [])
    gaps_text = "\n".join(f"- {g}" for g in gaps)

    prompt = SELF_REVIEW_PROMPT.format(
        research_question=research_question,
        num_papers=len(analyzed_papers),
        paper_list=paper_list,
        review_summary=review_summary,
        themes=themes_text or "None identified",
        gaps=gaps_text or "None identified",
    )

    response = await call_llm(prompt, model=model, temperature=0.2, max_tokens=2000)

    # Parse JSON response (same pattern as other agents)
    response = response.strip()
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
    response = response.strip()

    data = json.loads(response)

    # Build ReviewAssessment with caps on list lengths
    return ReviewAssessment(
        overall_score=float(data.get("overall_score", 0.5)),
        relevance_score=float(data.get("relevance_score", 0.5)),
        coverage_score=float(data.get("coverage_score", 0.5)),
        coherence_score=float(data.get("coherence_score", 0.5)),
        coverage_gaps=data.get("coverage_gaps", [])[:5],
        irrelevant_papers=data.get("irrelevant_papers", [])[:5],
        weak_claims=data.get("weak_claims", [])[:5],
        suggestions=data.get("suggestions", [])[:5],
        needs_additional_search=bool(data.get("needs_additional_search", False)),
        additional_queries=data.get("additional_queries", [])[:5],
    )


async def self_review_agent(state: LitScribeState) -> Dict[str, Any]:
    """Main entry point for the Self-Review Agent.

    Called by the LangGraph workflow after synthesis to assess
    the quality of the generated review.

    Always returns current_agent="complete" (no loop-back in Phase 9.1).
    Never raises — falls back to neutral assessment on any error.

    Args:
        state: Current workflow state

    Returns:
        State updates with self_review assessment
    """
    research_question = state["research_question"]
    synthesis = state.get("synthesis")
    analyzed_papers = state.get("analyzed_papers", [])
    errors = list(state.get("errors", []))

    logger.info("Self-Review Agent starting")

    # Skip if no synthesis to review
    if synthesis is None:
        logger.warning("No synthesis to review, skipping self-review")
        return {
            "self_review": _build_fallback_assessment("No synthesis available"),
            "current_agent": "complete",
        }

    try:
        assessment = await assess_review_quality(
            research_question=research_question,
            analyzed_papers=analyzed_papers,
            synthesis=synthesis,
        )

        # Log key findings
        irrelevant_count = len(assessment.get("irrelevant_papers", []))
        if irrelevant_count > 0:
            logger.warning(f"Self-review found {irrelevant_count} potentially irrelevant papers")
        logger.info(
            f"Self-review complete: overall={assessment['overall_score']:.2f}, "
            f"relevance={assessment['relevance_score']:.2f}, "
            f"coverage={assessment['coverage_score']:.2f}, "
            f"coherence={assessment['coherence_score']:.2f}"
        )

        return {
            "self_review": assessment,
            "current_agent": "complete",
        }

    except (json.JSONDecodeError, LLMError) as e:
        error_msg = f"Self-review failed: {e}"
        logger.warning(error_msg)
        errors.append(error_msg)
        return {
            "self_review": _build_fallback_assessment(str(e)),
            "errors": errors,
            "current_agent": "complete",
        }
    except Exception as e:
        error_msg = f"Self-review unexpected error: {e}"
        logger.warning(error_msg)
        errors.append(error_msg)
        return {
            "self_review": _build_fallback_assessment(str(e)),
            "errors": errors,
            "current_agent": "complete",
        }


__all__ = [
    "self_review_agent",
    "assess_review_quality",
]
