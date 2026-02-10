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
    from agents.synthesis_agent import count_words
    wc = count_words(review_text)
    if wc <= max_words:
        return review_text
    # For CJK text, truncate by characters (~1 char ≈ 1 word-equivalent)
    max_chars = max_words * 2  # rough ratio for mixed CJK/Latin
    truncated = review_text[:max_chars]
    return truncated + f"\n\n[... truncated, {wc} words total]"


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
    tracker=None,
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

    response = await call_llm(prompt, model=model, temperature=0.2, max_tokens=2000, tracker=tracker, agent_name="self_review", task_type="self_review")

    # Parse JSON response (robust extraction for reasoning models)
    from agents.tools import extract_json
    data = extract_json(response)

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

    # Skip if disabled (Phase 9.5 ablation)
    if state.get("disable_self_review", False):
        logger.info("Self-review disabled (ablation mode), skipping")
        return {
            "self_review": _build_fallback_assessment("Self-review disabled (ablation)"),
            "current_agent": "complete",
        }

    # Skip if no synthesis to review
    if synthesis is None:
        logger.warning("No synthesis to review, skipping self-review")
        return {
            "self_review": _build_fallback_assessment("No synthesis available"),
            "current_agent": "complete",
        }

    sources = state.get("sources", [])
    iteration_count = state.get("iteration_count", 0)
    llm_config = state.get("llm_config", {})
    model = llm_config.get("model")
    from utils.token_tracker import get_tracker
    tracker = get_tracker()

    try:
        assessment = await assess_review_quality(
            research_question=research_question,
            analyzed_papers=analyzed_papers,
            synthesis=synthesis,
            model=model,
            tracker=tracker,
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

        updates: Dict[str, Any] = {
            "self_review": assessment,
            "current_agent": "complete",
        }

        # Step 4b: Filter out irrelevant papers from analyzed_papers
        irrelevant_ids = {
            p.get("paper_id") for p in assessment.get("irrelevant_papers", [])
        }
        if irrelevant_ids:
            cleaned_papers = [
                p for p in analyzed_papers
                if p.get("paper_id") not in irrelevant_ids
            ]
            removed = len(analyzed_papers) - len(cleaned_papers)
            if removed > 0:
                logger.info(f"Self-review: removed {removed} irrelevant papers from analyzed_papers")
                updates["analyzed_papers"] = cleaned_papers
                # Sync papers_to_analyze so supervisor doesn't route back to critical_reading
                cleaned_ids = {p.get("paper_id") for p in cleaned_papers}
                papers_to_analyze = state.get("papers_to_analyze", [])
                updates["papers_to_analyze"] = [
                    p for p in papers_to_analyze
                    if (p.get("paper_id") or p.get("arxiv_id") or p.get("doi")) in cleaned_ids
                ]

        # Step 4c: Loop-back if quality is low and online search is available
        if (
            assessment.get("overall_score", 1.0) < 0.6
            and assessment.get("needs_additional_search", False)
            and iteration_count < 8
        ):
            if sources:
                # Online mode: loop back to discovery for more papers
                logger.info("Self-review: low quality score, routing back to discovery")
                updates["current_agent"] = "discovery"
            else:
                # Local-only mode: flag for CLI to handle
                logger.info("Self-review: low quality score in local-only mode, setting quality warning")
                updates["_quality_warning"] = True

        return updates

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
