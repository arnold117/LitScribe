from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from litscribe.models.plan import ResearchPlan
from litscribe.models.analysis import PaperAnalysis
from litscribe.models.review import ReviewOutput
from litscribe.models.assessment import ReviewAssessment
from litscribe.models.paper import Paper

logger = logging.getLogger(__name__)

MIN_PAPERS_FOR_SYNTHESIS = 6


@dataclass
class PipelineState:
    research_question: str = ""
    language: str = "en"
    plan: ResearchPlan | None = None
    papers: list[Paper] = field(default_factory=list)
    analyses: list[PaperAnalysis] = field(default_factory=list)
    graph: dict | None = None
    synthesis: ReviewOutput | None = None
    assessment: ReviewAssessment | None = None
    iteration: int = 0
    max_iterations: int = 3
    circuit_breaker_used: bool = False
    domain: str = "General"
    extra_queries: list[str] = field(default_factory=list)
    contradiction_report: Any = None


def determine_recommendation(state: PipelineState) -> str:
    if state.plan is None:
        return "DELEGATE to planner"

    if not state.papers:
        return "CALL discover_papers"

    if not state.analyses:
        return "DELEGATE to reader with the discovered papers"

    if len(state.analyses) < MIN_PAPERS_FOR_SYNTHESIS and not state.circuit_breaker_used:
        if state.iteration < state.max_iterations:
            return (
                f"CALL discover_papers with broader queries — only {len(state.analyses)} papers "
                f"(need {MIN_PAPERS_FOR_SYNTHESIS}), disable domain filter"
            )
        return "DELEGATE to synthesizer — few papers but retries exhausted"

    if state.assessment is not None:
        a = state.assessment
        needs_loop_back = (
            a.score < 0.65
            or (getattr(a, "needs_additional_search", False) and a.coverage_score < 0.7)
        )
        if needs_loop_back and state.iteration < state.max_iterations:
            refined = getattr(a, "refined_queries", []) or []
            q_hint = f" with refined queries: {refined[:3]}" if refined else ""
            return f"LOOP BACK: CALL discover_papers{q_hint}"
        return "COMPLETE"

    if state.synthesis is not None and state.assessment is None:
        return "DELEGATE to reviewer"

    if state.graph is None and len(state.analyses) >= 5:
        return "CALL build_knowledge_graph"

    if state.synthesis is None:
        return "DELEGATE to synthesizer"

    return "COMPLETE"


def check_status(state: PipelineState) -> dict:
    recommendation = determine_recommendation(state)

    logger.info(
        f"check_status: papers={len(state.papers)}, analyses={len(state.analyses)}, "
        f"iter={state.iteration}, rec={recommendation}"
    )

    return {
        "research_question": state.research_question,
        "papers_found": len(state.papers),
        "papers_analyzed": len(state.analyses),
        "has_plan": state.plan is not None,
        "has_graph": state.graph is not None,
        "has_synthesis": state.synthesis is not None,
        "has_review": state.assessment is not None,
        "review_score": state.assessment.score if state.assessment else None,
        "coverage_score": state.assessment.coverage_score if state.assessment else None,
        "iteration": state.iteration,
        "max_iterations": state.max_iterations,
        "domain": state.domain,
        "recommendation": recommendation,
    }
