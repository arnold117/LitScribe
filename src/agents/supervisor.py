"""Supervisor Agent for LitScribe.

The Supervisor orchestrates the workflow between agents, making routing
decisions based on the current state. It follows a simple linear workflow
but can be extended for more complex routing logic.
"""

import json
import logging
from typing import Any, Dict, Literal

from agents.prompts import LOOPBACK_QUERY_REFINEMENT_PROMPT, WORKFLOW_STATUS_PROMPT
from agents.state import LitScribeState
from agents.tools import call_llm, call_llm_for_json

logger = logging.getLogger(__name__)

# Minimum papers required before synthesis; fewer triggers circuit breaker loop-back
MIN_PAPERS_FOR_SYNTHESIS = 6


def get_workflow_status(state: LitScribeState) -> Dict[str, Any]:
    """Analyze current workflow state.

    Args:
        state: Current LitScribeState

    Returns:
        Status summary dict
    """
    search_completed = state.get("search_results") is not None
    papers_found = len(state.get("papers_to_analyze", []))
    papers_analyzed = len(state.get("analyzed_papers", []))
    graphrag_completed = state.get("knowledge_graph") is not None
    synthesis_generated = state.get("synthesis") is not None
    errors = state.get("errors", [])

    return {
        "search_completed": search_completed,
        "papers_found": papers_found,
        "papers_analyzed": papers_analyzed,
        "graphrag_completed": graphrag_completed,
        "graphrag_enabled": state.get("graphrag_enabled", True),
        "synthesis_generated": synthesis_generated,
        "error_count": len(errors),
        "latest_errors": errors[-3:] if errors else [],
    }


def determine_next_agent(
    state: LitScribeState,
) -> Literal["planning", "discovery", "critical_reading", "graphrag", "synthesis", "self_review", "complete"]:
    """Determine which agent should run next based on state.

    Follows a linear workflow:
    0. Planning: if no research plan yet (Phase 9.2)
    1. Discovery: if no search results yet
    2. Critical Reading: if papers found but not analyzed
    3. GraphRAG: if papers analyzed but no knowledge graph (Phase 7.5)
    4. Synthesis: if GraphRAG done (or disabled) but no synthesis
    5. Self-Review: if synthesis done but no self-review (Phase 9.1)
    6. Complete: if self-review done or errors prevent progress

    Args:
        state: Current workflow state

    Returns:
        Name of next agent to run
    """
    current = state.get("current_agent", "supervisor")
    errors = state.get("errors", [])

    # Check for fatal errors (too many errors)
    if len(errors) > 10:
        logger.warning("Too many errors, ending workflow")
        return "complete"

    # Check iteration limit
    iteration = state.get("iteration_count", 0)
    if iteration > 10:  # Increased from 5 to accommodate GraphRAG
        logger.warning("Iteration limit reached, ending workflow")
        return "complete"

    # State-based routing
    research_plan = state.get("research_plan")
    search_results = state.get("search_results")
    papers_to_analyze = state.get("papers_to_analyze", [])
    analyzed_papers = state.get("analyzed_papers", [])
    knowledge_graph = state.get("knowledge_graph")
    graphrag_enabled = state.get("graphrag_enabled", True)
    synthesis = state.get("synthesis")

    # Phase 0: Need to plan (Phase 9.2)
    if research_plan is None:
        return "planning"

    # Phase 1: Need to discover papers
    if search_results is None:
        return "discovery"

    # Phase 2: Have papers but haven't analyzed them all
    if papers_to_analyze and len(analyzed_papers) < len(papers_to_analyze):
        return "critical_reading"

    # Phase 3: GraphRAG (if enabled and not done)
    if graphrag_enabled and analyzed_papers and knowledge_graph is None:
        return "graphrag"

    # Phase 4: All papers analyzed (and GraphRAG done if enabled), need synthesis
    # Circuit breaker: if too few papers, try broadened discovery before writing a bad review
    if analyzed_papers and synthesis is None:
        already_retried = state.get("_circuit_breaker_retried", False)
        if (
            len(analyzed_papers) < MIN_PAPERS_FOR_SYNTHESIS
            and state.get("sources")
            and iteration < 8
            and not already_retried
        ):
            logger.warning(
                f"Circuit breaker: only {len(analyzed_papers)} papers for synthesis "
                f"(minimum {MIN_PAPERS_FOR_SYNTHESIS}), routing back to discovery with relaxed filters"
            )
            return "discovery"
        if len(analyzed_papers) < MIN_PAPERS_FOR_SYNTHESIS:
            logger.warning(
                f"Circuit breaker: only {len(analyzed_papers)} papers for synthesis "
                f"(minimum {MIN_PAPERS_FOR_SYNTHESIS}), but retries exhausted — proceeding with warning"
            )
        return "synthesis"

    # Phase 5: Synthesis done but self-review not done (Phase 9.1)
    if synthesis is not None and state.get("self_review") is None:
        return "self_review"

    # Phase 5b: Self-review requested loop-back to discovery
    # Trigger when: (LLM requests more search AND coverage is genuinely low) OR overall score is poor
    self_review = state.get("self_review")
    if (
        self_review is not None
        and ((self_review.get("needs_additional_search", False)
              and self_review.get("coverage_score", 1.0) < 0.7)
             or self_review.get("overall_score", 1.0) < 0.65)
        and state.get("sources")  # Has online sources
        and iteration < 8
    ):
        # Reset self_review and synthesis so they re-run after discovery
        return "discovery"

    # All done
    return "complete"


async def supervisor_agent(state: LitScribeState) -> Dict[str, Any]:
    """Main entry point for the Supervisor Agent.

    The supervisor analyzes the current state and decides which
    agent should run next in the workflow.

    Args:
        state: Current workflow state

    Returns:
        State updates with routing decision
    """
    research_question = state.get("research_question", "")
    current_agent = state.get("current_agent", "supervisor")
    iteration = state.get("iteration_count", 0)

    logger.info(f"Supervisor: iteration {iteration}, current agent: {current_agent}")

    # Get workflow status
    status = get_workflow_status(state)

    # Determine next agent
    next_agent = determine_next_agent(state)

    graphrag_status = "done" if status["graphrag_completed"] else ("pending" if status["graphrag_enabled"] else "disabled")
    logger.info(
        f"Supervisor routing: {current_agent} -> {next_agent} "
        f"(papers: {status['papers_found']}/{status['papers_analyzed']}, "
        f"graphrag: {graphrag_status}, errors: {status['error_count']})"
    )

    updates: Dict[str, Any] = {
        "current_agent": next_agent,
        "iteration_count": iteration + 1,
    }

    # Circuit breaker: too few papers for synthesis, route back to discovery with relaxed filters
    if (
        next_agent == "discovery"
        and state.get("synthesis") is None
        and state.get("self_review") is None
        and state.get("analyzed_papers")
    ):
        logger.info(
            f"Circuit breaker: {len(state.get('analyzed_papers', []))} papers < {MIN_PAPERS_FOR_SYNTHESIS}, "
            f"resetting for broader search"
        )
        updates["_circuit_breaker_retried"] = True
        updates["search_results"] = None  # Force full re-search
        updates["knowledge_graph"] = None
        updates["disable_domain_filter"] = True  # Broaden search by removing domain restrictions

    # When looping back to discovery from self-review, use incremental strategy:
    # keep high-relevance papers, clear downstream, inject additional_queries
    self_review = state.get("self_review")
    if (
        next_agent == "discovery"
        and self_review is not None
        and state.get("synthesis") is not None
    ):
        analyzed = state.get("analyzed_papers", [])
        keep = [p for p in analyzed if p.get("relevance_score", 0) >= 0.5]
        keep_ids = {p.get("paper_id") for p in keep if p.get("paper_id")}

        logger.info(
            f"Incremental loop-back: keeping {len(keep)}/{len(analyzed)} "
            f"high-relevance papers, clearing downstream state"
        )
        updates["synthesis"] = None
        updates["self_review"] = None
        updates["knowledge_graph"] = None
        updates["analyzed_papers"] = keep
        updates["papers_to_analyze"] = [
            p for p in state.get("papers_to_analyze", [])
            if (p.get("paper_id") or p.get("arxiv_id") or p.get("doi")) in keep_ids
        ]
        updates["parsed_documents"] = {
            k: v for k, v in state.get("parsed_documents", {}).items()
            if k in keep_ids
        }
        # Refine additional_queries: use LLM to combine coverage_gaps + plan context
        # into precise Boolean queries, falling back to self-review's raw queries on error
        extra_queries = self_review.get("additional_queries", [])
        coverage_gaps = self_review.get("coverage_gaps", [])
        research_plan = state.get("research_plan")
        plan_subtopics = research_plan.get("sub_topics", []) if research_plan else []

        if coverage_gaps and plan_subtopics:
            try:
                from utils.token_tracker import get_tracker
                tracker = get_tracker()
                subtopics_text = "\n".join(
                    f"- {t.get('name', '?')}: {t.get('description', '')[:150]}"
                    for t in plan_subtopics if t.get("selected", True)
                )
                gaps_text = "\n".join(f"- {g}" for g in coverage_gaps)
                initial_text = "\n".join(f"- {q}" for q in extra_queries) if extra_queries else "None"

                prompt = LOOPBACK_QUERY_REFINEMENT_PROMPT.format(
                    research_question=state.get("research_question", ""),
                    plan_subtopics=subtopics_text,
                    coverage_gaps=gaps_text,
                    initial_queries=initial_text,
                )
                refined = await call_llm_for_json(
                    prompt, temperature=0.3, max_tokens=800,
                    tracker=tracker, agent_name="discovery",
                )
                if isinstance(refined, list) and refined:
                    logger.info(
                        f"Loop-back query refinement: {len(refined)} refined queries "
                        f"(was {len(extra_queries)} from self-review)"
                    )
                    extra_queries = refined
            except Exception as e:
                logger.warning(f"Loop-back query refinement failed ({e}), using self-review queries")

        if extra_queries:
            logger.info(f"Injecting {len(extra_queries)} additional queries for discovery")
            updates["additional_queries"] = extra_queries

    return updates


async def supervisor_with_llm(state: LitScribeState) -> Dict[str, Any]:
    """Enhanced supervisor that uses LLM for complex routing decisions.

    Use this version when you need more sophisticated routing logic
    that considers the content of errors, paper quality, etc.

    Args:
        state: Current workflow state

    Returns:
        State updates with routing decision
    """
    status = get_workflow_status(state)

    prompt = WORKFLOW_STATUS_PROMPT.format(
        research_question=state.get("research_question", ""),
        search_completed=status["search_completed"],
        papers_found=status["papers_found"],
        papers_analyzed=status["papers_analyzed"],
        synthesis_generated=status["synthesis_generated"],
        errors=", ".join(status["latest_errors"]) if status["latest_errors"] else "None",
        iteration_count=state.get("iteration_count", 0),
    )

    try:
        from utils.token_tracker import get_tracker
        tracker = get_tracker()
        response = await call_llm(prompt, temperature=0.1, max_tokens=300, tracker=tracker, agent_name="supervisor")

        # Parse JSON response
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        result = json.loads(response)
        next_agent = result.get("next_agent", "complete")

        # Validate next_agent
        valid_agents = {"discovery", "critical_reading", "synthesis", "complete"}
        if next_agent not in valid_agents:
            next_agent = determine_next_agent(state)

        logger.info(f"LLM Supervisor decision: {next_agent}, reason: {result.get('reasoning', 'N/A')}")

        return {
            "current_agent": next_agent,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    except Exception as e:
        logger.warning(f"LLM supervisor failed, falling back to rule-based: {e}")
        return await supervisor_agent(state)


# Export
__all__ = [
    "supervisor_agent",
    "supervisor_with_llm",
    "determine_next_agent",
    "get_workflow_status",
]
