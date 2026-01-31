"""Supervisor Agent for LitScribe.

The Supervisor orchestrates the workflow between agents, making routing
decisions based on the current state. It follows a simple linear workflow
but can be extended for more complex routing logic.
"""

import json
import logging
from typing import Any, Dict, Literal

from agents.prompts import WORKFLOW_STATUS_PROMPT
from agents.state import LitScribeState
from agents.tools import call_llm

logger = logging.getLogger(__name__)


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
) -> Literal["discovery", "critical_reading", "graphrag", "synthesis", "complete"]:
    """Determine which agent should run next based on state.

    Follows a linear workflow:
    1. Discovery: if no search results yet
    2. Critical Reading: if papers found but not analyzed
    3. GraphRAG: if papers analyzed but no knowledge graph (Phase 7.5)
    4. Synthesis: if GraphRAG done (or disabled) but no synthesis
    5. Complete: if synthesis done or errors prevent progress

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
    search_results = state.get("search_results")
    papers_to_analyze = state.get("papers_to_analyze", [])
    analyzed_papers = state.get("analyzed_papers", [])
    knowledge_graph = state.get("knowledge_graph")
    graphrag_enabled = state.get("graphrag_enabled", True)
    synthesis = state.get("synthesis")

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
    if analyzed_papers and synthesis is None:
        return "synthesis"

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

    return {
        "current_agent": next_agent,
        "iteration_count": iteration + 1,
    }


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
        response = await call_llm(prompt, temperature=0.1, max_tokens=300)

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
