"""LangGraph workflow definition for LitScribe.

This module defines the main workflow graph that orchestrates
the multi-agent literature review process.
"""

import logging
from typing import Any, Dict, Literal, Optional

from langgraph.graph import END, StateGraph

from agents.critical_reading_agent import critical_reading_agent
from agents.discovery_agent import discovery_agent
from agents.state import LitScribeState, create_initial_state
from agents.supervisor import supervisor_agent
from agents.synthesis_agent import synthesis_agent

logger = logging.getLogger(__name__)


def should_continue(state: LitScribeState) -> Literal["discovery", "critical_reading", "synthesis", "complete"]:
    """Routing function for the conditional edge.

    Determines which node to visit next based on current_agent in state.

    Args:
        state: Current workflow state

    Returns:
        Name of next node to visit
    """
    current = state.get("current_agent", "complete")

    # Map state to graph node names
    if current == "discovery":
        return "discovery"
    elif current == "critical_reading":
        return "critical_reading"
    elif current == "synthesis":
        return "synthesis"
    else:
        return "complete"


def create_review_graph() -> StateGraph:
    """Create the LangGraph workflow for literature review.

    The workflow follows this structure:

    ```
    START
      |
      v
    supervisor --> discovery --> supervisor
                      |              |
                      v              v
              critical_reading --> synthesis --> END
    ```

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(LitScribeState)

    # Add nodes for each agent
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("discovery", discovery_agent)
    workflow.add_node("critical_reading", critical_reading_agent)
    workflow.add_node("synthesis", synthesis_agent)

    # Set the entry point
    workflow.set_entry_point("supervisor")

    # Add conditional edges based on supervisor's routing decision
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "discovery": "discovery",
            "critical_reading": "critical_reading",
            "synthesis": "synthesis",
            "complete": END,
        }
    )

    # After each agent completes, return to supervisor for next decision
    workflow.add_edge("discovery", "supervisor")
    workflow.add_edge("critical_reading", "supervisor")
    workflow.add_edge("synthesis", "supervisor")

    return workflow


def compile_graph(
    checkpointer: Optional[Any] = None,
    interrupt_before: Optional[list] = None,
    interrupt_after: Optional[list] = None,
) -> Any:
    """Compile the workflow graph with optional checkpointing.

    Args:
        checkpointer: Optional checkpointer for state persistence
        interrupt_before: Nodes to interrupt before (for human-in-the-loop)
        interrupt_after: Nodes to interrupt after

    Returns:
        Compiled graph ready for invocation
    """
    workflow = create_review_graph()

    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
    if interrupt_after:
        compile_kwargs["interrupt_after"] = interrupt_after

    return workflow.compile(**compile_kwargs)


async def run_literature_review(
    research_question: str,
    max_papers: int = 10,
    sources: Optional[list] = None,
    review_type: Literal["systematic", "narrative", "scoping"] = "narrative",
    llm_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a complete literature review workflow.

    This is the main entry point for running a literature review.
    It creates the initial state, compiles the graph, and runs
    the workflow to completion.

    Args:
        research_question: The research question to explore
        max_papers: Maximum number of papers to analyze
        sources: List of sources to search (default: arxiv, semantic_scholar)
        review_type: Type of literature review
        llm_config: LLM configuration options
        verbose: Whether to log progress

    Returns:
        Final state containing the literature review
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Create initial state
    initial_state = create_initial_state(
        research_question=research_question,
        max_papers=max_papers,
        sources=sources,
        review_type=review_type,
        llm_config=llm_config or {},
    )

    # Compile the graph
    graph = compile_graph()

    logger.info(f"Starting literature review for: {research_question}")

    # Run the workflow
    final_state = await graph.ainvoke(initial_state)

    logger.info("Literature review complete")

    return final_state


async def run_with_streaming(
    research_question: str,
    max_papers: int = 10,
    sources: Optional[list] = None,
    review_type: Literal["systematic", "narrative", "scoping"] = "narrative",
    llm_config: Optional[Dict[str, Any]] = None,
):
    """Run literature review with streaming updates.

    Yields state updates as the workflow progresses, useful for
    real-time progress tracking in UIs.

    Args:
        research_question: The research question
        max_papers: Maximum papers to analyze
        sources: Sources to search
        review_type: Type of review
        llm_config: LLM configuration

    Yields:
        State snapshots as workflow progresses
    """
    initial_state = create_initial_state(
        research_question=research_question,
        max_papers=max_papers,
        sources=sources,
        review_type=review_type,
        llm_config=llm_config or {},
    )

    graph = compile_graph()

    async for state in graph.astream(initial_state):
        yield state


# Convenience function for sync usage
def run_review_sync(
    research_question: str,
    max_papers: int = 10,
    sources: Optional[list] = None,
    review_type: Literal["systematic", "narrative", "scoping"] = "narrative",
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Synchronous wrapper for running literature review.

    Args:
        research_question: The research question
        max_papers: Maximum papers to analyze
        sources: Sources to search
        review_type: Type of review
        llm_config: LLM configuration

    Returns:
        Final state with literature review
    """
    import asyncio
    return asyncio.run(run_literature_review(
        research_question=research_question,
        max_papers=max_papers,
        sources=sources,
        review_type=review_type,
        llm_config=llm_config,
    ))


# Export
__all__ = [
    "create_review_graph",
    "compile_graph",
    "run_literature_review",
    "run_with_streaming",
    "run_review_sync",
]
