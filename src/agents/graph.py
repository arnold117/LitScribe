"""LangGraph workflow definition for LitScribe.

This module defines the main workflow graph that orchestrates
the multi-agent literature review process.

Supports SQLite checkpointing for breakpoint resume (Phase 6.5).
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from agents.critical_reading_agent import critical_reading_agent
from agents.discovery_agent import discovery_agent
from agents.self_review_agent import self_review_agent
from agents.state import LitScribeState, create_initial_state
from agents.supervisor import supervisor_agent
from agents.synthesis_agent import synthesis_agent
from cache.database import get_cache_db

# Note: graphrag_agent is imported inside create_review_graph() to avoid circular imports

logger = logging.getLogger(__name__)


def should_continue(state: LitScribeState) -> Literal["discovery", "critical_reading", "graphrag", "synthesis", "self_review", "complete"]:
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
    elif current == "graphrag":
        return "graphrag"
    elif current == "synthesis":
        return "synthesis"
    elif current == "self_review":
        return "self_review"
    else:
        return "complete"


def create_review_graph() -> StateGraph:
    """Create the LangGraph workflow for literature review.

    The workflow follows this structure (Phase 7.5 with GraphRAG):

    ```
    START
      |
      v
    supervisor --> discovery --> supervisor
                      |              |
                      v              v
              critical_reading --> graphrag --> synthesis --> END
    ```

    Returns:
        Compiled StateGraph ready for execution
    """
    # Lazy import to avoid circular dependency
    from graphrag.integration import graphrag_agent

    # Initialize the graph with our state schema
    workflow = StateGraph(LitScribeState)

    # Add nodes for each agent
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("discovery", discovery_agent)
    workflow.add_node("critical_reading", critical_reading_agent)
    workflow.add_node("graphrag", graphrag_agent)  # Phase 7.5
    workflow.add_node("synthesis", synthesis_agent)
    workflow.add_node("self_review", self_review_agent)  # Phase 9.1

    # Set the entry point
    workflow.set_entry_point("supervisor")

    # Add conditional edges based on supervisor's routing decision
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "discovery": "discovery",
            "critical_reading": "critical_reading",
            "graphrag": "graphrag",
            "synthesis": "synthesis",
            "self_review": "self_review",
            "complete": END,
        }
    )

    # After each agent completes, return to supervisor for next decision
    workflow.add_edge("discovery", "supervisor")
    workflow.add_edge("critical_reading", "supervisor")
    workflow.add_edge("graphrag", "supervisor")  # Phase 7.5
    workflow.add_edge("synthesis", "supervisor")
    workflow.add_edge("self_review", "supervisor")  # Phase 9.1

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


def get_checkpoint_db_path() -> str:
    """Get the path to the checkpoint database.

    Uses the same database as the cache system for consistency.

    Returns:
        Path to the SQLite database file
    """
    db = get_cache_db()
    return str(db.db_path)


async def run_literature_review(
    research_question: str,
    max_papers: int = 10,
    sources: Optional[list] = None,
    review_type: Literal["systematic", "narrative", "scoping"] = "narrative",
    cache_enabled: bool = True,
    llm_config: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    checkpoint_enabled: bool = True,
    verbose: bool = True,
    graphrag_enabled: bool = True,
    batch_size: int = 20,
    local_files: Optional[list] = None,
    language: str = "en",
) -> Dict[str, Any]:
    """Run a complete literature review workflow.

    This is the main entry point for running a literature review.
    It creates the initial state, compiles the graph, and runs
    the workflow to completion.

    Supports SQLite checkpointing for breakpoint resume.

    Args:
        research_question: The research question to explore
        max_papers: Maximum number of papers to analyze (supports up to 500)
        sources: List of sources to search (default: arxiv, semantic_scholar)
        review_type: Type of literature review
        cache_enabled: Whether to use local cache (default: True)
        llm_config: LLM configuration options
        thread_id: Optional thread ID for checkpointing (auto-generated if not provided)
        checkpoint_enabled: Whether to enable checkpointing (default: True)
        verbose: Whether to log progress
        graphrag_enabled: Whether to enable GraphRAG knowledge graph (default: True)
        batch_size: Batch size for processing papers (default: 20)
        local_files: List of local PDF file paths to include in review
        language: Output language for review text (default: "en")

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
        cache_enabled=cache_enabled,
        llm_config=llm_config or {},
        graphrag_enabled=graphrag_enabled,
        batch_size=batch_size,
        local_files=local_files or [],
        language=language,
    )

    logger.info(f"Starting literature review for: {research_question}")

    # Run with or without checkpointing
    if checkpoint_enabled:
        # Generate thread_id if not provided
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        logger.info(f"Checkpointing enabled with thread_id: {thread_id}")

        # Use async context manager for checkpointer
        db_path = get_checkpoint_db_path()
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            graph = compile_graph(checkpointer=checkpointer)
            final_state = await graph.ainvoke(initial_state, config)
    else:
        graph = compile_graph()
        final_state = await graph.ainvoke(initial_state)

    logger.info("Literature review complete")

    # Include thread_id in result for resume capability
    if checkpoint_enabled and thread_id:
        final_state["_thread_id"] = thread_id

    return final_state


async def resume_literature_review(
    thread_id: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Resume a literature review from a checkpoint.

    Args:
        thread_id: The thread ID of the review to resume
        verbose: Whether to log progress

    Returns:
        Final state containing the literature review
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    logger.info(f"Resuming literature review with thread_id: {thread_id}")

    db_path = get_checkpoint_db_path()
    config = {"configurable": {"thread_id": thread_id}}

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = compile_graph(checkpointer=checkpointer)

        # Get the current state from checkpoint
        state_snapshot = await graph.aget_state(config)

        if state_snapshot is None or state_snapshot.values is None:
            raise ValueError(f"No checkpoint found for thread_id: {thread_id}")

        logger.info(f"Resuming from state: current_agent={state_snapshot.values.get('current_agent')}")

        # Resume the workflow
        final_state = await graph.ainvoke(None, config)

    logger.info("Literature review resumed and completed")

    return final_state


async def get_review_state(thread_id: str) -> Optional[Dict[str, Any]]:
    """Get the current state of a literature review by thread_id.

    Args:
        thread_id: The thread ID of the review

    Returns:
        Current state dict or None if not found
    """
    db_path = get_checkpoint_db_path()
    config = {"configurable": {"thread_id": thread_id}}

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = compile_graph(checkpointer=checkpointer)
        state_snapshot = await graph.aget_state(config)

        if state_snapshot and state_snapshot.values:
            return dict(state_snapshot.values)
    return None


async def run_with_streaming(
    research_question: str,
    max_papers: int = 10,
    sources: Optional[list] = None,
    review_type: Literal["systematic", "narrative", "scoping"] = "narrative",
    cache_enabled: bool = True,
    llm_config: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    checkpoint_enabled: bool = True,
    graphrag_enabled: bool = True,
    batch_size: int = 20,
    language: str = "en",
):
    """Run literature review with streaming updates.

    Yields state updates as the workflow progresses, useful for
    real-time progress tracking in UIs.

    Supports SQLite checkpointing for breakpoint resume.

    Args:
        research_question: The research question
        max_papers: Maximum papers to analyze (supports up to 500)
        sources: Sources to search
        review_type: Type of review
        cache_enabled: Whether to use local cache
        llm_config: LLM configuration
        thread_id: Optional thread ID for checkpointing
        checkpoint_enabled: Whether to enable checkpointing
        graphrag_enabled: Whether to enable GraphRAG knowledge graph
        batch_size: Batch size for processing papers
        language: Output language for review text (default: "en")

    Yields:
        State snapshots as workflow progresses
    """
    initial_state = create_initial_state(
        research_question=research_question,
        max_papers=max_papers,
        sources=sources,
        review_type=review_type,
        cache_enabled=cache_enabled,
        llm_config=llm_config or {},
        graphrag_enabled=graphrag_enabled,
        batch_size=batch_size,
        language=language,
    )

    if checkpoint_enabled:
        # Generate thread_id if not provided
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        logger.info(f"Checkpointing enabled with thread_id: {thread_id}")

        db_path = get_checkpoint_db_path()
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            graph = compile_graph(checkpointer=checkpointer)
            async for state in graph.astream(initial_state, config):
                yield state
    else:
        graph = compile_graph()
        async for state in graph.astream(initial_state):
            yield state


# Convenience function for sync usage
def run_review_sync(
    research_question: str,
    max_papers: int = 10,
    sources: Optional[list] = None,
    review_type: Literal["systematic", "narrative", "scoping"] = "narrative",
    cache_enabled: bool = True,
    llm_config: Optional[Dict[str, Any]] = None,
    checkpoint_enabled: bool = True,
    graphrag_enabled: bool = True,
    batch_size: int = 20,
    language: str = "en",
) -> Dict[str, Any]:
    """Synchronous wrapper for running literature review.

    Args:
        research_question: The research question
        max_papers: Maximum papers to analyze (supports up to 500)
        sources: Sources to search
        review_type: Type of review
        cache_enabled: Whether to use local cache
        llm_config: LLM configuration
        checkpoint_enabled: Whether to enable checkpointing
        graphrag_enabled: Whether to enable GraphRAG knowledge graph
        batch_size: Batch size for processing papers
        language: Output language for review text (default: "en")

    Returns:
        Final state with literature review
    """
    import asyncio
    return asyncio.run(run_literature_review(
        research_question=research_question,
        max_papers=max_papers,
        sources=sources,
        review_type=review_type,
        cache_enabled=cache_enabled,
        llm_config=llm_config,
        checkpoint_enabled=checkpoint_enabled,
        graphrag_enabled=graphrag_enabled,
        batch_size=batch_size,
        language=language,
    ))


# Export
__all__ = [
    "create_review_graph",
    "compile_graph",
    "run_literature_review",
    "resume_literature_review",
    "get_review_state",
    "run_with_streaming",
    "run_review_sync",
    "get_checkpoint_db_path",
]
