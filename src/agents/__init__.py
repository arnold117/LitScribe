"""LitScribe Multi-Agent System.

This package implements the multi-agent workflow for academic literature review:
- Discovery Agent: Find relevant papers across multiple sources
- Critical Reading Agent: Parse and analyze paper content
- Synthesis Agent: Generate literature review from findings
- Supervisor: Orchestrate the workflow between agents
"""

from agents.critical_reading_agent import (
    analyze_single_paper,
    critical_reading_agent,
    extract_key_findings,
)
from agents.discovery_agent import (
    discovery_agent,
    expand_queries,
    search_all_sources,
    select_papers,
    snowball_sampling,
)
from agents.graph import (
    compile_graph,
    create_review_graph,
    run_literature_review,
    run_review_sync,
    run_with_streaming,
)
from agents.state import (
    LitScribeState,
    PaperSummary,
    SearchResult,
    SynthesisOutput,
    ThemeCluster,
    create_initial_state,
)
from agents.supervisor import (
    determine_next_agent,
    supervisor_agent,
)
from agents.synthesis_agent import (
    analyze_gaps,
    format_citations,
    generate_review,
    identify_themes,
    synthesis_agent,
)

__all__ = [
    # State types
    "LitScribeState",
    "PaperSummary",
    "SearchResult",
    "SynthesisOutput",
    "ThemeCluster",
    "create_initial_state",
    # Discovery Agent
    "discovery_agent",
    "expand_queries",
    "search_all_sources",
    "select_papers",
    "snowball_sampling",
    # Critical Reading Agent
    "critical_reading_agent",
    "analyze_single_paper",
    "extract_key_findings",
    # Synthesis Agent
    "synthesis_agent",
    "identify_themes",
    "analyze_gaps",
    "generate_review",
    "format_citations",
    # Supervisor
    "supervisor_agent",
    "determine_next_agent",
    # Graph
    "create_review_graph",
    "compile_graph",
    "run_literature_review",
    "run_with_streaming",
    "run_review_sync",
]
