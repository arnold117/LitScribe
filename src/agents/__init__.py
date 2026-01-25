"""LitScribe Multi-Agent System.

This package implements the multi-agent workflow for academic literature review:
- Discovery Agent: Find relevant papers across multiple sources
- Critical Reading Agent: Parse and analyze paper content
- Synthesis Agent: Generate literature review from findings
"""

from agents.state import (
    LitScribeState,
    PaperSummary,
    SearchResult,
    SynthesisOutput,
    ThemeCluster,
    create_initial_state,
)

__all__ = [
    "LitScribeState",
    "PaperSummary",
    "SearchResult",
    "SynthesisOutput",
    "ThemeCluster",
    "create_initial_state",
]
