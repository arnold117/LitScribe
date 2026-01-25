"""State definitions for LitScribe multi-agent system.

This module defines the shared state schema used by all agents in the LangGraph workflow.
The state acts as a collaborative whiteboard where agents read and write information.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages


class PaperSummary(TypedDict):
    """Summary of an analyzed paper produced by Critical Reading Agent."""

    paper_id: str  # Unique identifier (DOI, arXiv ID, or S2 ID)
    title: str
    authors: List[str]
    year: int
    abstract: str

    # Analysis results
    key_findings: List[str]  # 3-5 key findings extracted from paper
    methodology: str  # Summary of research methodology
    strengths: List[str]  # Paper strengths
    limitations: List[str]  # Paper limitations

    # Metadata
    relevance_score: float  # 0-1, relevance to research question
    citations: int
    venue: str

    # Source tracking
    pdf_available: bool
    source: str  # "arxiv", "semantic_scholar", "pubmed", etc.


class SearchResult(TypedDict):
    """Result from Discovery Agent's search phase."""

    query: str  # Original research question
    expanded_queries: List[str]  # LLM-generated expanded queries
    papers: List[Dict[str, Any]]  # List of UnifiedPaper dicts
    source_counts: Dict[str, int]  # Papers found per source
    total_found: int
    search_timestamp: str


class ThemeCluster(TypedDict):
    """A thematic cluster identified across papers."""

    theme: str  # Theme name/title
    description: str  # What this theme covers
    paper_ids: List[str]  # Papers belonging to this theme
    key_points: List[str]  # Main points within this theme


class SynthesisOutput(TypedDict):
    """Output from Synthesis Agent."""

    # Thematic analysis
    themes: List[ThemeCluster]

    # Gap analysis
    gaps: List[str]  # Identified research gaps
    future_directions: List[str]  # Suggested future research

    # Generated review
    review_text: str  # The literature review narrative

    # Citations
    citations_formatted: List[str]  # Formatted citation list (APA/MLA)

    # Metadata
    word_count: int
    papers_cited: int


class LitScribeState(TypedDict):
    """Main state shared across all agents in the LitScribe workflow.

    This state follows the LangGraph pattern of using TypedDict with
    annotated reducers for managing state updates across agent nodes.

    Workflow:
    1. User provides research_question
    2. Discovery Agent: expands query, searches sources, selects papers
    3. Critical Reading Agent: parses PDFs, extracts findings, creates summaries
    4. Synthesis Agent: identifies themes, finds gaps, generates review
    """

    # === Core Input ===
    research_question: str  # The user's research question/topic

    # === Message History ===
    # Uses add_messages reducer to accumulate conversation history
    messages: Annotated[List, add_messages]

    # === Discovery Stage ===
    search_results: Optional[SearchResult]  # Results from multi-source search
    papers_to_analyze: List[Dict[str, Any]]  # Selected papers for analysis

    # === Critical Reading Stage ===
    analyzed_papers: List[PaperSummary]  # Completed paper summaries
    parsed_documents: Dict[str, Dict[str, Any]]  # paper_id -> ParsedDocument

    # === Synthesis Stage ===
    synthesis: Optional[SynthesisOutput]  # Final synthesis output

    # === Workflow Control ===
    current_agent: Literal[
        "supervisor",
        "discovery",
        "critical_reading",
        "synthesis",
        "complete"
    ]
    iteration_count: int  # Track iterations for loop control

    # === Error Tracking ===
    errors: List[str]  # Accumulated error messages

    # === Configuration ===
    max_papers: int  # Maximum papers to analyze
    sources: List[str]  # Sources to search ["arxiv", "semantic_scholar", "pubmed"]
    review_type: Literal["systematic", "narrative", "scoping"]  # Type of review

    # === LLM Configuration ===
    llm_config: Dict[str, Any]  # LLM settings passed to agents


def create_initial_state(
    research_question: str,
    max_papers: int = 10,
    sources: Optional[List[str]] = None,
    review_type: Literal["systematic", "narrative", "scoping"] = "narrative",
    llm_config: Optional[Dict[str, Any]] = None,
) -> LitScribeState:
    """Create an initial state for a new literature review workflow.

    Args:
        research_question: The research question to explore
        max_papers: Maximum number of papers to analyze (default: 10)
        sources: List of sources to search (default: arxiv, semantic_scholar)
        review_type: Type of literature review to generate
        llm_config: LLM configuration dict

    Returns:
        Initialized LitScribeState ready for the workflow
    """
    if sources is None:
        sources = ["arxiv", "semantic_scholar"]

    if llm_config is None:
        llm_config = {}

    return LitScribeState(
        research_question=research_question,
        messages=[],
        search_results=None,
        papers_to_analyze=[],
        analyzed_papers=[],
        parsed_documents={},
        synthesis=None,
        current_agent="supervisor",
        iteration_count=0,
        errors=[],
        max_papers=max_papers,
        sources=sources,
        review_type=review_type,
        llm_config=llm_config,
    )
