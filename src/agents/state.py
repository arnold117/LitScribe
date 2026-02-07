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


# === Phase 7.5: GraphRAG Types ===


class ExtractedEntity(TypedDict):
    """An entity extracted from a paper (method, dataset, metric, concept)."""

    entity_id: str  # Unique ID (normalized name hash)
    name: str  # Canonical name
    entity_type: str  # "method", "dataset", "metric", "concept"
    aliases: List[str]  # Alternative names/acronyms
    description: str  # Brief description
    paper_ids: List[str]  # Papers mentioning this entity
    frequency: int  # Total mentions across papers


class EntityMention(TypedDict):
    """A mention of an entity in a specific paper."""

    entity_id: str
    paper_id: str
    context: str  # Surrounding text
    section: str  # "abstract", "methods", "results", etc.
    confidence: float  # 0-1 extraction confidence


class GraphEdge(TypedDict):
    """An edge in the knowledge graph."""

    source_id: str
    target_id: str
    edge_type: str  # "cites", "uses_method", "evaluates_on", "co_occurs"
    weight: float  # Edge weight/strength
    paper_ids: List[str]  # Papers supporting this relationship


class Community(TypedDict):
    """A detected community in the knowledge graph."""

    community_id: str
    level: int  # Hierarchy level (0=top, higher=more specific)
    entities: List[str]  # Entity IDs in this community
    papers: List[str]  # Paper IDs in this community
    summary: str  # LLM-generated community summary
    parent_id: Optional[str]  # Parent community at higher level
    children_ids: List[str]  # Child communities


class KnowledgeGraphData(TypedDict):
    """Complete knowledge graph data for state storage."""

    entities: Dict[str, ExtractedEntity]  # entity_id -> entity
    mentions: List[EntityMention]
    edges: List[GraphEdge]
    communities: List[Community]
    global_summary: str  # Global synthesis from community summaries
    stats: Dict[str, Any]  # node_count, edge_count, etc.


class BatchProcessingState(TypedDict):
    """State for batch processing progress."""

    total_papers: int
    processed_papers: int
    current_batch: int
    total_batches: int
    batch_size: int
    batch_summaries: List[Dict[str, Any]]  # Sub-theme summaries per batch


class LitScribeState(TypedDict):
    """Main state shared across all agents in the LitScribe workflow.

    This state follows the LangGraph pattern of using TypedDict with
    annotated reducers for managing state updates across agent nodes.

    Workflow:
    1. User provides research_question
    2. Discovery Agent: expands query, searches sources, selects papers
    3. Critical Reading Agent: parses PDFs, extracts findings, creates summaries
    4. GraphRAG Agent (Phase 7.5): builds knowledge graph, detects communities
    5. Synthesis Agent: identifies themes, finds gaps, generates review
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

    # === GraphRAG Stage (Phase 7.5) ===
    knowledge_graph: Optional[KnowledgeGraphData]  # Knowledge graph data
    graphrag_enabled: bool  # Feature flag for GraphRAG processing
    batch_state: Optional[BatchProcessingState]  # Batch processing progress

    # === Synthesis Stage ===
    synthesis: Optional[SynthesisOutput]  # Final synthesis output
    technology_comparison: Optional[Dict[str, Any]]  # Tech comparison table

    # === Workflow Control ===
    current_agent: Literal[
        "supervisor",
        "discovery",
        "critical_reading",
        "graphrag",
        "synthesis",
        "complete"
    ]
    iteration_count: int  # Track iterations for loop control

    # === Error Tracking ===
    errors: List[str]  # Accumulated error messages

    # === Configuration ===
    max_papers: int  # Maximum papers to analyze (supports up to 500)
    sources: List[str]  # Sources to search ["arxiv", "semantic_scholar", "pubmed"]
    review_type: Literal["systematic", "narrative", "scoping"]  # Type of review
    cache_enabled: bool  # Whether to use local cache for search/PDF/parse
    batch_size: int  # Batch size for processing papers (default: 20)
    local_files: List[str]  # Local PDF file paths to include in review

    # === LLM Configuration ===
    llm_config: Dict[str, Any]  # LLM settings passed to agents


def create_initial_state(
    research_question: str,
    max_papers: int = 10,
    sources: Optional[List[str]] = None,
    review_type: Literal["systematic", "narrative", "scoping"] = "narrative",
    cache_enabled: bool = True,
    llm_config: Optional[Dict[str, Any]] = None,
    graphrag_enabled: bool = True,
    batch_size: int = 20,
    local_files: Optional[List[str]] = None,
) -> LitScribeState:
    """Create an initial state for a new literature review workflow.

    Args:
        research_question: The research question to explore
        max_papers: Maximum number of papers to analyze (default: 10, max: 500)
        sources: List of sources to search (default: arxiv, semantic_scholar, pubmed)
        review_type: Type of literature review to generate
        cache_enabled: Whether to use local cache (default: True)
        llm_config: LLM configuration dict
        graphrag_enabled: Enable GraphRAG knowledge graph (default: True)
        batch_size: Batch size for processing papers (default: 20)
        local_files: List of local PDF file paths to include in review

    Returns:
        Initialized LitScribeState ready for the workflow
    """
    if sources is None:
        sources = ["arxiv", "semantic_scholar", "pubmed"]

    if llm_config is None:
        llm_config = {}

    if local_files is None:
        local_files = []

    # Cap max_papers at 500
    max_papers = min(max_papers, 500)

    return LitScribeState(
        research_question=research_question,
        messages=[],
        search_results=None,
        papers_to_analyze=[],
        analyzed_papers=[],
        parsed_documents={},
        knowledge_graph=None,
        graphrag_enabled=graphrag_enabled,
        batch_state=None,
        synthesis=None,
        technology_comparison=None,
        current_agent="supervisor",
        iteration_count=0,
        errors=[],
        max_papers=max_papers,
        sources=sources,
        review_type=review_type,
        cache_enabled=cache_enabled,
        batch_size=batch_size,
        local_files=local_files,
        llm_config=llm_config,
    )
