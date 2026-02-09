"""Tool wrappers for LitScribe agents.

This module wraps the existing MCP server functions as LangChain-compatible tools
that can be used by LangGraph agents. Each tool handles error conversion and
rate limiting appropriately.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.errors import (
    AgentError,
    ErrorType,
    LLMError,
    PDFNotFoundError,
    PDFParseError,
    RateLimitError,
    retry_on_rate_limit,
    semantic_scholar_limiter,
)


# =============================================================================
# Discovery Tools - For finding papers across sources
# =============================================================================

@retry_on_rate_limit(max_attempts=3)
async def unified_search(
    query: str,
    sources: Optional[List[str]] = None,
    max_per_source: int = 10,
    sort_by: str = "relevance",
    deduplicate: bool = True,
    arxiv_categories: Optional[List[str]] = None,
    s2_fields: Optional[List[str]] = None,
    pubmed_mesh: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Search multiple academic sources and return deduplicated results.

    Args:
        query: Search query string
        sources: List of sources to search (default: arxiv, semantic_scholar)
        max_per_source: Maximum results per source
        sort_by: Sort order (relevance, citations, year, completeness)
        deduplicate: Whether to deduplicate results
        arxiv_categories: arXiv category filters
        s2_fields: Semantic Scholar field filters
        pubmed_mesh: PubMed MeSH term filters

    Returns:
        Dictionary with papers, source_counts, and metadata
    """
    if sources is None:
        sources = ["arxiv", "semantic_scholar", "pubmed"]

    try:
        from aggregators.unified_search import search_all_sources
        return await search_all_sources(
            query=query,
            sources=sources,
            max_per_source=max_per_source,
            deduplicate=deduplicate,
            sort_by=sort_by,
            arxiv_categories=arxiv_categories,
            s2_fields=s2_fields,
            pubmed_mesh=pubmed_mesh,
        )
    except Exception as e:
        if "rate limit" in str(e).lower():
            raise RateLimitError(str(e), original_error=e)
        raise AgentError(ErrorType.NETWORK_ERROR, str(e), original_error=e)


@retry_on_rate_limit(max_attempts=3)
async def search_semantic_scholar(
    query: str,
    limit: int = 20,
    year: Optional[str] = None,
    fields_of_study: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Search Semantic Scholar for papers.

    Args:
        query: Search query string
        limit: Maximum number of results
        year: Filter by year or year range (e.g., "2020", "2020-2024")
        fields_of_study: Filter by fields

    Returns:
        Dictionary with search results
    """
    async with semantic_scholar_limiter:
        try:
            from services.semantic_scholar import search_papers
            return await search_papers(
                query=query,
                limit=limit,
                year=year,
                fields_of_study=fields_of_study,
            )
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                raise RateLimitError(str(e), original_error=e)
            raise AgentError(ErrorType.NETWORK_ERROR, str(e), original_error=e)


@retry_on_rate_limit(max_attempts=3)
async def get_paper_details(paper_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific paper.

    Args:
        paper_id: Paper identifier (DOI, arXiv:XXXX, PMID:XXXX, or S2 ID)

    Returns:
        Dictionary with paper details including abstract, citations, references
    """
    async with semantic_scholar_limiter:
        try:
            from services.semantic_scholar import get_paper
            return await get_paper(paper_id)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                raise RateLimitError(str(e), original_error=e)
            raise AgentError(ErrorType.NETWORK_ERROR, str(e), original_error=e)


@retry_on_rate_limit(max_attempts=3)
async def get_paper_citations(paper_id: str, limit: int = 20) -> Dict[str, Any]:
    """Get papers that cite the specified paper.

    Args:
        paper_id: Paper identifier
        limit: Maximum number of citations to return

    Returns:
        Dictionary with citing papers
    """
    async with semantic_scholar_limiter:
        try:
            from services.semantic_scholar import get_paper_citations as s2_citations
            return await s2_citations(paper_id, limit=limit)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                raise RateLimitError(str(e), original_error=e)
            raise AgentError(ErrorType.NETWORK_ERROR, str(e), original_error=e)


@retry_on_rate_limit(max_attempts=3)
async def get_paper_references(paper_id: str, limit: int = 20) -> Dict[str, Any]:
    """Get papers referenced by the specified paper.

    Args:
        paper_id: Paper identifier
        limit: Maximum number of references to return

    Returns:
        Dictionary with referenced papers
    """
    async with semantic_scholar_limiter:
        try:
            from services.semantic_scholar import get_paper_references as s2_refs
            return await s2_refs(paper_id, limit=limit)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                raise RateLimitError(str(e), original_error=e)
            raise AgentError(ErrorType.NETWORK_ERROR, str(e), original_error=e)


@retry_on_rate_limit(max_attempts=3)
async def get_recommendations(paper_id: str, limit: int = 10) -> Dict[str, Any]:
    """Get recommended papers similar to the specified paper.

    Args:
        paper_id: Paper identifier
        limit: Maximum number of recommendations

    Returns:
        Dictionary with recommended papers
    """
    async with semantic_scholar_limiter:
        try:
            from services.semantic_scholar import get_recommendations as s2_recs
            return await s2_recs(paper_id, limit=limit)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                raise RateLimitError(str(e), original_error=e)
            raise AgentError(ErrorType.NETWORK_ERROR, str(e), original_error=e)


# =============================================================================
# Critical Reading Tools - For PDF acquisition and parsing
# =============================================================================

async def download_arxiv_pdf(arxiv_id: str) -> Dict[str, Any]:
    """Download PDF for an arXiv paper.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2401.00001")

    Returns:
        Dictionary with local file path
    """
    try:
        from services.arxiv import download_pdf
        return await download_pdf(arxiv_id)
    except Exception as e:
        raise PDFNotFoundError(
            f"Failed to download arXiv PDF {arxiv_id}: {e}",
            original_error=e,
            context={"arxiv_id": arxiv_id}
        )


async def get_zotero_pdf_path(item_key: str) -> Dict[str, Any]:
    """Get local PDF path for a Zotero item.

    Args:
        item_key: Zotero item key

    Returns:
        Dictionary with PDF path if available
    """
    try:
        from services.zotero import get_item_pdf_path
        return await get_item_pdf_path(item_key)
    except Exception as e:
        raise PDFNotFoundError(
            f"Failed to get Zotero PDF path for {item_key}: {e}",
            original_error=e,
            context={"item_key": item_key}
        )


async def parse_pdf(pdf_path: str, backend: str = "pymupdf") -> Dict[str, Any]:
    """Parse a PDF file into structured markdown.

    Args:
        pdf_path: Path to the PDF file
        backend: Parser backend ("pymupdf" or "marker")

    Returns:
        Dictionary with markdown, sections, tables, references
    """
    try:
        from services.pdf_parser import parse_pdf as pdf_parse
        return await pdf_parse(pdf_path, backend=backend)
    except Exception as e:
        raise PDFParseError(
            f"Failed to parse PDF {pdf_path}: {e}",
            original_error=e,
            context={"pdf_path": pdf_path, "backend": backend}
        )


async def extract_pdf_section(pdf_path: str, section_name: str) -> Dict[str, Any]:
    """Extract a specific section from a PDF.

    Args:
        pdf_path: Path to the PDF file
        section_name: Name of section to extract (e.g., "Abstract", "Methods")

    Returns:
        Dictionary with section content
    """
    try:
        from services.pdf_parser import extract_section
        return await extract_section(pdf_path, section_name)
    except Exception as e:
        raise PDFParseError(
            f"Failed to extract section '{section_name}' from {pdf_path}: {e}",
            original_error=e,
            context={"pdf_path": pdf_path, "section_name": section_name}
        )


async def extract_pdf_tables(pdf_path: str) -> Dict[str, Any]:
    """Extract all tables from a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with list of extracted tables
    """
    try:
        from services.pdf_parser import extract_all_tables
        return await extract_all_tables(pdf_path)
    except Exception as e:
        raise PDFParseError(
            f"Failed to extract tables from {pdf_path}: {e}",
            original_error=e,
            context={"pdf_path": pdf_path}
        )


# =============================================================================
# LLM Tools - For text generation and analysis
# =============================================================================

async def call_llm(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    tracker: Optional[Any] = None,
    agent_name: str = "unknown",
) -> str:
    """Call LLM for text generation.

    Args:
        prompt: The prompt to send to the LLM
        model: Model to use (default from config)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        tracker: Optional TokenTracker to record usage
        agent_name: Name of the calling agent (for tracking)

    Returns:
        Generated text response
    """
    try:
        from litellm import acompletion
        from utils.config import Config

        if model is None:
            model = Config.LITELLM_MODEL

        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        if content is None:
            raise LLMError(
                "LLM returned empty content",
                context={"model": model, "prompt_length": len(prompt)}
            )

        if tracker and hasattr(response, "usage") and response.usage:
            tracker.record(agent_name, model, {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            })

        return content

    except LLMError:
        raise
    except Exception as e:
        raise LLMError(
            f"LLM call failed: {e}",
            original_error=e,
            context={"model": model, "prompt_length": len(prompt)}
        )


async def call_llm_with_system(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    tracker: Optional[Any] = None,
    agent_name: str = "unknown",
) -> str:
    """Call LLM with system and user prompts.

    Args:
        system_prompt: System message defining agent behavior
        user_prompt: User message/query
        model: Model to use (default from config)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        tracker: Optional TokenTracker to record usage
        agent_name: Name of the calling agent (for tracking)

    Returns:
        Generated text response
    """
    try:
        from litellm import acompletion
        from utils.config import Config

        if model is None:
            model = Config.LITELLM_MODEL

        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content

        if tracker and hasattr(response, "usage") and response.usage:
            tracker.record(agent_name, model, {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            })

        return content

    except Exception as e:
        raise LLMError(
            f"LLM call failed: {e}",
            original_error=e,
            context={"model": model}
        )


# =============================================================================
# Zotero Tools - For library management
# =============================================================================

async def search_zotero(query: str, limit: int = 20) -> Dict[str, Any]:
    """Search Zotero library.

    Args:
        query: Search query
        limit: Maximum results to return

    Returns:
        Dictionary with matching items
    """
    try:
        from services.zotero import search_items
        return await search_items(query, limit=limit)
    except Exception as e:
        raise AgentError(
            ErrorType.NETWORK_ERROR,
            f"Zotero search failed: {e}",
            original_error=e
        )


async def create_or_get_zotero_collection(
    name: str,
    parent_collection: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a Zotero collection or return existing one.

    Args:
        name: Collection name
        parent_collection: Parent collection key (optional)

    Returns:
        Dictionary with collection key and name
    """
    try:
        from services.zotero import create_or_get_collection
        return await create_or_get_collection(name, parent_collection)
    except Exception as e:
        raise AgentError(
            ErrorType.NETWORK_ERROR,
            f"Zotero collection operation failed: {e}",
            original_error=e
        )


async def save_papers_to_zotero(
    papers: List[Dict[str, Any]],
    collection_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Save papers to a Zotero collection.

    Args:
        papers: List of paper dicts with doi or arxiv_id
        collection_key: Target collection key

    Returns:
        Dictionary with save results
    """
    try:
        from services.zotero import save_papers_to_collection
        return await save_papers_to_collection(papers, collection_key)
    except Exception as e:
        raise AgentError(
            ErrorType.NETWORK_ERROR,
            f"Zotero save failed: {e}",
            original_error=e
        )


async def import_zotero_collection(
    collection_key: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Import papers from a Zotero collection.

    Args:
        collection_key: Zotero collection key
        limit: Maximum papers to import

    Returns:
        Dictionary with papers in unified format
    """
    try:
        from services.zotero import import_collection_papers
        return await import_collection_papers(collection_key, limit=limit)
    except Exception as e:
        raise AgentError(
            ErrorType.NETWORK_ERROR,
            f"Zotero import failed: {e}",
            original_error=e
        )


async def write_analysis_to_zotero(
    item_key: str,
    analysis: Dict[str, Any],
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Write LitScribe analysis results as a Zotero note.

    Args:
        item_key: Zotero item key
        analysis: Analysis dict with key_findings, methodology, etc.
        tags: Optional tags for the note

    Returns:
        Dictionary with note creation result
    """
    try:
        from services.zotero import add_note

        # Format analysis as HTML note
        findings = analysis.get("key_findings", [])
        methodology = analysis.get("methodology", "")
        strengths = analysis.get("strengths", [])
        limitations = analysis.get("limitations", [])

        html_parts = ["<h2>LitScribe Analysis</h2>"]

        if findings:
            html_parts.append("<h3>Key Findings</h3><ul>")
            for f in findings:
                html_parts.append(f"<li>{f}</li>")
            html_parts.append("</ul>")

        if methodology:
            html_parts.append(f"<h3>Methodology</h3><p>{methodology}</p>")

        if strengths:
            html_parts.append("<h3>Strengths</h3><ul>")
            for s in strengths:
                html_parts.append(f"<li>{s}</li>")
            html_parts.append("</ul>")

        if limitations:
            html_parts.append("<h3>Limitations</h3><ul>")
            for l in limitations:
                html_parts.append(f"<li>{l}</li>")
            html_parts.append("</ul>")

        note_content = "\n".join(html_parts)
        note_tags = (tags or []) + ["LitScribe"]

        return await add_note(item_key, note_content, tags=note_tags)
    except Exception as e:
        raise AgentError(
            ErrorType.NETWORK_ERROR,
            f"Zotero note write failed: {e}",
            original_error=e
        )


# =============================================================================
# Tool Registry - For agent access
# =============================================================================

DISCOVERY_TOOLS = [
    unified_search,
    search_semantic_scholar,
    get_paper_details,
    get_paper_citations,
    get_paper_references,
    get_recommendations,
]

ZOTERO_TOOLS = [
    search_zotero,
    create_or_get_zotero_collection,
    save_papers_to_zotero,
    import_zotero_collection,
    write_analysis_to_zotero,
]

CRITICAL_READING_TOOLS = [
    download_arxiv_pdf,
    get_zotero_pdf_path,
    parse_pdf,
    extract_pdf_section,
    extract_pdf_tables,
]

SYNTHESIS_TOOLS = [
    call_llm,
    call_llm_with_system,
]

ALL_TOOLS = DISCOVERY_TOOLS + ZOTERO_TOOLS + CRITICAL_READING_TOOLS + SYNTHESIS_TOOLS
