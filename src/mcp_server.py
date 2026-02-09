"""LitScribe Unified MCP Server.

Exposes high-level literature review tools for external MCP clients
(Claude Desktop, Cursor, etc.) via stdio or streamable-http transport.

Usage:
    litscribe-mcp                  # stdio transport (default)
    litscribe-mcp --http 8080      # streamable-http on port 8080
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

from fastmcp import FastMCP

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

mcp = FastMCP(
    "litscribe",
    description="LitScribe - Autonomous Academic Literature Review Engine",
)


@mcp.tool()
async def search_literature(
    query: str,
    sources: Optional[List[str]] = None,
    limit: int = 20,
) -> dict:
    """Search academic literature across multiple databases.

    Searches arXiv, PubMed, Semantic Scholar, and/or Zotero in parallel,
    deduplicates results, and returns ranked papers.

    Args:
        query: Search query string (supports natural language)
        sources: Databases to search. Options: "arxiv", "pubmed",
                 "semantic_scholar", "zotero". Default: all except zotero.
        limit: Maximum results per source (default: 20, max: 100)

    Returns:
        Dictionary with total count, source breakdown, and paper list
    """
    from aggregators.unified_search import search_all_sources

    limit = min(limit, 100)

    result = await search_all_sources(
        query=query,
        sources=sources,
        max_per_source=limit,
        deduplicate=True,
        sort_by="relevance",
    )

    return result


@mcp.tool()
async def review_topic(
    question: str,
    max_papers: int = 10,
    sources: Optional[List[str]] = None,
    language: str = "en",
    enable_graphrag: bool = True,
) -> dict:
    """Generate a complete literature review on a research topic.

    Runs the full LitScribe multi-agent pipeline: planning, discovery,
    critical reading, knowledge graph construction, synthesis, and
    self-review. Returns structured review with citations.

    Args:
        question: Research question or topic to review
        max_papers: Maximum papers to analyze (default: 10, max: 500)
        sources: Databases to search (default: arxiv, pubmed, semantic_scholar)
        language: Output language - "en" for English, "zh" for Chinese, etc.
        enable_graphrag: Enable GraphRAG knowledge graph (default: True)

    Returns:
        Dictionary with review text, themes, gaps, citations, and metadata
    """
    from agents.graph import run_with_streaming

    max_papers = min(max_papers, 500)

    # Collect final state from streaming
    final_state = None
    async for state in run_with_streaming(
        research_question=question,
        max_papers=max_papers,
        sources=sources,
        language=language,
        graphrag_enabled=enable_graphrag,
    ):
        final_state = state

    if final_state is None:
        return {"error": "Review workflow produced no output"}

    # Extract the last state from the streaming output
    # LangGraph astream yields {node_name: state_update} dicts
    last_state = {}
    if isinstance(final_state, dict):
        for key, value in final_state.items():
            if isinstance(value, dict):
                last_state.update(value)
            else:
                last_state = final_state
                break

    # Build response
    synthesis = last_state.get("synthesis", {})
    if not synthesis and "synthesis" in final_state:
        synthesis = final_state["synthesis"] or {}

    return {
        "question": question,
        "review_text": synthesis.get("review_text", ""),
        "themes": synthesis.get("themes", []),
        "research_gaps": synthesis.get("research_gaps", []),
        "papers_analyzed": len(last_state.get("analyzed_papers", [])),
        "language": language,
    }


@mcp.tool()
async def get_paper_info(
    paper_id: str,
) -> dict:
    """Get detailed information about a specific academic paper.

    Supports multiple identifier formats: DOI, arXiv ID, PubMed ID,
    or Semantic Scholar ID.

    Args:
        paper_id: Paper identifier. Formats:
            - DOI: "10.1234/example" or "DOI:10.1234/example"
            - arXiv: "2301.00001" or "arXiv:2301.00001"
            - PubMed: "PMID:12345678"
            - Semantic Scholar: full hex ID

    Returns:
        Paper metadata including title, authors, abstract, citations,
        references, and links
    """
    from services.semantic_scholar import get_paper

    return await get_paper(paper_id)


@mcp.tool()
async def export_review(
    session_id: str,
    format: str = "markdown",
) -> dict:
    """Export a previously generated literature review.

    Args:
        session_id: Session ID from a completed review (supports prefix match)
        format: Export format - "markdown", "json", "bibtex"

    Returns:
        Dictionary with exported content and file path
    """
    from versioning.review_versions import get_session, get_version

    session = get_session(session_id)
    if session is None:
        return {"error": f"Session '{session_id}' not found"}

    # Get the latest version
    latest_version = session.get("current_version", 1)
    version = get_version(session["session_id"], latest_version)

    if version is None:
        return {"error": f"No version found for session '{session_id}'"}

    state_json = version.get("state_json", "{}")
    state = json.loads(state_json) if isinstance(state_json, str) else state_json

    synthesis = state.get("synthesis", {})
    review_text = synthesis.get("review_text", "")

    if format == "json":
        return {
            "session_id": session["session_id"],
            "format": "json",
            "content": state,
        }
    elif format == "bibtex":
        from exporters.bibtex_exporter import generate_bibtex
        analyzed = state.get("analyzed_papers", [])
        bibtex = generate_bibtex(analyzed)
        return {
            "session_id": session["session_id"],
            "format": "bibtex",
            "content": bibtex,
        }
    else:
        # Default: markdown
        return {
            "session_id": session["session_id"],
            "format": "markdown",
            "content": review_text,
        }


def main():
    """Run the LitScribe MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="LitScribe MCP Server")
    parser.add_argument(
        "--http",
        type=int,
        metavar="PORT",
        help="Run with streamable-http transport on PORT",
    )
    args = parser.parse_args()

    if args.http:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=args.http)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
