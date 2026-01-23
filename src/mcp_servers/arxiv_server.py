"""arXiv MCP Server - Search and download papers from arXiv."""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import arxiv

from models.paper import Paper
from utils.config import Config

# Initialize FastMCP server
mcp = FastMCP("arxiv-server")


def _parse_arxiv_result(result: arxiv.Result) -> Paper:
    """Convert arxiv.Result to Paper model."""
    return Paper.from_arxiv_result(result)


@mcp.tool()
async def search_papers(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    category: Optional[str] = None,
) -> dict:
    """
    Search arXiv for papers matching the query.

    Args:
        query: Search query (supports arXiv query syntax)
        max_results: Maximum number of results (default: 10, max: 100)
        sort_by: Sort criterion - "relevance", "lastUpdatedDate", or "submittedDate"
        sort_order: Sort order - "ascending" or "descending"
        category: Filter by arXiv category (e.g., "cs.AI", "cs.CL", "stat.ML")

    Returns:
        Dictionary with search results
    """
    max_results = min(max_results, 100)

    # Map sort options
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
    }
    order_map = {
        "ascending": arxiv.SortOrder.Ascending,
        "descending": arxiv.SortOrder.Descending,
    }

    sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
    order = order_map.get(sort_order, arxiv.SortOrder.Descending)

    # Build query with category filter
    if category:
        query = f"cat:{category} AND ({query})"

    loop = asyncio.get_event_loop()

    def do_search():
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=order,
        )
        results = list(client.results(search))
        return results

    results = await loop.run_in_executor(None, do_search)

    papers = [_parse_arxiv_result(r) for r in results]

    return {
        "query": query,
        "count": len(papers),
        "papers": [p.to_dict() for p in papers],
    }


@mcp.tool()
async def get_paper_metadata(arxiv_id: str) -> dict:
    """
    Get detailed metadata for a specific arXiv paper.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2301.00001" or "2301.00001v1")

    Returns:
        Full paper metadata
    """
    loop = asyncio.get_event_loop()

    def do_fetch():
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))
        return results[0] if results else None

    result = await loop.run_in_executor(None, do_fetch)

    if not result:
        return {"error": f"Paper {arxiv_id} not found"}

    paper = _parse_arxiv_result(result)
    return paper.to_dict()


@mcp.tool()
async def download_pdf(
    arxiv_id: str,
    save_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> dict:
    """
    Download PDF for an arXiv paper.

    Args:
        arxiv_id: arXiv paper ID
        save_dir: Directory to save PDF (default: LITSCRIBE_DATA_DIR/pdfs)
        filename: Custom filename (default: {arxiv_id}.pdf)

    Returns:
        Dictionary with download info and local path
    """
    loop = asyncio.get_event_loop()

    # Set default save directory
    if save_dir:
        pdf_dir = Path(save_dir)
    else:
        pdf_dir = Config.DATA_DIR / "pdfs"

    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Set filename
    if filename:
        pdf_path = pdf_dir / filename
    else:
        # Clean arxiv_id for filename (remove version suffix if present)
        clean_id = arxiv_id.replace("/", "_").replace(":", "_")
        pdf_path = pdf_dir / f"{clean_id}.pdf"

    def do_download():
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))

        if not results:
            return None, "Paper not found"

        paper = results[0]
        # Download PDF
        paper.download_pdf(dirpath=str(pdf_dir), filename=pdf_path.name)
        return paper, None

    paper, error = await loop.run_in_executor(None, do_download)

    if error:
        return {"error": error}

    return {
        "arxiv_id": arxiv_id,
        "title": paper.title,
        "pdf_path": str(pdf_path),
        "pdf_url": paper.pdf_url,
        "downloaded": pdf_path.exists(),
    }


@mcp.tool()
async def get_paper_by_doi(doi: str) -> dict:
    """
    Search arXiv for a paper by DOI.

    Args:
        doi: DOI of the paper

    Returns:
        Paper metadata if found
    """
    loop = asyncio.get_event_loop()

    def do_search():
        client = arxiv.Client()
        # Search by DOI in all fields
        search = arxiv.Search(
            query=f'doi:"{doi}"',
            max_results=5,
        )
        results = list(client.results(search))
        return results

    results = await loop.run_in_executor(None, do_search)

    if not results:
        return {"error": f"No arXiv paper found with DOI {doi}"}

    # Return the first match
    paper = _parse_arxiv_result(results[0])
    return paper.to_dict()


@mcp.tool()
async def get_recent_papers(
    category: str,
    max_results: int = 10,
) -> dict:
    """
    Get recent papers from a specific arXiv category.

    Args:
        category: arXiv category (e.g., "cs.AI", "cs.CL", "stat.ML", "cs.LG")
        max_results: Maximum number of results (default: 10)

    Returns:
        List of recent papers
    """
    max_results = min(max_results, 50)

    loop = asyncio.get_event_loop()

    def do_search():
        client = arxiv.Client()
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        results = list(client.results(search))
        return results

    results = await loop.run_in_executor(None, do_search)

    papers = [_parse_arxiv_result(r) for r in results]

    return {
        "category": category,
        "count": len(papers),
        "papers": [p.to_dict() for p in papers],
    }


@mcp.tool()
async def search_by_author(
    author_name: str,
    max_results: int = 20,
) -> dict:
    """
    Search arXiv for papers by a specific author.

    Args:
        author_name: Author name to search for
        max_results: Maximum number of results (default: 20)

    Returns:
        List of papers by the author
    """
    max_results = min(max_results, 100)

    loop = asyncio.get_event_loop()

    def do_search():
        client = arxiv.Client()
        search = arxiv.Search(
            query=f'au:"{author_name}"',
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        results = list(client.results(search))
        return results

    results = await loop.run_in_executor(None, do_search)

    papers = [_parse_arxiv_result(r) for r in results]

    return {
        "author": author_name,
        "count": len(papers),
        "papers": [p.to_dict() for p in papers],
    }


@mcp.tool()
async def batch_get_papers(arxiv_ids: List[str]) -> dict:
    """
    Get metadata for multiple arXiv papers at once.

    Args:
        arxiv_ids: List of arXiv IDs (max 50)

    Returns:
        List of paper metadata
    """
    arxiv_ids = arxiv_ids[:50]  # Limit

    loop = asyncio.get_event_loop()

    def do_fetch():
        client = arxiv.Client()
        search = arxiv.Search(id_list=arxiv_ids)
        results = list(client.results(search))
        return results

    results = await loop.run_in_executor(None, do_fetch)

    papers = [_parse_arxiv_result(r) for r in results]

    return {
        "requested": len(arxiv_ids),
        "found": len(papers),
        "papers": [p.to_dict() for p in papers],
    }


def main():
    """Run the arXiv MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="arXiv MCP Server")
    parser.add_argument("--stdio", action="store_true", help="Use stdio transport")
    args = parser.parse_args()

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
