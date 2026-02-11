"""arXiv service - Search and download papers from arXiv.

Rate limiting strategy:
- Global singleton client (reuses HTTP connection)
- 3.5s cooldown between requests (arXiv asks for >= 3s)
- Client-level retry with 5s delay (handles transient errors)
- Cooldown enforced in blocking thread (run_in_executor)
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import arxiv

from models.paper import Paper
from utils.config import Config

logger = logging.getLogger(__name__)

# Global singleton client — reuses connection, respects delay_seconds between pages
_arxiv_client: Optional[arxiv.Client] = None
_last_request_time: float = 0.0
_ARXIV_COOLDOWN = 3.5  # seconds between requests (arXiv asks for 3s minimum)


def _get_client() -> arxiv.Client:
    """Get or create the global arXiv client singleton."""
    global _arxiv_client
    if _arxiv_client is None:
        _arxiv_client = arxiv.Client(
            page_size=20,
            num_retries=3,
            delay_seconds=5,
        )
    return _arxiv_client


def _wait_cooldown():
    """Block until cooldown period has passed since last request."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _ARXIV_COOLDOWN:
        wait = _ARXIV_COOLDOWN - elapsed
        logger.debug(f"arXiv cooldown: waiting {wait:.1f}s")
        time.sleep(wait)
    _last_request_time = time.monotonic()


def _parse_arxiv_result(result: arxiv.Result) -> Paper:
    """Convert arxiv.Result to Paper model."""
    return Paper.from_arxiv_result(result)


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
        _wait_cooldown()
        client = _get_client()
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
        _wait_cooldown()
        client = _get_client()
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))
        return results[0] if results else None

    result = await loop.run_in_executor(None, do_fetch)

    if not result:
        return {"error": f"Paper {arxiv_id} not found"}

    paper = _parse_arxiv_result(result)
    return paper.to_dict()


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
        _wait_cooldown()
        client = _get_client()
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
        _wait_cooldown()
        client = _get_client()
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
        _wait_cooldown()
        client = _get_client()
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
        _wait_cooldown()
        client = _get_client()
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
        _wait_cooldown()
        client = _get_client()
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
