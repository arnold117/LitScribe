"""Semantic Scholar MCP Server - Search academic papers via Semantic Scholar API."""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

import aiohttp

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.config import Config

# API configuration
BASE_URL = "https://api.semanticscholar.org/graph/v1"
HEADERS = {}

# Add API key if available (optional, increases rate limit)
if hasattr(Config, "SEMANTIC_SCHOLAR_API_KEY") and Config.SEMANTIC_SCHOLAR_API_KEY:
    HEADERS["x-api-key"] = Config.SEMANTIC_SCHOLAR_API_KEY


async def _make_request(endpoint: str, params: dict = None) -> dict:
    """Make async request to Semantic Scholar API."""
    url = f"{BASE_URL}/{endpoint}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=HEADERS) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 429:
                return {"error": "Rate limit exceeded. Wait a moment and retry."}
            else:
                text = await response.text()
                return {"error": f"API error {response.status}: {text[:200]}"}


async def search_papers(
    query: str,
    limit: int = 20,
    year: Optional[str] = None,
    fields_of_study: Optional[List[str]] = None,
    open_access_only: bool = False,
) -> dict:
    """
    Search for papers on Semantic Scholar.

    Args:
        query: Search query string
        limit: Maximum number of results (default: 20, max: 100)
        year: Filter by year or year range (e.g., "2020", "2020-2024")
        fields_of_study: Filter by fields (e.g., ["Computer Science", "Medicine"])
        open_access_only: Only return papers with open access PDFs

    Returns:
        Dictionary with search results including papers and metadata
    """
    limit = min(limit, 100)

    # Build query parameters
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,authors,year,citationCount,abstract,url,openAccessPdf,externalIds,venue,fieldsOfStudy,tldr",
    }

    if year:
        params["year"] = year

    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    if open_access_only:
        params["openAccessPdf"] = ""

    result = await _make_request("paper/search", params)

    if "error" in result:
        return result

    papers = []
    for paper in result.get("data", []):
        papers.append(_format_paper(paper))

    return {
        "query": query,
        "total": result.get("total", 0),
        "count": len(papers),
        "papers": papers,
    }


async def get_paper(
    paper_id: str,
) -> dict:
    """
    Get detailed information for a specific paper.

    Args:
        paper_id: Paper identifier - can be:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - DOI (e.g., "10.18653/v1/N18-3011" or "DOI:10.18653/v1/N18-3011")
            - arXiv ID (e.g., "arXiv:1706.03762")
            - PubMed ID (e.g., "PMID:19872477")
            - Corpus ID (e.g., "CorpusId:37220927")

    Returns:
        Detailed paper information
    """
    params = {
        "fields": "paperId,title,authors,year,citationCount,referenceCount,abstract,url,openAccessPdf,externalIds,venue,fieldsOfStudy,tldr,citations.paperId,citations.title,references.paperId,references.title",
    }

    result = await _make_request(f"paper/{paper_id}", params)

    if "error" in result:
        return result

    return _format_paper_detail(result)


async def get_paper_citations(
    paper_id: str,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    Get papers that cite a given paper.

    Args:
        paper_id: Paper identifier (S2 ID, DOI, arXiv ID, etc.)
        limit: Maximum number of citations to return (default: 50, max: 1000)
        offset: Offset for pagination

    Returns:
        List of citing papers
    """
    limit = min(limit, 1000)

    params = {
        "fields": "paperId,title,authors,year,citationCount,abstract,url",
        "limit": limit,
        "offset": offset,
    }

    result = await _make_request(f"paper/{paper_id}/citations", params)

    if "error" in result:
        return result

    citations = []
    for item in (result.get("data") or []):
        citing_paper = item.get("citingPaper") or {}
        if citing_paper:
            citations.append(_format_paper(citing_paper))

    return {
        "paper_id": paper_id,
        "total": result.get("total", len(citations)),
        "count": len(citations),
        "offset": offset,
        "citations": citations,
    }


async def get_paper_references(
    paper_id: str,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    Get papers that a given paper cites (its references).

    Args:
        paper_id: Paper identifier (S2 ID, DOI, arXiv ID, etc.)
        limit: Maximum number of references to return (default: 50, max: 1000)
        offset: Offset for pagination

    Returns:
        List of referenced papers
    """
    limit = min(limit, 1000)

    params = {
        "fields": "paperId,title,authors,year,citationCount,abstract,url",
        "limit": limit,
        "offset": offset,
    }

    result = await _make_request(f"paper/{paper_id}/references", params)

    if "error" in result:
        return result

    references = []
    for item in (result.get("data") or []):
        cited_paper = item.get("citedPaper") or {}
        if cited_paper:
            references.append(_format_paper(cited_paper))

    return {
        "paper_id": paper_id,
        "total": result.get("total", len(references)),
        "count": len(references),
        "offset": offset,
        "references": references,
    }


async def search_by_author(
    author_name: str,
    limit: int = 20,
) -> dict:
    """
    Search for an author and their papers.

    Args:
        author_name: Author name to search for
        limit: Maximum number of papers to return per author

    Returns:
        Author information and their papers
    """
    # First search for author
    params = {
        "query": author_name,
        "limit": 5,
        "fields": "authorId,name,affiliations,paperCount,citationCount,hIndex",
    }

    result = await _make_request("author/search", params)

    if "error" in result:
        return result

    authors = []
    for author in result.get("data", []):
        author_id = author.get("authorId")

        # Get author's papers
        papers_params = {
            "fields": "paperId,title,year,citationCount,venue",
            "limit": limit,
        }
        papers_result = await _make_request(f"author/{author_id}/papers", papers_params)

        author_papers = []
        if "data" in papers_result:
            author_papers = [_format_paper_brief(p) for p in papers_result["data"]]

        authors.append({
            "author_id": author_id,
            "name": author.get("name", ""),
            "affiliations": author.get("affiliations", []),
            "paper_count": author.get("paperCount", 0),
            "citation_count": author.get("citationCount", 0),
            "h_index": author.get("hIndex", 0),
            "papers": author_papers,
        })

    return {
        "query": author_name,
        "count": len(authors),
        "authors": authors,
    }


async def get_recommendations(
    paper_id: str,
    limit: int = 20,
) -> dict:
    """
    Get paper recommendations based on a given paper.

    Args:
        paper_id: Paper identifier to get recommendations for
        limit: Maximum number of recommendations (default: 20, max: 500)

    Returns:
        List of recommended papers
    """
    limit = min(limit, 500)

    params = {
        "fields": "paperId,title,authors,year,citationCount,abstract,url,openAccessPdf",
        "limit": limit,
    }

    result = await _make_request(f"recommendations/v1/papers/forpaper/{paper_id}", params)

    if "error" in result:
        return result

    papers = []
    for paper in result.get("recommendedPapers", []):
        papers.append(_format_paper(paper))

    return {
        "source_paper_id": paper_id,
        "count": len(papers),
        "recommendations": papers,
    }


async def batch_get_papers(
    paper_ids: List[str],
) -> dict:
    """
    Get details for multiple papers at once.

    Args:
        paper_ids: List of paper identifiers (max 500)

    Returns:
        List of paper details
    """
    paper_ids = paper_ids[:500]

    params = {
        "fields": "paperId,title,authors,year,citationCount,abstract,url,openAccessPdf,externalIds,venue",
    }

    # POST request for batch
    url = f"{BASE_URL}/paper/batch"

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            params=params,
            json={"ids": paper_ids},
            headers={**HEADERS, "Content-Type": "application/json"},
        ) as response:
            if response.status == 200:
                result = await response.json()
            else:
                text = await response.text()
                return {"error": f"API error {response.status}: {text[:200]}"}

    papers = []
    for paper in result:
        if paper:  # Some IDs may not be found
            papers.append(_format_paper(paper))

    return {
        "requested": len(paper_ids),
        "found": len(papers),
        "papers": papers,
    }


def _format_paper(paper: dict) -> dict:
    """Format paper data for output."""
    authors = []
    for author in (paper.get("authors") or []):
        authors.append(author.get("name", ""))

    external_ids = paper.get("externalIds", {}) or {}

    pdf_url = None
    open_access = paper.get("openAccessPdf")
    if open_access:
        pdf_url = open_access.get("url")

    tldr = None
    if paper.get("tldr"):
        tldr = paper["tldr"].get("text")

    return {
        "paper_id": paper.get("paperId", ""),
        "title": paper.get("title", ""),
        "authors": authors,
        "year": paper.get("year"),
        "citation_count": paper.get("citationCount", 0),
        "abstract": paper.get("abstract", ""),
        "venue": paper.get("venue", ""),
        "url": paper.get("url", ""),
        "pdf_url": pdf_url,
        "doi": external_ids.get("DOI"),
        "arxiv_id": external_ids.get("ArXiv"),
        "pmid": external_ids.get("PubMed"),
        "fields_of_study": paper.get("fieldsOfStudy", []),
        "tldr": tldr,
    }


def _format_paper_detail(paper: dict) -> dict:
    """Format detailed paper data."""
    base = _format_paper(paper)

    # Add reference count
    base["reference_count"] = paper.get("referenceCount", 0)

    # Add brief citation list
    citations = []
    for item in (paper.get("citations") or [])[:10]:
        if item:
            citations.append({
                "paper_id": item.get("paperId", ""),
                "title": item.get("title", ""),
            })
    base["top_citations"] = citations

    # Add brief reference list
    references = []
    for item in (paper.get("references") or [])[:10]:
        if item:
            references.append({
                "paper_id": item.get("paperId", ""),
                "title": item.get("title", ""),
            })
    base["top_references"] = references

    return base


def _format_paper_brief(paper: dict) -> dict:
    """Format brief paper data."""
    return {
        "paper_id": paper.get("paperId", ""),
        "title": paper.get("title", ""),
        "year": paper.get("year"),
        "citation_count": paper.get("citationCount", 0),
        "venue": paper.get("venue", ""),
    }


