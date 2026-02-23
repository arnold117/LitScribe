"""OpenAlex API client — search 250M+ academic works across all disciplines.

Covers PubMed, CrossRef, arXiv, bioRxiv, medRxiv, DBLP and more.
No domain-specific filters needed; broad coverage by default.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"

# Polite pool: set a mailto for higher rate limits (no API key needed)
MAILTO = "litscribe@example.com"


def _reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
    """Reconstruct plaintext abstract from OpenAlex inverted index format.

    OpenAlex stores abstracts as {word: [position1, position2, ...]} for
    copyright reasons. We rebuild by sorting words by their positions.
    """
    if not inverted_index:
        return ""
    # Build (position, word) pairs
    position_word = []
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word.append((pos, word))
    position_word.sort(key=lambda x: x[0])
    return " ".join(w for _, w in position_word)


def _format_paper(work: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAlex work object to our standard paper dict format."""
    # Extract authors
    authors = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append(name)

    # Extract DOI (strip URL prefix if present)
    doi = work.get("doi") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    # Extract best PDF URL from locations
    pdf_url = None
    for loc in work.get("locations") or []:
        if loc.get("pdf_url"):
            pdf_url = loc["pdf_url"]
            break

    # Extract venue
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    venue = source.get("display_name") or ""

    # Reconstruct abstract
    abstract = _reconstruct_abstract(
        work.get("abstract_inverted_index") or {}
    )

    # Extract external IDs
    ids = work.get("ids") or {}
    pmid = ids.get("pmid") or ""
    if pmid.startswith("https://pubmed.ncbi.nlm.nih.gov/"):
        pmid = pmid.split("/")[-1]

    openalex_id = work.get("id") or ""
    if openalex_id.startswith("https://openalex.org/"):
        openalex_id = openalex_id[len("https://openalex.org/"):]

    return {
        "paper_id": openalex_id,
        "title": work.get("display_name") or work.get("title") or "",
        "authors": authors,
        "year": work.get("publication_year") or 0,
        "citation_count": work.get("cited_by_count") or 0,
        "abstract": abstract,
        "venue": venue,
        "url": work.get("doi") or "",
        "pdf_url": pdf_url,
        "doi": doi if doi else None,
        "pmid": pmid if pmid else None,
        "fields_of_study": [
            kw.get("display_name")
            for kw in (work.get("keywords") or [])
            if kw.get("display_name")
        ],
    }


async def search_papers(
    query: str,
    max_results: int = 20,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    min_citations: Optional[int] = None,
) -> dict:
    """Search OpenAlex for academic papers.

    Args:
        query: Search query (supports boolean AND/OR/NOT)
        max_results: Maximum results to return (max 200 per page)
        year_from: Filter publications from this year
        year_to: Filter publications up to this year
        min_citations: Post-filter by minimum citation count

    Returns:
        Dict with query, count, and papers list
    """
    params: Dict[str, Any] = {
        "search": query,
        "per_page": min(max_results, 200),
        "mailto": MAILTO,
    }

    # Build filter string
    filters = []
    if year_from and year_to:
        filters.append(f"publication_year:{year_from}-{year_to}")
    elif year_from:
        filters.append(f"publication_year:>{year_from - 1}")
    elif year_to:
        filters.append(f"publication_year:<{year_to + 1}")

    if min_citations:
        filters.append(f"cited_by_count:>{min_citations - 1}")

    if filters:
        params["filter"] = ",".join(filters)

    url = f"{BASE_URL}/works"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    data = await response.json()
                elif response.status == 429:
                    logger.warning("OpenAlex rate limit hit")
                    return {"query": query, "count": 0, "papers": []}
                else:
                    text = await response.text()
                    logger.warning(f"OpenAlex API error {response.status}: {text[:200]}")
                    return {"query": query, "count": 0, "papers": []}
    except (aiohttp.ClientError, TimeoutError) as e:
        logger.warning(f"OpenAlex request failed: {e}")
        return {"query": query, "count": 0, "papers": []}

    papers = []
    for work in data.get("results") or []:
        formatted = _format_paper(work)
        papers.append(formatted)
        if len(papers) >= max_results:
            break

    return {
        "query": query,
        "count": len(papers),
        "papers": papers,
    }
