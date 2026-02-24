"""Europe PMC API client — search ~40M life science articles.

Superset of PubMed with European journals, preprints, patents, and clinical guidelines.
Free API, no authentication required.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def _format_paper(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Europe PMC result to standard paper dict format."""
    # Parse authors from authorString (semicolon or comma separated)
    author_str = hit.get("authorString") or ""
    if author_str:
        # Europe PMC typically uses "Smith J, Jones A, Brown B" (comma-separated)
        # but some records use semicolons: "Smith J; Jones A; Brown B"
        if ";" in author_str:
            authors = [a.strip().rstrip(".") for a in author_str.split(";") if a.strip()]
        else:
            authors = [a.strip() for a in author_str.split(",") if a.strip()]
    else:
        authors = []

    # Extract PMC ID and build PDF URL for open access articles
    pmcid = hit.get("pmcid") or ""
    pdf_url = None
    if pmcid and hit.get("isOpenAccess") == "Y":
        pdf_url = f"https://europepmc.org/articles/{pmcid}/pdf"

    # Extract DOI
    doi = hit.get("doi") or ""

    # Build paper_id: prefer PMID, then PMC ID, then Europe PMC internal ID
    pmid = hit.get("pmid") or ""
    paper_id = pmid or pmcid or hit.get("id") or ""

    return {
        "paper_id": paper_id,
        "title": hit.get("title") or "",
        "authors": authors,
        "year": int(hit.get("pubYear") or 0),
        "citation_count": int(hit.get("citedByCount") or 0),
        "abstract": hit.get("abstractText") or "",
        "venue": hit.get("journalTitle") or "",
        "url": f"https://doi.org/{doi}" if doi else "",
        "pdf_url": pdf_url,
        "doi": doi if doi else None,
        "pmid": pmid if pmid else None,
        "pmc_id": pmcid if pmcid else None,
        "pub_type": hit.get("pubType") or "",
        "is_open_access": hit.get("isOpenAccess") == "Y",
    }


async def search_papers(
    query: str,
    max_results: int = 20,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    min_citations: Optional[int] = None,
) -> dict:
    """Search Europe PMC for academic papers.

    Args:
        query: Search query (supports Europe PMC query syntax)
        max_results: Maximum results to return (max 100 per page)
        year_from: Filter publications from this year
        year_to: Filter publications up to this year
        min_citations: Post-filter by minimum citation count

    Returns:
        Dict with query, count, and papers list
    """
    # Build query with year filter
    full_query = query
    if year_from and year_to:
        full_query += f" AND PUB_YEAR:[{year_from} TO {year_to}]"
    elif year_from:
        full_query += f" AND PUB_YEAR:[{year_from} TO 2099]"
    elif year_to:
        full_query += f" AND PUB_YEAR:[1900 TO {year_to}]"

    params = {
        "query": full_query,
        "resultType": "core",  # Include abstracts
        "pageSize": min(max_results, 100),
        "format": "json",
        "sort": "RELEVANCE",
    }

    url = f"{BASE_URL}/search"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    data = await response.json()
                elif response.status == 429:
                    logger.warning("Europe PMC rate limit hit")
                    return {"query": query, "count": 0, "papers": []}
                else:
                    text = await response.text()
                    logger.warning(f"Europe PMC API error {response.status}: {text[:200]}")
                    return {"query": query, "count": 0, "papers": []}
    except (aiohttp.ClientError, TimeoutError) as e:
        logger.warning(f"Europe PMC request failed: {e}")
        return {"query": query, "count": 0, "papers": []}

    results = data.get("resultList", {}).get("result", [])

    papers = []
    for hit in results:
        formatted = _format_paper(hit)
        # Post-filter by citation count
        if min_citations and formatted["citation_count"] < min_citations:
            continue
        papers.append(formatted)
        if len(papers) >= max_results:
            break

    return {
        "query": query,
        "count": len(papers),
        "papers": papers,
    }
