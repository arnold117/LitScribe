"""Europe PMC search service — ~40M life science articles."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import aiohttp

from litscribe.models.paper import Paper

logger = logging.getLogger(__name__)

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def _parse_authors(author_str: str) -> list[str]:
    """Parse Europe PMC authorString (semicolon or comma separated)."""
    if not author_str:
        return []
    if ";" in author_str:
        return [a.strip().rstrip(".") for a in author_str.split(";") if a.strip()]
    return [a.strip() for a in author_str.split(",") if a.strip()]


def _hit_to_paper(hit: Dict[str, Any]) -> Paper:
    """Convert a Europe PMC result hit to a Paper."""
    authors = _parse_authors(hit.get("authorString") or "")

    pmid = hit.get("pmid") or ""
    pmcid = hit.get("pmcid") or ""
    source_id = pmid or pmcid or hit.get("id", "")

    # PDF URL for open access PMC articles
    pdf_url: Optional[str] = None
    if pmcid and hit.get("isOpenAccess") == "Y":
        pdf_url = f"https://europepmc.org/articles/{pmcid}/pdf"

    doi = hit.get("doi") or ""

    # Journal title is nested: journalInfo.journal.title
    venue = ""
    journal_info = hit.get("journalInfo") or {}
    if isinstance(journal_info, dict):
        journal = journal_info.get("journal") or {}
        venue = journal.get("title") or ""

    return Paper(
        paper_id=f"europepmc:{source_id}",
        title=hit.get("title") or "",
        authors=authors,
        abstract=hit.get("abstractText") or "",
        year=int(hit.get("pubYear") or 0),
        sources={"europe_pmc": source_id},
        venue=venue,
        citations=int(hit.get("citedByCount") or 0),
        doi=doi,
        pdf_urls=[pdf_url] if pdf_url else [],
    )


class EuropePMCService:
    """Europe PMC search service. Free API, no authentication required."""

    source_name = "europe_pmc"

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **filters,
    ) -> list[Paper]:
        """Search Europe PMC for academic papers.

        Supported filters:
            year_from (int): earliest publication year
            year_to (int): latest publication year
            min_citations (int): minimum citation count (post-filter)
        """
        year_from: Optional[int] = filters.get("year_from")
        year_to: Optional[int] = filters.get("year_to")
        min_citations: Optional[int] = filters.get("min_citations")

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
            "resultType": "core",
            "pageSize": min(max_results, 100),
            "format": "json",
            # Note: do NOT pass sort= — it breaks the API (returns hitCount:None).
        }

        url = f"{BASE_URL}/search"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                    elif response.status == 429:
                        logger.warning("Europe PMC rate limit hit")
                        return []
                    else:
                        text = await response.text()
                        logger.warning(
                            "Europe PMC API error %s: %s",
                            response.status,
                            text[:200],
                        )
                        return []
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.warning("Europe PMC request failed: %s", exc)
            return []

        results = data.get("resultList", {}).get("result", [])

        papers: list[Paper] = []
        for hit in results:
            paper = _hit_to_paper(hit)
            if min_citations and paper.citations < min_citations:
                continue
            papers.append(paper)
            if len(papers) >= max_results:
                break

        return papers
