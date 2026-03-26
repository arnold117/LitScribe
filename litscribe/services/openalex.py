"""OpenAlex search service — 250M+ academic works across all disciplines."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import aiohttp

from litscribe.models.paper import Paper

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"
MAILTO = "litscribe@example.com"


def _reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
    """Reconstruct plaintext abstract from OpenAlex inverted index format.

    OpenAlex stores abstracts as {word: [position1, position2, ...]} for
    copyright reasons. We rebuild by sorting words by their positions.
    """
    if not inverted_index:
        return ""
    position_word = []
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word.append((pos, word))
    position_word.sort(key=lambda x: x[0])
    return " ".join(w for _, w in position_word)


def _work_to_paper(work: Dict[str, Any]) -> Paper:
    """Convert OpenAlex work object to a Paper."""
    # Extract authors
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in work.get("authorships") or []
    ]

    # Extract OpenAlex ID (strip URL prefix)
    openalex_id = work.get("id") or ""
    if openalex_id.startswith("https://openalex.org/"):
        openalex_id = openalex_id[len("https://openalex.org/"):]

    # Extract DOI (strip URL prefix)
    doi = work.get("doi") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    # Extract best PDF URL from locations
    pdf_url: Optional[str] = None
    for loc in work.get("locations") or []:
        if loc.get("pdf_url"):
            pdf_url = loc["pdf_url"]
            break

    # Extract venue from primary location
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    venue = source.get("display_name") or ""

    abstract = _reconstruct_abstract(work.get("abstract_inverted_index") or {})

    return Paper(
        paper_id=f"openalex:{openalex_id}",
        title=work.get("display_name") or work.get("title") or "",
        authors=authors,
        abstract=abstract,
        year=work.get("publication_year") or 0,
        sources={"openalex": openalex_id},
        venue=venue,
        citations=work.get("cited_by_count") or 0,
        doi=doi,
        pdf_urls=[pdf_url] if pdf_url else [],
    )


class OpenAlexService:
    """OpenAlex search service. Polite pool — no API key needed."""

    source_name = "openalex"

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **filters,
    ) -> list[Paper]:
        """Search OpenAlex for academic papers.

        Supported filters:
            year_from (int): earliest publication year
            year_to (int): latest publication year
            min_citations (int): minimum citation count (post-filter)
        """
        year_from: Optional[int] = filters.get("year_from")
        year_to: Optional[int] = filters.get("year_to")
        min_citations: Optional[int] = filters.get("min_citations")

        params: Dict[str, Any] = {
            "search": query,
            "per_page": min(max_results, 200),
            "mailto": MAILTO,
        }

        # Build filter string
        filter_parts: List[str] = []
        if year_from and year_to:
            filter_parts.append(f"publication_year:{year_from}-{year_to}")
        elif year_from:
            filter_parts.append(f"publication_year:>{year_from - 1}")
        elif year_to:
            filter_parts.append(f"publication_year:<{year_to + 1}")

        if min_citations:
            filter_parts.append(f"cited_by_count:>{min_citations - 1}")

        if filter_parts:
            params["filter"] = ",".join(filter_parts)

        url = f"{BASE_URL}/works"

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
                        logger.warning("OpenAlex rate limit hit")
                        return []
                    else:
                        text = await response.text()
                        logger.warning(
                            "OpenAlex API error %s: %s",
                            response.status,
                            text[:200],
                        )
                        return []
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.warning("OpenAlex request failed: %s", exc)
            return []

        papers: list[Paper] = []
        for work in data.get("results") or []:
            papers.append(_work_to_paper(work))
            if len(papers) >= max_results:
                break

        return papers
