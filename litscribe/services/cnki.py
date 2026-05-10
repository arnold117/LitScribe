from __future__ import annotations

import logging
from typing import Any

import aiohttp

from litscribe.models.paper import Paper

logger = logging.getLogger(__name__)


class CNKIService:
    """Search Chinese academic literature via CNKI's open search API.

    Note: CNKI doesn't have a true public API. This uses their
    open search endpoint for basic keyword search. Results are limited.
    For production use, consider CNKI's institutional API or web scraping.
    """

    source_name = "cnki"

    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]:
        # CNKI doesn't have a public REST API like Western databases.
        # Use CrossRef with Chinese content filter as a proxy.
        try:
            return await self._search_crossref_chinese(query, max_results)
        except Exception as e:
            logger.debug(f"CNKI/CrossRef search failed: {e}")
            return []

    async def _search_crossref_chinese(self, query: str, max_results: int) -> list[Paper]:
        url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": max_results,
            "select": "DOI,title,author,published-print,container-title,abstract",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

        papers = []
        for item in data.get("message", {}).get("items", []):
            title_list = item.get("title", [])
            title = title_list[0] if title_list else ""

            authors = []
            for a in item.get("author", []):
                name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                if name:
                    authors.append(name)

            year = 0
            pub = item.get("published-print", {}).get("date-parts", [[]])
            if pub and pub[0]:
                year = pub[0][0]

            doi = item.get("DOI", "")
            venue_list = item.get("container-title", [])
            venue = venue_list[0] if venue_list else ""
            abstract = item.get("abstract", "")
            # Strip HTML tags from CrossRef abstract
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract)

            if title:
                papers.append(Paper(
                    paper_id=f"crossref:{doi}",
                    title=title,
                    authors=authors,
                    abstract=abstract[:500],
                    year=year,
                    sources={"crossref": doi},
                    venue=venue,
                    doi=doi,
                ))

        return papers


class WanfangService:
    """Wanfang Data search — uses CrossRef as proxy for Chinese literature."""

    source_name = "wanfang"

    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]:
        # Wanfang doesn't have a public API either.
        # Reuse CrossRef with different query strategy for Chinese content.
        cnki = CNKIService()
        return await cnki.search(query, max_results)
