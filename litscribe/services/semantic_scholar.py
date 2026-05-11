"""Semantic Scholar search service — S2 Graph API v1."""
from __future__ import annotations

from typing import Optional

import aiohttp

from litscribe.models.paper import Paper
from litscribe.services.base import SearchService

BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = (
    "paperId,title,authors,year,citationCount,abstract,url,"
    "openAccessPdf,externalIds,venue"
)


class SemanticScholarService:
    """Semantic Scholar search service implementing the SearchService protocol."""

    source_name = "semantic_scholar"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["x-api-key"] = api_key

    async def _api_request(self, endpoint: str, params: dict) -> dict:
        """Make an async GET request to the S2 Graph API."""
        url = f"{BASE_URL}/{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self._headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    return {"error": "Rate limit exceeded. Wait a moment and retry."}
                else:
                    text = await resp.text()
                    return {"error": f"API error {resp.status}: {text[:200]}"}

    @staticmethod
    def _convert(paper: dict) -> Paper:
        """Convert a raw S2 paper dict to a v2 Paper model."""
        external_ids = paper.get("externalIds", {}) or {}
        open_access = paper.get("openAccessPdf") or {}
        return Paper(
            paper_id=f"s2:{paper.get('paperId', '')}",
            title=paper.get("title", ""),
            authors=[a.get("name", "") for a in (paper.get("authors") or [])],
            abstract=paper.get("abstract", "") or "",
            year=paper.get("year") or 0,
            sources={"semantic_scholar": paper.get("paperId", "")},
            venue=paper.get("venue", "") or "",
            citations=paper.get("citationCount", 0) or 0,
            doi=external_ids.get("DOI", "") or "",
            pdf_urls=[open_access["url"]] if open_access.get("url") else [],
        )

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **filters,
    ) -> list[Paper]:
        """Search Semantic Scholar and return a list of Paper objects.

        Supported filters:
          year (str): year or year range, e.g. "2020" or "2020-2024"
          fields_of_study (list[str]): e.g. ["Computer Science"]
          min_citations (int): post-filter — drop papers below this threshold
        """
        year: Optional[str] = filters.get("year")
        fields_of_study: Optional[list[str]] = filters.get("fields_of_study")
        min_citations: Optional[int] = filters.get("min_citations")

        # Fetch extra results when post-filtering by citations
        fetch_limit = min(max_results * 2, 100) if min_citations else min(max_results, 100)

        params: dict = {
            "query": query,
            "limit": fetch_limit,
            "fields": _FIELDS,
        }
        if year:
            params["year"] = year
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        result = await self._api_request("paper/search", params)

        if "error" in result:
            return []

        papers: list[Paper] = []
        for raw in result.get("data", []):
            paper = self._convert(raw)
            if min_citations is not None and paper.citations < min_citations:
                continue
            papers.append(paper)

        # Sort by citation count (most impactful first)
        papers.sort(key=lambda p: p.citations, reverse=True)
        return papers[:max_results]
