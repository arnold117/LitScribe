"""arXiv search service — ported from src/services/arxiv.py.

Rate limiting strategy:
- Global singleton client (reuses HTTP connection)
- 3.5s cooldown between requests (arXiv asks for >= 3s)
- Client-level retry with 5s delay (handles transient errors)
- Cooldown enforced in blocking thread (run_in_executor)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import arxiv

from litscribe.models.paper import Paper
from litscribe.services.base import SearchService

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


def _wait_cooldown() -> None:
    """Block until cooldown period has passed since last request."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _ARXIV_COOLDOWN:
        wait = _ARXIV_COOLDOWN - elapsed
        logger.debug(f"arXiv cooldown: waiting {wait:.1f}s")
        time.sleep(wait)
    _last_request_time = time.monotonic()


def _result_to_paper(result: arxiv.Result) -> Paper:
    """Convert arxiv.Result to v2 Paper model."""
    raw_id = result.entry_id.split("/")[-1].split("v")[0]
    return Paper(
        paper_id=f"arxiv:{raw_id}",
        title=result.title.replace("\n", " "),
        authors=[a.name for a in result.authors],
        abstract=result.summary.replace("\n", " ") if result.summary else "",
        year=result.published.year if result.published else 0,
        sources={"arxiv": raw_id},
        venue=", ".join(result.categories) if result.categories else "",
        doi=result.doi or "",
        pdf_urls=[result.pdf_url] if result.pdf_url else [],
    )


def _do_search(
    query: str,
    max_results: int,
    sort_criterion: arxiv.SortCriterion,
    sort_order: arxiv.SortOrder,
) -> list[arxiv.Result]:
    """Blocking search executed in a thread executor."""
    _wait_cooldown()
    client = _get_client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_criterion,
        sort_order=sort_order,
    )
    return list(client.results(search))


class ArxivService:
    """arXiv search service implementing the SearchService protocol."""

    source_name = "arxiv"

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **filters,
    ) -> list[Paper]:
        """Search arXiv for papers matching the query.

        Supported **filters:
            sort_by (str): "relevance" | "lastUpdatedDate" | "submittedDate"
            sort_order (str): "ascending" | "descending"
            category (str): arXiv category filter, e.g. "cs.CL"
            year_from (int): earliest submission year (inclusive)
            year_to (int): latest submission year (inclusive)
        """
        max_results = min(max_results, 100)

        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        order_map = {
            "ascending": arxiv.SortOrder.Ascending,
            "descending": arxiv.SortOrder.Descending,
        }

        sort_criterion = sort_map.get(
            filters.get("sort_by", "relevance"), arxiv.SortCriterion.Relevance
        )
        sort_order = order_map.get(
            filters.get("sort_order", "descending"), arxiv.SortOrder.Descending
        )

        # Build query with optional category filter
        category = filters.get("category")
        if category:
            query = f"cat:{category} AND ({query})"

        # Add date range filter via arXiv query syntax
        year_from = filters.get("year_from")
        year_to = filters.get("year_to")
        if year_from or year_to:
            date_from = f"{year_from}0101" if year_from else "19000101"
            date_to = f"{year_to}1231" if year_to else "20991231"
            query = f"({query}) AND submittedDate:[{date_from} TO {date_to}]"

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, _do_search, query, max_results, sort_criterion, sort_order
        )

        return [_result_to_paper(r) for r in results]


# Verify protocol compliance at import time (type-checker friendly)
assert isinstance(ArxivService(), SearchService)
