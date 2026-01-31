"""Cached versions of agent tools for LitScribe.

Wraps the existing tools with caching functionality for:
- Search results
- Paper metadata
- PDF downloads
- Parsed documents
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from cache.database import get_cache_db
from cache.paper_cache import PaperCache
from cache.parse_cache import ParseCache
from cache.pdf_cache import PDFCache
from cache.search_cache import SearchCache

logger = logging.getLogger(__name__)


class CachedTools:
    """Cached wrapper for agent tools."""

    def __init__(self, cache_enabled: bool = True, ttl_hours: int = 24):
        """Initialize cached tools.

        Args:
            cache_enabled: Whether caching is enabled
            ttl_hours: TTL for search cache
        """
        self.cache_enabled = cache_enabled
        if cache_enabled:
            db = get_cache_db()
            self.paper_cache = PaperCache(db)
            self.search_cache = SearchCache(db, ttl_hours=ttl_hours)
            self.pdf_cache = PDFCache(db)
            self.parse_cache = ParseCache(db)
        else:
            self.paper_cache = None
            self.search_cache = None
            self.pdf_cache = None
            self.parse_cache = None

    async def search_with_cache(
        self,
        query: str,
        sources: List[str],
        max_per_source: int = 20,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Search with caching support.

        Args:
            query: Search query
            sources: List of sources to search
            max_per_source: Max results per source
            force_refresh: If True, bypass cache

        Returns:
            Combined search results
        """
        from agents.tools import unified_search

        if not self.cache_enabled or force_refresh:
            result = await unified_search(
                query=query,
                sources=sources,
                max_per_source=max_per_source,
                deduplicate=True,
            )
            if self.cache_enabled:
                # Cache the results
                await self._cache_search_results(query, sources, result)
            return result

        # Check cache for each source
        cached_papers = []
        sources_to_fetch = []

        for source in sources:
            cached = await self.search_cache.get_async(query, source)
            if cached:
                results, total = cached
                cached_papers.extend(results)
                logger.info(f"Cache hit for '{query}' on {source}: {len(results)} papers")
            else:
                sources_to_fetch.append(source)

        # Fetch missing sources
        if sources_to_fetch:
            logger.info(f"Fetching from {sources_to_fetch} (not cached)")
            result = await unified_search(
                query=query,
                sources=sources_to_fetch,
                max_per_source=max_per_source,
                deduplicate=True,
            )
            new_papers = result.get("papers", [])

            # Cache the new results by source
            source_papers = {}
            for paper in new_papers:
                src = paper.get("source", "unknown")
                if src not in source_papers:
                    source_papers[src] = []
                source_papers[src].append(paper)

            for src, papers in source_papers.items():
                await self.search_cache.save_async(query, src, papers, len(papers))
                # Also cache paper metadata
                await self.paper_cache.save_many_async(papers)

            cached_papers.extend(new_papers)

        # Deduplicate combined results
        unique_papers = self._deduplicate_papers(cached_papers)

        return {
            "papers": unique_papers,
            "total_found": len(unique_papers),
            "from_cache": len(sources) - len(sources_to_fetch),
            "sources_fetched": sources_to_fetch,
        }

    async def _cache_search_results(
        self,
        query: str,
        sources: List[str],
        result: Dict[str, Any]
    ) -> None:
        """Cache search results by source.

        Args:
            query: Search query
            sources: Sources that were searched
            result: Search result dict
        """
        papers = result.get("papers", [])

        # Group by source
        source_papers = {}
        for paper in papers:
            src = paper.get("source", "unknown")
            if src not in source_papers:
                source_papers[src] = []
            source_papers[src].append(paper)

        # Cache each source separately
        for src, src_papers in source_papers.items():
            await self.search_cache.save_async(query, src, src_papers, len(src_papers))

        # Cache paper metadata
        await self.paper_cache.save_many_async(papers)

    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate papers by ID or title.

        Args:
            papers: List of papers

        Returns:
            Deduplicated list
        """
        seen_ids = set()
        seen_titles = set()
        unique = []

        for paper in papers:
            paper_id = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("doi")
            title = paper.get("title", "").lower().strip()

            if paper_id and paper_id in seen_ids:
                continue
            if title and title in seen_titles:
                continue

            if paper_id:
                seen_ids.add(paper_id)
            if title:
                seen_titles.add(title)
            unique.append(paper)

        return unique

    async def get_pdf_with_cache(
        self,
        paper: Dict[str, Any],
        force_download: bool = False,
    ) -> Optional[Path]:
        """Get PDF with caching support.

        Args:
            paper: Paper metadata
            force_download: If True, re-download even if cached

        Returns:
            Path to PDF file or None
        """
        from agents.tools import download_arxiv_pdf, get_zotero_pdf_path

        paper_id = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("doi")
        if not paper_id:
            return None

        # Check cache first
        if self.cache_enabled and not force_download:
            cached_path = await self.pdf_cache.get_path_async(paper_id)
            if cached_path and cached_path.exists():
                logger.info(f"PDF cache hit for {paper_id}")
                return cached_path

        # Download PDF
        pdf_path = None
        arxiv_id = paper.get("arxiv_id")
        zotero_key = paper.get("zotero_key")

        if arxiv_id:
            try:
                result = await download_arxiv_pdf(arxiv_id)
                pdf_path = result.get("pdf_path")
            except Exception as e:
                logger.warning(f"Failed to download arXiv PDF {arxiv_id}: {e}")

        if not pdf_path and zotero_key:
            try:
                result = await get_zotero_pdf_path(zotero_key)
                pdf_path = result.get("pdf_path")
            except Exception as e:
                logger.warning(f"Failed to get Zotero PDF {zotero_key}: {e}")

        # Cache the PDF path
        if pdf_path and self.cache_enabled:
            pdf_path = Path(pdf_path)
            if pdf_path.exists():
                await self.pdf_cache.save_async(paper_id, pdf_path)
                return pdf_path

        return Path(pdf_path) if pdf_path else None

    async def parse_pdf_with_cache(
        self,
        paper_id: str,
        pdf_path: Path,
        force_reparse: bool = False,
    ) -> Dict[str, Any]:
        """Parse PDF with caching support.

        Args:
            paper_id: Paper identifier
            pdf_path: Path to PDF file
            force_reparse: If True, re-parse even if cached

        Returns:
            Parsed document dict
        """
        from agents.tools import parse_pdf

        # Check cache first
        if self.cache_enabled and not force_reparse:
            cached = await self.parse_cache.get_async(paper_id)
            if cached:
                logger.info(f"Parse cache hit for {paper_id}")
                return cached

        # Parse the PDF
        result = await parse_pdf(str(pdf_path), backend="pymupdf")

        # Cache the result
        if self.cache_enabled:
            await self.parse_cache.save_async(paper_id, result)

        return result

    async def get_paper_from_cache(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get paper metadata from cache.

        Args:
            paper_id: Paper identifier

        Returns:
            Paper dict or None
        """
        if not self.cache_enabled:
            return None
        return await self.paper_cache.get_async(paper_id)

    async def save_paper_to_cache(self, paper: Dict[str, Any]) -> str:
        """Save paper metadata to cache.

        Args:
            paper: Paper dict

        Returns:
            Normalized paper ID
        """
        if not self.cache_enabled:
            return paper.get("paper_id", "")
        return await self.paper_cache.save_async(paper)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache stats dict
        """
        if not self.cache_enabled:
            return {"cache_enabled": False}

        db = get_cache_db()
        stats = db.get_stats()
        stats["cache_enabled"] = True
        return stats


# Global cached tools instance
_cached_tools: Optional[CachedTools] = None


def get_cached_tools(cache_enabled: bool = True) -> CachedTools:
    """Get or create the global cached tools instance.

    Args:
        cache_enabled: Whether to enable caching

    Returns:
        CachedTools instance
    """
    global _cached_tools
    if _cached_tools is None:
        _cached_tools = CachedTools(cache_enabled=cache_enabled)
    return _cached_tools


# Export
__all__ = ["CachedTools", "get_cached_tools"]
