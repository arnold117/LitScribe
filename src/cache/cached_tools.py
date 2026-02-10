"""Cached versions of agent tools for LitScribe.

Wraps the existing tools with caching functionality for:
- Search results (local-first: SQLite cache -> Zotero -> External API)
- Paper metadata
- PDF downloads
- Parsed documents
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from cache.database import get_cache_db
from cache.paper_cache import PaperCache
from cache.parse_cache import ParseCache
from cache.pdf_cache import PDFCache
from cache.search_cache import SearchCache
from utils.config import Config

logger = logging.getLogger(__name__)


def _zotero_available() -> bool:
    """Check if Zotero API is configured."""
    return bool(Config.ZOTERO_API_KEY and Config.ZOTERO_LIBRARY_ID)


async def resolve_zotero_collection(value: str) -> Optional[str]:
    """Resolve a Zotero collection value to a collection key.

    If value looks like an 8-char alphanumeric key (e.g. 'ABC123XY'), use it directly.
    Otherwise treat it as a collection name and look up or create it.

    Args:
        value: Collection key or name

    Returns:
        Collection key string, or None if resolution fails
    """
    if not _zotero_available():
        logger.warning("Zotero not configured, cannot resolve collection")
        return None

    # Heuristic: Zotero keys are exactly 8 alphanumeric characters
    if len(value) == 8 and value.isalnum():
        return value

    # Treat as collection name — look up or create
    try:
        from services.zotero import create_or_get_collection
        result = await create_or_get_collection(name=value)
        if "error" in result:
            logger.error(f"Failed to resolve Zotero collection '{value}': {result['error']}")
            return None
        key = result.get("key", "")
        if result.get("created"):
            logger.info(f"Created new Zotero collection '{value}' (key: {key})")
        else:
            logger.info(f"Found existing Zotero collection '{value}' (key: {key})")
        return key
    except Exception as e:
        logger.error(f"Failed to resolve Zotero collection '{value}': {e}")
        return None


def _zotero_item_to_paper(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Zotero item dict to unified paper format.

    Args:
        item: Zotero item dict (from ZoteroItem.to_dict())

    Returns:
        Paper dict in unified format compatible with the agent pipeline
    """
    # Extract author names from creators
    authors = []
    for creator in item.get("creators", []):
        if creator.get("creatorType") == "author":
            name = f"{creator.get('firstName', '')} {creator.get('lastName', '')}".strip()
            if name:
                authors.append(name)

    # Parse year from date
    year = 0
    date_str = item.get("date", "")
    if date_str:
        year_match = re.search(r"(\d{4})", date_str)
        if year_match:
            year = int(year_match.group(1))

    paper_id = item.get("doi") or f"zotero:{item.get('key', '')}"

    return {
        "paper_id": paper_id,
        "title": item.get("title", ""),
        "authors": authors,
        "abstract": item.get("abstract", ""),
        "year": year,
        "venue": item.get("publication_title", ""),
        "citations": 0,
        "doi": item.get("doi"),
        "arxiv_id": item.get("arxiv_id"),
        "url": item.get("url", ""),
        "source": "zotero",
        "zotero_key": item.get("key", ""),
        "search_origin": "zotero_local",
    }


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

    async def search_zotero_first(
        self,
        query: str,
        zotero_collection: Optional[str] = None,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """Search Zotero personal library as a priority local source.

        This is called before external APIs to leverage papers the user
        already has in their library.

        Args:
            query: Search query
            zotero_collection: Specific Zotero collection to search (optional)
            limit: Maximum results

        Returns:
            List of papers in unified format from Zotero
        """
        if not _zotero_available():
            return []

        try:
            from services.zotero import search_items

            result = await search_items(
                query=query,
                collection=zotero_collection,
                limit=limit,
            )
            items = result.get("items", [])
            papers = [_zotero_item_to_paper(item) for item in items]

            if papers:
                logger.info(
                    f"Zotero local search for '{query}': {len(papers)} papers found"
                )
                # Cache Zotero papers
                if self.cache_enabled:
                    await self.paper_cache.save_many_async(papers)

            return papers

        except Exception as e:
            logger.warning(f"Zotero local search failed: {e}")
            return []

    async def search_with_cache(
        self,
        query: str,
        sources: List[str],
        max_per_source: int = 20,
        force_refresh: bool = False,
        zotero_collection: Optional[str] = None,
        arxiv_categories: Optional[List[str]] = None,
        s2_fields: Optional[List[str]] = None,
        pubmed_mesh: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search with local-first strategy: SQLite cache -> Zotero -> External API.

        Search order:
        1. SQLite cache — check cached results per source
        2. Zotero library — search personal library (if configured)
        3. External APIs — only fetch from sources not covered by cache

        Args:
            query: Search query
            sources: List of sources to search
            max_per_source: Max results per source
            force_refresh: If True, bypass cache
            zotero_collection: Zotero collection to search (optional)
            arxiv_categories: arXiv category filters (e.g. ["q-bio.BM"])
            s2_fields: Semantic Scholar field filters (e.g. ["Biology"])
            pubmed_mesh: PubMed MeSH term filters (e.g. ["Alkaloids"])

        Returns:
            Combined search results with origin tracking
        """
        from agents.tools import unified_search

        if not self.cache_enabled or force_refresh:
            # Still try Zotero first even without cache
            zotero_papers = await self.search_zotero_first(
                query, zotero_collection
            )

            result = await unified_search(
                query=query,
                sources=sources,
                max_per_source=max_per_source,
                deduplicate=True,
                arxiv_categories=arxiv_categories,
                s2_fields=s2_fields,
                pubmed_mesh=pubmed_mesh,
            )
            all_papers = zotero_papers + result.get("papers", [])

            if self.cache_enabled:
                await self._cache_search_results(query, sources, result)

            unique_papers = self._deduplicate_papers(all_papers)
            source_counts = dict(result.get("source_counts", {}))
            if zotero_papers:
                source_counts["zotero"] = len(zotero_papers)
            return {
                "papers": unique_papers,
                "total_found": len(unique_papers),
                "from_cache": 0,
                "from_zotero": len(zotero_papers),
                "source_counts": source_counts,
                "sources_fetched": sources,
            }

        # === Step 1: Check SQLite cache for each source ===
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

        # === Step 2: Search Zotero library (always, if configured) ===
        zotero_papers = await self.search_zotero_first(
            query, zotero_collection
        )

        # === Step 3: Fetch from external APIs (only uncached sources) ===
        fetched_source_counts = {}
        if sources_to_fetch:
            logger.info(f"Fetching from {sources_to_fetch} (not cached)")
            result = await unified_search(
                query=query,
                sources=sources_to_fetch,
                max_per_source=max_per_source,
                deduplicate=True,
                arxiv_categories=arxiv_categories,
                s2_fields=s2_fields,
                pubmed_mesh=pubmed_mesh,
            )
            new_papers = result.get("papers", [])
            fetched_source_counts = dict(result.get("source_counts", {}))

            # Cache the new results by source
            source_papers = {}
            for paper in new_papers:
                src = paper.get("source", "unknown")
                if src not in source_papers:
                    source_papers[src] = []
                source_papers[src].append(paper)

            for src, papers in source_papers.items():
                await self.search_cache.save_async(query, src, papers, len(papers))
                await self.paper_cache.save_many_async(papers)

            cached_papers.extend(new_papers)

        # === Step 4: Merge and deduplicate (Zotero papers first for priority) ===
        all_papers = zotero_papers + cached_papers
        unique_papers = self._deduplicate_papers(all_papers)

        # Build source_counts from fetched results + Zotero
        source_counts = fetched_source_counts
        if zotero_papers:
            source_counts["zotero"] = len(zotero_papers)

        return {
            "papers": unique_papers,
            "total_found": len(unique_papers),
            "from_cache": len(sources) - len(sources_to_fetch),
            "from_zotero": len(zotero_papers),
            "source_counts": source_counts,
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

        Papers earlier in the list have priority (Zotero > cache > external).

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
        """Get PDF with caching support. Checks Zotero local storage first.

        Search order: PDF cache -> Zotero local storage -> arXiv download

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

        pdf_path = None
        zotero_key = paper.get("zotero_key")
        arxiv_id = paper.get("arxiv_id")

        # Try Zotero local storage first (no network needed)
        if zotero_key:
            try:
                result = await get_zotero_pdf_path(zotero_key)
                local_path = result.get("local_path")
                if local_path and Path(local_path).exists():
                    pdf_path = local_path
                    logger.info(f"PDF found in Zotero local storage: {zotero_key}")
            except Exception as e:
                logger.warning(f"Failed to get Zotero PDF {zotero_key}: {e}")

        # Fall back to arXiv download
        if not pdf_path and arxiv_id:
            try:
                result = await download_arxiv_pdf(arxiv_id)
                pdf_path = result.get("pdf_path")
            except Exception as e:
                logger.warning(f"Failed to download arXiv PDF {arxiv_id}: {e}")

        # Try Unpaywall (legal OA lookup by DOI)
        if not pdf_path:
            doi = paper.get("doi")
            if doi:
                try:
                    from services.unpaywall import get_oa_pdf_url
                    from agents.critical_reading_agent import _download_pdf_from_url
                    oa_url = await get_oa_pdf_url(doi)
                    if oa_url:
                        downloaded = await _download_pdf_from_url(oa_url, paper_id)
                        if downloaded:
                            pdf_path = downloaded
                            logger.info(f"Downloaded PDF via Unpaywall for {paper_id}")
                except Exception as e:
                    logger.warning(f"Unpaywall lookup failed for {paper_id}: {e}")

        # Try PMC (PubMed Central free full text)
        if not pdf_path:
            pmc_id = paper.get("pmc_id")
            if pmc_id:
                try:
                    from agents.critical_reading_agent import _download_pdf_from_url
                    pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"
                    downloaded = await _download_pdf_from_url(pmc_url, paper_id)
                    if downloaded:
                        pdf_path = downloaded
                        logger.info(f"Downloaded PDF from PMC for {paper_id} ({pmc_id})")
                except Exception as e:
                    logger.warning(f"PMC PDF download failed for {pmc_id}: {e}")

        # Fall back to direct URL download (e.g. Semantic Scholar openAccessPdf)
        if not pdf_path:
            pdf_urls = paper.get("pdf_urls") or []
            single_url = paper.get("pdf_url")
            if single_url and single_url not in pdf_urls:
                pdf_urls.append(single_url)
            for url in pdf_urls:
                try:
                    from agents.critical_reading_agent import _download_pdf_from_url
                    downloaded = await _download_pdf_from_url(url, paper_id)
                    if downloaded:
                        pdf_path = downloaded
                        logger.info(f"Downloaded PDF from URL for {paper_id}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to download PDF from {url}: {e}")

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
