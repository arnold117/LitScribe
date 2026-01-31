"""Paper metadata cache for LitScribe.

Caches paper metadata to avoid repeated API calls.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite

from cache.database import CacheDatabase, get_cache_db

logger = logging.getLogger(__name__)


class PaperCache:
    """Cache for paper metadata."""

    def __init__(self, db: Optional[CacheDatabase] = None):
        """Initialize paper cache.

        Args:
            db: CacheDatabase instance. Uses global instance if not provided.
        """
        self.db = db or get_cache_db()

    def _normalize_paper_id(self, paper_id: str, source: Optional[str] = None) -> str:
        """Normalize paper ID format.

        Args:
            paper_id: Raw paper ID
            source: Source name (arxiv, pubmed, semantic_scholar)

        Returns:
            Normalized paper ID with source prefix
        """
        # Already has prefix
        if ":" in paper_id and paper_id.split(":")[0] in ["arxiv", "pmid", "s2", "doi"]:
            return paper_id

        # Detect source from ID format
        if source:
            source_lower = source.lower()
            if source_lower in ["arxiv", "arx"]:
                return f"arxiv:{paper_id}"
            elif source_lower in ["pubmed", "pmid"]:
                return f"pmid:{paper_id}"
            elif source_lower in ["semantic_scholar", "s2"]:
                return f"s2:{paper_id}"
            elif source_lower == "doi":
                return f"doi:{paper_id}"

        # Try to infer from format
        if paper_id.startswith("10."):  # DOI format
            return f"doi:{paper_id}"
        elif paper_id.isdigit() and len(paper_id) >= 7:  # PMID
            return f"pmid:{paper_id}"

        return paper_id

    def _paper_to_dict(self, row: aiosqlite.Row) -> Dict[str, Any]:
        """Convert database row to paper dictionary.

        Args:
            row: Database row

        Returns:
            Paper dictionary
        """
        paper = dict(row)
        # Parse JSON fields
        if paper.get("authors"):
            paper["authors"] = json.loads(paper["authors"])
        if paper.get("metadata"):
            paper["metadata"] = json.loads(paper["metadata"])
        return paper

    # Synchronous methods

    def get(self, paper_id: str, source: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get paper by ID (sync).

        Args:
            paper_id: Paper identifier
            source: Source name for ID normalization

        Returns:
            Paper dictionary or None if not found
        """
        self.db.init_schema()
        normalized_id = self._normalize_paper_id(paper_id, source)

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM papers WHERE paper_id = ?",
                (normalized_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._paper_to_dict(row)
        return None

    def save(self, paper: Dict[str, Any]) -> str:
        """Save paper to cache (sync).

        Args:
            paper: Paper dictionary with at least 'paper_id' and 'title'

        Returns:
            Normalized paper ID
        """
        self.db.init_schema()

        paper_id = self._normalize_paper_id(
            paper.get("paper_id", paper.get("id", "")),
            paper.get("source")
        )

        authors = paper.get("authors", [])
        if isinstance(authors, list):
            authors = json.dumps(authors)

        metadata = paper.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)

        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO papers (
                    paper_id, title, authors, abstract, year, venue,
                    citations, source, doi, pdf_url, metadata, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(paper_id) DO UPDATE SET
                    title = excluded.title,
                    authors = COALESCE(excluded.authors, authors),
                    abstract = COALESCE(excluded.abstract, abstract),
                    year = COALESCE(excluded.year, year),
                    venue = COALESCE(excluded.venue, venue),
                    citations = COALESCE(excluded.citations, citations),
                    doi = COALESCE(excluded.doi, doi),
                    pdf_url = COALESCE(excluded.pdf_url, pdf_url),
                    metadata = COALESCE(excluded.metadata, metadata),
                    updated_at = datetime('now')
            """, (
                paper_id,
                paper.get("title", ""),
                authors,
                paper.get("abstract"),
                paper.get("year"),
                paper.get("venue"),
                paper.get("citations"),
                paper.get("source"),
                paper.get("doi"),
                paper.get("pdf_url"),
                metadata,
            ))
            conn.commit()

        logger.debug(f"Cached paper: {paper_id}")
        return paper_id

    def save_many(self, papers: List[Dict[str, Any]]) -> int:
        """Save multiple papers to cache (sync).

        Args:
            papers: List of paper dictionaries

        Returns:
            Number of papers saved
        """
        count = 0
        for paper in papers:
            try:
                self.save(paper)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to cache paper: {e}")
        return count

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Full-text search papers (sync).

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching papers
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT papers.*
                FROM papers_fts
                JOIN papers ON papers_fts.paper_id = papers.paper_id
                WHERE papers_fts MATCH ?
                LIMIT ?
            """, (query, limit))
            return [self._paper_to_dict(row) for row in cursor.fetchall()]

    def get_by_source(self, source: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get papers by source (sync).

        Args:
            source: Source name
            limit: Maximum results

        Returns:
            List of papers from the source
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM papers WHERE source = ? ORDER BY created_at DESC LIMIT ?",
                (source, limit)
            )
            return [self._paper_to_dict(row) for row in cursor.fetchall()]

    def exists(self, paper_id: str, source: Optional[str] = None) -> bool:
        """Check if paper exists in cache (sync).

        Args:
            paper_id: Paper identifier
            source: Source name

        Returns:
            True if paper is cached
        """
        return self.get(paper_id, source) is not None

    # Asynchronous methods

    async def get_async(self, paper_id: str, source: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get paper by ID (async).

        Args:
            paper_id: Paper identifier
            source: Source name

        Returns:
            Paper dictionary or None
        """
        await self.db.init_schema_async()
        normalized_id = self._normalize_paper_id(paper_id, source)

        async with aiosqlite.connect(self.db.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM papers WHERE paper_id = ?",
                (normalized_id,)
            )
            row = await cursor.fetchone()
            if row:
                return self._paper_to_dict(row)
        return None

    async def save_async(self, paper: Dict[str, Any]) -> str:
        """Save paper to cache (async).

        Args:
            paper: Paper dictionary

        Returns:
            Normalized paper ID
        """
        await self.db.init_schema_async()

        paper_id = self._normalize_paper_id(
            paper.get("paper_id", paper.get("id", "")),
            paper.get("source")
        )

        authors = paper.get("authors", [])
        if isinstance(authors, list):
            authors = json.dumps(authors)

        metadata = paper.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)

        async with aiosqlite.connect(self.db.db_path) as db:
            await db.execute("""
                INSERT INTO papers (
                    paper_id, title, authors, abstract, year, venue,
                    citations, source, doi, pdf_url, metadata, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(paper_id) DO UPDATE SET
                    title = excluded.title,
                    authors = COALESCE(excluded.authors, authors),
                    abstract = COALESCE(excluded.abstract, abstract),
                    year = COALESCE(excluded.year, year),
                    venue = COALESCE(excluded.venue, venue),
                    citations = COALESCE(excluded.citations, citations),
                    doi = COALESCE(excluded.doi, doi),
                    pdf_url = COALESCE(excluded.pdf_url, pdf_url),
                    metadata = COALESCE(excluded.metadata, metadata),
                    updated_at = datetime('now')
            """, (
                paper_id,
                paper.get("title", ""),
                authors,
                paper.get("abstract"),
                paper.get("year"),
                paper.get("venue"),
                paper.get("citations"),
                paper.get("source"),
                paper.get("doi"),
                paper.get("pdf_url"),
                metadata,
            ))
            await db.commit()

        logger.debug(f"Cached paper: {paper_id}")
        return paper_id

    async def save_many_async(self, papers: List[Dict[str, Any]]) -> int:
        """Save multiple papers (async).

        Args:
            papers: List of paper dictionaries

        Returns:
            Number saved
        """
        count = 0
        for paper in papers:
            try:
                await self.save_async(paper)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to cache paper: {e}")
        return count

    async def exists_async(self, paper_id: str, source: Optional[str] = None) -> bool:
        """Check if paper exists (async).

        Args:
            paper_id: Paper identifier
            source: Source name

        Returns:
            True if cached
        """
        paper = await self.get_async(paper_id, source)
        return paper is not None


# Export
__all__ = ["PaperCache"]
