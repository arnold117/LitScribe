"""Parsed document cache for LitScribe.

Caches PDF parsing results to avoid re-parsing.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import aiosqlite

from cache.database import CacheDatabase, get_cache_db

logger = logging.getLogger(__name__)


class ParseCache:
    """Cache for parsed PDF documents."""

    def __init__(self, db: Optional[CacheDatabase] = None):
        """Initialize parse cache.

        Args:
            db: CacheDatabase instance
        """
        self.db = db or get_cache_db()

    def _serialize_parsed(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize parsed document for storage.

        Args:
            parsed: Parsed document dict

        Returns:
            Serialized dict with JSON strings for complex fields
        """
        return {
            "markdown": parsed.get("markdown", ""),
            "sections": json.dumps(parsed.get("sections", [])),
            "tables": json.dumps(parsed.get("tables", [])),
            "equations": json.dumps(parsed.get("equations", [])),
            "references": json.dumps(parsed.get("references", [])),
            "word_count": parsed.get("word_count", 0),
            "page_count": parsed.get("page_count", 0),
        }

    def _deserialize_parsed(self, row: aiosqlite.Row) -> Dict[str, Any]:
        """Deserialize parsed document from storage.

        Args:
            row: Database row

        Returns:
            Parsed document dict
        """
        return {
            "paper_id": row["paper_id"],
            "markdown": row["markdown"] or "",
            "sections": json.loads(row["sections"]) if row["sections"] else [],
            "tables": json.loads(row["tables"]) if row["tables"] else [],
            "equations": json.loads(row["equations"]) if row["equations"] else [],
            "references": json.loads(row["references"]) if row["references"] else [],
            "word_count": row["word_count"] or 0,
            "page_count": row["page_count"] or 0,
            "parsed_at": row["parsed_at"],
        }

    # Synchronous methods

    def get(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get cached parsed document (sync).

        Args:
            paper_id: Paper identifier

        Returns:
            Parsed document dict or None
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM parsed_docs WHERE paper_id = ?",
                (paper_id,)
            )
            row = cursor.fetchone()
            if row:
                logger.debug(f"Parse cache hit for {paper_id}")
                return self._deserialize_parsed(row)

        logger.debug(f"Parse cache miss for {paper_id}")
        return None

    def exists(self, paper_id: str) -> bool:
        """Check if parsed document is cached (sync).

        Args:
            paper_id: Paper identifier

        Returns:
            True if cached
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM parsed_docs WHERE paper_id = ?",
                (paper_id,)
            )
            return cursor.fetchone() is not None

    def save(self, paper_id: str, parsed: Dict[str, Any]) -> None:
        """Save parsed document to cache (sync).

        Args:
            paper_id: Paper identifier
            parsed: Parsed document dict
        """
        self.db.init_schema()
        serialized = self._serialize_parsed(parsed)

        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO parsed_docs (
                    paper_id, markdown, sections, tables, equations,
                    references, word_count, page_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    markdown = excluded.markdown,
                    sections = excluded.sections,
                    tables = excluded.tables,
                    equations = excluded.equations,
                    references = excluded.references,
                    word_count = excluded.word_count,
                    page_count = excluded.page_count,
                    parsed_at = datetime('now')
            """, (
                paper_id,
                serialized["markdown"],
                serialized["sections"],
                serialized["tables"],
                serialized["equations"],
                serialized["references"],
                serialized["word_count"],
                serialized["page_count"],
            ))
            conn.commit()

        logger.info(f"Cached parsed document for {paper_id} ({serialized['word_count']} words)")

    def remove(self, paper_id: str) -> bool:
        """Remove parsed document from cache (sync).

        Args:
            paper_id: Paper identifier

        Returns:
            True if removed
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM parsed_docs WHERE paper_id = ?",
                (paper_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_markdown(self, paper_id: str) -> Optional[str]:
        """Get just the markdown content (sync).

        Args:
            paper_id: Paper identifier

        Returns:
            Markdown string or None
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT markdown FROM parsed_docs WHERE paper_id = ?",
                (paper_id,)
            )
            row = cursor.fetchone()
            if row:
                return row["markdown"]
        return None

    def get_sections(self, paper_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get parsed sections (sync).

        Args:
            paper_id: Paper identifier

        Returns:
            List of sections or None
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT sections FROM parsed_docs WHERE paper_id = ?",
                (paper_id,)
            )
            row = cursor.fetchone()
            if row and row["sections"]:
                return json.loads(row["sections"])
        return None

    def get_references(self, paper_id: str) -> Optional[List[str]]:
        """Get parsed references (sync).

        Args:
            paper_id: Paper identifier

        Returns:
            List of references or None
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT references FROM parsed_docs WHERE paper_id = ?",
                (paper_id,)
            )
            row = cursor.fetchone()
            if row and row["references"]:
                return json.loads(row["references"])
        return None

    def get_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all cached parsed documents (sync).

        Args:
            limit: Maximum entries

        Returns:
            List of parsed documents (summary only, no full markdown)
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT paper_id, word_count, page_count, parsed_at
                FROM parsed_docs
                ORDER BY parsed_at DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    # Asynchronous methods

    async def get_async(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get cached parsed document (async).

        Args:
            paper_id: Paper identifier

        Returns:
            Parsed document dict or None
        """
        await self.db.init_schema_async()

        async with aiosqlite.connect(self.db.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM parsed_docs WHERE paper_id = ?",
                (paper_id,)
            )
            row = await cursor.fetchone()
            if row:
                logger.debug(f"Parse cache hit for {paper_id}")
                return self._deserialize_parsed(row)

        logger.debug(f"Parse cache miss for {paper_id}")
        return None

    async def exists_async(self, paper_id: str) -> bool:
        """Check if parsed document is cached (async).

        Args:
            paper_id: Paper identifier

        Returns:
            True if cached
        """
        await self.db.init_schema_async()

        async with aiosqlite.connect(self.db.db_path) as db:
            cursor = await db.execute(
                "SELECT 1 FROM parsed_docs WHERE paper_id = ?",
                (paper_id,)
            )
            return await cursor.fetchone() is not None

    async def save_async(self, paper_id: str, parsed: Dict[str, Any]) -> None:
        """Save parsed document to cache (async).

        Args:
            paper_id: Paper identifier
            parsed: Parsed document dict
        """
        await self.db.init_schema_async()
        serialized = self._serialize_parsed(parsed)

        async with aiosqlite.connect(self.db.db_path) as db:
            await db.execute("""
                INSERT INTO parsed_docs (
                    paper_id, markdown, sections, tables, equations,
                    references, word_count, page_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    markdown = excluded.markdown,
                    sections = excluded.sections,
                    tables = excluded.tables,
                    equations = excluded.equations,
                    references = excluded.references,
                    word_count = excluded.word_count,
                    page_count = excluded.page_count,
                    parsed_at = datetime('now')
            """, (
                paper_id,
                serialized["markdown"],
                serialized["sections"],
                serialized["tables"],
                serialized["equations"],
                serialized["references"],
                serialized["word_count"],
                serialized["page_count"],
            ))
            await db.commit()

        logger.info(f"Cached parsed document for {paper_id}")

    async def remove_async(self, paper_id: str) -> bool:
        """Remove parsed document from cache (async).

        Args:
            paper_id: Paper identifier

        Returns:
            True if removed
        """
        await self.db.init_schema_async()

        async with aiosqlite.connect(self.db.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM parsed_docs WHERE paper_id = ?",
                (paper_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def get_markdown_async(self, paper_id: str) -> Optional[str]:
        """Get just the markdown content (async).

        Args:
            paper_id: Paper identifier

        Returns:
            Markdown string or None
        """
        await self.db.init_schema_async()

        async with aiosqlite.connect(self.db.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT markdown FROM parsed_docs WHERE paper_id = ?",
                (paper_id,)
            )
            row = await cursor.fetchone()
            if row:
                return row["markdown"]
        return None


# Export
__all__ = ["ParseCache"]
