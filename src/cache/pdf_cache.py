"""PDF download cache for LitScribe.

Tracks downloaded PDFs and their locations to avoid re-downloading.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import aiosqlite

from cache.database import CacheDatabase, get_cache_db

logger = logging.getLogger(__name__)

# Default PDF storage directory
DEFAULT_PDF_DIR = Path("data/pdfs")


class PDFCache:
    """Cache for tracking PDF downloads."""

    def __init__(
        self,
        db: Optional[CacheDatabase] = None,
        pdf_dir: Optional[Path] = None
    ):
        """Initialize PDF cache.

        Args:
            db: CacheDatabase instance
            pdf_dir: Directory for storing PDFs
        """
        self.db = db or get_cache_db()
        self.pdf_dir = pdf_dir or Path(os.getenv("LITSCRIBE_DATA_DIR", DEFAULT_PDF_DIR))
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file.

        Args:
            file_path: Path to file

        Returns:
            Hex-encoded SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _normalize_paper_id(self, paper_id: str) -> str:
        """Normalize paper ID for consistency.

        Args:
            paper_id: Paper identifier

        Returns:
            Normalized ID
        """
        return paper_id.replace("/", "_").replace(":", "_")

    def get_pdf_path(self, paper_id: str) -> Path:
        """Get expected PDF path for a paper.

        Args:
            paper_id: Paper identifier

        Returns:
            Path where PDF would be stored
        """
        normalized = self._normalize_paper_id(paper_id)
        return self.pdf_dir / f"{normalized}.pdf"

    # Synchronous methods

    def get(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get PDF cache entry (sync).

        Args:
            paper_id: Paper identifier

        Returns:
            Cache entry dict with pdf_path, file_hash, etc. or None
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM pdfs WHERE paper_id = ?",
                (paper_id,)
            )
            row = cursor.fetchone()
            if row:
                entry = dict(row)
                # Verify file still exists
                pdf_path = Path(entry["pdf_path"])
                if pdf_path.exists():
                    return entry
                else:
                    # File was deleted, remove from cache
                    self.remove(paper_id)
        return None

    def exists(self, paper_id: str) -> bool:
        """Check if PDF is cached and file exists (sync).

        Args:
            paper_id: Paper identifier

        Returns:
            True if PDF is cached and file exists
        """
        entry = self.get(paper_id)
        return entry is not None

    def get_path(self, paper_id: str) -> Optional[Path]:
        """Get PDF file path if cached (sync).

        Args:
            paper_id: Paper identifier

        Returns:
            Path to PDF file or None
        """
        entry = self.get(paper_id)
        if entry:
            return Path(entry["pdf_path"])
        return None

    def save(self, paper_id: str, pdf_path: Path) -> Dict[str, Any]:
        """Record PDF download (sync).

        Args:
            paper_id: Paper identifier
            pdf_path: Path to downloaded PDF

        Returns:
            Cache entry dict
        """
        self.db.init_schema()

        file_hash = self._compute_file_hash(pdf_path)
        file_size = pdf_path.stat().st_size

        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO pdfs (paper_id, pdf_path, file_hash, file_size)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    pdf_path = excluded.pdf_path,
                    file_hash = excluded.file_hash,
                    file_size = excluded.file_size,
                    downloaded_at = datetime('now')
            """, (
                paper_id,
                str(pdf_path.absolute()),
                file_hash,
                file_size,
            ))
            conn.commit()

        logger.info(f"Cached PDF for {paper_id}: {pdf_path}")
        return {
            "paper_id": paper_id,
            "pdf_path": str(pdf_path.absolute()),
            "file_hash": file_hash,
            "file_size": file_size,
        }

    def remove(self, paper_id: str, delete_file: bool = False) -> bool:
        """Remove PDF from cache (sync).

        Args:
            paper_id: Paper identifier
            delete_file: If True, also delete the PDF file

        Returns:
            True if entry was removed
        """
        self.db.init_schema()

        entry = self.get(paper_id)
        if entry and delete_file:
            pdf_path = Path(entry["pdf_path"])
            if pdf_path.exists():
                pdf_path.unlink()
                logger.info(f"Deleted PDF file: {pdf_path}")

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM pdfs WHERE paper_id = ?",
                (paper_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def verify_integrity(self, paper_id: str) -> bool:
        """Verify PDF file integrity (sync).

        Args:
            paper_id: Paper identifier

        Returns:
            True if file exists and hash matches
        """
        entry = self.get(paper_id)
        if not entry:
            return False

        pdf_path = Path(entry["pdf_path"])
        if not pdf_path.exists():
            return False

        current_hash = self._compute_file_hash(pdf_path)
        return current_hash == entry["file_hash"]

    def get_all(self, limit: int = 100) -> list:
        """Get all cached PDFs (sync).

        Args:
            limit: Maximum entries to return

        Returns:
            List of cache entries
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM pdfs ORDER BY downloaded_at DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_total_size(self) -> int:
        """Get total size of cached PDFs in bytes (sync).

        Returns:
            Total size in bytes
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT SUM(file_size) FROM pdfs")
            result = cursor.fetchone()[0]
            return result or 0

    # Asynchronous methods

    async def get_async(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get PDF cache entry (async).

        Args:
            paper_id: Paper identifier

        Returns:
            Cache entry or None
        """
        await self.db.init_schema_async()

        async with aiosqlite.connect(self.db.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM pdfs WHERE paper_id = ?",
                (paper_id,)
            )
            row = await cursor.fetchone()
            if row:
                entry = dict(row)
                pdf_path = Path(entry["pdf_path"])
                if pdf_path.exists():
                    return entry
                else:
                    await self.remove_async(paper_id)
        return None

    async def exists_async(self, paper_id: str) -> bool:
        """Check if PDF is cached (async).

        Args:
            paper_id: Paper identifier

        Returns:
            True if cached
        """
        entry = await self.get_async(paper_id)
        return entry is not None

    async def get_path_async(self, paper_id: str) -> Optional[Path]:
        """Get PDF path if cached (async).

        Args:
            paper_id: Paper identifier

        Returns:
            Path or None
        """
        entry = await self.get_async(paper_id)
        if entry:
            return Path(entry["pdf_path"])
        return None

    async def save_async(self, paper_id: str, pdf_path: Path) -> Dict[str, Any]:
        """Record PDF download (async).

        Args:
            paper_id: Paper identifier
            pdf_path: Path to PDF

        Returns:
            Cache entry
        """
        await self.db.init_schema_async()

        file_hash = self._compute_file_hash(pdf_path)
        file_size = pdf_path.stat().st_size

        async with aiosqlite.connect(self.db.db_path) as db:
            await db.execute("""
                INSERT INTO pdfs (paper_id, pdf_path, file_hash, file_size)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    pdf_path = excluded.pdf_path,
                    file_hash = excluded.file_hash,
                    file_size = excluded.file_size,
                    downloaded_at = datetime('now')
            """, (
                paper_id,
                str(pdf_path.absolute()),
                file_hash,
                file_size,
            ))
            await db.commit()

        logger.info(f"Cached PDF for {paper_id}: {pdf_path}")
        return {
            "paper_id": paper_id,
            "pdf_path": str(pdf_path.absolute()),
            "file_hash": file_hash,
            "file_size": file_size,
        }

    async def remove_async(self, paper_id: str, delete_file: bool = False) -> bool:
        """Remove PDF from cache (async).

        Args:
            paper_id: Paper identifier
            delete_file: If True, delete file too

        Returns:
            True if removed
        """
        await self.db.init_schema_async()

        if delete_file:
            entry = await self.get_async(paper_id)
            if entry:
                pdf_path = Path(entry["pdf_path"])
                if pdf_path.exists():
                    pdf_path.unlink()

        async with aiosqlite.connect(self.db.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM pdfs WHERE paper_id = ?",
                (paper_id,)
            )
            await db.commit()
            return cursor.rowcount > 0


# Export
__all__ = ["PDFCache", "DEFAULT_PDF_DIR"]
