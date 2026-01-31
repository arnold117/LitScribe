"""SQLite database management for LitScribe cache.

This module handles database connection, initialization, and schema management.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import aiosqlite

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path("cache")
DEFAULT_DB_NAME = "litscribe.db"

# Database schema version for migrations
SCHEMA_VERSION = 1

# Core SQL schema definition (required tables)
CORE_SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Paper metadata cache
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT,
    abstract TEXT,
    year INTEGER,
    venue TEXT,
    citations INTEGER,
    source TEXT,
    doi TEXT,
    pdf_url TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- PDF download records
CREATE TABLE IF NOT EXISTS pdfs (
    paper_id TEXT PRIMARY KEY,
    pdf_path TEXT NOT NULL,
    file_hash TEXT,
    file_size INTEGER,
    downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

-- Parsed document cache
CREATE TABLE IF NOT EXISTS parsed_docs (
    paper_id TEXT PRIMARY KEY,
    markdown TEXT,
    sections TEXT,
    tables TEXT,
    equations TEXT,
    paper_references TEXT,
    word_count INTEGER,
    page_count INTEGER,
    parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

-- Search results cache with TTL
CREATE TABLE IF NOT EXISTS search_cache (
    query_hash TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    source TEXT NOT NULL,
    results TEXT,
    total_found INTEGER,
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- LLM response cache
CREATE TABLE IF NOT EXISTS llm_cache (
    prompt_hash TEXT PRIMARY KEY,
    prompt_preview TEXT,
    response TEXT NOT NULL,
    model TEXT,
    tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Command execution history
CREATE TABLE IF NOT EXISTS command_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command TEXT NOT NULL,
    args TEXT,
    research_question TEXT,
    result_path TEXT,
    papers_found INTEGER,
    papers_analyzed INTEGER,
    status TEXT DEFAULT 'running',
    error_message TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_search_cache_expires ON search_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_command_logs_status ON command_logs(status);
CREATE INDEX IF NOT EXISTS idx_command_logs_started ON command_logs(started_at);
"""

# FTS schema (optional, may fail on some SQLite versions)
FTS_SCHEMA_SQL = """
-- Full-text search index for papers
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    paper_id,
    title,
    abstract,
    content='papers',
    content_rowid='rowid'
);
"""

# FTS triggers (separate for easier debugging)
FTS_TRIGGERS_SQL = [
    """CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
        INSERT INTO papers_fts(rowid, paper_id, title, abstract)
        VALUES (NEW.rowid, NEW.paper_id, NEW.title, NEW.abstract);
    END""",
    """CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
        INSERT INTO papers_fts(papers_fts, rowid, paper_id, title, abstract)
        VALUES('delete', OLD.rowid, OLD.paper_id, OLD.title, OLD.abstract);
    END""",
    """CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
        INSERT INTO papers_fts(papers_fts, rowid, paper_id, title, abstract)
        VALUES('delete', OLD.rowid, OLD.paper_id, OLD.title, OLD.abstract);
        INSERT INTO papers_fts(rowid, paper_id, title, abstract)
        VALUES (NEW.rowid, NEW.paper_id, NEW.title, NEW.abstract);
    END""",
]

# Combined schema for backward compatibility
SCHEMA_SQL = CORE_SCHEMA_SQL


class CacheDatabase:
    """SQLite cache database manager.

    Provides both sync and async interfaces for database operations.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize cache database.

        Args:
            db_path: Path to SQLite database file. Defaults to cache/litscribe.db
        """
        if db_path is None:
            cache_dir = Path(os.getenv("LITSCRIBE_CACHE_DIR", DEFAULT_CACHE_DIR))
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / DEFAULT_DB_NAME

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def init_schema(self) -> None:
        """Initialize database schema (synchronous)."""
        if self._initialized:
            return

        with self.get_connection() as conn:
            # Check current schema version
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if cursor.fetchone() is None:
                # Fresh database, apply core schema
                conn.executescript(CORE_SCHEMA_SQL)

                # Try to apply FTS (optional, may fail on some SQLite versions)
                try:
                    conn.executescript(FTS_SCHEMA_SQL)
                    for trigger_sql in FTS_TRIGGERS_SQL:
                        conn.execute(trigger_sql)
                    logger.debug("FTS5 support enabled")
                except Exception as e:
                    logger.warning(f"FTS5 not available: {e}")

                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,)
                )
                logger.info(f"Initialized cache database at {self.db_path}")
            else:
                # Check for migrations
                cursor = conn.execute("SELECT MAX(version) FROM schema_version")
                current_version = cursor.fetchone()[0] or 0
                if current_version < SCHEMA_VERSION:
                    self._apply_migrations(conn, current_version)

            conn.commit()

        self._initialized = True

    async def init_schema_async(self) -> None:
        """Initialize database schema (asynchronous)."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if await cursor.fetchone() is None:
                # Apply core schema
                await db.executescript(CORE_SCHEMA_SQL)

                # Try to apply FTS (optional)
                try:
                    await db.executescript(FTS_SCHEMA_SQL)
                    for trigger_sql in FTS_TRIGGERS_SQL:
                        await db.execute(trigger_sql)
                    logger.debug("FTS5 support enabled")
                except Exception as e:
                    logger.warning(f"FTS5 not available: {e}")

                await db.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,)
                )
                logger.info(f"Initialized cache database at {self.db_path}")
            else:
                cursor = await db.execute("SELECT MAX(version) FROM schema_version")
                row = await cursor.fetchone()
                current_version = row[0] if row else 0
                if current_version < SCHEMA_VERSION:
                    await self._apply_migrations_async(db, current_version)

            await db.commit()

        self._initialized = True

    def _apply_migrations(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Apply database migrations (sync)."""
        # Add migration logic here as schema evolves
        logger.info(f"Migrating database from version {from_version} to {SCHEMA_VERSION}")
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,)
        )

    async def _apply_migrations_async(self, db: aiosqlite.Connection, from_version: int) -> None:
        """Apply database migrations (async)."""
        logger.info(f"Migrating database from version {from_version} to {SCHEMA_VERSION}")
        await db.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,)
        )

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a synchronous database connection.

        Yields:
            SQLite connection with row factory
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
        try:
            yield conn
        finally:
            conn.close()

    async def get_async_connection(self) -> aiosqlite.Connection:
        """Get an asynchronous database connection.

        Returns:
            Async SQLite connection
        """
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("PRAGMA journal_mode = WAL")
        return db

    # Utility methods

    def execute(self, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Execute SQL query synchronously.

        Args:
            sql: SQL query string
            params: Query parameters

        Returns:
            List of rows
        """
        self.init_schema()
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            results = cursor.fetchall()
            conn.commit()
            return results

    async def execute_async(self, sql: str, params: Tuple = ()) -> List[aiosqlite.Row]:
        """Execute SQL query asynchronously.

        Args:
            sql: SQL query string
            params: Query parameters

        Returns:
            List of rows
        """
        await self.init_schema_async()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(sql, params)
            results = await cursor.fetchall()
            await db.commit()
            return results

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        self.init_schema()
        stats = {}
        with self.get_connection() as conn:
            # Count records in each table
            for table in ["papers", "pdfs", "parsed_docs", "search_cache", "llm_cache", "command_logs"]:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            # Database file size
            stats["db_size_mb"] = round(self.db_path.stat().st_size / (1024 * 1024), 2)

            # Recent activity
            cursor = conn.execute(
                "SELECT COUNT(*) FROM command_logs WHERE started_at > datetime('now', '-7 days')"
            )
            stats["commands_last_7_days"] = cursor.fetchone()[0]

        return stats

    def clear_expired_cache(self) -> int:
        """Clear expired search cache entries.

        Returns:
            Number of entries cleared
        """
        self.init_schema()
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM search_cache WHERE expires_at < datetime('now')"
            )
            count = cursor.rowcount
            conn.commit()
            logger.info(f"Cleared {count} expired cache entries")
            return count

    def vacuum(self) -> None:
        """Optimize database by running VACUUM."""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            logger.info("Database vacuumed successfully")


# Global cache database instance
_cache_db: Optional[CacheDatabase] = None


def get_cache_db(db_path: Optional[Path] = None) -> CacheDatabase:
    """Get or create the global cache database instance.

    Args:
        db_path: Optional path to database file

    Returns:
        CacheDatabase instance
    """
    global _cache_db
    if _cache_db is None or (db_path is not None and _cache_db.db_path != db_path):
        _cache_db = CacheDatabase(db_path)
    return _cache_db


def init_cache(db_path: Optional[Path] = None) -> CacheDatabase:
    """Initialize the cache system.

    Args:
        db_path: Optional path to database file

    Returns:
        Initialized CacheDatabase instance
    """
    db = get_cache_db(db_path)
    db.init_schema()
    return db


# Export
__all__ = [
    "CacheDatabase",
    "get_cache_db",
    "init_cache",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DB_NAME",
]
