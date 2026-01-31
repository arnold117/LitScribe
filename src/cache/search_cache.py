"""Search results cache with TTL for LitScribe.

Caches search results to avoid repeated API calls within TTL period.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

from cache.database import CacheDatabase, get_cache_db

logger = logging.getLogger(__name__)

# Default TTL: 24 hours
DEFAULT_TTL_HOURS = 24


class SearchCache:
    """Cache for search results with TTL support."""

    def __init__(self, db: Optional[CacheDatabase] = None, ttl_hours: int = DEFAULT_TTL_HOURS):
        """Initialize search cache.

        Args:
            db: CacheDatabase instance
            ttl_hours: Time-to-live in hours for cached results
        """
        self.db = db or get_cache_db()
        self.ttl_hours = ttl_hours

    def _hash_query(self, query: str, source: str) -> str:
        """Create hash for query+source combination.

        Args:
            query: Search query
            source: Source name

        Returns:
            SHA256 hash of query+source
        """
        key = f"{source}:{query.lower().strip()}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def _parse_results(self, row: aiosqlite.Row) -> Tuple[List[Dict[str, Any]], int]:
        """Parse cached results from row.

        Args:
            row: Database row

        Returns:
            Tuple of (results list, total found)
        """
        results_json = row["results"]
        results = json.loads(results_json) if results_json else []
        total_found = row["total_found"] or len(results)
        return results, total_found

    # Synchronous methods

    def get(
        self,
        query: str,
        source: str,
        ignore_expired: bool = False
    ) -> Optional[Tuple[List[Dict[str, Any]], int]]:
        """Get cached search results (sync).

        Args:
            query: Search query
            source: Source name
            ignore_expired: If True, return results even if expired

        Returns:
            Tuple of (results, total_found) or None if not cached/expired
        """
        self.db.init_schema()
        query_hash = self._hash_query(query, source)

        with self.db.get_connection() as conn:
            if ignore_expired:
                cursor = conn.execute(
                    "SELECT * FROM search_cache WHERE query_hash = ?",
                    (query_hash,)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM search_cache WHERE query_hash = ? AND expires_at > datetime('now')",
                    (query_hash,)
                )

            row = cursor.fetchone()
            if row:
                logger.debug(f"Cache hit for query: {query[:50]}... (source: {source})")
                return self._parse_results(row)

        logger.debug(f"Cache miss for query: {query[:50]}... (source: {source})")
        return None

    def save(
        self,
        query: str,
        source: str,
        results: List[Dict[str, Any]],
        total_found: Optional[int] = None,
        ttl_hours: Optional[int] = None
    ) -> str:
        """Save search results to cache (sync).

        Args:
            query: Search query
            source: Source name
            results: List of search results
            total_found: Total number found (may be more than results)
            ttl_hours: Custom TTL, uses default if not specified

        Returns:
            Query hash
        """
        self.db.init_schema()
        query_hash = self._hash_query(query, source)
        ttl = ttl_hours if ttl_hours is not None else self.ttl_hours
        expires_at = datetime.now() + timedelta(hours=ttl)

        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO search_cache (
                    query_hash, query, source, results, total_found, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(query_hash) DO UPDATE SET
                    results = excluded.results,
                    total_found = excluded.total_found,
                    searched_at = datetime('now'),
                    expires_at = excluded.expires_at
            """, (
                query_hash,
                query,
                source,
                json.dumps(results),
                total_found or len(results),
                expires_at.isoformat(),
            ))
            conn.commit()

        logger.debug(f"Cached {len(results)} results for query: {query[:50]}... (expires: {expires_at})")
        return query_hash

    def invalidate(self, query: str, source: str) -> bool:
        """Invalidate cached results for a query (sync).

        Args:
            query: Search query
            source: Source name

        Returns:
            True if entry was invalidated
        """
        self.db.init_schema()
        query_hash = self._hash_query(query, source)

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM search_cache WHERE query_hash = ?",
                (query_hash,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear_expired(self) -> int:
        """Clear all expired cache entries (sync).

        Returns:
            Number of entries cleared
        """
        return self.db.clear_expired_cache()

    def clear_source(self, source: str) -> int:
        """Clear all cache entries for a source (sync).

        Args:
            source: Source name

        Returns:
            Number of entries cleared
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM search_cache WHERE source = ?",
                (source,)
            )
            count = cursor.rowcount
            conn.commit()
            return count

    def get_all_cached_queries(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all cached queries (for debugging/stats).

        Args:
            source: Filter by source if provided

        Returns:
            List of cache entries (without full results)
        """
        self.db.init_schema()

        with self.db.get_connection() as conn:
            if source:
                cursor = conn.execute("""
                    SELECT query_hash, query, source, total_found, searched_at, expires_at
                    FROM search_cache WHERE source = ?
                    ORDER BY searched_at DESC
                """, (source,))
            else:
                cursor = conn.execute("""
                    SELECT query_hash, query, source, total_found, searched_at, expires_at
                    FROM search_cache
                    ORDER BY searched_at DESC
                """)
            return [dict(row) for row in cursor.fetchall()]

    # Asynchronous methods

    async def get_async(
        self,
        query: str,
        source: str,
        ignore_expired: bool = False
    ) -> Optional[Tuple[List[Dict[str, Any]], int]]:
        """Get cached search results (async).

        Args:
            query: Search query
            source: Source name
            ignore_expired: If True, return results even if expired

        Returns:
            Tuple of (results, total_found) or None
        """
        await self.db.init_schema_async()
        query_hash = self._hash_query(query, source)

        async with aiosqlite.connect(self.db.db_path) as db:
            db.row_factory = aiosqlite.Row
            if ignore_expired:
                cursor = await db.execute(
                    "SELECT * FROM search_cache WHERE query_hash = ?",
                    (query_hash,)
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM search_cache WHERE query_hash = ? AND expires_at > datetime('now')",
                    (query_hash,)
                )

            row = await cursor.fetchone()
            if row:
                logger.debug(f"Cache hit for query: {query[:50]}... (source: {source})")
                return self._parse_results(row)

        logger.debug(f"Cache miss for query: {query[:50]}... (source: {source})")
        return None

    async def save_async(
        self,
        query: str,
        source: str,
        results: List[Dict[str, Any]],
        total_found: Optional[int] = None,
        ttl_hours: Optional[int] = None
    ) -> str:
        """Save search results to cache (async).

        Args:
            query: Search query
            source: Source name
            results: List of search results
            total_found: Total found count
            ttl_hours: Custom TTL

        Returns:
            Query hash
        """
        await self.db.init_schema_async()
        query_hash = self._hash_query(query, source)
        ttl = ttl_hours if ttl_hours is not None else self.ttl_hours
        expires_at = datetime.now() + timedelta(hours=ttl)

        async with aiosqlite.connect(self.db.db_path) as db:
            await db.execute("""
                INSERT INTO search_cache (
                    query_hash, query, source, results, total_found, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(query_hash) DO UPDATE SET
                    results = excluded.results,
                    total_found = excluded.total_found,
                    searched_at = datetime('now'),
                    expires_at = excluded.expires_at
            """, (
                query_hash,
                query,
                source,
                json.dumps(results),
                total_found or len(results),
                expires_at.isoformat(),
            ))
            await db.commit()

        logger.debug(f"Cached {len(results)} results for query: {query[:50]}...")
        return query_hash

    async def invalidate_async(self, query: str, source: str) -> bool:
        """Invalidate cached results (async).

        Args:
            query: Search query
            source: Source name

        Returns:
            True if entry was invalidated
        """
        await self.db.init_schema_async()
        query_hash = self._hash_query(query, source)

        async with aiosqlite.connect(self.db.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM search_cache WHERE query_hash = ?",
                (query_hash,)
            )
            await db.commit()
            return cursor.rowcount > 0


# Export
__all__ = ["SearchCache", "DEFAULT_TTL_HOURS"]
