"""Failed papers queue management for LitScribe.

This module handles persistence and retrieval of papers that failed
during analysis, supporting retry mechanisms.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from cache.database import get_cache_db

logger = logging.getLogger(__name__)


async def record_failed_paper(
    paper_id: str,
    title: str,
    error_message: str,
    paper_data: Dict[str, Any],
    research_question: Optional[str] = None,
    max_retries: int = 3,
) -> int:
    """Record a failed paper in the retry queue.

    Args:
        paper_id: Unique identifier for the paper
        title: Paper title
        error_message: Error that caused the failure
        paper_data: Original paper metadata (for retry)
        research_question: Research question context
        max_retries: Maximum retry attempts allowed

    Returns:
        ID of the inserted record
    """
    db = get_cache_db()
    await db.init_schema_async()

    import aiosqlite
    async with aiosqlite.connect(db.db_path) as conn:
        cursor = await conn.execute(
            """
            INSERT INTO failed_papers
            (paper_id, title, error_message, paper_data, research_question, max_retries)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                paper_id,
                title,
                error_message,
                json.dumps(paper_data, ensure_ascii=False),
                research_question,
                max_retries,
            )
        )
        await conn.commit()
        record_id = cursor.lastrowid
        logger.info(f"Recorded failed paper: {paper_id} ({title})")
        return record_id


async def get_pending_retries(
    research_question: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get papers that are pending retry.

    Args:
        research_question: Filter by research question (optional)
        limit: Maximum number of records to return

    Returns:
        List of failed paper records
    """
    db = get_cache_db()
    await db.init_schema_async()

    import aiosqlite
    async with aiosqlite.connect(db.db_path) as conn:
        conn.row_factory = aiosqlite.Row

        if research_question:
            cursor = await conn.execute(
                """
                SELECT * FROM failed_papers
                WHERE resolved_at IS NULL
                AND retry_count < max_retries
                AND research_question = ?
                ORDER BY failed_at ASC
                LIMIT ?
                """,
                (research_question, limit)
            )
        else:
            cursor = await conn.execute(
                """
                SELECT * FROM failed_papers
                WHERE resolved_at IS NULL
                AND retry_count < max_retries
                ORDER BY failed_at ASC
                LIMIT ?
                """,
                (limit,)
            )

        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "paper_id": row["paper_id"],
                "title": row["title"],
                "error_message": row["error_message"],
                "paper_data": json.loads(row["paper_data"]) if row["paper_data"] else {},
                "research_question": row["research_question"],
                "retry_count": row["retry_count"],
                "max_retries": row["max_retries"],
                "failed_at": row["failed_at"],
                "last_retry_at": row["last_retry_at"],
            }
            for row in rows
        ]


async def increment_retry_count(record_id: int) -> None:
    """Increment the retry count for a failed paper.

    Args:
        record_id: ID of the failed_papers record
    """
    db = get_cache_db()

    import aiosqlite
    async with aiosqlite.connect(db.db_path) as conn:
        await conn.execute(
            """
            UPDATE failed_papers
            SET retry_count = retry_count + 1,
                last_retry_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (record_id,)
        )
        await conn.commit()


async def mark_resolved(
    record_id: int,
    resolution: str = "success",
) -> None:
    """Mark a failed paper as resolved.

    Args:
        record_id: ID of the failed_papers record
        resolution: Resolution type ('success', 'skipped', 'permanent_failure')
    """
    db = get_cache_db()

    import aiosqlite
    async with aiosqlite.connect(db.db_path) as conn:
        await conn.execute(
            """
            UPDATE failed_papers
            SET resolved_at = CURRENT_TIMESTAMP,
                resolution = ?
            WHERE id = ?
            """,
            (resolution, record_id)
        )
        await conn.commit()
        logger.info(f"Marked failed paper {record_id} as {resolution}")


async def mark_resolved_by_paper_id(
    paper_id: str,
    resolution: str = "success",
) -> None:
    """Mark all pending failures for a paper as resolved.

    Args:
        paper_id: Paper ID to resolve
        resolution: Resolution type
    """
    db = get_cache_db()

    import aiosqlite
    async with aiosqlite.connect(db.db_path) as conn:
        await conn.execute(
            """
            UPDATE failed_papers
            SET resolved_at = CURRENT_TIMESTAMP,
                resolution = ?
            WHERE paper_id = ? AND resolved_at IS NULL
            """,
            (resolution, paper_id)
        )
        await conn.commit()


async def get_failure_stats() -> Dict[str, Any]:
    """Get statistics about failed papers.

    Returns:
        Dictionary with failure statistics
    """
    db = get_cache_db()
    await db.init_schema_async()

    import aiosqlite
    async with aiosqlite.connect(db.db_path) as conn:
        # Total failures
        cursor = await conn.execute("SELECT COUNT(*) FROM failed_papers")
        total = (await cursor.fetchone())[0]

        # Pending retries
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM failed_papers WHERE resolved_at IS NULL AND retry_count < max_retries"
        )
        pending = (await cursor.fetchone())[0]

        # Resolved successfully
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM failed_papers WHERE resolution = 'success'"
        )
        resolved_success = (await cursor.fetchone())[0]

        # Permanent failures
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM failed_papers WHERE resolution = 'permanent_failure' OR (resolved_at IS NULL AND retry_count >= max_retries)"
        )
        permanent = (await cursor.fetchone())[0]

        return {
            "total_failures": total,
            "pending_retries": pending,
            "resolved_success": resolved_success,
            "permanent_failures": permanent,
        }


async def clear_resolved(days_old: int = 30) -> int:
    """Clear resolved failures older than specified days.

    Args:
        days_old: Clear records older than this many days

    Returns:
        Number of records deleted
    """
    db = get_cache_db()

    import aiosqlite
    async with aiosqlite.connect(db.db_path) as conn:
        cursor = await conn.execute(
            f"""
            DELETE FROM failed_papers
            WHERE resolved_at IS NOT NULL
            AND resolved_at < datetime('now', '-{days_old} days')
            """
        )
        count = cursor.rowcount
        await conn.commit()
        logger.info(f"Cleared {count} old resolved failure records")
        return count


# Export
__all__ = [
    "record_failed_paper",
    "get_pending_retries",
    "increment_retry_count",
    "mark_resolved",
    "mark_resolved_by_paper_id",
    "get_failure_stats",
    "clear_resolved",
]
