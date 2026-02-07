"""Review session and version management for iterative refinement (Phase 9.3).

Provides session lifecycle management and diff-based version tracking
using Python's difflib for generating unified diffs between versions.
"""

import difflib
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from cache.database import get_cache_db

logger = logging.getLogger(__name__)


def create_session(
    research_question: str,
    review_type: str = "narrative",
    language: str = "en",
    thread_id: Optional[str] = None,
    state_snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a new review session.

    Args:
        research_question: The research question
        review_type: Type of review
        language: Output language
        thread_id: LangGraph checkpoint thread_id
        state_snapshot: Serialized state for session storage

    Returns:
        session_id (UUID string)
    """
    db = get_cache_db()
    db.init_schema()
    session_id = str(uuid.uuid4())
    snapshot_json = json.dumps(state_snapshot, default=str) if state_snapshot else None

    with db.get_connection() as conn:
        conn.execute(
            """INSERT INTO review_sessions
               (session_id, research_question, review_type, language, thread_id, state_snapshot)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, research_question, review_type, language, thread_id, snapshot_json),
        )
        conn.commit()

    logger.info(f"Created session {session_id} for: {research_question[:50]}")
    return session_id


def save_version(
    session_id: str,
    review_text: str,
    word_count: int,
    papers_cited: int,
    instruction: Optional[str] = None,
    previous_text: Optional[str] = None,
) -> int:
    """Save a new version of the review.

    Auto-increments version_number. Generates diff from previous_text if provided.

    Args:
        session_id: Session to save version for
        review_text: Full review text
        word_count: Word count
        papers_cited: Number of papers cited
        instruction: User instruction that triggered this version (None for v1)
        previous_text: Previous review text (for diff generation)

    Returns:
        version_number of the new version
    """
    db = get_cache_db()
    db.init_schema()

    with db.get_connection() as conn:
        # Get next version number
        cursor = conn.execute(
            "SELECT MAX(version_number) FROM review_versions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        version_number = (row[0] or 0) + 1

        # Generate diff
        diff_text = None
        if previous_text is not None and version_number > 1:
            diff_text = generate_diff(previous_text, review_text)

        conn.execute(
            """INSERT INTO review_versions
               (session_id, version_number, review_text, word_count, papers_cited, instruction, diff_text)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, version_number, review_text, word_count, papers_cited, instruction, diff_text),
        )

        # Update session timestamp
        conn.execute(
            "UPDATE review_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()

    logger.info(f"Saved version {version_number} for session {session_id}")
    return version_number


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session metadata.

    Args:
        session_id: Session ID (supports prefix match)

    Returns:
        Session dict or None
    """
    db = get_cache_db()
    db.init_schema()

    with db.get_connection() as conn:
        # Try exact match first
        cursor = conn.execute(
            "SELECT * FROM review_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()

        # Try prefix match if exact match fails
        if row is None:
            cursor = conn.execute(
                "SELECT * FROM review_sessions WHERE session_id LIKE ? LIMIT 1",
                (session_id + "%",),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)


def list_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    """List all sessions, most recent first.

    Args:
        limit: Maximum number of sessions to return

    Returns:
        List of session dicts
    """
    db = get_cache_db()
    db.init_schema()

    with db.get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM review_sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_version(session_id: str, version_number: int) -> Optional[Dict[str, Any]]:
    """Get a specific version.

    Args:
        session_id: Session ID
        version_number: Version number

    Returns:
        Version dict or None
    """
    db = get_cache_db()
    db.init_schema()

    # Resolve prefix
    session = get_session(session_id)
    if session is None:
        return None
    resolved_id = session["session_id"]

    with db.get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM review_versions WHERE session_id = ? AND version_number = ?",
            (resolved_id, version_number),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_latest_version(session_id: str) -> Optional[Dict[str, Any]]:
    """Get the latest version for a session.

    Args:
        session_id: Session ID

    Returns:
        Version dict or None
    """
    db = get_cache_db()
    db.init_schema()

    # Resolve prefix
    session = get_session(session_id)
    if session is None:
        return None
    resolved_id = session["session_id"]

    with db.get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM review_versions WHERE session_id = ? ORDER BY version_number DESC LIMIT 1",
            (resolved_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_all_versions(session_id: str) -> List[Dict[str, Any]]:
    """Get all versions for a session, ordered by version_number.

    Args:
        session_id: Session ID

    Returns:
        List of version dicts
    """
    db = get_cache_db()
    db.init_schema()

    # Resolve prefix
    session = get_session(session_id)
    if session is None:
        return []
    resolved_id = session["session_id"]

    with db.get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM review_versions WHERE session_id = ? ORDER BY version_number ASC",
            (resolved_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_diff(session_id: str, v1: int, v2: int) -> str:
    """Generate unified diff between two versions.

    Args:
        session_id: Session ID
        v1: First version number
        v2: Second version number

    Returns:
        Unified diff string, or empty string if versions not found
    """
    ver1 = get_version(session_id, v1)
    ver2 = get_version(session_id, v2)

    if ver1 is None or ver2 is None:
        return ""

    return generate_diff(ver1["review_text"], ver2["review_text"])


def rollback(session_id: str, target_version: int) -> int:
    """Rollback to a target version by creating a new version with the old text.

    Args:
        session_id: Session ID
        target_version: Version number to rollback to

    Returns:
        New version number

    Raises:
        ValueError: If target version not found
    """
    target = get_version(session_id, target_version)
    if target is None:
        raise ValueError(f"Version {target_version} not found for session {session_id}")

    latest = get_latest_version(session_id)
    previous_text = latest["review_text"] if latest else None

    return save_version(
        session_id=target["session_id"],
        review_text=target["review_text"],
        word_count=target.get("word_count", len(target["review_text"].split())),
        papers_cited=target.get("papers_cited", 0),
        instruction=f"Rollback to version {target_version}",
        previous_text=previous_text,
    )


def update_session_state(session_id: str, state_snapshot: Dict[str, Any]) -> None:
    """Update the stored state snapshot for a session.

    Args:
        session_id: Session ID
        state_snapshot: New state snapshot
    """
    db = get_cache_db()
    db.init_schema()

    # Resolve prefix
    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session not found: {session_id}")
    resolved_id = session["session_id"]

    snapshot_json = json.dumps(state_snapshot, default=str)

    with db.get_connection() as conn:
        conn.execute(
            "UPDATE review_sessions SET state_snapshot = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (snapshot_json, resolved_id),
        )
        conn.commit()


def generate_diff(old_text: str, new_text: str) -> str:
    """Generate unified diff between two texts.

    Args:
        old_text: Previous version text
        new_text: New version text

    Returns:
        Unified diff string
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="previous",
        tofile="current",
    )
    return "".join(diff)


__all__ = [
    "create_session",
    "save_version",
    "get_session",
    "list_sessions",
    "get_version",
    "get_latest_version",
    "get_all_versions",
    "get_diff",
    "rollback",
    "update_session_state",
    "generate_diff",
]
