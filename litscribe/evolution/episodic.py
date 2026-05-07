"""Episodic memory — FTS5-based cross-session recall."""
from __future__ import annotations

from litscribe.store.sqlite import SQLiteStore


class EpisodicMemory:
    """Records and recalls research session episodes via SQLite FTS5."""

    def __init__(self, store: SQLiteStore):
        self._store = store

    async def record(
        self,
        session_id: str,
        question: str,
        outcome_score: float,
        key_events: list[str],
    ) -> None:
        """Persist a session episode with a joined summary of key events."""
        summary = "; ".join(key_events)
        await self._store.save_episode(
            session_id=session_id,
            question=question,
            outcome_score=outcome_score,
            summary=summary,
        )

    async def recall(self, query: str, limit: int = 5) -> list[dict]:
        """Full-text search across episodes, ranked by relevance.

        Converts the query into an FTS5 OR expression so that any matching
        term scores a hit (prevents zero results when not all terms appear).
        """
        terms = query.strip().split()
        fts_query = " OR ".join(terms) if len(terms) > 1 else query
        return await self._store.recall(fts_query, limit)
