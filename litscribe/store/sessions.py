from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from litscribe.models.review import ReviewOutput
from litscribe.tools.status import PipelineState


class SessionStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    async def _db(self):
        db = await aiosqlite.connect(self.db_path)
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                research_question TEXT NOT NULL,
                language TEXT DEFAULT 'en',
                domain TEXT DEFAULT '',
                papers_count INTEGER DEFAULT 0,
                analyses_count INTEGER DEFAULT 0,
                score REAL DEFAULT 0.0,
                review_text TEXT DEFAULT '',
                word_count INTEGER DEFAULT 0,
                themes TEXT DEFAULT '[]',
                state_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS review_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                review_text TEXT NOT NULL,
                word_count INTEGER DEFAULT 0,
                instruction TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await db.commit()
        return db

    async def save_session(self, state: PipelineState) -> str:
        session_id = str(uuid.uuid4())[:8]
        db = await self._db()

        themes = []
        review_text = ""
        word_count = 0
        if state.synthesis:
            review_text = state.synthesis.text
            word_count = state.synthesis.word_count
            themes = [t.name for t in state.synthesis.themes]

        await db.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id, research_question, language, domain,
                papers_count, analyses_count, score, review_text,
                word_count, themes, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (session_id, state.research_question, state.language,
             state.domain, len(state.papers), len(state.analyses),
             state.assessment.score if state.assessment else 0.0,
             review_text, word_count, json.dumps(themes, ensure_ascii=False)),
        )

        if review_text:
            await db.execute(
                """INSERT INTO review_versions
                   (session_id, version, review_text, word_count)
                   VALUES (?, 1, ?, ?)""",
                (session_id, review_text, word_count),
            )

        await db.commit()
        await db.close()
        return session_id

    async def save_refinement(self, session_id: str, review: ReviewOutput, instruction: str) -> int:
        db = await self._db()

        row = await db.execute_fetchall(
            "SELECT MAX(version) FROM review_versions WHERE session_id = ?",
            (session_id,),
        )
        version = (row[0][0] or 0) + 1

        await db.execute(
            """INSERT INTO review_versions
               (session_id, version, review_text, word_count, instruction)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, version, review.text, review.word_count, instruction),
        )

        await db.execute(
            """UPDATE sessions SET review_text = ?, word_count = ?, updated_at = datetime('now')
               WHERE session_id = ?""",
            (review.text, review.word_count, session_id),
        )

        await db.commit()
        await db.close()
        return version

    async def list_sessions(self) -> list[dict]:
        db = await self._db()
        rows = await db.execute_fetchall(
            """SELECT session_id, research_question, language, domain,
                      papers_count, word_count, score, created_at
               FROM sessions ORDER BY created_at DESC LIMIT 20""",
        )
        await db.close()

        return [
            {
                "session_id": r[0],
                "question": r[1],
                "language": r[2],
                "domain": r[3],
                "papers": r[4],
                "words": r[5],
                "score": r[6],
                "created_at": r[7],
            }
            for r in rows
        ]

    async def get_session(self, session_id: str) -> dict | None:
        db = await self._db()
        rows = await db.execute_fetchall(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        await db.close()

        if not rows:
            return None

        r = rows[0]
        cols = ["session_id", "research_question", "language", "domain",
                "papers_count", "analyses_count", "score", "review_text",
                "word_count", "themes", "state_json", "created_at", "updated_at"]
        return dict(zip(cols, r))

    async def get_versions(self, session_id: str) -> list[dict]:
        db = await self._db()
        rows = await db.execute_fetchall(
            """SELECT version, word_count, instruction, created_at
               FROM review_versions WHERE session_id = ?
               ORDER BY version""",
            (session_id,),
        )
        await db.close()
        return [
            {"version": r[0], "words": r[1], "instruction": r[2], "created_at": r[3]}
            for r in rows
        ]
