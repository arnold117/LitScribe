"""SQLite storage layer with FTS5 for LitScribe episodic memory."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from litscribe.models.analysis import ParsedDoc
from litscribe.models.paper import Paper

SCHEMA_VERSION = 6

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id    TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    authors     TEXT NOT NULL,   -- JSON array
    abstract    TEXT NOT NULL,
    year        INTEGER,
    sources     TEXT NOT NULL,   -- JSON object
    venue       TEXT DEFAULT '',
    citations   INTEGER DEFAULT 0,
    doi         TEXT DEFAULT '',
    pdf_urls    TEXT DEFAULT '[]',  -- JSON array
    relevance_score    REAL DEFAULT 0.0,
    completeness_score REAL DEFAULT 0.0,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pdfs (
    paper_id    TEXT PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    size_bytes  INTEGER DEFAULT 0,
    downloaded_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS parsed_docs (
    paper_id    TEXT PRIMARY KEY,
    markdown    TEXT NOT NULL DEFAULT '',
    sections    TEXT NOT NULL DEFAULT '[]',  -- JSON array
    word_count  INTEGER DEFAULT 0,
    parsed_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,
    research_question TEXT NOT NULL,
    review_type       TEXT NOT NULL DEFAULT 'standard',
    language          TEXT NOT NULL DEFAULT 'en',
    created_at        TEXT DEFAULT (datetime('now')),
    updated_at        TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    question    TEXT NOT NULL,
    outcome_score REAL DEFAULT 0.0,
    summary     TEXT NOT NULL DEFAULT '',
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    session_id UNINDEXED,
    question,
    summary,
    content='episodes',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS episodes_ai
AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, session_id, question, summary)
    VALUES (new.id, new.session_id, new.question, new.summary);
END;

CREATE TABLE IF NOT EXISTS review_versions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL,
    version_number INTEGER NOT NULL DEFAULT 1,
    review_text    TEXT NOT NULL,
    word_count     INTEGER DEFAULT 0,
    instruction    TEXT DEFAULT '',
    diff_text      TEXT DEFAULT '',
    created_at     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS skills_meta (
    skill_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT DEFAULT '',
    success_rate REAL DEFAULT 0.0,
    use_count   INTEGER DEFAULT 0,
    last_used   TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);
"""


class SQLiteStore:
    """Async SQLite store with FTS5 full-text search for episodic memory."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create tables and set schema version."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_DDL)
        # Upsert schema version
        await self._db.execute(
            "INSERT OR REPLACE INTO schema_version(version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    async def list_tables(self) -> list[str]:
        """Return a list of real (non-virtual-shadow) table names."""
        async with self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ) as cursor:
            rows = await cursor.fetchall()
        return [row["name"] for row in rows]

    # ------------------------------------------------------------------
    # Papers
    # ------------------------------------------------------------------

    async def save_papers(self, papers: list[Paper]) -> None:
        """Upsert a list of Paper objects."""
        await self._db.executemany(
            """
            INSERT INTO papers
                (paper_id, title, authors, abstract, year, sources, venue,
                 citations, doi, pdf_urls, relevance_score, completeness_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                title              = excluded.title,
                authors            = excluded.authors,
                abstract           = excluded.abstract,
                year               = excluded.year,
                sources            = excluded.sources,
                venue              = excluded.venue,
                citations          = excluded.citations,
                doi                = excluded.doi,
                pdf_urls           = excluded.pdf_urls,
                relevance_score    = excluded.relevance_score,
                completeness_score = excluded.completeness_score
            """,
            [
                (
                    p.paper_id,
                    p.title,
                    json.dumps(p.authors),
                    p.abstract,
                    p.year,
                    json.dumps(p.sources),
                    p.venue,
                    p.citations,
                    p.doi,
                    json.dumps(p.pdf_urls),
                    p.relevance_score,
                    p.completeness_score,
                )
                for p in papers
            ],
        )
        await self._db.commit()

    async def get_paper(self, paper_id: str) -> Paper | None:
        """Retrieve a Paper by its ID, or None if not found."""
        async with self._db.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return Paper(
            paper_id=row["paper_id"],
            title=row["title"],
            authors=json.loads(row["authors"]),
            abstract=row["abstract"],
            year=row["year"],
            sources=json.loads(row["sources"]),
            venue=row["venue"] or "",
            citations=row["citations"] or 0,
            doi=row["doi"] or "",
            pdf_urls=json.loads(row["pdf_urls"]),
            relevance_score=row["relevance_score"] or 0.0,
            completeness_score=row["completeness_score"] or 0.0,
        )

    # ------------------------------------------------------------------
    # Parsed documents
    # ------------------------------------------------------------------

    async def save_parsed(self, paper_id: str, doc: ParsedDoc) -> None:
        """Upsert a ParsedDoc."""
        await self._db.execute(
            """
            INSERT INTO parsed_docs (paper_id, markdown, sections, word_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                markdown   = excluded.markdown,
                sections   = excluded.sections,
                word_count = excluded.word_count,
                parsed_at  = datetime('now')
            """,
            (paper_id, doc.markdown, json.dumps(doc.sections), doc.word_count),
        )
        await self._db.commit()

    async def get_parsed(self, paper_id: str) -> ParsedDoc | None:
        """Retrieve a ParsedDoc by paper_id, or None if not found."""
        async with self._db.execute(
            "SELECT * FROM parsed_docs WHERE paper_id = ?", (paper_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return ParsedDoc(
            paper_id=row["paper_id"],
            markdown=row["markdown"],
            sections=json.loads(row["sections"]),
            word_count=row["word_count"],
        )

    # ------------------------------------------------------------------
    # Episodes (episodic memory) + FTS5 recall
    # ------------------------------------------------------------------

    async def save_episode(
        self,
        *,
        session_id: str,
        question: str,
        outcome_score: float = 0.0,
        summary: str = "",
    ) -> int:
        """Insert an episode and automatically index it in FTS5 via trigger."""
        async with self._db.execute(
            """
            INSERT INTO episodes (session_id, question, outcome_score, summary)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, question, outcome_score, summary),
        ) as cursor:
            rowid = cursor.lastrowid
        await self._db.commit()
        return rowid

    async def recall(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Full-text search episodes using FTS5 MATCH, ranked by BM25."""
        async with self._db.execute(
            """
            SELECT e.id, e.session_id, e.question, e.outcome_score, e.summary, e.created_at
            FROM episodes_fts
            JOIN episodes e ON episodes_fts.rowid = e.id
            WHERE episodes_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def create_session(
        self, session_id: str, question: str, review_type: str = "standard", language: str = "en"
    ):
        await self._db.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id, research_question, review_type, language)
               VALUES (?, ?, ?, ?)""",
            (session_id, question, review_type, language),
        )
        await self._db.commit()

    async def get_session(self, session_id: str) -> dict | None:
        cursor = await self._db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "session_id": row["session_id"],
            "research_question": row["research_question"],
            "review_type": row["review_type"],
            "language": row["language"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def list_sessions(self, limit: int = 20) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT session_id, research_question, review_type, created_at FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {"session_id": row["session_id"], "question": row["research_question"],
             "review_type": row["review_type"], "created_at": row["created_at"]}
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Review Versions
    # ------------------------------------------------------------------

    async def save_version(
        self, session_id: str, version_number: int, review_text: str,
        word_count: int = 0, instruction: str = "", diff_text: str = "",
    ):
        await self._db.execute(
            """INSERT OR REPLACE INTO review_versions
               (session_id, version_number, review_text, word_count, instruction, diff_text)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, version_number, review_text, word_count, instruction, diff_text),
        )
        await self._db.commit()

    async def get_versions(self, session_id: str) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM review_versions WHERE session_id = ? ORDER BY version_number",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "session_id": row["session_id"],
                "version_number": row["version_number"],
                "review_text": row["review_text"],
                "word_count": row["word_count"],
                "instruction": row["instruction"],
                "diff_text": row["diff_text"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    async def get_latest_version(self, session_id: str) -> dict | None:
        cursor = await self._db.execute(
            "SELECT * FROM review_versions WHERE session_id = ? ORDER BY version_number DESC LIMIT 1",
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "session_id": row["session_id"],
            "version_number": row["version_number"],
            "review_text": row["review_text"],
            "word_count": row["word_count"],
            "instruction": row["instruction"],
            "diff_text": row["diff_text"],
            "created_at": row["created_at"],
        }
