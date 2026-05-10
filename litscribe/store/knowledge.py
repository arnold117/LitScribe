from __future__ import annotations

import json
import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


class KnowledgeStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    async def _db(self):
        db = await aiosqlite.connect(self.db_path)
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                topic TEXT NOT NULL,
                finding TEXT NOT NULL,
                paper_id TEXT DEFAULT '',
                cite_key TEXT DEFAULT '',
                confidence REAL DEFAULT 0.5,
                signature TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge(domain)
        """)
        await db.commit()
        return db

    async def save_findings(
        self,
        domain: str,
        topic: str,
        analyses: list,
        key_map: dict[str, str] | None = None,
    ) -> int:
        db = await self._db()
        count = 0
        for a in analyses:
            pid = a.paper_id if hasattr(a, "paper_id") else a.get("paper_id", "")
            findings = a.key_findings if hasattr(a, "key_findings") else a.get("key_findings", [])
            relevance = a.relevance_score if hasattr(a, "relevance_score") else a.get("relevance_score", 0.5)
            cite_key = key_map.get(pid, "") if key_map else ""

            for finding in findings[:3]:
                from litscribe.tools.integrity import sign_finding
                sig = sign_finding(domain, topic, finding)
                await db.execute(
                    """INSERT INTO knowledge (domain, topic, finding, paper_id, cite_key, confidence, signature)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (domain, topic, finding, pid, cite_key, relevance, sig),
                )
                count += 1

        await db.commit()
        await db.close()
        logger.info(f"Knowledge: saved {count} findings for '{topic[:30]}' in {domain}")
        return count

    async def query(self, domain: str = "", topic: str = "", limit: int = 20) -> list[dict]:
        db = await self._db()

        query = "SELECT domain, topic, finding, cite_key, confidence, created_at FROM knowledge"
        conditions = []
        params: list = []
        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if topic:
            safe_topic = topic.replace("%", "").replace("_", "")[:100]
            conditions.append("(topic LIKE ? OR finding LIKE ?)")
            params.extend([f"%{safe_topic}%", f"%{safe_topic}%"])

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
        params.append(limit)

        rows = await db.execute_fetchall(
            query.replace("SELECT domain, topic, finding, cite_key, confidence, created_at",
                          "SELECT domain, topic, finding, cite_key, confidence, created_at, signature"),
            params,
        )
        await db.close()

        from litscribe.tools.integrity import verify_finding
        verified = []
        tampered = 0
        for r in rows:
            sig = r[6] if len(r) > 6 else ""
            if sig and not verify_finding(r[0], r[1], r[2], sig):
                tampered += 1
                logger.warning(f"Tampered knowledge entry detected: {r[2][:50]}")
                continue
            verified.append(
                {"domain": r[0], "topic": r[1], "finding": r[2],
                 "cite_key": r[3], "confidence": r[4], "created_at": r[5]}
            )

        if tampered:
            logger.warning(f"Knowledge integrity: {tampered} tampered entries rejected")
        return verified

    async def get_context_for_review(self, research_question: str, domain: str) -> str:
        findings = await self.query(domain=domain, topic=research_question[:50], limit=10)
        if not findings:
            return ""

        lines = ["## Prior Knowledge (from previous reviews)\n"]
        for f in findings:
            cite = f" [@{f['cite_key']}]" if f["cite_key"] else ""
            lines.append(f"- {f['finding']}{cite}")

        return "\n".join(lines)
