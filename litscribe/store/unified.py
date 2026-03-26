from __future__ import annotations
from pathlib import Path
from litscribe.models.paper import Paper
from litscribe.models.analysis import ParsedDoc
from litscribe.store.sqlite import SQLiteStore
from litscribe.store.vectors import VectorStore


class UnifiedStore:
    def __init__(self, db_path: Path, chroma_path: Path):
        self.sqlite = SQLiteStore(db_path)
        self.vectors = VectorStore(chroma_path)

    async def initialize(self):
        await self.sqlite.initialize()

    async def close(self):
        await self.sqlite.close()

    async def save_papers(self, papers: list[Paper]):
        await self.sqlite.save_papers(papers)

    async def get_paper(self, paper_id: str) -> Paper | None:
        return await self.sqlite.get_paper(paper_id)

    async def save_parsed(self, paper_id: str, doc: ParsedDoc):
        await self.sqlite.save_parsed(paper_id, doc)

    async def get_parsed(self, paper_id: str) -> ParsedDoc | None:
        return await self.sqlite.get_parsed(paper_id)

    async def save_episode(self, session_id: str, question: str, outcome_score: float, summary: str):
        await self.sqlite.save_episode(session_id=session_id, question=question, outcome_score=outcome_score, summary=summary)

    async def recall(self, query: str, limit: int = 5) -> list[dict]:
        return await self.sqlite.recall(query, limit)

    def add_embeddings(self, texts: list[str], metadatas: list[dict], ids: list[str], collection: str):
        self.vectors.add_texts(collection, texts, metadatas, ids)

    def semantic_search(self, query: str, collection: str, n: int = 10) -> list[dict]:
        return self.vectors.search(query, collection, n)
