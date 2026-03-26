"""Semantic memory — ChromaDB knowledge store and user profile modeling."""
from __future__ import annotations

from litscribe.models.analysis import PaperAnalysis
from litscribe.store.vectors import VectorStore

SEMANTIC_COLLECTION = "semantic_memory"
USER_COLLECTION = "user_profiles"


class SemanticMemory:
    """Stores distilled research knowledge and user preferences in ChromaDB."""

    def __init__(self, vectors: VectorStore):
        self._vectors = vectors

    def absorb(self, analyses: list[PaperAnalysis]) -> None:
        """Index key findings from PaperAnalysis objects into the semantic store."""
        texts, metadatas, ids = [], [], []
        for a in analyses:
            for i, finding in enumerate(a.key_findings):
                texts.append(finding)
                metadatas.append({"paper_id": a.paper_id, "type": "finding"})
                ids.append(f"{a.paper_id}:finding:{i}")
        if texts:
            self._vectors.add_texts(SEMANTIC_COLLECTION, texts, metadatas, ids)

    def search(self, query: str, n: int = 10) -> list[dict]:
        """Semantic search over absorbed knowledge."""
        return self._vectors.search(query, SEMANTIC_COLLECTION, n)

    def update_user_profile(self, user_id: str, domain: str, preferences: dict) -> None:
        """Upsert a user profile document."""
        text = f"User domain: {domain}. Preferences: {preferences}"
        self._vectors.add_texts(
            USER_COLLECTION,
            texts=[text],
            metadatas=[{"user_id": user_id, "domain": domain, **preferences}],
            ids=[user_id],
        )

    def get_user_profile(self, user_id: str) -> dict | None:
        """Retrieve a user profile by ID, or None if not found."""
        results = self._vectors.search(user_id, USER_COLLECTION, n=1)
        if results and results[0]["id"] == user_id:
            return results[0]
        return None
