from __future__ import annotations

from pathlib import Path

import chromadb
import chromadb.errors


class VectorStore:
    def __init__(self, path: Path):
        self._client = chromadb.PersistentClient(path=str(path))

    def _get_or_create(self, name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(name=name)

    def add_texts(self, collection: str, texts: list[str], metadatas: list[dict], ids: list[str]):
        coll = self._get_or_create(collection)
        coll.add(documents=texts, metadatas=metadatas, ids=ids)

    def search(self, query: str, collection: str, n: int = 10) -> list[dict]:
        try:
            coll = self._client.get_collection(collection)
        except (ValueError, chromadb.errors.NotFoundError):
            return []
        count = coll.count()
        if count == 0:
            return []
        results = coll.query(query_texts=[query], n_results=min(n, count))
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return output

    def delete(self, collection: str, ids: list[str]):
        try:
            coll = self._client.get_collection(collection)
            coll.delete(ids=ids)
        except (ValueError, chromadb.errors.NotFoundError):
            pass
