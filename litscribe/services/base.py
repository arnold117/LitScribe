from __future__ import annotations
from typing import Protocol, runtime_checkable
from litscribe.models.paper import Paper

@runtime_checkable
class SearchService(Protocol):
    source_name: str
    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]: ...

def dedup_papers(papers: list[Paper]) -> list[Paper]:
    seen: dict[str, Paper] = {}
    for p in papers:
        if p.paper_id in seen:
            existing = seen[p.paper_id]
            merged_sources = {**existing.sources, **p.sources}
            merged_urls = list(set(existing.pdf_urls + p.pdf_urls))
            seen[p.paper_id] = existing.model_copy(update={"sources": merged_sources, "pdf_urls": merged_urls, "relevance_score": max(existing.relevance_score, p.relevance_score)})
        else:
            seen[p.paper_id] = p
    return list(seen.values())
