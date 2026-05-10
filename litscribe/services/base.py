from __future__ import annotations
from typing import Protocol, runtime_checkable
from litscribe.models.paper import Paper

@runtime_checkable
class SearchService(Protocol):
    source_name: str
    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]: ...

def dedup_papers(papers: list[Paper]) -> list[Paper]:
    seen_id: dict[str, Paper] = {}
    seen_doi: dict[str, str] = {}  # doi → paper_id

    for p in papers:
        # Check DOI dedup first
        if p.doi and p.doi in seen_doi:
            existing_id = seen_doi[p.doi]
            existing = seen_id[existing_id]
            merged_sources = {**existing.sources, **p.sources}
            merged_urls = list(set(existing.pdf_urls + p.pdf_urls))
            seen_id[existing_id] = existing.model_copy(update={
                "sources": merged_sources,
                "pdf_urls": merged_urls,
                "relevance_score": max(existing.relevance_score, p.relevance_score),
            })
            continue

        # Check paper_id dedup
        if p.paper_id in seen_id:
            existing = seen_id[p.paper_id]
            merged_sources = {**existing.sources, **p.sources}
            merged_urls = list(set(existing.pdf_urls + p.pdf_urls))
            seen_id[p.paper_id] = existing.model_copy(update={
                "sources": merged_sources,
                "pdf_urls": merged_urls,
                "relevance_score": max(existing.relevance_score, p.relevance_score),
            })
        else:
            seen_id[p.paper_id] = p
            if p.doi:
                seen_doi[p.doi] = p.paper_id

    return list(seen_id.values())
