import pytest
from litscribe.models.paper import Paper


class MockSearchService:
    source_name = "mock"
    async def search(self, query: str, max_results: int = 10, **filters) -> list[Paper]:
        return [Paper(paper_id="mock:1", title=f"Result for: {query}", authors=["Author"], abstract="Abstract", year=2024, sources={"mock": "1"})]


def test_mock_service_implements_protocol():
    from litscribe.services.base import SearchService
    service = MockSearchService()
    assert isinstance(service, SearchService)


@pytest.mark.asyncio
async def test_mock_service_returns_papers():
    service = MockSearchService()
    results = await service.search("test query")
    assert len(results) == 1
    assert results[0].paper_id == "mock:1"


def test_dedup_papers():
    from litscribe.services.base import dedup_papers
    papers = [
        Paper(paper_id="a", title="Paper A", authors=["X"], abstract="abs", year=2024, sources={"arxiv": "1"}),
        Paper(paper_id="b", title="Paper B", authors=["Y"], abstract="abs", year=2024, sources={"s2": "2"}),
        Paper(paper_id="a", title="Paper A dup", authors=["X"], abstract="abs", year=2024, sources={"pubmed": "3"}),
    ]
    deduped = dedup_papers(papers)
    assert len(deduped) == 2
    paper_a = next(p for p in deduped if p.paper_id == "a")
    assert "arxiv" in paper_a.sources
    assert "pubmed" in paper_a.sources
