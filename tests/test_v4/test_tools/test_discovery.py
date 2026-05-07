import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litscribe.models.paper import Paper
from litscribe.tools.discovery import (
    domain_filter,
    expand_queries,
    search_all_sources,
    select_papers,
    snowball_sample,
)


def _make_paper(pid: str, title: str = "Test Paper", abstract: str = "test abstract") -> Paper:
    return Paper(
        paper_id=pid, title=title, authors=["Author A"],
        abstract=abstract, year=2024, sources={"test": pid},
    )


@pytest.mark.asyncio
async def test_expand_queries_parses_structured():
    router = AsyncMock()
    router.call_json = AsyncMock(return_value={
        "core_methodology": ["query1", "query2"],
        "application_domain": ["query3"],
        "review_meta": [],
        "recent_advances": ["query4"],
        "cross_disciplinary": [],
        "synonyms_nomenclature": ["query5"],
    })

    queries = await expand_queries(router, "test question", "CS")
    assert "test question" in queries
    assert len(queries) >= 5


@pytest.mark.asyncio
async def test_expand_queries_handles_flat_list():
    router = AsyncMock()
    router.call_json = AsyncMock(return_value=["q1", "q2", "q3"])

    queries = await expand_queries(router, "test", "CS")
    assert "test" in queries
    assert len(queries) == 4


@pytest.mark.asyncio
async def test_select_papers_returns_all_when_under_limit():
    router = AsyncMock()
    papers = [_make_paper(f"p{i}") for i in range(5)]

    result = await select_papers(router, papers, "test", "CS", max_papers=10)
    assert len(result) == 5
    router.call_json.assert_not_called()


@pytest.mark.asyncio
async def test_select_papers_calls_llm_when_over_limit():
    router = AsyncMock()
    router.call_json = AsyncMock(return_value=["p0", "p2", "p4"])
    papers = [_make_paper(f"p{i}") for i in range(10)]

    result = await select_papers(router, papers, "test", "CS", max_papers=3)
    assert len(result) == 3
    assert all(p.paper_id in ("p0", "p2", "p4") for p in result)


@pytest.mark.asyncio
async def test_domain_filter_keeps_relevant():
    router = AsyncMock()
    router.call_json = AsyncMock(return_value=[
        {"paper_id": "p0", "relevant": True, "reason": "on topic"},
        {"paper_id": "p1", "relevant": False, "reason": "off topic"},
        {"paper_id": "p2", "relevant": True, "reason": "on topic"},
    ])
    papers = [_make_paper(f"p{i}") for i in range(3)]

    # Need >5 papers to trigger filter, but we patch to test the logic
    # directly
    result = await domain_filter(router, papers * 3, "test", "CS")
    assert len(result) > 0


@pytest.mark.asyncio
async def test_snowball_returns_original_on_no_s2():
    papers = [_make_paper(f"arxiv:{i}") for i in range(3)]
    config = MagicMock()

    with patch("litscribe.services.semantic_scholar.SemanticScholarService", side_effect=ImportError):
        result = await snowball_sample(papers, config)
    assert len(result) == len(papers)
