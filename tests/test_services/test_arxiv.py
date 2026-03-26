import pytest
from unittest.mock import MagicMock, patch, AsyncMock


def test_arxiv_service_implements_protocol():
    from litscribe.services.arxiv import ArxivService
    from litscribe.services.base import SearchService
    service = ArxivService()
    assert isinstance(service, SearchService)
    assert service.source_name == "arxiv"


@pytest.mark.asyncio
async def test_arxiv_search_converts_to_paper_model():
    from litscribe.services.arxiv import ArxivService
    # Mock the arxiv library
    mock_result = MagicMock()
    mock_result.entry_id = "http://arxiv.org/abs/2412.15115v1"
    mock_result.title = "Test Paper Title"
    mock_result.authors = [MagicMock(name="Alice"), MagicMock(name="Bob")]
    mock_result.summary = "Test abstract"
    mock_result.published = MagicMock(year=2024)
    mock_result.categories = ["cs.CL", "cs.AI"]
    mock_result.doi = "10.1234/test"
    mock_result.pdf_url = "http://arxiv.org/pdf/2412.15115v1"

    # Fix: MagicMock.name is special, need to set it differently
    mock_result.authors[0].name = "Alice"
    mock_result.authors[1].name = "Bob"

    with patch("litscribe.services.arxiv._do_search", return_value=[mock_result]):
        service = ArxivService()
        results = await service.search("test query", max_results=10)
        assert len(results) == 1
        assert results[0].paper_id == "arxiv:2412.15115"
        assert results[0].title == "Test Paper Title"
        assert "arxiv" in results[0].sources
