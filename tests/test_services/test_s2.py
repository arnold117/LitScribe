import pytest
from unittest.mock import AsyncMock, patch


def test_s2_service_implements_protocol():
    from litscribe.services.semantic_scholar import SemanticScholarService
    from litscribe.services.base import SearchService
    service = SemanticScholarService()
    assert isinstance(service, SearchService)
    assert service.source_name == "semantic_scholar"


@pytest.mark.asyncio
async def test_s2_search_converts_to_paper_model():
    from litscribe.services.semantic_scholar import SemanticScholarService
    mock_response = {
        "total": 1, "data": [{
            "paperId": "abc123", "title": "Test S2 Paper",
            "authors": [{"name": "Alice"}], "year": 2024,
            "citationCount": 42, "abstract": "S2 abstract",
            "venue": "NeurIPS", "url": "https://...",
            "openAccessPdf": {"url": "https://pdf.example.com/paper.pdf"},
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "2412.15115"},
        }]
    }
    with patch("litscribe.services.semantic_scholar.SemanticScholarService._api_request", return_value=mock_response):
        service = SemanticScholarService()
        results = await service.search("test", max_results=10)
        assert len(results) == 1
        assert results[0].paper_id == "s2:abc123"
        assert results[0].citations == 42
        assert len(results[0].pdf_urls) == 1
