import pytest
from unittest.mock import patch, MagicMock


def test_pubmed_service_implements_protocol():
    from litscribe.services.pubmed import PubMedService
    from litscribe.services.base import SearchService
    service = PubMedService()
    assert isinstance(service, SearchService)
    assert service.source_name == "pubmed"


@pytest.mark.asyncio
async def test_pubmed_search_converts_to_paper_model():
    from litscribe.services.pubmed import PubMedService
    mock_record = {
        "PMID": "12345678",
        "TI": "Test PubMed Paper",
        "AU": ["Smith J", "Jones A"],
        "AB": "Test abstract text",
        "DP": "2024 Jan",
        "SO": "Nature. 2024;123:45-67",
        "AID": ["10.1038/test [doi]"],
    }

    with patch("litscribe.services.pubmed.PubMedService._fetch_records", return_value=[mock_record]):
        service = PubMedService()
        results = await service.search("test", max_results=10)
        assert len(results) == 1
        assert results[0].paper_id == "pmid:12345678"
        assert results[0].doi == "10.1038/test"
        assert results[0].year == 2024
