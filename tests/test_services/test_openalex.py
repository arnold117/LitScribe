import pytest
from unittest.mock import patch, AsyncMock, MagicMock


def test_openalex_service_implements_protocol():
    from litscribe.services.openalex import OpenAlexService
    from litscribe.services.base import SearchService

    service = OpenAlexService()
    assert isinstance(service, SearchService)
    assert service.source_name == "openalex"


@pytest.mark.asyncio
async def test_openalex_search_converts_to_paper_model():
    from litscribe.services.openalex import OpenAlexService

    mock_work = {
        "id": "https://openalex.org/W2741809807",
        "display_name": "Attention Is All You Need",
        "authorships": [
            {"author": {"display_name": "Ashish Vaswani"}},
            {"author": {"display_name": "Noam Shazeer"}},
        ],
        "abstract_inverted_index": {
            "The": [0],
            "dominant": [1],
            "sequence": [2],
        },
        "publication_year": 2017,
        "cited_by_count": 50000,
        "doi": "https://doi.org/10.48550/arXiv.1706.03762",
        "primary_location": {
            "source": {"display_name": "arXiv"},
        },
        "locations": [
            {"pdf_url": "https://arxiv.org/pdf/1706.03762"},
        ],
    }

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"results": [mock_work]})

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_get)

    with patch("litscribe.services.openalex.aiohttp.ClientSession", return_value=mock_session):
        service = OpenAlexService()
        results = await service.search("attention mechanism", max_results=5)

    assert len(results) == 1
    paper = results[0]
    assert paper.paper_id == "openalex:W2741809807"
    assert paper.title == "Attention Is All You Need"
    assert "Ashish Vaswani" in paper.authors
    assert "Noam Shazeer" in paper.authors
    assert paper.abstract == "The dominant sequence"
    assert paper.year == 2017
    assert paper.citations == 50000
    assert paper.doi == "10.48550/arXiv.1706.03762"
    assert paper.venue == "arXiv"
    assert "https://arxiv.org/pdf/1706.03762" in paper.pdf_urls
    assert "openalex" in paper.sources


@pytest.mark.asyncio
async def test_openalex_search_returns_empty_on_error():
    from litscribe.services.openalex import OpenAlexService

    mock_response = MagicMock()
    mock_response.status = 500
    mock_response.text = AsyncMock(return_value="Internal Server Error")

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_get)

    with patch("litscribe.services.openalex.aiohttp.ClientSession", return_value=mock_session):
        service = OpenAlexService()
        results = await service.search("test")

    assert results == []
