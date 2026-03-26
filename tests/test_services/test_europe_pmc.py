import pytest
from unittest.mock import patch, AsyncMock, MagicMock


def test_europe_pmc_service_implements_protocol():
    from litscribe.services.europe_pmc import EuropePMCService
    from litscribe.services.base import SearchService

    service = EuropePMCService()
    assert isinstance(service, SearchService)
    assert service.source_name == "europe_pmc"


@pytest.mark.asyncio
async def test_europe_pmc_search_converts_to_paper_model():
    from litscribe.services.europe_pmc import EuropePMCService

    mock_hit = {
        "pmid": "12345678",
        "pmcid": "PMC1234567",
        "title": "CRISPR-Cas9 Gene Editing in Human Cells",
        "authorString": "Zhang F; Cong L; Ran FA",
        "abstractText": "We report the use of CRISPR-Cas9 for genome editing.",
        "pubYear": "2013",
        "citedByCount": 8000,
        "doi": "10.1126/science.1231143",
        "isOpenAccess": "Y",
        "journalInfo": {
            "journal": {"title": "Science"}
        },
    }

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "resultList": {"result": [mock_hit]}
    })

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_get)

    with patch("litscribe.services.europe_pmc.aiohttp.ClientSession", return_value=mock_session):
        service = EuropePMCService()
        results = await service.search("CRISPR gene editing", max_results=5)

    assert len(results) == 1
    paper = results[0]
    assert paper.paper_id == "europepmc:12345678"
    assert paper.title == "CRISPR-Cas9 Gene Editing in Human Cells"
    assert "Zhang F" in paper.authors
    assert "Cong L" in paper.authors
    assert "Ran FA" in paper.authors
    assert "CRISPR-Cas9" in paper.abstract
    assert paper.year == 2013
    assert paper.citations == 8000
    assert paper.doi == "10.1126/science.1231143"
    assert paper.venue == "Science"
    assert "https://europepmc.org/articles/PMC1234567/pdf" in paper.pdf_urls
    assert "europe_pmc" in paper.sources


@pytest.mark.asyncio
async def test_europe_pmc_author_comma_separated():
    from litscribe.services.europe_pmc import EuropePMCService

    mock_hit = {
        "pmid": "99999999",
        "title": "Comma Author Test",
        "authorString": "Smith J, Jones A, Brown B",
        "abstractText": "",
        "pubYear": "2020",
        "citedByCount": 10,
        "isOpenAccess": "N",
        "journalInfo": {},
    }

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "resultList": {"result": [mock_hit]}
    })

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_get)

    with patch("litscribe.services.europe_pmc.aiohttp.ClientSession", return_value=mock_session):
        service = EuropePMCService()
        results = await service.search("test", max_results=5)

    assert len(results) == 1
    assert results[0].authors == ["Smith J", "Jones A", "Brown B"]


@pytest.mark.asyncio
async def test_europe_pmc_min_citations_filter():
    from litscribe.services.europe_pmc import EuropePMCService

    hits = [
        {"pmid": "1", "title": "A", "authorString": "", "abstractText": "",
         "pubYear": "2020", "citedByCount": 5, "isOpenAccess": "N", "journalInfo": {}},
        {"pmid": "2", "title": "B", "authorString": "", "abstractText": "",
         "pubYear": "2021", "citedByCount": 100, "isOpenAccess": "N", "journalInfo": {}},
    ]

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"resultList": {"result": hits}})

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_get)

    with patch("litscribe.services.europe_pmc.aiohttp.ClientSession", return_value=mock_session):
        service = EuropePMCService()
        results = await service.search("test", max_results=10, min_citations=50)

    assert len(results) == 1
    assert results[0].paper_id == "europepmc:2"
