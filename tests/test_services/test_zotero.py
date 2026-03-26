import pytest
from unittest.mock import patch, MagicMock


def test_zotero_service_implements_protocol():
    from litscribe.services.zotero import ZoteroService
    from litscribe.services.base import SearchService

    service = ZoteroService(api_key="test", library_id="12345")
    assert isinstance(service, SearchService)
    assert service.source_name == "zotero"


@pytest.mark.asyncio
async def test_zotero_search_converts_to_paper_model():
    from litscribe.services.zotero import ZoteroService

    mock_item = {
        "data": {
            "key": "ABCD1234",
            "itemType": "journalArticle",
            "title": "Deep Learning for Natural Language Processing",
            "creators": [
                {"creatorType": "author", "lastName": "LeCun", "firstName": "Yann"},
                {"creatorType": "author", "lastName": "Bengio", "firstName": "Yoshua"},
            ],
            "abstractNote": "A survey of deep learning methods for NLP.",
            "date": "2015-05-28",
            "publicationTitle": "Nature",
            "DOI": "10.1038/nature14539",
            "collections": [],
            "tags": [],
        }
    }

    mock_zot = MagicMock()
    mock_zot.items.return_value = [mock_item]

    with patch("litscribe.services.zotero.ZoteroService._make_client", return_value=mock_zot):
        service = ZoteroService(api_key="test", library_id="12345")
        results = await service.search("deep learning NLP", max_results=5)

    assert len(results) == 1
    paper = results[0]
    assert paper.paper_id == "zotero:ABCD1234"
    assert paper.title == "Deep Learning for Natural Language Processing"
    assert "LeCun Yann" in paper.authors
    assert "Bengio Yoshua" in paper.authors
    assert paper.abstract == "A survey of deep learning methods for NLP."
    assert paper.year == 2015
    assert paper.venue == "Nature"
    assert paper.doi == "10.1038/nature14539"
    assert "zotero" in paper.sources
    assert paper.sources["zotero"] == "ABCD1234"


@pytest.mark.asyncio
async def test_zotero_search_filters_attachments():
    from litscribe.services.zotero import ZoteroService

    items = [
        {"data": {"key": "K1", "itemType": "journalArticle", "title": "Paper A",
                  "creators": [], "date": "2020", "publicationTitle": "", "DOI": "",
                  "abstractNote": ""}},
        {"data": {"key": "K2", "itemType": "attachment", "title": "PDF",
                  "creators": [], "date": "", "publicationTitle": "", "DOI": "",
                  "abstractNote": ""}},
        {"data": {"key": "K3", "itemType": "note", "title": "My note",
                  "creators": [], "date": "", "publicationTitle": "", "DOI": "",
                  "abstractNote": ""}},
    ]

    mock_zot = MagicMock()
    mock_zot.items.return_value = items

    with patch("litscribe.services.zotero.ZoteroService._make_client", return_value=mock_zot):
        service = ZoteroService(api_key="test", library_id="12345")
        results = await service.search("anything", max_results=10)

    assert len(results) == 1
    assert results[0].paper_id == "zotero:K1"


@pytest.mark.asyncio
async def test_zotero_search_returns_empty_on_exception():
    from litscribe.services.zotero import ZoteroService

    with patch(
        "litscribe.services.zotero.ZoteroService._make_client",
        side_effect=RuntimeError("no credentials"),
    ):
        service = ZoteroService()
        results = await service.search("test")

    assert results == []
