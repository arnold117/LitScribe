"""Tests that all service adapters implement the SearchService protocol."""
import pytest
from litscribe.services.base import SearchService


def test_all_services_implement_protocol():
    from litscribe.services.arxiv import ArxivService
    from litscribe.services.pubmed import PubMedService
    from litscribe.services.semantic_scholar import SemanticScholarService
    from litscribe.services.openalex import OpenAlexService
    from litscribe.services.europe_pmc import EuropePMCService
    from litscribe.services.zotero import ZoteroService

    for cls in [
        ArxivService,
        PubMedService,
        SemanticScholarService,
        OpenAlexService,
        EuropePMCService,
        ZoteroService,
    ]:
        service = cls()
        assert isinstance(service, SearchService), f"{cls.__name__} does not implement SearchService"
        assert hasattr(service, "source_name"), f"{cls.__name__} missing source_name"


def test_source_names_are_unique():
    from litscribe.services.arxiv import ArxivService
    from litscribe.services.pubmed import PubMedService
    from litscribe.services.semantic_scholar import SemanticScholarService
    from litscribe.services.openalex import OpenAlexService
    from litscribe.services.europe_pmc import EuropePMCService
    from litscribe.services.zotero import ZoteroService

    services = [
        ArxivService(),
        PubMedService(),
        SemanticScholarService(),
        OpenAlexService(),
        EuropePMCService(),
        ZoteroService(),
    ]
    names = [s.source_name for s in services]
    assert len(names) == len(set(names)), "Service source_names must be unique"


def test_pdf_service_stub_exists():
    from litscribe.services.pdf import PDFService

    svc = PDFService()
    assert svc is not None


def test_exporters_are_importable():
    from litscribe.exporters.bibtex import papers_to_bibtex, escape_bibtex
    from litscribe.exporters.pandoc import export_review, ExportFormat
    from litscribe.exporters.citation_formatter import format_citation, CitationStyle

    assert callable(papers_to_bibtex)
    assert callable(format_citation)
    assert callable(export_review)
