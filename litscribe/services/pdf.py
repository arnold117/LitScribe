"""PDF parser service stub. Port from src/services/pdf_parser.py."""
from __future__ import annotations

from litscribe.models.analysis import ParsedDoc


class PDFService:
    """PDF download and parsing service. Port from src/services/pdf_parser.py."""

    async def parse(self, url: str, paper_id: str) -> ParsedDoc:
        raise NotImplementedError("Port from src/services/pdf_parser.py")
