"""Exporters for LitScribe literature reviews.

This module provides various export formats:
- BibTeX citations
- Multiple citation styles (APA, MLA, IEEE, Chicago, GB/T 7714)
- Pandoc export (Word, PDF via LaTeX)
"""

from exporters.bibtex_exporter import (
    BibTeXExporter,
    generate_bibtex,
    generate_bibtex_entry,
)
from exporters.citation_formatter import (
    CitationFormatter,
    CitationStyle,
    format_citation,
)
from exporters.pandoc_exporter import (
    ExportConfig,
    ExportFormat,
    PandocExporter,
    export_review,
    generate_review_markdown,
)

__all__ = [
    # BibTeX
    "BibTeXExporter",
    "generate_bibtex",
    "generate_bibtex_entry",
    # Citation formatting
    "CitationFormatter",
    "CitationStyle",
    "format_citation",
    # Pandoc export
    "ExportConfig",
    "ExportFormat",
    "PandocExporter",
    "export_review",
    "generate_review_markdown",
]
