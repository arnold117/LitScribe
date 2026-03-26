"""Citation formatter. Port from src/exporters/citation_formatter.py."""
from __future__ import annotations


def format_citation(paper: object, style: str = "apa") -> str:
    """Format a single paper as a citation string in the given style.

    Port from src/exporters/citation_formatter.py.
    Supported styles: apa, mla, chicago, ieee.
    """
    raise NotImplementedError("Port from src/exporters/citation_formatter.py")


def format_citations(papers: list, style: str = "apa") -> list[str]:
    """Format multiple papers as citation strings.

    Port from src/exporters/citation_formatter.py.
    """
    raise NotImplementedError("Port from src/exporters/citation_formatter.py")
