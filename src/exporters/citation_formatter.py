"""Citation formatter for multiple academic styles.

Supports:
- APA 7th Edition
- MLA 9th Edition
- IEEE
- Chicago (Author-Date)
- GB/T 7714-2015 (Chinese national standard)
"""

from enum import Enum
from typing import List, Optional

from agents.state import PaperSummary


class CitationStyle(Enum):
    """Supported citation styles."""

    APA = "apa"
    MLA = "mla"
    IEEE = "ieee"
    CHICAGO = "chicago"
    GB_T_7714 = "gbt7714"  # Chinese national standard


def format_authors_apa(authors: List[str], max_authors: int = 20) -> str:
    """Format authors in APA style.

    Rules:
    - Up to 20 authors: list all
    - 21+ authors: first 19, ..., last author

    Args:
        authors: List of author names
        max_authors: Maximum authors before truncation

    Returns:
        Formatted author string
    """
    if not authors:
        return ""

    def format_name(name: str) -> str:
        """Convert 'First Last' to 'Last, F.' format."""
        name = name.strip()
        if "," in name:
            parts = name.split(",")
            last = parts[0].strip()
            first = parts[1].strip() if len(parts) > 1 else ""
            initials = ". ".join(w[0].upper() for w in first.split() if w)
            return f"{last}, {initials}." if initials else last
        else:
            parts = name.split()
            if len(parts) == 1:
                return parts[0]
            last = parts[-1]
            initials = ". ".join(w[0].upper() for w in parts[:-1] if w)
            return f"{last}, {initials}." if initials else last

    formatted = [format_name(a) for a in authors]

    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    elif len(formatted) <= max_authors:
        return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
    else:
        # 21+ authors: first 19, ..., last
        return ", ".join(formatted[:19]) + ", ... " + formatted[-1]


def format_authors_mla(authors: List[str]) -> str:
    """Format authors in MLA style.

    Rules:
    - 1 author: Last, First.
    - 2 authors: Last, First, and First Last.
    - 3+ authors: Last, First, et al.

    Args:
        authors: List of author names

    Returns:
        Formatted author string
    """
    if not authors:
        return ""

    def format_first(name: str) -> str:
        """Format first author as 'Last, First'."""
        name = name.strip()
        if "," in name:
            return name
        parts = name.split()
        if len(parts) == 1:
            return parts[0]
        return f"{parts[-1]}, {' '.join(parts[:-1])}"

    def format_other(name: str) -> str:
        """Format subsequent authors as 'First Last'."""
        name = name.strip()
        if "," not in name:
            return name
        parts = name.split(",")
        return f"{parts[1].strip()} {parts[0].strip()}"

    if len(authors) == 1:
        return format_first(authors[0])
    elif len(authors) == 2:
        return f"{format_first(authors[0])}, and {format_other(authors[1])}"
    else:
        return f"{format_first(authors[0])}, et al."


def format_authors_ieee(authors: List[str]) -> str:
    """Format authors in IEEE style.

    Format: F. Last, F. M. Last, and F. Last

    Args:
        authors: List of author names

    Returns:
        Formatted author string
    """
    if not authors:
        return ""

    def format_name(name: str) -> str:
        """Convert to 'F. M. Last' format."""
        name = name.strip()
        if "," in name:
            parts = name.split(",")
            last = parts[0].strip()
            first = parts[1].strip() if len(parts) > 1 else ""
            initials = " ".join(w[0].upper() + "." for w in first.split() if w)
            return f"{initials} {last}" if initials else last
        else:
            parts = name.split()
            if len(parts) == 1:
                return parts[0]
            last = parts[-1]
            initials = " ".join(w[0].upper() + "." for w in parts[:-1] if w)
            return f"{initials} {last}" if initials else last

    formatted = [format_name(a) for a in authors]

    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    else:
        return ", ".join(formatted[:-1]) + ", and " + formatted[-1]


def format_authors_chicago(authors: List[str]) -> str:
    """Format authors in Chicago style (Author-Date).

    Rules:
    - 1 author: Last, First
    - 2-3 authors: Last, First, First Last, and First Last
    - 4+ authors: Last, First, et al.

    Args:
        authors: List of author names

    Returns:
        Formatted author string
    """
    if not authors:
        return ""

    def format_first(name: str) -> str:
        """Format first author as 'Last, First'."""
        name = name.strip()
        if "," in name:
            return name
        parts = name.split()
        if len(parts) == 1:
            return parts[0]
        return f"{parts[-1]}, {' '.join(parts[:-1])}"

    def format_other(name: str) -> str:
        """Format subsequent authors as 'First Last'."""
        name = name.strip()
        if "," not in name:
            return name
        parts = name.split(",")
        return f"{parts[1].strip()} {parts[0].strip()}"

    if len(authors) == 1:
        return format_first(authors[0])
    elif len(authors) <= 3:
        result = format_first(authors[0])
        for i, a in enumerate(authors[1:], 1):
            if i == len(authors) - 1:
                result += f", and {format_other(a)}"
            else:
                result += f", {format_other(a)}"
        return result
    else:
        return f"{format_first(authors[0])}, et al."


def format_authors_gbt7714(authors: List[str], max_authors: int = 3) -> str:
    """Format authors in GB/T 7714-2015 style (Chinese standard).

    Rules:
    - Up to 3 authors: list all, separated by commas
    - 4+ authors: first 3, 等 (et al.)
    - Names in original order (no comma inversion for Chinese)

    Args:
        authors: List of author names
        max_authors: Maximum authors before truncation (default: 3)

    Returns:
        Formatted author string
    """
    if not authors:
        return ""

    # For Chinese names, keep original order
    # For Western names, keep as-is (no inversion required in references)
    if len(authors) <= max_authors:
        return ", ".join(authors)
    else:
        return ", ".join(authors[:max_authors]) + ", 等"


class CitationFormatter:
    """Format citations in various academic styles."""

    def __init__(self, style: CitationStyle = CitationStyle.APA):
        """Initialize formatter with a style.

        Args:
            style: Citation style to use
        """
        self.style = style

    def format_paper(self, paper: PaperSummary, index: Optional[int] = None) -> str:
        """Format a single paper citation.

        Args:
            paper: Paper summary to format
            index: Optional index for numbered styles (IEEE)

        Returns:
            Formatted citation string
        """
        if self.style == CitationStyle.APA:
            return self._format_apa(paper)
        elif self.style == CitationStyle.MLA:
            return self._format_mla(paper)
        elif self.style == CitationStyle.IEEE:
            return self._format_ieee(paper, index)
        elif self.style == CitationStyle.CHICAGO:
            return self._format_chicago(paper)
        elif self.style == CitationStyle.GB_T_7714:
            return self._format_gbt7714(paper, index)
        else:
            return self._format_apa(paper)

    def _format_apa(self, paper: PaperSummary) -> str:
        """Format in APA 7th Edition style.

        Format: Author(s) (Year). Title. Venue. DOI/URL
        """
        authors = format_authors_apa(paper.get("authors", []))
        year = paper.get("year", "n.d.")
        title = paper.get("title", "Untitled")
        venue = paper.get("venue", "")
        paper_id = paper.get("paper_id", "")

        # Build citation
        parts = []

        # Author (Year)
        if authors:
            parts.append(f"{authors} ({year}).")
        else:
            parts.append(f"({year}).")

        # Title (italicized in actual rendering)
        parts.append(f"{title}.")

        # Venue
        if venue:
            parts.append(f"*{venue}*.")

        # DOI or identifier
        if paper_id:
            if paper_id.startswith("10."):
                parts.append(f"https://doi.org/{paper_id}")
            elif "arxiv" in paper.get("source", ""):
                parts.append(f"https://arxiv.org/abs/{paper_id.replace('arxiv:', '')}")

        return " ".join(parts)

    def _format_mla(self, paper: PaperSummary) -> str:
        """Format in MLA 9th Edition style.

        Format: Author(s). "Title." Venue, Year.
        """
        authors = format_authors_mla(paper.get("authors", []))
        year = paper.get("year", "n.d.")
        title = paper.get("title", "Untitled")
        venue = paper.get("venue", "")

        parts = []

        # Author
        if authors:
            parts.append(f"{authors}.")

        # Title in quotes
        parts.append(f'"{title}."')

        # Venue (italicized)
        if venue:
            parts.append(f"*{venue}*,")

        # Year
        parts.append(f"{year}.")

        return " ".join(parts)

    def _format_ieee(self, paper: PaperSummary, index: Optional[int] = None) -> str:
        """Format in IEEE style.

        Format: [#] F. Last, F. Last, "Title," Venue, Year.
        """
        authors = format_authors_ieee(paper.get("authors", []))
        year = paper.get("year", "")
        title = paper.get("title", "Untitled")
        venue = paper.get("venue", "")

        parts = []

        # Index number
        if index is not None:
            parts.append(f"[{index}]")

        # Authors
        if authors:
            parts.append(f"{authors},")

        # Title in quotes
        parts.append(f'"{title},"')

        # Venue (italicized)
        if venue:
            parts.append(f"*{venue}*,")

        # Year
        if year:
            parts.append(f"{year}.")
        else:
            # Remove trailing comma and add period
            if parts:
                parts[-1] = parts[-1].rstrip(",") + "."

        return " ".join(parts)

    def _format_chicago(self, paper: PaperSummary) -> str:
        """Format in Chicago style (Author-Date).

        Format: Author(s). Year. "Title." Venue.
        """
        authors = format_authors_chicago(paper.get("authors", []))
        year = paper.get("year", "n.d.")
        title = paper.get("title", "Untitled")
        venue = paper.get("venue", "")

        parts = []

        # Author
        if authors:
            parts.append(f"{authors}.")

        # Year
        parts.append(f"{year}.")

        # Title in quotes
        parts.append(f'"{title}."')

        # Venue (italicized)
        if venue:
            parts.append(f"*{venue}*.")

        return " ".join(parts)

    def _format_gbt7714(self, paper: PaperSummary, index: Optional[int] = None) -> str:
        """Format in GB/T 7714-2015 style (Chinese national standard).

        Format: [#] 作者. 题名[文献类型]. 刊名, 年, 卷(期): 页码.
        For journal articles: [J]
        For conference papers: [C]
        For electronic resources: [EB/OL]
        """
        authors = format_authors_gbt7714(paper.get("authors", []))
        year = paper.get("year", "")
        title = paper.get("title", "Untitled")
        venue = paper.get("venue", "")
        source = paper.get("source", "")

        # Determine document type
        if "conference" in venue.lower() or "proceedings" in venue.lower():
            doc_type = "[C]"
        elif source == "arxiv":
            doc_type = "[EB/OL]"
        else:
            doc_type = "[J]"

        parts = []

        # Index number
        if index is not None:
            parts.append(f"[{index}]")

        # Authors
        if authors:
            parts.append(f"{authors}.")

        # Title with document type
        parts.append(f"{title}{doc_type}.")

        # Venue
        if venue:
            parts.append(f"{venue},")

        # Year
        if year:
            parts.append(f"{year}.")

        return " ".join(parts)

    def format_papers(
        self,
        papers: List[PaperSummary],
        numbered: bool = False,
    ) -> List[str]:
        """Format multiple paper citations.

        Args:
            papers: List of paper summaries
            numbered: Whether to include numbers (for IEEE/GB/T 7714)

        Returns:
            List of formatted citation strings
        """
        citations = []
        for i, paper in enumerate(papers, 1):
            index = i if numbered else None
            citations.append(self.format_paper(paper, index))
        return citations


def format_citation(
    paper: PaperSummary,
    style: CitationStyle = CitationStyle.APA,
) -> str:
    """Convenience function to format a single citation.

    Args:
        paper: Paper summary to format
        style: Citation style to use

    Returns:
        Formatted citation string
    """
    formatter = CitationFormatter(style)
    return formatter.format_paper(paper)


# Export
__all__ = [
    "CitationStyle",
    "CitationFormatter",
    "format_citation",
    "format_authors_apa",
    "format_authors_mla",
    "format_authors_ieee",
    "format_authors_chicago",
    "format_authors_gbt7714",
]
