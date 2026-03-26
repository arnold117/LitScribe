"""BibTeX exporter for LitScribe v2.

Generates BibTeX bibliography files from Paper models.
Supports automatic entry type detection and proper escaping.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List

from litscribe.models.paper import Paper


@dataclass
class BibTeXEntry:
    """A single BibTeX entry."""

    entry_type: str  # article, inproceedings, misc, etc.
    cite_key: str
    fields: Dict[str, str]

    def to_bibtex(self) -> str:
        """Convert to BibTeX string format."""
        lines = [f"@{self.entry_type}{{{self.cite_key},"]
        for key, value in self.fields.items():
            if value:
                escaped = escape_bibtex(value)
                lines.append(f"  {key} = {{{escaped}}},")
        lines.append("}")
        return "\n".join(lines)


def escape_bibtex(text: str) -> str:
    """Escape special LaTeX/BibTeX characters.

    Args:
        text: Raw text to escape

    Returns:
        Text with special characters escaped
    """
    if not text:
        return ""

    replacements = [
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]

    result = text
    for old, new in replacements:
        result = re.sub(rf"(?<!\\){re.escape(old)}", new, result)

    return result


def normalize_author_name(name: str) -> str:
    """Normalize author name for BibTeX (Last, First format)."""
    name = name.strip()
    if "," in name:
        return name
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return f"{parts[1]}, {parts[0]}"
    else:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"


def generate_cite_key(authors: list[str], year: int, title: str) -> str:
    """Generate a citation key from author list, year, and title.

    Format: FirstAuthorLastname_Year_FirstWordOfTitle

    Args:
        authors: List of author name strings
        year: Publication year
        title: Paper title

    Returns:
        Citation key string
    """
    if authors:
        first_author = authors[0]
        if "," in first_author:
            last_name = first_author.split(",")[0]
        else:
            parts = first_author.split()
            if not parts:
                last_name = "Unknown"
            elif len(parts) >= 2 and len(parts[-1]) == 1:
                # Format "LastName I" — last token is a single initial
                last_name = parts[0]
            else:
                last_name = parts[-1]
    else:
        last_name = "Unknown"

    last_name = unicodedata.normalize("NFKD", last_name)
    last_name = last_name.encode("ascii", "ignore").decode("ascii")
    last_name = re.sub(r"[^a-zA-Z]", "", last_name)

    year_str = str(year) if year else "XXXX"

    title_words = re.sub(r"[^\w\s]", "", title).split()
    stop_words = {"a", "an", "the", "on", "in", "of", "for", "to", "with"}
    first_word = "paper"
    for word in title_words:
        if word.lower() not in stop_words:
            first_word = word
            break

    return f"{last_name}{year_str}{first_word}".lower()


def detect_entry_type(venue: str) -> str:
    """Detect the appropriate BibTeX entry type from a venue string.

    Args:
        venue: Publication venue name

    Returns:
        BibTeX entry type: 'article', 'inproceedings', or 'misc'
    """
    venue_lower = venue.lower()

    conference_keywords = [
        "conference", "proceedings", "workshop", "symposium",
        "icml", "neurips", "nips", "iclr", "cvpr", "iccv", "acl",
        "emnlp", "naacl", "aaai", "ijcai", "sigir", "kdd", "www",
    ]
    if any(kw in venue_lower for kw in conference_keywords):
        return "inproceedings"

    journal_keywords = [
        "journal", "transactions", "review", "letters", "magazine",
        "nature", "science", "cell", "lancet", "nejm", "jama",
    ]
    if any(kw in venue_lower for kw in journal_keywords):
        return "article"

    if "arxiv" in venue_lower:
        return "misc"

    return "misc"


def _generate_bibtex_entry(paper: Paper, used_keys: set[str] | None = None) -> BibTeXEntry:
    """Generate a BibTeX entry from a v2 Paper model."""
    entry_type = detect_entry_type(paper.venue)
    cite_key = generate_cite_key(paper.authors, paper.year, paper.title)

    # Ensure unique key
    if used_keys is not None:
        if cite_key in used_keys:
            suffix = ord("a")
            while f"{cite_key}{chr(suffix)}" in used_keys:
                suffix += 1
            cite_key = f"{cite_key}{chr(suffix)}"
        used_keys.add(cite_key)

    fields: Dict[str, str] = {}

    if paper.title:
        fields["title"] = paper.title

    if paper.authors:
        normalized = [normalize_author_name(a) for a in paper.authors]
        fields["author"] = " and ".join(normalized)

    if paper.year:
        fields["year"] = str(paper.year)

    if paper.abstract:
        fields["abstract"] = paper.abstract

    if entry_type == "article":
        fields["journal"] = paper.venue if paper.venue else "Unknown Journal"
    elif entry_type == "inproceedings":
        fields["booktitle"] = paper.venue if paper.venue else "Unknown Conference"

    if paper.doi:
        fields["doi"] = paper.doi
    elif paper.paper_id.startswith("10."):
        fields["doi"] = paper.paper_id

    # arXiv source handling
    arxiv_id = paper.sources.get("arxiv", "")
    if arxiv_id:
        fields["eprint"] = arxiv_id
        fields["archiveprefix"] = "arXiv"

    return BibTeXEntry(entry_type=entry_type, cite_key=cite_key, fields=fields)


def papers_to_bibtex(papers: list[Paper]) -> str:
    """Generate a BibTeX bibliography string from a list of v2 Paper models.

    Args:
        papers: List of Paper objects

    Returns:
        Complete BibTeX file content as a string
    """
    used_keys: set[str] = set()
    entries = [_generate_bibtex_entry(p, used_keys) for p in papers]

    lines = [
        "% BibTeX bibliography generated by LitScribe",
        f"% Contains {len(entries)} entries",
        "%",
        "% Citation keys follow the format: AuthorYearFirstword",
        "% e.g., smith2023deep for 'Deep Learning for NLP' by Smith (2023)",
        "",
    ]
    for entry in entries:
        lines.append(entry.to_bibtex())
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "BibTeXEntry",
    "escape_bibtex",
    "normalize_author_name",
    "generate_cite_key",
    "detect_entry_type",
    "papers_to_bibtex",
]
