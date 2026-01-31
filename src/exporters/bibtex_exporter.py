"""BibTeX exporter for LitScribe.

Generates BibTeX bibliography files from analyzed papers.
Supports automatic entry type detection and proper escaping.
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.state import LitScribeState, PaperSummary


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
                # Escape special characters
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

    # BibTeX special characters that need escaping
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
        # Don't escape if already escaped
        result = re.sub(rf"(?<!\\){re.escape(old)}", new, result)

    return result


def normalize_author_name(name: str) -> str:
    """Normalize author name for BibTeX.

    Converts various formats to "Lastname, Firstname" format.

    Args:
        name: Author name in any format

    Returns:
        Normalized name for BibTeX
    """
    name = name.strip()

    # Already in "Last, First" format
    if "," in name:
        return name

    # Handle "First Last" format
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return f"{parts[1]}, {parts[0]}"
    else:
        # Assume last word is surname, rest is first name
        return f"{parts[-1]}, {' '.join(parts[:-1])}"


def generate_cite_key(paper: PaperSummary) -> str:
    """Generate a unique citation key for a paper.

    Format: FirstAuthorLastname_Year_FirstWordOfTitle

    Args:
        paper: Paper summary

    Returns:
        Citation key string
    """
    # Get first author's last name
    authors = paper.get("authors", [])
    if authors:
        first_author = authors[0]
        # Extract last name
        if "," in first_author:
            last_name = first_author.split(",")[0]
        else:
            parts = first_author.split()
            last_name = parts[-1] if parts else "Unknown"
    else:
        last_name = "Unknown"

    # Normalize: remove accents, special chars
    last_name = unicodedata.normalize("NFKD", last_name)
    last_name = last_name.encode("ascii", "ignore").decode("ascii")
    last_name = re.sub(r"[^a-zA-Z]", "", last_name)

    # Get year
    year = paper.get("year", 0)
    year_str = str(year) if year else "XXXX"

    # Get first significant word of title
    title = paper.get("title", "")
    # Remove common articles
    title_words = re.sub(r"[^\w\s]", "", title).split()
    stop_words = {"a", "an", "the", "on", "in", "of", "for", "to", "with"}
    first_word = "paper"
    for word in title_words:
        if word.lower() not in stop_words:
            first_word = word
            break

    return f"{last_name}{year_str}{first_word}".lower()


def detect_entry_type(paper: PaperSummary) -> str:
    """Detect the appropriate BibTeX entry type.

    Args:
        paper: Paper summary

    Returns:
        BibTeX entry type (article, inproceedings, misc, etc.)
    """
    venue = paper.get("venue", "").lower()
    source = paper.get("source", "").lower()

    # Conference indicators
    conference_keywords = [
        "conference", "proceedings", "workshop", "symposium",
        "icml", "neurips", "nips", "iclr", "cvpr", "iccv", "acl",
        "emnlp", "naacl", "aaai", "ijcai", "sigir", "kdd", "www",
    ]
    if any(kw in venue for kw in conference_keywords):
        return "inproceedings"

    # Journal indicators
    journal_keywords = [
        "journal", "transactions", "review", "letters", "magazine",
        "nature", "science", "cell", "lancet", "nejm", "jama",
    ]
    if any(kw in venue for kw in journal_keywords):
        return "article"

    # arXiv preprints
    if source == "arxiv" or "arxiv" in venue:
        return "misc"

    # PubMed typically means journal article
    if source == "pubmed":
        return "article"

    # Default to misc for unknown sources
    return "misc"


def generate_bibtex_entry(paper: PaperSummary) -> BibTeXEntry:
    """Generate a BibTeX entry from a paper summary.

    Args:
        paper: Paper summary from Critical Reading Agent

    Returns:
        BibTeXEntry object
    """
    entry_type = detect_entry_type(paper)
    cite_key = generate_cite_key(paper)

    fields = {}

    # Title
    title = paper.get("title", "")
    if title:
        fields["title"] = title

    # Authors
    authors = paper.get("authors", [])
    if authors:
        # BibTeX author format: "Last1, First1 and Last2, First2"
        normalized = [normalize_author_name(a) for a in authors]
        fields["author"] = " and ".join(normalized)

    # Year
    year = paper.get("year", 0)
    if year:
        fields["year"] = str(year)

    # Abstract
    abstract = paper.get("abstract", "")
    if abstract:
        fields["abstract"] = abstract

    # Venue-specific fields
    venue = paper.get("venue", "")
    if entry_type == "article":
        fields["journal"] = venue if venue else "Unknown Journal"
    elif entry_type == "inproceedings":
        fields["booktitle"] = venue if venue else "Unknown Conference"

    # Paper ID handling
    paper_id = paper.get("paper_id", "")
    source = paper.get("source", "")

    # DOI
    if paper_id.startswith("10."):
        fields["doi"] = paper_id
    elif "/" in paper_id and "." in paper_id:
        # Might be a DOI
        fields["doi"] = paper_id

    # arXiv ID
    if source == "arxiv" or paper_id.startswith("arxiv:"):
        arxiv_id = paper_id.replace("arxiv:", "")
        fields["eprint"] = arxiv_id
        fields["archiveprefix"] = "arXiv"
        # Extract primary category if available
        if "." in arxiv_id:
            category = arxiv_id.split(".")[0]
            fields["primaryclass"] = category

    # PubMed ID
    if source == "pubmed" or paper_id.startswith("pmid:"):
        pmid = paper_id.replace("pmid:", "")
        fields["pmid"] = pmid

    # Semantic Scholar ID
    if source == "semantic_scholar" or paper_id.startswith("s2:"):
        s2_id = paper_id.replace("s2:", "")
        fields["note"] = f"Semantic Scholar ID: {s2_id}"

    # Keywords from findings (optional, useful for organization)
    findings = paper.get("key_findings", [])
    if findings:
        # Extract key terms as keywords
        keywords = []
        for finding in findings[:3]:
            words = finding.split()[:5]  # First 5 words of each finding
            keywords.extend([w.lower() for w in words if len(w) > 4])
        if keywords:
            fields["keywords"] = ", ".join(set(keywords[:10]))

    return BibTeXEntry(
        entry_type=entry_type,
        cite_key=cite_key,
        fields=fields,
    )


class BibTeXExporter:
    """Export literature review papers to BibTeX format."""

    def __init__(self, papers: List[PaperSummary]):
        """Initialize with papers to export.

        Args:
            papers: List of paper summaries
        """
        self.papers = papers
        self.entries: List[BibTeXEntry] = []
        self._used_keys: set = set()

    def _ensure_unique_key(self, key: str) -> str:
        """Ensure citation key is unique by adding suffix if needed."""
        if key not in self._used_keys:
            self._used_keys.add(key)
            return key

        # Add suffix a, b, c, etc.
        suffix = ord("a")
        while f"{key}{chr(suffix)}" in self._used_keys:
            suffix += 1
        unique_key = f"{key}{chr(suffix)}"
        self._used_keys.add(unique_key)
        return unique_key

    def generate(self) -> str:
        """Generate BibTeX content for all papers.

        Returns:
            Complete BibTeX file content
        """
        self.entries = []
        self._used_keys = set()

        for paper in self.papers:
            entry = generate_bibtex_entry(paper)
            entry.cite_key = self._ensure_unique_key(entry.cite_key)
            self.entries.append(entry)

        # Generate header comment
        lines = [
            "% BibTeX bibliography generated by LitScribe",
            f"% Contains {len(self.entries)} entries",
            "%",
            "% Citation keys follow the format: AuthorYearFirstword",
            "% e.g., smith2023deep for 'Deep Learning for NLP' by Smith (2023)",
            "",
        ]

        # Add entries
        for entry in self.entries:
            lines.append(entry.to_bibtex())
            lines.append("")

        return "\n".join(lines)

    def save(self, filepath: Path) -> Path:
        """Save BibTeX to file.

        Args:
            filepath: Path to save the .bib file

        Returns:
            Path to the saved file
        """
        content = self.generate()
        filepath = Path(filepath)

        # Ensure .bib extension
        if filepath.suffix != ".bib":
            filepath = filepath.with_suffix(".bib")

        filepath.write_text(content, encoding="utf-8")
        return filepath

    def get_cite_keys(self) -> List[str]:
        """Get list of citation keys for use in documents.

        Returns:
            List of citation keys
        """
        if not self.entries:
            self.generate()
        return [entry.cite_key for entry in self.entries]


def generate_bibtex(
    state: LitScribeState,
    output_path: Optional[Path] = None,
) -> str:
    """Generate BibTeX from a LitScribe state.

    Convenience function for quick export.

    Args:
        state: LitScribe workflow state
        output_path: Optional path to save the .bib file

    Returns:
        BibTeX content string
    """
    papers = state.get("analyzed_papers", [])
    exporter = BibTeXExporter(papers)
    content = exporter.generate()

    if output_path:
        exporter.save(output_path)

    return content


# Export
__all__ = [
    "BibTeXExporter",
    "BibTeXEntry",
    "generate_bibtex",
    "generate_bibtex_entry",
    "generate_cite_key",
    "escape_bibtex",
]
