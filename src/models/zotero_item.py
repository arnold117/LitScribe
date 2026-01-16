"""Zotero item model."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ZoteroItem:
    """Represents a Zotero library item."""

    key: str  # Zotero item key
    item_type: str  # article, book, conferencePaper, etc.
    title: str
    creators: List[Dict[str, str]] = field(default_factory=list)
    # creators: [{"firstName": "John", "lastName": "Doe", "creatorType": "author"}]

    abstract: str = ""
    date: str = ""
    url: str = ""
    pdf_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    collections: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    # Extended metadata
    publication_title: str = ""  # Journal/conference name
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: Optional[str] = None
    issn: Optional[str] = None
    arxiv_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "key": self.key,
            "item_type": self.item_type,
            "title": self.title,
            "creators": self.creators,
            "abstract": self.abstract,
            "date": self.date,
            "url": self.url,
            "pdf_path": self.pdf_path,
            "tags": self.tags,
            "collections": self.collections,
            "extra": self.extra,
            "publication_title": self.publication_title,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "doi": self.doi,
            "issn": self.issn,
            "arxiv_id": self.arxiv_id,
        }

    @property
    def authors(self) -> List[str]:
        """Get list of author names."""
        authors = []
        for creator in self.creators:
            if creator.get("creatorType") == "author":
                first = creator.get("firstName", "")
                last = creator.get("lastName", "")
                name = f"{first} {last}".strip()
                if name:
                    authors.append(name)
        return authors

    @property
    def has_pdf(self) -> bool:
        """Check if PDF attachment exists."""
        return self.pdf_path is not None and self.pdf_path != ""

    def __repr__(self) -> str:
        authors_str = ", ".join(self.authors[:2])
        if len(self.authors) > 2:
            authors_str += " et al."
        return f"ZoteroItem(title='{self.title[:40]}...', authors=[{authors_str}])"


@dataclass
class Fulltext:
    """Represents fulltext content from Zotero."""

    content: str
    pdf_path: str
    indexed_chars: int

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "pdf_path": self.pdf_path,
            "indexed_chars": self.indexed_chars,
        }
