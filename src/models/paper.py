"""Base paper model for arXiv papers."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Paper:
    """Represents an arXiv paper."""

    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    pdf_url: str
    published: datetime
    categories: List[str] = field(default_factory=list)
    citations: Optional[int] = None
    updated: Optional[datetime] = None
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "pdf_url": self.pdf_url,
            "published": self.published.isoformat() if self.published else None,
            "categories": self.categories,
            "citations": self.citations,
            "updated": self.updated.isoformat() if self.updated else None,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "doi": self.doi,
        }

    @classmethod
    def from_arxiv_result(cls, result) -> "Paper":
        """Create Paper instance from arxiv API result."""
        return cls(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title,
            authors=[author.name for author in result.authors],
            abstract=result.summary,
            pdf_url=result.pdf_url,
            published=result.published,
            categories=[cat for cat in result.categories],
            updated=result.updated,
            comment=result.comment,
            journal_ref=result.journal_ref,
            doi=result.doi,
        )
