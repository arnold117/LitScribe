"""Google Scholar article model."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ScholarArticle:
    """Represents a Google Scholar article."""

    scholar_id: str
    title: str
    authors: List[str]
    year: int
    venue: str  # Journal or conference name
    citations: int
    url: str
    abstract: Optional[str] = None  # Scholar doesn't always have abstracts
    pdf_url: Optional[str] = None
    source: Optional[str] = None  # "PDF", "HTML", "arXiv", "PubMed", etc.
    related_articles_url: Optional[str] = None
    cited_by_url: Optional[str] = None
    versions: List[str] = field(default_factory=list)  # Different versions of the paper

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "scholar_id": self.scholar_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "citations": self.citations,
            "url": self.url,
            "abstract": self.abstract,
            "pdf_url": self.pdf_url,
            "source": self.source,
            "related_articles_url": self.related_articles_url,
            "cited_by_url": self.cited_by_url,
            "versions": self.versions,
        }

    @property
    def has_pdf(self) -> bool:
        """Check if PDF is available."""
        return self.pdf_url is not None
