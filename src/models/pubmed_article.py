"""PubMed article model."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class PubMedArticle:
    """Represents a PubMed article."""

    pmid: str
    title: str
    authors: List[str]
    abstract: str
    journal: str
    publication_date: datetime
    doi: Optional[str] = None
    pmc_id: Optional[str] = None  # PubMed Central ID (if full text available)
    mesh_terms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    url: Optional[str] = None
    citations: Optional[int] = None
    affiliation: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "pmid": self.pmid,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "journal": self.journal,
            "publication_date": self.publication_date.isoformat()
            if self.publication_date
            else None,
            "doi": self.doi,
            "pmc_id": self.pmc_id,
            "mesh_terms": self.mesh_terms,
            "keywords": self.keywords,
            "url": self.url,
            "citations": self.citations,
            "affiliation": self.affiliation,
        }

    @property
    def full_text_available(self) -> bool:
        """Check if full text is available via PubMed Central."""
        return self.pmc_id is not None
