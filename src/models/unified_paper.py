"""Unified paper model that integrates information from multiple sources."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class UnifiedPaper:
    """
    Unified representation of a paper, integrating metadata from multiple sources.

    This model merges information from arXiv, PubMed, Google Scholar, and other sources
    to provide the most complete view of a paper.
    """

    title: str
    authors: List[str]
    abstract: str
    year: int

    # Source identifiers - maps source name to source-specific ID
    sources: Dict[str, str] = field(default_factory=dict)
    # Examples: {"arxiv": "2412.15115", "scholar": "12345", "pubmed": "PMID678"}

    # Core metadata
    venue: str = ""  # Journal or conference name
    citations: int = 0
    pdf_urls: List[str] = field(default_factory=list)  # Multiple PDF sources

    # Quality metrics
    relevance_score: float = 0.0  # Relevance to query (0-1)
    completeness_score: float = 0.0  # Metadata completeness (0-1)

    # Extended fields
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmc_id: Optional[str] = None
    arxiv_id: Optional[str] = None
    scholar_id: Optional[str] = None

    # Specialized metadata
    mesh_terms: List[str] = field(default_factory=list)  # From PubMed
    categories: List[str] = field(default_factory=list)  # From arXiv
    keywords: List[str] = field(default_factory=list)

    # Additional fields
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "sources": self.sources,
            "venue": self.venue,
            "citations": self.citations,
            "pdf_urls": self.pdf_urls,
            "relevance_score": self.relevance_score,
            "completeness_score": self.completeness_score,
            "doi": self.doi,
            "pmid": self.pmid,
            "pmc_id": self.pmc_id,
            "arxiv_id": self.arxiv_id,
            "scholar_id": self.scholar_id,
            "mesh_terms": self.mesh_terms,
            "categories": self.categories,
            "keywords": self.keywords,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "url": self.url,
        }

    @property
    def num_sources(self) -> int:
        """Number of sources this paper was found in."""
        return len(self.sources)

    @property
    def has_full_text(self) -> bool:
        """Check if full text PDF is available."""
        return len(self.pdf_urls) > 0

    @property
    def primary_source(self) -> Optional[str]:
        """Get the primary source (one with most complete metadata)."""
        if not self.sources:
            return None

        # Priority: arXiv > PubMed > Scholar > others
        priority = ["arxiv", "pubmed", "scholar"]
        for source in priority:
            if source in self.sources:
                return source

        return list(self.sources.keys())[0]

    def calculate_completeness_score(self) -> float:
        """
        Calculate metadata completeness score (0-1).

        Considers:
        - Title, authors, abstract presence
        - DOI availability
        - PDF availability
        - Citation count
        - Source-specific metadata
        """
        score = 0.0
        total_fields = 10

        # Core fields (3 points)
        if self.title:
            score += 1
        if self.authors:
            score += 1
        if self.abstract and len(self.abstract) > 50:
            score += 1

        # Identifiers (2 points)
        if self.doi:
            score += 1
        if self.arxiv_id or self.pmid:
            score += 1

        # PDF availability (2 points)
        if self.pdf_urls:
            score += 2

        # Citations (1 point)
        if self.citations and self.citations > 0:
            score += 1

        # Source-specific metadata (2 points)
        if self.mesh_terms or self.categories:
            score += 1
        if self.venue:
            score += 1

        self.completeness_score = score / total_fields
        return self.completeness_score

    def merge_with(self, other: "UnifiedPaper") -> "UnifiedPaper":
        """
        Merge this paper with another, keeping the most complete information.

        Args:
            other: Another UnifiedPaper to merge with

        Returns:
            A new UnifiedPaper with merged information
        """
        merged = UnifiedPaper(
            title=self.title if len(self.title) > len(other.title) else other.title,
            authors=list(set((self.authors or []) + (other.authors or []))),
            abstract=self.abstract if len(self.abstract) > len(other.abstract) else other.abstract,
            year=self.year or other.year,
            sources={**self.sources, **other.sources},
            venue=self.venue or other.venue,
            citations=max(self.citations, other.citations),
            pdf_urls=list(set((self.pdf_urls or []) + (other.pdf_urls or []))),
            relevance_score=max(self.relevance_score, other.relevance_score),
        )

        # Merge identifiers (prefer non-None values)
        merged.doi = self.doi or other.doi
        merged.pmid = self.pmid or other.pmid
        merged.pmc_id = self.pmc_id or other.pmc_id
        merged.arxiv_id = self.arxiv_id or other.arxiv_id
        merged.scholar_id = self.scholar_id or other.scholar_id

        # Merge specialized metadata (guard against None fields from APIs)
        merged.mesh_terms = list(set((self.mesh_terms or []) + (other.mesh_terms or [])))
        merged.categories = list(set((self.categories or []) + (other.categories or [])))
        merged.keywords = list(set((self.keywords or []) + (other.keywords or [])))

        # Other fields
        merged.comment = self.comment or other.comment
        merged.journal_ref = self.journal_ref or other.journal_ref
        merged.url = self.url or other.url

        # Recalculate completeness score
        merged.calculate_completeness_score()

        return merged

    def __repr__(self) -> str:
        """String representation."""
        sources_str = ", ".join(self.sources.keys())
        return (
            f"UnifiedPaper(title='{self.title[:50]}...', "
            f"authors={len(self.authors)}, "
            f"year={self.year}, "
            f"citations={self.citations}, "
            f"sources=[{sources_str}])"
        )
