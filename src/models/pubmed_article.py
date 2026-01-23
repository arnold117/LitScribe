"""PubMed article model."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PubMedArticle:
    """Represents a PubMed article."""

    pmid: str
    title: str
    authors: List[str]
    abstract: str
    journal: str
    publication_date: Optional[datetime] = None
    doi: Optional[str] = None
    pmc_id: Optional[str] = None  # PubMed Central ID (if full text available)
    mesh_terms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    url: Optional[str] = None
    citations: Optional[int] = None
    affiliation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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

    @property
    def pubmed_url(self) -> str:
        """Get PubMed URL for this article."""
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def pmc_url(self) -> Optional[str]:
        """Get PubMed Central URL if available."""
        if self.pmc_id:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{self.pmc_id}/"
        return None

    @classmethod
    def from_medline_record(cls, record: Dict[str, Any]) -> "PubMedArticle":
        """Create PubMedArticle from Biopython Medline record."""
        pub_date = None
        if "DP" in record:
            try:
                date_str = record["DP"]
                parts = date_str.split()
                if len(parts) >= 3:
                    pub_date = datetime.strptime(f"{parts[0]} {parts[1]} {parts[2]}", "%Y %b %d")
                elif len(parts) == 2:
                    pub_date = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y %b")
                elif len(parts) == 1:
                    pub_date = datetime.strptime(parts[0], "%Y")
            except (ValueError, IndexError):
                pass

        authors = record.get("AU", [])
        if isinstance(authors, str):
            authors = [authors]

        mesh_terms = record.get("MH", [])
        if isinstance(mesh_terms, str):
            mesh_terms = [mesh_terms]

        keywords = record.get("OT", [])
        if isinstance(keywords, str):
            keywords = [keywords]

        # Extract DOI
        doi = None
        lid = record.get("LID", "")
        if "[doi]" in lid:
            doi = lid.replace(" [doi]", "")
        elif record.get("AID"):
            aids = record.get("AID", [])
            if isinstance(aids, str):
                aids = [aids]
            for aid in aids:
                if "[doi]" in aid:
                    doi = aid.replace(" [doi]", "")
                    break

        return cls(
            pmid=record.get("PMID", ""),
            title=record.get("TI", ""),
            authors=authors,
            abstract=record.get("AB", ""),
            journal=record.get("JT", record.get("TA", "")),
            publication_date=pub_date,
            doi=doi,
            pmc_id=record.get("PMC"),
            mesh_terms=mesh_terms,
            keywords=keywords,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{record.get('PMID', '')}/",
            affiliation=record.get("AD") if isinstance(record.get("AD"), str) else (record.get("AD", [""])[0] if record.get("AD") else None),
        )
