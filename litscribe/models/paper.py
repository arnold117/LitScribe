from pydantic import BaseModel


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    sources: dict[str, str]
    venue: str = ""
    citations: int = 0
    doi: str = ""
    pdf_urls: list[str] = []
    relevance_score: float = 0.0
    completeness_score: float = 0.0


class SearchMeta(BaseModel):
    total_found: int = 0
    sources_queried: list[str] = []
    queries_used: list[str] = []
    snowball_rounds: int = 0
