from pydantic import BaseModel


class ParsedDoc(BaseModel):
    paper_id: str
    markdown: str = ""
    sections: list[dict] = []
    word_count: int = 0


class PaperAnalysis(BaseModel):
    paper_id: str
    key_findings: list[str]
    methodology: str = ""
    strengths: list[str] = []
    limitations: list[str] = []
    relevance_score: float = 0.0
    themes: list[str] = []
