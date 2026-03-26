from pydantic import BaseModel


class Citation(BaseModel):
    paper_id: str
    claim: str
    section: str = ""


class Theme(BaseModel):
    name: str
    description: str
    paper_ids: list[str] = []


class ReviewOutput(BaseModel):
    text: str
    citations: list[Citation] = []
    themes: list[Theme] = []
    word_count: int = 0
    language: str = "en"
