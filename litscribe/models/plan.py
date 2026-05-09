from enum import Enum
from pydantic import BaseModel


class ReviewTier(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class SubTopic(BaseModel):
    name: str
    keywords: list[str] = []
    estimated_papers: int = 10


class ResearchPlan(BaseModel):
    question: str
    sub_topics: list[SubTopic]
    domain: str
    tier: ReviewTier
    max_papers: int = 40
    language: str = "en"
    target_words: int = 0
