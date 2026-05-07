from pydantic import BaseModel


class ReviewAssessment(BaseModel):
    passed: bool
    score: float
    feedback: str
    refined_queries: list[str] | None = None
    coverage_score: float = 0.0
    weak_claims: list[str] = []
