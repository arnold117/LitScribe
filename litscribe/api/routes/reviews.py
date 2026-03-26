"""Reviews CRUD endpoints."""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/reviews", tags=["reviews"])

# In-memory store: review_id -> review dict
_reviews: dict[str, dict[str, Any]] = {}


class ReviewRequest(BaseModel):
    question: str
    max_papers: int = 40
    tier: str = "standard"
    language: str = "en"
    graphrag: bool = True
    model: str | None = None


class ReviewResponse(BaseModel):
    review_id: str
    status: str
    question: str
    tier: str


@router.post("", status_code=202, response_model=ReviewResponse)
async def create_review(req: ReviewRequest) -> ReviewResponse:
    """Start a new literature review job."""
    review_id = str(uuid.uuid4())
    record: dict[str, Any] = {
        "review_id": review_id,
        "status": "started",
        "question": req.question,
        "tier": req.tier,
        "max_papers": req.max_papers,
        "language": req.language,
        "result": None,
    }
    _reviews[review_id] = record
    return ReviewResponse(
        review_id=review_id,
        status="started",
        question=req.question,
        tier=req.tier,
    )


@router.get("", response_model=list[ReviewResponse])
async def list_reviews() -> list[ReviewResponse]:
    """List all reviews."""
    return [
        ReviewResponse(
            review_id=r["review_id"],
            status=r["status"],
            question=r["question"],
            tier=r["tier"],
        )
        for r in _reviews.values()
    ]


@router.get("/{review_id}", response_model=ReviewResponse)
async def get_review(review_id: str) -> ReviewResponse:
    """Get a review by ID."""
    record = _reviews.get(review_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Review not found")
    return ReviewResponse(
        review_id=record["review_id"],
        status=record["status"],
        question=record["question"],
        tier=record["tier"],
    )


@router.delete("/{review_id}", status_code=204)
async def delete_review(review_id: str) -> None:
    """Delete a review by ID."""
    if review_id not in _reviews:
        raise HTTPException(status_code=404, detail="Review not found")
    del _reviews[review_id]
