"""Memory API endpoints — skills and episodic memory."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("/skills", response_model=list[dict])
async def list_skills() -> list[dict]:
    """List all available procedural skills."""
    # Future: load from SkillEvolver / ProceduralMemory
    return []


@router.get("/episodes", response_model=list[dict])
async def search_episodes(q: str = "") -> list[dict]:
    """Search episodic memory for relevant past sessions."""
    # Future: query EpisodicMemory with q
    return []


@router.put("/skills/{skill_id}")
async def update_skill(skill_id: str, payload: dict[str, Any]) -> dict[str, str]:
    """Update or upsert a skill by ID."""
    # Future: persist via ProceduralMemory
    return {"status": "updated"}
