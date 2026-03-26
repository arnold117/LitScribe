from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# Will be wired to store in production via deps
_store = None


@router.get("")
async def list_sessions(limit: int = 20) -> list[dict]:
    if _store:
        return await _store.list_sessions(limit)
    return []


@router.get("/{session_id}")
async def get_session(session_id: str):
    if not _store:
        raise HTTPException(status_code=503, detail="Store not initialized")
    session = await _store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/{session_id}/versions")
async def get_versions(session_id: str) -> list[dict]:
    if not _store:
        return []
    return await _store.get_versions(session_id)
