from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()

# Active connections per review_id
_connections: dict[str, list[WebSocket]] = defaultdict(list)


async def broadcast_progress(review_id: str, step: str, detail: str = ""):
    """Called by pipeline's on_progress callback to push updates."""
    message = json.dumps({"review_id": review_id, "step": step, "detail": detail})
    dead = []
    for ws in _connections.get(review_id, []):
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _connections[review_id].remove(ws)


@router.websocket("/ws/reviews/{review_id}")
async def review_progress(websocket: WebSocket, review_id: str):
    await websocket.accept()
    _connections[review_id].append(websocket)
    try:
        while True:
            # Keep connection alive, client can send pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        _connections[review_id].remove(websocket)
