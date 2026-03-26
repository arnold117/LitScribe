import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    from litscribe.api.main import create_app
    return create_app()


@pytest.mark.asyncio
async def test_websocket_connect_and_ping(app):
    from starlette.testclient import TestClient
    client = TestClient(app)
    with client.websocket_connect("/ws/reviews/test123") as ws:
        ws.send_text("ping")
        data = ws.receive_json()
        assert data["type"] == "pong"


@pytest.mark.asyncio
async def test_broadcast_progress():
    from litscribe.api.websocket import broadcast_progress, _connections
    # No connections — should not error
    await broadcast_progress("nonexistent", "planning", "starting")
    assert True  # No error = pass
