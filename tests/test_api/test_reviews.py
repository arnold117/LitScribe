import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    from litscribe.api.main import create_app

    return create_app()


@pytest.mark.asyncio
async def test_create_review(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/reviews",
            json={"question": "LLM reasoning", "max_papers": 10, "tier": "quick"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "review_id" in data
        assert data["status"] == "started"


@pytest.mark.asyncio
async def test_get_review_not_found(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/reviews/nonexistent")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_reviews(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/reviews")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
