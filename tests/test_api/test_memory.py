import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    from litscribe.api.main import create_app

    return create_app()


@pytest.mark.asyncio
async def test_list_skills(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/memory/skills")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_search_episodes(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/memory/episodes", params={"q": "test"})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
