import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def store(tmp_data_dir):
    from litscribe.store.sqlite import SQLiteStore
    s = SQLiteStore(tmp_data_dir / "test.db")
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_create_and_get_session(store):
    await store.create_session("s1", "LLM reasoning", "standard", "en")
    session = await store.get_session("s1")
    assert session is not None
    assert session["research_question"] == "LLM reasoning"


@pytest.mark.asyncio
async def test_list_sessions(store):
    await store.create_session("s1", "Topic A")
    await store.create_session("s2", "Topic B")
    sessions = await store.list_sessions()
    assert len(sessions) == 2


@pytest.mark.asyncio
async def test_get_session_not_found(store):
    result = await store.get_session("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_save_and_get_versions(store):
    await store.create_session("s1", "Test")
    await store.save_version("s1", 1, "First draft", word_count=100)
    await store.save_version("s1", 2, "Revised draft", word_count=150, instruction="Add more detail")
    versions = await store.get_versions("s1")
    assert len(versions) == 2
    assert versions[0]["version_number"] == 1
    assert versions[1]["version_number"] == 2


@pytest.mark.asyncio
async def test_get_latest_version(store):
    await store.create_session("s1", "Test")
    await store.save_version("s1", 1, "v1 text", word_count=100)
    await store.save_version("s1", 2, "v2 text", word_count=200)
    latest = await store.get_latest_version("s1")
    assert latest is not None
    assert latest["version_number"] == 2
    assert latest["review_text"] == "v2 text"


@pytest.mark.asyncio
async def test_get_latest_version_none(store):
    result = await store.get_latest_version("nonexistent")
    assert result is None
