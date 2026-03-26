import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def sqlite_store(tmp_data_dir):
    from litscribe.store.sqlite import SQLiteStore
    store = SQLiteStore(tmp_data_dir / "test.db")
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_initialize_creates_tables(sqlite_store):
    tables = await sqlite_store.list_tables()
    assert "papers" in tables
    assert "parsed_docs" in tables
    assert "episodes" in tables
    assert "sessions" in tables
    assert "skills_meta" in tables


@pytest.mark.asyncio
async def test_save_and_get_paper(sqlite_store):
    from litscribe.models.paper import Paper
    paper = Paper(
        paper_id="arxiv:2412.15115", title="Test Paper", authors=["Alice"],
        abstract="Abstract text", year=2024, sources={"arxiv": "2412.15115"},
    )
    await sqlite_store.save_papers([paper])
    result = await sqlite_store.get_paper("arxiv:2412.15115")
    assert result is not None
    assert result.title == "Test Paper"


@pytest.mark.asyncio
async def test_get_paper_not_found(sqlite_store):
    result = await sqlite_store.get_paper("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_save_and_get_parsed_doc(sqlite_store):
    from litscribe.models.analysis import ParsedDoc
    doc = ParsedDoc(paper_id="x", markdown="# Title\nContent", word_count=2)
    await sqlite_store.save_parsed("x", doc)
    result = await sqlite_store.get_parsed("x")
    assert result is not None
    assert result.markdown == "# Title\nContent"


@pytest.mark.asyncio
async def test_episode_save_and_fts5_recall(sqlite_store):
    await sqlite_store.save_episode(
        session_id="sess1", question="LLM reasoning capabilities",
        outcome_score=0.85, summary="Searched arxiv and S2 for reasoning papers. Found 35 papers.",
    )
    await sqlite_store.save_episode(
        session_id="sess2", question="Protein folding methods",
        outcome_score=0.7, summary="Searched PubMed for protein structure prediction. Found 20 papers.",
    )
    results = await sqlite_store.recall("reasoning LLM", limit=5)
    assert len(results) >= 1
    assert results[0]["session_id"] == "sess1"


@pytest.mark.asyncio
async def test_episode_recall_empty(sqlite_store):
    results = await sqlite_store.recall("nonexistent topic")
    assert results == []
