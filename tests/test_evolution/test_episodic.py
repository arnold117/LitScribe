import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def episodic(tmp_data_dir):
    from litscribe.evolution.episodic import EpisodicMemory
    from litscribe.store.sqlite import SQLiteStore
    store = SQLiteStore(tmp_data_dir / "test.db")
    await store.initialize()
    mem = EpisodicMemory(store)
    yield mem
    await store.close()


@pytest.mark.asyncio
async def test_record_and_recall(episodic):
    await episodic.record(
        session_id="s1", question="LLM reasoning capabilities",
        outcome_score=0.85, key_events=["Searched arXiv", "Found 35 papers", "Synthesis score 0.85"],
    )
    results = await episodic.recall("LLM reasoning")
    assert len(results) >= 1
    assert results[0]["session_id"] == "s1"
    assert "arXiv" in results[0]["summary"]


@pytest.mark.asyncio
async def test_recall_returns_most_relevant(episodic):
    await episodic.record(session_id="s1", question="LLM reasoning", outcome_score=0.8, key_events=["searched arxiv for reasoning"])
    await episodic.record(session_id="s2", question="protein folding", outcome_score=0.9, key_events=["searched pubmed for protein"])
    results = await episodic.recall("protein biology")
    assert results[0]["session_id"] == "s2"
