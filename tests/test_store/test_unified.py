import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def store(tmp_data_dir):
    from litscribe.store.unified import UnifiedStore
    s = UnifiedStore(db_path=tmp_data_dir / "test.db", chroma_path=tmp_data_dir / "vectors")
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_unified_store_paper_roundtrip(store):
    from litscribe.models.paper import Paper
    paper = Paper(paper_id="test1", title="T", authors=["A"], abstract="abs", year=2024, sources={"arxiv": "123"})
    await store.save_papers([paper])
    result = await store.get_paper("test1")
    assert result is not None
    assert result.title == "T"


@pytest.mark.asyncio
async def test_unified_store_semantic_search(store):
    store.add_embeddings(
        texts=["deep learning for NLP"], metadatas=[{"source": "p1"}],
        ids=["e1"], collection="semantic_memory",
    )
    results = store.semantic_search("NLP neural networks", collection="semantic_memory", n=1)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_unified_store_episode_roundtrip(store):
    await store.save_episode(session_id="s1", question="test query", outcome_score=0.8, summary="found papers")
    results = await store.recall("test query")
    assert len(results) == 1
