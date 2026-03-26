import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def memory(tmp_data_dir, tmp_skills_dir):
    from litscribe.evolution.memory_manager import MemoryManager
    mgr = MemoryManager(db_path=tmp_data_dir / "test.db", chroma_path=tmp_data_dir / "vectors", skills_dir=tmp_skills_dir)
    await mgr.initialize()
    yield mgr
    await mgr.close()


@pytest.mark.asyncio
async def test_memory_manager_has_all_layers(memory):
    assert memory.episodic is not None
    assert memory.semantic is not None
    assert memory.procedural is not None
    assert memory.evolver is not None


@pytest.mark.asyncio
async def test_memory_manager_episode_roundtrip(memory):
    await memory.episodic.record(session_id="s1", question="test", outcome_score=0.8, key_events=["event1"])
    results = await memory.episodic.recall("test")
    assert len(results) == 1
