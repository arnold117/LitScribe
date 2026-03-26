import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def evolver(tmp_data_dir, tmp_skills_dir):
    from litscribe.evolution.skill_evolver import SkillEvolver
    from litscribe.evolution.episodic import EpisodicMemory
    from litscribe.evolution.procedural import ProceduralMemory
    from litscribe.store.sqlite import SQLiteStore
    from litscribe.store.vectors import VectorStore
    sqlite = SQLiteStore(tmp_data_dir / "test.db")
    await sqlite.initialize()
    vectors = VectorStore(tmp_data_dir / "vectors")
    episodic = EpisodicMemory(sqlite)
    procedural = ProceduralMemory(tmp_skills_dir, vectors)
    evolver = SkillEvolver(episodic=episodic, procedural=procedural)
    yield evolver
    await sqlite.close()


@pytest.mark.asyncio
async def test_should_extract_skill_high_score_complex(evolver):
    assert evolver.should_extract_skill(score=0.85, complexity=7) is True


@pytest.mark.asyncio
async def test_should_not_extract_skill_low_score(evolver):
    assert evolver.should_extract_skill(score=0.4, complexity=7) is False


@pytest.mark.asyncio
async def test_should_not_extract_skill_low_complexity(evolver):
    assert evolver.should_extract_skill(score=0.9, complexity=2) is False


@pytest.mark.asyncio
async def test_record_failure(evolver):
    await evolver.record_failure(session_id="s1", question="test question", score=0.3, feedback="Coverage too low")
    results = await evolver.episodic.recall("test question")
    assert len(results) == 1
    assert "FAILURE" in results[0]["summary"]


@pytest.mark.asyncio
async def test_inject_skills_adds_to_instructions(evolver):
    evolver.procedural.save_skill(name="NLP Search", domain="NLP", trigger="NLP queries", strategy="Search arXiv cs.CL first", learned_adjustments=[])
    instructions = evolver.inject_skills(domain="NLP", task_type="discovery")
    assert "NLP Search" in instructions
    assert "arXiv" in instructions
