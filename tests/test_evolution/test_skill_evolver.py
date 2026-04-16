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


# ---- TaskMetrics ----

def test_metrics_non_trivial_many_subtopics():
    from litscribe.evolution.skill_evolver import TaskMetrics
    m = TaskMetrics(sub_topic_count=4, papers_found=20, papers_relevant=15)
    assert m.is_non_trivial is True


def test_metrics_non_trivial_low_signal_to_noise():
    from litscribe.evolution.skill_evolver import TaskMetrics
    m = TaskMetrics(papers_found=100, papers_relevant=8)
    assert m.signal_to_noise == 0.08
    assert m.is_non_trivial is True


def test_metrics_non_trivial_many_loopbacks():
    from litscribe.evolution.skill_evolver import TaskMetrics
    m = TaskMetrics(loop_back_count=2)
    assert m.is_non_trivial is True


def test_metrics_non_trivial_many_sources():
    from litscribe.evolution.skill_evolver import TaskMetrics
    m = TaskMetrics(source_count=3)
    assert m.is_non_trivial is True


def test_metrics_trivial():
    from litscribe.evolution.skill_evolver import TaskMetrics
    m = TaskMetrics(sub_topic_count=1, papers_found=10, papers_relevant=8, loop_back_count=0, source_count=1)
    assert m.is_non_trivial is False


def test_metrics_signal_to_noise_zero_papers():
    from litscribe.evolution.skill_evolver import TaskMetrics
    m = TaskMetrics(papers_found=0, papers_relevant=0)
    assert m.signal_to_noise == 1.0  # no papers = not noisy


# ---- should_extract_skill ----

@pytest.mark.asyncio
async def test_should_extract_high_score_non_trivial(evolver):
    from litscribe.evolution.skill_evolver import TaskMetrics
    m = TaskMetrics(sub_topic_count=5, papers_found=50, papers_relevant=40)
    assert evolver.should_extract_skill(score=0.85, metrics=m) is True


@pytest.mark.asyncio
async def test_should_not_extract_low_score(evolver):
    from litscribe.evolution.skill_evolver import TaskMetrics
    m = TaskMetrics(sub_topic_count=5)
    assert evolver.should_extract_skill(score=0.4, metrics=m) is False


@pytest.mark.asyncio
async def test_should_not_extract_trivial_task(evolver):
    from litscribe.evolution.skill_evolver import TaskMetrics
    m = TaskMetrics(sub_topic_count=1, papers_found=10, papers_relevant=9, source_count=1)
    assert evolver.should_extract_skill(score=0.9, metrics=m) is False


# ---- record_failure ----

@pytest.mark.asyncio
async def test_record_failure(evolver):
    await evolver.record_failure(session_id="s1", question="test question", score=0.3, feedback="Coverage too low")
    results = await evolver.episodic.recall("test question")
    assert len(results) == 1
    assert "FAILURE" in results[0]["summary"]


# ---- inject_skills ----

@pytest.mark.asyncio
async def test_inject_skills_adds_to_instructions(evolver):
    evolver.procedural.save_skill(name="NLP Search", domain="NLP", trigger="NLP queries", strategy="Search arXiv cs.CL first", learned_adjustments=[])
    instructions = evolver.inject_skills(domain="NLP", task_type="discovery")
    assert "NLP Search" in instructions
    assert "arXiv" in instructions


# ---- post_task_evaluate ----

@pytest.mark.asyncio
async def test_post_task_evaluate_extracts_skill(evolver):
    from litscribe.evolution.skill_evolver import TaskMetrics
    m = TaskMetrics(sub_topic_count=4, papers_found=30, papers_relevant=20, source_count=3)
    await evolver.post_task_evaluate(
        session_id="sess-abc123",
        question="NLP transformers survey",
        score=0.85,
        metrics=m,
        domain="NLP",
        trace_summary="Searched arXiv and S2, filtered by year, synthesized themes.",
    )
    skills = evolver.procedural.list_skills()
    assert len(skills) == 1
    assert "NLP" in skills[0]["domain"]


@pytest.mark.asyncio
async def test_post_task_evaluate_records_failure(evolver):
    from litscribe.evolution.skill_evolver import TaskMetrics
    m = TaskMetrics(sub_topic_count=2)
    await evolver.post_task_evaluate(
        session_id="sess-fail",
        question="obscure topic",
        score=0.3,
        metrics=m,
        domain="Other",
        trace_summary="Could not find enough papers.",
    )
    episodes = await evolver.episodic.recall("obscure topic")
    assert any("FAILURE" in e["summary"] for e in episodes)
