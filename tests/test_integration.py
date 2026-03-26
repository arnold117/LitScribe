"""End-to-end integration test — pipeline + memory + skill evolution."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from litscribe.models.plan import ReviewTier


@pytest.mark.asyncio
async def test_full_pipeline_with_mock_llm(tmp_data_dir, tmp_skills_dir):
    from litscribe.config import Config
    from litscribe.store.unified import UnifiedStore
    from litscribe.evolution.memory_manager import MemoryManager
    from litscribe.agents.pipeline import LitScribePipeline
    from litscribe.agents.planner import create_plan
    from litscribe.models.paper import Paper
    from litscribe.models.analysis import PaperAnalysis
    from litscribe.models.review import ReviewOutput
    from litscribe.models.assessment import ReviewAssessment

    store = UnifiedStore(
        db_path=tmp_data_dir / "test.db",
        chroma_path=tmp_data_dir / "vectors",
    )
    await store.initialize()

    memory = MemoryManager(
        db_path=tmp_data_dir / "test.db",
        chroma_path=tmp_data_dir / "vectors",
        skills_dir=tmp_skills_dir,
    )
    await memory.initialize()

    mock_llm = AsyncMock(
        return_value='{"sub_topics": [{"name": "CoT", "keywords": ["chain of thought"], "estimated_papers": 10}], "domain": "NLP/AI"}'
    )

    async def plan_fn(question, **kw):
        return await create_plan(question, ReviewTier.QUICK, 10, "en", mock_llm)

    async def discover_fn(plan, **kw):
        return [
            Paper(
                paper_id="p1",
                title="CoT Paper",
                authors=["A"],
                abstract="About CoT",
                year=2024,
                sources={"arxiv": "1"},
            )
        ]

    async def read_fn(papers, **kw):
        return [
            PaperAnalysis(
                paper_id="p1",
                key_findings=["CoT improves reasoning"],
                relevance_score=0.9,
            )
        ]

    async def synthesize_fn(analyses, **kw):
        return ReviewOutput(text="CoT review text", word_count=500)

    async def review_fn(output, **kw):
        return ReviewAssessment(passed=True, score=0.85, feedback="Good coverage")

    pipeline = LitScribePipeline(
        plan_fn=plan_fn,
        discover_fn=discover_fn,
        read_fn=read_fn,
        synthesize_fn=synthesize_fn,
        review_fn=review_fn,
    )
    result = await pipeline.run("LLM reasoning", max_papers=10, tier=ReviewTier.QUICK)
    assert result.text == "CoT review text"

    # Skill evolution trigger check
    assert memory.evolver.should_extract_skill(score=0.85, complexity=5)

    # Episodic memory
    await memory.episodic.record(
        session_id="test",
        question="LLM reasoning",
        outcome_score=0.85,
        key_events=["Found 1 paper", "Score 0.85"],
    )
    recalled = await memory.episodic.recall("LLM reasoning")
    assert len(recalled) == 1

    # Semantic memory
    memory.semantic.absorb(
        [
            PaperAnalysis(
                paper_id="p1",
                key_findings=["CoT improves reasoning"],
                relevance_score=0.9,
            )
        ]
    )
    sem_results = memory.semantic.search("chain of thought reasoning")
    assert len(sem_results) >= 1

    await store.close()
    await memory.close()
