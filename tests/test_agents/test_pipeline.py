import pytest
import pytest_asyncio
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_pipeline_runs_all_steps():
    from litscribe.agents.pipeline import LitScribePipeline

    call_log = []

    async def mock_plan(question, **kwargs):
        call_log.append("plan")
        from litscribe.models.plan import ResearchPlan, SubTopic, ReviewTier

        return ResearchPlan(
            question=question,
            sub_topics=[SubTopic(name="t1", keywords=["k"])],
            domain="NLP",
            tier=ReviewTier.QUICK,
            max_papers=10,
            language="en",
        )

    async def mock_discover(plan, **kwargs):
        call_log.append("discover")
        from litscribe.models.paper import Paper

        return [
            Paper(
                paper_id="p1",
                title="P1",
                authors=["A"],
                abstract="abs",
                year=2024,
                sources={"arxiv": "1"},
            )
        ]

    async def mock_read(papers, **kwargs):
        call_log.append("read")
        from litscribe.models.analysis import PaperAnalysis

        return [PaperAnalysis(paper_id="p1", key_findings=["f1"], relevance_score=0.9)]

    async def mock_synthesize(analyses, **kwargs):
        call_log.append("synthesize")
        from litscribe.models.review import ReviewOutput

        return ReviewOutput(text="Review text", word_count=100)

    async def mock_review(output, **kwargs):
        call_log.append("review")
        from litscribe.models.assessment import ReviewAssessment

        return ReviewAssessment(passed=True, score=0.85, feedback="Good")

    pipeline = LitScribePipeline(
        plan_fn=mock_plan,
        discover_fn=mock_discover,
        read_fn=mock_read,
        synthesize_fn=mock_synthesize,
        review_fn=mock_review,
    )
    result = await pipeline.run("LLM reasoning", max_papers=10)
    assert result.text == "Review text"
    assert call_log == ["plan", "discover", "read", "synthesize", "review"]


@pytest.mark.asyncio
async def test_pipeline_loops_on_failed_review():
    from litscribe.agents.pipeline import LitScribePipeline
    from litscribe.models.plan import ResearchPlan, SubTopic, ReviewTier
    from litscribe.models.paper import Paper
    from litscribe.models.analysis import PaperAnalysis
    from litscribe.models.review import ReviewOutput
    from litscribe.models.assessment import ReviewAssessment

    review_call_count = 0

    async def mock_plan(question, **kwargs):
        return ResearchPlan(
            question=question,
            sub_topics=[SubTopic(name="t1", keywords=["k"])],
            domain="NLP",
            tier=ReviewTier.QUICK,
            max_papers=10,
            language="en",
        )

    async def mock_discover(plan, **kwargs):
        return [
            Paper(
                paper_id="p1",
                title="P1",
                authors=["A"],
                abstract="abs",
                year=2024,
                sources={},
            )
        ]

    async def mock_read(papers, **kwargs):
        return [PaperAnalysis(paper_id="p1", key_findings=["f1"], relevance_score=0.9)]

    async def mock_synthesize(analyses, **kwargs):
        return ReviewOutput(text="Review", word_count=100)

    async def mock_review(output, **kwargs):
        nonlocal review_call_count
        review_call_count += 1
        if review_call_count < 2:
            return ReviewAssessment(
                passed=False,
                score=0.4,
                feedback="Needs work",
                refined_queries=["more papers"],
            )
        return ReviewAssessment(passed=True, score=0.8, feedback="OK")

    pipeline = LitScribePipeline(
        plan_fn=mock_plan,
        discover_fn=mock_discover,
        read_fn=mock_read,
        synthesize_fn=mock_synthesize,
        review_fn=mock_review,
        max_iterations=3,
    )
    result = await pipeline.run("test", max_papers=10)
    assert result.text == "Review"
    assert review_call_count == 2


@pytest.mark.asyncio
async def test_pipeline_with_memory_evolves_skills(tmp_path):
    """Pipeline with MemoryManager absorbs analyses and evaluates the task."""
    from litscribe.agents.pipeline import LitScribePipeline
    from litscribe.evolution.memory_manager import MemoryManager
    from litscribe.models.plan import ResearchPlan, SubTopic, ReviewTier
    from litscribe.models.paper import Paper
    from litscribe.models.analysis import PaperAnalysis
    from litscribe.models.review import ReviewOutput
    from litscribe.models.assessment import ReviewAssessment

    mm = MemoryManager(
        db_path=tmp_path / "test.db",
        chroma_path=tmp_path / "vectors",
        skills_dir=tmp_path / "skills",
    )
    await mm.initialize()

    async def mock_plan(question, **kwargs):
        return ResearchPlan(
            question=question,
            sub_topics=[
                SubTopic(name="t1", keywords=["a"]),
                SubTopic(name="t2", keywords=["b"]),
                SubTopic(name="t3", keywords=["c"]),
            ],
            domain="AI",
            tier=ReviewTier.STANDARD,
            max_papers=30,
            language="en",
        )

    async def mock_discover(plan, **kwargs):
        return [
            Paper(paper_id=f"p{i}", title=f"P{i}", authors=["A"],
                  abstract="abs", year=2024, sources={"arxiv": str(i), "s2": str(i)})
            for i in range(20)
        ]

    async def mock_read(papers, **kwargs):
        return [
            PaperAnalysis(paper_id=p.paper_id, key_findings=[f"finding for {p.paper_id}"],
                          relevance_score=0.8 if int(p.paper_id[1:]) < 12 else 0.3)
            for p in papers
        ]

    async def mock_synthesize(analyses, **kwargs):
        return ReviewOutput(text="Comprehensive review.", word_count=500)

    async def mock_review(output, **kwargs):
        return ReviewAssessment(passed=True, score=0.85, feedback="Solid review")

    pipeline = LitScribePipeline(
        plan_fn=mock_plan,
        discover_fn=mock_discover,
        read_fn=mock_read,
        synthesize_fn=mock_synthesize,
        review_fn=mock_review,
        memory=mm,
    )
    result = await pipeline.run("AI safety survey")

    assert result.text == "Comprehensive review."

    # Semantic memory should have absorbed findings
    hits = mm.semantic.search("finding", n=5)
    assert len(hits) > 0

    # Evolver should have extracted a skill (score=0.85, 3 sub_topics = non-trivial)
    skills = mm.procedural.list_skills()
    assert len(skills) == 1
    assert "AI" in skills[0]["domain"]

    await mm.close()
