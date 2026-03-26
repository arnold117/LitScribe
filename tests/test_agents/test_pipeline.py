import pytest
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
