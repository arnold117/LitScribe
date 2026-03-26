import pytest
from pydantic import ValidationError


def test_paper_model():
    from litscribe.models.paper import Paper
    p = Paper(
        paper_id="arxiv:2412.15115", title="Test Paper", authors=["Alice", "Bob"],
        abstract="An abstract.", year=2024, sources={"arxiv": "2412.15115"},
    )
    assert p.paper_id == "arxiv:2412.15115"
    assert p.relevance_score == 0.0
    assert p.pdf_urls == []


def test_paper_requires_title():
    from litscribe.models.paper import Paper
    with pytest.raises(ValidationError):
        Paper(paper_id="x", authors=[], abstract="", year=2024, sources={})


def test_research_plan_model():
    from litscribe.models.plan import ResearchPlan, SubTopic, ReviewTier
    plan = ResearchPlan(
        question="LLM reasoning",
        sub_topics=[SubTopic(name="chain of thought", keywords=["CoT", "reasoning"])],
        domain="NLP/AI", tier=ReviewTier.STANDARD,
    )
    assert len(plan.sub_topics) == 1
    assert plan.tier == ReviewTier.STANDARD


def test_review_output_model():
    from litscribe.models.review import ReviewOutput, Citation, Theme
    review = ReviewOutput(
        text="A review.",
        citations=[Citation(paper_id="x", claim="claim", section="intro")],
        themes=[Theme(name="theme1", description="desc", paper_ids=["x"])],
        word_count=100,
    )
    assert review.word_count == 100


def test_review_assessment_model():
    from litscribe.models.assessment import ReviewAssessment
    assessment = ReviewAssessment(
        passed=False, score=0.45, feedback="Needs more coverage",
        refined_queries=["LLM reasoning chains"],
    )
    assert not assessment.passed
    assert assessment.refined_queries is not None


def test_paper_analysis_model():
    from litscribe.models.analysis import PaperAnalysis
    analysis = PaperAnalysis(
        paper_id="arxiv:2412.15115", key_findings=["Finding 1"],
        methodology="Experimental", strengths=["Strong design"],
        limitations=["Small sample"], relevance_score=0.85,
    )
    assert analysis.relevance_score == 0.85


def test_models_are_serializable():
    from litscribe.models.paper import Paper
    p = Paper(paper_id="x", title="Test", authors=["A"], abstract="abs", year=2024, sources={"arxiv": "123"})
    data = p.model_dump()
    p2 = Paper.model_validate(data)
    assert p == p2
