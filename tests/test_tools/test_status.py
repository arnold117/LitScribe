from litscribe.tools.status import PipelineState, check_status, determine_recommendation
from litscribe.models.assessment import ReviewAssessment


def test_empty_state_recommends_planner():
    state = PipelineState(research_question="test")
    rec = determine_recommendation(state)
    assert "planner" in rec.lower()


def test_after_plan_recommends_discover():
    from litscribe.models.plan import ResearchPlan, SubTopic
    state = PipelineState(research_question="test")
    state.plan = ResearchPlan(
        question="test", sub_topics=[SubTopic(name="t1", keywords=["k1"], estimated_papers=5)],
        domain="CS", tier="standard", max_papers=40, language="en", target_words=5000,
    )
    rec = determine_recommendation(state)
    assert "discover" in rec.lower()


def test_few_papers_triggers_circuit_breaker():
    from litscribe.models.plan import ResearchPlan, SubTopic
    from litscribe.models.paper import Paper
    from litscribe.models.analysis import PaperAnalysis

    state = PipelineState(research_question="test")
    state.plan = ResearchPlan(
        question="test", sub_topics=[], domain="CS",
        tier="standard", max_papers=40, language="en", target_words=5000,
    )
    state.papers = [Paper(paper_id=f"p{i}", title=f"Paper {i}", authors=[], abstract="", year=2024, sources={"test": f"p{i}"}) for i in range(3)]
    state.analyses = [
        PaperAnalysis(paper_id=f"p{i}", key_findings=[], methodology="", strengths=[], limitations=[], relevance_score=0.5, themes=[])
        for i in range(3)
    ]
    rec = determine_recommendation(state)
    assert "broader" in rec.lower() or "discover" in rec.lower()


def test_good_score_completes():
    from litscribe.models.plan import ResearchPlan, SubTopic
    from litscribe.models.paper import Paper
    from litscribe.models.analysis import PaperAnalysis
    from litscribe.models.review import ReviewOutput

    state = PipelineState(research_question="test")
    state.plan = ResearchPlan(
        question="test", sub_topics=[], domain="CS",
        tier="standard", max_papers=40, language="en", target_words=5000,
    )
    state.papers = [Paper(paper_id=f"p{i}", title=f"P{i}", authors=[], abstract="", year=2024, sources={"test": f"p{i}"}) for i in range(10)]
    state.analyses = [
        PaperAnalysis(paper_id=f"p{i}", key_findings=["f"], methodology="m", strengths=["s"], limitations=["l"], relevance_score=0.8, themes=[])
        for i in range(10)
    ]
    state.synthesis = ReviewOutput(text="review", citations=[], themes=[], word_count=3000, language="en")
    state.assessment = ReviewAssessment(passed=True, score=0.8, feedback="good", refined_queries=[], coverage_score=0.9, weak_claims=[])

    rec = determine_recommendation(state)
    assert "COMPLETE" in rec


def test_low_score_loops_back():
    from litscribe.models.plan import ResearchPlan
    from litscribe.models.paper import Paper
    from litscribe.models.analysis import PaperAnalysis
    from litscribe.models.review import ReviewOutput

    state = PipelineState(research_question="test")
    state.plan = ResearchPlan(
        question="test", sub_topics=[], domain="CS",
        tier="standard", max_papers=40, language="en", target_words=5000,
    )
    state.papers = [Paper(paper_id=f"p{i}", title=f"P{i}", authors=[], abstract="", year=2024, sources={"test": f"p{i}"}) for i in range(10)]
    state.analyses = [
        PaperAnalysis(paper_id=f"p{i}", key_findings=["f"], methodology="m", strengths=[], limitations=[], relevance_score=0.5, themes=[])
        for i in range(10)
    ]
    state.synthesis = ReviewOutput(text="bad review", citations=[], themes=[], word_count=500, language="en")
    state.assessment = ReviewAssessment(
        passed=False, score=0.4, feedback="poor", refined_queries=["more papers"], coverage_score=0.3, weak_claims=[],
    )
    state.iteration = 1

    rec = determine_recommendation(state)
    assert "LOOP BACK" in rec


def test_check_status_returns_dict():
    state = PipelineState(research_question="test question")
    result = check_status(state)
    assert isinstance(result, dict)
    assert result["research_question"] == "test question"
    assert "recommendation" in result
