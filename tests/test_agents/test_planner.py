import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_planner_creates_research_plan():
    from litscribe.agents.planner import create_plan
    from litscribe.models.plan import ReviewTier

    mock_llm = AsyncMock(
        return_value='{"sub_topics": [{"name": "chain of thought", "keywords": ["CoT", "reasoning"], "estimated_papers": 15}], "domain": "NLP/AI"}'
    )
    plan = await create_plan(
        question="LLM reasoning capabilities",
        tier=ReviewTier.STANDARD,
        max_papers=40,
        language="en",
        llm_call=mock_llm,
    )
    assert plan.question == "LLM reasoning capabilities"
    assert len(plan.sub_topics) >= 1
    assert plan.domain == "NLP/AI"
    assert plan.tier == ReviewTier.STANDARD


@pytest.mark.asyncio
async def test_planner_handles_llm_returning_string_subtopics():
    from litscribe.agents.planner import create_plan
    from litscribe.models.plan import ReviewTier

    mock_llm = AsyncMock(
        return_value='{"sub_topics": "chain of thought, prompt engineering", "domain": "NLP"}'
    )
    plan = await create_plan(
        question="LLM reasoning",
        tier=ReviewTier.QUICK,
        max_papers=20,
        language="en",
        llm_call=mock_llm,
    )
    assert len(plan.sub_topics) >= 1
    assert all(st.name for st in plan.sub_topics)
