"""Tests for Agno agent wrappers and the Agno-backed pipeline factory."""
from __future__ import annotations


def test_create_planner_agent():
    from litscribe.agents.agno_agents import create_planner_agent

    agent = create_planner_agent(
        model="openai/qwen-plus", api_key="test", api_base="http://test"
    )
    assert agent.name == "Planner"


def test_create_reader_agent():
    from litscribe.agents.agno_agents import create_reader_agent

    agent = create_reader_agent(api_key="test", api_base="http://test")
    assert agent.name == "CriticalReader"


def test_create_synthesizer_agent():
    from litscribe.agents.agno_agents import create_synthesizer_agent

    agent = create_synthesizer_agent(api_key="test", api_base="http://test")
    assert agent.name == "Synthesizer"


def test_create_reviewer_agent():
    from litscribe.agents.agno_agents import create_reviewer_agent

    agent = create_reviewer_agent(api_key="test", api_base="http://test")
    assert agent.name == "SelfReviewer"


def test_create_all_agents():
    from litscribe.agents.agno_agents import (
        create_planner_agent,
        create_reader_agent,
        create_synthesizer_agent,
        create_reviewer_agent,
    )

    agents = [
        create_planner_agent(api_key="test", api_base="http://test"),
        create_reader_agent(api_key="test", api_base="http://test"),
        create_synthesizer_agent(api_key="test", api_base="http://test"),
        create_reviewer_agent(api_key="test", api_base="http://test"),
    ]
    names = [a.name for a in agents]
    assert len(set(names)) == 4  # all unique


def test_agents_have_instructions():
    from litscribe.agents.agno_agents import (
        create_planner_agent,
        create_reader_agent,
        create_synthesizer_agent,
        create_reviewer_agent,
    )

    for factory in (
        create_planner_agent,
        create_reader_agent,
        create_synthesizer_agent,
        create_reviewer_agent,
    ):
        agent = factory(api_key="test", api_base="http://test")
        assert agent.instructions, f"{agent.name} should have instructions"


def test_agents_have_litellm_model():
    from litscribe.agents.agno_agents import create_planner_agent
    from agno.models.litellm import LiteLLM

    agent = create_planner_agent(
        model="openai/qwen-plus", api_key="test", api_base="http://test"
    )
    assert isinstance(agent.model, LiteLLM)
    assert agent.model.id == "openai/qwen-plus"


def test_agno_pipeline_factory():
    from litscribe.agents.agno_pipeline import create_agno_pipeline
    from litscribe.config import Config

    config = Config()
    pipeline = create_agno_pipeline(config)
    assert pipeline is not None


def test_agno_pipeline_has_all_fns():
    from litscribe.agents.agno_pipeline import create_agno_pipeline
    from litscribe.agents.pipeline import LitScribePipeline
    from litscribe.config import Config

    config = Config()
    pipeline = create_agno_pipeline(config)
    assert isinstance(pipeline, LitScribePipeline)
    assert callable(pipeline.plan_fn)
    assert callable(pipeline.discover_fn)
    assert callable(pipeline.read_fn)
    assert callable(pipeline.synthesize_fn)
    assert callable(pipeline.review_fn)
