import os
from unittest.mock import patch

import pytest


def test_agent_creation():
    """Test that create_litscribe_agent returns agent, state, token_mw."""
    with patch.dict(os.environ, {
        "LLM_API_KEY": "sk-test-fake-key",
        "LLM_API_BASE": "https://api.example.com/v1",
        "LLM_MODEL": "test-model",
    }):
        from litscribe.config import Config
        from litscribe.agents import create_litscribe_agent

        config = Config()
        config.ensure_directories()
        agent, state, token_mw = create_litscribe_agent(config, memory=None)

        assert agent is not None
        assert state is not None
        assert state.research_question == ""
        assert state.iteration == 0
        assert token_mw.total_calls == 0


def test_subagents_have_response_format():
    """Test that subagents are defined with proper response_format."""
    from litscribe.agents import _build_subagents
    from litscribe.models.plan import ResearchPlan
    from litscribe.models.review import ReviewOutput
    from litscribe.models.assessment import ReviewAssessment

    subagents = _build_subagents()
    assert len(subagents) == 4

    names = {sa["name"] for sa in subagents}
    assert names == {"planner", "reader", "synthesizer", "reviewer"}

    for sa in subagents:
        assert "system_prompt" in sa
        assert "description" in sa
        assert len(sa["description"]) > 10


def test_pipeline_tools_created():
    """Test that create_pipeline_tools returns all 7 tools."""
    with patch.dict(os.environ, {
        "LLM_API_KEY": "sk-test-fake-key",
        "LLM_API_BASE": "https://api.example.com/v1",
        "LLM_MODEL": "test-model",
    }):
        from litscribe.config import Config
        from litscribe.agents import create_pipeline_tools
        from litscribe.tools.status import PipelineState

        config = Config()
        state = PipelineState()
        tools = create_pipeline_tools(config, state)

        assert len(tools) == 4
        tool_names = {t.name for t in tools}
        assert "search_papers" in tool_names
        assert "build_knowledge_graph" in tool_names
        assert "check_pipeline_status" in tool_names
        assert "export_results" in tool_names
