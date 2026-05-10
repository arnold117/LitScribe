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


def test_agent_has_tools():
    """Test that agent has the expected tools."""
    with patch.dict(os.environ, {
        "LLM_API_KEY": "sk-test-fake-key",
        "LLM_API_BASE": "https://api.example.com/v1",
        "LLM_MODEL": "test-model",
    }):
        from litscribe.config import Config
        from litscribe.agents import create_litscribe_agent

        config = Config()
        config.ensure_directories()
        agent, state, token_mw = create_litscribe_agent(config)

        assert agent is not None
        assert state.research_question == ""


def test_pipeline_tools_created():
    """Test that create_pipeline_tools returns the 3 tools."""
    with patch.dict(os.environ, {
        "LLM_API_KEY": "sk-test-fake-key",
        "LLM_API_BASE": "https://api.example.com/v1",
        "LLM_MODEL": "test-model",
    }):
        from litscribe.config import Config
        from litscribe.agents import create_pipeline_tools, _build_model
        from litscribe.tools.status import PipelineState

        config = Config()
        state = PipelineState()
        model = _build_model(config)
        tools = create_pipeline_tools(config, state, model, memory=None)

        assert len(tools) == 6
        tool_names = {t.name for t in tools}
        assert "run_review" in tool_names
        assert "search_papers" in tool_names
        assert "refine_review" in tool_names
        assert "analyze_draft" in tool_names
        assert "suggest_review_outline" in tool_names
        assert "export_results" in tool_names
