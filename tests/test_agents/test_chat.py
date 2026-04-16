"""Tests for the ChatAgent."""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from litscribe.agents.chat import ChatAgent
from litscribe.config import Config
from litscribe.evolution.memory_manager import MemoryManager
from litscribe.llm.router import LLMRouter


@pytest_asyncio.fixture
async def chat_agent(tmp_path):
    cfg = Config()
    cfg.llm.api_key = "test"
    mm = MemoryManager(
        db_path=tmp_path / "test.db",
        chroma_path=tmp_path / "vectors",
        skills_dir=tmp_path / "skills",
    )
    await mm.initialize()
    router = LLMRouter(cfg)
    agent = ChatAgent(config=cfg, memory=mm, router=router)
    yield agent
    await mm.close()


@pytest.mark.asyncio
async def test_chat_plain_conversation(chat_agent):
    """Non-tool responses are passed through."""
    with patch.object(chat_agent.router, "call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = "Hello! How can I help with your research?"
        response = await chat_agent.send("hi")
        assert "Hello" in response
        assert len(chat_agent.history) == 2  # user + assistant


@pytest.mark.asyncio
async def test_chat_search_memory(chat_agent):
    """search_memory tool is dispatched correctly."""
    tool_response = '{"tool": "search_memory", "query": "transformers"}'
    with patch.object(chat_agent.router, "call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = tool_response
        response = await chat_agent.send("what do you know about transformers?")
        assert "No relevant memories" in response or "findings" in response.lower()


@pytest.mark.asyncio
async def test_chat_list_skills_empty(chat_agent):
    """list_skills returns empty message when no skills exist."""
    tool_response = '{"tool": "list_skills"}'
    with patch.object(chat_agent.router, "call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = tool_response
        response = await chat_agent.send("what skills have you learned?")
        assert "No skills" in response


@pytest.mark.asyncio
async def test_chat_export_no_review(chat_agent):
    """export without prior review returns helpful message."""
    tool_response = '{"tool": "export", "format": "bibtex"}'
    with patch.object(chat_agent.router, "call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = tool_response
        response = await chat_agent.send("export to bibtex")
        assert "No review" in response


@pytest.mark.asyncio
async def test_chat_stages_message_during_pipeline(chat_agent):
    """Messages sent during pipeline run are staged, not lost."""
    chat_agent._pipeline_running = True
    response = await chat_agent.send("add year filter")
    assert "staged" in response.lower()
    assert chat_agent._stage_queue.qsize() == 1
    chat_agent._pipeline_running = False


@pytest.mark.asyncio
async def test_parse_tool_call_with_fences(chat_agent):
    """Tool calls wrapped in markdown fences are parsed."""
    raw = '```json\n{"tool": "list_skills"}\n```'
    result = chat_agent._parse_tool_call(raw)
    assert result is not None
    assert result["tool"] == "list_skills"


@pytest.mark.asyncio
async def test_parse_tool_call_plain_text(chat_agent):
    """Plain text is not mistaken for a tool call."""
    result = chat_agent._parse_tool_call("Just a normal response.")
    assert result is None
