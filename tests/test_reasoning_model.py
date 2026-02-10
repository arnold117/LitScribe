#!/usr/bin/env python
"""Tests for DeepSeek thinking mode / reasoning model routing.

Run with: pytest tests/test_reasoning_model.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# === Test 1: _is_reasoning_model detection ===

def test_is_reasoning_model_deepseek_reasoner():
    from agents.tools import _is_reasoning_model
    assert _is_reasoning_model("deepseek/deepseek-reasoner") is True


def test_is_reasoning_model_deepseek_r1():
    from agents.tools import _is_reasoning_model
    assert _is_reasoning_model("deepseek/deepseek-r1") is True


def test_is_reasoning_model_o1():
    from agents.tools import _is_reasoning_model
    assert _is_reasoning_model("openai/o1-preview") is True


def test_is_reasoning_model_o3():
    from agents.tools import _is_reasoning_model
    assert _is_reasoning_model("openai/o3-mini") is True


def test_is_not_reasoning_model_chat():
    from agents.tools import _is_reasoning_model
    assert _is_reasoning_model("deepseek/deepseek-chat") is False


def test_is_not_reasoning_model_claude():
    from agents.tools import _is_reasoning_model
    assert _is_reasoning_model("anthropic/claude-sonnet-4-5-20250929") is False


def test_is_not_reasoning_model_haiku():
    from agents.tools import _is_reasoning_model
    assert _is_reasoning_model("anthropic/claude-haiku-4-5-20251001") is False


# === Test 2: _resolve_model routing ===

def test_resolve_model_explicit_wins():
    """Explicit model always takes priority."""
    from agents.tools import _resolve_model
    with patch("utils.config.Config") as mock_cfg:
        mock_cfg.LITELLM_MODEL = "deepseek/deepseek-chat"
        mock_cfg.LITELLM_REASONING_MODEL = "deepseek/deepseek-reasoner"
        result = _resolve_model("anthropic/claude-sonnet-4-5-20250929", task_type="synthesis")
        assert result == "anthropic/claude-sonnet-4-5-20250929"


def test_resolve_model_synthesis_uses_reasoning():
    """Synthesis task should use reasoning model when configured."""
    from agents.tools import _resolve_model
    with patch("utils.config.Config") as mock_cfg:
        mock_cfg.LITELLM_MODEL = "deepseek/deepseek-chat"
        mock_cfg.LITELLM_REASONING_MODEL = "deepseek/deepseek-reasoner"
        result = _resolve_model(None, task_type="synthesis")
        assert result == "deepseek/deepseek-reasoner"


def test_resolve_model_self_review_uses_reasoning():
    from agents.tools import _resolve_model
    with patch("utils.config.Config") as mock_cfg:
        mock_cfg.LITELLM_MODEL = "deepseek/deepseek-chat"
        mock_cfg.LITELLM_REASONING_MODEL = "deepseek/deepseek-reasoner"
        result = _resolve_model(None, task_type="self_review")
        assert result == "deepseek/deepseek-reasoner"


def test_resolve_model_refinement_uses_reasoning():
    from agents.tools import _resolve_model
    with patch("utils.config.Config") as mock_cfg:
        mock_cfg.LITELLM_MODEL = "deepseek/deepseek-chat"
        mock_cfg.LITELLM_REASONING_MODEL = "deepseek/deepseek-reasoner"
        result = _resolve_model(None, task_type="refinement")
        assert result == "deepseek/deepseek-reasoner"


def test_resolve_model_discovery_uses_default():
    """Non-heavy tasks should use default model."""
    from agents.tools import _resolve_model
    with patch("utils.config.Config") as mock_cfg:
        mock_cfg.LITELLM_MODEL = "deepseek/deepseek-chat"
        mock_cfg.LITELLM_REASONING_MODEL = "deepseek/deepseek-reasoner"
        result = _resolve_model(None, task_type="discovery")
        assert result == "deepseek/deepseek-chat"


def test_resolve_model_no_reasoning_configured():
    """When no reasoning model configured, always use default."""
    from agents.tools import _resolve_model
    with patch("utils.config.Config") as mock_cfg:
        mock_cfg.LITELLM_MODEL = "deepseek/deepseek-chat"
        mock_cfg.LITELLM_REASONING_MODEL = ""
        result = _resolve_model(None, task_type="synthesis")
        assert result == "deepseek/deepseek-chat"


def test_resolve_model_no_task_type():
    """No task_type should use default model."""
    from agents.tools import _resolve_model
    with patch("utils.config.Config") as mock_cfg:
        mock_cfg.LITELLM_MODEL = "deepseek/deepseek-chat"
        mock_cfg.LITELLM_REASONING_MODEL = "deepseek/deepseek-reasoner"
        result = _resolve_model(None, task_type=None)
        assert result == "deepseek/deepseek-chat"


# === Test 3: REASONING_TASK_TYPES ===

def test_reasoning_task_types_set():
    from agents.tools import REASONING_TASK_TYPES
    assert "synthesis" in REASONING_TASK_TYPES
    assert "self_review" in REASONING_TASK_TYPES
    assert "refinement" in REASONING_TASK_TYPES
    assert "discovery" not in REASONING_TASK_TYPES
    assert "planning" not in REASONING_TASK_TYPES
    assert "critical_reading" not in REASONING_TASK_TYPES


# === Test 4: extract_json robustness ===

def test_extract_json_clean():
    from agents.tools import extract_json
    result = extract_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_extract_json_array():
    from agents.tools import extract_json
    result = extract_json('["a", "b", "c"]')
    assert result == ["a", "b", "c"]


def test_extract_json_with_code_fence():
    from agents.tools import extract_json
    text = '```json\n{"key": "value"}\n```'
    result = extract_json(text)
    assert result == {"key": "value"}


def test_extract_json_with_thinking_prefix():
    """Reasoning models may prepend thinking text before JSON."""
    from agents.tools import extract_json
    text = """Let me analyze this carefully.

The research question covers several aspects including...

Based on my analysis:

{"complexity_score": 3, "sub_topics": [{"name": "Topic A"}]}"""
    result = extract_json(text)
    assert result["complexity_score"] == 3


def test_extract_json_with_thinking_and_trailing():
    from agents.tools import extract_json
    text = """I'll think about this step by step.

{"action_type": "add_content", "target_section": "methods", "details": "Add LoRA discussion"}

This should work well for the review."""
    result = extract_json(text)
    assert result["action_type"] == "add_content"


def test_extract_json_array_with_prefix():
    from agents.tools import extract_json
    text = """Here are the expanded queries:

["CRISPR gene editing", "genome engineering tools", "CRISPR-Cas9 applications"]

These queries cover the main aspects."""
    result = extract_json(text)
    assert len(result) == 3
    assert "CRISPR gene editing" in result


def test_extract_json_no_json_raises():
    from agents.tools import extract_json
    with pytest.raises(json.JSONDecodeError):
        extract_json("This is just plain text with no JSON at all.")


def test_extract_json_nested():
    from agents.tools import extract_json
    text = '{"outer": {"inner": [1, 2, 3]}, "key": "value"}'
    result = extract_json(text)
    assert result["outer"]["inner"] == [1, 2, 3]


# === Test 5: call_llm signature has task_type ===

def test_call_llm_has_task_type_param():
    import inspect
    from agents.tools import call_llm
    sig = inspect.signature(call_llm)
    params = list(sig.parameters.keys())
    assert "task_type" in params


def test_call_llm_with_system_has_task_type_param():
    import inspect
    from agents.tools import call_llm_with_system
    sig = inspect.signature(call_llm_with_system)
    params = list(sig.parameters.keys())
    assert "task_type" in params


# === Test 6: Config has LITELLM_REASONING_MODEL ===

def test_config_has_reasoning_model():
    from utils.config import Config
    assert hasattr(Config, "LITELLM_REASONING_MODEL")


# === Entrypoint ===

async def main():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=str(Path(__file__).parent.parent),
    )
    return result.returncode


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
