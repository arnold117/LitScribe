import os
import pytest
from pathlib import Path


def test_config_loads_defaults():
    from litscribe.config import Config

    cfg = Config()
    assert cfg.llm.default_model == "openai/qwen-plus"
    assert cfg.llm.api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert cfg.data_dir.is_absolute()


def test_config_loads_env(monkeypatch):
    from litscribe.config import Config

    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test123")
    monkeypatch.setenv("LITSCRIBE_DEFAULT_MODEL", "openai/qwen-max")
    cfg = Config()
    assert cfg.llm.api_key == "sk-test123"
    assert cfg.llm.default_model == "openai/qwen-max"


def test_config_task_models_default():
    from litscribe.config import Config

    cfg = Config()
    assert "synthesis" in cfg.llm.task_models
    assert "query_expansion" in cfg.llm.task_models


def test_config_yaml_override(tmp_path):
    from litscribe.config import Config

    yaml_content = """
llm:
  default_model: "openai/qwen-turbo"
  task_models:
    synthesis: "openai/deepseek-r1"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    cfg = Config(config_path=config_file)
    assert cfg.llm.default_model == "openai/qwen-turbo"
    assert cfg.llm.task_models["synthesis"] == "openai/deepseek-r1"
    # Non-overridden task models still have defaults
    assert "query_expansion" in cfg.llm.task_models


def test_config_data_directories(tmp_path, monkeypatch):
    from litscribe.config import Config

    monkeypatch.setenv("LITSCRIBE_DATA_DIR", str(tmp_path))
    cfg = Config()
    assert cfg.data_dir == tmp_path
    assert cfg.db_path == tmp_path / "litscribe.db"
    assert cfg.chroma_path == tmp_path / "vectors"
    assert cfg.skills_dir == Path.home() / ".litscribe" / "skills"
