import pytest

def test_resolve_model_default():
    from litscribe.llm.router import LLMRouter
    from litscribe.config import Config
    config = Config()
    router = LLMRouter(config)
    assert router.resolve_model("query_expansion") == "openai/qwen-turbo"
    assert router.resolve_model("synthesis") == "openai/qwen-max"
    assert router.resolve_model("unknown_task") == "openai/qwen-plus"

def test_resolve_model_custom_config(tmp_path):
    from litscribe.llm.router import LLMRouter
    from litscribe.config import Config
    yaml_content = 'llm:\n  default_model: "openai/qwen-turbo"\n  task_models:\n    synthesis: "openai/deepseek-r1"\n'
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    config = Config(config_path=config_file)
    router = LLMRouter(config)
    assert router.resolve_model("synthesis") == "openai/deepseek-r1"
    assert router.resolve_model("unknown") == "openai/qwen-turbo"
