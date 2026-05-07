from unittest.mock import MagicMock, patch

from langchain_core.messages import SystemMessage

from litscribe.middleware.evolution import EvolutionMiddleware


def test_skill_injection_into_system_prompt():
    memory = MagicMock()
    memory.evolver.inject_skills.return_value = "## Relevant Skill\nUse X strategy"

    mw = EvolutionMiddleware(memory)
    mw.set_domain("Biology")

    request = MagicMock()
    request.system_message = SystemMessage(content="You are a helpful assistant.")
    request.override = MagicMock(return_value=request)

    handler = MagicMock(return_value="response")

    mw.wrap_model_call(request, handler)

    request.override.assert_called_once()
    call_kwargs = request.override.call_args
    new_sys = call_kwargs.kwargs.get("system_message") or call_kwargs[1].get("system_message")
    assert "Relevant Skill" in new_sys.content
    handler.assert_called_once()


def test_no_injection_when_no_skills():
    memory = MagicMock()
    memory.evolver.inject_skills.return_value = ""

    mw = EvolutionMiddleware(memory)
    request = MagicMock()
    request.system_message = SystemMessage(content="base prompt")

    handler = MagicMock(return_value="response")

    mw.wrap_model_call(request, handler)

    request.override.assert_not_called()
    handler.assert_called_once_with(request)


def test_graceful_on_memory_error():
    memory = MagicMock()
    memory.evolver.inject_skills.side_effect = RuntimeError("db error")

    mw = EvolutionMiddleware(memory)
    request = MagicMock()
    request.system_message = SystemMessage(content="base")

    handler = MagicMock(return_value="response")

    result = mw.wrap_model_call(request, handler)

    handler.assert_called_once()
    assert result == "response"
