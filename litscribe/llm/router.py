from __future__ import annotations
import re
import json
import logging
from typing import Any
import litellm
from litscribe.config import Config
from litscribe.llm.tracker import TokenTracker

logger = logging.getLogger(__name__)
REASONING_PATTERNS = re.compile(r"reasoner|deepseek-r1|o1-|o3-|o4-", re.IGNORECASE)

class LLMRouter:
    def __init__(self, config: Config):
        self.config = config
        self.tracker = TokenTracker()

    def resolve_model(self, task_type: str | None = None) -> str:
        if task_type and task_type in self.config.llm.task_models:
            return self._ensure_litellm_prefix(self.config.llm.task_models[task_type])
        return self._ensure_litellm_prefix(self.config.llm.default_model)

    def _ensure_litellm_prefix(self, model: str) -> str:
        if "/" in model:
            return model
        api_base = self.config.llm.api_base.lower()
        if "deepseek" in api_base:
            return f"deepseek/{model}"
        if "dashscope" in api_base or "aliyun" in api_base:
            return f"openai/{model}"
        if "openai.com" in api_base:
            return model
        return f"openai/{model}"

    def _is_reasoning_model(self, model: str) -> bool:
        return bool(REASONING_PATTERNS.search(model))

    async def call(self, messages: list[dict], task_type: str | None = None, model_override: str | None = None, temperature: float = 0.7, max_tokens: int = 4096, agent_name: str = "unknown") -> str:
        model = model_override or self.resolve_model(task_type)
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if self.config.llm.api_key:
            kwargs["api_key"] = self.config.llm.api_key
        if self.config.llm.api_base:
            kwargs["api_base"] = self.config.llm.api_base
        if not self._is_reasoning_model(model):
            kwargs["temperature"] = temperature
        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content or ""
        if self._is_reasoning_model(model):
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if response.usage:
            self.tracker.record(agent_name, model, {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens})
        return content

    async def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        task_type: str | None = None,
        agent_name: str = "unknown",
    ) -> dict[str, Any] | None:
        """Call the LLM with OpenAI-style tool definitions.

        Returns a dict ``{"name": ..., "arguments": {...}}`` if the model
        invoked a tool, or ``None`` if it responded with plain text.
        The plain text content is appended to ``messages`` as an assistant
        message so the caller can read it.
        """
        model = self.resolve_model(task_type)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "max_tokens": 4096,
        }
        if self.config.llm.api_key:
            kwargs["api_key"] = self.config.llm.api_key
        if self.config.llm.api_base:
            kwargs["api_base"] = self.config.llm.api_base
        if not self._is_reasoning_model(model):
            kwargs["temperature"] = 0.7

        try:
            response = await litellm.acompletion(**kwargs)
        except Exception:
            # Model may not support tools — caller should fall back
            logger.debug("call_with_tools failed, model may not support tools", exc_info=True)
            return None

        msg = response.choices[0].message
        if response.usage:
            self.tracker.record(agent_name, model, {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            })

        # Check for tool call in response
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tc = msg.tool_calls[0]
            fn = tc.function
            try:
                args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else fn.arguments
            except json.JSONDecodeError:
                args = {}
            return {"name": fn.name, "arguments": args}

        return None

    async def call_json(self, messages: list[dict], task_type: str | None = None, agent_name: str = "unknown", max_retries: int = 2) -> dict:
        for attempt in range(max_retries + 1):
            temp = 0.7 if attempt == 0 else max(0.3, 0.7 - attempt * 0.2)
            raw = await self.call(messages, task_type=task_type, temperature=temp, agent_name=agent_name)
            try:
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```\w*\n?", "", raw)
                    raw = re.sub(r"\n?```$", "", raw)
                return json.loads(raw)
            except json.JSONDecodeError:
                if attempt == max_retries:
                    raise
                logger.warning(f"JSON parse failed (attempt {attempt + 1}), retrying")
