from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class ModelAdapter:
    """Wraps any LangChain chat model to provide the same call/call_json interface as LLMRouter."""

    def __init__(self, model: BaseChatModel):
        self.model = model

    async def call(self, messages: list[dict], task_type: str | None = None,
                   temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> str:
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            prompt = messages[0].get("content", "")
        else:
            prompt = str(messages)

        result = await self.model.ainvoke(prompt)
        return result.content

    async def call_json(self, messages: list[dict], task_type: str | None = None,
                        max_retries: int = 2, **kwargs) -> dict | list:
        for attempt in range(max_retries + 1):
            raw = await self.call(messages, task_type=task_type)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```\w*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                if attempt == max_retries:
                    raise
                logger.warning(f"JSON parse failed (attempt {attempt + 1}), retrying")
        return {}
