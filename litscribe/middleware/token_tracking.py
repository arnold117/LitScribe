from __future__ import annotations

import logging

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest

logger = logging.getLogger(__name__)


class TokenTrackingMiddleware(AgentMiddleware):

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    def wrap_model_call(self, request: ModelRequest, handler):
        self.total_calls += 1
        response = handler(request)
        self._extract_usage(response)
        return response

    async def awrap_model_call(self, request: ModelRequest, handler):
        self.total_calls += 1
        response = await handler(request)
        self._extract_usage(response)
        return response

    def _extract_usage(self, response):
        try:
            msg = response if hasattr(response, "usage_metadata") else getattr(response, "message", None)
            if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                usage = msg.usage_metadata
                self.total_prompt_tokens += usage.get("input_tokens", 0)
                self.total_completion_tokens += usage.get("output_tokens", 0)
        except Exception:
            pass

    def summary(self) -> dict:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_calls": self.total_calls,
        }
