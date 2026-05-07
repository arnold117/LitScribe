from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ToolCallRequest
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)


class EvolutionMiddleware(AgentMiddleware):

    def __init__(self, memory_manager):
        self.memory = memory_manager
        self._domain: str = "General"

    def set_domain(self, domain: str):
        self._domain = domain

    def wrap_model_call(self, request: ModelRequest, handler):
        try:
            skill_context = self.memory.evolver.inject_skills(self._domain, "planning")
            if skill_context and request.system_message:
                new_content = request.system_message.content + "\n\n" + skill_context
                request = request.override(system_message=SystemMessage(content=new_content))
        except Exception as e:
            logger.debug(f"Skill injection skipped: {e}")

        return handler(request)

    async def awrap_model_call(self, request: ModelRequest, handler):
        try:
            skill_context = self.memory.evolver.inject_skills(self._domain, "planning")
            if skill_context and request.system_message:
                new_content = request.system_message.content + "\n\n" + skill_context
                request = request.override(system_message=SystemMessage(content=new_content))
        except Exception as e:
            logger.debug(f"Skill injection skipped: {e}")

        return await handler(request)

    def after_agent(self, state, runtime):
        return None

    async def aafter_agent(self, state, runtime):
        return None
