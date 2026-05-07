from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ToolCallRequest
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)


class EvolutionMiddleware(AgentMiddleware):

    def __init__(self, memory_manager, pipeline_state=None):
        self.memory = memory_manager
        self._domain: str = "General"
        self._pipeline_state = pipeline_state

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
        self._run_post_task_evaluate()
        return None

    async def aafter_agent(self, state, runtime):
        self._run_post_task_evaluate()
        return None

    def _run_post_task_evaluate(self):
        ps = self._pipeline_state
        if ps is None or ps.assessment is None:
            return
        try:
            from litscribe.evolution.skill_evolver import TaskMetrics
            metrics = TaskMetrics(
                sub_topic_count=len(ps.plan.sub_topics) if ps.plan and ps.plan.sub_topics else 0,
                papers_found=len(ps.papers),
                papers_relevant=len(ps.analyses),
                loop_back_count=max(0, ps.iteration - 1),
                source_count=len({s for p in ps.papers for s in p.sources}),
            )
            self.memory.evolver.post_task_evaluate(
                session_id=f"session-{id(ps)}",
                domain=self._domain,
                score=ps.assessment.score,
                metrics=metrics,
            )
            logger.info(f"Post-task evaluate: score={ps.assessment.score:.2f}, domain={self._domain}")
        except Exception as e:
            logger.warning(f"Post-task evaluate failed: {e}")
