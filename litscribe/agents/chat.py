"""Chat agent — conversational entry point that can invoke the review pipeline.

The chat agent is the user-facing interface. It decides when to trigger
a full literature review, answer questions from memory, or just converse.
Runs as a REPL locally or can be driven over WebSocket.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from litscribe.config import Config
from litscribe.evolution.memory_manager import MemoryManager
from litscribe.llm.router import LLMRouter
from litscribe.models.review import ReviewOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are LitScribe, an AI research assistant that helps users with academic \
literature reviews. You have access to several tools:

1. **review** — Run a full literature review pipeline. Use when the user asks \
for a survey, literature review, or research overview on a topic. \
Call with: {"tool": "review", "question": "<research question>", \
"tier": "quick|standard|comprehensive", "max_papers": 40, "language": "en"}

2. **search_memory** — Search past research sessions and knowledge. Use when \
the user asks about prior work or wants to recall past findings. \
Call with: {"tool": "search_memory", "query": "<search terms>"}

3. **list_skills** — Show learned research strategies. \
Call with: {"tool": "list_skills"}

4. **export** — Export a review to a specific format. \
Call with: {"tool": "export", "format": "bibtex|apa|mla"}

If the user is just chatting, asking a question, or you don't need a tool, \
respond directly without calling any tool.

When you need a tool, respond with ONLY a JSON block: {"tool": "...", ...}
When you don't, respond in natural language.
"""


class ChatAgent:
    """Stateful chat agent with tool dispatch."""

    def __init__(
        self,
        config: Config,
        memory: MemoryManager,
        router: LLMRouter,
    ) -> None:
        self.config = config
        self.memory = memory
        self.router = router
        self.history: list[dict[str, str]] = []
        self._last_review: ReviewOutput | None = None
        self._stage_queue: asyncio.Queue[str] = asyncio.Queue()
        self._pipeline_running = False

    async def send(self, user_message: str) -> str:
        """Process a user message and return the assistant response."""
        # If pipeline is running, stage the message
        if self._pipeline_running:
            await self._stage_queue.put(user_message)
            return "[message staged — will be processed after current review completes]"

        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        raw = await self.router.call(
            messages, task_type="planning", agent_name="chat"
        )

        # Check if the response is a tool call
        tool_call = self._parse_tool_call(raw)
        if tool_call:
            result = await self._dispatch(tool_call)
            self.history.append({"role": "assistant", "content": result})
            return result

        self.history.append({"role": "assistant", "content": raw})
        return raw

    def _parse_tool_call(self, text: str) -> dict | None:
        """Try to extract a JSON tool call from the LLM response."""
        text = text.strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "tool" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    async def _dispatch(self, call: dict[str, Any]) -> str:
        """Route a parsed tool call to the appropriate handler."""
        tool = call.get("tool", "")
        if tool == "review":
            return await self._handle_review(call)
        elif tool == "search_memory":
            return await self._handle_search_memory(call)
        elif tool == "list_skills":
            return await self._handle_list_skills()
        elif tool == "export":
            return self._handle_export(call)
        return f"Unknown tool: {tool}"

    async def _handle_review(self, call: dict[str, Any]) -> str:
        """Run the full review pipeline."""
        from litscribe.agents.agno_pipeline import create_agno_pipeline

        question = call.get("question", "")
        if not question:
            return "Please provide a research question."

        self._pipeline_running = True
        try:
            pipeline = create_agno_pipeline(self.config, memory=self.memory)
            self._last_review = await pipeline.run(
                question,
                tier=call.get("tier", "standard"),
                max_papers=call.get("max_papers", 40),
                language=call.get("language", "en"),
            )
            result = (
                f"Review complete ({self._last_review.word_count} words).\n\n"
                f"{self._last_review.text[:500]}..."
                if len(self._last_review.text) > 500
                else f"Review complete ({self._last_review.word_count} words).\n\n{self._last_review.text}"
            )
        except Exception as e:
            logger.exception("Pipeline failed")
            result = f"Review failed: {e}"
        finally:
            self._pipeline_running = False

        # Process staged messages
        await self._drain_stage_queue()

        return result

    async def _handle_search_memory(self, call: dict[str, Any]) -> str:
        """Search episodic and semantic memory."""
        query = call.get("query", "")
        if not query:
            return "Please provide a search query."

        episodes = await self.memory.episodic.recall(query, limit=3)
        semantic = self.memory.semantic.search(query, n=3)

        parts: list[str] = []
        if episodes:
            parts.append("**Past sessions:**")
            for ep in episodes:
                parts.append(f"- {ep.get('question', '?')} (score: {ep.get('outcome_score', '?')})")
        if semantic:
            parts.append("**Related findings:**")
            for hit in semantic:
                parts.append(f"- {hit.get('document', '?')[:120]}")
        if not parts:
            return "No relevant memories found."
        return "\n".join(parts)

    async def _handle_list_skills(self) -> str:
        """List learned procedural skills."""
        skills = self.memory.procedural.list_skills()
        if not skills:
            return "No skills learned yet."
        lines = []
        for s in skills:
            lines.append(
                f"- **{s['name']}** (v{s['version']}, domain: {s['domain']}): "
                f"{s['trigger']}"
            )
        return "\n".join(lines)

    def _handle_export(self, call: dict[str, Any]) -> str:
        """Export the last review."""
        if self._last_review is None:
            return "No review to export. Run a review first."
        fmt = call.get("format", "bibtex")
        return f"Export to {fmt} is not yet wired. Last review has {self._last_review.word_count} words."

    async def _drain_stage_queue(self) -> None:
        """Process any messages that were staged during pipeline execution."""
        while not self._stage_queue.empty():
            staged = await self._stage_queue.get()
            logger.info("Processing staged message: %s", staged[:80])
            # Feed staged messages back through the agent
            response = await self.send(staged)
            # In REPL mode the response gets printed by the caller;
            # here we just append to history so context is maintained
            logger.info("Staged response: %s", response[:80])
