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
"""

# OpenAI-style tool definitions for function calling
_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "review",
            "description": "Run a full literature review pipeline on a research topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The research question to review."},
                    "tier": {"type": "string", "enum": ["quick", "standard", "comprehensive"], "default": "standard"},
                    "max_papers": {"type": "integer", "default": 40},
                    "language": {"type": "string", "default": "en"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search past research sessions and accumulated knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "Show learned research strategies and skills.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export",
            "description": "Export the last review to a citation format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {"type": "string", "enum": ["bibtex", "apa", "mla", "ieee", "chicago"]},
                },
                "required": ["format"],
            },
        },
    },
]


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

        # Strategy 1: try function calling (most reliable)
        tool_call = await self._try_function_calling(messages)

        # Strategy 2: fall back to text-based tool dispatch
        if tool_call is None:
            raw = await self.router.call(
                messages, task_type="planning", agent_name="chat"
            )
            tool_call = self._parse_tool_call(raw)
            if tool_call is None:
                # Plain conversation — no tool needed
                self.history.append({"role": "assistant", "content": raw})
                return raw

        result = await self._dispatch(tool_call)
        self.history.append({"role": "assistant", "content": result})
        return result

        self.history.append({"role": "assistant", "content": raw})
        return raw

    async def _try_function_calling(self, messages: list[dict]) -> dict | None:
        """Attempt OpenAI-style function calling via the LLM router.

        Returns a normalised tool-call dict ``{"tool": ..., ...}`` or ``None``
        if the model doesn't support tools or chose not to call one.
        """
        result = await self.router.call_with_tools(
            messages, tools=_TOOL_DEFINITIONS, task_type="planning", agent_name="chat"
        )
        if result is None:
            return None
        # Normalise: function calling returns {"name": ..., "arguments": {...}}
        # but _dispatch expects {"tool": ..., **args}
        return {"tool": result["name"], **result.get("arguments", {})}

    def _parse_tool_call(self, text: str) -> dict | None:
        """Fuzzy-extract a JSON tool call from mixed LLM output.

        Handles: pure JSON, markdown-fenced JSON, and JSON embedded in text.
        """
        text = text.strip()

        # Try 1: pure JSON (with optional markdown fences)
        cleaned = text
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```\w*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "tool" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        # Try 2: extract first JSON object from mixed text
        match = re.search(r"\{[^{}]*\"tool\"[^{}]*\}", text)
        if match:
            try:
                data = json.loads(match.group())
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
        staged_responses = await self._drain_stage_queue()
        if staged_responses:
            result += "\n\n--- Staged messages processed ---\n" + "\n".join(staged_responses)

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
        """Export the last review's cited papers."""
        if self._last_review is None:
            return "No review to export. Run a review first."

        # Collect Paper objects from citations
        papers = self._collect_cited_papers()
        if not papers:
            return "No cited papers to export."

        fmt = call.get("format", "bibtex")

        if fmt == "bibtex":
            from litscribe.exporters.bibtex import papers_to_bibtex
            return papers_to_bibtex(papers)

        from litscribe.exporters.citation_formatter import CitationStyle, format_citations
        style_map = {
            "apa": CitationStyle.APA,
            "mla": CitationStyle.MLA,
            "ieee": CitationStyle.IEEE,
            "chicago": CitationStyle.CHICAGO,
            "gbt7714": CitationStyle.GB_T_7714,
        }
        style = style_map.get(fmt, CitationStyle.APA)
        return format_citations(papers, style=style)

    def _collect_cited_papers(self) -> list:
        """Build Paper objects from the last review's citations."""
        if self._last_review is None:
            return []
        from litscribe.models.paper import Paper
        papers = []
        for cit in self._last_review.citations:
            papers.append(Paper(
                paper_id=cit.paper_id,
                title=cit.claim or cit.paper_id,
                authors=[],
                abstract="",
                year=0,
                sources={},
            ))
        return papers

    async def _drain_stage_queue(self) -> list[str]:
        """Process staged messages and return their responses."""
        responses: list[str] = []
        while not self._stage_queue.empty():
            staged = await self._stage_queue.get()
            logger.info("Processing staged message: %s", staged[:80])
            response = await self.send(staged)
            responses.append(response)
        return responses
