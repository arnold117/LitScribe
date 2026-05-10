from __future__ import annotations

import json
import logging
from typing import Any

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from litscribe.config import Config
from litscribe.evolution.memory_manager import MemoryManager
from litscribe.middleware.evolution import EvolutionMiddleware
from litscribe.middleware.token_tracking import TokenTrackingMiddleware
from litscribe.prompts.supervisor import SUPERVISOR_PROMPT
from litscribe.tools.status import PipelineState

logger = logging.getLogger(__name__)


def _build_model(config: Config) -> ChatOpenAI:
    model_name = config.llm.default_model
    if "/" in model_name:
        model_name = model_name.split("/", 1)[1]

    kwargs = dict(
        model=model_name,
        openai_api_key=config.llm.api_key,
        openai_api_base=config.llm.api_base,
        temperature=0.1,
        timeout=300,
        max_retries=2,
    )
    if "deepseek" in config.llm.api_base.lower() and any(
        k in model_name.lower() for k in ("v4", "reasoner")
    ):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    return ChatOpenAI(**kwargs)


def create_pipeline_tools(config: Config, state: PipelineState, model: ChatOpenAI, memory=None):

    @tool
    async def run_review(
        research_question: str,
        max_papers: int = 40,
        language: str = "en",
        instructions: str = "",
    ) -> str:
        """Run a complete literature review pipeline. This handles plan → search → read → graphrag → synthesize → review automatically."""
        from litscribe.tools.pipeline import run_review as _run

        state.research_question = research_question
        state.language = language

        result = await _run(
            model=model, config=config, state=state,
            max_papers=max_papers, user_instructions=instructions,
            memory=memory,
        )

        # Auto-save to file
        if state.synthesis:
            from litscribe.tools.output import save_review
            filepath = save_review(state.synthesis.text, research_question)
            result += f"\n\n📄 Saved to: {filepath}"

        return result

    @tool
    async def search_papers(queries: str, max_papers: int = 20) -> str:
        """Search academic databases with queries (comma-separated). Use for quick searches without full review."""
        from litscribe.tools.search import search_all_sources

        query_list = [q.strip() for q in queries.split(",") if q.strip()]
        if not query_list:
            return "No queries provided."

        papers = await search_all_sources(query_list, config, max_per_source=max_papers)
        state.papers = papers[:max_papers]

        summaries = []
        for p in state.papers[:10]:
            authors = ", ".join(p.authors[:2]) if p.authors else "Unknown"
            summaries.append(f"- {p.title} ({authors}, {p.year})")

        return f"Found {len(papers)} papers, kept {len(state.papers)}:\n" + "\n".join(summaries)

    @tool
    async def refine_review(instruction: str) -> str:
        """Modify an existing review based on user instruction. Examples: 'add a section about delivery methods', 'expand the methodology discussion', 'remove the part about X', 'rewrite the conclusion'."""
        from litscribe.tools.refinement import refine_review as _refine
        from litscribe.prompts.utils import format_summaries_for_prompt

        if state.synthesis is None:
            return "No review to refine. Run a review first."

        papers_ctx = ""
        if state.analyses:
            from litscribe.tools.synthesis import _enrich_analyses_with_papers
            enriched = _enrich_analyses_with_papers(state.analyses, state.papers)
            papers_ctx = format_summaries_for_prompt(enriched, max_chars=5000)

        old_text = state.synthesis.text
        old_words = state.synthesis.word_count
        new_review = await _refine(
            model=model, current_review=state.synthesis,
            instruction_text=instruction,
            research_question=state.research_question,
            papers_context=papers_ctx,
            language=state.language,
            config=config,
        )
        state.synthesis = new_review

        # Show diff stats
        from litscribe.tools.diff import diff_stats
        stats = diff_stats(old_text, new_review.text)

        from litscribe.tools.output import save_review
        filepath = save_review(new_review.text, state.research_question)

        return (
            f"Review refined: {new_review.word_count} words (was {old_words}).\n"
            f"Changes: +{stats['added']} lines, -{stats['removed']} lines\n"
            f"📄 Updated: {filepath}"
        )

    @tool
    async def analyze_draft(draft_text: str, paper_abstracts: str = "") -> str:
        """Analyze a user's draft review and suggest improvements. Pass the draft text and optionally paper abstracts (comma-separated)."""
        from litscribe.tools.local_review import review_draft
        from litscribe.models.paper import Paper
        from litscribe.models.analysis import PaperAnalysis

        papers = []
        analyses = []
        if paper_abstracts:
            for i, abstract in enumerate(paper_abstracts.split("|||")):
                p = Paper(paper_id=f"local:{i}", title=f"Paper {i+1}", authors=[], abstract=abstract.strip(), year=2024, sources={"local": str(i)})
                papers.append(p)
                analyses.append(PaperAnalysis(paper_id=p.paper_id, key_findings=[abstract.strip()[:200]], methodology="", strengths=[], limitations=[], relevance_score=0.5, themes=[]))

        result = await review_draft(model, draft_text, papers, analyses)
        if "error" in result:
            return f"Error: {result['error']}"

        lines = ["**Draft Review Analysis:**\n"]
        for s in result.get("strengths", []):
            lines.append(f"✅ {s}")
        for w in result.get("weaknesses", []):
            lines.append(f"❌ {w.get('issue','')}: {w.get('suggestion','')}")
        for m in result.get("missing_topics", []):
            lines.append(f"❓ Missing: {m}")
        if result.get("revised_outline"):
            lines.append("\n**Suggested outline:** " + " → ".join(result["revised_outline"]))
        return "\n".join(lines)

    @tool
    async def suggest_review_outline(paper_abstracts: str) -> str:
        """Given paper abstracts (separated by |||), suggest what review to write and what's missing."""
        from litscribe.tools.local_review import suggest_outline
        from litscribe.models.paper import Paper
        from litscribe.models.analysis import PaperAnalysis

        papers = []
        analyses = []
        for i, abstract in enumerate(paper_abstracts.split("|||")):
            abstract = abstract.strip()
            if not abstract:
                continue
            p = Paper(paper_id=f"local:{i}", title=f"Paper {i+1}", authors=[], abstract=abstract, year=2024, sources={"local": str(i)})
            papers.append(p)
            analyses.append(PaperAnalysis(paper_id=p.paper_id, key_findings=[abstract[:200]], methodology="", strengths=[], limitations=[], relevance_score=0.5, themes=[]))

        result = await suggest_outline(model, papers, analyses)
        if "error" in result:
            return f"Error: {result['error']}"

        lines = [f"**Suggested question:** {result.get('suggested_question', '?')}\n"]
        for t in result.get("themes", []):
            lines.append(f"📚 {t.get('name','')}: {t.get('description','')[:60]}")
        lines.append(f"\n**Outline:** {' → '.join(result.get('proposed_outline', []))}")
        if result.get("gaps"):
            lines.append(f"\n**Missing:** {', '.join(result.get('gaps', []))}")
        if result.get("search_queries"):
            lines.append(f"\n**Search for:** {', '.join(result.get('search_queries', []))}")
        return "\n".join(lines)

    @tool
    async def export_results(format: str = "markdown", style: str = "apa") -> str:
        """Export the review. Formats: markdown, bibtex, citations."""
        from litscribe.tools.export import export_review

        if state.synthesis is None:
            return "No review to export. Run a review first."

        result = await export_review(state.synthesis, state.papers, format, style)
        return result.get("content", "Export failed")[:3000]

    return [run_review, search_papers, refine_review, analyze_draft, suggest_review_outline, export_results]


def create_litscribe_agent(
    config: Config,
    memory: MemoryManager | None = None,
):
    model = _build_model(config)
    state = PipelineState()

    tools = create_pipeline_tools(config, state, model, memory=memory)

    middleware = []
    if memory:
        evolution_mw = EvolutionMiddleware(memory, pipeline_state=state)
        middleware.append(evolution_mw)
    token_mw = TokenTrackingMiddleware()
    middleware.append(token_mw)

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=SUPERVISOR_PROMPT,
        middleware=middleware,
        name="litscribe",
    )

    return agent, state, token_mw
