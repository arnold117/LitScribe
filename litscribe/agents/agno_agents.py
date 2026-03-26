"""Agno Agent wrappers for each LitScribe pipeline step."""
from __future__ import annotations

from agno.agent import Agent
from agno.models.litellm import LiteLLM


def create_planner_agent(
    model: str = "openai/qwen-plus",
    api_key: str = "",
    api_base: str = "",
) -> Agent:
    """Create the research-planning Agno Agent."""
    return Agent(
        name="Planner",
        model=LiteLLM(id=model, api_key=api_key or None, api_base=api_base or None),
        instructions=[
            "You are a research planning agent.",
            "Given a research question, decompose it into sub-topics for a literature review.",
            "Return JSON with sub_topics (list of {name, keywords, estimated_papers}) and domain.",
        ],
        markdown=True,
    )


def create_reader_agent(
    model: str = "openai/qwen-plus",
    api_key: str = "",
    api_base: str = "",
) -> Agent:
    """Create the critical-reading Agno Agent."""
    return Agent(
        name="CriticalReader",
        model=LiteLLM(id=model, api_key=api_key or None, api_base=api_base or None),
        instructions=[
            "You are a critical reading agent for academic papers.",
            "Analyze papers and extract: key findings, methodology, strengths, limitations.",
            "Return structured analysis for each paper.",
        ],
        markdown=True,
    )


def create_synthesizer_agent(
    model: str = "openai/qwen-max",
    api_key: str = "",
    api_base: str = "",
) -> Agent:
    """Create the synthesis Agno Agent."""
    return Agent(
        name="Synthesizer",
        model=LiteLLM(id=model, api_key=api_key or None, api_base=api_base or None),
        instructions=[
            "You are a synthesis agent that writes literature reviews.",
            "Identify themes, analyze gaps, and produce a comprehensive review.",
            "Every claim must be backed by citations.",
        ],
        markdown=True,
    )


def create_reviewer_agent(
    model: str = "openai/qwen-plus",
    api_key: str = "",
    api_base: str = "",
) -> Agent:
    """Create the self-review Agno Agent."""
    return Agent(
        name="SelfReviewer",
        model=LiteLLM(id=model, api_key=api_key or None, api_base=api_base or None),
        instructions=[
            "You are a self-review agent.",
            "Evaluate the quality of a literature review: coverage, accuracy, citation grounding.",
            "Return a score (0-1), whether it passes (score >= 0.65), and feedback.",
        ],
        markdown=True,
    )
