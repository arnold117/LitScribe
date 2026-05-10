from __future__ import annotations

import logging
from typing import Any

from litscribe.models.paper import Paper
from litscribe.models.review import ReviewOutput

logger = logging.getLogger(__name__)


def export_bibtex(papers: list[Paper]) -> str:
    from litscribe.tools.cite_keys import assign_cite_keys, build_bibtex
    key_map = assign_cite_keys(papers)
    return build_bibtex(papers, key_map)


def export_citations(papers: list[Paper], style: str = "apa") -> str:
    from litscribe.exporters.citation_formatter import format_citations
    return format_citations([p.model_dump() for p in papers], style=style)


async def export_review(
    review: ReviewOutput,
    papers: list[Paper],
    format: str = "markdown",
    style: str = "apa",
) -> dict[str, Any]:
    if format == "bibtex":
        content = export_bibtex(papers)
        return {"format": "bibtex", "content": content}

    if format == "citations":
        content = export_citations(papers, style=style)
        return {"format": f"citations_{style}", "content": content}

    return {"format": "markdown", "content": review.text}
