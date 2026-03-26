"""Community summarizer stub. Port from src/graphrag/summarizer.py."""
from __future__ import annotations

from typing import Callable, Awaitable


async def summarize_communities(
    communities: list[dict],
    llm_call: Callable[..., Awaitable[str]],
) -> list[dict]:
    """Generate natural-language summaries for each detected community.

    Port from src/graphrag/summarizer.py.
    """
    raise NotImplementedError("Port from src/graphrag/summarizer.py")
