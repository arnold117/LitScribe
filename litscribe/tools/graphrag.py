from __future__ import annotations

import logging
from typing import Any

from litscribe.models.analysis import PaperAnalysis

logger = logging.getLogger(__name__)


async def build_knowledge_graph(
    analyses: list[PaperAnalysis],
    llm_call,
) -> dict[str, Any]:
    from litscribe.plugins.graphrag.plugin import GraphRAGPlugin

    plugin = GraphRAGPlugin(llm_call=llm_call)
    analysis_dicts = [a.model_dump() for a in analyses]

    try:
        result = await plugin.process(analysis_dicts)
        n_entities = len(result.get("entities", []))
        n_communities = len(result.get("communities", []))
        logger.info(f"GraphRAG: {n_entities} entities, {n_communities} communities")
        return result
    except Exception as e:
        logger.warning(f"GraphRAG failed: {e}")
        return {"entities": [], "mentions": [], "communities": [], "graph": None}
