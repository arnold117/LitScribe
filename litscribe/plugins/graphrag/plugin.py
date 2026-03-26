"""GraphRAG plugin entry point — orchestrates entity extraction, linking,
graph construction, and community detection."""
from __future__ import annotations

from typing import Any, Callable, Awaitable

from litscribe.models.analysis import PaperAnalysis


class GraphRAGPlugin:
    """Optional GraphRAG enrichment plugin with injected LLM calls.

    Usage::

        plugin = GraphRAGPlugin(llm_call=my_llm_fn)
        result = await plugin.process(analyses)
        # result keys: entities, mentions, graph, communities
    """

    def __init__(self, llm_call: Callable[..., Awaitable[str]]) -> None:
        self.llm_call = llm_call

    async def process(self, analyses: list[PaperAnalysis]) -> dict[str, Any]:
        """Run the full GraphRAG pipeline on a list of PaperAnalysis objects.

        Returns a dict with:
        - entities: mapping entity_id -> entity dict
        - mentions: list of mention dicts
        - graph: NetworkX Graph of entity co-occurrences
        - communities: list of community dicts
        """
        from litscribe.plugins.graphrag.entity_extractor import extract_entities
        from litscribe.plugins.graphrag.linker import link_entities
        from litscribe.plugins.graphrag.graph_builder import build_graph
        from litscribe.plugins.graphrag.community_detector import detect_communities

        entities, mentions = await extract_entities(analyses, self.llm_call)
        linked = link_entities(entities)
        graph = build_graph(linked, mentions)
        communities = detect_communities(graph)

        return {
            "entities": {e["entity_id"]: e for e in linked},
            "mentions": mentions,
            "graph": graph,
            "communities": communities,
        }
